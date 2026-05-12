"""
eta_convergence_inference_deterministic.py
────────────────────────────
Runs Deterministic inference for a bus trip that can start at any section.
Loads a 9-row Excel file (one row per possible starting point) and
produces a convergence chart of the total ETA as the bus works through 
successive sections (No Uncertainty Bands).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

num_segment = 9

# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL ARCHITECTURE (STANDARD PYTORCH)
# ══════════════════════════════════════════════════════════════════════════════

class DeterministicLocalIsolationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments):
        super().__init__()
        self.num_segments = num_segments
        # Standard PyTorch nn.ModuleList
        self.nets = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_segments)
        ])
            
    def forward(self, x_inputs):
        outputs = []
        for i in range(self.num_segments):
            out = torch.nn.functional.silu(self.nets[i](x_inputs[i]))
            outputs.append(out)
        return outputs

class DeterministicNeighborMixingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2):
        super().__init__()
        self.num_segments = num_segments
        
        # 1. Attention Networks (Standard Linear)
        self.att_nets = nn.ModuleList([
            nn.Linear(input_dim * 2, 2) for _ in range(num_segments)
        ])
        
        # 2. Processing Networks (Standard Linear)
        net_input_dim = input_dim * 2
        self.nets_1 = nn.ModuleList([
            nn.Linear(net_input_dim, output_dim) for _ in range(num_segments)
        ])
        self.nets_2 = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_segments)
        ])
        
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        
    def forward(self, prev_layer_outputs):
        outputs = []
        for i in range(self.num_segments):
            h_self = prev_layer_outputs[i]
            
            if i < self.num_segments - 1:
                h_right = prev_layer_outputs[i+1]
            else:
                h_right = torch.zeros_like(h_self)

            # --- Deterministic Attention ---
            context = torch.cat([h_self, h_right], dim=1) 
            raw_attention_scores = self.att_nets[i](context) 
            attention_weights = torch.nn.functional.softmax(raw_attention_scores, dim=1)
            
            alpha_self = attention_weights[:, 0].unsqueeze(1)  
            alpha_right = attention_weights[:, 1].unsqueeze(1) 

            self_feat_weighted = h_self * alpha_self
            right_feat_weighted = h_right * alpha_right
            
            combined = torch.cat([self_feat_weighted, right_feat_weighted], dim=1)
            
            # --- Standard Processing ---
            out = self.nets_1[i](combined)
            out = self.dropout_1(out) 
            out = torch.nn.functional.silu(out)
            
            out = self.nets_2[i](out)
            out = self.dropout_2(out)
            out = torch.nn.functional.silu(out)
            
            outputs.append(out)
            
        return outputs

# ==========================================
# 3. DETERMINISTIC "MATRIX" GNN MODEL
# ==========================================
class DeterministicMatrixGNN(nn.Module):
    def __init__(self, num_sections=9, global_dim=9, local_dim=4, hidden_dim=32):
        super().__init__()
        self.num_sections = num_sections
        input_dim = global_dim + local_dim + 1 
            
        self.embedding_layer = DeterministicLocalIsolationLayer(input_dim, hidden_dim, num_sections)
        
        self.prop_layers = nn.ModuleList([
            DeterministicNeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2)
            for _ in range(3)
        ])
        
        # --- Output Heads ---
        # In a deterministic model, we ONLY predict the Mean (loc).
        # We DO NOT predict scale or df, because there is no probability distribution!
        self.heads_loc = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_sections)
        ])

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        
        # Virtual Clock
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        inputs_list = []
        for i in range(self.num_sections):
            loc_i = all_sections_data[:, i, :]
            time_i = accumulated_time[:, i, :]
            inp = torch.cat([global_features, loc_i, time_i], dim=1)
            inputs_list.append(inp)
            
        h_current = self.embedding_layer(inputs_list)
            
        for layer in self.prop_layers:
            h_current = layer(h_current)
        
        all_locs = []
        
        for i in range(self.num_sections):
            final_feat = h_current[i] 
            
            # Predict only the point-estimate (the mean time)
            loc = self.heads_loc[i](final_feat)
            all_locs.append(loc)
        
        # Return a list of tensors, shape: [Batch, 1] for each section
        return all_locs


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_trip_snapshot(file_path: str, device: torch.device):
    print(f"Reading trip snapshot: {file_path}")
    df = pd.read_excel(file_path, header=None, skiprows=1)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    raw = df.values.astype(np.float32)

    x_global   = torch.tensor(raw[:, 0:9], dtype=torch.float32, device=device)
    time_spent = raw[:, 18]                                                          

    local_start = 9 + num_segment + 1   
    x_local_raw = raw[:, local_start : local_start + num_segment * 4]
    x_local = torch.tensor(
        x_local_raw.reshape(-1, num_segment, 4), dtype=torch.float32, device=device
    )

    print(f"  Loaded {raw.shape[0]} rows  |  elapsed times: {time_spent.round(1)}")
    return x_global, x_local, time_spent


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DETERMINISTIC CONVERGENCE CHART (NO BOLLINGER BANDS)
# ══════════════════════════════════════════════════════════════════════════════

def plot_eta_convergence(section_ids, etas, time_spent,
                         num_sections=9, output_path="eta_convergence_deterministic.png"):
    s   = np.asarray(section_ids, dtype=float)
    mu  = np.asarray(etas)
    el  = np.asarray(time_spent)

    # ── Style ──────────────────────────────────────────────────────────────
    BG      = "#0D1117"
    PANEL   = "#161B22"
    ACCENT  = "#e74c3c"  # Switched to a nice deterministic Red
    ACCENT2 = "#3FB950"
    ELAPSED = "#F0883E"
    GRID    = "#21262D"
    TEXT    = "#E6EDF3"
    MUTED   = "#8B949E"

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    ax.set_facecolor(PANEL)

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # ── Section markers ────────────────────────────────────────────────────
    for i in range(num_sections):
        ax.axvline(i, color=GRID, linewidth=0.6, linestyle="--", zorder=0)

    # ── Mean ETA line (Deterministic) ──────────────────────────────────────
    ax.plot(s, mu, color=ACCENT, linewidth=2.5, zorder=4,
            label="Total Forecasted ETA", marker="o", markersize=6,
            markerfacecolor=BG, markeredgecolor=ACCENT, markeredgewidth=2)

    # ── Elapsed time area (already-spent portion) ──────────────────────────
    ax.fill_between(s, 0, el,
                    color=ELAPSED, alpha=0.18, label="Elapsed (known)", zorder=1)
    ax.plot(s, el, color=ELAPSED, linewidth=1.8, linestyle=":", zorder=3,
            marker="s", markersize=5,
            markerfacecolor=BG, markeredgecolor=ELAPSED, markeredgewidth=1.8)

    # ── Remaining-time annotation bar ─────────────────────────────────────
    remaining = mu - el
    ax.bar(s, remaining, bottom=el,
           color=ACCENT2, alpha=0.15, width=0.35, zorder=1, label="Predicted remaining")

    # ── Annotate exact ETA at each point ───────────────────────────────────
    for i, (xi, yi) in enumerate(zip(s, mu)):
        ax.annotate(f"{yi:.0f}s",
                    xy=(xi, yi),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, color=TEXT,
                    fontweight="bold")

    # ── Labels & formatting ────────────────────────────────────────────────
    ax.set_xlabel("Bus position  (starting section)", color=MUTED, fontsize=11, labelpad=10)
    ax.set_ylabel("Time  (seconds)", color=MUTED, fontsize=11, labelpad=10)
    ax.set_title("Total ETA Convergence as Bus Progresses (Deterministic)",
                 color=TEXT, fontsize=14, fontweight="bold", pad=16)

    ax.set_xticks(range(num_sections))
    ax.set_xticklabels([f"Sec {i}" for i in range(num_sections)], color=MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/60:.1f} min" if v >= 120 else f"{v:.0f} s"
    ))

    leg = ax.legend(loc="upper right", framealpha=0.25,
                    facecolor=PANEL, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=9)

    fig.text(0.5, 0.01,
             "Predictions stabilize as fewer unobserved sections remain",
             ha="center", color=MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nChart saved → {output_path}")
    
    try:
        plt.show()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── File paths  (edit these) ───────────────────────────────────────────
    # MUST point to your STANDARD PYTORCH saved state_dict file
    saved_params_path  = "deterministic_model.pt" 
    saved_scaler_path  = "deterministic_scaler.pkl"
    trip_snapshot_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy_weather_4_D.xlsx"
    output_chart_path  = "eta_convergence_deterministic_chart_4D.png"

    # ── 1. Load scaler & trip data ─────────────────────────────────────────
    print("\n── Loading scaler & trip snapshot ──")
    loaded_scaler = joblib.load(saved_scaler_path)
    x_global_trip, x_local_trip, time_spent = load_trip_snapshot(
        trip_snapshot_path, device
    )

    # ── 2. Reconstruct model ───────────────────────────────────────────────
    print("\n── Initialising Deterministic Model ──")
    deterministic_model = DeterministicMatrixGNN(
        num_sections=num_segment, global_dim=9, local_dim=4,
        hidden_dim=32
    ).to(device)

    # Load Standard PyTorch Weights
    state_dict = torch.load(saved_params_path, map_location=device)
    deterministic_model.load_state_dict(state_dict, strict=False)

    # MUST set to Eval mode to turn off Dropout
    deterministic_model.eval()
    print("Weights loaded successfully.")

    # ── 3. Single-Pass Inference for each starting section ─────────────────
    print(f"\n── Running Deterministic Inference ({num_segment} sections) ──")
    
    all_etas = []

    with torch.no_grad():
        for start_sec in range(num_segment):
            xg = x_global_trip[start_sec:start_sec+1]   # (1, 9)
            xl = x_local_trip[start_sec:start_sec+1]    # (1, 9, 4)
            elapsed = float(time_spent[start_sec])

            # Forward pass (Returns a list of 9 tensors)
            preds_list = deterministic_model(xg, xl)
            
            # If the model happened to return a tuple of lists, extract just the locations
            if isinstance(preds_list, tuple):
                preds_list = preds_list[0]

            # Merge list of tensors into one tensor of shape (1, 9)
            preds_scaled = torch.cat(preds_list, dim=1).cpu().numpy()

            # Inverse-transform 
            preds_real = loaded_scaler.inverse_transform(preds_scaled)  # (1, 9)

            # Sum the predicted travel times ONLY for the remaining sections
            remaining = preds_real[0, start_sec:].sum()
            total_eta  = elapsed + remaining

            all_etas.append(total_eta)

            print(f"  Start sec {start_sec}: elapsed={elapsed:.1f}s  "
                  f"remaining={remaining:.1f}s  total={total_eta:.1f}s")

    # ── 4. Plot ────────────────────────────────────────────────────────────
    print("\n── Plotting ──")
    plot_eta_convergence(
        section_ids=list(range(num_segment)),
        etas=all_etas,
        time_spent=time_spent,
        num_sections=num_segment,
        output_path=output_chart_path,
    )