"""
eta_convergence_inference.py
────────────────────────────
Runs Bayesian inference for a bus trip that can start at any section.
Loads a 9-row Excel file (one row per possible starting point) and
produces a Bollinger-band convergence chart of the total ETA as the
bus works through successive sections.

Expected Excel layout (same base structure as training data):
  Cols  0– 8  : global features  (x_global)
  Cols  9–17  : section-level targets  (not used at inference time)
  Col  18     : elapsed travel time so far  (seconds already spent)
  Cols 19–54  : local section features  (9 sections × 4 features)
"""

import statistics
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from sklearn.model_selection import train_test_split   # kept for helper below

num_segment = 9


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
# 1.  MODEL ARCHITECTURE  (must match the saved weights exactly)
# ══════════════════════════════════════════════════════════════════════════════

class LocalIsolationLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments, device="cuda"):
        super().__init__()
        self.num_segments = num_segments
        self.nets = PyroModuleList([])
        for i in range(num_segments):
            net = PyroModule[nn.Linear](input_dim, output_dim)
            z = torch.tensor(0., device=device)
            s = torch.tensor(1., device=device)
            df = torch.tensor(15., device=device)
            net.weight = PyroSample(dist.StudentT(df, z, s).expand([output_dim, input_dim]).to_event(2))
            net.bias   = PyroSample(dist.StudentT(df, z, s).expand([output_dim]).to_event(1))
            self.nets.append(net)

    def forward(self, x_inputs):
        return [torch.nn.functional.silu(self.nets[i](x_inputs[i]))
                for i in range(self.num_segments)]


class NeighborMixingLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device="cuda"):
        super().__init__()
        self.num_segments = num_segments
        ls = torch.tensor(2.0, device=device)
        lz = torch.tensor(0.0, device=device)
        sc = torch.tensor(1.0, device=device)
        z  = torch.tensor(0.0, device=device)
        ws = torch.tensor(1.0, device=device)
        bs = torch.tensor(0.1, device=device)

        self.w_self  = PyroSample(dist.Normal(ls, sc).expand([num_segments]).to_event(1))
        self.w_right = PyroSample(dist.Normal(lz, sc).expand([num_segments]).to_event(1))

        self.nets_1 = PyroModuleList([])
        self.nets_2 = PyroModuleList([])
        for i in range(num_segments):
            n1 = PyroModule[nn.Linear](input_dim * 2, output_dim)
            n1.weight = PyroSample(dist.Normal(z, ws).expand([output_dim, input_dim*2]).to_event(2))
            n1.bias   = PyroSample(dist.Normal(z, bs).expand([output_dim]).to_event(1))
            self.nets_1.append(n1)
        self.dropout_1 = PyroModule[nn.Dropout](p=dropout_rate)

        for i in range(num_segments):
            n2 = PyroModule[nn.Linear](output_dim, output_dim)
            n2.weight = PyroSample(dist.Normal(z, ws).expand([output_dim, output_dim]).to_event(2))
            n2.bias   = PyroSample(dist.Normal(z, bs).expand([output_dim]).to_event(1))
            self.nets_2.append(n2)
        self.dropout_2 = PyroModule[nn.Dropout](p=dropout_rate)

    def forward(self, prev):
        out = []
        for i in range(self.num_segments):
            sf = prev[i] * torch.nn.functional.softplus(self.w_self[i])
            rf = (prev[i+1] * torch.nn.functional.softplus(self.w_right[i])
                  if i < self.num_segments - 1
                  else torch.zeros_like(sf))
            x = self.nets_1[i](torch.cat([sf, rf], dim=1))
            x = torch.nn.functional.silu(self.dropout_1(x))
            x = self.nets_2[i](x)
            x = torch.nn.functional.silu(self.dropout_2(x))
            out.append(x)
        return out

class MatrixGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=8, device = 'cuda', pnt_1 = None):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        input_dim = global_dim + local_dim + 1 
        self.pnt_1 = pnt_1
        
        # --- Layer 0: Input Projection (Unique per segment) ---
        
            
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        
        # --- Propagation Layers (The Matrix) ---
        # "Different function for each layer each section"
        # We create N layers. Each layer contains N unique networks.
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(3) # Layer depth = num_segments
        ])
        
        # --- Output Heads ---
        final_dim = hidden_dim
        
        self.heads_loc = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df = PyroModuleList([])
        
        for i in range(self.num_sections):
            loc_std_dev = 2
            # Tensors for Heads
            zero = torch.tensor(0., device=device)
            loc_std = torch.tensor(1.0, device=device)
            loc_bias_mu = torch.tensor(0., device=device)
            loc_bias_std = torch.tensor(1., device=device)

            scale_std = torch.tensor(0.3, device=device)
            scale_bias_mu = torch.tensor(0., device=device)
            scale_bias_std = torch.tensor(3.0, device=device)

            df_std = torch.tensor(1, device=device)
            df_bias_mu = torch.tensor(0., device=device)
            df_bias_std = torch.tensor(3.0, device=device)
            
            # Loc Head
            h_loc = PyroModule[nn.Linear](final_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(zero, loc_std).expand([1, final_dim]).to_event(2))
            h_loc.bias = PyroSample(dist.Normal(loc_bias_mu, loc_bias_std).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)
            
            # Scale Head
            h_scale = PyroModule[nn.Linear](final_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(zero, scale_std).expand([1, final_dim]).to_event(2))
            h_scale.bias = PyroSample(dist.Normal(scale_bias_mu, scale_bias_std).expand([1]).to_event(1))
            self.heads_scale.append(h_scale)
            
            # DF Head
            h_df = PyroModule[nn.Linear](final_dim, 1)
            h_df.weight = PyroSample(dist.Normal(zero, df_std).expand([1, final_dim]).to_event(2))
            h_df.bias = PyroSample(dist.Normal(df_bias_mu, df_bias_std).expand([1]).to_event(1))
            self.heads_df.append(h_df)

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        
        # Initialize a single running clock (not a vector of zeros for all sections)
        current_time = torch.zeros(batch_size, self.num_sections).to(device)
        
        all_locs = []
        all_scales = []
        all_dfs = []
        
        # --- THE AUTOREGRESSIVE WATERFALL LOOP ---
        for current_section in range(self.num_sections):
            # 1. Build the Input List for the GNN
            inputs_list = []
            for i in range(self.num_sections):
                loc_i = all_sections_data[:, i, :]
                # THE CRITICAL INJECTION LOGIC
                if i == current_section:
                    # If we are looking at the section we are currently predicting,
                    # we inject the REAL running clock time.
                    time_i = current_time[:, i:i+1] 
                    #time_i = torch.zeros(batch_size, 1).to(device)
                elif i < current_section:
                    # For sections in the PAST, we could inject their actual historical times,
                    # but for simplicity, feeding the current clock is often enough, 
                    # or you can feed 0 if you want them to be "static anchors".
                    # Let's feed the current clock to show "how far past" they are.
                    time_i = current_time[:, i:i+1] 
                    #time_i = torch.zeros(batch_size, 1).to(device)
                    
                else:
                    # For sections in the FUTURE, we don't know the time yet.
                    # We feed 0.0 (or you could feed current_time as a baseline).
                    # Let's feed 0.0 to indicate "unreached".
                    time_i = torch.zeros(batch_size, 1).to(device)
                if self.pnt_1 is None:
                    self.pnt_1 = True
                    #print(current_time.abs().mean().item(), current_time)
                
                if time_i.abs().mean().item() > 15:
                    time_i = torch.zeros(batch_size, 1).to(device)
                #print(time_i.abs().mean().item())
                    #print(f"Section {i} Time Injection: {time_i.abs().mean().item():.2f}")
                #print(time_i.abs().mean().item())
                inp = torch.cat([global_features, loc_i, time_i], dim=1)
                inputs_list.append(inp)
                
            # 2. Run the full GNN (Embedding + Mixing) to get context
            h_current = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h_current = layer(h_current)
            
            # 3. Extract ONLY the prediction for the current_section
            final_feat = h_current[current_section] 
            
            loc = self.heads_loc[current_section](final_feat)
            if self.pnt_1 is True:
                self.pnt_1 = False
                #print(loc)
            #print("loc:", loc.mean().item())
            scale = torch.nn.functional.softplus(self.heads_scale[current_section](final_feat)) + 1e-3 
            df = torch.nn.functional.softplus(self.heads_df[current_section](final_feat)) + 2.5
            
            # 4. Store the predictions
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            #print(all_locs[-1].mean().item())
            #print(all_locs[-1].mean().item())
            # 5. UPDATE THE CLOCK for the next loop iteration
            # We add the predicted travel time of THIS section to the running total.
            for j in range(current_section, self.num_sections):
                current_time[:, j:j+1] = current_time[:, j:j+1] + loc
            #print(loc.abs().mean().item())
            
            #current_time = current_time + loc
            
        return all_locs, all_scales, all_dfs


def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    with pyro.poutine.scale(scale=kl_weight):
        locs, scales, dfs = bnn_model(x_global, x_local)
    if total_size is None:
        total_size = x_global.shape[0]
    with pyro.plate("data", size=total_size,
                    subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            d = dist.StudentT(dfs[i].squeeze(-1),
                              locs[i].squeeze(-1),
                              scales[i].squeeze(-1))
            obs = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", d, obs=obs)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_trip_snapshot(file_path: str, device: torch.device):
    """
    Load the 9-row 'live snapshot' Excel file.

    Expected column layout
    ─────────────────────
    0– 8  : global features          (x_global)
    9–17  : section targets          (not used here – model is inference-only)
    18    : elapsed time so far (s)  (time_spent)
    19–54 : local section features   (x_local, 9 sections × 4 cols)

    Returns
    ───────
    x_global   : Tensor (9, 9)
    x_local    : Tensor (9, 9, 4)
    time_spent : np.ndarray (9,)   seconds already travelled when row i was recorded
    """
    print(f"Reading trip snapshot: {file_path}")
    df = pd.read_excel(file_path, header=None, skiprows=1)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    raw = df.values.astype(np.float32)

    x_global   = torch.tensor(raw[:, 0:9], dtype=torch.float32, device=device)
    time_spent = raw[:, 18]                                                          # col S

    local_start = 9 + num_segment + 1   # = 19
    x_local_raw = raw[:, local_start : local_start + num_segment * 4]
    x_local = torch.tensor(
        x_local_raw.reshape(-1, num_segment, 4), dtype=torch.float32, device=device
    )

    print(f"  Loaded {raw.shape[0]} rows  |  elapsed times: {time_spent.round(1)}")
    return x_global, x_local, time_spent


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BOLLINGER-BAND CONVERGENCE CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_eta_convergence(section_ids, means, stds, deter_means, time_spent,
                         num_sections=9, output_path="eta_convergence.png"):
    """
    Draws a transit-style Bollinger-band convergence chart.

    Parameters
    ──────────
    section_ids : array-like  starting section for each row  (0 … num_sections-1)
    means       : array-like  mean total ETA (elapsed + predicted remaining), seconds
    stds        : array-like  std of total ETA across MC samples, seconds
    time_spent  : array-like  elapsed time at each starting section, seconds
    """
    s   = np.asarray(section_ids, dtype=float)
    mu  = np.asarray(means)
    sg  = np.asarray(stds)
    el  = np.asarray(time_spent)

    # ── Style ──────────────────────────────────────────────────────────────
    BG      = "#0D1117"
    PANEL   = "#161B22"
    ACCENT  = "#58A6FF"
    ACCENT_D  = "#e74c3c"
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

    # ── Bollinger bands ────────────────────────────────────────────────────
    ax.fill_between(s, mu - 2*sg, mu + 2*sg,
                    color=ACCENT, alpha=0.12, label="±2σ band", zorder=1)
    ax.fill_between(s, mu - sg,   mu + sg,
                    color=ACCENT, alpha=0.25, label="±1σ band", zorder=2)

    # ── Mean ETA line (Bayesian) ───────────────────────────────────────────
    ax.plot(s, mu, color=ACCENT, linewidth=2.5, zorder=4,
            label="Mean total ETA", marker="o", markersize=6,
            markerfacecolor=BG, markeredgecolor=ACCENT, markeredgewidth=2)
    
    # ── Mean ETA line (Deterministic) ──────────────────────────────────────
    ax.plot(s, deter_means, color=ACCENT_D, linewidth=2.5, zorder=4,
            label="Total Forecasted ETA", marker="o", markersize=6,
            markerfacecolor=BG, markeredgecolor=ACCENT_D, markeredgewidth=2)

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
    ax.plot(s, mu, color=ACCENT2, alpha=0.0)   # invisible, just for legend

    # ── Annotate std at each point ─────────────────────────────────────────
    for i, (xi, yi, si) in enumerate(zip(s, mu, sg)):
        ax.annotate(f"±{si:.0f}s",
                    xy=(xi, yi + 2*sg[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5, color=MUTED,
                    fontfamily="monospace")
        
    # ── Annotate exact ETA at each point ───────────────────────────────────
    for i, (xi, yi) in enumerate(zip(s, mu)):
        ax.annotate(f"{yi:.0f}s",
                    xy=(xi, yi),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, color=TEXT,
                    fontweight="bold")

    # ── Labels & formatting ────────────────────────────────────────────────
    ax.set_xlabel("Bus position  (starting section)", color=MUTED, fontsize=11,
                  labelpad=10)
    ax.set_ylabel("Time  (seconds)", color=MUTED, fontsize=11, labelpad=10)
    ax.set_title("Total ETA Convergence as Bus Progresses",
                 color=TEXT, fontsize=14, fontweight="bold", pad=16)

    ax.set_xticks(range(num_sections))
    ax.set_xticklabels([f"Sec {i}" for i in range(num_sections)],
                       color=MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/60:.1f} min" if v >= 120 else f"{v:.0f} s"
    ))

    # ── Legend ─────────────────────────────────────────────────────────────
    leg = ax.legend(loc="upper right", framealpha=0.25,
                    facecolor=PANEL, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=9)

    # ── Subtitle note ──────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             "Bands shrink as fewer sections remain · uncertainty can widen near congestion",
             ha="center", color=MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nChart saved → {output_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── File paths  (edit these) ───────────────────────────────────────────
    saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu.pt"
    saved_scaler_path = "y_scaler.pkl"
    #trip_snapshot_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy3_flagged.xlsx"   # your new 9-row file
    trip_snapshot_path = "Jumpy1.xlsx"   # your new 9-row file
    output_chart_path  = "Jumpy1.png"

    NUM_SAMPLES = 200   # MC samples per starting section

    # ── 1. Load scaler & trip data ─────────────────────────────────────────
    print("\n── Loading scaler & trip snapshot ──")
    loaded_scaler = joblib.load(saved_scaler_path)
    x_global_trip, x_local_trip, time_spent = load_trip_snapshot(
        trip_snapshot_path, device
    )

    # ── 2. Reconstruct model & guide ──────────────────────────────────────
    print("\n── Initialising model ──")
    pyro.clear_param_store()
    pyro.get_param_store().load(saved_params_path, map_location=device.type)

    bnn_model = MatrixGNN(
        num_sections=num_segment, global_dim=9, local_dim=4,
        hidden_dim=32, device=device
    ).to(device)

    base_guide = AutoDiagonalNormal(model_fn).to(device)

    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=0.1):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true,
                              total_size=total_size, kl_weight=kl_weight)

    # Warm-up trace to bind param store → guide
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        guide_fn(x_global_trip[0:1], x_local_trip[0:1], y_true=dummy_y)

    bnn_model.eval()
    base_guide.eval()
    print("Weights loaded & linked.")

    # ── 3. Inference for each starting section ─────────────────────────────
    print(f"\n── Running inference  ({NUM_SAMPLES} MC samples × {num_segment} sections) ──")
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=NUM_SAMPLES)

    all_means, all_stds = [], []

    for start_sec in range(num_segment):
        xg = x_global_trip[start_sec:start_sec+1]   # (1, 9)
        xl = x_local_trip[start_sec:start_sec+1]    # (1, 9, 4)
        elapsed = float(time_spent[start_sec])

        samples = predictive(xg, xl)

        # Collect scaled predictions for each section: shape (NUM_SAMPLES, 9)
        samples_scaled = np.stack(
            [samples[f"obs_section_{i}"].squeeze().cpu().numpy()
             for i in range(num_segment)],
            axis=1
        )  # (NUM_SAMPLES, 9)

        # Inverse-transform each MC sample row
        samples_real = loaded_scaler.inverse_transform(samples_scaled)  # (NUM_SAMPLES, 9)

        # Only sum sections from start_sec onwards (remaining journey)
        remaining_per_sample = samples_real[:, start_sec:].sum(axis=1)   # (NUM_SAMPLES,)
        total_eta_per_sample  = elapsed + remaining_per_sample            # (NUM_SAMPLES,)

        mean_eta = total_eta_per_sample.mean()
        std_eta  = total_eta_per_sample.std()
        all_means.append(mean_eta)
        all_stds.append(std_eta)

        print(f"  Start sec {start_sec}: elapsed={elapsed:.1f}s  "
              f"remaining={mean_eta-elapsed:.1f}s  total={mean_eta:.1f}s  ±{std_eta:.1f}s")
        
        
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
        means=all_means,
        stds=all_stds,
        deter_means=all_etas,
        time_spent=time_spent,
        num_sections=num_segment,
        output_path=output_chart_path,
    )