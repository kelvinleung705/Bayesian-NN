"""
eta_convergence_inference.py  (multi-trip variance edition)
────────────────────────────────────────────────────────────
Loads N trips (each N_SECTIONS rows) from a single Excel file,
runs Bayesian + Deterministic inference, and reports the
cross-trip variance of total ETA at each possible starting section.
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

num_segment = 9
NUM_TRIPS   = 50        # ← set to actual number of trips in your file


# ══════════════════════════════════════════════════════════════════════════════
# (Architecture classes unchanged – paste your original ones here)
# ══════════════════════════════════════════════════════════════════════════════
# ... DeterministicLocalIsolationLayer, DeterministicNeighborMixingLayer,
#     DeterministicMatrixGNN, LocalIsolationLayer, NeighborMixingLayer,
#     MatrixGNN, model_fn  (identical to original)
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
# 2.  DATA LOADING  – now returns a list of (x_global, x_local, time_spent)
# ══════════════════════════════════════════════════════════════════════════════

def load_all_trips(file_path: str, device: torch.device,
                   num_trips: int = NUM_TRIPS,
                   n_sections: int = num_segment):
    """
    Load the multi-trip Excel file.
    Rows 0..(n_sections-1)      → trip 0
    Rows n_sections..(2*n-1)    → trip 1
    …

    Returns
    ───────
    trips : list of (x_global, x_local, time_spent) tuples, one per trip
    """
    print(f"Reading {file_path}  (expecting {num_trips} trips × {n_sections} rows each)")
    df = pd.read_excel(file_path, header=None, skiprows=1)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    raw = df.values.astype(np.float32)

    expected_rows = num_trips * n_sections
    if raw.shape[0] < expected_rows:
        raise ValueError(
            f"File has only {raw.shape[0]} rows, "
            f"but {num_trips} trips × {n_sections} sections = {expected_rows} rows expected."
        )

    trips = []
    local_start = 9 + n_sections + 1   # col 19 (same as original)

    for t in range(num_trips):
        row_start = t * n_sections
        row_end   = row_start + n_sections
        block = raw[row_start:row_end]          # shape (n_sections, n_cols)

        x_global   = torch.tensor(block[:, 0:9],
                                  dtype=torch.float32, device=device)
        time_spent = block[:, 18]               # col S – elapsed seconds
        x_local_raw = block[:, local_start : local_start + n_sections * 4]
        x_local = torch.tensor(
            x_local_raw.reshape(-1, n_sections, 4),
            dtype=torch.float32, device=device
        )
        trips.append((x_global, x_local, time_spent))

    print(f"  Loaded {num_trips} trips OK.")
    return trips


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VARIANCE SUMMARY CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_trip_variance(
        section_ids,
        # Bayesian arrays  (shape: num_sections)
        bayes_mean_of_means,  bayes_std_of_means,
        bayes_mean_of_stds,
        # Deterministic arrays (shape: num_sections)
        deter_mean_of_etas,   deter_std_of_etas,
        num_sections=9,
        output_path="eta_variance_across_trips.png"):
    """
    Two-panel chart:
      Top   – cross-trip mean ± std of total ETA (Bayesian vs Deterministic)
      Bottom – cross-trip std (spread) of total ETA per starting section
    """
    s = np.asarray(section_ids, dtype=float)

    BG    = "#0D1117"; PANEL = "#161B22"; ACCENT = "#58A6FF"
    RED   = "#e74c3c"; GREEN = "#3FB950"; MUTED  = "#8B949E"
    TEXT  = "#E6EDF3"; GRID  = "#21262D"

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), facecolor=BG,
                             gridspec_kw={"hspace": 0.42})

    for ax in axes:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        for i in range(num_sections):
            ax.axvline(i, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
        ax.set_xticks(range(num_sections))
        ax.set_xticklabels([f"Sec {i}" for i in range(num_sections)],
                           color=MUTED, fontsize=9)
        ax.tick_params(axis="y", colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    # ── Panel A: cross-trip mean ± std of total ETA ────────────────────────
    ax = axes[0]
    ax.fill_between(s,
                    bayes_mean_of_means - bayes_std_of_means,
                    bayes_mean_of_means + bayes_std_of_means,
                    color=ACCENT, alpha=0.20, label="Bayesian ±1 cross-trip σ")
    ax.plot(s, bayes_mean_of_means, color=ACCENT, linewidth=2.5,
            marker="o", markersize=6, markerfacecolor=BG,
            markeredgecolor=ACCENT, markeredgewidth=2,
            label="Bayesian – mean ETA (avg over trips)")

    ax.fill_between(s,
                    deter_mean_of_etas - deter_std_of_etas,
                    deter_mean_of_etas + deter_std_of_etas,
                    color=RED, alpha=0.20, label="Deterministic ±1 cross-trip σ")
    ax.plot(s, deter_mean_of_etas, color=RED, linewidth=2.5,
            marker="o", markersize=6, markerfacecolor=BG,
            markeredgecolor=RED, markeredgewidth=2,
            label="Deterministic – mean ETA (avg over trips)")

    ax.set_title("Cross-Trip Mean Total ETA per Starting Section",
                 color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Total ETA (seconds)", color=MUTED, fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/60:.1f} min" if v >= 120 else f"{v:.0f} s"))
    ax.legend(framealpha=0.25, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=9, loc="upper right")

    # ── Panel B: cross-trip std (variability) per starting section ─────────
    ax = axes[1]
    width = 0.3
    ax.bar(s - width/2, bayes_std_of_means,  width=width,
           color=ACCENT, alpha=0.75, label="Bayesian σ of mean ETA across trips")
    ax.bar(s + width/2, deter_std_of_etas, width=width,
           color=RED,   alpha=0.75, label="Deterministic σ of ETA across trips")

    # also overlay the Bayesian model's own average internal uncertainty
    ax.plot(s, bayes_mean_of_stds, color=GREEN, linewidth=2,
            linestyle="--", marker="D", markersize=5,
            markerfacecolor=BG, markeredgecolor=GREEN, markeredgewidth=1.8,
            label="Bayesian avg within-sample σ (model uncertainty)")

    # annotate bars
    for xi, bv, dv in zip(s, bayes_std_of_means, deter_std_of_etas):
        ax.text(xi - width/2, bv + 1, f"{bv:.1f}", ha="center",
                fontsize=7, color=ACCENT, fontfamily="monospace")
        ax.text(xi + width/2, dv + 1, f"{dv:.1f}", ha="center",
                fontsize=7, color=RED,   fontfamily="monospace")

    ax.set_title("Cross-Trip Variance (Std Dev) of Total ETA per Starting Section",
                 color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Bus position (starting section)", color=MUTED,
                  fontsize=11, labelpad=8)
    ax.set_ylabel("Std Dev of Total ETA (s)", color=MUTED, fontsize=11, labelpad=8)
    ax.legend(framealpha=0.25, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=9, loc="upper right")

    fig.text(0.5, 0.01,
             f"Statistics computed over {NUM_TRIPS} trips · "
             "Bayesian MC σ = within-trip model uncertainty",
             ha="center", color=MUTED, fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nChart saved → {output_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── File paths ─────────────────────────────────────────────────────────
    saved_params_path  = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu.pt"
    saved_scaler_path  = "y_scaler.pkl"
    trip_snapshot_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy_50.xlsx"
    output_chart_path  = "eta_variance_across_trips.png"
    NUM_SAMPLES = 50

    # ── 1. Load scaler & all trips ─────────────────────────────────────────
    print("\n── Loading scaler & all trips ──")
    loaded_scaler = joblib.load(saved_scaler_path)
    trips = load_all_trips(trip_snapshot_path, device,
                           num_trips=NUM_TRIPS, n_sections=num_segment)

    # ── 2. Rebuild Bayesian model ──────────────────────────────────────────
    print("\n── Initialising Bayesian model ──")
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

    # warm-up trace
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        guide_fn(trips[0][0][0:1], trips[0][1][0:1], y_true=dummy_y)

    bnn_model.eval()
    base_guide.eval()
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=NUM_SAMPLES)

    # ── 3. Rebuild Deterministic model ────────────────────────────────────
    print("\n── Initialising Deterministic model ──")
    deterministic_model = DeterministicMatrixGNN(
        num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32
    ).to(device)
    state_dict = torch.load(saved_params_path, map_location=device)
    deterministic_model.load_state_dict(state_dict, strict=False)
    deterministic_model.eval()

    # ── 4. Collect per-trip ETA for every starting section ────────────────
    #
    # Layout of accumulators:
    #   bayes_eta_means[sec]  → list of mean_eta  (one float per trip)
    #   bayes_eta_stds[sec]   → list of std_eta   (one float per trip)
    #   deter_etas[sec]       → list of total_eta (one float per trip)
    #
    bayes_eta_means = [[] for _ in range(num_segment)]
    bayes_eta_stds  = [[] for _ in range(num_segment)]
    deter_etas      = [[] for _ in range(num_segment)]

    print(f"\n── Running inference over {NUM_TRIPS} trips ──")

    for trip_idx, (x_global_trip, x_local_trip, time_spent) in enumerate(trips):
        print(f"\n  Trip {trip_idx + 1}/{NUM_TRIPS}")

        # ── Bayesian ──────────────────────────────────────────────────────
        for start_sec in range(num_segment):
            xg = x_global_trip[start_sec:start_sec+1]
            xl = x_local_trip[start_sec:start_sec+1]
            elapsed = float(time_spent[start_sec])

            samples = predictive(xg, xl)
            samples_scaled = np.stack(
                [samples[f"obs_section_{i}"].squeeze().cpu().numpy()
                 for i in range(num_segment)],
                axis=1
            )                                               # (NUM_SAMPLES, 9)
            samples_real = loaded_scaler.inverse_transform(samples_scaled)
            remaining_per_sample = samples_real[:, start_sec:].sum(axis=1)
            total_eta_per_sample = elapsed + remaining_per_sample

            bayes_eta_means[start_sec].append(total_eta_per_sample.mean())
            bayes_eta_stds[start_sec].append(total_eta_per_sample.std())
        print(f"    Bayesian ETA at Sec {start_sec}: "
              f"{bayes_eta_means[start_sec][-1]:.1f}s ± "
              f"{bayes_eta_stds[start_sec][-1]:.1f}s (MC uncertainty)")

        # ── Deterministic ─────────────────────────────────────────────────
        with torch.no_grad():
            for start_sec in range(num_segment):
                xg = x_global_trip[start_sec:start_sec+1]
                xl = x_local_trip[start_sec:start_sec+1]
                elapsed = float(time_spent[start_sec])

                preds_list = deterministic_model(xg, xl)
                if isinstance(preds_list, tuple):
                    preds_list = preds_list[0]

                preds_scaled = torch.cat(preds_list, dim=1).cpu().numpy()
                preds_real   = loaded_scaler.inverse_transform(preds_scaled)
                remaining    = preds_real[0, start_sec:].sum()
                total_eta    = elapsed + remaining

                deter_etas[start_sec].append(float(total_eta))

    # ── 5. Aggregate across trips ──────────────────────────────────────────
    print("\n── Cross-trip statistics per starting section ──")
    print(f"{'Sec':>4} | {'Bayes μ(ETA)':>13} {'Bayes σ(ETA)':>13} "
          f"{'Bayes μ(σ_MC)':>14} | {'Deter μ(ETA)':>13} {'Deter σ(ETA)':>13}")
    print("-" * 80)

    bayes_mean_of_means = np.zeros(num_segment)
    bayes_std_of_means  = np.zeros(num_segment)
    bayes_mean_of_stds  = np.zeros(num_segment)   # avg within-trip MC uncertainty
    deter_mean_of_etas  = np.zeros(num_segment)
    deter_std_of_etas   = np.zeros(num_segment)

    for sec in range(num_segment):
        bm = np.array(bayes_eta_means[sec])
        bs = np.array(bayes_eta_stds[sec])
        dm = np.array(deter_etas[sec])

        bayes_mean_of_means[sec] = bm.mean()
        bayes_std_of_means[sec]  = bm.std()
        bayes_mean_of_stds[sec]  = bs.mean()
        deter_mean_of_etas[sec]  = dm.mean()
        deter_std_of_etas[sec]   = dm.std()

        print(f"  {sec:>2} | "
              f"{bm.mean():>12.1f}s "
              f"{bm.std():>12.1f}s "
              f"{bs.mean():>13.1f}s  | "
              f"{dm.mean():>12.1f}s "
              f"{dm.std():>12.1f}s")

    # ── 6. Plot ────────────────────────────────────────────────────────────
    print("\n── Plotting ──")
    plot_cross_trip_variance(
        section_ids       = list(range(num_segment)),
        bayes_mean_of_means = bayes_mean_of_means,
        bayes_std_of_means  = bayes_std_of_means,
        bayes_mean_of_stds  = bayes_mean_of_stds,
        deter_mean_of_etas  = deter_mean_of_etas,
        deter_std_of_etas   = deter_std_of_etas,
        num_sections      = num_segment,
        output_path       = output_chart_path,
    )