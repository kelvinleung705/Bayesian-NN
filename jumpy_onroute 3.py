import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

num_segment = 9

# ==========================================
# 1. MODEL ARCHITECTURE (Must match training exactly)
# ==========================================
class LocalIsolationLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.nets = PyroModuleList([])
        for i in range(num_segments):
            net = PyroModule[nn.Linear](input_dim, output_dim)
            zero      = torch.tensor(0.,  device=device)
            point_one = torch.tensor(1.,  device=device)
            df        = torch.tensor(15., device=device)
            net.weight = PyroSample(dist.StudentT(df, zero, point_one).expand([output_dim, input_dim]).to_event(2))
            net.bias   = PyroSample(dist.StudentT(df, zero, point_one).expand([output_dim]).to_event(1))
            self.nets.append(net)

    def forward(self, x_inputs):
        outputs = []
        for i in range(self.num_segments):
            out = torch.nn.functional.silu(self.nets[i](x_inputs[i]))
            outputs.append(out)
        return outputs


class NeighborMixingLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.num_segments = num_segments

        loc_self = torch.tensor(2.,  device=device)
        loc_side = torch.tensor(0.0, device=device)
        scale    = torch.tensor(1.0, device=device)
        zero     = torch.tensor(0.,  device=device)
        w_scale  = torch.tensor(1.0, device=device)
        b_scale  = torch.tensor(0.1, device=device)

        self.w_self  = PyroSample(dist.Normal(loc_self, scale).expand([num_segments]).to_event(1))
        self.w_right = PyroSample(dist.Normal(loc_side, scale).expand([num_segments]).to_event(1))

        self.nets_1 = PyroModuleList([])
        self.nets_2 = PyroModuleList([])

        for i in range(num_segments):
            net_input_dim = input_dim * 2
            net_1 = PyroModule[nn.Linear](net_input_dim, output_dim)
            net_1.weight = PyroSample(dist.Normal(zero, w_scale).expand([output_dim, net_input_dim]).to_event(2))
            net_1.bias   = PyroSample(dist.Normal(zero, b_scale).expand([output_dim]).to_event(1))
            self.nets_1.append(net_1)

        self.dropout_1 = PyroModule[nn.Dropout](p=dropout_rate)

        for i in range(num_segments):
            net_2 = PyroModule[nn.Linear](output_dim, output_dim)
            net_2.weight = PyroSample(dist.Normal(zero, w_scale).expand([output_dim, output_dim]).to_event(2))
            net_2.bias   = PyroSample(dist.Normal(zero, b_scale).expand([output_dim]).to_event(1))
            self.nets_2.append(net_2)

        self.dropout_2 = PyroModule[nn.Dropout](p=dropout_rate)

    def forward(self, prev_layer_outputs):
        outputs = []
        for i in range(self.num_segments):
            ws = torch.nn.functional.softplus(self.w_self[i])
            wr = torch.nn.functional.softplus(self.w_right[i])
            self_feat = prev_layer_outputs[i] * ws

            right_feat = prev_layer_outputs[i + 1] * wr if i < self.num_segments - 1 else torch.zeros_like(self_feat)

            combined = torch.cat([self_feat, right_feat], dim=1)
            out = self.nets_1[i](combined)
            out = self.dropout_1(out)
            out = torch.nn.functional.silu(out)
            out = self.nets_2[i](out)
            out = self.dropout_2(out)
            out = torch.nn.functional.silu(out)
            outputs.append(out)
        return outputs


class MatrixGNN(PyroModule):
    def __init__(self, num_sections=9, global_dim=9, local_dim=4, hidden_dim=32, device='cuda'):
        super().__init__()
        self.num_sections = num_sections
        input_dim = global_dim + local_dim + 1

        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(3)
        ])

        final_dim = hidden_dim
        self.heads_loc   = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df    = PyroModuleList([])

        for i in range(self.num_sections):
            zero           = torch.tensor(0.,  device=device)
            loc_std        = torch.tensor(1.0, device=device)
            loc_bias_mu    = torch.tensor(0.,  device=device)
            loc_bias_std   = torch.tensor(1.,  device=device)
            scale_std      = torch.tensor(0.3, device=device)
            scale_bias_mu  = torch.tensor(0.,  device=device)
            scale_bias_std = torch.tensor(3.0, device=device)
            df_std         = torch.tensor(1.,  device=device)
            df_bias_mu     = torch.tensor(0.,  device=device)
            df_bias_std    = torch.tensor(3.0, device=device)

            h_loc = PyroModule[nn.Linear](final_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(zero, loc_std).expand([1, final_dim]).to_event(2))
            h_loc.bias   = PyroSample(dist.Normal(loc_bias_mu, loc_bias_std).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)

            h_scale = PyroModule[nn.Linear](final_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(zero, scale_std).expand([1, final_dim]).to_event(2))
            h_scale.bias   = PyroSample(dist.Normal(scale_bias_mu, scale_bias_std).expand([1]).to_event(1))
            self.heads_scale.append(h_scale)

            h_df = PyroModule[nn.Linear](final_dim, 1)
            h_df.weight = PyroSample(dist.Normal(zero, df_std).expand([1, final_dim]).to_event(2))
            h_df.bias   = PyroSample(dist.Normal(df_bias_mu, df_bias_std).expand([1]).to_event(1))
            self.heads_df.append(h_df)

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)

        inputs_list = []
        for i in range(self.num_sections):
            inp = torch.cat([global_features, all_sections_data[:, i, :], accumulated_time[:, i, :]], dim=1)
            inputs_list.append(inp)

        h_current = self.embedding_layer(inputs_list)
        for layer in self.prop_layers:
            h_current = layer(h_current)

        all_locs, all_scales, all_dfs = [], [], []
        for i in range(self.num_sections):
            final_feat = h_current[i]
            loc   = self.heads_loc[i](final_feat)
            scale = torch.nn.functional.softplus(self.heads_scale[i](final_feat)) + 1e-3
            df    = torch.nn.functional.softplus(self.heads_df[i](final_feat)) + 2.5
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)

        return all_locs, all_scales, all_dfs


bnn_model = None  # assigned globally before model_fn is called

def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    # Must match training model_fn exactly
    with pyro.poutine.scale(scale=kl_weight):
        locs, scales, dfs = bnn_model(x_global, x_local)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(dfs[i].squeeze(-1), locs[i].squeeze(-1), scales[i].squeeze(-1))
            target = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)


# ==========================================
# 2. DATA LOADING
# ==========================================
def load_realtime_trip_data(file_path):
    """
    Reads the 9-row Excel file.
    Row s = features when the bus is at the START of section s.
    Column index 18 (0-based) = seconds already spent travelling.
    """
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)

    if len(df) != num_segment:
        print(f"WARNING: Expected {num_segment} rows, found {len(df)}.")

    end_col = 9 + num_segment + 1 + (num_segment * 4)
    df_subset = df.iloc[:, 0:end_col].apply(pd.to_numeric, errors='coerce').fillna(0)
    raw_data_np = df_subset.values.astype(np.float32)

    x_global = torch.tensor(raw_data_np[:, 0:9], dtype=torch.float32)

    raw_local = raw_data_np[:, 9 + num_segment + 1 : 9 + num_segment + 1 + (num_segment * 4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)

    # Column 18 = time already spent (seconds)
    time_spent_array = raw_data_np[:, 18]

    # Raw section times (columns 9 to 9+num_segment) for ground truth
    y_raw = raw_data_np[:, 9 : 9 + num_segment]
    true_total_time = float(np.sum(y_raw[0, :]))  # Full trip time from row 0

    return x_global, x_local, time_spent_array, true_total_time


# ==========================================
# 3. WEIGHT LOADING (Fixed)
# ==========================================
def load_guide_weights(guide, model_file, x_global, x_local, device):
    # Step 1: Prime so guide knows the latent structure
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        guide(x_global[0:1], x_local[0:1], y_true=dummy_y)
    print(f"  Guide primed with {pyro.param('AutoDiagonalNormal.loc').shape[0]} params")

    # Step 2: Now that sizes match, .load() works correctly
    # It loads BOTH params and constraints — manual assignment was missing constraints
    pyro.get_param_store().load(model_file, map_location=device)
    print(f"  Loaded {pyro.param('AutoDiagonalNormal.loc').shape[0]} params from {model_file}")

    # Sanity check — scale should all be positive after constraint transform
    scale_vals = pyro.param('AutoDiagonalNormal.scale')
    assert not torch.isnan(scale_vals).any(), "NaN in scale after load!"
    assert (scale_vals > 0).all(), "Negative scale values after load!"
    print("  Sanity check passed — no NaN, all scales positive.")


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CONFIG ---
    REALTIME_EXCEL_FILE = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy.xlsx"
    MODEL_FILE          = "ghost_bus_model_cycle_0.1_2000_df10_KL_Sample.pt"
    SCALER_FILE         = "y_scaler.pkl"
    NUM_SAMPLES         = 100
    START_SECTION       = 0   # change to e.g. 3 to simulate bus starting at section 3

    # 1. LOAD
    loaded_scaler = joblib.load(SCALER_FILE)
    x_global, x_local, time_spent_array, true_total_time = load_realtime_trip_data(REALTIME_EXCEL_FILE)
    x_global = x_global.to(device)
    x_local  = x_local.to(device)

    # 2. BUILD MODEL & LOAD WEIGHTS
    print("\n--- Loading Bayesian Model ---")
    pyro.clear_param_store()

    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4,
                          hidden_dim=32, device=str(device)).to(device)
    guide = AutoDiagonalNormal(model_fn).to(device)

    load_guide_weights(guide, MODEL_FILE, x_global, x_local, device)

    # 3. INFERENCE
    # Run Predictive once on all 9 rows simultaneously.
    # samples["obs_section_i"] → shape [NUM_SAMPLES, 9]
    # samples["obs_section_i"][:, s] = predictions for section i using features from row s
    print(f"\n--- Running Inference (start_section={START_SECTION}, {NUM_SAMPLES} samples) ---")

    predictive = Predictive(model_fn, guide=guide, num_samples=NUM_SAMPLES)

    with torch.no_grad():
        samples = predictive(x_global, x_local)

    # 4. BUILD ETA CURVES
    active_sections = list(range(START_SECTION, num_segment))
    plot_etas_mean  = []
    plot_etas_std   = []
    plot_etas_upper = []
    plot_etas_lower = []

    for s in active_sections:
        time_spent = float(time_spent_array[s])

        # Collect scaled predictions for ALL 9 sections, using features from row s
        # Shape: [NUM_SAMPLES, num_segment]
        preds_scaled = np.stack(
            [samples[f"obs_section_{i}"][:, s].cpu().numpy() for i in range(num_segment)],
            axis=1
        )

        # Inverse-transform the full [NUM_SAMPLES, 9] matrix at once — correct approach
        preds_real = loaded_scaler.inverse_transform(preds_scaled)  # [NUM_SAMPLES, 9]

        # Sum only the REMAINING sections (s to end) per sample
        remaining_per_sample = preds_real[:, s:].sum(axis=1)  # [NUM_SAMPLES]

        # ETA = time already spent + remaining
        eta_per_sample = time_spent + remaining_per_sample

        plot_etas_mean.append(eta_per_sample.mean())
        plot_etas_std.append(eta_per_sample.std())
        plot_etas_upper.append(np.percentile(eta_per_sample, 95))
        plot_etas_lower.append(np.percentile(eta_per_sample, 5))

        print(f"  Sec {s:2d} | spent={time_spent:6.1f}s | "
              f"remaining={remaining_per_sample.mean():6.1f}±{remaining_per_sample.std():.1f}s | "
              f"ETA={eta_per_sample.mean():6.1f}s")

    plot_etas_mean  = np.array(plot_etas_mean)
    plot_etas_std   = np.array(plot_etas_std)
    plot_etas_upper = np.array(plot_etas_upper)
    plot_etas_lower = np.array(plot_etas_lower)

    # 5. PLOT
    print("\nGenerating convergence plot...")
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(14, 8))

    x_axis           = np.array(active_sections)
    time_spent_active = time_spent_array[START_SECTION:]

    # Ground truth line
    ax.axhline(y=true_total_time, color='#e74c3c', linestyle='--', linewidth=2.5,
               label=f'Actual Total Trip Time ({true_total_time:.1f}s)', zorder=1)

    # 90% predictive interval (5th–95th percentile)
    ax.fill_between(x_axis, plot_etas_lower, plot_etas_upper,
                    color='#3498db', alpha=0.20, label='90% Predictive Interval', zorder=2)

    # ±1 StdDev inner band
    ax.fill_between(x_axis, plot_etas_mean - plot_etas_std, plot_etas_mean + plot_etas_std,
                    color='#3498db', alpha=0.40, label='±1 StdDev Band', zorder=3)

    # Mean ETA trajectory
    ax.plot(x_axis, plot_etas_mean, color='#2c3e50', linewidth=3,
            marker='o', markersize=8, label='Mean ETA Forecast', zorder=4)

    # Time already spent
    ax.plot(x_axis, time_spent_active, color='gray', linestyle=':',
            linewidth=2, marker='x', markersize=8, label='Time Spent So Far', zorder=5)

    ax.set_title('Dynamic Real-Time ETA Convergence\n(Bollinger bands narrow as bus progresses)',
                 fontsize=17, fontweight='bold', pad=18)
    ax.set_xlabel('Bus Location (Start of Section)', fontsize=14)
    ax.set_ylabel('Total Trip Time (Seconds)', fontsize=14)
    ax.set_xlim(active_sections[0], active_sections[-1])
    ax.set_xticks(x_axis)
    ax.set_xticklabels([f"Sec {s}" for s in active_sections], fontsize=12)

    y_min = max(0, min(plot_etas_lower.min(), true_total_time) * 0.92)
    y_max = max(plot_etas_upper.max(), true_total_time) * 1.08
    ax.set_ylim(y_min, y_max)

    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    sns.despine()
    plt.tight_layout()

    out = 'eta_convergence_cone.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()