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
    def __init__(self, num_sections=9, global_dim=9, local_dim=4,
                 hidden_dim=32, device="cuda"):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        inp = global_dim + local_dim + 1

        self.embedding_layer = LocalIsolationLayer(inp, hidden_dim, num_sections, device)
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections,
                                dropout_rate=0.2, device=device)
            for _ in range(3)
        ])
        self.heads_loc   = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df    = PyroModuleList([])

        for i in range(num_sections):
            z = torch.tensor(0., device=device)
            h_loc = PyroModule[nn.Linear](hidden_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(z, torch.tensor(1.0, device=device)).expand([1, hidden_dim]).to_event(2))
            h_loc.bias   = PyroSample(dist.Normal(z, torch.tensor(1.0, device=device)).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)

            h_sc = PyroModule[nn.Linear](hidden_dim, 1)
            h_sc.weight = PyroSample(dist.Normal(z, torch.tensor(0.3, device=device)).expand([1, hidden_dim]).to_event(2))
            h_sc.bias   = PyroSample(dist.Normal(z, torch.tensor(3.0, device=device)).expand([1]).to_event(1))
            self.heads_scale.append(h_sc)

            h_df = PyroModule[nn.Linear](hidden_dim, 1)
            h_df.weight = PyroSample(dist.Normal(z, torch.tensor(1.0, device=device)).expand([1, hidden_dim]).to_event(2))
            h_df.bias   = PyroSample(dist.Normal(z, torch.tensor(3.0, device=device)).expand([1]).to_event(1))
            self.heads_df.append(h_df)

    def forward(self, x_global, x_local):
        bs = x_global.shape[0]
        acc = torch.zeros(bs, self.num_sections, 1, device=x_global.device)
        inputs_list = [
            torch.cat([x_global, x_local[:, i, :], acc[:, i, :]], dim=1)
            for i in range(self.num_sections)
        ]
        h = self.embedding_layer(inputs_list)
        for layer in self.prop_layers:
            h = layer(h)

        locs, scales, dfs = [], [], []
        for i in range(self.num_sections):
            f = h[i]
            locs.append(self.heads_loc[i](f))
            scales.append(torch.nn.functional.softplus(self.heads_scale[i](f)) + 1e-3)
            dfs.append(torch.nn.functional.softplus(self.heads_df[i](f)) + 2.5)
        return locs, scales, dfs


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

def plot_eta_convergence(section_ids, means, stds, time_spent,
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

    # ── Mean ETA line ──────────────────────────────────────────────────────
    ax.plot(s, mu, color=ACCENT, linewidth=2.5, zorder=4,
            label="Mean total ETA", marker="o", markersize=6,
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
    ax.plot(s, mu, color=ACCENT2, alpha=0.0)   # invisible, just for legend

    # ── Annotate std at each point ─────────────────────────────────────────
    for i, (xi, yi, si) in enumerate(zip(s, mu, sg)):
        ax.annotate(f"±{si:.0f}s",
                    xy=(xi, yi + 2*sg[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5, color=MUTED,
                    fontfamily="monospace")

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
    saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_Sample.pt"
    saved_scaler_path = "y_scaler.pkl"
    trip_snapshot_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy2_flagged.xlsx"   # your new 9-row file
    #trip_snapshot_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy3_flagged.xlsx"   # your new 9-row file
    output_chart_path  = "eta_convergence.png"

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

    # ── 4. Plot ────────────────────────────────────────────────────────────
    print("\n── Plotting ──")
    plot_eta_convergence(
        section_ids=list(range(num_segment)),
        means=all_means,
        stds=all_stds,
        time_spent=time_spent,
        num_sections=num_segment,
        output_path=output_chart_path,
    )