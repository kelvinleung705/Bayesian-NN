"""
eta_in_trip_variance.py  (50-trip, ABCD-checkpoint edition)
─────────────────────────────────────────────────────────────
For each of 50 trips (9 rows each = 450 rows total) computes:
 
  Full in-trip std  : std of ETA across all 9 starting sections
  ABCD in-trip std  : std of ETA at checkpoints A/B/C/D only
                      A=sec0, B=sec2, C=sec5, D=sec8
 
Reports per-trip values and averages for both Bayesian & Deterministic.
 
Also saves one overlay chart per trip showing:
  - Bayesian mean ETA with ±1σ / ±2σ Bollinger bands
  - Deterministic ETA line
  - Elapsed time area
"""
 
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
import os
 
# ── Config ─────────────────────────────────────────────────────────────────
num_segment  = 9
NUM_TRIPS    = 50
NUM_SAMPLES  = 50   # Bayesian MC draws per section
 
# A=seg1(idx0)  B=seg3(idx2)  C=seg6(idx5)  D=seg9(idx8)
ABCD_INDICES = [0, 2, 5, 8]
ABCD_LABELS  = {0: "A", 2: "B", 5: "C", 8: "D"}
 
# ── Colour palette (shared across all charts) ──────────────────────────────
BG      = "#0D1117"
PANEL   = "#161B22"
BLUE    = "#58A6FF"
RED     = "#e74c3c"
GREEN   = "#3FB950"
ELAPSED = "#F0883E"
GRID    = "#21262D"
TEXT    = "#E6EDF3"
MUTED   = "#8B949E"
ORANGE  = "#F0883E"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
 
class DeterministicLocalIsolationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments):
        super().__init__()
        self.num_segments = num_segments
        self.nets = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_segments)
        ])
 
    def forward(self, x_inputs):
        return [torch.nn.functional.silu(self.nets[i](x_inputs[i]))
                for i in range(self.num_segments)]
 
 
class DeterministicNeighborMixingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2):
        super().__init__()
        self.num_segments = num_segments
        self.att_nets = nn.ModuleList([
            nn.Linear(input_dim * 2, 2) for _ in range(num_segments)
        ])
        self.nets_1 = nn.ModuleList([
            nn.Linear(input_dim * 2, output_dim) for _ in range(num_segments)
        ])
        self.nets_2 = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_segments)
        ])
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
 
    def forward(self, prev_layer_outputs):
        outputs = []
        for i in range(self.num_segments):
            h_self  = prev_layer_outputs[i]
            h_right = prev_layer_outputs[i+1] if i < self.num_segments - 1 \
                      else torch.zeros_like(h_self)
 
            context = torch.cat([h_self, h_right], dim=1)
            att_w   = torch.nn.functional.softmax(self.att_nets[i](context), dim=1)
            combined = torch.cat([h_self * att_w[:, 0:1],
                                  h_right * att_w[:, 1:2]], dim=1)
 
            out = torch.nn.functional.silu(self.dropout_1(self.nets_1[i](combined)))
            out = torch.nn.functional.silu(self.dropout_2(self.nets_2[i](out)))
            outputs.append(out)
        return outputs
 
 
class DeterministicMatrixGNN(nn.Module):
    def __init__(self, num_sections=9, global_dim=9, local_dim=4, hidden_dim=32):
        super().__init__()
        self.num_sections = num_sections
        self.embedding_layer = DeterministicLocalIsolationLayer(
            global_dim + local_dim + 1, hidden_dim, num_sections)
        self.prop_layers = nn.ModuleList([
            DeterministicNeighborMixingLayer(hidden_dim, hidden_dim,
                                             num_sections, dropout_rate=0.2)
            for _ in range(3)
        ])
        self.heads_loc = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_sections)
        ])
 
    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1, device=device)
        inputs_list = [
            torch.cat([global_features,
                       all_sections_data[:, i, :],
                       accumulated_time[:, i, :]], dim=1)
            for i in range(self.num_sections)
        ]
        h = self.embedding_layer(inputs_list)
        for layer in self.prop_layers:
            h = layer(h)
        return [self.heads_loc[i](h[i]) for i in range(self.num_sections)]
 
 
# ── Bayesian layers ────────────────────────────────────────────────────────
 
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
        z  = torch.tensor(0.0, device=device)
        ws = torch.tensor(1.0, device=device)
        bs = torch.tensor(0.1, device=device)
 
        self.w_self  = PyroSample(dist.Normal(torch.tensor(2.0, device=device),
                                               torch.tensor(1.0, device=device))
                                   .expand([num_segments]).to_event(1))
        self.w_right = PyroSample(dist.Normal(z, torch.tensor(1.0, device=device))
                                   .expand([num_segments]).to_event(1))
 
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
                  if i < self.num_segments - 1 else torch.zeros_like(sf))
            x = self.nets_1[i](torch.cat([sf, rf], dim=1))
            x = torch.nn.functional.silu(self.dropout_1(x))
            x = self.nets_2[i](x)
            x = torch.nn.functional.silu(self.dropout_2(x))
            out.append(x)
        return out
 
 
class MatrixGNN(PyroModule):
    def __init__(self, num_sections=9, global_dim=9, local_dim=4,
                 hidden_dim=32, device="cuda", pnt_1=None):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        self.pnt_1 = pnt_1
        input_dim = global_dim + local_dim + 1
 
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, 0.2, device)
            for _ in range(2)
        ])
 
        self.heads_loc   = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df    = PyroModuleList([])
 
        for i in range(num_sections):
            zero = torch.tensor(0., device=device)
 
            h_loc = PyroModule[nn.Linear](hidden_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(zero, torch.tensor(1.0, device=device)).expand([1, hidden_dim]).to_event(2))
            h_loc.bias   = PyroSample(dist.Normal(zero, torch.tensor(1.0, device=device)).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)
 
            h_sc = PyroModule[nn.Linear](hidden_dim, 1)
            h_sc.weight = PyroSample(dist.Normal(zero, torch.tensor(0.3, device=device)).expand([1, hidden_dim]).to_event(2))
            h_sc.bias   = PyroSample(dist.Normal(zero, torch.tensor(3.0, device=device)).expand([1]).to_event(1))
            self.heads_scale.append(h_sc)
 
            h_df = PyroModule[nn.Linear](hidden_dim, 1)
            h_df.weight = PyroSample(dist.Normal(zero, torch.tensor(1.0, device=device)).expand([1, hidden_dim]).to_event(2))
            h_df.bias   = PyroSample(dist.Normal(zero, torch.tensor(3.0, device=device)).expand([1]).to_event(1))
            self.heads_df.append(h_df)
 
    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        current_time = torch.zeros(batch_size, self.num_sections, device=device)
 
        all_locs, all_scales, all_dfs = [], [], []
 
        for current_section in range(self.num_sections):
            inputs_list = []
            for i in range(self.num_sections):
                loc_i = all_sections_data[:, i, :]
                if i <= current_section:
                    time_i = current_time[:, i:i+1]
                else:
                    time_i = torch.zeros(batch_size, 1, device=device)
                if time_i.abs().mean().item() > 15:
                    time_i = torch.zeros(batch_size, 1, device=device)
                inp = torch.cat([global_features, loc_i, time_i], dim=1)
                inputs_list.append(inp)
 
            h = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h = layer(h)
 
            feat  = h[current_section]
            loc   = self.heads_loc[current_section](feat)
            scale = torch.nn.functional.softplus(self.heads_scale[current_section](feat)) + 1e-3
            df    = torch.nn.functional.softplus(self.heads_df[current_section](feat)) + 2.5
 
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
 
            for j in range(current_section, self.num_sections):
                current_time[:, j] = current_time[:, j] + loc.squeeze(-1)
 
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
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
 
def load_all_trips(file_path, device, num_trips=NUM_TRIPS, n_sec=num_segment):
    df  = pd.read_excel(file_path, header=None, skiprows=1)
    df  = df.apply(pd.to_numeric, errors="coerce").dropna()
    raw = df.values.astype(np.float32)
 
    expected = num_trips * n_sec
    if raw.shape[0] < expected:
        raise ValueError(
            f"Expected {expected} rows ({num_trips} trips × {n_sec} sections) "
            f"but file only has {raw.shape[0]} rows."
        )
 
    local_start = 9 + n_sec + 1   # col 19
    trips = []
    for t in range(num_trips):
        rs  = t * n_sec
        blk = raw[rs : rs + n_sec]
        x_global   = torch.tensor(blk[:, 0:9], dtype=torch.float32, device=device)
        time_spent = blk[:, 18]
        x_local    = torch.tensor(
            blk[:, local_start: local_start + n_sec * 4].reshape(-1, n_sec, 4),
            dtype=torch.float32, device=device,
        )
        trips.append((x_global, x_local, time_spent))
 
    print(f"Loaded {num_trips} trips ({raw.shape[0]} rows).")
    return trips
 
 
# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
 
def bayes_eta_for_trip(x_global_trip, x_local_trip, time_spent,
                        predictive, scaler, n_sec=num_segment):
    """
    Returns
    ───────
    etas      : (n_sec,)            mean total ETA at each starting section
    mc_stds   : (n_sec,)            within-section MC std
    all_samples : (n_sec, NUM_SAMPLES)  raw total-ETA samples per section
    """
    etas, mc_stds, all_samples = [], [], []
    for s in range(n_sec):
        xg      = x_global_trip[s:s+1]
        xl      = x_local_trip[s:s+1]
        elapsed = float(time_spent[s])
 
        samps = predictive(xg, xl)
        samps_scaled = np.stack(
            [samps[f"obs_section_{i}"].squeeze().cpu().numpy() for i in range(n_sec)],
            axis=1,
        )
        samps_real = scaler.inverse_transform(samps_scaled)
        total      = elapsed + samps_real[:, s:].sum(axis=1)
 
        etas.append(total.mean())
        mc_stds.append(total.std())
        all_samples.append(total)
 
    return np.array(etas), np.array(mc_stds), np.array(all_samples)
 
 
def deter_eta_for_trip(x_global_trip, x_local_trip, time_spent,
                        model, scaler, device, n_sec=num_segment):
    etas = []
    with torch.no_grad():
        for s in range(n_sec):
            xg      = x_global_trip[s:s+1]
            xl      = x_local_trip[s:s+1]
            elapsed = float(time_spent[s])
 
            preds = model(xg, xl)
            if isinstance(preds, tuple):
                preds = preds[0]
 
            scaled = torch.cat(preds, dim=1).cpu().numpy()
            real   = scaler.inverse_transform(scaled)
            etas.append(elapsed + real[0, s:].sum())
 
    return np.array(etas)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PER-TRIP OVERLAY CHART  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_trip_overlay(trip_idx, time_spent,
                      bayes_means, bayes_stds,
                      deter_etas,
                      num_sections=num_segment,
                      output_path=None):
    """
    One chart per trip:  Bayesian Bollinger bands + Deterministic line.
 
    Parameters
    ──────────
    trip_idx    : int
    time_spent  : (n_sec,) elapsed seconds at each starting section
    bayes_means : (n_sec,) Bayesian mean total ETA
    bayes_stds  : (n_sec,) Bayesian MC std of total ETA
    deter_etas  : (n_sec,) Deterministic total ETA
    """
    s   = np.arange(num_sections, dtype=float)
    mu  = np.asarray(bayes_means)
    sg  = np.asarray(bayes_stds)
    de  = np.asarray(deter_etas)
    el  = np.asarray(time_spent)
 
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
 
    # section dividers
    for i in range(num_sections):
        lc = ORANGE if i in ABCD_INDICES else GRID
        ax.axvline(i, color=lc, linewidth=0.8 if i in ABCD_INDICES else 0.5,
                   linestyle="--" if i not in ABCD_INDICES else ":",
                   alpha=0.7 if i in ABCD_INDICES else 1.0, zorder=0)
 
    # ── Bayesian Bollinger bands ───────────────────────────────────────────
    ax.fill_between(s, mu - 2*sg, mu + 2*sg,
                    color=BLUE, alpha=0.10, label="Bayes ±2σ", zorder=1)
    ax.fill_between(s, mu - sg,   mu + sg,
                    color=BLUE, alpha=0.22, label="Bayes ±1σ", zorder=2)
 
    ax.plot(s, mu, color=BLUE, linewidth=2.5, zorder=5,
            label="Bayesian mean ETA",
            marker="o", markersize=6,
            markerfacecolor=BG, markeredgecolor=BLUE, markeredgewidth=2)
 
    # ── Deterministic line ─────────────────────────────────────────────────
    ax.plot(s, de, color=RED, linewidth=2.2, linestyle="--", zorder=5,
            label="Deterministic ETA",
            marker="D", markersize=5,
            markerfacecolor=BG, markeredgecolor=RED, markeredgewidth=2)
 
    # ── Elapsed area ───────────────────────────────────────────────────────
    ax.fill_between(s, 0, el,
                    color=ELAPSED, alpha=0.15, label="Elapsed (known)", zorder=1)
    ax.plot(s, el, color=ELAPSED, linewidth=1.6, linestyle=":",
            marker="s", markersize=4,
            markerfacecolor=BG, markeredgecolor=ELAPSED, markeredgewidth=1.6,
            zorder=3)
 
    # ── Remaining bars (Bayesian) ──────────────────────────────────────────
    ax.bar(s, mu - el, bottom=el,
           color=BLUE, alpha=0.10, width=0.35, zorder=1)
 
    # ── ABCD checkpoint labels ─────────────────────────────────────────────
    for idx, lbl in ABCD_LABELS.items():
        ax.annotate(lbl,
                    xy=(idx, max(mu[idx] + 2*sg[idx], de[idx]) + 0),
                    xytext=(0, 22), textcoords="offset points",
                    ha="center", color=ORANGE, fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=BG,
                              ec=ORANGE, alpha=0.7))
 
    # ── ±σ annotations ────────────────────────────────────────────────────
    for i, (xi, yi, si) in enumerate(zip(s, mu, sg)):
        ax.annotate(f"±{si:.0f}s",
                    xy=(xi, yi + 2*si),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7, color=MUTED, fontfamily="monospace")
 
    # ── Diff annotation at each section ───────────────────────────────────
    for i in range(num_sections):
        diff = de[i] - mu[i]
        sign = "+" if diff >= 0 else ""
        ax.annotate(f"{sign}{diff:.0f}s",
                    xy=(s[i], (mu[i] + de[i]) / 2),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center",
                    fontsize=6.5, color=MUTED, fontfamily="monospace",
                    alpha=0.75)
 
    # ── Formatting ─────────────────────────────────────────────────────────
    ax.set_xlabel("Bus position  (starting section)  ▲ = ABCD checkpoint",
                  color=MUTED, fontsize=10, labelpad=10)
    ax.set_ylabel("Total ETA  (seconds)", color=MUTED, fontsize=10, labelpad=10)
    ax.set_title(
        f"Trip {trip_idx + 1:>3}  —  Bayesian (Bollinger) vs Deterministic ETA",
        color=TEXT, fontsize=13, fontweight="bold", pad=14,
    )
 
    sec_labels = [
        f"{'ABCD'[ABCD_INDICES.index(i)]}={i}" if i in ABCD_INDICES else str(i)
        for i in range(num_sections)
    ]
    ax.set_xticks(range(num_sections))
    ax.set_xticklabels(sec_labels, color=MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
 
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/60:.1f} min" if v >= 120 else f"{v:.0f} s"
    ))
 
    # summary stats in legend
    bayes_intrip_std = mu.std()
    deter_intrip_std = de.std()
    ax.legend(
        title=(f"In-trip σ →  Bayes: {bayes_intrip_std:.1f}s   "
               f"Deter: {deter_intrip_std:.1f}s"),
        title_fontsize=8,
        loc="upper right", framealpha=0.30,
        facecolor=PANEL, edgecolor=GRID,
        labelcolor=TEXT, fontsize=9,
    )
 
    fig.text(0.5, 0.01,
             "Bollinger bands = Bayesian MC uncertainty  ·  "
             "Dashed red = deterministic point estimate  ·  "
             "Orange area = elapsed travel time",
             ha="center", color=MUTED, fontsize=7.5)
 
    plt.tight_layout(rect=[0, 0.03, 1, 1])
 
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {output_path}")
    plt.close(fig)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
 
def print_summary(bayes_full_std, deter_full_std,
                  bayes_abcd_std, deter_abcd_std,
                  num_trips=NUM_TRIPS):
    hdr = (f"{'Trip':>5} │ "
           f"{'B full σ':>9} {'D full σ':>9} │ "
           f"{'B ABCD σ':>9} {'D ABCD σ':>9}")
    sep = "─" * len(hdr)
 
    print("\n" + "═"*len(hdr))
    print("  IN-TRIP ETA VARIANCE  (std of ETA estimates across sections)")
    print("  Full = all 9 sections │ ABCD = checkpoints at sec 0,2,5,8")
    print("═"*len(hdr))
    print(hdr)
    print(sep)
    for t in range(num_trips):
        print(f"  {t+1:>3}  │ "
              f"{bayes_full_std[t]:>8.1f}s {deter_full_std[t]:>8.1f}s │ "
              f"{bayes_abcd_std[t]:>8.1f}s {deter_abcd_std[t]:>8.1f}s")
    print(sep)
    print(f"  {'AVG':>3}  │ "
          f"{bayes_full_std.mean():>8.1f}s {deter_full_std.mean():>8.1f}s │ "
          f"{bayes_abcd_std.mean():>8.1f}s {deter_abcd_std.mean():>8.1f}s")
    print(f"  {'STD':>3}  │ "
          f"{bayes_full_std.std():>8.1f}s {deter_full_std.std():>8.1f}s │ "
          f"{bayes_abcd_std.std():>8.1f}s {deter_abcd_std.std():>8.1f}s")
    print("═"*len(hdr) + "\n")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE SUMMARY PLOT  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_results(bayes_etas_all, deter_etas_all, bayes_mc_stds_all,
                 bayes_full_std, deter_full_std,
                 bayes_abcd_std, deter_abcd_std,
                 num_trips=NUM_TRIPS, num_sections=num_segment,
                 output_path="eta_in_trip_variance.png"):
 
    s          = np.arange(num_sections)
    bayes_sec_mean  = bayes_etas_all.mean(axis=0)
    bayes_sec_std   = bayes_etas_all.std(axis=0)
    deter_sec_mean  = deter_etas_all.mean(axis=0)
    deter_sec_std   = deter_etas_all.std(axis=0)
    avg_mc_std      = bayes_mc_stds_all.mean(axis=0)
    avg_b_full = bayes_full_std.mean()
    avg_d_full = deter_full_std.mean()
    avg_b_abcd = bayes_abcd_std.mean()
    avg_d_abcd = deter_abcd_std.mean()
 
    fig = plt.figure(figsize=(16, 15), facecolor=BG)
    gs  = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.35,
                           height_ratios=[1.5, 1.2, 1.1])
    ax_traj = fig.add_subplot(gs[0, :])
    ax_mean = fig.add_subplot(gs[1, 0])
    ax_scat = fig.add_subplot(gs[1, 1])
    ax_bar  = fig.add_subplot(gs[2, 0])
    ax_mc   = fig.add_subplot(gs[2, 1])
 
    sec_labels = [
        f"{'ABCD'[ABCD_INDICES.index(i)]}={i}" if i in ABCD_INDICES else str(i)
        for i in range(num_sections)
    ]
 
    def style(ax, ylabel="Total ETA", show_xticks=True):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9, labelpad=6)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v/60:.1f}m" if v >= 120 else f"{v:.0f}s")
        )
        if show_xticks:
            ax.set_xticks(s)
            ax.set_xticklabels(sec_labels, color=MUTED, fontsize=8)
            for i in ABCD_INDICES:
                ax.axvline(i, color=ORANGE, linewidth=0.9, linestyle=":", alpha=0.6, zorder=0)
            for i in range(num_sections):
                ax.axvline(i, color=GRID, linewidth=0.4, linestyle="--", zorder=0)
 
    style(ax_traj)
    for t in range(num_trips):
        ax_traj.plot(s, bayes_etas_all[t], color=BLUE, alpha=0.18, linewidth=1.0, zorder=2)
        ax_traj.plot(s, deter_etas_all[t], color=RED,  alpha=0.18, linewidth=1.0,
                     linestyle="--", zorder=2)
    ax_traj.plot(s, bayes_sec_mean, color=BLUE, linewidth=2.8,
                 marker="o", markersize=7, markerfacecolor=BG,
                 markeredgecolor=BLUE, markeredgewidth=2, zorder=5,
                 label=f"Bayesian avg   (full σ={avg_b_full:.1f}s, ABCD σ={avg_b_abcd:.1f}s)")
    ax_traj.plot(s, deter_sec_mean, color=RED, linewidth=2.8,
                 marker="o", markersize=7, markerfacecolor=BG,
                 markeredgecolor=RED, markeredgewidth=2, zorder=5, linestyle="--",
                 label=f"Deterministic avg   (full σ={avg_d_full:.1f}s, ABCD σ={avg_d_abcd:.1f}s)")
    for idx, lbl in ABCD_LABELS.items():
        ax_traj.scatter([idx], [bayes_sec_mean[idx]], color=ORANGE, s=90, zorder=6, marker="^")
        ax_traj.annotate(lbl, (idx, bayes_sec_mean[idx]),
                         xytext=(0, 12), textcoords="offset points",
                         color=ORANGE, fontsize=9, fontweight="bold", ha="center")
    ax_traj.set_title(
        f"Total ETA at Each Starting Section – All {num_trips} Trips Overlaid",
        color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax_traj.set_xlabel("Starting section  (▲ = ABCD checkpoint)", color=MUTED, fontsize=10)
    ax_traj.legend(framealpha=0.3, facecolor=PANEL, edgecolor=GRID,
                   labelcolor=TEXT, fontsize=9)
 
    style(ax_mean)
    ax_mean.fill_between(s, bayes_sec_mean-bayes_sec_std, bayes_sec_mean+bayes_sec_std,
                         color=BLUE, alpha=0.18)
    ax_mean.plot(s, bayes_sec_mean, color=BLUE, linewidth=2.2, marker="o",
                 markersize=5, markerfacecolor=BG, markeredgecolor=BLUE,
                 markeredgewidth=2, label="Bayesian")
    ax_mean.fill_between(s, deter_sec_mean-deter_sec_std, deter_sec_mean+deter_sec_std,
                         color=RED, alpha=0.18)
    ax_mean.plot(s, deter_sec_mean, color=RED, linewidth=2.2, marker="o",
                 markersize=5, markerfacecolor=BG, markeredgecolor=RED,
                 markeredgewidth=2, linestyle="--", label="Deterministic")
    ax_mean.set_title("Mean ETA ± Cross-Trip Std per Section",
                      color=TEXT, fontsize=11, pad=8)
    ax_mean.set_xlabel("Starting section", color=MUTED, fontsize=9)
    ax_mean.legend(framealpha=0.3, facecolor=PANEL, edgecolor=GRID,
                   labelcolor=TEXT, fontsize=8)
 
    ax_scat.set_facecolor(PANEL)
    for sp in ax_scat.spines.values(): sp.set_edgecolor(GRID)
    ax_scat.tick_params(colors=MUTED, labelsize=9)
    ax_scat.grid(axis="y", color=GRID, linewidth=0.5)
    ax_scat.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    trip_ids = np.arange(1, num_trips+1)
    ax_scat.scatter(trip_ids, bayes_full_std, color=BLUE,  s=35, alpha=0.85, label="Bayesian full", zorder=4)
    ax_scat.scatter(trip_ids, deter_full_std, color=RED,   s=35, alpha=0.85, marker="D", label="Deter full", zorder=4)
    ax_scat.scatter(trip_ids, bayes_abcd_std, color=BLUE,  s=35, alpha=0.85, marker="^", label="Bayesian ABCD", zorder=4)
    ax_scat.scatter(trip_ids, deter_abcd_std, color=RED,   s=35, alpha=0.85, marker="v", label="Deter ABCD", zorder=4)
    for val, col, ls in [(avg_b_full,BLUE,"-"),(avg_d_full,RED,"-"),(avg_b_abcd,BLUE,":"),(avg_d_abcd,RED,":")]:
        ax_scat.axhline(val, color=col, linewidth=1.4, linestyle=ls, alpha=0.7)
    ax_scat.set_title("In-Trip ETA Std per Trip\n(solid=full 9 secs, dotted=ABCD only)",
                      color=TEXT, fontsize=11, pad=8)
    ax_scat.set_xlabel("Trip index", color=MUTED, fontsize=9)
    ax_scat.set_ylabel("Std of ETA across sections (s)", color=MUTED, fontsize=9)
    ax_scat.legend(framealpha=0.3, facecolor=PANEL, edgecolor=GRID,
                   labelcolor=TEXT, fontsize=7, ncol=2)
 
    ax_bar.set_facecolor(PANEL)
    for sp in ax_bar.spines.values(): sp.set_edgecolor(GRID)
    ax_bar.tick_params(colors=MUTED, labelsize=9)
    ax_bar.grid(axis="y", color=GRID, linewidth=0.5)
    x    = np.array([0, 1, 3, 4])
    vals = [avg_b_full, avg_d_full, avg_b_abcd, avg_d_abcd]
    colors_bar = [BLUE, RED, BLUE, RED]
    alphas = [0.85, 0.85, 0.50, 0.50]
    xlbls = ["Bayes\nFull", "Deter\nFull", "Bayes\nABCD", "Deter\nABCD"]
    bars = ax_bar.bar(x, vals, color=colors_bar, width=0.7, zorder=3)
    for bar, al in zip(bars, alphas): bar.set_alpha(al)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}s",
                    ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(xlbls, color=TEXT, fontsize=10)
    ax_bar.set_title("Avg In-Trip ETA Std\n(lower = more consistent estimates)",
                     color=TEXT, fontsize=11, pad=8)
    ax_bar.set_ylabel("Avg std of ETA (s)", color=MUTED, fontsize=9)
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    ax_bar.set_ylim(0, max(vals)*1.40)
    for model_x, b_val, d_val, col in [([0,1],avg_b_full,avg_d_full,GREEN),
                                        ([3,4],avg_b_abcd,avg_d_abcd,ORANGE)]:
        diff = abs(b_val-d_val)
        ymax = max(b_val, d_val)
        ax_bar.annotate("", xy=(model_x[1], ymax*1.12), xytext=(model_x[0], ymax*1.12),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
        ax_bar.text(sum(model_x)/2, ymax*1.15, f"Δ{diff:.1f}s", ha="center", color=col, fontsize=9)
 
    ax_mc.set_facecolor(PANEL)
    for sp in ax_mc.spines.values(): sp.set_edgecolor(GRID)
    ax_mc.tick_params(colors=MUTED, labelsize=9)
    ax_mc.grid(color=GRID, linewidth=0.5)
    ax_mc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    ax_mc.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    ax_mc.scatter(avg_mc_std, bayes_sec_std, color=BLUE, s=70, zorder=4, alpha=0.9)
    for i in range(num_sections):
        lbl = ABCD_LABELS.get(i, str(i))
        col = ORANGE if i in ABCD_INDICES else MUTED
        ax_mc.annotate(lbl, (avg_mc_std[i], bayes_sec_std[i]),
                       textcoords="offset points", xytext=(6,3),
                       color=col, fontsize=8,
                       fontweight="bold" if i in ABCD_INDICES else "normal")
    lim = max(avg_mc_std.max(), bayes_sec_std.max()) * 1.15
    ax_mc.plot([0, lim], [0, lim], color=GRID, linewidth=1, linestyle=":")
    ax_mc.set_xlim(0, lim); ax_mc.set_ylim(0, lim)
    ax_mc.set_title("Bayesian MC Uncertainty\nvs Cross-Trip ETA Spread per Section",
                    color=TEXT, fontsize=11, pad=8)
    ax_mc.set_xlabel("Avg within-section MC std (model uncertainty)", color=MUTED, fontsize=9)
    ax_mc.set_ylabel("Cross-trip std of mean ETA", color=MUTED, fontsize=9)
    ax_mc.text(lim*0.05, lim*0.88, "Above diagonal:\nreal spread > model\nuncertainty",
               color=MUTED, fontsize=7.5)
 
    fig.suptitle(
        f"In-Trip ETA Convergence Stability  ·  {num_trips} Trips  ·  {num_sections} Sections\n"
        f"Full 9-sec avg σ →  Bayesian: {avg_b_full:.1f}s   Deterministic: {avg_d_full:.1f}s     "
        f"ABCD avg σ →  Bayesian: {avg_b_abcd:.1f}s   Deterministic: {avg_d_abcd:.1f}s",
        color=TEXT, fontsize=13, fontweight="bold", y=0.998,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Aggregate chart saved → {output_path}")
    plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    #saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu4_fixed_10000_new_encoding.pt"
    saved_params_path = "ghost_bus_model_cycle_0.7_clamp_5_29_1526.pt"
    saved_scaler_path  = "y_scaler.pkl"
    trip_snapshot_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy_50.xlsx"
    output_chart_path  = "eta_in_trip_variance.png"
    per_trip_dir       = "eta_per_trip_charts"   # ← folder for n per-trip PNGs
 
    os.makedirs(per_trip_dir, exist_ok=True)
 
    # ── 1. Data ────────────────────────────────────────────────────────────
    scaler = joblib.load(saved_scaler_path)
    trips  = load_all_trips(trip_snapshot_path, device)
 
    # ── 2. Bayesian model ──────────────────────────────────────────────────
    pyro.clear_param_store()
    pyro.get_param_store().load(saved_params_path, map_location=device.type)
 
    bnn_model = MatrixGNN(
        num_sections=num_segment, global_dim=9, local_dim=4,
        hidden_dim=32, device=device,
    ).to(device)
 
    base_guide = AutoDiagonalNormal(model_fn).to(device)
 
    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=0.1):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true,
                              total_size=total_size, kl_weight=kl_weight)
 
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        guide_fn(trips[0][0][0:1], trips[0][1][0:1], y_true=dummy_y)
 
    bnn_model.eval()
    base_guide.eval()
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=NUM_SAMPLES)
 
    # ── 3. Deterministic model ─────────────────────────────────────────────
    det_model = DeterministicMatrixGNN(
        num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32,
    ).to(device)
    det_model.load_state_dict(
        torch.load(saved_params_path, map_location=device), strict=False)
    det_model.eval()
 
    # ── 4. Collect ETAs ────────────────────────────────────────────────────
    bayes_etas_all    = np.zeros((NUM_TRIPS, num_segment))
    bayes_mc_stds_all = np.zeros((NUM_TRIPS, num_segment))
    deter_etas_all    = np.zeros((NUM_TRIPS, num_segment))
 
    for t, (xg_trip, xl_trip, ts) in enumerate(trips):
        print(f"Trip {t+1:>3}/{NUM_TRIPS} ...", end="  ")
 
        etas_b, stds_b, _ = bayes_eta_for_trip(xg_trip, xl_trip, ts, predictive, scaler)
        etas_d            = deter_eta_for_trip(xg_trip, xl_trip, ts, det_model, scaler, device)
 
        bayes_etas_all[t]    = etas_b
        bayes_mc_stds_all[t] = stds_b
        deter_etas_all[t]    = etas_d
 
        b_full_std  = etas_b.std()
        b_full_mean = etas_b.mean()
        d_full_std  = etas_d.std()
        d_full_mean = etas_d.mean()
        b_abcd = etas_b[ABCD_INDICES].std()
        d_abcd = etas_d[ABCD_INDICES].std()
        print(f"Bayes full std={b_full_std:.1f}s, Variability={b_full_std/b_full_mean*100:.1f}% │ "
              f"Deter full std={d_full_std:.1f}s, Variability={d_full_std/d_full_mean*100:.1f}%")
 
        # ── per-trip overlay chart ─────────────────────────────────────────
        out_path = os.path.join(per_trip_dir, f"trip_{t+1:03d}_overlay.png")
        plot_trip_overlay(
            trip_idx    = t,
            time_spent  = ts,
            bayes_means = etas_b,
            bayes_stds  = stds_b,
            deter_etas  = etas_d,
            output_path = out_path,
        )
 
    # ── 5. In-trip std per trip ────────────────────────────────────────────
    bayes_full_std = bayes_etas_all.std(axis=1)
    deter_full_std = deter_etas_all.std(axis=1)
    bayes_abcd_std = bayes_etas_all[:, ABCD_INDICES].std(axis=1)
    deter_abcd_std = deter_etas_all[:, ABCD_INDICES].std(axis=1)
 
    # ── 6. Summary table ───────────────────────────────────────────────────
    print_summary(bayes_full_std, deter_full_std,
                  bayes_abcd_std, deter_abcd_std)
 
    # ── 7. Aggregate plot ──────────────────────────────────────────────────
    plot_results(
        bayes_etas_all    = bayes_etas_all,
        deter_etas_all    = deter_etas_all,
        bayes_mc_stds_all = bayes_mc_stds_all,
        bayes_full_std    = bayes_full_std,
        deter_full_std    = deter_full_std,
        bayes_abcd_std    = bayes_abcd_std,
        deter_abcd_std    = deter_abcd_std,
    )
 
    print(f"\n✓ Done — {NUM_TRIPS} per-trip charts in '{per_trip_dir}/'")