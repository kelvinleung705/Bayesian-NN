"""
Fast Batched Evaluation — Bayesian Last Layer MatrixGNN
=======================================================
Architecture: deterministic encoder (nn.Module) + Bayesian output heads (PyroSample).
Loads a saved param store (.pt) + scaler (.pkl) and runs vectorised inference.
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
from torch.utils.data import TensorDataset, DataLoader

num_segment = 9


# ==========================================
# 1. DATA LOADING
# ==========================================
def process_validation_data(file_path, loaded_scaler):
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)

    end = 9 + num_segment + 1 + (num_segment * 4) + 2
    df_subset = df.iloc[:, 0:end].dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce').dropna()
    raw = df_subset.values.astype(np.float32)

    x_global = torch.tensor(raw[:, 0:9], dtype=torch.float32)
    raw_local = raw[:, 9 + num_segment + 1: 9 + num_segment + 1 + (num_segment * 4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)

    y_raw = raw[:, 9:9 + num_segment]
    y_scaled = torch.tensor(loaded_scaler.transform(y_raw), dtype=torch.float32)

    print(f"Rows loaded: {raw.shape[0]}")
    print(f"y_raw mean per segment: {y_raw.mean(axis=0)}")
    print(f"y_raw std  per segment: {y_raw.std(axis=0)}")
    return x_global, x_local, y_scaled


# ==========================================
# 2. MODEL ARCHITECTURE  (must match training)
#    Encoder = deterministic nn.Module
#    Heads   = Bayesian PyroModule + PyroSample
# ==========================================
class LocalIsolationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.nets = nn.ModuleList([
            nn.Linear(input_dim, output_dim, device=device)
            for _ in range(num_segments)
        ])

    def forward(self, x_inputs):
        return [
            torch.nn.functional.silu(self.nets[i](x_inputs[i]))
            for i in range(self.num_segments)
        ]


class NeighborMixingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.w_self  = nn.Parameter(torch.full((num_segments,), 2.0, device=device))
        self.w_right = nn.Parameter(torch.zeros(num_segments, device=device))

        net_input_dim = input_dim * 2
        self.nets_1 = nn.ModuleList([
            nn.Linear(net_input_dim, output_dim, device=device)
            for _ in range(num_segments)
        ])
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.nets_2 = nn.ModuleList([
            nn.Linear(output_dim, output_dim, device=device)
            for _ in range(num_segments)
        ])
        self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, prev):
        outputs = []
        for i in range(self.num_segments):
            ws = torch.nn.functional.softplus(self.w_self[i])
            wr = torch.nn.functional.softplus(self.w_right[i])
            self_feat  = prev[i] * ws
            right_feat = prev[i + 1] * wr if i < self.num_segments - 1 else torch.zeros_like(self_feat)

            out = self.nets_1[i](torch.cat([self_feat, right_feat], dim=1))
            out = torch.nn.functional.silu(self.dropout_1(out))
            out = self.nets_2[i](out)
            out = torch.nn.functional.silu(self.dropout_2(out))
            outputs.append(out)
        return outputs


class MatrixGNN(PyroModule):
    def __init__(self, num_sections=9, global_dim=9, local_dim=4,
                 hidden_dim=16, device='cuda'):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        input_dim = global_dim + local_dim + 1

        # Deterministic encoder
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        self.prop_layers = nn.ModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(2)
        ])

        # Bayesian output heads
        final_dim = hidden_dim
        self.heads_loc   = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df    = PyroModuleList([])

        zero = torch.tensor(0., device=device)
        for _ in range(num_sections):
            """
            h_loc = PyroModule[nn.Linear](hidden_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(zero, torch.tensor(1.0, device=device)).expand([1, hidden_dim]).to_event(2))
            h_loc.bias   = PyroSample(dist.Normal(zero, torch.tensor(3.0, device=device)).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)

            h_scale = PyroModule[nn.Linear](hidden_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(zero, torch.tensor(0.3, device=device)).expand([1, hidden_dim]).to_event(2))
            h_scale.bias   = PyroSample(dist.Normal(zero, torch.tensor(3.0, device=device)).expand([1]).to_event(1))
            self.heads_scale.append(h_scale)

            h_df = PyroModule[nn.Linear](hidden_dim, 1)
            h_df.weight = PyroSample(dist.Normal(zero, torch.tensor(1.0, device=device)).expand([1, hidden_dim]).to_event(2))
            h_df.bias   = PyroSample(dist.Normal(zero, torch.tensor(3.0, device=device)).expand([1]).to_event(1))
            self.heads_df.append(h_df)
            """
            
            
            zero = torch.tensor(0., device=device)
            loc_std = torch.tensor(1.0, device=device)
            loc_bias_mu = torch.tensor(0., device=device)
            loc_bias_std = torch.tensor(3., device=device)

            scale_std = torch.tensor(0.3, device=device)    #0.3
            scale_bias_mu = torch.tensor(0.0, device=device) # 從 1.5 開始，避免過度自信    #3.0
            scale_bias_std = torch.tensor(1.0, device=device)   #3.0

            df_std = torch.tensor(1.0, device=device)
            df_bias_mu = torch.tensor(0.0, device=device)
            df_bias_std = torch.tensor(3.0, device=device)
            
            h_loc = PyroModule[nn.Linear](final_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(zero, loc_std).expand([1, final_dim]).to_event(2))
            h_loc.bias = PyroSample(dist.Normal(loc_bias_mu, loc_bias_std).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)
            
            h_scale = PyroModule[nn.Linear](final_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(zero, scale_std).expand([1, final_dim]).to_event(2))
            h_scale.bias = PyroSample(dist.Normal(scale_bias_mu, scale_bias_std).expand([1]).to_event(1))
            self.heads_scale.append(h_scale)
            
            h_df = PyroModule[nn.Linear](final_dim, 1)
            h_df.weight = PyroSample(dist.Normal(zero, df_std).expand([1, final_dim]).to_event(2))
            h_df.bias = PyroSample(dist.Normal(df_bias_mu, df_bias_std).expand([1]).to_event(1))
            self.heads_df.append(h_df)

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        current_time = torch.zeros(batch_size, 1, device=device)
        all_locs, all_scales, all_dfs = [], [], []

        for s in range(self.num_sections):
            inputs_list = []
            for i in range(self.num_sections):
                loc_i  = all_sections_data[:, i, :]
                time_i = current_time if i <= s else torch.zeros(batch_size, 1, device=device)
                time_i = torch.clamp(time_i, -15.0, 15.0)
                inputs_list.append(torch.cat([global_features, loc_i, time_i], dim=1))

            h = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h = layer(h)

            feat  = h[s]
            loc   = self.heads_loc[s](feat)
            scale = torch.nn.functional.softplus(self.heads_scale[s](feat)) + 1e-3
            df    = torch.nn.functional.softplus(self.heads_df[s](feat)) + 2.5

            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            current_time = current_time + loc

        return all_locs, all_scales, all_dfs


# ==========================================
# 3. PYRO MODEL FUNCTION
# ==========================================
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    with pyro.poutine.scale(scale=kl_weight):
        locs, scales, dfs = bnn_model(x_global, x_local)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size,
                    subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            pyro.sample(
                f"obs_section_{i}",
                dist.StudentT(dfs[i].squeeze(-1), locs[i].squeeze(-1), scales[i].squeeze(-1)),
                obs=y_true[:, i] if y_true is not None else None,
            )


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    # ---- config ---------------------------------------------------------------
    SAVED_PARAMS = "ghost_bus_model_bll_claude.pt"
    SAVED_SCALER = "y_scaler_bll_claude.pkl"
    FILE_PATH    = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_new.xlsx"
    HIDDEN_DIM   = 16       # must match training
    NUM_SAMPLES  = 200      # posterior samples per prediction
    BATCH_SIZE   = 1024     # GPU batch size for inference
    # ---------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load scaler & data ---------------------------------------------------
    print("\n--- Loading scaler & data ---")
    loaded_scaler = joblib.load(SAVED_SCALER)
    x_global_val, x_local_val, y_val = process_validation_data(FILE_PATH, loaded_scaler)
    x_global_val = x_global_val.to(device)
    x_local_val  = x_local_val.to(device)
    y_val        = y_val.to(device)

    # ---- rebuild model & load weights -----------------------------------------
    print("\n--- Loading model weights ---")
    pyro.clear_param_store()
    pyro.get_param_store().load(SAVED_PARAMS, map_location=device.type)

    bnn_model  = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4,
                            hidden_dim=HIDDEN_DIM, device=device).to(device)
    base_guide = AutoDiagonalNormal(model_fn).to(device)

    # Warm-up trace: links guide variational params to the loaded param store
    with torch.no_grad():
        dummy_y = torch.zeros(1, num_segment, device=device)
        base_guide(x_global_val[:1], x_local_val[:1], dummy_y, total_size=1)

    bnn_model.eval()
    base_guide.eval()

    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true,
                              total_size=total_size, kl_weight=kl_weight)

    print("Weights loaded. Starting batched inference...")

    # ---- batched inference ----------------------------------------------------
    predictive  = Predictive(model_fn, guide=guide_fn, num_samples=NUM_SAMPLES)
    val_loader  = DataLoader(
        TensorDataset(x_global_val, x_local_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    all_pred_scaled = []   # (N, num_segment)
    all_std_scaled  = []
    all_y_scaled    = []

    with torch.no_grad():
        for batch_idx, (x_g, x_l, y_b) in enumerate(val_loader):
            # samples[obs_section_i] has shape (NUM_SAMPLES, batch_size)
            samples = predictive(x_g, x_l)

            # Stack → (batch_size, num_segment)
            means = torch.stack(
                [samples[f"obs_section_{i}"].squeeze(-1).mean(dim=0)
                 for i in range(num_segment)], dim=1
            ).cpu().numpy()

            stds = torch.stack(
                [samples[f"obs_section_{i}"].squeeze(-1).std(dim=0)
                 for i in range(num_segment)], dim=1
            ).cpu().numpy()

            all_pred_scaled.append(means)
            all_std_scaled.append(stds)
            all_y_scaled.append(y_b.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                done = min((batch_idx + 1) * BATCH_SIZE, len(x_global_val))
                print(f"  {done}/{len(x_global_val)} samples processed...")

    # ---- inverse-transform everything at once ---------------------------------
    pred_scaled   = np.vstack(all_pred_scaled)   # (N, S)
    std_scaled    = np.vstack(all_std_scaled)
    y_np          = np.vstack(all_y_scaled)

    pred_real   = loaded_scaler.inverse_transform(pred_scaled)   # seconds
    actual_real = loaded_scaler.inverse_transform(y_np)
    std_real    = std_scaled * loaded_scaler.scale_              # linear scale

    # ---- vectorised metrics ---------------------------------------------------
    total_pred = pred_real.sum(axis=1)     # (N,)
    total_act  = actual_real.sum(axis=1)
    total_std  = np.sqrt((std_real ** 2).sum(axis=1))

    N = len(total_act)

    # trip-level coverage
    trip_in_bound = (total_act >= total_pred - total_std) & \
                    (total_act <= total_pred + total_std)
    within_bound_count = trip_in_bound.sum()

    # section-level coverage
    sec_in_bound = (actual_real >= pred_real - std_real) & \
                   (actual_real <= pred_real + std_real)
    section_hits = sec_in_bound.sum()                  # total cells
    section_hits_per_trip = sec_in_bound.sum(axis=1)   # (N,) — hits per trip

    # error metrics
    errors     = total_act - total_pred
    abs_errors = np.abs(errors)
    mae        = abs_errors.mean()
    bias       = errors.mean()
    rmse       = np.sqrt((errors ** 2).mean())

    # confidence ratio (prediction / uncertainty)
    valid_mask    = total_std > 0
    conf_ratio    = (total_pred[valid_mask] / total_std[valid_mask]).mean()

    # extreme samples: those where uncertainty > prediction
    extreme_mask  = total_std >= total_pred
    extreme_count = extreme_mask.sum()

    # running std (for trend visibility)
    pred_running_std = pd.Series(total_pred).expanding().std().fillna(0).values
    conf_running_std = pd.Series(total_std).expanding().std().fillna(0).values
    act_running_std  = pd.Series(total_act).expanding().std().fillna(0).values

    # ---- per-sample printout --------------------------------------------------
    PRINT_LIMIT = N   # set smaller for a quick peek, e.g. 20

    # Running counters that mirror the original training-script accumulation
    run_within_bound   = 0          # trip-level hits so far
    run_section_hits   = 0          # section-level hits so far
    run_ratio_sum      = 0.0        # sum of (pred/std) for valid samples
    run_error_total    = 0.0        # signed error accumulator
    run_abs_error      = 0.0        # absolute error accumulator
    valid_loc_sum      = 0.0
    valid_std_sum      = 0.0
    valid_count        = 0
    overload           = 0

    for j in range(PRINT_LIMIT):
        # Update running trip/section counters for every sample (matches original)
        run_within_bound += int(trip_in_bound[j])
        run_section_hits += int(sec_in_bound[j].sum())
        run_error_total  += total_act[j] - total_pred[j]
        run_abs_error    += abs_errors[j]
        if total_std[j] > 0:
            run_ratio_sum += total_pred[j] / total_std[j]

        # Extreme samples: skip detailed printing but keep counters
        if extreme_mask[j]:
            overload += 1
            continue

        valid_loc_sum += total_pred[j]
        valid_std_sum += total_std[j]
        valid_count   += 1

        error_rate         = run_error_total  / (j + 1)
        error_rate_squared = run_abs_error    / (j + 1)

        print(f"\n--- Sample {j} ---")
        for i in range(num_segment):
            within = "YES" if sec_in_bound[j, i] else "NO"
            print(
                f"  Sec {i}: Pred {pred_real[j,i]:.1f}s | "
                f"Actual {actual_real[j,i]:.1f}s | "
                f"Conf +/- {std_real[j,i]:.1f}s | "
                f"Within Bound? {within}"
            )

        print(f"\nTotal ETA: {total_pred[j]:.2f} seconds (Actual: {total_act[j]:.2f})")
        print(f"\nWithin Bound? : {'YES' if trip_in_bound[j] else 'NO'}")
        print(f"Confidence: +/- {total_std[j]:.2f} seconds")
        conf_level = total_pred[j] / total_std[j] if total_std[j] > 0 else 0
        print(f"Confidence Level: {conf_level}")
        print(
            f"\nPrediction Std Deviation: {pred_running_std[j]:.2f} , "
            f"Confidence Std Deviation: {conf_running_std[j]:.2f} , "
            f"Actual Std Deviation: {act_running_std[j]:.2f})"
        )
        print(f"Error: {error_rate_squared}")
        print(f"Error Tendency: {error_rate}")
        print(f"\n")
        print(f"總共 {j + 1} 筆驗證資料中，有 {run_within_bound} 筆落在預測區間內。")
        print(f"平均 {num_segment} Section，有 {run_section_hits / (j + 1)} section 落在預測區間內。")
        print(f"平均置信度指標: {run_ratio_sum / (j + 1)}")

    avg_loc = valid_loc_sum / valid_count if valid_count else 0.0
    avg_std = valid_std_sum / valid_count if valid_count else 0.0

    # ---- final summary --------------------------------------------------------
    print(f"\n==============================================")
    print(f"總共 {N} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
    print(f"平均 {num_segment} Section，有 {section_hits / N:.2f} section 落在預測區間內。")
    print(f"平均置信度指標: {conf_ratio:.2f}")
    print(f"過濾 {overload} 筆極端值後的 平均預測時間: {avg_loc:.2f}")
    print(f"過濾 {overload} 筆極端值後的 平均置信度: {avg_std:.2f}")
    print(f"MAE: {mae:.2f}s  |  RMSE: {rmse:.2f}s  |  Bias: {bias:.2f}s")
    print("==============================================")