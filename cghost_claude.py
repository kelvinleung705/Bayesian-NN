import statistics

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList, PyroParam
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyro.optim import PyroLRScheduler
import joblib

num_segment = 9

# ==========================================
# 1. DATA PRE-PROCESSING
# ==========================================
def process_raw_data(file_path):
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)

    end = 9 + num_segment + 1 + (num_segment * 4) + 2
    df_subset = df.iloc[:, 0:end]
    df_subset = df_subset.dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    raw_data_np = df_subset.values.astype(np.float32)

    x_global = torch.tensor(raw_data_np[:, 0:9], dtype=torch.float32)
    raw_local = raw_data_np[:, 9 + num_segment + 1: 9 + num_segment + 1 + (num_segment * 4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)

    y_raw = raw_data_np[:, 9:9 + num_segment]
    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y_raw), dtype=torch.float32)

    print("Total rows loaded:", raw_data_np.shape[0])
    print("Sample y_raw row 0:", y_raw[0])
    print("Sample x_local row 0:", raw_local[0])
    print("y_raw mean per segment:", y_raw.mean(axis=0))
    print("y_raw std per segment:", y_raw.std(axis=0))
    return x_global, x_local, y_scaled, scaler_y


# ==========================================
# 2. DETERMINISTIC ENCODER LAYERS
#    These are plain nn.Module — no PyroSample,
#    no stochastic weights. They are optimised
#    via the Pyro param store exactly like any
#    standard PyTorch module nested inside a
#    PyroModule parent (MatrixGNN).
# ==========================================

class LocalIsolationLayer(PyroModule):          # <-- nn.Module, NOT PyroModule
    """Per-segment input projection. Fully deterministic."""
    def __init__(self, input_dim, output_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.nets = PyroModuleList([
            PyroModule[nn.Linear](input_dim, output_dim).to(device) 
            for _ in range(num_segments)
        ])

    def forward(self, x_inputs):
        return [
            torch.nn.functional.silu(self.nets[i](x_inputs[i]))
            for i in range(self.num_segments)
        ]


class NeighborMixingLayer(PyroModule):          # <-- nn.Module, NOT PyroModule
    """Spatial mixing with deterministic learnable weights. Fully deterministic."""
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.device = device

        # Learnable spatial blend scalars (deterministic nn.Parameter)
        self.w_self  = PyroParam(torch.full((num_segments,), 2.0, device=device))
        self.w_right = PyroParam(torch.zeros(num_segments, device=device))

        net_input_dim = input_dim * 2
        self.nets_1 = PyroModuleList([
            PyroModule[nn.Linear](net_input_dim, output_dim).to(device)
            for _ in range(num_segments)
        ])
        #self.dropout_1 = nn.Dropout(p=dropout_rate)

        self.nets_2 = PyroModuleList([
            PyroModule[nn.Linear](output_dim, output_dim).to(device)
            for _ in range(num_segments)
        ])
        #self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, prev_layer_outputs):
        outputs = []
        for i in range(self.num_segments):
            ws = torch.nn.functional.softplus(self.w_self[i])
            wr = torch.nn.functional.softplus(self.w_right[i])

            self_feat = prev_layer_outputs[i] * ws
            if i < self.num_segments - 1:
                right_feat = prev_layer_outputs[i + 1] * wr
            else:
                right_feat = torch.zeros_like(self_feat)

            combined = torch.cat([self_feat, right_feat], dim=1)

            out = self.nets_1[i](combined)
            #out = self.dropout_1(out)
            out = torch.nn.functional.silu(out)

            out = self.nets_2[i](out)
            #out = self.dropout_2(out)
            out = torch.nn.functional.silu(out)

            outputs.append(out)
        return outputs


# ==========================================
# 3. THE "MATRIX" GNN MODEL
#    MatrixGNN stays PyroModule because it owns
#    the Bayesian output heads (PyroSample).
#    The encoder sub-modules are nn.Module and
#    are stored as normal attributes — Pyro will
#    register their parameters in the param store
#    automatically via PyroModule.__setattr__.
# ==========================================
class MatrixGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4,
                 hidden_dim=8, device='cuda'):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        input_dim = global_dim + local_dim + 1

        # -------------------------------------------------------------------
        # DETERMINISTIC ENCODER
        # Plain nn.Module instances assigned to a PyroModule. Their weights
        # live in Pyro's param store and are updated by the SVI optimizer, but
        # they are NOT sampled — they have a single point estimate, not a
        # posterior distribution. AutoDiagonalNormal will NOT create
        # variational parameters for them.
        # -------------------------------------------------------------------
        self.embedding_layer = LocalIsolationLayer(
            input_dim, hidden_dim, num_sections, device
        )
        self.prop_layers = PyroModuleList([          # <-- PyroModuleList, not nn.ModuleList
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections,
                                dropout_rate=0.2, device=device)
            for _ in range(2)
        ])

        # -------------------------------------------------------------------
        # BAYESIAN DECODER (last layer only)
        # Each head is a PyroModule[nn.Linear] with PyroSample weight and
        # bias. AutoDiagonalNormal will learn a mean+std for every weight
        # and bias in these heads — and only these heads.
        # -------------------------------------------------------------------
        self.heads_loc   = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df    = PyroModuleList([])

        zero = torch.tensor(0., device=device)
        loc_std = torch.tensor(1.0, device=device)
        loc_bias_mu = torch.tensor(0., device=device)
        loc_bias_std = torch.tensor(3., device=device) #1.

        scale_std = torch.tensor(0.3, device=device)
        scale_bias_mu = torch.tensor(0., device=device) #1.0
        scale_bias_std = torch.tensor(1.0, device=device) # 3.0

        df_std = torch.tensor(1., device=device)
        df_bias_mu = torch.tensor(0., device=device)
        df_bias_std = torch.tensor(3.0, device=device)

        for i in range(self.num_sections):
            h_loc = PyroModule[nn.Linear](hidden_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(zero, loc_std).expand([1, hidden_dim]).to_event(2))
            h_loc.bias = PyroSample(dist.Normal(loc_bias_mu, loc_bias_std).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)

            h_scale = PyroModule[nn.Linear](hidden_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(zero, scale_std).expand([1, hidden_dim]).to_event(2))
            h_scale.bias = PyroSample(dist.Normal(scale_bias_mu, scale_bias_std).expand([1]).to_event(1))
            self.heads_scale.append(h_scale)

            h_df = PyroModule[nn.Linear](hidden_dim, 1)
            h_df.weight = PyroSample(dist.Normal(zero, df_std).expand([1, hidden_dim]).to_event(2))
            h_df.bias = PyroSample(dist.Normal(zero, df_bias_std).expand([1]).to_event(1))
            self.heads_df.append(h_df)

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device

        current_time = torch.zeros(batch_size, 1, device=device)
        all_locs, all_scales, all_dfs = [], [], []

        for current_section in range(self.num_sections):
            # Build per-segment input list (autoregressive clock injection)
            inputs_list = []
            for i in range(self.num_sections):
                loc_i = all_sections_data[:, i, :]
                if i <= current_section:
                    time_i = current_time
                else:
                    time_i = torch.zeros(batch_size, 1, device=device)
                time_i = torch.clamp(time_i, min=-15.0, max=15.0)
                inputs_list.append(torch.cat([global_features, loc_i, time_i], dim=1))

            # --- DETERMINISTIC ENCODER FORWARD PASS ---
            h = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h = layer(h)

            # --- BAYESIAN DECODER FORWARD PASS ---
            # Only h[current_section] is fed to the Bayesian heads.
            feat = h[current_section]

            loc   = self.heads_loc[current_section](feat)
            scale = torch.nn.functional.softplus(
                self.heads_scale[current_section](feat)
            ) + 1e-3
            df    = torch.nn.functional.softplus(
                self.heads_df[current_section](feat)
            ) + 2.5 #3.5

            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)

            current_time = current_time + loc

        return all_locs, all_scales, all_dfs


# ==========================================
# 4. MODEL / GUIDE FUNCTIONS
# ==========================================
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    # PyroSample sites (Bayesian head weights) are called inside bnn_model.forward().
    # Wrapping with poutine.scale(kl_weight) up-weights the KL (prior) terms
    # without touching the likelihood, which is declared outside this block.
    with pyro.poutine.scale(scale=kl_weight):
        locs, scales, dfs = bnn_model(x_global, x_local)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size,
                    subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(
                dfs[i].squeeze(), locs[i].squeeze(), scales[i].squeeze()
            )
            target = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)


def get_ll_kl(model_fn, guide, x_g, x_l, y, total_size, kl_weight):
    guide_trace = pyro.poutine.trace(guide).get_trace(
        x_g, x_l, y, total_size=total_size, kl_weight=kl_weight
    )
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model_fn, trace=guide_trace)
    ).get_trace(x_g, x_l, y, total_size=total_size, kl_weight=kl_weight)

    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()

    ll = kl = 0.0
    for name, site in model_trace.nodes.items():
        if site["type"] != "sample":
            continue
        if site["is_observed"]:
            ll += site["log_prob_sum"]
        else:
            if name not in guide_trace.nodes:
                continue
            kl += guide_trace.nodes[name]["log_prob_sum"] - site["log_prob_sum"]

    return ll.item(), kl.item()


# ==========================================
# 5. TRAINING
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_new.xlsx"
    x_global_all, x_local_all, y_all, scaler_y = process_raw_data(file_path)

    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.00001, random_state=42)

    x_global_train = x_global_all[train_idx]
    x_local_train  = x_local_all[train_idx]
    y_train        = y_all[train_idx]

    x_global_val = x_global_all[val_idx].to(device)
    x_local_val  = x_local_all[val_idx].to(device)
    y_val        = y_all[val_idx].to(device)

    pyro.clear_param_store()

    # Instantiate model — encoder is deterministic, only heads are Bayesian
    bnn_model = MatrixGNN(
        num_sections=num_segment, global_dim=9, local_dim=4,
        hidden_dim=16, device=device
    ).to(device)

    # AutoDiagonalNormal will discover PyroSample sites in model_fn and
    # create one mean + one std variational parameter per scalar weight.
    # Encoder weights (nn.Parameter) are NOT discovered — they are updated
    # directly by the optimizer as regular parameters via Pyro's param store.
    base_guide = AutoDiagonalNormal(model_fn).to(device)

    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true,
                              total_size=total_size, kl_weight=kl_weight)

    CYCLE_LENGTH = 1250

    optimizer_args = {
        "optimizer": torch.optim.AdamW,
        "optim_args": {"lr": 0.001, "weight_decay": 0.0},
    }

    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=CYCLE_LENGTH, T_mult=1, eta_min=0.0001
        )

    scheduler = PyroLRScheduler(scheduler_constructor, optimizer_args)
    svi = SVI(model_fn, guide_fn, scheduler, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    epochs     = 4000
    batch_size = 734
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_size    = len(train_dataset)
    print(f"Training dataset size: {total_size}")

    for epoch in range(epochs):
        epoch_loss = epoch_ll = epoch_kl = 0.0

        relative_epoch = epoch % CYCLE_LENGTH
        ramp_epochs    = 625
        max_beta       = 0.5
        if relative_epoch < ramp_epochs:
            current_kl_weight = max(0.00001, (relative_epoch / ramp_epochs) * max_beta)
        else:
            current_kl_weight = max_beta

        for x_g_batch, x_l_batch, y_batch in train_loader:
            x_g_batch = x_g_batch.to(device)
            x_l_batch = x_l_batch.to(device)
            y_batch   = y_batch.to(device)

            loss = svi.step(x_g_batch, x_l_batch, y_batch,
                            total_size=total_size, kl_weight=current_kl_weight)
            epoch_loss += loss

        scheduler.step()

        with torch.no_grad():
            ll, kl = get_ll_kl(
                model_fn, guide_fn,
                x_g_batch, x_l_batch, y_batch,
                total_size=total_size, kl_weight=current_kl_weight,
            )
            epoch_ll += ll
            epoch_kl += kl

        if epoch % 100 == 0 or epoch == epochs - 1:
            current_lr = list(scheduler.optim_objs.values())[0].optimizer.param_groups[0]["lr"]
            avg_loss   = epoch_loss / len(train_loader)
            print(
                f"Epoch {epoch:05d} | LR: {current_lr:.6f} | KL Wt: {current_kl_weight:.3f} | "
                f"ELBO: {avg_loss:.2f} | LL: {epoch_ll:.2f} | Beta*KL: {epoch_kl:.2f} | "
                f"KL: {epoch_kl / current_kl_weight:.2f}"
            )

    pyro.get_param_store().save("ghost_bus_model_bll_claude.pt")
    joblib.dump(scaler_y, "y_scaler_bll_claude.pkl")
    print("\nModel and scaler saved.")

    # ==========================================
    # 6. INFERENCE
    # ==========================================
    bnn_model.eval()
    base_guide.eval()

    predictive = Predictive(model_fn, guide=guide_fn, num_samples=50)

    list_of_predict    = []
    list_of_confidence = []
    list_of_actual     = []
    list_of_predict_sections    = [[] for _ in range(num_segment)]
    list_of_confidence_sections = [[] for _ in range(num_segment)]
    list_of_actual_sections     = [[] for _ in range(num_segment)]

    within_bound_count       = 0
    number_of_ratio          = 0
    section_within_bound_counts = 0
    error_abs_total = error_rate_squared = error_total = 0

    for j in range(len(x_global_val)):
        val_x_g = x_global_val[j:j + 1]
        val_x_l = x_local_val[j:j + 1]

        samples = predictive(val_x_g, val_x_l)

        pred_means_scaled, pred_stds_scaled, actuals_scaled = [], [], []
        for i in range(num_segment):
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            pred_means_scaled.append(sec_samples.mean().item())
            pred_stds_scaled.append(sec_samples.std().item())
            actuals_scaled.append(y_val[j, i].item())

        pred_real   = scaler_y.inverse_transform([pred_means_scaled])[0]
        actual_real = scaler_y.inverse_transform([actuals_scaled])[0]
        std_real    = np.array(pred_stds_scaled) * scaler_y.scale_

        total_pred = 0
        trip_section_within_bound = 0

        print(f"\n--- Sample {j} ---")
        for i in range(num_segment):
            list_of_predict_sections[i].append(pred_real[i])
            sec_pred_std = (statistics.pvariance(list_of_predict_sections[i]) ** 0.5
                            if len(list_of_predict_sections[i]) > 1 else 0.0)
            list_of_confidence_sections[i].append(std_real[i])
            sec_conf_std = (statistics.pvariance(list_of_confidence_sections[i]) ** 0.5
                            if len(list_of_confidence_sections[i]) > 1 else 0.0)
            list_of_actual_sections[i].append(actual_real[i])
            sec_act_std = (statistics.pvariance(list_of_actual_sections[i]) ** 0.5
                           if len(list_of_actual_sections[i]) > 1 else 0.0)

            within = (pred_real[i] - std_real[i]) <= actual_real[i] <= (pred_real[i] + std_real[i])
            print(
                f"  Sec {i}: Pred {pred_real[i]:.1f}s | Actual {actual_real[i]:.1f}s | "
                f"Conf +/- {std_real[i]:.1f}s | Pred Dev {sec_pred_std:.1f}s | "
                f"Conf Dev {sec_conf_std:.1f}s | Act Dev {sec_act_std:.1f}s | "
                f"{'YES' if within else 'NO'}"
            )
            total_pred += pred_real[i]
            if within:
                trip_section_within_bound += 1

        section_within_bound_counts += trip_section_within_bound

        total_act = actual_real.sum()
        total_std = np.sqrt(np.sum(std_real ** 2))

        if (total_pred - total_std) <= total_act <= (total_pred + total_std):
            within_bound_count += 1

        list_of_predict.append(total_pred)
        pred_std_dev = (statistics.pvariance(list_of_predict) ** 0.5
                        if len(list_of_predict) > 1 else 0.0)
        list_of_confidence.append(total_std)
        conf_std_dev = (statistics.pvariance(list_of_confidence) ** 0.5
                        if len(list_of_confidence) > 1 else 0.0)
        list_of_actual.append(total_act)
        act_std_dev = (statistics.pvariance(list_of_actual) ** 0.5
                       if len(list_of_actual) > 1 else 0.0)

        if total_std > 0:
            number_of_ratio += total_pred / total_std

        error_total     += (total_act - total_pred)
        error_abs_total += abs(total_act - total_pred)
        error_rate           = error_total / (j + 1)
        error_rate_squared   = error_abs_total / (j + 1)

        print(f"\nTotal ETA: {total_pred:.2f}s  (Actual: {total_act:.2f}s)")
        print(f"Within bound? {'YES' if (total_pred - total_std) <= total_act <= (total_pred + total_std) else 'NO'}")
        print(f"Confidence: +/- {total_std:.2f}s")
        print(f"Pred Dev: {pred_std_dev:.2f} | Conf Dev: {conf_std_dev:.2f} | Act Dev: {act_std_dev:.2f}")
        print(f"MAE: {error_rate_squared:.2f}  |  Error Tendency: {error_rate:.2f}")
        print(f"\n{j + 1} 筆驗證，{within_bound_count} 筆落在預測區間。")
        print(f"平均 section 命中: {section_within_bound_counts / len(x_global_val):.2f}")
        print(f"平均置信度指標: {number_of_ratio / len(x_global_val):.2f}")