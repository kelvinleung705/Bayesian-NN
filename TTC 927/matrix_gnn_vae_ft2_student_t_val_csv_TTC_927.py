import statistics
import math
import os

from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceMeanField_ELBO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyro.optim import ExponentialLR
import joblib

num_segment = 9
LATENT_DIM = 64   # z in R^64

# ==========================================
# 1. DATA PRE-PROCESSING
# ==========================================
def process_raw_data(file_path, scaler_y=None):
    """
    Reads either .csv, .xlsx, or .xls files.
    If scaler_y is provided, transforms y using it (useful for validation).
    Otherwise, fits a new StandardScaler.
    """
    print(f"Reading {file_path}...")
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, header=None, skiprows=1)
    elif ext == '.csv':
        df = pd.read_csv(file_path, header=None, skiprows=1)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Must be .csv, .xlsx, or .xls")

    end = 9 + num_segment + 1 + (num_segment * 4) + 2
    df_subset = df.iloc[:, 0:end]
    df_subset = df_subset.dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    raw_data_np = df_subset.values.astype(np.float32)

    x_global = torch.tensor(raw_data_np[:, 0:9], dtype=torch.float32)
    raw_local = raw_data_np[:, 9 + num_segment + 1 : 9 + num_segment + 1 + (num_segment * 4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)

    y_raw = raw_data_np[:, 9 : 9 + num_segment].sum(axis=1)
    y_raw_2d = y_raw.reshape(-1, 1)

    if scaler_y is None:
        scaler_y = StandardScaler()
        y_scaled_2d = scaler_y.fit_transform(y_raw_2d)
    else:
        y_scaled_2d = scaler_y.transform(y_raw_2d)

    y_scaled = torch.tensor(y_scaled_2d, dtype=torch.float32).squeeze(-1)

    print(f"Loaded {raw_data_np.shape[0]} rows from {file_path}")
    print(f"y_raw mean: {y_raw.mean():.2f} | y_raw std: {y_raw.std():.2f}")
    return x_global, x_local, y_scaled, scaler_y, y_raw


# ==========================================
# 2. ENCODER phi
# ==========================================
class LocalIsolationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.nets = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in range(num_segments)
        ])

    def forward(self, x_inputs):
        return [torch.nn.functional.silu(self.nets[i](x_inputs[i]))
                for i in range(self.num_segments)]


class NeighborMixingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        net_in = input_dim * 2

        self.w_self  = nn.Parameter(torch.full((num_segments,), 1.0))
        self.w_right = nn.Parameter(torch.full((num_segments,), 0.0))

        self.nets_1 = nn.ModuleList([
            nn.Linear(net_in, output_dim) for _ in range(num_segments)
        ])
        self.nets_2 = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_segments)
        ])

        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, prev):
        out = []
        for i in range(self.num_segments):
            ws = torch.nn.functional.softplus(self.w_self[i])
            wr = torch.nn.functional.softplus(self.w_right[i])
            self_feat  = prev[i] * ws
            right_feat = prev[i+1] * wr if i < self.num_segments - 1 else torch.zeros_like(self_feat)

            h = torch.cat([self_feat, right_feat], dim=1)
            h = self.dropout_1(self.nets_1[i](h))
            h = torch.nn.functional.silu(h)
            h = self.dropout_2(self.nets_2[i](h))
            h = torch.nn.functional.silu(h)
            out.append(h)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        self.gate   = nn.Linear(hidden_dim, 1, bias=True)
        self.v_proj = nn.Linear(hidden_dim, self.out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def forward(self, h_stack: torch.Tensor, return_weights: bool = False):
        scores  = self.gate(h_stack).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        weights = self.dropout(weights)

        v = self.v_proj(h_stack)
        pooled = (weights.unsqueeze(-1) * v).sum(dim=1)

        if return_weights:
            return pooled, weights
        return pooled


class EncoderPhi(nn.Module):
    def __init__(self, global_dim, local_dim, hidden_dim, num_segments, latent_dim, device='cuda'):
        super().__init__()
        self.head_dropout = nn.Dropout(p=0.1)
        self.num_segments = num_segments
        self.latent_dim   = latent_dim
        input_dim = global_dim + local_dim

        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_segments, device)
        self.prop_layers = nn.ModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_segments, dropout_rate=0.2, device=device)
            for _ in range(2)
        ])
        self.attn_pool = AttentionPooling(hidden_dim, hidden_dim, dropout=0.1)

        pool_dim = hidden_dim
        self.head_mu     = nn.Linear(pool_dim, latent_dim)
        self.head_logvar = nn.Linear(pool_dim, latent_dim)

        nn.init.constant_(self.head_logvar.bias, -4.0)
        nn.init.xavier_uniform_(self.head_mu.weight, gain=1.0)
        nn.init.xavier_uniform_(self.head_logvar.weight, gain=1.0)
        nn.init.zeros_(self.head_mu.bias)

    def forward(self, global_features, all_sections_data, return_attn: bool = False):
        inputs_list = [
            torch.cat([global_features, all_sections_data[:, i, :]], dim=1)
            for i in range(self.num_segments)
        ]

        h = self.embedding_layer(inputs_list)
        for layer in self.prop_layers:
            h = layer(h)

        h_stack = torch.stack(h, dim=1)

        if return_attn:
            h_pool, attn_weights = self.attn_pool(h_stack, return_weights=True)
        else:
            h_pool = self.attn_pool(h_stack)

        h_pool = self.head_dropout(h_pool)
        mu     = self.head_mu(h_pool)
        logvar = self.head_logvar(h_pool)

        if return_attn:
            return mu, logvar, attn_weights
        return mu, logvar


# ==========================================
# 3. BOTTLENECK
# ==========================================
def gaussian_reparameterise(mu, logvar, force_sample=False):
    sigma = torch.exp(0.5 * logvar)
    if not torch.is_grad_enabled() and not force_sample:
        return mu, sigma
    eps = torch.randn_like(mu)
    return mu + eps * sigma, sigma


def kl_gaussian_to_standard_normal(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1)


# ==========================================
# 4. DECODER theta
# ==========================================
class DecoderTheta(nn.Module):
    def __init__(self, latent_dim, num_segments, global_dim=9, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.latent_dim   = latent_dim
        self.drop         = nn.Dropout(0.1)

        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim + global_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            self.drop
        )
        self.tower_a = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            self.drop,
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
        )
        self.tower_b = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            self.drop,
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
        )

        self.heads_a  = nn.Linear(8, 1)
        self.heads_b  = nn.Linear(8, 1)
        self.heads_df = nn.Linear(8, 1)

        nn.init.xavier_uniform_(self.heads_a.weight, gain=1.0)
        nn.init.xavier_uniform_(self.heads_b.weight, gain=1.5)
        nn.init.constant_(self.heads_b.bias, 1.0)

        nn.init.xavier_uniform_(self.heads_df.weight, gain=0.1)
        nn.init.constant_(self.heads_df.bias, 2.0)

    def forward(self, z, x_global):
        dec_in = torch.cat([z, x_global], dim=-1)
        initial_h = self.dec1(dec_in)

        h_a = self.tower_a(initial_h)
        a_e = self.heads_a(h_a)

        h_b = self.tower_b(initial_h)
        b_e = self.heads_b(h_b)
        b_t = torch.exp(0.5 * torch.clamp(b_e, min=-10.0, max=10.0)) + 1e-3

        df_e = self.heads_df(h_b)
        #nu = torch.nn.functional.softplus(df_e) + 2.05
        nu = 4.1 + 13 * torch.sigmoid(df_e)

        return (a_e, b_t, nu)


# ==========================================
# 5. FULL MODEL: MatrixGNN_VAE
# ==========================================
class MatrixGNN_VAE(nn.Module):
    def __init__(self, num_sections=9, global_dim=9, local_dim=4,
                 hidden_dim=13, latent_dim=LATENT_DIM, device='cuda'):
        super().__init__()
        self.num_sections = num_sections
        self.latent_dim   = latent_dim
        self.device       = device

        self.encoder = EncoderPhi(
            global_dim=global_dim,
            local_dim=local_dim,
            hidden_dim=hidden_dim,
            num_segments=num_sections,
            latent_dim=latent_dim,
            device=device,
        )

        self.decoder = DecoderTheta(
            latent_dim=latent_dim,
            num_segments=num_sections,
            device=device,
        )

    def forward(self, global_features, all_sections_data, return_attn=False, force_sample=False, kl_weight=1.0):
        if return_attn:
            mu, logvar, attn_weights = self.encoder(global_features, all_sections_data, return_attn=True)
        else:
            mu, logvar = self.encoder(global_features, all_sections_data)

        z, sigma = gaussian_reparameterise(mu, logvar, force_sample=force_sample)
        kl = kl_gaussian_to_standard_normal(mu, logvar)

        locs, scalers, dfs = self.decoder(z, global_features)

        if return_attn:
            return locs, scalers, dfs, kl, attn_weights
        return locs, scalers, dfs, kl


# ==========================================
# 6. PYRO MODEL & GUIDE
# ==========================================
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    pyro.module("decoder", bnn_model.decoder)
    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0]):
        z_prior_mu = x_global.new_zeros(x_global.shape[0], bnn_model.latent_dim)
        z_prior_sigma = x_global.new_ones(x_global.shape[0], bnn_model.latent_dim)

        with pyro.poutine.scale(scale=kl_weight):
            z = pyro.sample("latent_z", dist.Normal(z_prior_mu, z_prior_sigma).to_event(1))

        a_e, b_t, df = bnn_model.decoder(z, x_global)

        pyro.sample(
            "obs",
            dist.StudentT(df.squeeze(-1), a_e.squeeze(-1), b_t.squeeze(-1)),
            obs=y_true
        )


def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    pyro.module("encoder", bnn_model.encoder)
    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0]):
        mu, logvar = bnn_model.encoder(x_global, x_local)
        sigma = torch.exp(0.5 * logvar)

        with pyro.poutine.scale(scale=kl_weight):
            pyro.sample("latent_z", dist.Normal(mu, sigma).to_event(1))


# ==========================================
# 7. METRICS EVALUATION
# ==========================================
def get_ll_kl(x_g, x_l, y):
    with torch.no_grad():
        locs, scales, dfs, kl = bnn_model(x_g, x_l)
        loc   = locs.squeeze(-1)
        scale = scales.squeeze(-1)
        df    = dfs.squeeze(-1)

        d  = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
        ll = d.log_prob(y[:]).sum().item()
        kl = kl.mean().item()
    return ll, kl


def evaluate_metrics(model, val_loader, scaler_y, device, n_samples=50):
    model.eval()
    all_preds = []
    all_actuals = []
    total_nll = 0.0
    total_crps = 0.0
    total_samples_count = 0
    within_ci_count = 0

    # 取得原始尺度轉換所需的乘數 (標準差)
    scaler_mean_t = torch.tensor(scaler_y.mean_[0], device=device, dtype=torch.float32)
    scaler_scale_t = torch.tensor(scaler_y.scale_[0], device=device, dtype=torch.float32)
    scale_factor = scaler_y.scale_[0]

    with torch.no_grad():
        for x_g_b, x_l_b, y_b in val_loader:
            x_g_b = x_g_b.to(device)
            x_l_b = x_l_b.to(device)
            y_b = y_b.to(device)
            B = y_b.size(0)

            mu, logvar = model.encoder(x_g_b, x_l_b)
            sigma = torch.exp(0.5 * logvar)

            # MC 採樣計算分佈特徵
            batch_locs, batch_scales, batch_dfs = [], [], []
            batch_y_samples = []  # 用於 CRPS 計算的採樣值
            for _ in range(n_samples):
                eps = torch.randn_like(mu)
                z = mu + eps * sigma
                a_e, b_t, df = model.decoder(z, x_g_b)
                
                loc_i = a_e.squeeze(-1)
                scale_i = b_t.squeeze(-1)
                df_i = df.squeeze(-1)

                batch_locs.append(loc_i)
                batch_scales.append(scale_i)
                batch_dfs.append(df_i)

                # 從預測的 Student-t 分佈中抽樣出具體的可能時間 y
                dist_sample = torch.distributions.StudentT(
                    df=df_i, loc=loc_i, scale=scale_i
                )
                batch_y_samples.append(dist_sample.sample())

            
            # ----------------------------------------------------
            # 1. 計算 CRPS (在真實秒數空間中向量化高速計算)
            # ----------------------------------------------------
            # y_samples_scaled: [n_samples, B]
            y_samples_scaled = torch.stack(batch_y_samples, dim=0)

            # 還原為真實秒數
            y_samples_real = (y_samples_scaled * scaler_scale_t + scaler_mean_t)  # [n_samples, B]
            y_true_real_t = y_b * scaler_scale_t + scaler_mean_t  # [B]

            # Term 1: E|Y_pred - y| (第一項：預測採樣與真實值的平均距離)
            term1 = torch.abs(y_samples_real - y_true_real_t.unsqueeze(0)).mean(dim=0)  # [B]

            # Term 2: 0.5 * E|Y_pred - Y'_pred| (第二項：預測採樣之間的內部多樣性距離)
            # 使用 O(M log M) 的排序向量化技巧取代雙重迴圈，運算速度快 100 倍
            sorted_samples, _ = torch.sort(y_samples_real, dim=0)  # [M, B]
            M = n_samples
            idx = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(1)  # [M, 1]
            weights = 2 * idx - M + 1  # 係數矩陣
            term2 = (sorted_samples * weights).sum(dim=0) / (M * M)  # [B]

            # 當前 Batch 的平均 CRPS (單位: 秒)
            batch_crps = term1 - term2
            total_crps += batch_crps.sum().item()
            
            mean_loc = torch.stack(batch_locs).mean(dim=0)
            mean_scale = torch.stack(batch_scales).mean(dim=0)
            mean_df = torch.stack(batch_dfs).mean(dim=0)


            # 計算 NLL (負對數似然)
            dist_val = torch.distributions.StudentT(df=mean_df, loc=mean_loc, scale=mean_scale)
            total_nll += -dist_val.log_prob(y_b).sum().item()
            total_samples_count += B #y_b.size(0) 

            # 計算 68.3% (1-sigma) CI 覆蓋率 (還原到原始秒數空間)
            y_real = scaler_y.inverse_transform(y_b.cpu().numpy().reshape(-1, 1)).ravel()
            pred_real = scaler_y.inverse_transform(mean_loc.cpu().numpy().reshape(-1, 1)).ravel()
            scale_real = mean_scale.cpu().numpy() * scale_factor

            # Student-t 的 68.3% 區間半寬約為 t_factor * scale
            # 當 df > 4 時，1-sigma 的臨界值約在 1.05 ~ 1.15 之間
            ci_half_width = scale_real * 1.00
            within_ci = (y_real >= (pred_real - ci_half_width)) & (y_real <= (pred_real + ci_half_width))
            within_ci_count += np.sum(within_ci)

            all_preds.extend(mean_loc.cpu().numpy())
            all_actuals.extend(y_b.cpu().numpy())

    model.train()

    preds_real = scaler_y.inverse_transform(np.array(all_preds).reshape(-1, 1)).ravel()
    actuals_real = scaler_y.inverse_transform(np.array(all_actuals).reshape(-1, 1)).ravel()

    mae = np.mean(np.abs(actuals_real - preds_real))
    rmse = np.sqrt(np.mean((actuals_real - preds_real) ** 2))
    mape = np.mean(np.abs((actuals_real - preds_real) / np.clip(np.abs(actuals_real), 1e-6, None))) * 100.0
    avg_nll = total_nll / total_samples_count
    coverage = (within_ci_count / total_samples_count) * 100.0
    avg_crps = total_crps / total_samples_count  # 新增：全局平均 CRPS (秒)

    return rmse, mae, mape, avg_nll, coverage, avg_crps


# Add this function to the bottom of your existing vae_model.py

import os
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import PyroLRScheduler


def train_single_run(
    run_id,
    train_file,
    val_file,
    output_dir,
    epochs=4000,
    batch_size=734,
    delta=0.6,
    device_name="cuda",
):
    """
    Trains a single model instance and logs validation metrics to a dedicated folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"[{run_id}] Running on device: {device}")

    # 1. Load Data
    x_g_train, x_l_train, y_train, scaler_y, _ = process_raw_data(train_file)

    if os.path.exists(val_file):
        x_g_val, x_l_val, y_val, _, _ = process_raw_data(val_file, scaler_y=scaler_y)
    else:
        print(f"[{run_id}] Validation file not found. Performing 90/10 split...")
        idx = np.arange(x_g_train.shape[0])
        t_idx, v_idx = train_test_split(idx, test_size=0.1, random_state=42)
        x_g_val, x_l_val, y_val = x_g_train[v_idx], x_l_train[v_idx], y_train[v_idx]
        x_g_train, x_l_train, y_train = x_g_train[t_idx], x_l_train[t_idx], y_train[t_idx]

    train_loader = DataLoader(TensorDataset(x_g_train, x_l_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_g_val, x_l_val, y_val), batch_size=256, shuffle=False)

    # 2. Build Model (Isolate parameter store per process)
    pyro.clear_param_store()
    model = MatrixGNN_VAE(
        num_sections=num_segment,
        global_dim=9,
        local_dim=4,
        hidden_dim=64,
        latent_dim=LATENT_DIM,
        device=device,
    ).to(device)

    # 3. Setup Closures for Pyro
    def local_model_fn(x_g, x_l, y=None, total_size=None, kl_weight=1.0):
        pyro.module(f"decoder_{run_id}", model.decoder)
        if total_size is None:
            total_size = x_g.shape[0]
        with pyro.plate(f"data_{run_id}", size=total_size, subsample_size=x_g.shape[0]):
            z_mu = x_g.new_zeros(x_g.shape[0], model.latent_dim)
            z_sigma = x_g.new_ones(x_g.shape[0], model.latent_dim)
            with pyro.poutine.scale(scale=kl_weight):
                z = pyro.sample("latent_z", dist.Normal(z_mu, z_sigma).to_event(1))
            a_e, b_t, df = model.decoder(z, x_g)
            pyro.sample("obs", dist.StudentT(df.squeeze(-1), a_e.squeeze(-1), b_t.squeeze(-1)), obs=y)

    def local_guide_fn(x_g, x_l, y=None, total_size=None, kl_weight=1.0):
        pyro.module(f"encoder_{run_id}", model.encoder)
        if total_size is None:
            total_size = x_g.shape[0]
        with pyro.plate(f"data_{run_id}", size=total_size, subsample_size=x_g.shape[0]):
            mu, logvar = model.encoder(x_g, x_l)
            sigma = torch.exp(0.5 * logvar)
            with pyro.poutine.scale(scale=kl_weight):
                pyro.sample("latent_z", dist.Normal(mu, sigma).to_event(1))

    CYCLE_LENGTH = 1000
    optimizer_args = {"optimizer": torch.optim.AdamW, "optim_args": {"lr": 0.001, "weight_decay": 0.0001}}
    scheduler = PyroLRScheduler(
        lambda opt: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=CYCLE_LENGTH, T_mult=1, eta_min=0.0001),
        optimizer_args,
    )
    svi = SVI(local_model_fn, local_guide_fn, scheduler, loss=Trace_ELBO())

    metrics_log_path = os.path.join(output_dir, "metrics_log.csv")
    metrics_records = []

    total_size = len(train_loader.dataset)
    ramp_epochs, down_epoch, max_beta = 500, 1000, delta

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        rel_epoch = (epoch - 1) % CYCLE_LENGTH
        if rel_epoch < ramp_epochs:
            current_kl = max(0.00001, (rel_epoch / ramp_epochs) * max_beta)
        elif rel_epoch < down_epoch:
            current_kl = max_beta
        else:
            current_kl = 0.00001

        for x_g_b, x_l_b, y_b in train_loader:
            svi.step(x_g_b.to(device), x_l_b.to(device), y_b.to(device),
                     total_size=total_size, kl_weight=current_kl)

        scheduler.step()

        # Evaluate and save at step 1000 intervals
        if epoch % 1000 == 0 or epoch == epochs:
            rmse, mae, mape, avg_nll, coverage, avg_crps = evaluate_metrics(model, val_loader, scaler_y, device, n_samples=50)
            print(f"[{run_id}] Epoch {epoch:04d} -> RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.2f}%")

            # Save metrics row
            record = {"epoch": epoch, "rmse": rmse, "mae": mae, "mape": mape, "avg_nll": avg_nll, "coverage": coverage, "avg_crps": avg_crps}
            metrics_records.append(record)
            pd.DataFrame(metrics_records).to_csv(metrics_log_path, index=False)

            # Save models and scalers into the dedicated object directory
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pt"))
            joblib.dump(scaler_y, os.path.join(output_dir, f"scaler_epoch_{epoch}.pkl"))

    print(f"[{run_id}] Training completed successfully.")

# ==========================================
# 8. TRAINING & MONITORING
# ==========================================
if __name__ == "__main__":
    from pyro.optim import PyroLRScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths: Compatible with CSV, XLSX, or XLS
    train_file_path = "samples/train_trips.csv"
    val_file_path   = "samples/validation_trips.csv"

    # Process training data & fit y scaler
    x_g_train, x_l_train, y_train, scaler_y, _ = process_raw_data(train_file_path)

    # Process validation data using fitted training scaler
    has_val_file = os.path.exists(val_file_path)
    if has_val_file:
        x_g_val, x_l_val, y_val, _, _ = process_raw_data(val_file_path, scaler_y=scaler_y)
        val_dataset = TensorDataset(x_g_val, x_l_val, y_val)
        val_loader  = DataLoader(val_dataset, batch_size=256, shuffle=False)
    else:
        print(f"Warning: Validation file '{val_file_path}' not found! Split will be made from training set.")
        idx = np.arange(x_g_train.shape[0])
        train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
        x_g_val, x_l_val, y_val = x_g_train[val_idx], x_l_train[val_idx], y_train[val_idx]
        x_g_train, x_l_train, y_train = x_g_train[train_idx], x_l_train[train_idx], y_train[train_idx]
        val_dataset = TensorDataset(x_g_val, x_l_val, y_val)
        val_loader  = DataLoader(val_dataset, batch_size=256, shuffle=False)

    pyro.clear_param_store()
    bnn_model = MatrixGNN_VAE(
        num_sections=num_segment,
        global_dim=9,
        local_dim=4,
        hidden_dim=64,
        latent_dim=LATENT_DIM,
        device=device,
    ).to(device)

    CYCLE_LENGTH = 1000
    optimizer_args = {
        "optimizer": torch.optim.AdamW,
        "optim_args": {"lr": 0.0005, "weight_decay": 0.0001},
    }

    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=CYCLE_LENGTH, T_mult=1, eta_min=0.0001
        )

    scheduler = PyroLRScheduler(scheduler_constructor, optimizer_args)
    svi = SVI(model_fn, guide_fn, scheduler, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    epochs        = 4000
    batch_size    = 734
    train_dataset = TensorDataset(x_g_train, x_l_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_size    = len(train_dataset)

    ramp_epochs = 500
    down_epoch  = 1000
    max_beta    = 0.6

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        relative_epoch = (epoch - 1) % CYCLE_LENGTH

        if relative_epoch < ramp_epochs:
            current_kl_weight = max(0.00001, (relative_epoch / ramp_epochs) * max_beta)
        elif relative_epoch < down_epoch:
            current_kl_weight = max_beta
        else:
            current_kl_weight = 0.00001

        for x_g_batch, x_l_batch, y_batch in train_loader:
            x_g_batch = x_g_batch.to(device)
            x_l_batch = x_l_batch.to(device)
            y_batch   = y_batch.to(device)
            loss = svi.step(x_g_batch, x_l_batch, y_batch,
                            total_size=total_size, kl_weight=current_kl_weight)
            epoch_loss += loss

        scheduler.step()

        # Regular training step log
        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                ll, kl = get_ll_kl(x_g_batch, x_l_batch, y_batch)
            current_lr = list(scheduler.optim_objs.values())[0].optimizer.param_groups[0]["lr"] if scheduler.optim_objs else 0.001
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch:05d} | LR: {current_lr:.6f} | KL Wt: {current_kl_weight:.3f} | ELBO Loss: {avg_loss:.2f} | LL: {ll:.2f} | KL: {kl:.2f}")

        # Run Validation & Save Checkpoint every 1000 epochs (and final epoch)
        if epoch % 1000 == 0 or epoch == epochs:
            print(f"\n{'='*25} VALIDATION @ EPOCH {epoch} {'='*25}")
            rmse, mae, mape, avg_nll, coverage, avg_crps = evaluate_metrics(bnn_model, val_loader, scaler_y, device, n_samples=50)
            print(f"Validation RMSE : {rmse:.3f}s")
            print(f"Validation MAE  : {mae:.3f}s")
            print(f"Validation MAPE : {mape:.2f}%")
            print(f"Validation NLL  : {avg_nll:.3f}")
            print(f"Validation Coverage : {coverage:.2f}%")
            print(f"Validation CRPS : {avg_crps:.3f}")
            print(f"{'='*68}\n")

            # Save model checkpoint and scaler every 1000 epochs
            model_save_path = f"vae_model_epoch_{epoch}.pt"
            scaler_save_path = f"y_scaler_epoch_{epoch}.pkl"
            torch.save(bnn_model.state_dict(), model_save_path)
            joblib.dump(scaler_y, scaler_save_path)
            print(f"Checkpoint saved: {model_save_path} and {scaler_save_path}\n")

    print("\nTraining complete.")