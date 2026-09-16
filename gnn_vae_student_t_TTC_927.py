import math
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import PyroLRScheduler

num_segment = 9
LATENT_DIM = 64   # z ∈ ℝ⁶⁴

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
    raw_local = raw_data_np[:, 9 + num_segment + 1 : 9 + num_segment + 1 + (num_segment * 4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)

    # y_raw: 真實未縮放的總行程秒數 (1D numpy array)
    y_raw = raw_data_np[:, 9 : 9 + num_segment].sum(axis=1)
    
    y_raw_2d = y_raw.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_scaled_2d = scaler_y.fit_transform(y_raw_2d)
    y_scaled = torch.tensor(y_scaled_2d, dtype=torch.float32).squeeze(-1)

    print("Total rows loaded:", raw_data_np.shape[0])
    print("y_raw mean:", y_raw.mean())
    print("y_raw std:", y_raw.std())

    # 🔴 重點：回傳 5 個值，包含 y_raw
    return x_global, x_local, y_scaled, y_raw, scaler_y


# ==========================================
# 2. ENCODER φ
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
    def __init__(self, input_dim, output_dim, num_segments,
                 dropout_rate=0.2, device='cuda'):
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
            right_feat = prev[i+1] * wr if i < self.num_segments - 1 \
                         else torch.zeros_like(self_feat)

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
        scores  = self.gate(h_stack).squeeze(-1)       # [B, S]
        weights = torch.softmax(scores, dim=1)         # [B, S]
        weights = self.dropout(weights)

        v = self.v_proj(h_stack)                       # [B, S, out_dim]
        pooled = (weights.unsqueeze(-1) * v).sum(dim=1)  # [B, out_dim]

        if return_weights:
            return pooled, weights
        return pooled


class EncoderPhi(nn.Module):
    def __init__(self, global_dim, local_dim, hidden_dim,
                 num_segments, latent_dim, device='cuda'):
        super().__init__()
        self.head_dropout = nn.Dropout(p=0.1)
        self.num_segments = num_segments
        self.latent_dim   = latent_dim
        input_dim = global_dim + local_dim

        self.embedding_layer = LocalIsolationLayer(
            input_dim, hidden_dim, num_segments, device)
        self.prop_layers = nn.ModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_segments,
                                dropout_rate=0.2, device=device)
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
        inputs_list = []
        for i in range(self.num_segments):
            loc_i = all_sections_data[:, i, :]
            inputs_list.append(torch.cat([global_features, loc_i], dim=1))

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
# 3. BOTTLENECK (Gaussian Reparameterization)
# ==========================================
def gaussian_reparameterise(mu, logvar, force_sample=False):
    sigma = torch.exp(0.5 * logvar)
    if not torch.is_grad_enabled() and not force_sample:
        return mu, sigma

    eps = torch.randn_like(mu)
    z = mu + eps * sigma
    return z, sigma


def kl_gaussian_to_standard_normal(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1)


# ==========================================
# 4. DECODER θ (三塔架構：加入 Tower C 預測自由度 df)
# ==========================================
class DecoderTheta(nn.Module):
    def __init__(self, latent_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.latent_dim   = latent_dim

        self.drop = nn.Dropout(0.1)

        # ── Shared trunk
        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            self.drop
        )

        # ── TOWER A: 預測 ETA 平均值 (loc) ──
        self.tower_a = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 16), nn.LeakyReLU(0.1),
            nn.Linear(16, 8),  nn.LeakyReLU(0.1),
        )

        # ── TOWER B: 預測不確定性 / 尺度 (scale) ──
        self.tower_b = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(0.1), self.drop,
            nn.Linear(32, 16), nn.LeakyReLU(0.1),
            nn.Linear(16, 8),  nn.LeakyReLU(0.1),
        )

        # ── TOWER C: 預測 Student-t 的自由度 (df) ──
        self.tower_c = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(0.1), self.drop,
            nn.Linear(32, 16), nn.LeakyReLU(0.1),
            nn.Linear(16, 8),  nn.LeakyReLU(0.1),
        )

        self.heads_a = nn.Linear(8, 1)
        self.heads_b = nn.Linear(8, 1)
        self.heads_c = nn.Linear(8, 1)

        # 權重初始化
        nn.init.xavier_uniform_(self.heads_a.weight, gain=1.0)
        nn.init.xavier_uniform_(self.heads_b.weight, gain=1.5)
        nn.init.constant_(self.heads_b.bias, 1.0)

        # 自由度初始值：bias=2.0 經過 softplus 大約等於 2.13，總 df 初值約 4.13 (合理厚尾起步)
        nn.init.xavier_uniform_(self.heads_c.weight, gain=1.0)
        nn.init.constant_(self.heads_c.bias, 2.0)

    def forward(self, z):
        initial_h = self.dec1(z)

        # Tower A
        h_a = self.tower_a(initial_h)
        a_e = self.heads_a(h_a)

        # Tower B
        h_b = self.tower_b(initial_h)
        b_e = self.heads_b(h_b)
        b_t = torch.exp(0.5 * torch.clamp(b_e, min=-10.0, max=10.0)) + 1e-3

        # Tower C (保證 nu > 2 且加上 50.0 上限防止梯度消失)
        h_c = self.tower_c(initial_h)
        c_e = self.heads_c(h_c)
        df_t = torch.clamp(torch.nn.functional.softplus(c_e) + 2.0, min=2.01, max=15.0)

        return (a_e, b_t, df_t)


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

    def forward(self, global_features, all_sections_data, return_attn=False, force_sample=False):
        if return_attn:
            mu, logvar, attn_weights = self.encoder(global_features, all_sections_data, return_attn=True)
        else:
            mu, logvar = self.encoder(global_features, all_sections_data)

        z, sigma = gaussian_reparameterise(mu, logvar, force_sample=force_sample)
        kl = kl_gaussian_to_standard_normal(mu, logvar)

        # 解碼輸出 (a_e, b_t, df_t)
        locs, scalers, dfs = self.decoder(z)

        if return_attn:
            return locs, scalers, dfs, kl, attn_weights
        return locs, scalers, dfs, kl


# ==========================================
# 6. PYRO MODEL & GUIDE (Student-t 觀測似然)
# ==========================================
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    pyro.module("decoder", bnn_model.decoder)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0]):
        # 1. 潛在空間高斯先驗 p(z) ~ N(0, I)
        z_prior_mu = x_global.new_zeros(x_global.shape[0], bnn_model.latent_dim)
        z_prior_sigma = x_global.new_ones(x_global.shape[0], bnn_model.latent_dim)

        with pyro.poutine.scale(scale=kl_weight):
            z = pyro.sample("latent_z", dist.Normal(z_prior_mu, z_prior_sigma).to_event(1))

        # 2. 解碼得到 loc, scale, df
        a_e, b_t, df_t = bnn_model.decoder(z)

        # 3. 觀測層改為 Student-t 似然
        target = y_true[:] if y_true is not None else None
        pyro.sample(
            "obs",
            dist.StudentT(df=df_t.squeeze(-1), loc=a_e.squeeze(-1), scale=b_t.squeeze(-1)),
            obs=target
        )


def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    pyro.module("encoder", bnn_model.encoder)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0]):
        # 變分後驗 q(z|x) ~ N(mu, sigma^2)
        mu, logvar = bnn_model.encoder(x_global, x_local)
        sigma = torch.exp(0.5 * logvar)

        with pyro.poutine.scale(scale=kl_weight):
            pyro.sample("latent_z", dist.Normal(mu, sigma).to_event(1))


# ==========================================
# 7. LL / KL DIAGNOSTIC
# ==========================================
def get_ll_kl(model_fn, guide_fn, x_g, x_l, y, total_size):
    with torch.no_grad():
        locs, scales, dfs, kl = bnn_model(x_g, x_l)
        loc   = locs.squeeze(-1)
        scale = scales.squeeze(-1)
        df    = dfs.squeeze(-1)

        # 改用 Student-t 計算 Log-Likelihood
        d  = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
        ll = d.log_prob(y[:]).sum().item()
        kl = kl.mean().item()
    return ll, kl


# ==========================================
# 8. TRAINING & 9. INFERENCE
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    assert str(device) == "cuda", "CUDA not available — check your environment"

    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_Jan_Jun.xlsx"
    x_global_all, x_local_all, y_all, y_raw_all, scaler_y = process_raw_data(file_path)

    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.0000001, random_state=42)

    x_global_train = x_global_all[train_idx]
    x_local_train  = x_local_all[train_idx]
    y_train        = y_all[train_idx]
    y_raw_train    = y_raw_all[train_idx]    # 原始真實秒數

    x_global_val = x_global_all[val_idx].to(device)
    x_local_val  = x_local_all[val_idx].to(device)
    y_val        = y_all[val_idx].to(device)
    y_raw_val    = y_raw_all[val_idx]        # 驗證集的原始真實秒數

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
        "optim_args": {"lr": 0.001, "weight_decay": 0.0001},
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

    ramp_epochs = 500
    down_epoch  = 1000
    max_beta    = 0.6

    for epoch in range(epochs):
        epoch_loss = 0.0

        relative_epoch = epoch % CYCLE_LENGTH
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

        if epoch % 1 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                ll, kl = get_ll_kl(model_fn, guide_fn, x_g_batch, x_l_batch, y_batch, total_size=total_size)
            current_lr = list(scheduler.optim_objs.values())[0].optimizer.param_groups[0]["lr"] if scheduler.optim_objs else 0.001
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch:05d} | LR: {current_lr:.6f} | KL Wt: {current_kl_weight:.3f} | ELBO Loss: {avg_loss:.2f} | LL: {ll:.2f} | KL: {kl:.2f}")

    torch.save(bnn_model.state_dict(), "2025_Jan_Jun_vae_student-t_TTC_927.pt")
    joblib.dump(scaler_y, "y_scaler_vae_student-t_2025_Jan_Jun_TTC_927.pkl")
    print("\nModel weights and scaler saved successfully.")

    # ==========================================
    # 9. INFERENCE (Monte Carlo Sampling from Student-t)
    # ==========================================
    bnn_model.eval()

    def predict_mc_aggregate(x_global, x_local, n_samples=200):
        all_samples = []
        with torch.no_grad():
            mu, logvar = bnn_model.encoder(x_global, x_local)
            sigma = torch.exp(0.5 * logvar)

            for _ in range(n_samples):
                eps = torch.randn_like(mu)
                z = mu + eps * sigma

                # 正確呼叫解碼器取得 a_e, b_t, df_t
                a_e, b_t, df_t = bnn_model.decoder(z)

                obs_dist = torch.distributions.StudentT(
                    df=df_t.squeeze(-1),
                    loc=a_e.squeeze(-1),
                    scale=b_t.squeeze(-1),
                )
                all_samples.append(obs_dist.sample())

        return torch.stack(all_samples, dim=0)  # [MC_SAMPLES, BATCH_SIZE]

    list_of_predict = []
    list_of_actual  = []
    within_bound_count = 0
    error_abs_total = error_total = 0

    print("\n--- Starting Final Evaluation ---")
    for j in range(len(x_global_val)):
        val_x_g = x_global_val[j:j+1]
        val_x_l = x_local_val[j:j+1]

        # MC Sampling
        samples = predict_mc_aggregate(val_x_g, val_x_l, n_samples=200)

        pred_mean_scaled = samples.mean().item()
        pred_std_scaled  = samples.std().item()

        # 反正規化
        pred_real = scaler_y.inverse_transform([[pred_mean_scaled]])[0][0]
        actual_real = y_raw_val[j]
        std_real = pred_std_scaled * scaler_y.scale_[0]

        within = (pred_real - std_real) <= actual_real <= (pred_real + std_real)
        if within:
            within_bound_count += 1

        error_total += (actual_real - pred_real)
        error_abs_total += abs(actual_real - pred_real)

        list_of_predict.append(pred_real)
        list_of_actual.append(actual_real)

        print(f"\n--- Sample {j} ---")
        print(f"Total ETA: {pred_real:.2f}s  (Actual: {actual_real:.2f}s)")
        print(f"Confidence: ±{std_real:.2f}s | Within CI? {'YES' if within else 'NO'}")
        print(f"MAE so far: {error_abs_total/(j+1):.2f}s | Bias: {error_total/(j+1):.2f}s")
        print(f"Prediction Std Deviation: {np.std(list_of_predict):.2f} | Actual Std Deviation: {np.std(list_of_actual):.2f}")
        print(f"總共 {j+1} 筆，{within_bound_count} 筆落在區間內")

    if len(x_global_val) > 0:
        print("\n" + "="*60)
        print(f"SUMMARY  ({len(x_global_val)} trips, 200 MC samples)")
        print("="*60)
        print(f"  Trip-level within CI : {within_bound_count}/{len(x_global_val)} ({(within_bound_count/len(x_global_val))*100:.1f}%)")
        print(f"  Final MAE            : {error_abs_total/len(x_global_val):.2f}s")
        print(f"  Final Bias           : {error_total/len(x_global_val):.2f}s")