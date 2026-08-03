import statistics
import math

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
LATENT_DIM = 64   # z ∈ ℝ³²

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
    raw_local = raw_data_np[:, 9+num_segment+1:9+num_segment+1+(num_segment*4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)

    y_raw = raw_data_np[:, 9 : 9+num_segment].sum(axis=1)
    print(y_raw)
    y_raw_2d = y_raw.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_scaled_2d = scaler_y.fit_transform(y_raw_2d)
    y_scaled = torch.tensor(y_scaled_2d, dtype=torch.float32).squeeze(-1)

    print("Total rows loaded:", raw_data_np.shape[0])
    print("y_raw mean:", y_raw.mean())
    print("y_raw std:", y_raw.std())
    return x_global, x_local, y_scaled, scaler_y


# ==========================================
# 2. ENCODER φ  (~97% of all parameters)
#    LocalIsolation embed + 2× NeighborMixing
#    → three-way split: μ head, log σ² head, log ν head
# ==========================================

class LocalIsolationLayer(nn.Module):
    """
    Unique unshared linear projection per segment.
    Input:  list of [batch, input_dim]  (length = num_segments)
    Output: list of [batch, output_dim] after SiLU
    """
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
    """
    Causal left-to-right message passing per segment pair.
    self-feature weighted by w_self, right-neighbour by w_right.
    Two sub-layers with SiLU + Dropout.
    """
    def __init__(self, input_dim, output_dim, num_segments,
                 dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        net_in = input_dim * 2

        # Deterministic scalar mixing weights — one per segment
        # initialised near (2, 0) to match old prior means
        self.w_self  = nn.Parameter(torch.full((num_segments,),  1.0))
        self.w_right = nn.Parameter(torch.full((num_segments,),  0.0))

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


#class AttentionPooling(nn.Module):
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else hidden_dim

        # Q·K scoring — always in hidden_dim space
        self.gate   = nn.Linear(hidden_dim, 1, bias=True)

        # V projection — hidden_dim → out_dim
        # if out_dim == hidden_dim, this is still a useful learned transform
        self.v_proj = nn.Linear(hidden_dim, self.out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def forward(self, h_stack: torch.Tensor, return_weights: bool = False):
        # h_stack: [B, S, hidden_dim]

        # ── Scoring in hidden_dim space ──────────────────────────
        scores  = self.gate(h_stack).squeeze(-1)       # [B, S]
        weights = torch.softmax(scores, dim=1)         # [B, S]
        weights = self.dropout(weights)

        # ── Project each segment to out_dim (V step) ─────────────
        v = self.v_proj(h_stack)                       # [B, S, out_dim]

        # ── Weighted sum collapses S ──────────────────────────────
        pooled = (weights.unsqueeze(-1) * v).sum(dim=1)  # [B, out_dim]

        if return_weights:
            return pooled, weights
        return pooled



class EncoderPhi(nn.Module):
    """
    ENCODER φ
    ---------
    Runs LocalIsolation + 2× NeighborMixing, then pools across
    segments, and projects to three heads:
        μ      ∈ ℝ^{LATENT_DIM}   — latent mean
        log σ² ∈ ℝ^{LATENT_DIM}   — latent log-variance  (z-cloud spread)
        log ν  ∈ ℝ^{LATENT_DIM}   — latent log-df         (tail heaviness)
    """
    def __init__(self, global_dim, local_dim, hidden_dim,
                 num_segments, latent_dim, device='cuda'):
        super().__init__()
        self.head_dropout = nn.Dropout(p=0.1)
        self.num_segments = num_segments
        self.latent_dim   = latent_dim
        input_dim = global_dim + local_dim

        # ── Embedding + propagation (shared across segments)
        self.embedding_layer = LocalIsolationLayer(
            input_dim, hidden_dim, num_segments, device)
        self.prop_layers = nn.ModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_segments,
                                dropout_rate=0.2, device=device)
            for _ in range(2)
        ])
        
        self.attn_pool = AttentionPooling(hidden_dim, hidden_dim, dropout=0.1)

        # ── Three-way split heads  (deterministic, small)
        # Pool dim: hidden_dim * num_segments after mean-pooling → hidden_dim
        pool_dim = hidden_dim
        self.head_mu     = nn.Linear(pool_dim, latent_dim)
        self.head_logvar = nn.Linear(pool_dim, latent_dim)   # log σ²
        
        # ADD THESE THREE LINES immediately after:
        nn.init.constant_(self.head_logvar.bias, -4.0)   #0.0
        # ADD — weight initialisations for more aggressive output variation
        nn.init.xavier_uniform_(self.head_mu.weight,     gain=1.0)  # ← 2.0
        nn.init.xavier_uniform_(self.head_logvar.weight, gain=1.0)  # ← 2.0
        nn.init.zeros_(self.head_mu.bias)   

    def forward(self, global_features, all_sections_data, return_attn: bool = False):
        # Build per-segment inputs
        inputs_list = []
        for i in range(self.num_segments):
            loc_i = all_sections_data[:, i, :]
            inputs_list.append(torch.cat([global_features, loc_i], dim=1))

        # GCN pass
        h = self.embedding_layer(inputs_list)
        for layer in self.prop_layers:
            h = layer(h)

        # Mean-pool across segments  → [batch, hidden_dim]
        h_stack = torch.stack(h, dim=1)          # [B, S, hidden_dim]
        
        """
        h_pool  = h_stack.mean(dim=1)             # [B, hidden_dim]
        """
        
        if return_attn:
            h_pool, attn_weights = self.attn_pool(h_stack, return_weights=True)
        else:
            h_pool = self.attn_pool(h_stack)          # [B, hidden_dim]
        
        
        
        # Three-way split
        h_pool = self.head_dropout(h_pool)
        mu     = self.head_mu(h_pool)                          # z-cloud centre
        logvar = self.head_logvar(h_pool)                      # z-cloud spread

        if return_attn:
            return mu, logvar, attn_weights    # [B, S]
        
        return mu, logvar


# ==========================================
# 3. BOTTLENECK
#    q(z|x) = T(μ, diag(σ²), ν)  — Student-t posterior
#    Reparameterisation: z = μ + σ ⊙ (g / √(χ²_ν / ν))
#       g ~ N(0,I),  χ²_ν ~ chi-squared(ν)
#    Latent z ∈ ℝ³²
# ==========================================
"""
def student_t_reparameterise(mu, logvar, logdf, force_sample=False):
    sigma = torch.exp(0.5 * logvar)                       
    nu    = torch.nn.functional.softplus(logdf) + 3.0     

    # If we are not forcing it, and grads are off, use mean
    if not torch.is_grad_enabled() and not force_sample:                        
        return mu, sigma, nu

    g = torch.randn_like(mu)                              
    chi2 = torch.distributions.Chi2(df=nu.detach()).rsample()   
    z    = mu + sigma * g / torch.sqrt(chi2 / nu + 1e-8)
    return z, sigma, nu
"""
"""
def kl_student_t_to_standard_normal(mu, logvar, logdf, prior_var=1.0):   #2.0
    
    Approximate KL(q_StudentT(μ,σ²,ν) ‖ N(0,I)) via the
    Gaussian lower bound (tight when ν→∞, conservative otherwise).
    This is the pyro.factor term added to the ELBO.
    
    sigma2 = torch.exp(logvar)
    nu     = torch.nn.functional.softplus(logdf) + 2.0
    log_prior = math.log(prior_var)
    # KL[N(μ,σ²) ‖ N(0,1)] as base term
    kl = -0.5 * (1 + logvar - log_prior
                 - mu.pow(2) / prior_var
                 - sigma2 / prior_var)
    # Correction for heavier tails: log(ν/(ν-2)) per dim (≥ 0)
    tail_correction = 0.5 * torch.log(nu / (nu - 2.0 + 1e-8))
    return (kl + tail_correction).mean(dim=-1).mean()
"""

def gaussian_reparameterise(mu, logvar, force_sample=False):
    sigma = torch.exp(0.5 * logvar)                       
    if not torch.is_grad_enabled() and not force_sample:                        
        return mu, sigma

    eps = torch.randn_like(mu)                              
    z = mu + eps * sigma
    return z, sigma

def kl_gaussian_to_standard_normal(mu, logvar):
    """
    Standard KL( N(mu, sigma^2) || N(0, 1) )
    Returns shape [B] so Pyro can scale it in the plate.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1) # Sum over latent dimensions, keep batch

# ==========================================
# 4. DECODER θ
#    Dec-layer 1: Linear(32 → 64) + LayerNorm + LeakyReLU
#    Dec-layer 2: Linear(64 → 16) + LayerNorm + LeakyReLU   (Jₑ computed here)
#    ↓ split
#    a head (16→1): identity → committed ETA
#    b head (16→1): softplus → √Vₐ  (aleatoric component)
#    ──────────────────────────────────────
#    Epistemic propagation via Jacobian:
#      b²ₜ = Vₐ + Jₑ(μ)ᵀ · diag(σ²) · Jₑ(μ)
#    Output: aₑ ± bₜ
# ==========================================
#class DecoderTheta(nn.Module):
class DecoderTheta(nn.Module):
    """
    Shared decoder that maps latent z → (a_e, b_t) per segment.
    The Jacobian-based epistemic uncertainty propagation is computed
    analytically: b²ₜ = Vₐ + Jₑ(μ)ᵀ diag(σ²) Jₑ(μ)
    """
    def __init__(self, latent_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.latent_dim   = latent_dim

        # ── Shared trunk
        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),
        )
        """
        self.dec2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
        )   # Jₑ = ∂f/∂z is computed through these two layers
        self.dec3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
        )   # Jₑ = ∂f/∂z is computed through these two layers
        self.dec4_head_1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
        )   # Jₑ = ∂f/∂z is computed through these two layers
        self.dec4_head_2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
        )   # Jₑ = ∂f/∂z is computed through these two layers
        """
        self.tower_a = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
        )

        # ── TOWER B: Dedicated strictly to Uncertainty (b_t) ──
        self.tower_b = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
        )
        # ── Per-segment output heads
        # a head: identity activation → committed ETA
        """self.heads_a = nn.ModuleList([nn.Linear(4, 1)
                                      for _ in range(num_segments)])
                                      """
        
        self.heads_a = nn.Linear(8, 1)
        # b head: softplus → √Vₐ (aleatoric)
        self.heads_b = nn.Linear(8, 1)
        #self.heads_c = nn.Linear(8, 1)
        
        
        nn.init.xavier_uniform_(self.heads_a.weight, gain=1.0)
        #nn.init.xavier_uniform_(self.heads_a.weight, gain=0.5)
        
        nn.init.xavier_uniform_(self.heads_b.weight, gain=1.5)
        
        #nn.init.xavier_uniform_(self.heads_b.weight, gain=3.0)
        # 1.0 escapes the Student-T Trap, telling the network the data is valid signal, not outliers!
        
        nn.init.constant_(self.heads_b.bias, 1.0)
        
        #nn.init.constant_(self.heads_b.bias, 0.0)


    
    def forward_with_epistemic(self, z, sigma, mu):
        """
        One-pass forward. head_b directly predicts total uncertainty.
        Pyro's weight posterior via Predictive(num_samples=50) handles
        epistemic uncertainty at inference time.
        """
        initial_h = self.dec1(z)
        h_a = self.tower_a(initial_h)
        a_e = self.heads_a(h_a)
        h_b = self.tower_b(initial_h)
        b_e = self.heads_b(h_b)
        b_t = torch.exp(0.5 * torch.clamp(b_e, min=-10.0, max=10.0)) + 1e-3
        """
        h_a = self.dec4_head_1(h)       # [B, 8]
        h_b = self.dec4_head_2(h)       # [B, 8]
        #h_c = self.dec4_head_3(h)       # [B, 8]
        outputs = []
        a_e = self.heads_a(h_a)                                    # loc
        b_t = torch.nn.functional.softplus(self.heads_b(h_b)) + 1e-3  # scale
        #df_t = torch.nn.functional.softplus(self.heads_c(h_c)) + 7.0  # scale
        
        """
        outputs = (a_e, b_t)
        return outputs


# ==========================================
# 5. FULL MODEL:  MatrixGNN_VAE
# ==========================================

class MatrixGNN_VAE(nn.Module):
    """
    Full VAE-style bus ETA model.

    Forward returns  (all_locs, all_scales, all_dfs)
    compatible with the original model_fn / SVI training loop.

    Here:
        loc   = aₑ   (committed ETA, from decoder a-head)
        scale = bₜ   (total uncertainty half-width, aleatoric + epistemic)
        df    = ν    (Student-t tail heaviness from bottleneck)
    """
    def __init__(self, num_sections=9, global_dim=9, local_dim=4,
                 hidden_dim=13, latent_dim=LATENT_DIM, device='cuda'):
        super().__init__()
        self.num_sections = num_sections
        self.latent_dim   = latent_dim
        self.device       = device

        # ── Encoder φ
        self.encoder = EncoderPhi(
            global_dim=global_dim,
            local_dim=local_dim,
            hidden_dim=hidden_dim,
            num_segments=num_sections,
            latent_dim=latent_dim,
            device=device,
        )

        # ── Decoder θ
        self.decoder = DecoderTheta(
            latent_dim=latent_dim,
            num_segments=num_sections,
            device=device,
        )

    def forward(self, global_features, all_sections_data, return_attn=False, force_sample=False, kl_weight=1.0):
        # ── ENCODER → μ, log σ², log ν
        if return_attn:
            mu, logvar, attn_weights = self.encoder(global_features, all_sections_data, return_attn=True)
        else:
            mu, logvar = self.encoder(global_features, all_sections_data)

        # ── BOTTLENECK: Student-t reparameterisation
        z, sigma = gaussian_reparameterise(mu, logvar, force_sample=force_sample)

        # ── KL term injected into Pyro's ELBO
        kl = kl_gaussian_to_standard_normal(mu, logvar)

        # ── DECODER → per-segment (aₑ, bₜ)
        #seg_outputs = self.decoder.forward_with_epistemic(z, sigma, mu)
        outputs = self.decoder.forward_with_epistemic(z, sigma, mu)


        locs, scalers = outputs
        
        if return_attn:
            return locs, scalers, kl, attn_weights   # 5-tuple for diagnostics
        return locs, scalers, kl                     # 4-tuple for training


# ==========================================
# 6.  MODEL_FN  (Pyro probabilistic model)
# ==========================================
"""
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    # 1. TELL PYRO TO TRACK AND UPDATE THE WEIGHTS
    pyro.module("bnn_model", bnn_model)
    with pyro.poutine.scale(scale=kl_weight):
        loc, scale, df = bnn_model(x_global, x_local)
        
        # Squeeze the outputs from [batch_size, 1] to [batch_size]
        loc = loc.squeeze(-1)
        scale = scale.squeeze(-1)
        df = df.squeeze(-1)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size,
                    subsample_size=x_global.shape[0], dim=-1):
        dist_i = dist.StudentT(df, loc, scale)
        target = y_true[:] if y_true is not None else None
        pyro.sample(f"obs_full_trip", dist_i, obs=target)
"""
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    pyro.module("bnn_model", bnn_model)
    locs, scales, kl = bnn_model(x_global, x_local)   # no kl_weight arg needed
    
    loc   = locs.squeeze(-1)
    scale = scales.squeeze(-1)
    #df    = dfs.squeeze(-1)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0], dim=-1):
        pyro.factor("bottleneck_kl", -kl * kl_weight)
        dist_i = dist.Normal(loc, scale)
        target = y_true[:] if y_true is not None else None
        pyro.sample("obs_full_trip", dist_i, obs=target)


# ==========================================
# 7.  LL / KL DIAGNOSTIC
# ==========================================

def get_ll_kl(model_fn, guide_fn, x_g, x_l, y, total_size):
    with torch.no_grad():
        locs, scales, kl = bnn_model(x_g, x_l)   # 4-tuple now
        loc   = locs.squeeze(-1)
        scale = scales.squeeze(-1)
        #df    = dfs.squeeze(-1)

        d  = torch.distributions.Normal(loc=loc, scale=scale)
        ll = d.log_prob(y[:]).sum().item()
        kl = kl.mean().item()
        # kl already returned from forward(), no need to re-run encoder
    return ll, kl


# ==========================================
# 8.  TRAINING  (identical SVI loop)
# ==========================================

# ==========================================
# 8.  TRAINING & 9. AGGREGATE INFERENCE
# ==========================================
if __name__ == "__main__":
    from pyro.optim import PyroLRScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_new.xlsx"
    x_global_all, x_local_all, y_all, scaler_y = process_raw_data(file_path)

    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.0000001, random_state=42)

    x_global_train = x_global_all[train_idx]
    x_local_train  = x_local_all[train_idx]
    y_train        = y_all[train_idx]

    x_global_val = x_global_all[val_idx].to(device)
    x_local_val  = x_local_all[val_idx].to(device)
    y_val        = y_all[val_idx].to(device)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    assert str(device) == "cuda", "CUDA not available — check your environment"

    pyro.clear_param_store()
    bnn_model = MatrixGNN_VAE(
        num_sections=num_segment,
        global_dim=9,
        local_dim=4,
        hidden_dim=64,
        latent_dim=LATENT_DIM,
        device=device,
    ).to(device)

    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
        pass

    CYCLE_LENGTH = 1000
    optimizer_args = {
        "optimizer": torch.optim.AdamW,
        "optim_args": {"lr": 0.001, "weight_decay": 0.01},
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
    max_beta    = 0.8 # 0.8

    for epoch in range(epochs):
        epoch_loss = epoch_ll = epoch_kl = 0.0

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

        if (epoch % 1 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                ll, kl = get_ll_kl(model_fn, guide_fn, x_g_batch, x_l_batch, y_batch, total_size=total_size)
            current_lr = list(scheduler.optim_objs.values())[0].optimizer.param_groups[0]["lr"] if scheduler.optim_objs else 0.002
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch:05d} | LR: {current_lr:.6f} | KL Wt: {current_kl_weight:.3f} | ELBO Loss: {avg_loss:.2f} | LL: {ll:.2f} | KL: {kl:.2f}")

    #pyro.get_param_store().save("ghost_bus_vae_aggregate.pt")
    torch.save(bnn_model.state_dict(), "ghost_bus_vae_aggregate_0.8.pt")
    joblib.dump(scaler_y, "y_scaler_vae_aggregate.pkl")
    print("\nModel weights and scaler saved successfully.")

    # ==========================================
    # 9. CORRECTED INFERENCE (Aggregate)
    # ==========================================
    bnn_model.eval()

    def predict_mc_aggregate(x_global, x_local, n_samples=200):
        all_samples = []
        with torch.no_grad():
            mu, logvar = bnn_model.encoder(x_global, x_local)
            sigma = torch.exp(0.5 * logvar)                       

            for _ in range(n_samples):
                eps    = torch.randn_like(mu) 
                z    = mu + eps * sigma

                a_e, b_t = bnn_model.decoder.forward_with_epistemic(z, sigma, mu)

                obs_dist = torch.distributions.Normal(
                    loc=a_e.squeeze(-1),
                    scale=b_t.squeeze(-1),
                )
                all_samples.append(obs_dist.sample()) 
                
        bnn_model.eval() # Turn dropout back off        
        return torch.stack(all_samples, dim=0) # [MC_SAMPLES, BATCH_SIZE]

    list_of_predict = []
    list_of_actual  = []
    within_bound_count = 0
    error_abs_total = error_total = 0

    print("\n--- Starting Final Evaluation ---")
    for j in range(len(x_global_val)):
        val_x_g = x_global_val[j:j+1]
        val_x_l = x_local_val[j:j+1]
        
        # MC Sampling
        samples = predict_mc_aggregate(val_x_g, val_x_l, n_samples=200) # shape: [200, 1]
        
        pred_mean_scaled = samples.mean().item()
        pred_std_scaled  = samples.std().item()
        actual_scaled    = y_val[j].item()

        # Inverse Transform
        pred_real = scaler_y.inverse_transform([[pred_mean_scaled]])[0][0]
        actual_real = scaler_y.inverse_transform([[actual_scaled]])[0][0]
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

    print("\n" + "="*60)
    print(f"SUMMARY  ({len(x_global_val)} trips, 200 MC samples)")
    print("="*60)
    print(f"  Trip-level within CI : {within_bound_count}/{len(x_global_val)} ({(within_bound_count/len(x_global_val))*100:.1f}%)")
    print(f"  Final MAE            : {error_abs_total/len(x_global_val):.2f}s")
    print(f"  Final Bias           : {error_total/len(x_global_val):.2f}s")