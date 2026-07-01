"""
MatrixGNN — VAE-style Encoder → Bottleneck → Decoder
=====================================================
Architecture follows the diagram:

  Input x (global + local features)
       │
  ┌────▼──────────────────────┐
  │  ENCODER φ (~97% params)  │
  │  LocalIsolation  (embed)  │
  │  NeighborMixing  ×2       │
  │       ↓ three-way split   │
  │  μ head │ log σ² head │ log ν head  │
  └──────────────────────────-┘
       │        │          │
  ┌────▼──────────────────────┐
  │       BOTTLENECK          │
  │  q(z|x) = T(μ, diag(σ²), ν)  — Student-t posterior │
  │  Reparam: z = μ + σ ⊙ (g / √(χ²_ν / ν))           │
  │  Latent z ∈ ℝ³²                                     │
  └────────────────────────────┘
       │
  ┌────▼──────────────────────┐
  │      DECODER θ            │
  │  Dec-layer 1: Linear(32→64)  + LayerNorm + LeakyReLU │
  │  Dec-layer 2: Linear(64→16)  + LayerNorm + LeakyReLU │
  │       ↓ split                                        │
  │  a head (16→1): identity → committed ETA            │
  │  b head (16→1): softplus → √Vₐ aleatoric            │
  └────────────────────────────┘
       │
  b²ₜ = Vₐ + Jₑ(μ)ᵀ · diag(σ²) · Jₑ(μ)   (Jacobian propagation)
  Output: aₑ ± bₜ   (committed ETA ± uncertainty half-width)
"""

import statistics
import math

from requests import head
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
    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y_raw), dtype=torch.float32)

    print("Total rows loaded:", raw_data_np.shape[0])
    print("y_raw mean per segment:", y_raw.mean(axis=0))
    print("y_raw std  per segment:", y_raw.std(axis=0))
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
        self.w_self  = nn.Parameter(torch.full((num_segments,),  2.0))
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



class EncoderPhi(PyroModule):
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
        
        self.attn_pool = AttentionPooling(hidden_dim, 96, dropout=0.1)

        # ── Three-way split heads  (deterministic, small)
        # Pool dim: hidden_dim * num_segments after mean-pooling → hidden_dim
        pool_dim = 96 #hidden_dim
        self.head_mu     = nn.Linear(pool_dim, latent_dim)
        self.head_logvar = nn.Linear(pool_dim, latent_dim)   # log σ²
        self.head_logdf  = nn.Linear(pool_dim, latent_dim)   # log ν
        
        # ADD THESE THREE LINES immediately after:
        nn.init.constant_(self.head_logvar.bias, 2.0)   #0.5
        nn.init.constant_(self.head_logdf.bias, 1.0)  #-1.0
        
        # ADD — weight initialisations for more aggressive output variation
        nn.init.xavier_uniform_(self.head_mu.weight,     gain=2.0)  # ← fix 1
        nn.init.xavier_uniform_(self.head_logvar.weight, gain=2.0)  # ← fix 1
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
        
        
        self.head_dropout = nn.Dropout(p=0.1)
        # Three-way split
        h_pool = self.head_dropout(h_pool)
        mu     = self.head_mu(h_pool)                          # z-cloud centre
        logvar = self.head_logvar(h_pool)                      # z-cloud spread
        logdf  = self.head_logdf(h_pool)                       # tail heaviness

        if return_attn:
            return mu, logvar, logdf, attn_weights    # [B, S]
        
        return mu, logvar, logdf


# ==========================================
# 3. BOTTLENECK
#    q(z|x) = T(μ, diag(σ²), ν)  — Student-t posterior
#    Reparameterisation: z = μ + σ ⊙ (g / √(χ²_ν / ν))
#       g ~ N(0,I),  χ²_ν ~ chi-squared(ν)
#    Latent z ∈ ℝ³²
# ==========================================

def student_t_reparameterise(mu, logvar, logdf):
    """
    Student-t reparameterisation trick.
      σ  = exp(0.5 * logvar)
      ν  = softplus(logdf) + 2   (ν > 2 ensures finite variance)
      g  ~ N(0, I)
      u  ~ Chi²(ν)  ≡ Gamma(ν/2, 1/2)
      z  = μ + σ ⊙ g / √(u / ν)

    During inference (eval) we use the mean: z = μ.
    """
    sigma = torch.exp(0.5 * logvar)                       # [B, D]
    nu    = torch.nn.functional.softplus(logdf) + 2.0     # [B, D], ν > 2

    if not torch.is_grad_enabled():                        # inference: use mean
        return mu, sigma, nu

    # Reparameterise
    g = torch.randn_like(mu)                              # N(0,I)
    # χ²_ν = Gamma(ν/2, rate=0.5);  use torch.distributions for sampling
    chi2 = torch.distributions.Chi2(df=nu.detach()).rsample()   # [B, D]
    z    = mu + sigma * g / torch.sqrt(chi2 / nu + 1e-8)
    return z, sigma, nu


def kl_student_t_to_student_t_mc(z, mu, logvar, logdf, prior_nu=3.0, prior_scale=1.0):
    """
    Monte Carlo approximation of KL( q(z|x) || p(z) )
    where both q and p are Student-t distributions.
    """
    sigma = torch.exp(0.5 * logvar)
    nu = torch.nn.functional.softplus(logdf) + 2.0
    
    # q(z|x) - The predicted posterior distribution
    q_dist = torch.distributions.StudentT(df=nu, loc=mu, scale=sigma)
    
    # p(z) - The prior distribution (Standard Student-t)
    p_dist = torch.distributions.StudentT(
        df=torch.full_like(nu, prior_nu), 
        loc=torch.zeros_like(mu), 
        scale=torch.full_like(sigma, prior_scale)
    )
    
    # Evaluate log probabilities using the already sampled z
    log_q = q_dist.log_prob(z)  # [B, D]
    log_p = p_dist.log_prob(z)  # [B, D]
    
    # KL = Expectation of (log q(z) - log p(z))
    kl = (log_q - log_p).sum(dim=-1).mean()
    return kl


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

        # ── One complete individual trunk per segment
        self.trunks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 13),
                nn.LeakyReLU(0.1),
                nn.Linear(13, 8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, 4),
                nn.LeakyReLU(0.1)
            ) for _ in range(num_segments)
        ])

        # ── Per-segment output heads
        # a head: identity activation → committed ETA
        self.heads_a = nn.ModuleList([nn.Linear(4, 1)
                                      for _ in range(num_segments)])
        # b head: softplus → √Vₐ (aleatoric)
        self.heads_b = nn.ModuleList([nn.Linear(4, 1)
                                      for _ in range(num_segments)])
        for head in self.heads_b:
            nn.init.constant_(head.bias, -0.3)  # Pushes initial softplus output lower
        self.heads_d = nn.ModuleList([nn.Linear(4, 1)
                                      for _ in range(num_segments)])

    def _trunk(self, z, i):
        """Forward through the full trunk for segment i."""
        return self.trunks[i](z)
    """
    def forward_with_epistemic(self, z, sigma, mu):
        """
        
    """
        Full forward pass with Jacobian-based uncertainty propagation.

        Parameters
        ----------
        z     : [B, D]  sampled latent (or μ at inference)
        sigma : [B, D]  posterior σ from bottleneck
        mu    : [B, D]  posterior μ (used for Jacobian)

        Returns
        -------
        list of (a_e, b_t) per segment, each [B, 1]
        """
        
    """
        h = self._trunk(z)                    # [B, 16]
        outputs = []

        for i in range(self.num_segments):
            # ── Committed ETA  (a head)
            a_e = self.heads_a[i](h)          # [B, 1], raw (identity activation)

            # ── Aleatoric variance  Vₐ = softplus(b_raw)²
            b_raw = self.heads_b[i](h)        # [B, 1]
            sqrt_Va = torch.nn.functional.softplus(b_raw) + 1e-3   # √Vₐ
            Va = sqrt_Va ** 2                 # [B, 1]

            # ── Epistemic variance via Jacobian propagation
            #    Jₑ(μ) = ∂a_e/∂z  evaluated at z = μ
            #    b²_epistemic = Jₑᵀ diag(σ²) Jₑ
            #                 = ‖σ ⊙ Jₑ‖²   (element-wise)
            #
            #    We use torch.autograd.functional.jacobian for correctness.
            #    Shape of Jₑ: [B, 1, D]
            sigma2 = (sigma ** 2).detach()    # [B, D], stop-grad for stability

            # Compute ∂a_e/∂mu via a lightweight vjp trick (per-sample)
            # For efficiency we use a manual chain-rule through the trunk.
            with torch.enable_grad():
                mu_req = mu.detach().requires_grad_(True)
                h_mu   = self._trunk(mu_req)                 # [B, 16]
                a_mu   = self.heads_a[i](h_mu)               # [B, 1]
                # Jacobian rows: grad of each output scalar w.r.t. mu
                # vmap would be ideal; we use a loop over batch here.
                Je_batch = torch.zeros(mu.shape[0], mu.shape[1],
                                       device=mu.device)     # [B, D]
                for b in range(mu.shape[0]):
                    grads = torch.autograd.grad(
                        a_mu[b].sum(), mu_req,
                        retain_graph=True, create_graph=False)[0]
                    Je_batch[b] = grads[b]

            # b²_epistemic = Σ_d  σ²_d · J²_d   (element-wise quadratic form)
            b2_epistemic = (sigma2 * Je_batch ** 2).sum(dim=-1, keepdim=True)  # [B,1]

            # ── Total uncertainty
            b2_total = Va + b2_epistemic           # aleatoric + epistemic
            b_t      = torch.sqrt(b2_total + 1e-8) # [B, 1]  half-width

            outputs.append((a_e, b_t))

        return outputs  # list of (a_e, b_t), length = num_segments
    """
    def forward_with_epistemic(self, z, sigma, mu, nu):
        """
        One-pass forward. head_b directly predicts total uncertainty.
        Pyro's weight posterior via Predictive(num_samples=50) handles
        epistemic uncertainty at inference time.
        """
        outputs = []
        for i in range(self.num_segments):
            h = self._trunk(z, i)
            a_e = self.heads_a[i](h)                                    # loc
            b_t = torch.nn.functional.softplus(self.heads_b[i](h)) + 1e-3  # scale
            df_t = torch.nn.functional.softplus(self.heads_d[i](h)) + 2  # scale
            outputs.append((a_e, b_t, df_t))
        return outputs


# ==========================================
# 5. FULL MODEL:  MatrixGNN_VAE
# ==========================================

class MatrixGNN_VAE(PyroModule):
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
                 hidden_dim=12, latent_dim=LATENT_DIM, device='cuda'):
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

    def forward(self, global_features, all_sections_data, return_attn: bool = False):
        # ── ENCODER → μ, log σ², log ν
        if return_attn:
            mu, logvar, logdf, attn_weights = self.encoder(global_features, all_sections_data, return_attn=True)
        else:
            mu, logvar, logdf = self.encoder(global_features, all_sections_data)

        # ── BOTTLENECK: Student-t reparameterisation
        z, sigma, nu = student_t_reparameterise(mu, logvar, logdf)

        # ── KL term injected into Pyro's ELBO
        kl = kl_student_t_to_student_t_mc(z, mu, logvar, logdf)
        pyro.factor("bottleneck_kl", -kl)   # negative because Pyro maximises ELBO

        # ── DECODER → per-segment (aₑ, bₜ)
        #seg_outputs = self.decoder.forward_with_epistemic(z, sigma, mu)
        seg_outputs = self.decoder.forward_with_epistemic(z, sigma, mu, nu)

        all_locs   = []
        all_scales = []
        all_dfs    = []

        # Use the per-sample ν (mean over latent dims) as the df for each segment
        #nu_scalar = nu.mean(dim=-1, keepdim=True)   # [B, 1]

        for i in range(self.num_sections):
            a_e, b_t, df_t = seg_outputs[i]
            all_locs.append(a_e)
            all_scales.append(b_t)
            all_dfs.append(df_t)
        
        if return_attn:
            return all_locs, all_scales, all_dfs, attn_weights
        return all_locs, all_scales, all_dfs


# ==========================================
# 6.  MODEL_FN  (Pyro probabilistic model)
# ==========================================

def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    with pyro.poutine.scale(scale=kl_weight):
        locs, scales, dfs = bnn_model(x_global, x_local)

    if total_size is None:
        total_size = x_global.shape[0]

    with pyro.plate("data", size=total_size,
                    subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(
                dfs[i].squeeze(),
                locs[i].squeeze(),
                scales[i].squeeze(),
            )
            target = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)


# ==========================================
# 7.  LL / KL DIAGNOSTIC
# ==========================================

def get_ll_kl(model_fn, guide_fn, x_g, x_l, y, total_size):
    with torch.no_grad():
        
        locs, scales, dfs = bnn_model(x_g, x_l)
        ll = 0.0
        for i in range(len(locs)):
            d = torch.distributions.StudentT(
                df=dfs[i].squeeze(),
                loc=locs[i].squeeze(),
                scale=scales[i].squeeze(),
            )
            ll += d.log_prob(y[:, i]).sum().item()
        mu, logvar, logdf = bnn_model.encoder(x_g, x_l)
        sigma = torch.exp(0.5 * logvar)
        nu = torch.nn.functional.softplus(logdf) + 2.0
        g = torch.randn_like(mu)
        chi2 = torch.distributions.Chi2(df=nu).sample()
        z_sample = mu + sigma * g / torch.sqrt(chi2 / nu + 1e-8)
        
        # 4. Calculate Monte Carlo KL divergence
        kl = kl_student_t_to_student_t_mc(
            z_sample, mu, logvar, logdf, prior_nu=3.0
        ).item()
        
    return ll, kl    # ← already plain Python floats, return directly


# ==========================================
# 8.  TRAINING  (identical SVI loop)
# ==========================================

if __name__ == "__main__":
    from pyro.optim import PyroLRScheduler
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.infer.autoguide import AutoDiagonalNormal

    # ── Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data
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

    # ── Model  (new VAE architecture)
    pyro.clear_param_store()
    bnn_model = MatrixGNN_VAE(
        num_sections=num_segment,
        global_dim=9,
        local_dim=4,
        hidden_dim=13,
        latent_dim=LATENT_DIM,
        device=device,
    ).to(device)

    # ── Guide  (AutoDiagonalNormal over remaining Pyro latents)
    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
        pass   # no latent variables to guide — VAE handles z internally

    # ── Optimizer + cosine annealing
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

    # ── Training loop
    print("\n--- Starting Training ---")
    epochs     = 4000
    batch_size = 734
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_size    = len(train_dataset)
    print(f"Training dataset size: {total_size}")

    ramp_epochs = 500
    down_epoch  = 1000
    max_beta    = 0.9

    for epoch in range(epochs):
        epoch_loss = epoch_ll = epoch_kl = 0.0

        # KL annealing schedule (unchanged)
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
                ll, kl = get_ll_kl(model_fn, guide_fn,
                                   x_g_batch, x_l_batch, y_batch,
                                   total_size=total_size)
            epoch_ll += ll
            epoch_kl += kl
            current_lr = list(scheduler.optim_objs.values())[0] \
                .optimizer.param_groups[0]["lr"] if scheduler.optim_objs else 0.002
            avg_loss = epoch_loss / len(train_loader)
            """
            if (epoch == 0
                    or relative_epoch + 1 == CYCLE_LENGTH
                    or relative_epoch + 1 == ramp_epochs
                    or relative_epoch + 1 == down_epoch
                    or epoch + 1 == epochs
                    or relative_epoch == 0):
            """
                
            print(
                f"Epoch {epoch:05d} | LR: {current_lr:.6f} | "
                f"KL Wt: {current_kl_weight:.3f} | "
                f"ELBO Loss: {avg_loss:.2f} | "
                f"LL: {ll:.2f} | KL: {kl:.2f} | "
                f"ELBO check: {ll - kl:.2f}"
            )

    # ── Save
    pyro.get_param_store().save("ghost_bus_vae.pt")
    joblib.dump(scaler_y, "y_scaler_vae.pkl")
    print("\nModel weights and scaler saved successfully.")

    # ==========================================
    # 9. INFERENCE
    # ==========================================
    bnn_model.eval()

    def predict_mc(x_global, x_local, n_samples=50):
        """
        Draw n_samples predictions by sampling z from the bottleneck.
        Replaces Predictive — returns same dict format: {obs_section_i: [n_samples, B]}
        """
        all_samples = {f"obs_section_{i}": [] for i in range(num_segment)}

        with torch.no_grad():
            # Encoder runs once — deterministic
            mu, logvar, logdf = bnn_model.encoder(x_global, x_local)
            sigma = torch.exp(0.5 * logvar)                        # [B, D]
            nu    = torch.nn.functional.softplus(logdf) + 2.0      # [B, D]
            nu_scalar = nu.mean(dim=-1, keepdim=True)              # [B, 1]

            for _ in range(n_samples):
                # Sample z from Student-t posterior
                g    = torch.randn_like(mu)
                chi2 = torch.distributions.Chi2(df=nu).sample()   # [B, D]
                z    = mu + sigma * g / torch.sqrt(chi2 / nu + 1e-8)

                # Decode
                seg_outputs = bnn_model.decoder.forward_with_epistemic(z, sigma, mu, nu)

                for i, (a_e, b_t) in enumerate(seg_outputs):
                    obs_dist = torch.distributions.StudentT(
                        df=nu_scalar.squeeze(-1),
                        loc=a_e.squeeze(-1),
                        scale=b_t.squeeze(-1),
                    )
                    all_samples[f"obs_section_{i}"].append(obs_dist.sample())  # [B]

        # Stack → [n_samples, B] per segment
        return {k: torch.stack(v, dim=0) for k, v in all_samples.items()}
    
    

    list_of_predict   = []
    list_of_confidence = []
    list_of_actual    = []
    list_of_predict_sections    = [[] for _ in range(num_segment)]
    list_of_confidence_sections = [[] for _ in range(num_segment)]
    list_of_actual_sections     = [[] for _ in range(num_segment)]

    within_bound_count       = 0
    number_of_ratio          = 0
    section_within_bound_counts = 0
    error_abs_total = error_rate_squared = error_total = 0

    for j in range(len(x_global_val)):
        val_x_g = x_global_val[j:j+1]
        val_x_l = x_local_val[j:j+1]
        samples = predict_mc(val_x_g, val_x_l, n_samples=50)
        
        pred_means_scaled = []
        pred_stds_scaled  = []
        actuals_scaled    = []

        for i in range(num_segment):
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            pred_means_scaled.append(sec_samples.mean().item())
            pred_stds_scaled.append(sec_samples.std().item())
            actuals_scaled.append(y_val[j, i].item())

        pred_real   = scaler_y.inverse_transform([pred_means_scaled])[0]
        actual_real = scaler_y.inverse_transform([actuals_scaled])[0]
        std_real    = np.array(pred_stds_scaled) * scaler_y.scale_

        total_pred = trip_section_within_bound = 0
        print(f"\n--- Sample {j} ---")

        for i in range(num_segment):
            list_of_predict_sections[i].append(pred_real[i])
            spd = statistics.pvariance(list_of_predict_sections[i]) ** 0.5 \
                  if len(list_of_predict_sections[i]) > 1 else 0.0
            list_of_confidence_sections[i].append(std_real[i])
            scd = statistics.pvariance(list_of_confidence_sections[i]) ** 0.5 \
                  if len(list_of_confidence_sections[i]) > 1 else 0.0
            list_of_actual_sections[i].append(actual_real[i])
            sad = statistics.pvariance(list_of_actual_sections[i]) ** 0.5 \
                  if len(list_of_actual_sections[i]) > 1 else 0.0

            within = (pred_real[i] - std_real[i]) <= actual_real[i] <= (pred_real[i] + std_real[i])
            print(
                f"  Sec {i}: Pred {pred_real[i]:.1f}s | "
                f"Actual {actual_real[i]:.1f}s | "
                f"Conf ±{std_real[i]:.1f}s | "
                f"Within? {'YES' if within else 'NO'}"
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
        list_of_confidence.append(total_std)
        list_of_actual.append(total_act)
        if total_std > 0:
            number_of_ratio += total_pred / total_std

        error_total += (total_act - total_pred)
        error_abs_total += abs(total_act - total_pred)

        print(f"\nTotal ETA: {total_pred:.2f}s  (Actual: {total_act:.2f}s)")
        print(f"Confidence: ±{total_std:.2f}s")
        print(f"MAE so far: {error_abs_total/(j+1):.2f}s | "
              f"Bias: {error_total/(j+1):.2f}s")
        print(f"總共 {j+1} 筆，{within_bound_count} 筆落在區間內 | "
              f"平均置信度指標: {number_of_ratio/(j+1):.3f}")
