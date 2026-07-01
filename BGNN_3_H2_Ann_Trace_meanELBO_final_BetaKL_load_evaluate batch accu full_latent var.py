import statistics
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList, PyroParam
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

num_segment = 9

# ==========================================
# 1. DATA PRE-PROCESSING FOR INFERENCE
# ==========================================
def process_validation_data(file_path, loaded_scaler):
    """
    Reads data, transforms it, and perfectly recreates the 20% validation split
    used in the original training script so the samples match 1-to-1.
    """
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)

    end = 14 + num_segment + 1 + (num_segment * 4)
    df_subset = df.iloc[:, 0:end]
    df_subset = df_subset.dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    raw_data_np = df_subset.values.astype(np.float32)
    
    # Extract X features directly
    x_global_all = torch.tensor(raw_data_np[:, 0:9], dtype=torch.float32)
    
    raw_local = raw_data_np[:, 9+num_segment+1:9+num_segment+1+(num_segment*4)]
    x_local_all = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)
    
    # Extract and Normalize Targets (Y) using LOADED scaler
    y_raw = raw_data_np[:, 9:9+num_segment]
    y_scaled_all = torch.tensor(loaded_scaler.transform(y_raw), dtype=torch.float32)
    print("Total rows loaded:", raw_data_np.shape[0])
    print("Sample y_raw row 0:", y_raw[0])        # Should be travel times, not global features
    print("Sample x_local row 0:", raw_local[0])  # Should be local segment features
    print("y_raw mean per segment:", y_raw.mean(axis=0))
    print("y_raw std per segment:", y_raw.std(axis=0))
    
    # Replicate the original Validation Split
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_val = x_global_all[idx]
    x_local_val = x_local_all[idx]
    y_val = y_scaled_all[idx]
    
    return x_global_val, x_local_val, y_val


# ==========================================
# 2. DETERMINISTIC ENCODER LAYERS (Standard PyTorch)
# ==========================================
class LocalIsolationLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.nets = PyroModuleList([
            PyroModule[nn.Linear](input_dim, output_dim).to(device) 
            for _ in range(num_segments)
        ])
            
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
        self.device = device
        
        # 確定性參數
        self.w_self = PyroParam(torch.full((num_segments,), 2.0, device=device))
        self.w_right = PyroParam(torch.zeros(num_segments, device=device))
        
        self.nets_1 = PyroModuleList([])
        self.nets_2 = PyroModuleList([])
        
        for i in range(num_segments):
            net_input_dim = input_dim * 2
            net_1 = PyroModule[nn.Linear](net_input_dim, output_dim, device=device)
            self.nets_1.append(net_1)
    
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        
        for i in range(num_segments):
            net_2 = PyroModule[nn.Linear](output_dim, output_dim, device=device)
            self.nets_2.append(net_2)
        
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        
    def forward(self, prev_layer_outputs):
        outputs = []
        for i in range(self.num_segments):
            ws = torch.nn.functional.softplus(self.w_self[i])
            wr = torch.nn.functional.softplus(self.w_right[i])
            
            self_feat = prev_layer_outputs[i] * ws
            
            if i < self.num_segments - 1:
                right_feat = prev_layer_outputs[i+1] * wr
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
# 3. THE "MATRIX" GNN MODEL (BLL Wrapper)
# ==========================================
class MatrixGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=8, device='cuda', y_mean=None, y_std=None):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        input_dim = global_dim + local_dim + 1 
        
        # 儲存 Y 的縮放參數，用來在 forward 內安全縮放時鐘
        if y_mean is not None and y_std is not None:
            self.register_buffer('y_mean_total', y_mean.sum())
            self.register_buffer('y_std_total', y_std.sum())
        else:
            self.register_buffer('y_mean_total', torch.tensor(470.0, device=device)) # 預設大約 8 分鐘
            self.register_buffer('y_std_total', torch.tensor(80.0, device=device))

        # 確定性 Encoder
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        self.prop_layers = nn.ModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(2) 
        ])
        
        # 貝氏 Decoder Heads
        final_dim = hidden_dim
        self.heads_loc = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df = PyroModuleList([])
        
        for i in range(self.num_sections):
            zero = torch.tensor(0., device=device)
            loc_std = torch.tensor(1.0, device=device)
            loc_bias_mu = torch.tensor(0., device=device)
            loc_bias_std = torch.tensor(1., device=device)

            scale_std = torch.tensor(0.3, device=device)    #0.3
            scale_bias_mu = torch.tensor(0., device=device) # 從 1.5 開始，避免過度自信
            scale_bias_std = torch.tensor(1.0, device=device)

            df_std = torch.tensor(1.0, device=device)
            df_bias_mu = torch.tensor(0., device=device)
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
        
        current_time = torch.zeros(batch_size, 1).to(device)
        
        all_locs = []
        all_scales = []
        all_dfs = []
        
        # --- THE AUTOREGRESSIVE WATERFALL LOOP ---
        for current_section in range(self.num_sections):

            inputs_list = []
            for i in range(self.num_sections):
                loc_i = all_sections_data[:, i, :]
                
                if i <= current_section:
                    time_i = current_time
                else:
                    time_i = torch.zeros(batch_size, 1).to(device)
                
                
                time_i = torch.clamp(time_i, min=-15.0, max=15.0)
                inputs_list.append(torch.cat([global_features, loc_i, time_i], dim=1))
                
                
                
            h_current = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h_current = layer(h_current)
            
            final_feat = h_current[current_section] 
            
            loc = self.heads_loc[current_section](final_feat)
            scale = torch.nn.functional.softplus(self.heads_scale[current_section](final_feat)) + 1e-3 
            df = torch.nn.functional.softplus(self.heads_df[current_section](final_feat)) + 2.5
            
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            
            
            # 修正 2：非在位累加，維持計算圖完整
            current_time = current_time + loc
            
        return all_locs, all_scales, all_dfs



class MatrixGNN_WithLatent(MatrixGNN):
    """Subclass that also returns the GNN's final hidden states."""
    
    def forward_with_latent(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        current_time = torch.zeros(batch_size, 1).to(device)

        all_locs, all_scales, all_dfs = [], [], []
        all_hidden = []  # ← NEW: collect hidden states per section step

        for current_section in range(self.num_sections):
            inputs_list = []
            for i in range(self.num_sections):
                loc_i = all_sections_data[:, i, :]
                if i <= current_section:
                    time_i = current_time
                else:
                    time_i = torch.zeros(batch_size, 1).to(device)

                if time_i.abs().mean().item() > 15:
                    time_i = torch.zeros(batch_size, 1).to(device)

                inp = torch.cat([global_features, loc_i, time_i], dim=1)
                inputs_list.append(inp)

            h_current = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h_current = layer(h_current)

            # ← Grab the hidden state of the current section at this step
            all_hidden.append(h_current[current_section].detach().cpu())

            final_feat = h_current[current_section]
            loc   = self.heads_loc[current_section](final_feat)
            scale = torch.nn.functional.softplus(self.heads_scale[current_section](final_feat)) + 1e-3
            df    = torch.nn.functional.softplus(self.heads_df[current_section](final_feat)) + 2.5

            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            current_time = current_time + loc

        # Stack: (batch_size, num_segments * hidden_dim)
        latent = torch.cat(all_hidden, dim=1)
        return all_locs, all_scales, all_dfs, latent

def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
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
# NETRON EXPORT (Outside Main Scope)
# ==========================================
import torch.onnx

class NetronExportWrapper(PyroModule):
    """
    Wraps MatrixGNN with a frozen guide trace so all PyroSample
    weights become deterministic tensor ops that can be traced/exported.
    """
    def __init__(self, bnn_model, frozen_trace):
        super().__init__()
        self.bnn_model = bnn_model
        self.frozen_trace = frozen_trace

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor):
        with pyro.poutine.replay(trace=self.frozen_trace):
            locs, scales, dfs = self.bnn_model(x_global, x_local)
        
        loc_out   = torch.cat(locs,   dim=1)   
        scale_out = torch.cat(scales, dim=1)
        df_out    = torch.cat(dfs,    dim=1)
        return loc_out, scale_out, df_out


# ==========================================
# 5. INFERENCE & METRIC SUMMARY
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy
    #saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu.pt" # Replace with exact saved params file name
    #saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu_Jan_to_Apr.pt" # Replace with exact saved params file name
    #saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu4_fixed.pt" # Replace with exact saved params file name
    #saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu4_fixed_10000_new_encoding.pt"
    #saved_params_path = "ghost_bus_model_cycle_KL_9_accu4_fixed_8000_new_encoding.pt"
    saved_params_path = "ghost_bus_BLL.pt"
    #saved_params_path = "ghost_bus_model_cycle_0.7_clamp_5_27_1055.pt"
    print(saved_params_path)
    
    #saved_scaler_path = "y_scaler_1.pkl"  
    #saved_scaler_path = "y_scaler_4_fixed.pkl"              # Replace with exact saved scaler file name
    #saved_scaler_path = "y_scaler_4_fixed_10000.pkl"              # Replace with exact saved scaler file name
    #saved_scaler_path = "y_scaler_4_fixed_20000.pkl"              # Replace with exact saved scaler file name
    #saved_scaler_path = "y_scaler_Jan_to_Apr.pkl" 
    #saved_scaler_path = "y_scaler_4_trash.pkl"   
    saved_scaler_path = "y_scaler_4_full_d.pkl"              # Replace with exact saved scaler file name
    print(saved_scaler_path)   
    #file_path = "bad_visibility.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_month_sorted.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2026.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_June.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy2_flagged.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_40.xlsx"
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_new.xlsx"
    print(file_path)
    
    
    # ==========================================
    # 1. LOAD SCALER & DATA
    # ==========================================
    print("\n--- Loading Scaler & Processing Data ---")
    loaded_scaler = joblib.load(saved_scaler_path)
    
    x_global_val, x_local_val, y_val = process_validation_data(file_path, loaded_scaler)
    
    x_global_val = x_global_val.to(device)
    x_local_val = x_local_val.to(device)
    y_val = y_val.to(device)

    # ==========================================
    # 2. RECONSTRUCT & LOAD MODEL
    # ==========================================
    print("\n--- Initializing and Loading Model Weights ---")
    pyro.clear_param_store()

    # Load ParamStore FIRST so the Guide correctly latches onto the loaded weights
    pyro.get_param_store().load(saved_params_path, map_location=device.type)

    bnn_model = MatrixGNN_WithLatent(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32, device=device).to(device)
    base_guide = AutoDiagonalNormal(model_fn).to(device)
    bnn_model.eval()
    base_guide.eval()
    
    # NOTE: We purposely DO NOT call bnn_model.eval() here.
    # Your original training script did not use .eval() during inference, 
    # which left Dropout active and added variance to your targets.
    # To mimic the exact results you saw before, we leave it in train mode!
    
    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=0.1):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true, total_size=total_size, kl_weight=kl_weight)
            
    # Dummy trace to securely link network dimensions to Pyro's Param Store
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        guide_fn(x_global_val[0:1], x_local_val[0:1], y_true=dummy_y)

    print("Model parameters loaded and correctly linked! Starting Inference.")
    
    # ==========================================
# NETRON EXPORT — Add after Section 2 (Model Load)
# ==========================================
import torch.onnx

class NetronExportWrapper(nn.Module):
    """
    Wraps MatrixGNN with a frozen guide trace so all PyroSample
    weights become deterministic tensor ops that can be traced/exported.
    """
    def __init__(self, bnn_model, frozen_trace):
        super().__init__()
        self.bnn_model = bnn_model
        self.frozen_trace = frozen_trace

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor):
        # Replay frozen weights through the model — no live sampling
        with pyro.poutine.replay(trace=self.frozen_trace):
            locs, scales, dfs, feats = self.bnn_model(x_global, x_local)
        
        # Concatenate per-segment outputs into single tensors
        loc_out   = torch.cat(locs,   dim=1)   # (batch, num_segment)
        scale_out = torch.cat(scales, dim=1)
        df_out    = torch.cat(dfs,    dim=1)
        return loc_out, scale_out, df_out

# ==========================================
# 3. FAST BATCHED INFERENCE LOOP
# ==========================================
print("\n--- Starting Fast Batched Prediction Test ---")
    
# 1. Setup DataLoader for Batching
batch_size = 1024  # Process 1024 samples at the exact same time
val_dataset = TensorDataset(x_global_val, x_local_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
predictive = Predictive(model_fn, guide=guide_fn, num_samples=200)
    
all_pred_real = []
all_actual_real = []
all_std_real = []

# 2. Run Inference on GPU in Batches
with torch.no_grad():
    for x_g_batch, x_l_batch, y_batch in val_loader:
            
        # Run 50 samples for the ENTIRE batch at once!
        # Shape of each section will be: (50, batch_size)
        samples = predictive(x_g_batch, x_l_batch)
            
        batch_means_scaled = []
        batch_stds_scaled = []
            
        for i in range(num_segment):
            # Squeeze handles any trailing dimensions
            sec_samples = samples[f"obs_section_{i}"].squeeze(-1)
                
            # Calculate mean and std across the 50 samples for every row in the batch simultaneously
            batch_means_scaled.append(sec_samples.mean(dim=0))
            batch_stds_scaled.append(sec_samples.std(dim=0))
                
        # Stack into shape (batch_size, num_segments) and move to CPU exactly once
        batch_means_scaled = torch.stack(batch_means_scaled, dim=1).cpu().numpy()
        batch_stds_scaled = torch.stack(batch_stds_scaled, dim=1).cpu().numpy()
        y_batch_np = y_batch.cpu().numpy()
            
        # 3. Vectorized Inverse Transform (Instantaneous)
        pred_real = loaded_scaler.inverse_transform(batch_means_scaled)
        actual_real = loaded_scaler.inverse_transform(y_batch_np)
            
        # Standard deviation scales linearly with the scaler's scale_ factor
        std_real = batch_stds_scaled * loaded_scaler.scale_
            
        all_pred_real.append(pred_real)
        all_actual_real.append(actual_real)
        all_std_real.append(std_real)

# 4. Combine all batches into massive, fast NumPy arrays
pred_real = np.vstack(all_pred_real)
actual_real = np.vstack(all_actual_real)
std_real = np.vstack(all_std_real)
    
# ════════════════════════════════════════════════════
# ── t-SNE STEP 2: Collect Latent Vectors ────────────  ← INSERT HERE
# ════════════════════════════════════════════════════
all_latents  = []
all_act_real = []
all_feats = {i: [] for i in range(num_segment)}   # ← ADD
    
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── NEW: storage for direct loc / scale per segment ──
all_locs_direct   = {i: [] for i in range(num_segment)}
all_scales_direct = {i: [] for i in range(num_segment)}

with torch.no_grad():
    for x_g_batch, x_l_batch, y_batch in val_loader:

        # ---- Use a frozen guide trace for deterministic weights ----
        guide_trace = pyro.poutine.trace(guide_fn).get_trace(
            x_g_batch, x_l_batch
        )

        with pyro.poutine.replay(trace=guide_trace):
            locs, scales, _, latent = bnn_model.forward_with_latent(x_g_batch, x_l_batch)
                
        # ── NEW: collect loc and scale for each segment ──
        for i in range(num_segment):
            all_locs_direct[i].append(locs[i].squeeze(-1).cpu().numpy())
            all_scales_direct[i].append(scales[i].squeeze(-1).cpu().numpy())
            
        hidden_dim = latent.shape[1] // num_segment
        for s in range(num_segment):
            chunk = latent[:, s * hidden_dim : (s+1) * hidden_dim].cpu()
            all_feats[s].append(chunk)  

        all_latents.append(latent.cpu().numpy())

        # Inverse-transform y for coloring by actual total travel time
        y_np = y_batch.cpu().numpy()
        actual = loaded_scaler.inverse_transform(y_np)
        all_act_real.append(actual.sum(axis=1))   # total trip time per sample

latent_matrix = np.vstack(all_latents)       # (N, num_seg * hidden_dim)
color_values  = np.concatenate(all_act_real) # (N,) — total actual travel time
    

    
# ── NEW: concatenate loc / scale arrays ──
locs_direct   = {}
scales_direct = {}
for s in range(num_segment):
    locs_direct[s]   = np.concatenate(all_locs_direct[s])
    scales_direct[s] = np.concatenate(all_scales_direct[s])
    
# ✅ ADD THIS — concatenate each section's list into one tensor
for s in range(num_segment):
    all_feats[s] = torch.cat(all_feats[s], dim=0)   # ← ADD
    
# ════════════════════════════════════════════════════
# ── t-SNE STEP 3: Run t-SNE & Plot ──────────────────  ← INSERT HERE
# ════════════════════════════════════════════════════
    
    
print(f"Latent matrix shape: {latent_matrix.shape}")  # e.g. (5000, 288)

tsne = TSNE(
    n_components=2,
    perplexity=40,       # try 20–80; higher = more global structure
    learning_rate=200,
    max_iter=1000,
    random_state=42,
    init='pca',          # PCA init is more stable than random
    verbose=1
)
tsne_result = tsne.fit_transform(latent_matrix)  # (N, 2)

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Color by total actual travel time ---
sc = axes[0].scatter(
    tsne_result[:, 0], tsne_result[:, 1],
    c=color_values,
    cmap='plasma',
    alpha=0.6,
    s=8
)
plt.colorbar(sc, ax=axes[0], label='Total Actual Travel Time (s)')
axes[0].set_title('t-SNE Latent Space\n(colored by actual total travel time)')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

# --- Plot 2: Color by prediction error ---
pred_errors = np.abs(color_values - np.vstack(all_pred_real).sum(axis=1))
sc2 = axes[1].scatter(
    tsne_result[:, 0], tsne_result[:, 1],
    c=pred_errors,
    cmap='RdYlGn_r',     # red = high error, green = low
    alpha=0.6,
    s=8,
    vmax=np.percentile(pred_errors, 95)  # clip outliers
)
plt.colorbar(sc2, ax=axes[1], label='|Prediction Error| (s)')
axes[1].set_title('t-SNE Latent Space\n(colored by absolute prediction error)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig("tsne_latent_space.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → tsne_latent_space.png")
    
    
# ════════════════════════════════════════════════════
# ── BONUS: Color by Global Feature ──────────────────  ← INSERT HERE (optional)
# ════════════════════════════════════════════════════
    
# x_global_val[:, 0] — replace 0 with whichever column is time-of-day
time_of_day = x_global_val[:, 0].cpu().numpy()

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    tsne_result[:, 0], tsne_result[:, 1],
    c=time_of_day[:len(tsne_result)],
    cmap='twilight',
    alpha=0.6, s=8
)
plt.colorbar(sc, label='Global Feature [0] (e.g. time of day)')
plt.title('t-SNE colored by global input feature')
plt.savefig("tsne_by_global_feature.png", dpi=150)
    
    
# ════════════════════════════════════════════════════
# ── Mean Variance----------------- ──────────────────  ← INSERT HERE (optional)
# ════════════════════════════════════════════════════
    
    
hidden_dim   = next(iter(all_feats.values())).shape[1]
num_sections = num_segment

fig, axes = plt.subplots(1, num_sections, figsize=(5 * num_sections, 5), sharey=False)
if num_sections == 1:
    axes = [axes]

for s, ax in enumerate(axes):
    feats  = all_feats[s].numpy()
    mean   = feats.mean(axis=1)
    var    = feats.var(axis=1)
    mean_centered = mean - np.median(mean)
    var_centered = var - np.median(var)

    # ✅ Filter extremes using IQR
    def iqr_mask(arr, k=1.5):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        return (arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)

    mask      = iqr_mask(mean_centered) & iqr_mask(var_centered)
    mean_plot = mean_centered[mask]
    var_plot  = var_centered[mask]

    ax.scatter(mean_plot, var_plot,          # ✅ removed c=dim_idx, cmap
                color="steelblue",            # ✅ plain color, all samples same
               alpha=0.4,                    # ✅ transparent for dense overlap
               s=15,                         # ✅ small dots for N samples
               edgecolors="none")            # ✅ no border on small dots
    """
    for idx, (m, v) in zip(dim_idx, zip(mean_plot, var_plot)):
        ax.annotate(f"d{idx}", (m, v),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="dimgray")
    """

    # ✅ Compute symmetric limits so 0 is always dead center
    # ✅ REPLACE the x_abs/y_abs block with this
    x_abs = np.abs(mean_plot).max() if len(mean_plot) > 0 else 1.0

    # Separate positive/negative max so 0 is always the visual midpoint
    y_pos = var_plot[var_plot >= 0].max() if (var_plot >= 0).any() else 1.0
    y_neg = np.abs(var_plot[var_plot < 0].min()) if (var_plot < 0).any() else 1.0
    y_abs = max(y_pos, y_neg)   # ✅ take the larger side, mirror it to both
    pad   = 0.15   # 15% breathing room

    ax.set_xlim(-(x_abs * (1 + pad)), x_abs * (1 + pad))
    ax.set_ylim(-(y_abs * (1 + pad)), y_abs * (1 + pad))

    # Reference lines — always visually centred now
    ax.axvline(0, color="red",  linewidth=1.0, linestyle="--", alpha=0.7, label="mean = median")
    ax.axhline(0, color="blue", linewidth=1.0, linestyle="--", alpha=0.7, label="var = median(var)")

    n_filtered = (~mask).sum()
    ax.set_title(f"Section {s}  [{n_filtered} extreme dim(s) hidden]")
    ax.set_xlabel("Mean − Median  (skew)")
    ax.set_ylabel("Variance − Median(Variance)")
    ax.legend(fontsize=8)
        
    print(f"Section {s}: total={len(mean_centered)}, kept={mask.sum()}, removed={n_filtered}")
    print(f"  mean IQR={np.percentile(mean_centered,75)-np.percentile(mean_centered,25):.6f}")
    print(f"  var  IQR={np.percentile(var_centered,75)-np.percentile(var_centered,25):.6f}")
    print(f"  activation min={feats.min():.3f}, max={feats.max():.3f}")
    print(f"  activation mean={feats.mean():.3f}, std={feats.std():.3f}")
    print(f"  raw var min={feats.var(axis=1).min():.3f}, max={feats.var(axis=1).max():.3f}")
    print(f"  raw var median={np.median(feats.var(axis=1)):.3f}")

plt.suptitle("Per-Sample N(m,u) — Median Centered", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("latent_centered_no_extreme.png", dpi=150)
plt.show()
    
    
for s, ax in enumerate(axes):
    feats  = all_feats[s].numpy()
    mean   = feats.mean(axis=1)
    var    = feats.var(axis=1)

    mean_centered = mean - np.median(mean)
    var_centered  = var  - np.median(var)

    def iqr_mask(arr, k=10.5):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        return (arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)

    for k in [1.5, 2.0, 2.5, 3.0, None]:
        if k is None:
            mask = np.ones(len(mean_centered), dtype=bool)
            break
        mask = iqr_mask(mean_centered, k) & iqr_mask(var_centered, k)
        if mask.sum() / len(mean_centered) >= 0.90:
            break

    mean_plot = mean_centered[mask]
    var_plot  = var_centered[mask]
    n_filtered = (~mask).sum()

    # ✅ Per-sample prediction std for this section, same mask applied
    pred_std_plot = std_real[mask, s]          # (N_kept,) — B(s) per sample
    """
    scatter = ax.scatter(mean_plot, var_plot,
                         c=pred_std_plot,       # ← color by B(s)
                         cmap="plasma",
                         alpha=0.6,
                         s=10,
                         edgecolors="none",
                         linewidths=0,          # ✅ fully removes edge rendering
                        rasterized=True)
    """
    vmin_std = np.percentile(pred_std_plot, 5)
    vmax_std = np.percentile(pred_std_plot, 95)

    scatter = ax.scatter(mean_plot, var_plot,
                        c=pred_std_plot,
                        cmap="plasma",
                        alpha=0.6,
                        s=10,
                        edgecolors="none",
                        linewidths=0,
                        rasterized=True,
                        vmin=vmin_std,   # ← ignore bottom 5% extreme
                        vmax=vmax_std)   # ← ignore top 5% extreme

    plt.colorbar(scatter, ax=ax, label=f"Pred Std B(s) [seconds]")

    x_pos = mean_plot[mean_plot >= 0].max() if (mean_plot >= 0).any() else 1.0
    x_neg = np.abs(mean_plot[mean_plot < 0].min()) if (mean_plot < 0).any() else 1.0
    x_abs = max(x_pos, x_neg)

    y_pos = var_plot[var_plot >= 0].max() if (var_plot >= 0).any() else 1.0
    y_neg = np.abs(var_plot[var_plot < 0].min()) if (var_plot < 0).any() else 1.0
    y_abs = max(y_pos, y_neg)

    pad = 0.15
    ax.set_xlim(-x_abs * (1 + pad),  x_abs * (1 + pad))
    ax.set_ylim(-y_abs * (1 + pad),  y_abs * (1 + pad))

    ax.axvline(0, color="red",  linewidth=1.0, linestyle="--", alpha=0.7, label="median(m)")
    ax.axhline(0, color="blue", linewidth=1.0, linestyle="--", alpha=0.7, label="median(u)")

    ax.set_title(f"Section {s}  [{n_filtered} extreme sample(s) hidden]\nN={mask.sum()} entries")
    ax.set_xlabel("m − median(m)")
    ax.set_ylabel("u − median(u)")
    ax.legend(fontsize=8)
    
    
    
plt.suptitle("Per-Sample N(m,u) — Median Centered", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("latent_centered_no_extreme_color_variance.png", dpi=150)
plt.show()
    
    
# ════════════════════════════════════════════════════
# ── t-SNE: Color by Direct Loc (Predicted Mean) ─────
# ════════════════════════════════════════════════════
    
# Inverse-transform locs back to real seconds for readability
# locs are scaled, stack them into (N, num_segment) then inverse_transform
locs_cols   = []
scales_cols = []
for i in range(num_segment):
    locs_cols.append(locs_direct[i])
    scales_cols.append(scales_direct[i])

locs_matrix   = np.column_stack(locs_cols)
scales_matrix = np.column_stack(scales_cols)

locs_real   = loaded_scaler.inverse_transform(locs_matrix)        # (N, num_segment)
# Scales are std-like: multiply by scaler's per-segment std to get real seconds
scales_real = scales_matrix * loaded_scaler.scale_[np.newaxis, :] # (N, num_segment)

# ── Plot: one subplot per segment, colored by loc ──
# ════════════════════════════════════════════════════
# ── Loc vs Scale — Median Centered, Per Segment ─────
# ════════════════════════════════════════════════════

ncols = 3
nrows = (num_segment + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6 * ncols, 5 * nrows),
                         constrained_layout=True)
axes_flat = list(axes.flat)

for i in range(num_segment):
    ax = axes_flat[i]

    loc_col   = locs_real[:, i]    # (N,) in real seconds
    scale_col = scales_real[:, i]  # (N,) in real seconds

    # Median-center both axes so (0, 0) is always the visual midpoint
    loc_centered   = loc_col   - np.median(loc_col)
    scale_centered = scale_col - np.median(scale_col)

    # IQR filter to suppress extreme outliers
    def iqr_mask(arr, k=1.5):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        return (arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)

    mask = iqr_mask(loc_centered) & iqr_mask(scale_centered)
    loc_plot   = loc_centered[mask]
    scale_plot = scale_centered[mask]
    n_hidden   = (~mask).sum()
    
    loc_plot = loc_centered
    scale_plot = scale_centered
    
    ax.scatter(
        scale_plot, loc_plot,
        color='steelblue',
        alpha=0.4,
        s=10,
        edgecolors='none'
    )

    # Symmetric axis limits so median sits dead centre
    x_abs = np.abs(scale_plot).max() if len(scale_plot) > 0 else 1.0
    y_abs = np.abs(loc_plot).max()   if len(loc_plot)   > 0 else 1.0
    pad   = 0.15

    ax.set_xlim(-(x_abs * (1 + pad)), x_abs * (1 + pad))
    ax.set_ylim(-(y_abs * (1 + pad)), y_abs * (1 + pad))

    # Crosshair at (0, 0) = median of both
    ax.axvline(0, color='red',  linewidth=1.0, linestyle='--', alpha=0.7, label='median(scale)')
    ax.axhline(0, color='blue', linewidth=1.0, linestyle='--', alpha=0.7, label='median(loc)')

    ax.set_title(f'Segment {i}  [{n_hidden} outlier(s) hidden]')
    ax.set_xlabel('Scale − Median(Scale)  (uncertainty)')
    ax.set_ylabel('Loc − Median(Loc)  (predicted time)')
    ax.legend(fontsize=8)

    print(f"Seg {i}: median loc={np.median(loc_col):.1f}s  median scale={np.median(scale_col):.1f}s  kept={mask.sum()}/{len(mask)}")

# Hide unused subplots
for ax in axes_flat[num_segment:]:
    ax.set_visible(False)

fig.suptitle('Loc vs Scale — Median Centred (per segment)',
             fontsize=14, fontweight='bold')
plt.savefig('loc_vs_scale_median_centred_full.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → loc_vs_scale_median_centred_full.png")

    

# ==========================================
# 4. FAST VECTORIZED METRICS CALCULATION
# ==========================================
# Calculate Total ETAs (Sum across sections for each trip)
total_pred = pred_real.sum(axis=1)
total_act = actual_real.sum(axis=1)
    
# Total Standard Deviation (Square root of sum of variances)
total_std = np.sqrt(np.sum(std_real**2, axis=1))

# Calculate global metrics natively in NumPy (Millisecond execution)
total_samples = len(total_act)
    
# Bounds checking
lower_bound = total_pred - total_std
upper_bound = total_pred + total_std
within_bound_mask = (total_act >= lower_bound) & (total_act <= upper_bound)
within_bound_count = np.sum(within_bound_mask)
    
# Section-level bounds checking
sec_lower_bounds = pred_real - std_real
sec_upper_bounds = pred_real + std_real
sec_within_bounds_mask = (actual_real >= sec_lower_bounds) & (actual_real <= sec_upper_bounds)
section_within_bound_counts = np.sum(sec_within_bounds_mask) # Total correct sections

# Running Errors
error_rates = (total_act - total_pred)
abs_errors = np.abs(total_act - total_pred)
    
# Cumulative stats (Replacing the slow statistics.pvariance in a loop)
# Using pandas expanding to recreate your "running deviation" concept quickly
pred_running_std = pd.Series(total_pred).expanding().std().fillna(0).values
conf_running_std = pd.Series(total_std).expanding().std().fillna(0).values
act_running_std = pd.Series(total_act).expanding().std().fillna(0).values
    
number_of_ratio_sum = np.sum(total_pred[total_std > 0] / total_std[total_std > 0])

# ==========================================
# 5. OPTIONAL: PRINT OUT INDIVIDUAL RESULTS
# ==========================================
# Warning: Printing thousands of lines to the console is very slow!
# I recommend keeping this to the first 5 samples just to verify, 
# but I have left the full loop here in case you need it.
max_pred = 0
max_conf = 0
    
print_limit = len(x_global_val) # Change this to 5 if you just want a quick peek
all_loc = 0
all_std = 0
    
    
overload = 0
list_of_extreme = []
max_pred = 0
max_conf = 0
    
# 1. Create variables to sum only the VALID (non-extreme) samples
valid_loc_sum = 0.0
valid_std_sum = 0.0
valid_count = 0

for j in range(print_limit):
    # EXTREME CONDITION
    if total_pred[j] < total_std[j]:
        overload += 1
        list_of_extreme.append(j)
            
    # NORMAL / VALID CONDITION (Skips the extremes automatically)
    else:
        valid_loc_sum += total_pred[j]
        valid_std_sum += total_std[j]
        valid_count += 1
            
        print(f"\n--- Sample {j} ---")
        for i in range(num_segment):
            is_in_bound = "YES" if sec_within_bounds_mask[j, i] else "NO"
            print(f"  Sec {i}: Pred {pred_real[j, i]:.1f}s | Actual {actual_real[j, i]:.1f}s | Conf +/- {std_real[j, i]:.1f}s | Within Bound? {is_in_bound}")
            
        if total_pred[j] > max_pred:
            max_pred = total_pred[j]
        if total_std[j] > max_conf:
            max_conf = total_std[j]
                
        print(f"\nTotal ETA: {total_pred[j]:.2f} seconds (Actual: {total_act[j]:.2f})")
        print(f"Within Bound? : {'YES' if within_bound_mask[j] else 'NO'}")
        print(f"Confidence: +/- {total_std[j]:.2f} seconds")
            
        conf_level = total_pred[j] / total_std[j] if total_std[j] > 0 else 0
        print(f"Confidence Level: {conf_level:.2f}")
    
        # Current running error metrics
        current_error_tendency = np.sum(error_rates[:j+1]) / (j + 1)
        current_error_squared = np.sum(abs_errors[:j+1]) / (j + 1)
        
        print(f"\nPrediction Std Deviation: {pred_running_std[j]:.2f} , Confidence Std Deviation: {conf_running_std[j]:.2f} , Actual Std Deviation: {act_running_std[j]:.2f}")
        print(f"Error (MAE running): {current_error_squared:.2f}")
        print(f"Error Tendency (Bias running): {current_error_tendency:.2f}")


# 2. Safely calculate averages avoiding division by zero
avg_loc = valid_loc_sum / valid_count if valid_count > 0 else 0
avg_std = valid_std_sum / valid_count if valid_count > 0 else 0

# ==========================================
# 6. FINAL METRICS SUMMARY
# ==========================================
print("\n==============================================")
print(f"總共 {total_samples} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
print(f"平均 {num_segment} Section，有 {section_within_bound_counts / total_samples:.2f} section 落在預測區間內。")
print(f"平均置信度指標: {number_of_ratio_sum / total_samples:.2f}")
    
# 3. Print the new, clean averages
print(f"過濾 {overload} 筆極端值後的 平均預測時間: {avg_loc:.2f}")
print(f"過濾 {overload} 筆極端值後的 平均置信度: {avg_std:.2f}")
print("==============================================")
print(f"Max Extreme Pred: {max_pred:.2f}")
print(f"Max Extreme Conf: {max_conf:.2f}")
print(f"Total Overloads skipped: {overload}")
    
    
    