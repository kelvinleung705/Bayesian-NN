import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import statistics
import joblib 

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroModuleList, PyroParam
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import PyroLRScheduler

# 修正 1：正確的 StandardScaler 導入
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
    raw_local = raw_data_np[:, 9+num_segment+1:9+num_segment+1+(num_segment*4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)
    
    y_raw = raw_data_np[:, 9:9+num_segment]
    
    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y_raw), dtype=torch.float32)
    
    print("Total rows loaded:", raw_data_np.shape[0])
    print("Sample y_raw row 0:", y_raw[0])        
    print("Sample x_local row 0:", raw_local[0])  
    return x_global, x_local, y_scaled, scaler_y


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
        # ADD — prevents SiLU from saturating into fixed outputs
        self.norms = nn.ModuleList([
            nn.LayerNorm(output_dim).to(device)
            for _ in range(num_segments)
        ])

    def forward(self, x_inputs):
        outputs = []
        for i in range(self.num_segments):
            out = self.nets[i](x_inputs[i])
            out = self.norms[i](out)          # ADD
            out = torch.nn.functional.silu(out)
            outputs.append(out)
        return outputs

class NeighborMixingLayer(PyroModule): 
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.device = device
        
        # 確定性參數
        self.w_self = PyroParam(torch.full((num_segments,), 1.5, device=device))
        self.w_right = PyroParam(torch.full((num_segments,), 0.5, device=device))
        
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
            loc_std = torch.tensor(1.5, device=device) #1.0
            loc_bias_mu = torch.tensor(0.2, device=device) #0.1
            loc_bias_std = torch.tensor(0.3, device=device)  #1.0

            scale_std = torch.tensor(0.2, device=device)    #0.3
            scale_bias_mu = torch.tensor(0., device=device) # 從 1.5 開始，避免過度自信
            scale_bias_std = torch.tensor(1.0, device=device) #1.0

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

# ==========================================
# 4. PROBABILISTIC DEFINITION (Inverse Scale)
# ==========================================
bnn_model = None

def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    locs, scales, dfs = bnn_model(x_global, x_local)
    
    if total_size is None:
        total_size = x_global.shape[0]
    
    # 修正 4：反向數據縮放，保護梯度不消失
    data_scale = 1.0 / kl_weight
    
    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(dfs[i].squeeze(-1), locs[i].squeeze(-1), scales[i].squeeze(-1))
            target = y_true[:, i] if y_true is not None else None
            
            with pyro.poutine.scale(scale=data_scale):
                pyro.sample(f"obs_section_{i}", dist_i, obs=target)

def get_ll_kl(model_fn, guide, x_g, x_l, y, total_size):
    # 評估時使用真實的 kl_weight=1.0 獲取純淨日誌
    guide_trace = pyro.poutine.trace(guide).get_trace(x_g, x_l, y, total_size=total_size, kl_weight=1.0)
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model_fn, trace=guide_trace)
    ).get_trace(x_g, x_l, y, total_size=total_size, kl_weight=1.0)
    
    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()
    
    ll = 0.0
    kl = 0.0
    for name, site in model_trace.nodes.items():
        if site["type"] != "sample": continue
        if site["is_observed"]:
            ll += site["log_prob_sum"]
        else:
            if name not in guide_trace.nodes: continue
            log_p = site["log_prob_sum"]
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            kl += log_q - log_p 
    return ll.item(), kl.item()

# ==========================================
# 5. MAIN TRAINING & INFERENCE
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_new.xlsx"
    x_global_all, x_local_all, y_all, scaler_y = process_raw_data(file_path)
    
    # 修正 3：恢復正常的 20% 驗證集比例，避免驗證集為 0 崩潰
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_train = x_global_all[train_idx]
    x_local_train = x_local_all[train_idx]
    y_train = y_all[train_idx]
    
    x_global_val = x_global_all[val_idx].to(device)
    x_local_val = x_local_all[val_idx].to(device)
    y_val = y_all[val_idx].to(device)

    pyro.clear_param_store()

    # 提取縮放參數傳給模型
    y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
    y_std = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)

    bnn_model = MatrixGNN(
        num_sections=num_segment, 
        global_dim=9, 
        local_dim=4, 
        hidden_dim=32, 
        device=device,
        y_mean=y_mean,
        y_std=y_std
    ).to(device)
    
    # 標準的 Guide
    guide = AutoDiagonalNormal(model_fn, init_scale=0.5).to(device)
    
    CYCLE_LENGTH = 1000
    optimizer_args = {
        "optimizer": torch.optim.AdamW,
        "optim_args": {"lr": 0.002, "weight_decay": 0.01}
    }
    
    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=CYCLE_LENGTH, T_mult=1, eta_min=0.0001
        )
    
    scheduler = PyroLRScheduler(scheduler_constructor, optimizer_args)
    svi = SVI(model_fn, guide, scheduler, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    
    epochs = 4000
    batch_size = 734
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_size = len(train_dataset)
    print(f"Training dataset size: {total_size}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        relative_epoch = epoch % CYCLE_LENGTH
        ramp_epochs = 500  
        max_beta = 0.5
        if relative_epoch < ramp_epochs:
            current_kl_weight = max(0.00001, (relative_epoch / ramp_epochs)*max_beta)
        else:
            current_kl_weight = max_beta
            
        for x_g_batch, x_l_batch, y_batch in train_loader:
            x_g_batch, x_l_batch, y_batch = x_g_batch.to(device), x_l_batch.to(device), y_batch.to(device)
            
            raw_loss = svi.step(x_g_batch, x_l_batch, y_batch, total_size=total_size, kl_weight=current_kl_weight)
            
            # 還原 Loss 尺度用於記錄
            actual_loss = raw_loss * current_kl_weight
            epoch_loss += actual_loss
            
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            current_lr = list(scheduler.optim_objs.values())[0].optimizer.param_groups[0]["lr"] if scheduler.optim_objs else 0.002
            avg_loss = epoch_loss / len(train_loader)
            with torch.no_grad():
                ll, kl = get_ll_kl(model_fn, guide, x_global_val, x_local_val, y_val, total_size=total_size)
                ratio = abs(kl) / (abs(ll) + 1e-8)
                print(f"Epoch {epoch:05d} | LR: {current_lr:.6f} | KL Wt: {current_kl_weight:.3f} | ELBO Loss: {avg_loss:.2f} | LL: {ll:.2f} | KL: {kl:.2f} | Ratio: {ratio:.4f}")

    # 儲存模型
    pyro.get_param_store().save("ghost_bus_BLL_LayerNorm.pt")
    joblib.dump(scaler_y, "y_scaler.pkl")
    print("\nModel trained and saved successfully. Run plot.py to see your smooth Bollinger Bands!")