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



def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    with pyro.poutine.scale(scale=kl_weight):
        locs, scales, dfs = bnn_model(x_global, x_local)
    
    if total_size is None:
        total_size = x_global.shape[0]
    
    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(dfs[i].squeeze(), locs[i].squeeze(), scales[i].squeeze())
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
    
    saved_params_path = "ghost_bus_BLL.pt"
    saved_scaler_path = "y_scaler_4_full_d.pkl"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_new.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_June.xlsx"
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2026.xlsx"
    
    # 1. LOAD SCALER & DATA
    print("\n--- Loading Scaler & Processing Data ---")
    loaded_scaler = joblib.load(saved_scaler_path)
    
    x_global_val, x_local_val, y_val = process_validation_data(file_path, loaded_scaler)
    
    x_global_val = x_global_val.to(device)
    x_local_val = x_local_val.to(device)
    y_val = y_val.to(device)

    # 2. RECONSTRUCT MODEL & LOAD WEIGHTS
    print("\n--- Initializing and Loading Model Weights ---")
    pyro.clear_param_store()
    
    global bnn_model
    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32, device=device).to(device)
    base_guide = AutoDiagonalNormal(model_fn).to(device)
    
    bnn_model.eval()
    base_guide.eval()
    
    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=0.1):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true, total_size=total_size, kl_weight=kl_weight)
            
    # Dummy trace
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        guide_fn(x_global_val[0:1], x_local_val[0:1], y_true=dummy_y)

    pyro.get_param_store().load(saved_params_path, map_location=device.type)
    print("Model parameters loaded successfully! Starting Inference.")

    # 3. FAST BATCHED INFERENCE LOOP (Fixed Indentation!)
    print("\n--- Starting Fast Batched Prediction Test ---")
    
    batch_size = 1024  
    val_dataset = TensorDataset(x_global_val, x_local_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=50)
    
    all_pred_real = []
    all_actual_real = []
    all_std_real = []

    with torch.no_grad():
        for x_g_batch, x_l_batch, y_batch in val_loader:
            samples = predictive(x_g_batch, x_l_batch)
            
            batch_means_scaled = []
            batch_stds_scaled = []
            
            for i in range(num_segment):
                sec_samples = samples[f"obs_section_{i}"].squeeze(-1)
                batch_means_scaled.append(sec_samples.mean(dim=0))
                batch_stds_scaled.append(sec_samples.std(dim=0))
                
            batch_means_scaled = torch.stack(batch_means_scaled, dim=1).cpu().numpy()
            batch_stds_scaled = torch.stack(batch_stds_scaled, dim=1).cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()
            
            pred_real = loaded_scaler.inverse_transform(batch_means_scaled)
            actual_real = loaded_scaler.inverse_transform(y_batch_np)
            std_real = batch_stds_scaled * loaded_scaler.scale_
            
            all_pred_real.append(pred_real)
            all_actual_real.append(actual_real)
            all_std_real.append(std_real)

    pred_real = np.vstack(all_pred_real)
    actual_real = np.vstack(all_actual_real)
    std_real = np.vstack(all_std_real)

    # 4. FAST VECTORIZED METRICS CALCULATION
    total_pred = pred_real.sum(axis=1)
    total_act = actual_real.sum(axis=1)
    total_std = np.sqrt(np.sum(std_real**2, axis=1))

    total_samples = len(total_act)
    within_bound_count = 0
    number_of_ratio = 0
    section_within_bound_counts = 0 
    error_abs_total = 0
    error_rate_squared = 0
    error_total = 0
    
    list_of_predict = []
    list_of_confidence = []
    list_of_actual = []
    list_of_predict_sections = [[] for i in range(num_segment)]
    list_of_confidence_sections = [[] for i in range(num_segment)]
    list_of_actual_sections = [[] for i in range(num_segment)]

    print_limit = len(x_global_val) 
    all_loc = 0
    all_std = 0
    overload = 0
    
    for j in range(print_limit):
        if total_pred[j] > 2000 or total_std[j] > 500:
            overload += 1
        
        print(f"\n--- Sample {j} ---")
        for i in range(num_segment):
            is_in_bound = "YES" if (pred_real[j, i] - std_real[j, i]) <= actual_real[j, i] <= (pred_real[j, i] + std_real[j, i]) else "NO"
            print(f"  Sec {i}: Pred {pred_real[j, i]:.1f}s | Actual {actual_real[j, i]:.1f}s | Conf +/- {std_real[j, i]:.1f}s | Within Bound? {is_in_bound}")
            
            if is_in_bound == "YES":
                section_within_bound_counts += 1

        print(f"\nTotal ETA: {total_pred[j]:.2f} seconds (Actual: {total_act[j]:.2f})")
        
        in_bound = (total_pred[j] - total_std[j]) <= total_act[j] <= (total_pred[j] + total_std[j])
        if in_bound:
            within_bound_count += 1
            
        print(f"Within Bound? : {'YES' if in_bound else 'NO'}")
        print(f"Confidence: +/- {total_std[j]:.2f} seconds")
        
        all_loc += total_pred[j]
        all_std += total_std[j]
        
        conf_level = total_pred[j] / total_std[j] if total_std[j] > 0 else 0
        print(f"Confidence Level: {conf_level:.2f}")
    
        # Statistics math
        list_of_predict.append(total_pred[j])
        list_of_confidence.append(total_std[j])
        list_of_actual.append(total_act[j])
        
        prediction_std_dev = statistics.pstdev(list_of_predict) if len(list_of_predict) > 1 else 0.0
        confidence_std_dev = statistics.pstdev(list_of_confidence) if len(list_of_confidence) > 1 else 0.0
        actual_std_dev     = statistics.pstdev(list_of_actual) if len(list_of_actual) > 1 else 0.0
        
        if total_std[j] > 0:
            number_of_ratio += total_pred[j]/total_std[j]
            
        error_total += (total_act[j] - total_pred[j])
        error_rate = error_total/(j+1) 
        error_abs_total += abs(total_act[j] - total_pred[j])
        error_rate_squared = error_abs_total/(j+1) 

        print(f"\nPrediction Std Deviation: {prediction_std_dev:.2f} , Confidence Std Deviation: {confidence_std_dev:.2f} , Actual Std Deviation: {actual_std_dev:.2f}")
        print(f"Error (MAE running): {error_rate_squared:.2f}")
        print(f"Error Tendency (Bias running): {error_rate:.2f}")
        print(f"\n")
    
    print("\n==============================================")
    print(f"總共 {total_samples} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
    print(f"平均 {num_segment} Section，有 {section_within_bound_counts / (total_samples * num_segment):.2f} section 落在預測區間內。")
    print(f"平均置信度指標: {number_of_ratio / total_samples:.2f}")
    print(f"平均預測時間: {all_loc / print_limit:.2f}")
    print(f"平均置信度: {all_std / print_limit:.2f}")
    print("==============================================")
    print(f"最大預測時間: {max_pred:.2f}")
    print(f"最大置信度: {max_conf:.2f}")
    print(f"超限樣本數 (Overload): {overload}")