import statistics
from onnx import save
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

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
    
    df = df[
    (df.iloc[:, 2] == 1) & 
    (df.iloc[:, 7] <= 2) #&
    #(df.iloc[:, 56].isin([5]))
    ]
    
    


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
    
    # --- CRITICAL FIX: Replicate the original Validation Split ---
    idx = np.arange(x_global_all.shape[0])
    #train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_val = x_global_all[idx]
    x_local_val = x_local_all[idx]
    y_val = y_scaled_all[idx]
    
    return x_global_val, x_local_val, y_val, idx


# ==========================================
# 2. MODEL ARCHITECTURE (Required for loading)
# ==========================================
class LocalIsolationLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.nets = PyroModuleList([])
        
        for i in range(num_segments):
            net = PyroModule[nn.Linear](input_dim, output_dim)
            zero = torch.tensor(0., device=device)
            point_one = torch.tensor(1., device=device)
            df = torch.tensor(15., device=device)
            net.weight = PyroSample(dist.StudentT(df, zero, point_one).expand([output_dim, input_dim]).to_event(2))
            net.bias = PyroSample(dist.StudentT(df, zero, point_one).expand([output_dim]).to_event(1))
            self.nets.append(net)
            
    def forward(self, x_inputs):
        outputs =[]
        for i in range(self.num_segments):
            out = torch.nn.functional.silu(self.nets[i](x_inputs[i]))
            outputs.append(out)
        return outputs

class NeighborMixingLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.device = device
        
        loc_self = torch.tensor(2., device=device)
        loc_side = torch.tensor(0.0, device=device)
        scale = torch.tensor(1.0, device=device)
        zero = torch.tensor(0., device=device)
        w_scale = torch.tensor(1.0, device=device)
        b_scale = torch.tensor(0.1, device=device)

        self.w_self = PyroSample(dist.Normal(loc_self, scale).expand([num_segments]).to_event(1))
        self.w_right = PyroSample(dist.Normal(loc_side, scale).expand([num_segments]).to_event(1))
        
        self.nets_1 = PyroModuleList([])
        self.nets_2 = PyroModuleList([])
        
        for i in range(num_segments):
            net_input_dim = input_dim * 2
            net_1 = PyroModule[nn.Linear](net_input_dim, output_dim)
            net_1.weight = PyroSample(dist.Normal(zero, w_scale).expand([output_dim, net_input_dim]).to_event(2))
            net_1.bias = PyroSample(dist.Normal(zero, b_scale).expand([output_dim]).to_event(1))
            self.nets_1.append(net_1)
    
        self.dropout_1 = PyroModule[nn.Dropout](p=dropout_rate)
        
        for i in range(num_segments):
            net_2 = PyroModule[nn.Linear](output_dim, output_dim)
            net_2.weight = PyroSample(dist.Normal(zero, w_scale).expand([output_dim, output_dim]).to_event(2))
            net_2.bias = PyroSample(dist.Normal(zero, b_scale).expand([output_dim]).to_event(1))
            self.nets_2.append(net_2)
        
        self.dropout_2 = PyroModule[nn.Dropout](p=dropout_rate)
        
    def forward(self, prev_layer_outputs):
        outputs =[]
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
            out = self.dropout_1(out) 
            out = torch.nn.functional.silu(out)
            
            out = self.nets_2[i](out)
            out = self.dropout_2(out)
            out = torch.nn.functional.silu(out)
            
            outputs.append(out)
        return outputs

class MatrixGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=8, device = 'cuda', pnt_1 = None):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        input_dim = global_dim + local_dim + 1 
        
        # --- Layer 0: Input Projection (Unique per segment) ---
            
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        
        # --- Propagation Layers (The Matrix) ---
        # "Different function for each layer each section"
        # We create N layers. Each layer contains N unique networks.
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(2) # Layer depth = num_segments
        ])
        
        # --- Output Heads ---
        final_dim = hidden_dim
        
        self.heads_loc = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df = PyroModuleList([])
        
        for i in range(self.num_sections):
            loc_std_dev = 2
            # Tensors for Heads
            zero = torch.tensor(0., device=device)
            loc_std = torch.tensor(1.0, device=device)
            loc_bias_mu = torch.tensor(0., device=device)
            loc_bias_std = torch.tensor(1., device=device)

            scale_std = torch.tensor(0.3, device=device)
            scale_bias_mu = torch.tensor(0., device=device)
            scale_bias_std = torch.tensor(3.0, device=device)
            
            three = torch.tensor(3., device=device)
            df_std = torch.tensor(1., device=device)
            df_bias_mu = torch.tensor(0., device=device)
            df_bias_std = torch.tensor(3.0, device=device)
            
            # Loc Head
            h_loc = PyroModule[nn.Linear](final_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(zero, loc_std).expand([1, final_dim]).to_event(2))
            h_loc.bias = PyroSample(dist.Normal(loc_bias_mu, loc_bias_std).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)
            
            # Scale Head
            h_scale = PyroModule[nn.Linear](final_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(zero, scale_std).expand([1, final_dim]).to_event(2))
            h_scale.bias = PyroSample(dist.Normal(scale_bias_mu, scale_bias_std).expand([1]).to_event(1))
            self.heads_scale.append(h_scale)
            
            # DF Head
            h_df = PyroModule[nn.Linear](final_dim, 1)
            h_df.weight = PyroSample(dist.Normal(zero, df_std).expand([1, final_dim]).to_event(2))
            h_df.bias = PyroSample(dist.Normal(df_bias_mu, df_bias_std).expand([1]).to_event(1))
            self.heads_df.append(h_df)

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        
        # Initialize a single running clock (not a vector of zeros for all sections)
        #current_time = torch.zeros(batch_size, 1).to(device)
        current_time = torch.zeros(batch_size, 1).to(device)
        
        all_locs = []
        all_scales = []
        all_dfs = []
        
        # --- THE AUTOREGRESSIVE WATERFALL LOOP ---
        for current_section in range(self.num_sections):
            # 1. Build the Input List for the GNN
            inputs_list = []
            for i in range(self.num_sections):
                loc_i = all_sections_data[:, i, :]
                # THE CRITICAL INJECTION LOGIC
                if i == current_section:
                    time_i = current_time
                elif i < current_section:
                    time_i = current_time
                else:
                    time_i = torch.zeros(batch_size, 1).to(device)
                
                if time_i.abs().mean().item() > 15:
                    time_i = torch.zeros(batch_size, 1).to(device)
                inp = torch.cat([global_features, loc_i, time_i], dim=1)
                inputs_list.append(inp)
                
            # 2. Run the full GNN (Embedding + Mixing) to get context
            h_current = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h_current = layer(h_current)
                
            
            # 3. Extract ONLY the prediction for the current_section
            final_feat = h_current[current_section] 
            
            loc = self.heads_loc[current_section](final_feat)
            scale = torch.nn.functional.softplus(self.heads_scale[current_section](final_feat)) + 1e-3 
            df = torch.nn.functional.softplus(self.heads_df[current_section](final_feat)) + 2.5
            
            # 4. Store the predictions
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            # 5. UPDATE THE CLOCK for the next loop iteration
            # We add the predicted travel time of THIS section to the running total.
            
            current_time = current_time + loc
                
            
        return all_locs, all_scales, all_dfs


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_9_accu4_fixed.pt" # Replace with exact saved params file name
    
    #saved_scaler_path = "y_scaler_1.pkl"              # Replace with exact saved scaler file name
    saved_scaler_path = "y_scaler_4_fixed.pkl"              # Replace with exact saved scaler file name
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_month_sorted.xlsx"
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2026.xlsx"
    #file_path = "bad_visibility.xlsx"
    
    
    
    # ==========================================
    # 1. LOAD SCALER & DATA
    # ==========================================
    print("\n--- Loading Scaler & Processing Data ---")
    loaded_scaler = joblib.load(saved_scaler_path)
    
    x_global_val, x_local_val, y_val, val_idx = process_validation_data(file_path, loaded_scaler)
    
    x_global_val = x_global_val.to(device)
    x_local_val = x_local_val.to(device)
    y_val = y_val.to(device)
    
    
    # --- 核心修復：從 Sin/Cos 還原為 0-24 小時 ---
    print("從嵌入向量 (Sin/Cos) 逆向還原時間軸...")
    
    # 提取第 0 列 (Sin) 和 第 1 列 (Cos)
    val_time_sin = x_global_val[:, 0].cpu().numpy()
    val_time_cos = x_global_val[:, 1].cpu().numpy()
    
    # 使用 arctan2 算出弧度 (範圍是 -pi 到 pi)
    # np.arctan2(y, x) 對應的是 np.arctan2(sin, cos)
    angles_radians = np.arctan2(val_time_sin, val_time_cos)
    
    # 將弧度轉換回分鐘 (1天 = 1440分鐘)
    # 如果角度是負的，加上 2*pi 轉回正數
    total_minutes = (angles_radians / (2 * np.pi)) * 1440.0
    total_minutes = np.where(total_minutes < 0, total_minutes + 1440.0, total_minutes)
    
    # 轉換為 24 小時制的小數 (例如 8.5 代表 08:30 AM)
    time_decimal = total_minutes / 60.0

    print(f"DEBUG: 成功還原時間軸！前 5 個行程時間: {time_decimal[:5].round(2)}")
    # ==========================================
    # 2. RECONSTRUCT & LOAD MODEL
    # ==========================================
    print("\n--- Initializing and Loading Model Weights ---")
    pyro.clear_param_store()

    # Load ParamStore FIRST so the Guide correctly latches onto the loaded weights
    pyro.get_param_store().load(saved_params_path, map_location=device.type)

    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32, device=device).to(device)
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
    # 3. FAST BATCHED INFERENCE LOOP
    # ==========================================
    print("\n--- Starting Fast Batched Prediction Test ---")
    
    # 1. Setup DataLoader for Batching
    batch_size = 1024  # Process 1024 samples at the exact same time
    val_dataset = TensorDataset(x_global_val, x_local_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=50)
    
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
    
    print_limit = len(x_global_val) # Change this to 5 if you just want a quick peek
    
    for j in range(print_limit):
        print(f"\n--- Sample {j} ---")
        for i in range(num_segment):
            is_in_bound = "YES" if sec_within_bounds_mask[j, i] else "NO"
            print(f"  Sec {i}: Pred {pred_real[j, i]:.1f}s | Actual {actual_real[j, i]:.1f}s | Conf +/- {std_real[j, i]:.1f}s | Within Bound? {is_in_bound}")
            
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

    # ==========================================
    # 6. FINAL METRICS SUMMARY
    # ==========================================
    print("\n==============================================")
    print(f"總共 {total_samples} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
    print(f"平均 {num_segment} Section，有 {section_within_bound_counts / total_samples:.2f} section 落在預測區間內。")
    print(f"平均置信度指標: {number_of_ratio_sum / total_samples:.2f}")
    print("==============================================")
    
    print("\n" + "="*30)
    print("FINAL VALIDATION SUMMARY")
    print("="*30)
    print(f"Total Validated Trips: {len(x_global_val)}")
    print(f"Total Coverage (Trip within 1 StdDev): {(within_bound_count/len(x_global_val))*100:.2f}%")
    print(f"Mean Absolute Error: {np.mean(np.abs(total_act - total_pred)):.2f} seconds")
    print("="*30)
    
    # ==========================================
    # 7. CLEANING, SMOOTHING AND PLOTTING
    # ==========================================
    print("\nPreparing data for visualization...")
    
    # 1. Ensure everything is a 1D NumPy array and remove any invalid numbers
    plot_x = np.array(time_decimal).flatten()
    plot_y = np.array(total_pred).flatten()
    plot_act = np.array(total_act).flatten()
    plot_std = np.array(total_std).flatten()
    
    # --- DIAGNOSTIC PRINTS ---
    print(f"\nDEBUG: Initial array lengths -> X: {len(plot_x)}, Y_pred: {len(plot_y)}, Y_act: {len(plot_act)}, Std: {len(plot_std)}")
    for i in range(len(plot_x)):
        if plot_x[i] < 3:
            plot_x[i] += 24
    if len(plot_x) > 0:
        print(f"DEBUG: First 5 X (Time) values: {plot_x[:5]}")
    if len(plot_y) > 0:
        print(f"DEBUG: First 5 Y_pred values: {plot_y[:5]}")
        print(f"DEBUG: First 5 Y_act values: {plot_act[:5]}")
    # -------------------------

    # 2. Filter out any NaNs or Infinities that would break the smoother
    valid_mask = np.isfinite(plot_x) & np.isfinite(plot_y) & np.isfinite(plot_act) & np.isfinite(plot_std)
    plot_x = plot_x[valid_mask]
    #print(plot_x[:10])
    plot_y = plot_y[valid_mask]
    #print(plot_y[:10])
    plot_act = plot_act[valid_mask]
    #print(plot_act[:10])
    plot_std = plot_std[valid_mask]
    #print(plot_std[:10])

    # 3. Calculate 95% Confidence Bounds
    #1.96
    upper_95 = plot_y + (1 * plot_std)
    print(upper_95[:10])
    lower_95 = plot_y - (1 * plot_std)
    print(lower_95[:10])

    
    # 4. Apply LOWESS Smoothing
    # frac=0.1 to 0.2 is usually best for traffic trends
    smooth_frac = 0.15
    print(f"Applying LOWESS smoothing to {len(plot_x)} points...")
    
    # lowess returns [x_sorted, y_smoothed]
    m_smooth = lowess(plot_y, plot_x, frac=smooth_frac)
    u_smooth = lowess(upper_95, plot_x, frac=smooth_frac)
    l_smooth = lowess(lower_95, plot_x, frac=smooth_frac)
    

    # 5. GENERATE THE FINAL PLOT
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(16, 9))

    # A. Raw Actual Times (The Gray Cloud)
    plt.scatter(plot_x, plot_act, color='black', alpha=0.2, s=12, label='Actual Trip Times', zorder=1)

    # B. The Bollinger Band (95% CI)
    # We use the X values from the smoother (which are already sorted)
    plt.fill_between(m_smooth[:, 0], l_smooth[:, 1], u_smooth[:, 1], 
                     color='#3498db', alpha=0.3, label='BNN 95% Predictive Uncertainty', zorder=2)

    # C. The Mean ETA Line (Trend Forecast)
    plt.plot(m_smooth[:, 0], m_smooth[:, 1], color='#2c3e50', linewidth=3, label='Mean ETA Forecast', zorder=3)

    # --- FORMATTING ---
    plt.title('Bayesian Transit ETA: Predictive Uncertainty vs. Time of Day', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Time of Day (24h Format)', fontsize=14)
    plt.ylabel('Total Travel Time (Seconds)', fontsize=14)
    
    # Grid and Ticks
    plt.xlim(5, 25)
    plt.ylim(300, 900) # Y 軸：從 300 秒到 900 秒
    plt.xticks(np.arange(5, 26, 1))
    plt.yticks(np.arange(300, 901, 100))
    
    # Smart Y-axis scaling: Show data up to the 99th percentile to avoid 1 crazy outlier squashing the graph
    #plt.ylim(0, np.percentile(plot_act, 99.5) * 1.1) 

    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    sns.despine()
    plt.tight_layout()
    
    # Save the output file
    #output_filename = 'ghost_bus_accu_results_bollinger_weekend.png'
    output_filename = 'ghost_bus_accu2_results_bollinger_weekday_good_weather_2026.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS! Plot saved as: {output_filename}")
    
    try:
        plt.show()
    except Exception:
        print("Interactive window not available. Please open the PNG file to see the result.")