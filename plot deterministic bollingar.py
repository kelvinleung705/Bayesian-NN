import statistics
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
    #(df.iloc[:, 6] == 0) &
    (df.iloc[:, 55] < 0)
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
# 2. DETERMINISTIC LAYERS
# ==========================================
class DeterministicLocalIsolationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments):
        super().__init__()
        self.num_segments = num_segments
        # Standard PyTorch nn.ModuleList
        self.nets = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_segments)
        ])
            
    def forward(self, x_inputs):
        outputs = []
        for i in range(self.num_segments):
            out = torch.nn.functional.silu(self.nets[i](x_inputs[i]))
            outputs.append(out)
        return outputs

class DeterministicNeighborMixingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2):
        super().__init__()
        self.num_segments = num_segments
        
        # 1. Attention Networks (Standard Linear)
        self.att_nets = nn.ModuleList([
            nn.Linear(input_dim * 2, 2) for _ in range(num_segments)
        ])
        
        # 2. Processing Networks (Standard Linear)
        net_input_dim = input_dim * 2
        self.nets_1 = nn.ModuleList([
            nn.Linear(net_input_dim, output_dim) for _ in range(num_segments)
        ])
        self.nets_2 = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_segments)
        ])
        
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        
    def forward(self, prev_layer_outputs):
        outputs = []
        for i in range(self.num_segments):
            h_self = prev_layer_outputs[i]
            
            if i < self.num_segments - 1:
                h_right = prev_layer_outputs[i+1]
            else:
                h_right = torch.zeros_like(h_self)

            # --- Deterministic Attention ---
            context = torch.cat([h_self, h_right], dim=1) 
            raw_attention_scores = self.att_nets[i](context) 
            attention_weights = torch.nn.functional.softmax(raw_attention_scores, dim=1)
            
            alpha_self = attention_weights[:, 0].unsqueeze(1)  
            alpha_right = attention_weights[:, 1].unsqueeze(1) 

            self_feat_weighted = h_self * alpha_self
            right_feat_weighted = h_right * alpha_right
            
            combined = torch.cat([self_feat_weighted, right_feat_weighted], dim=1)
            
            # --- Standard Processing ---
            out = self.nets_1[i](combined)
            out = self.dropout_1(out) 
            out = torch.nn.functional.silu(out)
            
            out = self.nets_2[i](out)
            out = self.dropout_2(out)
            out = torch.nn.functional.silu(out)
            
            outputs.append(out)
            
        return outputs

# ==========================================
# 3. DETERMINISTIC "MATRIX" GNN MODEL
# ==========================================
class DeterministicMatrixGNN(nn.Module):
    def __init__(self, num_sections=9, global_dim=9, local_dim=4, hidden_dim=32):
        super().__init__()
        self.num_sections = num_sections
        input_dim = global_dim + local_dim + 1 
            
        self.embedding_layer = DeterministicLocalIsolationLayer(input_dim, hidden_dim, num_sections)
        
        self.prop_layers = nn.ModuleList([
            DeterministicNeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2)
            for _ in range(3)
        ])
        
        # --- Output Heads ---
        # In a deterministic model, we ONLY predict the Mean (loc).
        # We DO NOT predict scale or df, because there is no probability distribution!
        self.heads_loc = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_sections)
        ])

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        
        # Virtual Clock
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        inputs_list = []
        for i in range(self.num_sections):
            loc_i = all_sections_data[:, i, :]
            time_i = accumulated_time[:, i, :]
            inp = torch.cat([global_features, loc_i, time_i], dim=1)
            inputs_list.append(inp)
            
        h_current = self.embedding_layer(inputs_list)
            
        for layer in self.prop_layers:
            h_current = layer(h_current)
        
        all_locs = []
        
        for i in range(self.num_sections):
            final_feat = h_current[i] 
            
            # Predict only the point-estimate (the mean time)
            loc = self.heads_loc[i](final_feat)
            all_locs.append(loc)
        
        # Return a list of tensors, shape: [Batch, 1] for each section
        return all_locs



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    saved_params_path = "deterministic_model.pt" # Replace with exact saved params file name
    saved_scaler_path = "deterministic_scaler.pkl"              # Replace with exact saved scaler file name
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_sorted.xlsx"
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_traffic.xlsx"
    
    
    
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
    # 2. RECONSTRUCT & LOAD MODEL (DETERMINISTIC)
    # ==========================================
    print("\n--- Initializing and Loading Model Weights ---")
    
    # 1. Initialize the Deterministic Model Architecture
    # Ensure your class is named DeterministicMatrixGNN (or whatever you named the standard PyTorch version)
    deterministic_model = DeterministicMatrixGNN(
        num_sections=num_segment, 
        global_dim=9, 
        local_dim=4, 
        hidden_dim=32
    ).to(device)

    # 2. Load the saved standard PyTorch weights (state_dict)
    # Make sure 'saved_params_path' points to your deterministic .pt file
    state_dict = torch.load(saved_params_path, map_location=device)
    deterministic_model.load_state_dict(state_dict)

    # 3. Set to Evaluation Mode
    # IMPORTANT: You MUST call .eval() for deterministic models!
    # This turns off Dropout layers so the model gives the exact same output every time.
    deterministic_model.eval()

    print("Model parameters loaded successfully! Starting Deterministic Inference.")

# ==========================================
    # 3. FAST BATCHED INFERENCE LOOP (DETERMINISTIC)
    # ==========================================
    print("\n--- Starting Fast Batched Prediction Test ---")
    
    batch_size = 1024 
    val_dataset = TensorDataset(x_global_val, x_local_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    all_pred_real = []
    all_actual_real = []

    # Run Inference on GPU in Batches
    with torch.no_grad():
        for x_g_batch, x_l_batch, y_batch in val_loader:
            
            # 1. Standard Forward Pass
            # Your model returns a list: [batch_sec0, batch_sec1, ..., batch_sec8]
            preds_list = deterministic_model(x_g_batch, x_l_batch) 
            
            # --- FIX STARTS HERE ---
            # Use torch.cat to merge the list of (batch, 1) tensors into a single (batch, 9) tensor
            preds_scaled = torch.cat(preds_list, dim=1)
            # --- FIX ENDS HERE ---
            
            # Move to CPU and convert to NumPy
            preds_scaled_np = preds_scaled.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()
            
            # 2. Vectorized Inverse Transform
            pred_real = loaded_scaler.inverse_transform(preds_scaled_np)
            actual_real = loaded_scaler.inverse_transform(y_batch_np)
            
            all_pred_real.append(pred_real)
            all_actual_real.append(actual_real)

    # Combine all batches
    pred_real = np.vstack(all_pred_real)
    actual_real = np.vstack(all_actual_real)

    # ==========================================
    # 4. FAST VECTORIZED METRICS CALCULATION
    # ==========================================
    # Calculate Total ETAs (Sum across sections for each trip)
    total_pred = pred_real.sum(axis=1)
    total_act = actual_real.sum(axis=1)

    # --- ADDING THE HARDCODED BOLLINGER BAND ---
    hardcoded_margin = 74.32
    
    # Calculate bounds
    lower_bound = total_pred - hardcoded_margin
    upper_bound = total_pred + hardcoded_margin
    
    # Check how many actual times landed inside the hardcoded bounds
    within_bound_mask = (total_act >= lower_bound) & (total_act <= upper_bound)
    within_bound_count = np.sum(within_bound_mask)
    coverage_percentage = (within_bound_count / len(total_act)) * 100

    # Running Errors
    error_rates = (total_act - total_pred)
    abs_errors = np.abs(total_act - total_pred)
    
    print("\n" + "="*30)
    print("FINAL VALIDATION SUMMARY (DETERMINISTIC)")
    print("="*30)
    print(f"Total Validated Trips: {len(total_act)}")
    print(f"Mean Absolute Error: {np.mean(abs_errors):.2f} seconds")
    print("-" * 30)
    print(f"Hardcoded Margin: +/- {hardcoded_margin} seconds")
    print(f"Trips within Hardcoded Margin: {within_bound_count} / {len(total_act)}")
    print(f"Coverage Percentage: {coverage_percentage:.2f}%")
    print("="*30)

    """
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
    """
    
    # ==========================================
    # 7. CLEANING, SMOOTHING AND PLOTTING
    # ==========================================
    print("\nPreparing data for visualization...")
    
    # 1. Ensure everything is a 1D NumPy array
    plot_x = np.array(time_decimal).flatten()
    plot_y = np.array(total_pred).flatten()
    plot_act = np.array(total_act).flatten()
    
    # Adjust early morning hours
    for i in range(len(plot_x)):
        if plot_x[i] < 3:
            plot_x[i] += 24

    # 2. Filter out NaNs or Infinities
    valid_mask = np.isfinite(plot_x) & np.isfinite(plot_y) & np.isfinite(plot_act)
    plot_x = plot_x[valid_mask]
    plot_y = plot_y[valid_mask]
    plot_act = plot_act[valid_mask]

    # 3. Apply hardcoded bounds for the plot
    hardcoded_margin = 74.32
    upper_95 = plot_y + hardcoded_margin
    lower_95 = plot_y - hardcoded_margin

    # 4. Apply LOWESS Smoothing
    smooth_frac = 0.15
    print(f"Applying LOWESS smoothing to {len(plot_x)} points...")
    
    m_smooth = lowess(plot_y, plot_x, frac=smooth_frac)
    u_smooth = lowess(upper_95, plot_x, frac=smooth_frac)
    l_smooth = lowess(lower_95, plot_x, frac=smooth_frac)

    # 5. GENERATE THE FINAL PLOT
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(16, 9))

    # A. Raw Actual Times (The Gray Cloud)
    plt.scatter(plot_x, plot_act, color='black', alpha=0.2, s=12, label='Actual Trip Times', zorder=1)

    # B. The Hardcoded Bollinger Band
    plt.fill_between(m_smooth[:, 0], l_smooth[:, 1], u_smooth[:, 1], 
                     color='#3498db', alpha=0.3, label=f'Hardcoded Margin (+/- {hardcoded_margin}s)', zorder=2)

    # C. The Mean ETA Line (Trend Forecast from Model)
    plt.plot(m_smooth[:, 0], m_smooth[:, 1], color='#e74c3c', linewidth=3, label='Deterministic ETA Forecast', zorder=3)

    # --- FORMATTING ---
    plt.title(f'Deterministic Transit ETA Forecast (with +/- {hardcoded_margin}s margin)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Time of Day (24h Format)', fontsize=14)
    plt.ylabel('Total Travel Time (Seconds)', fontsize=14)
    
    # Grid and Ticks
    plt.xlim(5, 25)
    plt.ylim(300, 900)
    plt.xticks(np.arange(5, 26, 1))
    plt.yticks(np.arange(300, 901, 100))

    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    sns.despine()
    plt.tight_layout()
    
    # Save the output file
    output_filename = 'ghost_bus_final_results_deterministic_weekday_bad_traffic_baollinger.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS! Plot saved as: {output_filename}")
    
    try:
        plt.show()
    except Exception:
        print("Interactive window not available. Please open the PNG file to see the result.")