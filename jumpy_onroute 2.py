import os
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

num_segment = 9

# ==========================================
# 1. PASTE YOUR MODEL CLASSES HERE
# ==========================================
# (Paste LocalIsolationLayer, NeighborMixingLayer, MatrixGNN, and model_fn exactly as they are in your working script)
# [YOUR CLASSES GO HERE - I am omitting them for brevity, but you MUST paste them!]


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
    (df.iloc[:, 2] == 0) & 
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
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=8, device='cuda'):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        input_dim = global_dim + local_dim + 1 
            
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(3)
        ])
        
        final_dim = hidden_dim
        self.heads_loc = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df = PyroModuleList([])
        
        for i in range(self.num_sections):
            zero = torch.tensor(0., device=device)
            loc_std = torch.tensor(1.0, device=device)
            loc_bias_mu = torch.tensor(0., device=device)
            loc_bias_std = torch.tensor(1., device=device)

            scale_std = torch.tensor(0.3, device=device)
            scale_bias_mu = torch.tensor(0., device=device)
            scale_bias_std = torch.tensor(3.0, device=device)

            df_std = torch.tensor(1., device=device)
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
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        inputs_list =[]
        for i in range(self.num_sections):
            loc_i = all_sections_data[:, i, :]
            time_i = accumulated_time[:, i, :]
            inp = torch.cat([global_features, loc_i, time_i], dim=1)
            inputs_list.append(inp)
            
        h_current = self.embedding_layer(inputs_list)
            
        for layer in self.prop_layers:
            h_current = layer(h_current)
        
        all_locs, all_scales, all_dfs = [], [],[]
        
        for i in range(self.num_sections):
            final_feat = h_current[i] 
            loc = self.heads_loc[i](final_feat)
            scale = torch.nn.functional.softplus(self.heads_scale[i](final_feat)) + 1e-3 
            df = torch.nn.functional.softplus(self.heads_df[i](final_feat)) + 2.5
            
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
        
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


# ==========================================
# 2. DATA LOADING FOR REAL-TIME SIMULATION
# ==========================================
def load_realtime_trip_data(file_path):
    print(f"Reading Real-Time Trip Data from {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)
    
    if len(df) != num_segment:
        print(f"WARNING: Expected {num_segment} rows, but found {len(df)}.")

    end_col = 19 + (num_segment * 4)
    
    # Load and clean the raw numbers from Excel
    df_subset = df.iloc[:, 0:end_col].apply(pd.to_numeric, errors='coerce').fillna(0)
    raw_data_np = df_subset.values.astype(np.float32)
    
    # 1. Extract Pre-Standardized Global Features
    x_global = torch.tensor(raw_data_np[:, 0:9], dtype=torch.float32)
    
    # 2. Extract The "Time Already Spent" (Column 18) - This is RAW seconds
    time_spent_array = raw_data_np[:, 18]
    
    # 3. Extract Pre-Standardized Local Features (Starting at Col 19)
    raw_local = raw_data_np[:, 19 : 19 + (num_segment * 4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)
    
    # --- FIX: WE REMOVED THE LOG1P TRANSFORMS ---
    # Because x_local is ALREADY standardized in the Excel file, 
    # we do NOT modify it here. We feed it straight to the model.
    
    # 4. Extract the Ground Truth Total Time (RAW seconds)
    y_raw = raw_data_np[:, 9:9+num_segment]
    true_total_time = np.sum(y_raw[0, :]) 

    return x_global, x_local, time_spent_array, true_total_time

# ==========================================
# 3. MAIN EXECUTION & PLOTTING
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- CONFIGURATION ---
    # Put your specific 9-row Excel file here
    REALTIME_EXCEL_FILE = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy.xlsx" 
    MODEL_FILE = "ghost_bus_model_cycle_0.1_2000_df10_KL_Sample.pt"
    SCALER_FILE = "y_scaler.pkl"
    
    
    
    # 1. LOAD DATA & SCALER
    loaded_scaler = joblib.load(SCALER_FILE)
    x_global, x_local, time_spent_array, true_total_time = load_realtime_trip_data(REALTIME_EXCEL_FILE)
    
    x_global = x_global.to(device)
    x_local = x_local.to(device)

    # 2. LOAD MODEL WEIGHTS
    print("\n--- Loading Bayesian Model ---")
    pyro.clear_param_store()
    global bnn_model
    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32, device=device).to(device)
    base_guide = AutoDiagonalNormal(model_fn).to(device)
    
    # Dummy trace to initialize the exact parameter shapes
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        base_guide(x_global[0:1], x_local[0:1], y_true=dummy_y)

    # Load the parameters
    pyro.get_param_store().load(MODEL_FILE, map_location=device.type)
    print("Model loaded successfully!")

    # 3. RUN REAL-TIME INFERENCE SIMULATION
    print("\n--- Simulating Real-Time Journey ---")
    predictive = Predictive(model_fn, guide=base_guide, num_samples=100)
    
    plot_etas_mean = np.zeros(num_segment)
    plot_etas_std = np.zeros(num_segment)

    with torch.no_grad():
        samples = predictive(x_global, x_local)
        
        for current_loc in range(num_segment):
            
            time_already_spent = time_spent_array[current_loc]
            
            # --- THE FIX: We collect the raw scaled samples first ---
            # Shape will be [100 samples, number_of_future_segments]
            num_future_secs = num_segment - current_loc
            future_samples_scaled = np.zeros((100, num_future_secs))
            
            for idx, future_sec in enumerate(range(current_loc, num_segment)):
                # Grab the 100 samples for this specific section, at this specific moment
                future_samples_scaled[:, idx] = samples[f"obs_section_{future_sec}"][:, current_loc].squeeze().cpu().numpy()

            # --- THE FIX: Un-scale EVERYTHING before doing math ---
            # We need to un-scale a matrix of shape [100, num_future_secs].
            # Because your scaler expects 9 columns, we pad it with zeros, un-scale, and slice it back.
            dummy_matrix = np.zeros((100, num_segment))
            dummy_matrix[:, current_loc:] = future_samples_scaled
            
            # Now every sample in the matrix is in REAL SECONDS
            future_samples_seconds = loaded_scaler.inverse_transform(dummy_matrix)[:, current_loc:]
            
            # --- NOW we can safely sum the seconds for each of the 100 simulations ---
            # Shape becomes [100] (The total remaining time for each simulation)
            total_remaining_seconds_samples = np.sum(future_samples_seconds, axis=1)

            # Calculate the Mean and StdDev of those 100 simulations
            mean_remaining_sec = total_remaining_seconds_samples.mean()
            std_remaining_sec = total_remaining_seconds_samples.std()

            # Final ETA is the exact time spent + the predicted remaining seconds
            total_eta_mean = time_already_spent + mean_remaining_sec
            total_eta_std = std_remaining_sec 

            plot_etas_mean[current_loc] = total_eta_mean
            plot_etas_std[current_loc] = total_eta_std
            
            print(f"At Start of Sec {current_loc}: Spent {time_already_spent:5.1f}s | Est. Remaining {mean_remaining_sec:5.1f}s | Total ETA {total_eta_mean:5.1f}s ± {total_eta_std*1.96:4.1f}s")
    # ==========================================
    # 4. GENERATE THE "CONE OF UNCERTAINTY" PLOT
    # ==========================================
    print("\nGenerating Real-Time Convergence Plot...")
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(14, 8))

    x_axis = np.arange(num_segment)

    # Calculate Confidence Bounds (1 Standard Deviation = 68% Confidence)
    # You can change 1.0 to 1.96 if you want 95% Confidence
    upper_bound = plot_etas_mean + (1.0 * plot_etas_std)
    lower_bound = plot_etas_mean - (1.0 * plot_etas_std)

    # 1. The True Final Travel Time (Horizontal Red Dashed Line)
    plt.axhline(y=true_total_time, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Actual Total Trip Time ({true_total_time:.1f}s)', zorder=1)

    # 2. The Confidence Cone (Blue Shaded Area)
    plt.fill_between(x_axis, lower_bound, upper_bound, color='#3498db', alpha=0.3, label='Bayesian Predictive Uncertainty (± 1 StdDev)', zorder=2)

    # 3. The Dynamic Mean ETA Trajectory (Dark Blue Line)
    plt.plot(x_axis, plot_etas_mean, color='#2c3e50', linewidth=3, marker='o', markersize=8, label='Dynamic Mean ETA Forecast', zorder=3)

    # 4. The "Time Spent" Trajectory (Gray Line moving up from zero)
    plt.plot(x_axis, time_spent_array, color='gray', linestyle=':', linewidth=2, marker='x', label='Actual Time Spent So Far', zorder=4)

    # Formatting
    plt.title('Dynamic Real-Time ETA Convergence (The "Ghost Bus" Simulator)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Bus Location (Reached Start of Section)', fontsize=14)
    plt.ylabel('Total Trip Time (Seconds)', fontsize=14)
    
    plt.xlim(0, num_segment - 1)
    plt.xticks(x_axis, [f"Sec {i}" for i in x_axis], fontsize=12)
    
    # Smart Y-Axis limits
    y_min = min(lower_bound.min(), true_total_time, 0)
    y_max = max(upper_bound.max(), true_total_time) * 1.1
    plt.ylim(y_min, y_max)

    plt.legend(loc='upper right', fontsize=13, frameon=True, shadow=True)
    sns.despine()
    plt.tight_layout()
    
    output_filename = 'eta_convergence_cone.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS! Plot saved as: {output_filename}")
    plt.show()