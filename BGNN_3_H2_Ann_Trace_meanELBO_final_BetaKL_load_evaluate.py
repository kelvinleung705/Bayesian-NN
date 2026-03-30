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
import joblib
from sklearn.model_selection import train_test_split

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
    
    # --- CRITICAL FIX: Replicate the original Validation Split ---
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_val = x_global_all[val_idx]
    x_local_val = x_local_all[val_idx]
    y_val = y_scaled_all[val_idx]
    
    return x_global_val, x_local_val, y_val


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    saved_params_path = "ghost_bus_model_cycle_0.1_2000_df10_KL_Sample.pt" # Replace with exact saved params file name
    saved_scaler_path = "y_scaler.pkl"              # Replace with exact saved scaler file name
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_sorted.xlsx"
    
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
    # 3. INFERENCE LOOP
    # ==========================================
    print("\n--- Starting Final Prediction Test ---")
    
    
    list_of_predict =[]
    list_of_confidence = []
    list_of_actual =[]
    list_of_predict_sections = [[] for i in range(num_segment)]
    list_of_confidence_sections = [[] for i in range(num_segment)]
    list_of_actual_sections = [[] for i in range(num_segment)]
    
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=50)
    
    within_bound_count = 0
    number_of_ratio = 0
    section_within_bound_counts = 0 
    error_abs_total = 0
    error_rate_squared = 0
    error_total = 0
    
    for j in range(len(x_global_val)):
        val_x_g = x_global_val[j:j+1]
        val_x_l = x_local_val[j:j+1]
    
        samples = predictive(val_x_g, val_x_l)
        
        pred_means_scaled =[]
        pred_stds_scaled = []
        actuals_scaled =[]
        
        for i in range(num_segment):
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            mean_scaled = sec_samples.mean().item()
            std_scaled = sec_samples.std().item()
            
            pred_means_scaled.append(mean_scaled)
            pred_stds_scaled.append(std_scaled)
            actuals_scaled.append(y_val[j, i].item())
            
        pred_real = loaded_scaler.inverse_transform([pred_means_scaled])[0]
        actual_real = loaded_scaler.inverse_transform([actuals_scaled])[0]
        std_real = np.array(pred_stds_scaled) * loaded_scaler.scale_
        
        total_pred = 0
        trip_section_within_bound = 0
        
        print(f"\n--- Sample {j} ---")
        for i in range(num_segment):
            list_of_predict_sections[i].append(pred_real[i])
            if len(list_of_predict_sections[i]) > 1:
                section_prediction_std_deviation = statistics.pvariance(list_of_predict_sections[i]) ** 0.5
            else:
                section_prediction_std_deviation = 0.0
                
            list_of_confidence_sections[i].append(std_real[i])
            if len(list_of_confidence_sections[i]) > 1:
                section_confidence_std_deviation = statistics.pvariance(list_of_confidence_sections[i]) ** 0.5
            else:
                section_confidence_std_deviation = 0.0
                
            list_of_actual_sections[i].append(actual_real[i])
            if len(list_of_actual_sections[i]) > 1:
                section_actual_std_deviation = statistics.pvariance(list_of_actual_sections[i]) ** 0.5
            else:
                section_actual_std_deviation = 0.0
            
            print(f"  Sec {i}: Pred {pred_real[i]:.1f}s | Actual {actual_real[i]:.1f}s | Conf +/- {std_real[i]:.1f}s | Prediction Dev {section_prediction_std_deviation:.1f}s | Confidence Dev {section_confidence_std_deviation:.1f}s | Actual Dev {section_actual_std_deviation:.1f}s | Within Bound? {'YES' if (pred_real[i] - std_real[i]) <= actual_real[i] <= (pred_real[i] + std_real[i]) else 'NO'}")
            total_pred += pred_real[i]
            
            if actual_real[i] >= (pred_real[i] - std_real[i]) and actual_real[i] <= (pred_real[i] + std_real[i]):
                trip_section_within_bound += 1
                
        section_within_bound_counts += trip_section_within_bound
        
        final_mean = pred_real.sum()
        total_act = actual_real.sum()
        total_std = np.sqrt(np.sum(std_real**2)) 
        
        if total_act >= (total_pred - total_std) and total_act <= (total_pred + total_std):
            within_bound_count += 1
            
        list_of_predict.append(total_pred)
        if len(list_of_predict) > 1:
            prediction_std_deviation = statistics.pvariance(list_of_predict) ** 0.5
        else:
            prediction_std_deviation = 0.0
            
        list_of_confidence.append(total_std)
        if len(list_of_confidence) > 1:
            confidence_std_deviation = statistics.pvariance(list_of_confidence) ** 0.5
        else:
            confidence_std_deviation = 0.0
            
        list_of_actual.append(total_act)
        if len(list_of_actual) > 1:
            actual_std_deviation = statistics.pvariance(list_of_actual) ** 0.5
        else:
            actual_std_deviation = 0.0
        
        if total_std > 0:
            number_of_ratio += total_pred/total_std
            
        error_total += (total_act - total_pred)
        error_rate = error_total/(j+1) 
        error_abs_total += abs(total_act - total_pred)
        error_rate_squared = error_abs_total/(j+1) 

        print(f"\nTotal ETA: {total_pred:.2f} seconds (Actual: {total_act:.2f})")
        print(f"Within Bound? : {'YES' if (total_pred - total_std) <= total_act <= (total_pred + total_std) else 'NO'}")
        print(f"Confidence: +/- {total_std:.2f} seconds")
        print(f"Confidence Level: {total_pred/total_std if total_act>0 else 0}")
        print(f"\nPrediction Std Deviation: {prediction_std_deviation:.2f} , Confidence Std Deviation: {confidence_std_deviation:.2f} , Actual Std Deviation: {actual_std_deviation:.2f}")
        print(f"Error: {error_rate_squared}")
        print(f"Error Tendency: {error_rate}")
        print(f"\n")
    
        print(f"總共 {j + 1} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
        print(f"平均 {num_segment} Section，有 {section_within_bound_counts/len(x_global_val)} section 落在預測區間內。")
        print(f"平均置信度指標: {number_of_ratio/len(x_global_val)}")