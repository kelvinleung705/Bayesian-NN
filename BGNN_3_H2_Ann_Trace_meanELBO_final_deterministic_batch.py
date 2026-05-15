import torch
import torch.nn as nn
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
    Reads data, transforms it, and perfectly recreates the validation split
    used in the original training script so the samples match 1-to-1.
    """
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)
    """
    df = df[
    #(df.iloc[:, 2] == 1) & 
    #(df.iloc[:, 7] <= 2) #&
    #(df.iloc[:, 56 ] == 5)
    (df.iloc[:, 56].isin([5]))
    ]
    """
    

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
    
    return x_global_all, x_local_all, y_scaled_all


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
    def __init__(self, num_sections=9, global_dim=9, local_dim=4, hidden_dim=16):
        super().__init__()
        self.num_sections = num_sections
        input_dim = global_dim + local_dim + 1 
            
        self.embedding_layer = DeterministicLocalIsolationLayer(input_dim, hidden_dim, num_sections)
        
        self.prop_layers = nn.ModuleList([
            DeterministicNeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2)
            for _ in range(2)
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
        current_time = torch.zeros(batch_size, 1).to(device)
        
        all_locs = []
        
        
        # --- THE AUTOREGRESSIVE WATERFALL LOOP ---
        for current_section in range(self.num_sections):
            # 1. Build the Input List for the GNN
            inputs_list = []
            for i in range(self.num_sections):
                loc_i = all_sections_data[:, i, :]
                # THE CRITICAL INJECTION LOGIC
                if i == current_section:
                    # If we are looking at the section we are currently predicting,
                    # we inject the REAL running clock time.
                    #time_i = current_time
                    time_i = current_time
                    #time_i = torch.zeros(batch_size, 1).to(device)
                elif i < current_section:
                    # For sections in the PAST, we could inject their actual historical times,
                    # but for simplicity, feeding the current clock is often enough, 
                    # or you can feed 0 if you want them to be "static anchors".
                    # Let's feed the current clock to show "how far past" they are.
                    #time_i = current_time
                    time_i = current_time
                    #time_i = torch.zeros(batch_size, 1).to(device)
                    
                else:
                    # For sections in the FUTURE, we don't know the time yet.
                    # We feed 0.0 (or you could feed current_time as a baseline).
                    # Let's feed 0.0 to indicate "unreached".
                    #time_i = torch.zeros(batch_size, 1).to(device)
                    time_i = torch.zeros(batch_size, 1).to(device)
                
                if time_i.abs().mean().item() > 15:
                    time_i = torch.zeros(batch_size, 1).to(device)
                #time_i = torch.clamp(time_i, min=-15.0, max=15.0)
                #print(time_i.abs().mean().item())
                    #print(f"Section {i} Time Injection: {time_i.abs().mean().item():.2f}")
                #print(time_i.abs().mean().item())
                inp = torch.cat([global_features, loc_i, time_i], dim=1)
                inputs_list.append(inp)
                
            # 2. Run the full GNN (Embedding + Mixing) to get context
            h_current = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h_current = layer(h_current)
                
            
            # 3. Extract ONLY the prediction for the current_section
            final_feat = h_current[current_section] 
            
            loc = self.heads_loc[current_section](final_feat)
            
            # 4. Store the predictions
            all_locs.append(loc)
            #print(all_locs[-1].mean().item())
            #print(all_locs[-1].mean().item())
            # 5. UPDATE THE CLOCK for the next loop iteration
            # We add the predicted travel time of THIS section to the running total.
            
            current_time = current_time + loc
            
            #print(loc.abs().mean().item())
            
            #current_time = current_time + loc
            
        return all_locs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # NOTE: This MUST point to a standard PyTorch state_dict (.pt), not a Pyro ParamStore.
    #saved_params_path = "deterministic_model_jan_apr.pt" 
    saved_params_path = "deterministic_model.pt" 
    #saved_scaler_path = "deterministic_scaler_jan_apr.pkl"  
    saved_scaler_path = "deterministic_scaler.pkl"  
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_month_sorted.xlsx"  # Ensure this is the correct path to your validation data            
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2026.xlsx"
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_June.xlsx"
    
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
    # 2. RECONSTRUCT & LOAD MODEL (DETERMINISTIC)
    # ==========================================
    print("\n--- Initializing and Loading Model Weights ---")

    bnn_model = DeterministicMatrixGNN(
        num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32
    ).to(device)

    # Load Standard PyTorch Weights
    # strict=False allows it to load even if the saved model had extra Bayesian heads we aren't using
    try:
        state_dict = torch.load(saved_params_path, map_location=device)
        bnn_model.load_state_dict(state_dict)
        print("Model parameters loaded successfully!")
    except Exception as e:
        print(f"\n[WARNING]: Could not load weights. Ensure the .pt file is a standard PyTorch state_dict, NOT a Pyro ParamStore. Error: {e}")

    # MANDATORY: Turns off dropout for deterministic output
    bnn_model.eval()

    # ==========================================
    # 3. FAST BATCHED INFERENCE LOOP
    # ==========================================
    print("\n--- Starting Fast Batched Prediction Test ---")
    
    batch_size = 1024 
    val_dataset = TensorDataset(x_global_val, x_local_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    all_pred_real = []
    all_actual_real = []

    with torch.no_grad():
        for x_g_batch, x_l_batch, y_batch in val_loader:
            
            # Forward pass (Returns a list of 9 tensors)
            preds_list = bnn_model(x_g_batch, x_l_batch)
            
        
            preds_scaled = torch.cat(preds_list, dim=1)
            
            # Combine list of [batch, 1] tensors into a single [batch, 9] tensor
            preds_scaled_np = preds_scaled.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()
            
            # Vectorized Inverse Transform
            pred_real = loaded_scaler.inverse_transform(preds_scaled_np)
            actual_real = loaded_scaler.inverse_transform(y_batch_np)
            
            all_pred_real.append(pred_real)
            all_actual_real.append(actual_real)

    # Combine all batches into fast NumPy arrays
    pred_real = np.vstack(all_pred_real)
    actual_real = np.vstack(all_actual_real)

    # ==========================================
    # 4. FAST VECTORIZED METRICS CALCULATION
    # ==========================================
    total_pred = pred_real.sum(axis=1)
    total_act = actual_real.sum(axis=1)
    total_samples = len(total_act)

    # --- HARDCODED MARGIN FOR COVERAGE CALCULATION ---
    hardcoded_margin = 80  # Seconds
    
    lower_bound = total_pred - hardcoded_margin
    upper_bound = total_pred + hardcoded_margin
    within_bound_mask = (total_act >= lower_bound) & (total_act <= upper_bound)
    within_bound_count = np.sum(within_bound_mask)
    
    # Section-level hardcoded bounds checking
    sec_lower_bounds = pred_real - hardcoded_margin
    sec_upper_bounds = pred_real + hardcoded_margin
    sec_within_bounds_mask = (actual_real >= sec_lower_bounds) & (actual_real <= sec_upper_bounds)
    section_within_bound_counts = np.sum(sec_within_bounds_mask)

    # Running Errors
    error_rates = (total_act - total_pred)
    abs_errors = np.abs(total_act - total_pred)
    
    pred_running_std = pd.Series(total_pred).expanding().std().fillna(0).values
    act_running_std = pd.Series(total_act).expanding().std().fillna(0).values

    # ==========================================
    # 5. PRINT OUT INDIVIDUAL RESULTS
    # ==========================================
    print_limit = len(x_global_val) # Change to a smaller number if output is too long
    all_loc = 0
    
    for j in range(print_limit):
        print(f"\n--- Sample {j} ---")
        for i in range(num_segment):
            is_in_bound = "YES" if sec_within_bounds_mask[j, i] else "NO"
            print(f"  Sec {i}: Pred {pred_real[j, i]:.1f}s | Actual {actual_real[j, i]:.1f}s | Within +/- {hardcoded_margin}s? {is_in_bound}")
            
        print(f"\nTotal ETA: {total_pred[j]:.2f} seconds (Actual: {total_act[j]:.2f})")
        print(f"Within Bound (+/- {hardcoded_margin}s)? : {'YES' if within_bound_mask[j] else 'NO'}")
        
        all_loc += total_pred[j]
        current_error_tendency = np.sum(error_rates[:j+1]) / (j + 1)
        current_error_squared = np.sum(abs_errors[:j+1]) / (j + 1)
        
        print(f"Prediction Std Deviation: {pred_running_std[j]:.2f} , Actual Std Deviation: {act_running_std[j]:.2f}")
        print(f"Error (MAE running): {current_error_squared:.2f}")
        print(f"Error Tendency (Bias running): {current_error_tendency:.2f}")

    # ==========================================
    # 6. FINAL METRICS SUMMARY
    # ==========================================
    print("\n==============================================")
    print("        FINAL DETERMINISTIC SUMMARY           ")
    print("==============================================")
    print(f"Hardcoded Acceptable Margin : +/- {hardcoded_margin}s")
    print(f"Total Validated Trips       : {total_samples}")
    print(f"Trips within margin         : {within_bound_count} ({(within_bound_count/total_samples)*100:.2f}%)")
    print(f"Average Sections w/in margin: {section_within_bound_counts / total_samples:.2f} / {num_segment}")
    print(f"Mean Average Prediction ETA : {all_loc / total_samples:.2f} seconds")
    print(f"Overall Mean Absolute Error : {np.mean(abs_errors):.2f} seconds")
    print("==============================================")