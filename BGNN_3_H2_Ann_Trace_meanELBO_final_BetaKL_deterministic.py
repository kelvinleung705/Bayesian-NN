import statistics

from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import pyro
from annealing import Annealer
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceMeanField_ELBO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyro.optim import ExponentialLR
import joblib # You might need to pip install joblib

num_segment = 9
# ==========================================
# 1. DATA PRE-PROCESSING 
# ==========================================
def process_raw_data(file_path):
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)
    df = df[
    #(df.iloc[:, 2] == 1) & 
    #(df.iloc[:, 7] <= 2) #&
    #(df.iloc[:, 56 ] == 5)
    (df.iloc[:, 56].isin([1, 2, 3, 4]))
    ]
    end = 14 + num_segment + 1 + (num_segment * 4)
    df_subset = df.iloc[:, 0:end]
    df_subset = df_subset.dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    raw_data_np = df_subset.values.astype(np.float32)
    
    # 1. Extract pre-normalized X features directly
    #14
    x_global = torch.tensor(raw_data_np[:, 0:9], dtype=torch.float32)
    
    raw_local = raw_data_np[:, 9+num_segment+1:9+num_segment+1+(num_segment*4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)
    
    # 2. Extract and Normalize Targets (Y)
    y_raw = raw_data_np[:, 9:9+num_segment]
    
    # We still scale Y so the Bayesian priors (mean=0) work correctly.
    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y_raw), dtype=torch.float32)
    
    # Return x_global, x_local, the normalized Y, and the Y scaler (for inverse transform later)
    return x_global, x_local, y_scaled, scaler_y


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
        current_time = torch.zeros(batch_size, self.num_sections).to(device)
        
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
                    time_i = current_time[:, i:i+1] 
                    #time_i = torch.zeros(batch_size, 1).to(device)
                elif i < current_section:
                    # For sections in the PAST, we could inject their actual historical times,
                    # but for simplicity, feeding the current clock is often enough, 
                    # or you can feed 0 if you want them to be "static anchors".
                    # Let's feed the current clock to show "how far past" they are.
                    #time_i = current_time
                    time_i = current_time[:, i:i+1] 
                    #time_i = torch.zeros(batch_size, 1).to(device)
                    
                else:
                    # For sections in the FUTURE, we don't know the time yet.
                    # We feed 0.0 (or you could feed current_time as a baseline).
                    # Let's feed 0.0 to indicate "unreached".
                    #time_i = torch.zeros(batch_size, 1).to(device)
                    time_i = current_time[:, i:i+1] 
                
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
            
            for j in range(current_section, self.num_sections):
                current_time[:, j:j+1] = current_time[:, j:j+1] + loc
            
            #print(loc.abs().mean().item())
            
            #current_time = current_time + loc
            
        return all_locs
# ==========================================
# 4. TRAINING & EVALUATION (Standard PyTorch)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_month_sorted.xlsx"
    
    print("\n--- Processing Data ---")
    x_global_all, x_local_all, y_all, scaler_y = process_raw_data(file_path)
    
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_train = x_global_all[train_idx].to(device)
    x_local_train = x_local_all[train_idx].to(device)
    y_train = y_all[train_idx].to(device)
    
    x_global_val = x_global_all[val_idx].to(device)
    x_local_val = x_local_all[val_idx].to(device)
    y_val = y_all[val_idx].to(device)
    
    x_global_all = x_global_all.to(device)
    x_local_all = x_local_all.to(device)
    y_final_all = y_all.to(device)

    # Initialize Model
    model = DeterministicMatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32).to(device)
    
    # Standard PyTorch Optimizer and Loss Function (MSE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    criterion = nn.MSELoss() # Mean Squared Error is standard for deterministic regression
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1, eta_min=0.0001)

    epochs = 750
    batch_size = 1024
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\n--- Starting Deterministic Training ({len(train_dataset)} samples) ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for x_g_batch, x_l_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass returns a list of 9 tensors
            locs = model(x_g_batch, x_l_batch)
            
            # Stack them into shape [Batch, 9] to match y_batch
            preds = torch.cat(locs, dim=1) 
            
            # Calculate standard MSE Loss
            loss = criterion(preds, y_batch)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()

        if epoch % 1 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch:05d} | LR: {current_lr:.6f} | MSE Loss (Scaled): {avg_loss:.4f}")
            
    # Save Model
    torch.save(model.state_dict(), "deterministic_model_jan_apr.pt")
    joblib.dump(scaler_y, "deterministic_scaler_jan_apr.pkl")
    print("Deterministic Model Saved.")

    # --- INFERENCE ---
    print("\n--- Final Validation (Deterministic) ---")
    model.eval() # Turn off Dropout
    
    with torch.no_grad():
        # Predict all validation data at once (no sampling needed!)
        val_locs = model(x_global_val, x_local_val)
        val_preds_scaled = torch.cat(val_locs, dim=1).cpu().numpy()
        
        # Inverse Transform
        pred_real = scaler_y.inverse_transform(val_preds_scaled)
        actual_real = scaler_y.inverse_transform(y_val.cpu().numpy())
        
        # Calculate Totals
        total_pred = np.sum(pred_real, axis=1)
        total_act = np.sum(actual_real, axis=1)
        
        mae = np.mean(np.abs(total_act - total_pred))
        mean_error = np.mean(total_act - total_pred)
        
        
        in_range = 0
        # Print a few examples
        for j in range(len(x_global_val)):
            if abs(total_act[j] - total_pred[j]) <= 69.36:  # Example condition for being in range
                in_range += 1
            print(f"Sample {j}: Pred {total_pred[j]:.1f}s | Actual {total_act[j]:.1f}s")
            print(f"Samples in range: {in_range}/{len(x_global_val)}")
        print(f"Deterministic Mean Absolute Error (MAE): {mae:.2f} seconds")
        print(f"Deterministic Mean Error: {mean_error:.2f} seconds")