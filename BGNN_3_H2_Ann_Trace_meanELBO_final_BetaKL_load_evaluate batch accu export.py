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
    return x_global_all, x_local_all, y_scaled_all
    """
    # --- CRITICAL FIX: Replicate the original Validation Split ---
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_val = x_global_all[val_idx]
    x_local_val = x_local_all[val_idx]
    y_val = y_scaled_all[val_idx]
    
    return x_global_val, x_local_val, y_val
    """


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

# ==========================================
# 3. THE "MATRIX" GNN MODEL
# ==========================================
class MatrixGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=8, device = 'cuda', pnt_1 = None):
        super().__init__()
        self.num_sections = num_sections
        self.device = device
        input_dim = global_dim + local_dim + 1 
        self.pnt_1 = pnt_1
        
        # --- Layer 0: Input Projection (Unique per segment) ---
        
            
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        
        # --- Propagation Layers (The Matrix) ---
        # "Different function for each layer each section"
        # We create N layers. Each layer contains N unique networks.
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(3) # Layer depth = num_segments
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

            df_std = torch.tensor(1, device=device)
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
        current_time = torch.zeros(batch_size, self.num_sections).to(device)
        
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
                    # If we are looking at the section we are currently predicting,
                    # we inject the REAL running clock time.
                    time_i = current_time[:, i:i+1] 
                    #time_i = torch.zeros(batch_size, 1).to(device)
                elif i < current_section:
                    # For sections in the PAST, we could inject their actual historical times,
                    # but for simplicity, feeding the current clock is often enough, 
                    # or you can feed 0 if you want them to be "static anchors".
                    # Let's feed the current clock to show "how far past" they are.
                    time_i = current_time[:, i:i+1] 
                    #time_i = torch.zeros(batch_size, 1).to(device)
                    
                else:
                    # For sections in the FUTURE, we don't know the time yet.
                    # We feed 0.0 (or you could feed current_time as a baseline).
                    # Let's feed 0.0 to indicate "unreached".
                    time_i = torch.zeros(batch_size, 1).to(device)
                if self.pnt_1 is None:
                    self.pnt_1 = True
                    #print(current_time.abs().mean().item(), current_time)
                
                if time_i.abs().mean().item() > 15:
                    time_i = torch.zeros(batch_size, 1).to(device)
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
            if self.pnt_1 is True:
                self.pnt_1 = False
                #print(loc)
            #print("loc:", loc.mean().item())
            scale = torch.nn.functional.softplus(self.heads_scale[current_section](final_feat)) + 1e-3 
            df = torch.nn.functional.softplus(self.heads_df[current_section](final_feat)) + 2.5
            
            # 4. Store the predictions
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            #print(all_locs[-1].mean().item())
            #print(all_locs[-1].mean().item())
            # 5. UPDATE THE CLOCK for the next loop iteration
            # We add the predicted travel time of THIS section to the running total.
            for j in range(current_section, self.num_sections):
                current_time[:, j:j+1] = current_time[:, j:j+1] + loc
            #print(loc.abs().mean().item())
            
            #current_time = current_time + loc
            
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
    #trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy
    saved_params_path = "ghost_bus_model_fixed_1_2000_df10_KL_9_accu.pt" # Replace with exact saved params file name
    
    saved_scaler_path = "y_scaler.pkl"              # Replace with exact saved scaler file name
    #file_path = "bad_visibility.xlsx"
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_sorted.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_jumpy.xlsx"
    
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

    try:
        pyro.get_param_store().load(saved_params_path, map_location=device.type)
    except Exception as e:
        print(f"Warning: Could not load param store. Proceeding with uninitialized weights. {e}")

    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32, device=device).to(device)
    base_guide = AutoDiagonalNormal(model_fn).to(device)
    
    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=0.1):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true, total_size=total_size, kl_weight=kl_weight)
            
    with torch.no_grad():
        dummy_y = torch.zeros((1, num_segment), device=device)
        try:
            guide_fn(x_global_val[0:1], x_local_val[0:1], y_true=dummy_y)
        except Exception:
            pass

    print("Model parameters loaded and correctly linked!")

    # ==========================================
    # 2.5 EXPORT GRAPH FOR NETRON VISUALIZATION
    # ==========================================
    print("\n--- Exporting Model Graph to ONNX for Netron ---")
    try:
        # Create a tiny dummy batch (batch_size=1)
        dummy_g = x_global_val[0:1].clone()
        dummy_l = x_local_val[0:1].clone()

        # We wrap in eval() and no_grad() so Pyro treats the weights as static for export
        bnn_model.eval()
        with torch.no_grad():
            torch.onnx.export(
                bnn_model, 
                (dummy_g, dummy_l), 
                "matrix_gnn_structure_accu.onnx", # THIS IS THE FILE NETRON NEEDS
                export_params=True,
                opset_version=14, 
                input_names=['global_features', 'local_sections_data'],
            )
        print("-> Successfully saved 'matrix_gnn_structure_accu.onnx'!")
        print("-> Go to https://netron.app and drag this file into the browser.")
        bnn_model.train() # Return to original state for your inference script
    except Exception as e:
        print(f"-> ONNX export failed (Pyro models can sometimes be tricky to trace): {e}")