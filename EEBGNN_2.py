from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyro.optim import ExponentialLR

num_segment = 9

# ==========================================
# 1. DATA PRE-PROCESSING 
# ==========================================
def process_raw_data(file_path):
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)

    end = 14 + num_segment + 1 + (num_segment * 4)
    df_subset = df.iloc[:, 0:end]
    df_subset = df_subset.dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    raw_data_np = df_subset.values.astype(np.float32)
    
    # 1. Extract pre-normalized X features directly
    x_global = torch.tensor(raw_data_np[:, 0:14], dtype=torch.float32)
    
    raw_local = raw_data_np[:, 14+num_segment+1:14+num_segment+1+(num_segment*4)]
    x_local = torch.tensor(raw_local.reshape(-1, num_segment, 4), dtype=torch.float32)
    
    # 2. Extract and Normalize Targets (Y)
    y_raw = raw_data_np[:, 14:14+num_segment]
    
    # We still scale Y so the Bayesian priors (mean=0) work correctly.
    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y_raw), dtype=torch.float32)
    
    # Return x_global, x_local, the normalized Y, and the Y scaler (for inverse transform later)
    return x_global, x_local, y_scaled, scaler_y


#==========================================
# 2. EMBEDDING LAYER (UNSHARED) LAYERS
# ==========================================

class LocalIsolationLayer(PyroModule):
    """
    LAYER 1: PURELY LOCAL
    Each segment has its own neural network.
    NO neighbor information is used here.
    """
    def __init__(self, input_dim, output_dim, num_segments):
        super().__init__()
        self.num_segments = num_segments
        self.nets = PyroModuleList([])
        
        for i in range(num_segments):
            # Simple Linear transformation for this specific segment
            net = PyroModule[nn.Linear](input_dim, output_dim)
            net.weight = PyroSample(dist.StudentT(3, 0., 0.2).expand([output_dim, input_dim]).to_event(2))
            net.bias = PyroSample(dist.StudentT(3, 0., 0.2).expand([output_dim]).to_event(1))
            self.nets.append(net)
            
    def forward(self, x_inputs):
        # x_inputs is a list of tensors, one per segment
        outputs = []
        for i in range(self.num_segments):
            out = self.nets[i](x_inputs[i])
            out = torch.nn.functional.leaky_relu(out, negative_slope=0.1)
            outputs.append(out)
        return outputs
    
    
# ==========================================
# 2. UNIQUE MIXING LAYER (With Dropout & Weights)
# ==========================================
class NeighborMixingLayer(PyroModule):
    """
    LAYER 2: MIXING
    Each segment has its own network.
    Input = Output of Previous Layer (Self) + Output of Previous Layer (Neighbors)
    """
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2):
        super().__init__()
        self.num_segments = num_segments
        
        """
        self.segment_importance = PyroSample(
            dist.Normal(1.0, 0.5).expand([num_segments]).to_event(1)
        )
        """
        self.w_self = PyroSample(dist.Normal(2.0, 0.5).expand([num_segments]).to_event(1))
        self.w_left = PyroSample(dist.Normal(0.0, 0.5).expand([num_segments]).to_event(1))
        self.w_right = PyroSample(dist.Normal(0.0, 0.5).expand([num_segments]).to_event(1))
        
        self.nets = PyroModuleList([])
        
        he_std = (2.0 / (input_dim * 3)) ** 0.5
        
        for i in range(num_segments):
            # Input size x3 because we concat [Self, Left, Right]
            net_input_dim = input_dim * 3 
            net = PyroModule[nn.Linear](net_input_dim, output_dim)
            net.weight = PyroSample(dist.StudentT(3, 0., 0.3).expand([output_dim, net_input_dim]).to_event(2))
            net.bias = PyroSample(dist.StudentT(3, 0., 0.3).expand([output_dim]).to_event(1))
            self.nets.append(net)
            
        self.dropout = PyroModule[nn.Dropout](p=dropout_rate)

    def forward(self, prev_layer_outputs):
        outputs = []
        raw_weights = torch.stack([self.w_self, self.w_left, self.w_right]) 
        weights = torch.nn.functional.softmax(raw_weights, dim=0) 
        for i in range(self.num_segments):
            # Apply individual weights using softmax to ensure positivity
            ws = weights[0, i]
            wl = weights[1, i]
            wr = weights[2, i]
            
            self_feat = prev_layer_outputs[i] * ws
            if i > 0:
                left_feat = prev_layer_outputs[i-1] * wl
            else:
                left_feat = torch.zeros_like(self_feat)

            if i < self.num_segments - 1:
                right_feat = prev_layer_outputs[i+1] * wr
            else:
                right_feat = torch.zeros_like(self_feat)

            combined = torch.cat([self_feat, left_feat, right_feat], dim=1)
            
            out = self.nets[i](combined)
            out = self.dropout(out) 
            out = torch.relu(out)
            outputs.append(out)
        return outputs

# ==========================================
# 3. THE "MATRIX" GNN MODEL
# ==========================================
class MatrixGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=8):
        super().__init__()
        self.num_sections = num_sections
        input_dim = global_dim + local_dim + 1 
        
        
        # --- Layer 0: Input Projection (Unique per segment) ---
        
            
        
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections)
        
        # --- Propagation Layers (The Matrix) ---
        # "Different function for each layer each section"
        # We create N layers. Each layer contains N unique networks.
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.1)
            for _ in range(num_sections) # Layer depth = num_segments
        ])
        
        # --- Output Heads ---
        final_dim = hidden_dim
        
        self.heads_loc = PyroModuleList([PyroModuleList([]) for _ in range(self.num_layers + 1)])
        self.heads_scale = PyroModuleList([PyroModuleList([]) for _ in range(self.num_layers + 1)])
        self.heads_df = PyroModuleList([PyroModuleList([]) for _ in range(self.num_layers + 1)])
        self.exit_gates = PyroModuleList([])
        
        
        self.num_layer = num_sections*2
        
        for layer_i in range(self.num_layers + 1):
            for sec_i in range(num_sections):
                # Loc Head (Wide prior)
                h_loc = PyroModule[nn.Linear](hidden_dim, 1)
                h_loc.weight = PyroSample(dist.Normal(0., 2.0).expand([1, hidden_dim]).to_event(2))
                h_loc.bias = PyroSample(dist.Normal(0., 1.0).expand([1]).to_event(1))
                self.heads_loc[layer_i].append(h_loc)
                
                # Scale Head 
                h_scale = PyroModule[nn.Linear](hidden_dim, 1)
                h_scale.weight = PyroSample(dist.Normal(0., 0.1).expand([1, hidden_dim]).to_event(2))
                h_scale.bias = PyroSample(dist.Normal(-5., 1.0).expand([1]).to_event(1))
                self.heads_scale[layer_i].append(h_scale)
                
                # DF Head
                h_df = PyroModule[nn.Linear](hidden_dim, 1)
                h_df.weight = PyroSample(dist.Normal(0., 0.1).expand([1, hidden_dim]).to_event(2))
                h_df.bias = PyroSample(dist.Normal(2., 0.5).expand([1]).to_event(1))
                self.heads_df[layer_i].append(h_df)
        
        
        for i in range(self.num_layer):
            # Exit Gate
            # Prior: Bias towards 0.5 (Neutral)
            gate_input_dim = hidden_dim + 2 
            gate = PyroModule[nn.Linear](gate_input_dim, 1)
            gate.weight = PyroSample(dist.Normal(0., 1).expand([1, gate_input_dim]).to_event(2))
            #-2.0
            gate.bias = PyroSample(dist.Normal(0., 1).expand([1]).to_event(1)) #0.2
            self.exit_gates.append(gate)
        
    
    def predict_step(self, h_current, layer_idx):
        locs, scales, dfs = [], [], []
        for i in range(self.num_sections):
            feat = h_current[i] 
            loc = self.heads_loc[layer_idx][i](feat)
            scale = torch.nn.functional.softplus(self.heads_scale[layer_idx][i](feat)) * 0.05 + 1e-3
            df = torch.nn.functional.softplus(self.heads_df[layer_idx][i](feat)) + 2.5
            
            locs.append(loc)
            scales.append(scale)
            dfs.append(df)
        return locs, scales, dfs
            

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        inputs_list = []
        for i in range(self.num_sections):
            loc_i = all_sections_data[:, i, :]
            time_i = accumulated_time[:, i, :]
            inp = torch.cat([global_features, loc_i, time_i], dim=1)
            inputs_list.append(inp)
            
        h_current = self.embedding_layer(inputs_list)
        
        all_layer_preds = [] 
        
        # Layer 0
        all_layer_preds.append(self.predict_step(h_current, layer_idx=0))

        # Layers 1 to 9
        for idx, layer in enumerate(self.prop_layers):
            h_current = layer(h_current)
            all_layer_preds.append(self.predict_step(h_current, layer_idx=idx+1))
            
        # Return a simple list of predictions for every layer
        return all_layer_preds
        
        
        # 3. Heads (Early Exit) 
        

        #return all_locs, all_scales, all_dfs

# ==========================================
# 4. EXECUTION
# ==========================================
def model_fn(x_global, x_local, y_true=None):
    (locs, scales, dfs), all_gates = bnn_model(x_global, x_local)
    with pyro.plate("data", x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(dfs[i].squeeze(), locs[i].squeeze(), scales[i].squeeze())
            target = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)
    
    # 2. Early-Exit Regularization (Energy Efficiency Penalty)
    total_gate_cost = 0
    # Loop over all layers except the last one
    for layer_idx, layer_gates in enumerate(all_gates[:-1]):
        for sec_i, sec_gate in enumerate(layer_gates):
            # Add to pyro deterministic to track it during inference!
            pyro.deterministic(f"gate_layer_{layer_idx}_section_{sec_i}", sec_gate)
            
            # Cost of NOT exiting = (1 - g). We want to minimize this.
            prob_continue = 1.0 - sec_gate
            total_gate_cost = total_gate_cost + prob_continue.sum()
            
    # Benchmark of "Precision" vs "Computation"
    # 0.0005 is very low -> Values Accuracy over Computation
    # // FIX 4: Force the model to exit!
    # A lambda of 0.0005 tells the model "Compute is practically free."
    # We increase this to 0.05 to force the network to use the early layers.
    reg_lambda = 0.018   #0.02
    pyro.factor("gate_regularization", -reg_lambda * total_gate_cost)

if __name__ == "__main__":
    from pyro.optim import PyroLRScheduler
    
    file_path = "trip_info5_2.xlsx"
    x_global_all, x_local_all, y_all, scaler_y  = process_raw_data(file_path)
    
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_train = x_global_all[train_idx]
    x_local_train = x_local_all[train_idx]
    y_train = y_all[train_idx]
    x_global_val = x_global_all[val_idx]
    x_local_val = x_local_all[val_idx]
    y_val = y_all[val_idx]

    # Initialize The Matrix Model
    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=14, local_dim=4, hidden_dim=32)
    
    guide = AutoDiagonalNormal(model_fn)
    
    """
    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=1)
    """

    
    optimizer = ExponentialLR({
        "optimizer": torch.optim.AdamW, # AdamW is often more stable than Adam
        "optim_args": {
            "lr": 0.005, 
            "weight_decay": 0.00 
        }, 
        "gamma": 0.995 # Slower decay (reduces by 0.5% instead of 1% per epoch)
    })
    """
    optimizer = PyroLRScheduler({
        "optimizer": torch.optim.AdamW,
        "optim_args": {"lr": 0.01, "weight_decay": 1e-4},
        "scheduler": scheduler_constructor
    })
    """

    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    pyro.clear_param_store()
    epochs = 100
    
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    mae_calc = torch.nn.L1Loss()

    for epoch in range(epochs):
        epoch_loss, epoch_mae, batches = 0, 0, 0
        for x_g_batch, x_l_batch, y_batch in train_loader:
            loss = svi.step(x_g_batch, x_l_batch, y_batch)
            #print(loss/256)
            epoch_loss += loss
            
            with torch.no_grad():
                (locs, _, _), _ = bnn_model(x_g_batch, x_l_batch)
                preds = torch.stack(locs, dim=1).squeeze()
                epoch_mae += mae_calc(preds, y_batch).item()
            batches += 1
            
        print(f"Epoch {epoch}: Train Loss {epoch_loss/len(train_loader):.2f}, MAE {epoch_mae/batches:.4f}")



    # ==========================================
    # 5. INFERENCE & GATE OBSERVATION
    # ==========================================
    print("\n--- Final Prediction Test (With Gate Decisions) ---")
    predictive = Predictive(model_fn, guide=guide, num_samples=50)
    
    within_bound_count = 0
    section_within_bound_counts = 0 
    number_of_ratio = 0
    
    # Just printing the first 10 for detailed viewing
    for j in range(min(200, len(x_global_val))):
        val_x_g = x_global_val[j:j+1]
        val_x_l = x_local_val[j:j+1]
    
        samples = predictive(val_x_g, val_x_l)
        
        pred_means_scaled = []
        pred_stds_scaled = []
        actuals_scaled = []
        exit_layers = []
        
        print(f"\n--- Sample {j} ---")
        trip_section_within_bound = 0
        
        for i in range(num_segment):
            # 1. Get the blended prediction
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            mean_scaled = sec_samples.mean().item()
            std_scaled = sec_samples.std().item()
            
            pred_means_scaled.append(mean_scaled)
            pred_stds_scaled.append(std_scaled)
            actuals_scaled.append(y_val[j, i].item())
            
            # 2. Determine "Hard Exit" layer based on the learned Gates
            # We look at the gates from Layer 0 upwards. 
            # First gate > 0.5 is the exit point.
            exited_at = num_segment # Default to final layer
            for l_idx in range(num_segment): # 0 to 9
                gate_key = f"gate_layer_{l_idx}_section_{i}"
                if gate_key in samples:
                    g_val = samples[gate_key].mean().item()
                    if g_val > 0.5:
                        exited_at = l_idx
                        break
            exit_layers.append(exited_at)

        # Inverse Transform
        pred_real = scaler_y.inverse_transform([pred_means_scaled])[0]
        actual_real = scaler_y.inverse_transform([actuals_scaled])[0]
        # Multiply std dev by scaler scale
        std_real = np.array(pred_stds_scaled) * scaler_y.scale_
        
        total_pred = 0
        
        for i in range(num_segment):
            print(f"  Sec {i}: Pred {pred_real[i]:.1f}s | Actual {actual_real[i]:.1f}s | Conf +/- {std_real[i]:.1f}s | Exited @ Layer {exit_layers[i]}")
            total_pred += pred_real[i]
            
            if actual_real[i] >= (pred_real[i] - std_real[i]) and actual_real[i] <= (pred_real[i] + std_real[i]):
                trip_section_within_bound += 1
                
        section_within_bound_counts += trip_section_within_bound
        
        total_act = actual_real.sum()
        total_std = np.sqrt(np.sum(std_real**2)) # Sum of variances for total trip std
        
        if total_act >= (total_pred - total_std) and total_act <= (total_pred + total_std):
            within_bound_count += 1
            
        print(f"  --> Total ETA: {total_pred:.1f}s (Actual: {total_act:.1f}s) | Conf +/- {total_std:.1f}s")
        print(f"  --> {'WITHIN' if (total_act >= (total_pred - total_std) and total_act <= (total_pred + total_std)) else 'OUTSIDE'} Bound.")
        if total_std > 0:
            number_of_ratio += total_pred/total_std

    print("\n--- Summary ---")
    print(f"Total Trips Within Bound: {within_bound_count} / {min(200, len(x_global_val))}")
    print(f"Avg Sections Within Bound: {section_within_bound_counts / min(200, len(x_global_val)):.2f} / {num_segment}")
    print(f"平均置信度指標: {number_of_ratio/min(200, len(x_global_val))}")
    

"""
    print("\n--- Final Prediction Test ---")
    predictive = Predictive(model_fn, guide=guide, num_samples=50)
    samples = predictive(x_global_val, x_local_val)
    
    total_actual = y_val.sum(dim=1)
    
    

    # --- D. Inference (Prediction) ---
    print("\n--- Final Prediction Test ---")
    
    # We can now process validation in larger batches without memory/shape errors
    val_dataset = TensorDataset(x_global_val, x_local_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Predictive helper
    predictive = Predictive(model_fn, guide=guide, num_samples=50)
    
    # Counters for accuracy
    total_samples = 0
    
    within_bound_count = 0
    number_of_ratio = 0
    section_within_bound_counts = 0 
    error_total = 0
    
    for j in range(len(x_global_val)):
    
        # Take the first item to predict
        val_x_g = x_global_val[j:j+1]
        val_x_l = x_local_val[j:j+1]
    
        # Run Monte Carlo Sampling (50 times)
        predictive = Predictive(model_fn, guide=guide, num_samples=50)
        samples = predictive(val_x_g, val_x_l)
    
        # Calculate Total ETA
        #total_time_samples = torch.zeros(50, 1)
        total_time_samples = torch.zeros(50)
    
        print("Predicted Section Times:")
        trip_section_within_bound_counts = 0
        for i in range(num_segment):
            # Get samples for this section
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            mean_t = sec_samples.mean().item()
            actual_t = y_val[j, i].item()
            print(f"  Section {i}: Pred {mean_t:.2f} | Actual {actual_t:.2f} | Conf +/- {sec_samples.std().item():.2f}")

            total_time_samples += sec_samples
            
            if actual_t >= mean_t - sec_samples.std().item() and actual_t <= mean_t + sec_samples.std().item():
                trip_section_within_bound_counts += 1
        section_within_bound_counts += trip_section_within_bound_counts

        final_mean = total_time_samples.mean().item()
        final_std = total_time_samples.std().item()
        actual_total = y_val[j].sum().item()
        
        
        if actual_total >= final_mean - final_std and actual_total <= final_mean + final_std:
            within_bound_count += 1
        if final_std > 0:
            number_of_ratio += final_mean/final_std
        error_total += (actual_total - final_mean)
        error_rate = error_total/(j+1) 

        print(f"\nTotal ETA: {final_mean:.2f} seconds (Actual: {actual_total:.2f})")
        print(f"\nWithin Bound? : {'YES' if (actual_total >= final_mean - final_std and actual_total <= final_mean + final_std) else 'NO'}")
        print(f"Confidence: +/- {final_std:.2f} seconds")
        print(f"Confidence Level: {final_mean/final_std if actual_total>0 else 0}")
        print(f"Error: {error_rate}")
        print(f"\n\n")
        
        print(f"總共 {j + 1} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
        print(f"平均 {num_segment} Section，有 {section_within_bound_counts/len(x_global_val)} section 落在預測區間內。")
        print(f"平均置信度指標: {number_of_ratio/len(x_global_val)}")
"""