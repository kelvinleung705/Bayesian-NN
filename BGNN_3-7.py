import statistics

from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceMeanField_ELBO
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
    def __init__(self, input_dim, output_dim, num_segments, device = 'cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.nets = PyroModuleList([])
        
        for i in range(num_segments):
            # Simple Linear transformation for this specific segment
            net = PyroModule[nn.Linear](input_dim, output_dim)
            zero = torch.tensor(0., device=device)
            point_one = torch.tensor(0.1, device=device)
            net.weight = PyroSample(dist.Normal(zero, point_one).expand([output_dim, input_dim]).to_event(2))
            net.bias = PyroSample(dist.Normal(zero, point_one).expand([output_dim]).to_event(1))
            self.nets.append(net)
            
    def forward(self, x_inputs):
        # x_inputs is a list of tensors, one per segment
        outputs = []
        for i in range(self.num_segments):
            out = torch.nn.functional.silu(self.nets[i](x_inputs[i]))
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
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.2, device = 'cuda'):
        super().__init__()
        self.num_segments = num_segments
        self.device = device
        
        loc_self = torch.tensor(2., device=device)
        loc_side = torch.tensor(0., device=device)
        scale = torch.tensor(0.5, device=device)
        
        zero = torch.tensor(0., device=device)
        w_scale = torch.tensor(0.2, device=device)
        b_scale = torch.tensor(0.1, device=device)

        self.w_self = PyroSample(dist.Normal(loc_self, scale).expand([num_segments]).to_event(1))
        self.w_left = PyroSample(dist.Normal(loc_side, scale).expand([num_segments]).to_event(1))
        self.w_right = PyroSample(dist.Normal(loc_side, scale).expand([num_segments]).to_event(1))
        
        self.nets = PyroModuleList([])
        
        he_std = (2.0 / (input_dim * 3)) ** 0.5
        
        for i in range(num_segments):
            # Input size x3 because we concat [Self, Left, Right]
            net_input_dim = input_dim * 3 
            net_1 = PyroModule[nn.Linear](net_input_dim, output_dim)
            net_1.weight = PyroSample(dist.Normal(zero, w_scale).expand([output_dim, net_input_dim]).to_event(2))
            
            net_1.bias = PyroSample(dist.Normal(zero, b_scale).expand([output_dim]).to_event(1))
            self.nets.append(net_1)
    
        self.dropout = PyroModule[nn.Dropout](p=dropout_rate)

    def forward(self, prev_layer_outputs):
        outputs = []
        for i in range(self.num_segments):
            # Apply individual weights using softmax to ensure positivity
            ws = torch.nn.functional.softmax(self.w_self[i], dim=0)
            wl = torch.nn.functional.softmax(self.w_left[i], dim=0)
            wr = torch.nn.functional.softmax(self.w_right[i], dim=0)

            self_feat = prev_layer_outputs[i] * ws
            
            if i > 0:
                left_feat = prev_layer_outputs[i-1] * 0
            else:
                left_feat = torch.zeros_like(self_feat)

            if i < self.num_segments - 1:
                right_feat = prev_layer_outputs[i+1] * wr
            else:
                right_feat = torch.zeros_like(self_feat)

            combined = torch.cat([self_feat, left_feat, right_feat], dim=1)
            
            out = self.nets[i](combined)
            out = self.dropout(out) 
            out = torch.nn.functional.silu(out)
            outputs.append(out)
        return outputs


# ==========================================
# 3. THE "MATRIX" GNN MODEL
# ==========================================
class MatrixGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=8, device = 'cuda'):
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
            for _ in range(num_sections) # Layer depth = num_segments
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

            scale_std = torch.tensor(0.1, device=device)
            scale_bias_mu = torch.tensor(0., device=device)
            scale_bias_std = torch.tensor(1.0, device=device)

            df_std = torch.tensor(0.1, device=device)
            df_bias_mu = torch.tensor(0., device=device)
            df_bias_std = torch.tensor(1.0, device=device)
            
            # Loc Head
            h_loc = PyroModule[nn.Sequential](PyroModule[nn.Linear](final_dim, 16), nn.SiLU(),PyroModule[nn.Linear](16, 1))
            h_loc.weight = PyroSample(dist.Normal(zero, loc_std).expand([1, final_dim]).to_event(2))
            h_loc.bias = PyroSample(dist.Normal(loc_bias_mu, loc_bias_std).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)
            
            # Scale Head
            h_scale = PyroModule[nn.Sequential](PyroModule[nn.Linear](final_dim, 16),nn.SiLU(),PyroModule[nn.Linear](16, 1))
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
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        # 1. Layer 0 (Local Input)
        inputs_list = []
        for i in range(self.num_sections):
            loc_i = all_sections_data[:, i, :]
            time_i = accumulated_time[:, i, :]
            inp = torch.cat([global_features, loc_i, time_i], dim=1)
            inputs_list.append(inp)
            
        h_current = self.embedding_layer(inputs_list)
            
        # 2. Propagation Layers
        for layer in self.prop_layers:
            h_current = layer(h_current)
        
        
        all_locs, all_scales, all_dfs = [], [], []
        
        
        
        for i in range(self.num_sections):
            # GET RID OF EARLY EXIT: Only use h_current[i]
            final_feat = h_current[i] 
            
            loc = self.heads_loc[i](final_feat)
            scale = torch.nn.functional.softplus(self.heads_scale[i](final_feat)) + 0.1
            df = torch.nn.functional.softplus(self.heads_df[i](final_feat)) + 2.5
            
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
        
        
        # 3. Heads (Early Exit) 

        return all_locs, all_scales, all_dfs

# ==========================================
# 4. EXECUTION
# ==========================================
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
    locs, scales, dfs = bnn_model(x_global, x_local)
    
    if total_size is None:
        total_size = x_global.shape[0]
    
    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0], dim=-1):
        # --- THE FIX FOR STUCK PREDICTIONS ---
        # We scale the Likelihood by 100.0.
        # This tells the model: "Fitting the data is 100x more important than the Prior."
        # Use kl_weight instead of 100.0!
        with pyro.poutine.scale(scale=kl_weight):
            for i in range(len(locs)):
                dist_i = dist.Normal(locs[i].squeeze(), scales[i].squeeze())
                #dist_i = dist.StudentT(df=dfs[i].squeeze(), loc=locs[i].squeeze(), scale=scales[i].squeeze())
                target = y_true[:, i] if y_true is not None else None
                pyro.sample(f"obs_section_{i}", dist_i, obs=target)

if __name__ == "__main__":
    from pyro.optim import PyroLRScheduler
    # 1. SETUP DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path = "trip_info_9_section_ver2_simplify_ultra.xlsx"
    x_global_all, x_local_all, y_all, scaler_y = process_raw_data(file_path)
    
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_train = x_global_all[train_idx]
    x_local_train = x_local_all[train_idx]
    y_train = y_all[train_idx]
    x_global_val = x_global_all[val_idx].to(device)
    x_local_val = x_local_all[val_idx].to(device)
    y_val = y_all[val_idx].to(device)

    # Initialize The Matrix Model
    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=14, local_dim=4, hidden_dim=32, device=device).to(device)
    
    guide = AutoDiagonalNormal(model_fn).to(device)
    
    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=1)
    
    
    optimizer = ExponentialLR({
    "optimizer": torch.optim.AdamW, # AdamW is often more stable than Adam
    "optim_args": {
        "lr": 0.0005, #0.002
        "weight_decay": 0.01 # AdamW expects higher weight decay values (usually 0.01 to 0.1)
    }, 
    "gamma": 0.995 # Slower decay (reduces by 0.5% instead of 1% per epoch)
})

    #svi = SVI(model_fn, guide, optimizer, loss=TraceMeanField_ELBO())
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    pyro.clear_param_store()
    epochs = 200
    
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    print(len(train_dataset))
    total_size = len(train_dataset)
    
    mae_calc = torch.nn.L1Loss()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mae = 0
        batches = 0
        
        # Calculate KL Annealing weight (goes from 0.0 to 1.0 over 50 epochs)
        # current_kl_weight = min(1.0, (epoch + 1) / 50.0)      #3/6/1928
        #current_kl_weight = min(0.1, (epoch + 1) / 100 * 0.05) #3/6/1058
        current_kl_weight = min(0.05, (epoch + 1) / 100 * 0.02) #3/7/1248  
        for x_g_batch, x_l_batch, y_batch in train_loader:
            x_g_batch = x_g_batch.to(device)
            x_l_batch = x_l_batch.to(device)
            y_batch = y_batch.to(device)
            #loss = svi.step(x_g_batch, x_l_batch, y_batch, total_size)
            # PASS BOTH total_size and kl_weight as keyword arguments
            loss = svi.step(x_g_batch, x_l_batch, y_batch, total_size=total_size, kl_weight=current_kl_weight)
            epoch_loss += loss
            """
            with torch.no_grad():
                locs, _, _ = bnn_model(x_g_batch, x_l_batch)
                preds = torch.stack(locs, dim=1).squeeze()
                epoch_mae += mae_calc(preds, y_batch).item()
            batches += 1
            """
        print(f"Epoch {epoch}: ELBO {epoch_loss/len(train_loader):.2f} | KL Weight: {current_kl_weight:.2f}")

    
    print("\n--- Final Prediction Test ---")
    predictive = Predictive(model_fn, guide=guide, num_samples=50)
    samples = predictive(x_global_val, x_local_val)
    
    total_actual = y_val.sum(dim=1)
    
    
    
    
# ==========================================
    # 5. INFERENCE 
    # ==========================================
    print("\n--- Final Prediction Test ---")
    list_of_predict = []
    list_of_actual = []
    list_of_predict_sections = [[] for i in range(num_segment)]
    list_of_confidence_sections = [[] for i in range(num_segment)]
    list_of_actual_sections = [[] for i in range(num_segment)]
    predictive = Predictive(model_fn, guide=guide, num_samples=50)

    
    # Counters for accuracy
    total_samples = 0
    
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
        
        pred_means_scaled = []
        pred_stds_scaled = []
        actuals_scaled = []
        
        total_trip_samples_real = torch.zeros(50, device=device)
        
        for i in range(num_segment):
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            mean_scaled = sec_samples.mean().item()
            std_scaled = sec_samples.std().item()
            
            pred_means_scaled.append(mean_scaled)
            pred_stds_scaled.append(std_scaled)
            actuals_scaled.append(y_val[j, i].item())
            
        # Inverse Transform
        pred_real = scaler_y.inverse_transform([pred_means_scaled])[0]
        actual_real = scaler_y.inverse_transform([actuals_scaled])[0]
        std_real = np.array(pred_stds_scaled) * scaler_y.scale_
        
        total_pred = 0
        trip_section_within_bound = 0
        
        # 6. CREATE ACCUMULATOR ON DEVICE
        
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
        print(f"\nWithin Bound? : {'YES' if (total_pred - total_std) <= total_act <= (total_pred + total_std) else 'NO'}")
        print(f"Confidence: +/- {total_std:.2f} seconds")
        print(f"Confidence Level: {total_pred/total_std if total_act>0 else 0}")
        print(f"\nPrediction Std Deviation: {prediction_std_deviation:.2f} , Actual Std Deviation: {actual_std_deviation:.2f})")
        print(f"Error: {error_rate_squared}")
        print(f"Error Tendency: {error_rate}")
        print(f"\n\n")
        
        print(f"總共 {j + 1} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
        print(f"平均 {num_segment} Section，有 {section_within_bound_counts/len(x_global_val)} section 落在預測區間內。")
        print(f"平均置信度指標: {number_of_ratio/len(x_global_val)}")