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
    data = torch.tensor(raw_data_np, dtype=torch.float32)
    
    x_global = data[:, 0:14]
    y_sections = data[:, 14:14+num_segment]
    raw_local = data[:, 14+num_segment+1:14+num_segment+1+(num_segment*4)]
    
    
    x_local = raw_local.view(-1, num_segment, 4)
    
    return x_global, x_local, y_sections


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
            net.weight = PyroSample(dist.Normal(0., 0.1).expand([output_dim, input_dim]).to_event(2))
            net.bias = PyroSample(dist.Normal(0., 0.1).expand([output_dim]).to_event(1))
            self.nets.append(net)
            
    def forward(self, x_inputs):
        # x_inputs is a list of tensors, one per segment
        outputs = []
        for i in range(self.num_segments):
            out = torch.nn.functional.leaky_relu(self.nets[i](x_inputs[i]), negative_slope=0.1)
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
            net.weight = PyroSample(dist.Normal(0., 0.2).expand([output_dim, net_input_dim]).to_event(2))
            
            net.bias = PyroSample(dist.Normal(0., 0.1).expand([output_dim]).to_event(1))
            self.nets.append(net)
            
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
        """
        self.input_layer = PyroModuleList([])
        for i in range(num_sections):
            l = PyroModule[nn.Linear](input_dim, hidden_dim)
            l.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
            l.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
            self.input_layer.append(l)
        """
            
        
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections)
        
        num_layer = num_sections
        
        # --- Propagation Layers (The Matrix) ---
        # "Different function for each layer each section"
        # We create N layers. Each layer contains N unique networks.
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2)
            for _ in range(num_layer) # Layer depth = num_segments
        ])
        
        # --- Output Heads ---
        final_dim = hidden_dim
        
        self.heads_loc = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df = PyroModuleList([])
        
        for i in range(self.num_sections):
            loc_std_dev = 2
            
            # Loc Head
            h_loc = PyroModule[nn.Linear](final_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(0., loc_std_dev).expand([1, final_dim]).to_event(2))
            h_loc.bias = PyroSample(dist.Normal(40., 20.0).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)
            
            # Scale Head
            h_scale = PyroModule[nn.Linear](final_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(0., 0.1).expand([1, final_dim]).to_event(2))
            h_scale.bias = PyroSample(dist.Normal(-7., 0.5).expand([1]).to_event(1)) 
            self.heads_scale.append(h_scale)
            
            # DF Head
            h_df = PyroModule[nn.Linear](final_dim, 1)
            h_df.weight = PyroSample(dist.Normal(0., 0.2).expand([1, final_dim]).to_event(2))
            h_df.bias = PyroSample(dist.Normal(2., 0.5).expand([1]).to_event(1))
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
            scale = torch.nn.functional.softplus(self.heads_scale[i](final_feat)) * 0.01 + 1e-3
            df = torch.nn.functional.softplus(self.heads_df[i](final_feat)) + 2.5
            
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
        
        
        # 3. Heads (Early Exit) 

        return all_locs, all_scales, all_dfs

# ==========================================
# 4. EXECUTION
# ==========================================
def model_fn(x_global, x_local, y_true=None):
    locs, scales, dfs = bnn_model(x_global, x_local)
    with pyro.plate("data", x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(dfs[i].squeeze(), locs[i].squeeze(), scales[i].squeeze())
            target = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)

if __name__ == "__main__":
    from pyro.optim import PyroLRScheduler
    
    file_path = "trip_info_9_section.xlsx"
    x_global_all, x_local_all, y_all = process_raw_data(file_path)
    
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
    
    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=1)

    
    optimizer = ExponentialLR({
    "optimizer": torch.optim.AdamW, # AdamW is often more stable than Adam
    "optim_args": {
        "lr": 0.005, 
        "weight_decay": 0.01 # AdamW expects higher weight decay values (usually 0.01 to 0.1)
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

    for epoch in range(epochs):
        epoch_loss = 0
        for x_g_batch, x_l_batch, y_batch in train_loader:
            loss = svi.step(x_g_batch, x_l_batch, y_batch)
            epoch_loss += loss
            
        print(f"Epoch {epoch}: Train Loss {epoch_loss/len(train_loader):.2f}")

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
    error_abs_total = 0
    error_rate_squared = 0
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
        error_abs_total += abs(actual_total - final_mean)
        error_rate_squared = error_abs_total/(j+1) 

        print(f"\nTotal ETA: {final_mean:.2f} seconds (Actual: {actual_total:.2f})")
        print(f"\nWithin Bound? : {'YES' if (actual_total >= final_mean - final_std and actual_total <= final_mean + final_std) else 'NO'}")
        print(f"Confidence: +/- {final_std:.2f} seconds")
        print(f"Confidence Level: {final_mean/final_std if actual_total>0 else 0}")
        print(f"Error: {error_rate_squared}")
        print(f"Error Tendency: {error_rate}")
        print(f"\n\n")
        
        print(f"總共 {j + 1} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
        print(f"平均 {num_segment} Section，有 {section_within_bound_counts/len(x_global_val)} section 落在預測區間內。")
        print(f"平均置信度指標: {number_of_ratio/len(x_global_val)}")
    
    
    