import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

num_segment = 3 

# ==========================================
# 1. DATA PRE-PROCESSING 
# ==========================================
def process_raw_data(file_path):
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, header=None, skiprows=1)

    end = 12 + num_segment + 1 + (num_segment * 4)
    df_subset = df.iloc[:, 0:end]
    df_subset = df_subset.dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    raw_data_np = df_subset.values.astype(np.float32)
    data = torch.tensor(raw_data_np, dtype=torch.float32)
    
    x_global = data[:, 0:12]
    y_sections = data[:, 12:12+num_segment]
    # Correctly reshape local data
    raw_local = data[:, 12+num_segment+1:12+num_segment+1+(num_segment*4)]
    x_local = raw_local.view(-1, num_segment, 4)
    
    return x_global, x_local, y_sections

# ==========================================
# 2. UNSHARED GNN LAYER (Unique weights per segment)
# ==========================================
class UnsharedGNNLayer(PyroModule):
    def __init__(self, input_dim, output_dim, num_segments=3):
        super().__init__()
        self.num_segments = num_segments
        
        # Create a SPECIFIC neural network for EACH segment
        # This acts like the "Waterfall" model but inside a GNN structure
        self.segment_nets = PyroModuleList([])
        
        for i in range(num_segments):
            # Input to each segment = [Self_Features + Neighbor_Features]
            # We assume neighbor features are aggregated (summed) before entering
            # So input size is effectively doubled (Self + Context)
            net = PyroModule[nn.Linear](input_dim * 2, output_dim)
            
            # Bayesian Priors
            net.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, input_dim * 2]).to_event(2))
            net.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))
            self.segment_nets.append(net)

    def forward(self, x_features):
        """
        x_features: List of tensors, where x_features[i] is [Batch, Dim] for segment i
        """
        outputs = []
        batch_size = x_features[0].shape[0]
        device = x_features[0].device
        
        for i in range(self.num_segments):
            # 1. Identify Neighbors (0-indexed chain)
            neighbors = []
            if i > 0: neighbors.append(i - 1)
            if i < self.num_segments - 1: neighbors.append(i + 1)
            
            # 2. Aggregate Neighbor Messages
            if len(neighbors) > 0:
                # Stack neighbors and sum them (Add aggregation)
                neighbor_feats = torch.stack([x_features[n] for n in neighbors], dim=0)
                neighbor_msg = torch.sum(neighbor_feats, dim=0)
            else:
                neighbor_msg = torch.zeros_like(x_features[i])
            
            # 3. Concatenate [Self, Neighbor_Sum]
            # This gives the model context without enforcing shared weights
            combined_input = torch.cat([x_features[i], neighbor_msg], dim=1)
            
            # 4. Pass through the UNIQUE network for this segment
            out = self.segment_nets[i](combined_input)
            out = torch.relu(out)
            outputs.append(out)
            
        return outputs

# ==========================================
# 3. THE "JUMP KNOWLEDGE" MODEL
# ==========================================
class IndependentEarlyExitGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, hidden_dim=16):
        super().__init__()
        self.num_sections = num_sections
        
        # Feature sizes
        input_dim = global_dim + local_dim + 1 # +1 for Time
        
        # --- Layer 1: Independent Projection ---
        # Maps raw inputs to hidden space. unique per segment.
        self.layer1 = UnsharedGNNLayer(input_dim, hidden_dim, num_sections)
        
        # --- Layer 2: Deep Interaction ---
        # Takes Layer 1 output, mixes neighbors again. Unique per segment.
        self.layer2 = UnsharedGNNLayer(hidden_dim, hidden_dim, num_sections)
        
        # --- Output Heads (Early Exit / Jump Knowledge) ---
        # The head sees: [Layer1_Output + Layer2_Output]
        # This allows "Early Exit" logic: if Layer 1 is enough, it ignores Layer 2 weights.
        final_input_dim = hidden_dim * 2
        
        self.heads_loc = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df = PyroModuleList([])
        
        for i in range(num_sections):
            # Loc Head
            h_loc = PyroModule[nn.Linear](final_input_dim, 1)
            h_loc.weight = PyroSample(dist.Normal(0., 1.).expand([1, final_input_dim]).to_event(2))
            h_loc.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
            self.heads_loc.append(h_loc)
            
            # Scale Head (Uncertainty)
            h_scale = PyroModule[nn.Linear](final_input_dim, 1)
            h_scale.weight = PyroSample(dist.Normal(0., 0.5).expand([1, final_input_dim]).to_event(2))
            h_scale.bias = PyroSample(dist.Normal(-3., 1.).expand([1]).to_event(1)) # Start confident
            self.heads_scale.append(h_scale)
            
            # DF Head (Tail shape)
            h_df = PyroModule[nn.Linear](final_input_dim, 1)
            h_df.weight = PyroSample(dist.Normal(0., 1.).expand([1, final_input_dim]).to_event(2))
            h_df.bias = PyroSample(dist.Normal(2., 1.).expand([1]).to_event(1))
            self.heads_df.append(h_df)

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        # 1. Prepare Initial Inputs
        input_list = []
        for i in range(self.num_sections):
            loc_i = all_sections_data[:, i, :]
            time_i = accumulated_time[:, i, :]
            # [Batch, 12+4+1]
            feat = torch.cat([global_features, loc_i, time_i], dim=1)
            input_list.append(feat)
            
        # 2. Layer 1 Pass (Unshared GNN)
        # Returns list of [Batch, Hidden] per segment
        h1_list = self.layer1(input_list)
        
        # 3. Layer 2 Pass (Unshared GNN)
        # Input is h1_list. Returns list of [Batch, Hidden]
        h2_list = self.layer2(h1_list)
        
        # 4. Jump Knowledge / Output Generation
        all_locs, all_scales, all_dfs = [], [], []
        
        for i in range(self.num_sections):
            # CONCATENATE Layer 1 and Layer 2 (Early Exit logic)
            # The model can choose to rely on H1 (simple) or H2 (complex)
            final_feat = torch.cat([h1_list[i], h2_list[i]], dim=1)
            
            loc = self.heads_loc[i](final_feat)
            
            scale = torch.nn.functional.softplus(self.heads_scale[i](final_feat))
            scale = scale * 0.1 + 1e-3 # Enforce small initial variance
            
            df = torch.nn.functional.softplus(self.heads_df[i](final_feat)) + 2.5
            
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            
        return all_locs, all_scales, all_dfs

# ==========================================
# 4. TRAINING & EXECUTION
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
    
    file_path = "trip_info3.xlsx"
    x_global_all, x_local_all, y_all = process_raw_data(file_path)
    
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_train = x_global_all[train_idx]
    x_local_train = x_local_all[train_idx]
    y_train = y_all[train_idx]
    x_global_val = x_global_all[val_idx]
    x_local_val = x_local_all[val_idx]
    y_val = y_all[val_idx]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Instantiate the Unshared Model
    bnn_model = IndependentEarlyExitGNN(num_sections=num_segment, global_dim=12, local_dim=4, hidden_dim=32)
    
    guide = AutoDiagonalNormal(model_fn)
    
    # Smart Optimizer (Cosine Annealing)
    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=1)

    optimizer = pyro.optim.Adam({"lr": 0.005})
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
    epochs = 200
    
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for x_g_batch, x_l_batch, y_batch in train_loader:
            loss = svi.step(x_g_batch, x_l_batch, y_batch)
            epoch_loss += loss
            
        print(f"Epoch {epoch}: Train Loss {epoch_loss/len(train_loader):.2f}")

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

        print(f"\nTotal ETA: {final_mean:.2f} seconds (Actual: {actual_total:.2f})")
        print(f"\nWithin Bound? : {'YES' if (actual_total >= final_mean - final_std and actual_total <= final_mean + final_std) else 'NO'}")
        print(f"Confidence: +/- {final_std:.2f} seconds")
        print(f"Confidence Level: {final_mean/final_std if actual_total>0 else 0}")
        print(f"\n\n")
        
        print(f"總共 {len(x_global_val)} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
        print(f"平均 {num_segment} Section，有 {section_within_bound_counts/len(x_global_val)} section 落在預測區間內。")
        print(f"平均置信度指標: {number_of_ratio/len(x_global_val)}")
    
    
    