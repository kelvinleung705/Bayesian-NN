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
from torch_geometric.nn import MessagePassing

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
    print(f"Successfully loaded data shape: {raw_data_np.shape}")
    
    data = torch.tensor(raw_data_np, dtype=torch.float32)
    x_global = data[:, 0:12]
    y_sections = data[:, 12:12+num_segment]
    raw_local = data[:, 12+num_segment+1:12+num_segment+1+(num_segment*4)]
    x_local = raw_local.view(-1, num_segment, 4)
    
    return x_global, x_local, y_sections

# ==========================================
# 2. BAYESIAN ADJACENT GNN MODEL
# ==========================================
class BayesianAdjacentGraphConv(PyroModule, MessagePassing):
    def __init__(self, in_channels, out_channels, layer_name=""):
        MessagePassing.__init__(self, aggr='mean')
        PyroModule.__init__(self)
        self.layer_name = layer_name
        
        self.neighbor_transform = PyroModule[nn.Linear](in_channels, out_channels)
        self.neighbor_transform.weight = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels, in_channels]).to_event(2)
        )
        self.neighbor_transform.bias = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels]).to_event(1)
        )
        
        self.self_transform = PyroModule[nn.Linear](in_channels, out_channels)
        self.self_transform.weight = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels, in_channels]).to_event(2)
        )
        self.self_transform.bias = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels]).to_event(1)
        )
        
    def forward(self, x, edge_index):
        neighbor_msg = self.propagate(edge_index, x=x)
        self_msg = self.self_transform(x)
        out = neighbor_msg + self_msg
        return out
    
    def message(self, x_j):
        return self.neighbor_transform(x_j)


class AdjacentBayesianGNN(PyroModule):
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, 
                 hidden_dim=12, num_gnn_layers=2, bidirectional=True):
        super().__init__()
        self.num_sections = num_sections
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.bidirectional = bidirectional
        
        node_feature_dim = global_dim + local_dim + 1
        
        self.input_projection = PyroModule[nn.Linear](node_feature_dim, hidden_dim)
        self.input_projection.weight = PyroSample(
            dist.Normal(0., 0.5).expand([hidden_dim, node_feature_dim]).to_event(2)
        )
        self.input_projection.bias = PyroSample(
            dist.Normal(0., 0.5).expand([hidden_dim]).to_event(1)
        )
        
        self.gnn_layers = PyroModuleList([
            BayesianAdjacentGraphConv(hidden_dim, hidden_dim, f"gnn_{i}")
            for i in range(num_gnn_layers)
        ])
        
        self.heads = PyroModuleList([])
        for i in range(num_sections):
            head = PyroModule[nn.Linear](hidden_dim, 3)
            head.weight = PyroSample(
                dist.Normal(0., 0.5).expand([3, hidden_dim]).to_event(2)
            )
            head.bias = PyroSample(
                dist.Normal(0., 0.5).expand([3]).to_event(1)
            )
            self.heads.append(head)
        
        # Learnable activation noise SCALE (Global parameter)
        self.activation_log_noise = PyroSample(
            dist.Normal(-2., 1.).expand([num_gnn_layers]).to_event(1)
        )
    
    def create_adjacent_graph(self, num_sections, bidirectional=True):
        edge_list = []
        for i in range(num_sections - 1):
            edge_list.append([i, i + 1])
            if bidirectional:
                edge_list.append([i + 1, i])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
    
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
    
        edge_index_template = self.create_adjacent_graph(
            self.num_sections, 
            bidirectional=self.bidirectional
        ).to(device)
    
        node_features_list = []
        for i in range(self.num_sections):
            local_i = all_sections_data[:, i, :]
            time_i = accumulated_time[:, i, :]
            node_feat = torch.cat([global_features, local_i, time_i], dim=1)
            node_features_list.append(node_feat)

        node_features = torch.stack(node_features_list, dim=1)
        node_features_flat = node_features.view(-1, node_features.shape[-1])
    
        edge_index_list = []
        for b in range(batch_size):
            offset = b * self.num_sections
            edge_index_list.append(edge_index_template + offset)
        edge_index_batched = torch.cat(edge_index_list, dim=1)
    
        h = self.input_projection(node_features_flat)
        h = torch.relu(h)
    
        for layer_idx, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(h, edge_index_batched)
            h = torch.relu(h)
        
            # FIX: Use torch.randn instead of pyro.sample for the noise values.
            # This allows dynamic batch sizes (h.shape) while still learning the noise scale.
            noise_scale = torch.exp(self.activation_log_noise[layer_idx])
            
            # Generate noise dynamically based on current batch shape
            noise = torch.randn_like(h) * noise_scale
            h = h + noise
    
        h = h.view(batch_size, self.num_sections, self.hidden_dim)
    
        all_locs, all_scales, all_dfs = [], [], []
    
        for i in range(self.num_sections):
            h_i = h[:, i, :]
            raw_output = self.heads[i](h_i)
        
            loc_i = raw_output[:, 0].unsqueeze(1)
            scale_i = torch.nn.functional.softplus(raw_output[:, 1]).unsqueeze(1) + 1e-3
            df_i = torch.nn.functional.softplus(raw_output[:, 2]).unsqueeze(1) + 2.5
        
            all_locs.append(loc_i)
            all_scales.append(scale_i)
            all_dfs.append(df_i)
        
            accumulated_time[:, i, :] = loc_i
    
        return all_locs, all_scales, all_dfs


# ==========================================
# 3. PROBABILISTIC MODEL
# ==========================================
def model_fn(x_global, x_local, y_true=None):
    locs, scales, dfs = bnn_gnn_model(x_global, x_local)
    
    with pyro.plate("data", x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            # 1. Flatten the predictions to 1D [Batch_Size]
            # .squeeze(-1) removes the trailing 1 dimension
            loc_i = locs[i].squeeze(-1)
            scale_i = scales[i].squeeze(-1)
            df_i = dfs[i].squeeze(-1)
            
            # T-Distribution for robustness against outliers
            #dist_i = dist.StudentT(dfs[i], locs[i], scales[i])
            dist_i = dist.StudentT(df_i, loc_i, scale_i)
            
            # Get target for this section if available
            # target = y_true[:, i].unsqueeze(1) if y_true is not None else None
            #target = y_true[:, i] if y_true is not None else None
            target = None
            if y_true is not None:
                target = y_true[:, i] # No unsqueeze needed!
            
            
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    file_path = "trip_info3.xlsx"
    print("Preprocessing...")
    
    # split date from global, local , and y
    x_global_all, x_local_all, y_all = process_raw_data(file_path)
    
    # split train and validation data by id
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    #training data set
    x_global_train = x_global_all[train_idx]
    x_local_train = x_local_all[train_idx]
    y_train = y_all[train_idx]
    
    #validation data set
    x_global_val = x_global_all[val_idx]
    x_local_val = x_local_all[val_idx]
    y_val = y_all[val_idx]

    print(f"Number of Traing data set: {len(train_idx)}, Number of Validation data set: {len(val_idx)}")

    # --- C. Setup Model & Training ---
    bnn_gnn_model = AdjacentBayesianGNN(num_sections=num_segment, global_dim=12, local_dim=4)
    guide = AutoDiagonalNormal(model_fn)
    optimizer = pyro.optim.Adam({"lr": 0.003})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    pyro.clear_param_store()
    epochs = 100
    
    # NOTE: drop_last=False is now safe because we handle dynamic noise shapes manually
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for x_g_batch, x_l_batch, y_batch in train_loader:
            loss = svi.step(x_g_batch, x_l_batch, y_batch)
            epoch_loss += loss
        print(f"Epoch {epoch}: Loss {epoch_loss / len(train_loader):.2f}")
        avg_train_loss = epoch_loss / len(train_loader)
        # ===== VALIDATION PHASE =====
        # Create validation dataloader
        val_dataset = TensorDataset(x_global_val, x_local_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        val_loss = 0
        for x_g_val_batch, x_l_val_batch, y_val_batch in val_loader:
            # Use evaluate_loss (no gradient updates)
            loss = svi.evaluate_loss(x_g_val_batch, x_l_val_batch, y_val_batch)
            val_loss += loss
    
        avg_val_loss = val_loss / len(val_loader)
    
        # Print both losses
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.2f}, Val Loss = {avg_val_loss:.2f}")

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
    