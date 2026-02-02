import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, PyroModuleList
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal  # Changed from AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.nn import MessagePassing


num_segment = 3  # Number of sections/segments

# ==========================================
# 1. DATA PRE-PROCESSING (The "Slicer")
# ==========================================
def process_raw_data(file_path):
    """
    Input: Excel file
    Output: 
        x_global: [N, 12]
        x_local:  [N, num_segment, 4]
        y_sections: [N, num_segment]
    """
    print(f"Reading {file_path}...")
    
    df = pd.read_excel(file_path, header=None, skiprows=1)

    end = 12 + num_segment + 1 + (num_segment * 4)
    df_subset = df.iloc[:, 0:end]

    # Clean the data
    df_subset = df_subset.dropna()
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    raw_data_np = df_subset.values.astype(np.float32)
    
    print(f"Successfully loaded data shape: {raw_data_np.shape}")
    
    # Convert to Tensor
    data = torch.tensor(raw_data_np, dtype=torch.float32)
    
    # Global Features (Indices 0-11)
    x_global = data[:, 0:12]
    
    # Targets (Indices 12 to 12+num_segment)
    y_sections = data[:, 12:12+num_segment]
    
    # Local Features
    raw_local = data[:, 12+num_segment+1:12+num_segment+1+(num_segment*4)]
    
    # Reshape into [Batch, num_segment, 4 Features]
    x_local = raw_local.view(-1, num_segment, 4)
    
    return x_global, x_local, y_sections


# ==========================================
# 2. BAYESIAN ADJACENT GNN MODEL
# ==========================================
class BayesianAdjacentGraphConv(PyroModule, MessagePassing):
    """
    Bayesian Graph Convolution that only passes messages to adjacent nodes
    """
    def __init__(self, in_channels, out_channels, layer_name=""):
        super().__init__(aggr='mean')  # Average messages from neighbors
        self.layer_name = layer_name
        
        # Bayesian transformation for neighbor messages
        self.neighbor_transform = PyroModule[nn.Linear](in_channels, out_channels)
        self.neighbor_transform.weight = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels, in_channels]).to_event(2)
        )
        self.neighbor_transform.bias = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels]).to_event(1)
        )
        
        # Bayesian transformation for self (the node itself)
        self.self_transform = PyroModule[nn.Linear](in_channels, out_channels)
        self.self_transform.weight = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels, in_channels]).to_event(2)
        )
        self.self_transform.bias = PyroSample(
            dist.Normal(0., 0.5).expand([out_channels]).to_event(1)
        )
        
    def forward(self, x, edge_index):
        """
        x: Node features [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        """
        # Propagate messages from neighbors
        neighbor_msg = self.propagate(edge_index, x=x)
        
        # Transform self features
        self_msg = self.self_transform(x)
        
        # Combine neighbor and self information
        out = neighbor_msg + self_msg
        
        return out
    
    def message(self, x_j):
        """
        Create messages from neighbors
        x_j: Features of neighbor nodes [num_edges, in_channels]
        """
        return self.neighbor_transform(x_j)


class AdjacentBayesianGNN(PyroModule):
    """
    Bayesian GNN where sections only communicate with adjacent neighbors
    """
    def __init__(self, num_sections=3, global_dim=12, local_dim=4, 
                 hidden_dim=8, num_gnn_layers=2, bidirectional=True):
        super().__init__()
        self.num_sections = num_sections
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.bidirectional = bidirectional
        
        # Initial feature dimension: global + local + time
        node_feature_dim = global_dim + local_dim + 1
        
        # Input projection: map raw features to hidden_dim
        self.input_projection = PyroModule[nn.Linear](node_feature_dim, hidden_dim)
        self.input_projection.weight = PyroSample(
            dist.Normal(0., 0.5).expand([hidden_dim, node_feature_dim]).to_event(2)
        )
        self.input_projection.bias = PyroSample(
            dist.Normal(0., 0.5).expand([hidden_dim]).to_event(1)
        )
        
        # GNN layers for message passing
        self.gnn_layers = PyroModuleList([
            BayesianAdjacentGraphConv(hidden_dim, hidden_dim, f"gnn_{i}")
            for i in range(num_gnn_layers)
        ])
        
        # Output heads (one per section)
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
        
        # Learnable activation noise per GNN layer
        self.activation_log_noise = PyroSample(
            dist.Normal(-2., 1.).expand([num_gnn_layers]).to_event(1)
        )
    
    def create_adjacent_graph(self, num_sections, bidirectional=True):
        """
        Create graph where each section only connects to adjacent sections
        """
        edge_list = []
        
        for i in range(num_sections - 1):
            # Forward edge: i → i+1
            edge_list.append([i, i + 1])
            
            if bidirectional:
                # Backward edge: i+1 → i
                edge_list.append([i + 1, i])
        
        # Convert to edge_index format [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def forward(self, global_features, all_sections_data):
        """
        global_features: [batch_size, global_dim]
        all_sections_data: [batch_size, num_sections, local_dim]
        """
        batch_size = global_features.shape[0]
        device = global_features.device
        
        # Initialize time
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        # STEP 1: CREATE ADJACENT GRAPH STRUCTURE
        edge_index = self.create_adjacent_graph(
            self.num_sections, 
            bidirectional=self.bidirectional
        ).to(device)
        
        # STEP 2: PREPARE NODE FEATURES FOR EACH SECTION
        node_features_list = []
        for i in range(self.num_sections):
            local_i = all_sections_data[:, i, :]  # [batch_size, local_dim]
            time_i = accumulated_time[:, i, :]    # [batch_size, 1]
            
            # Concatenate: [batch_size, global_dim + local_dim + 1]
            node_feat = torch.cat([global_features, local_i, time_i], dim=1)
            node_features_list.append(node_feat)
        
        # Stack: [batch_size, num_sections, feature_dim]
        node_features = torch.stack(node_features_list, dim=1)
        
        # STEP 3: BATCH PROCESSING
        num_nodes_per_graph = self.num_sections
        total_nodes = batch_size * num_nodes_per_graph
        node_features_flat = node_features.view(total_nodes, -1)
        
        # Create batched edge indices
        edge_index_batch_list = []
        for b in range(batch_size):
            offset = b * num_nodes_per_graph
            edge_index_batch_list.append(edge_index + offset)
        edge_index_batched = torch.cat(edge_index_batch_list, dim=1)
        
        # STEP 4: PROJECT INPUT TO HIDDEN DIMENSION
        h = self.input_projection(node_features_flat)
        h = torch.relu(h)
        
        # STEP 5: MESSAGE PASSING (GNN LAYERS)
        for layer_idx, gnn_layer in enumerate(self.gnn_layers):
            # Apply graph convolution
            h = gnn_layer(h, edge_index_batched)
            h = torch.relu(h)
            
            # Add Bayesian activation noise
            noise_scale = torch.exp(self.activation_log_noise[layer_idx])
            with pyro.plate(f"gnn_noise_{layer_idx}", total_nodes, dim=-2):
                noise = pyro.sample(
                    f"gnn_act_noise_{layer_idx}",
                    dist.Normal(0., noise_scale).expand([total_nodes, self.hidden_dim]).to_event(1)
                )
            h = h + noise
        
        # STEP 6: RESHAPE BACK TO BATCH FORMAT
        h = h.view(batch_size, num_nodes_per_graph, self.hidden_dim)
        
        # STEP 7: GENERATE PREDICTIONS FOR EACH SECTION
        all_locs = []
        all_scales = []
        all_dfs = []
        
        for i in range(self.num_sections):
            h_i = h[:, i, :]  # [batch_size, hidden_dim]
            
            # Apply output head
            raw_output = self.heads[i](h_i)
            
            # Extract distribution parameters
            loc_i = raw_output[:, 0].unsqueeze(1)
            scale_i = torch.nn.functional.softplus(raw_output[:, 1]).unsqueeze(1) + 1e-3
            df_i = torch.nn.functional.softplus(raw_output[:, 2]).unsqueeze(1) + 2.5
            
            all_locs.append(loc_i)
            all_scales.append(scale_i)
            all_dfs.append(df_i)
            
            # Update accumulated time
            accumulated_time[:, i, :] = loc_i
        
        return all_locs, all_scales, all_dfs


# ==========================================
# 3. PROBABILISTIC MODEL
# ==========================================
def model_fn(x_global, x_local, y_true=None):
    """Probabilistic model for training"""
    locs, scales, dfs = bnn_gnn_model(x_global, x_local)
    
    with pyro.plate("data", x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            loc_i = locs[i].squeeze(-1)
            scale_i = scales[i].squeeze(-1)
            df_i = dfs[i].squeeze(-1)
            
            dist_i = dist.StudentT(df_i, loc_i, scale_i)
            target = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    file_path = "trip_info3.xlsx"
    
    # --- B. Preprocess ---
    print("Preprocessing...")
    x_g_all, x_l_all, y_all = process_raw_data(file_path)
    
    print(f"Global Input Shape: {x_g_all.shape}")
    print(f"Local Input Shape:  {x_l_all.shape}")
    print(f"Target Shape:       {y_all.shape}")
    
    # --- Train/Val Split ---
    idx = np.arange(x_g_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    # Training set
    x_g_train = x_g_all[train_idx]
    x_l_train = x_l_all[train_idx]
    y_train = y_all[train_idx]

    # Validation set
    x_g_val = x_g_all[val_idx]
    x_l_val = x_l_all[val_idx]
    y_val = y_all[val_idx]

    print(f"Number of training data: {len(train_idx)}, Number of validation data: {len(val_idx)}")

    # --- C. Setup Model & Training ---
    bnn_gnn_model = AdjacentBayesianGNN(
        num_sections=num_segment, 
        global_dim=12, 
        local_dim=4, 
        hidden_dim=8,
        num_gnn_layers=2,  # Number of message-passing rounds
        bidirectional=True  # Sections can communicate both ways
    )
    
    guide = AutoDiagonalNormal(model_fn)  # Changed from AutoNormal
    optimizer = pyro.optim.Adam({"lr": 0.01})  # Increased learning rate
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    pyro.clear_param_store()
    epochs = 200  # Increased epochs
    
    train_dataset = TensorDataset(x_g_train, x_l_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for x_g_batch, x_l_batch, y_batch in train_loader:
            loss = svi.step(x_g_batch, x_l_batch, y_batch)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation phase
        val_dataset = TensorDataset(x_g_val, x_l_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        val_loss = 0
        for x_g_val_batch, x_l_val_batch, y_val_batch in val_loader:
            loss = svi.evaluate_loss(x_g_val_batch, x_l_val_batch, y_val_batch)
            val_loss += loss
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.2f}, Val Loss = {avg_val_loss:.2f}")

    # --- D. Inference (Prediction) ---
    print("\n--- Final Prediction Test ---")
    
    within_bound_count = 0
    number_of_ratio = 0
    section_within_bound_counts = 0 
    
    for j in range(len(x_g_val)):
        # Take one validation sample
        val_x_g = x_g_val[j:j+1]
        val_x_l = x_l_val[j:j+1]
    
        # Run Monte Carlo Sampling (100 samples for better uncertainty)
        predictive = Predictive(model_fn, guide=guide, num_samples=100)
        samples = predictive(val_x_g, val_x_l)
    
        # Calculate Total ETA
        total_time_samples = torch.zeros(100)
    
        print(f"\n=== Sample {j+1}/{len(x_g_val)} ===")
        print("Predicted Section Times:")
        trip_section_within_bound_counts = 0
        
        for i in range(num_segment):
            # Get samples for this section
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            mean_t = sec_samples.mean().item()
            std_t = sec_samples.std().item()
            actual_t = y_val[j, i].item()
            
            print(f"  Section {i}: Pred {mean_t:.2f} | Actual {actual_t:.2f} | Conf ±{std_t:.2f}")

            total_time_samples += sec_samples
            
            # Check if actual falls within ±1 std
            if actual_t >= mean_t - std_t and actual_t <= mean_t + std_t:
                trip_section_within_bound_counts += 1
        
        section_within_bound_counts += trip_section_within_bound_counts

        final_mean = total_time_samples.mean().item()
        final_std = total_time_samples.std().item()
        actual_total = y_val[j].sum().item()
        
        # Check if total falls within bounds
        if actual_total >= final_mean - final_std and actual_total <= final_mean + final_std:
            within_bound_count += 1
        
        if final_std > 0:
            number_of_ratio += final_mean / final_std

        print(f"\nTotal ETA: {final_mean:.2f} seconds (Actual: {actual_total:.2f})")
        print(f"Within Bound?: {'YES' if (actual_total >= final_mean - final_std and actual_total <= final_mean + final_std) else 'NO'}")
        print(f"Confidence: ±{final_std:.2f} seconds")
        print(f"Confidence Ratio: {final_mean/final_std if final_std > 0 else 0:.2f}")
    
    # Final statistics
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"Total {len(x_g_val)} validation datas，there are {within_bound_count} lands on the predicted interval.")
    print(f"Total accuracy: {within_bound_count/len(x_g_val)*100:.1f}%")
    print(f"平均 {num_segment} Section，有 {section_within_bound_counts/len(x_g_val):.2f} section lands on the predicted interval.")
    print(f"Section accuracy: {section_within_bound_counts/(len(x_g_val)*num_segment)*100:.1f}%")
    print(f"mean/error ratio: {number_of_ratio/len(x_g_val):.2f}")