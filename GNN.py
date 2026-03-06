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
            net = nn.Linear(input_dim, output_dim)
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
    def __init__(self, input_dim, output_dim, num_segments, dropout_rate=0.1):
        super().__init__()
        self.num_segments = num_segments
        
        """
        self.segment_importance = PyroSample(
            dist.Normal(1.0, 0.5).expand([num_segments]).to_event(1)
        )
        """
        self.w_self = nn.Parameter(torch.ones(num_segments) * 2.0)
        self.w_left = nn.Parameter(torch.ones(num_segments) * 0.1)
        self.w_right = nn.Parameter(torch.ones(num_segments) * 0.1)
        
        self.nets = PyroModuleList([])
        
        he_std = (2.0 / (input_dim * 3)) ** 0.5
        
        for i in range(num_segments):
            # Input size x3 because we concat [Self, Left, Right]
            self.nets.append(nn.Sequential(
                nn.Linear(input_dim * 3, output_dim * 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(output_dim * 2, output_dim)
            ))
            
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, prev_layer_outputs):
        outputs = []
        
        
        # Softmax ensures weights sum to 1.0
        #raw_weights = torch.stack([self.w_self, self.w_left, self.w_right]) 
        #weights = torch.nn.functional.softmax(raw_weights, dim=0) 
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
            #out = torch.relu(out)
            #out = self.layer_norm(out + prev_layer_outputs[i])
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
        
        # --- Propagation Layers (The Matrix) ---
        # "Different function for each layer each section"
        # We create N layers. Each layer contains N unique networks.
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.1)
            for _ in range(num_sections) # Layer depth = num_segments
        ])
        
        # --- Output Heads ---
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_sections)
        ])
        
        

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        device = global_features.device
        accumulated_time = torch.zeros(batch_size, self.num_sections, 1).to(device)
        
        # 1. Layer 0 (Local Input)
        inputs_list = []
        for i in range(self.num_sections):
            inp = torch.cat([global_features, all_sections_data[:, i, :], accumulated_time[:, i, :]], dim=1)
            inputs_list.append(inp)
            
        h_current = self.embedding_layer(inputs_list)
            
        # 2. Propagation Layers
        for layer in self.prop_layers:
            h_current = layer(h_current)
        
        
        preds = []
        for i in range(self.num_sections):
            # Shape: [batch_size, 1]
            pred_i = self.output_heads[i](h_current[i])
            preds.append(pred_i)
            
        # Concatenate into [batch_size, num_sections]
        return torch.cat(preds, dim=1)

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_global_train = x_global_all[train_idx].to(device)
    x_local_train = x_local_all[train_idx].to(device)
    y_train = y_all[train_idx].to(device)
    x_global_val = x_global_all[val_idx].to(device)
    x_local_val = x_local_all[val_idx].to(device)
    y_val = y_all[val_idx].to(device)

    # Initialize The Matrix Model
    gnn_model = MatrixGNN(num_sections=num_segment, global_dim=14, local_dim=4, hidden_dim=32).to(device)
    
    #guide = AutoDiagonalNormal(model_fn)
    
    #def scheduler_constructor(optim):
    #    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=1)

    
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss() # Standard Mean Squared Error
    mae_calc = nn.L1Loss()   # For monitoring
    """
    optimizer = ExponentialLR({
    "optimizer": optimizer, # AdamW is often more stable than Adam
    "optim_args": {
        "lr": 0.001, 
        "weight_decay": 0.01 # AdamW expects higher weight decay values (usually 0.01 to 0.1)
    }, 
    "gamma": 0.995 # Slower decay (reduces by 0.5% instead of 1% per epoch)
})
    """


    print("\n--- Starting Training ---")
    epochs = 50
    
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        gnn_model.train()
        epoch_loss = 0
        epoch_mae = 0
        
        for x_g_batch, x_l_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = gnn_model(x_g_batch, x_l_batch)
            
            # Calculate standard MSE loss
            loss = criterion(predictions, y_batch)
            
            # Backprop and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mae += mae_calc(predictions, y_batch).item()
            
        scheduler.step()
        
        print(f"Epoch {epoch}: MSE Loss {epoch_loss/len(train_loader):.4f} | Scaled MAE {epoch_mae/len(train_loader):.4f}")

    
    

    # --- D. Inference (Prediction) ---
    print("\n--- Final Prediction Test ---")
    gnn_model.eval()
    
    error_abs_total = 0
    error_total = 0
    
    num_test_samples = len(x_global_val)
    
    with torch.no_grad():
    
        for j in range(len(x_global_val)):
    
            # Take the first item to predict
            val_x_g = x_global_val[j:j+1]
            val_x_l = x_local_val[j:j+1]

            # Predict (Returns [1, 9] tensor)
            pred = gnn_model(val_x_g, val_x_l).cpu().numpy()[0]
            actual = y_val[j:j+1].cpu().numpy()[0]
            
            #if j < 10: print(f"\n--- Sample {j} ---")
            
            total_pred = 0
            for i in range(num_segment):
                #if j < 10:
                print(f"  Sec {i}: Pred {pred[i]:.1f}s | Actual {actual[i]:.1f}s")
                total_pred += pred[i]
                    
            total_act = actual.sum()
            
            
            error_total += (total_act - total_pred)
            error_rate = error_total / (j+1) 
            error_abs_total += abs(total_act - total_pred)
            error_rate_squared = error_abs_total / (j+1) 

            #if j < 10:
            print(f"  --> Total ETA: {total_pred:.1f}s (Actual: {total_act:.1f}s)")
            print(f"Error (Absolute): {error_rate_squared:.2f}s")
            print(f"Error Tendency (Bias): {error_rate:.2f}s")
            
        
            

    print("\n--- Summary ---")
    print(f"Average Absolute Error per trip: {error_rate_squared:.2f} seconds")
    
    