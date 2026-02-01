import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyro.nn import PyroModule, PyroSample, PyroModuleList # Import this

# ==========================================
# 1. DATA PRE-PROCESSING (The "Slicer")
# ==========================================
def process_raw_data(file_path):
    """
    Input: Numpy array of shape [N, 38]
    Output: 
        x_global: [N, 12]
        x_local:  [N, 5, 4] (With log transform applied to feature 0)
        y_sections: [N, 5] (The travel time targets)
    """
    # --- 1. Read the Excel File ---
    # header=0: Means the FIRST row (Index 0) is the header/column names.
    # If "skip first row" means Row 0 is garbage and Row 1 is the header: use header=1
    # If the file has NO headers at all and Row 0 is garbage: use header=None, skiprows=1
    
    print(f"Reading {file_path}...")
    
    # Scenario: The first row (index 0) contains descriptions/garbage, 
    # and the actual data starts after that.
    df = pd.read_excel(file_path, header=None, skiprows=1)

    # --- 2. Select the 33 Columns (Indices 0 to 32) ---
    # We use iloc (Integer Location) to grab columns 0 through 33.
    # Note: 0:38 in Python excludes 33, so it gets 0-32.
    df_subset = df.iloc[:, 0:33]

    # --- 3. Clean the Data ---
    # Drop rows that contain ANY missing values (NaN) to prevent crashes
    df_subset = df_subset.dropna()
    
    # Ensure all data is numeric (force convert errors to NaN, then drop again)
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.dropna()

    # --- 4. Convert to NumPy Array ---
    raw_data_np = df_subset.values.astype(np.float32)
    
    print(f"Successfully loaded data shape: {raw_data_np.shape}")
    
    
    # Convert to Tensor
    data = torch.tensor(raw_data_np, dtype=torch.float32)
    
    #return raw_data_np

    
    
    # --- 1. Global Features (Indices 0-11) ---
    x_global = data[:, 0:12]
    
    # --- 2. Targets (Indices 12-16) ---
    # These are the actual travel times for the 4 sections
    y_sections = data[:, 12:16] 
    
    # (Index 16 is Total Time, we don't strictly need it for training 
    #  since we train on sections, but you can keep it for validation)
    
    # --- 3. Local Features (Indices 17-32) ---
    # Total 20 columns. We assume 5 sections * 4 features.
    raw_local = data[:, 17:33]
    
    # Reshape into [Batch, 5 Sections, 4 Features]
    x_local = raw_local.view(-1, 4, 4)
    
    # --- 4. Apply Logarithm Encoding ---
    # User Requirement: "N+0 need logarithm encoding"
    # We apply log1p (log(x+1)) to ensure stability if x is 0.
    # Clone to avoid in-place error
    x_local_processed = x_local.clone()
    
    
    
    return x_global, x_local, y_sections

# ==========================================
# 2. THE WATERFALL BNN MODEL
# ==========================================
class SectionalWaterfallBNN(PyroModule):
    def __init__(self, num_sections=4, global_dim=12, local_dim=4, hidden_dim=32):
        super().__init__()
        self.num_sections = num_sections
        
        # Input to each block: Global + Local + Accumulated_Time(1)
        input_dim = global_dim + local_dim + 1 
        
        # Define 5 UNIQUE Processing Blocks (one for each section)
        self.blocks = PyroModuleList([])
        self.heads = PyroModuleList([])

        for i in range(num_sections):
            # The Brain for this section
            block = PyroModule()
            block.linear1 = PyroModule[nn.Linear](input_dim, hidden_dim)
            #block.linear2 = PyroModule[nn.Linear](hidden_dim, hidden_dim)
            
            # --- SET PRIORS FOR BLOCK LAYERS (Full Bayesian) ---
            block.linear1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
            block.linear1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
            
            #block.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, hidden_dim]).to_event(2))
            #block.linear2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
            
            # 2. Define the Bayesian Head
            head = PyroModule[nn.Linear](hidden_dim, 3) 
            
            # --- SET PRIORS FOR HEAD (Full Bayesian) ---
            head.weight = PyroSample(dist.Normal(0., 1.).expand([3, hidden_dim]).to_event(2))
            head.bias = PyroSample(dist.Normal(0., 1.).expand([3]).to_event(1))
            
            self.blocks.append(block)
            self.heads.append(head)

    def forward(self, global_features, all_sections_data):
        batch_size = global_features.shape[0]
        # Initialize "Virtual Clock" at 0
        accumulated_time = torch.zeros(batch_size, 1).to(global_features.device)
        
        all_locs = []
        all_scales = []
        all_dfs = []

        # The Waterfall Loop
        for i in range(self.num_sections):
            # 1. Get local data for this section [Batch, 4]
            local_features = all_sections_data[:, i, :] 
            
            # 2. Injection: [Global | Local | Accumulated Time]
            x_input = torch.cat([global_features, local_features, accumulated_time], dim=1)
            
            # 3. Process
            h = self.blocks[i].linear1(x_input)
            h = torch.relu(h)
            raw_output = self.heads[i](h)
            
            # 4. Format Output Parameters
            loc_i = raw_output[:, 0].unsqueeze(1) # Mean time
            # Uncertainty must be positive
            scale_i = torch.nn.functional.softplus(raw_output[:, 1]).unsqueeze(1) + 1e-3 
            # DoF must be > 2.0
            df_i = torch.nn.functional.softplus(raw_output[:, 2]).unsqueeze(1) + 2.5 
            
            # 5. Store
            all_locs.append(loc_i)
            all_scales.append(scale_i)
            all_dfs.append(df_i)
            
            # 6. Propagate Clock (Add predicted time to accumulator)
            accumulated_time = accumulated_time + loc_i

        return all_locs, all_scales, all_dfs

# ==========================================
# 3. PROBABILISTIC DEFINITION (For Training)
# ==========================================
def model_fn(x_global, x_local, y_true=None):
    # Run network
    locs, scales, dfs = bnn_model(x_global, x_local)
    
    # Observe data
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
            #target = y_true[:, i].unsqueeze(1) if y_true is not None else None
            target = None
            if y_true is not None:
                target = y_true[:, i] # No unsqueeze needed!
            
            # Calculate Loss
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- A. Generate Dummy Data (Simulating your 0-37 index array) ---
    print("Generating dummy data...")
    N_SAMPLES = 500
    
    # Indices 0-11: Global
    raw_global = np.random.randn(N_SAMPLES, 12)
    # Indices 12-15: Targets (Section Times, e.g., around 5 mins)
    raw_targets = np.abs(np.random.randn(N_SAMPLES, 4) + 0.0)
    # Index 16: Total Time
    raw_total = np.sum(raw_targets, axis=1, keepdims=True)
    # Indices 17-32: Local Features (Assuming raw positive values for log transform)
    raw_local = np.abs(np.random.randn(N_SAMPLES, 20) * 10) 
    
    # Combine into one big [N, 38] array like your source
    full_raw_data = np.hstack([raw_global, raw_targets, raw_total, raw_local])
    file_path = "trip_info.xlsx"  # Using the array directly for this example
    
    # --- B. Preprocess ---
    print("Preprocessing...")
    x_g_all, x_l_all, y_all = process_raw_data(file_path)
    
    print(f"Global Input Shape: {x_g_all.shape}") # Should be [500, 12]
    print(f"Local Input Shape:  {x_l_all.shape}") # Should be [500, 5, 4]
    print(f"Target Shape:       {y_all.shape}")   # Should be [500, 5]
    
    # --- NEW: 分割數據 (Train/Val Split) ---
    idx = np.arange(x_g_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    
    
    # 訓練集
    x_g_train = x_g_all[train_idx]
    x_l_train = x_l_all[train_idx]
    y_train = y_all[train_idx]

    # 驗證集（這部分數據模型在訓練時從未見過）
    x_g_val = x_g_all[val_idx]
    x_l_val = x_l_all[val_idx]
    y_val = y_all[val_idx]

    print(f"訓練集樣本數: {len(train_idx)}, 驗證集樣本數: {len(val_idx)}")

    # --- C. Setup Model & Training ---
    bnn_model = SectionalWaterfallBNN(num_sections=4, global_dim=12, local_dim=4)
    guide = AutoNormal(model_fn)
    optimizer = pyro.optim.Adam({"lr": 0.01})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    pyro.clear_param_store()
    epochs = 200
    
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        loss = svi.step(x_g_train, x_l_train, y_train)
        pbar.set_description(f"Loss: {loss:.2f}")

    # --- D. Inference (Prediction) ---
    print("\n--- Final Prediction Test ---")
    # Take the first item to predict
    test_x_g = x_g_train[0:1]
    test_x_l = x_l_train[0:1]
    
    # Run Monte Carlo Sampling (50 times)
    predictive = Predictive(model_fn, guide=guide, num_samples=50)
    samples = predictive(test_x_g, test_x_l)
    
    # Calculate Total ETA
    #total_time_samples = torch.zeros(50, 1)
    total_time_samples = torch.zeros(50)
    
    print("Predicted Section Times:")
    for i in range(5):
        # Get samples for this section
        sec_samples = samples[f"obs_section_{i}"].squeeze()
        mean_t = sec_samples.mean().item()
        actual_t = y_train[0, i].item()
        print(f"  Section {i}: Pred {mean_t:.2f} | Actual {actual_t:.2f}")
        
        total_time_samples += sec_samples

    final_mean = total_time_samples.mean().item()
    final_std = total_time_samples.std().item()
    actual_total = y_train[0].sum().item()

    print(f"\nTotal ETA: {final_mean:.2f} mins (Actual: {actual_total:.2f})")
    print(f"Confidence: +/- {final_std:.2f} mins")