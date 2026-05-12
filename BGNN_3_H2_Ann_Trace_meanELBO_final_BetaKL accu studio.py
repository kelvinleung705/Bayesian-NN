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
            point_one = torch.tensor(1, device=device)
            df = torch.tensor(15., device=device)
            net.weight = PyroSample(dist.StudentT(df, zero, point_one).expand([output_dim, input_dim]).to_event(2))
            net.bias = PyroSample(dist.StudentT(df, zero, point_one).expand([output_dim]).to_event(1))
            self.nets.append(net)
            
    def forward(self, x_inputs):
        # x_inputs is a list of tensors, one per segment
        outputs = []
        all_sampled_weights = []
        for i in range(self.num_segments):
            out = torch.nn.functional.silu(self.nets[i](x_inputs[i]))
            outputs.append(out)
            all_sampled_weights.append(self.nets[i].weight.abs().mean())
        avg_weight = torch.stack(all_sampled_weights).mean()
        w_std = self.nets[0].weight.std()
        #print(f"Global Sampled Weight Mean: {avg_weight.item():.4f} | Spread (Std): {w_std.item():.4f}")
        return outputs
    
    
# ==========================================
# 2. UNIQUE MIXING LAYER (With Dropout & Weights)
# ==========================================
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

            df_std = torch.tensor(1., device=device)
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
        #current_time = torch.zeros(batch_size, 1).to(device)
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
                if self.pnt_1 is None:
                    self.pnt_1 = True
                    #print(current_time.abs().mean().item(), current_time)
                
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

# ==========================================
# 4. EXECUTION
# ==========================================
# ==========================================
# 4. EXECUTION & MODEL FUNCTION
# ==========================================

# CRITICAL FIX 1: We declare bnn_model as global so model_fn can see it
bnn_model = None

def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0): 
    # CRITICAL FIX 2: Do NOT scale the model call! Let the priors remain N(0,1)
    locs, scales, dfs = bnn_model(x_global, x_local)
    
    if total_size is None:
        total_size = x_global.shape[0]
    
    # CRITICAL FIX 3: THE INVERSE SCALE TRICK
    # By scaling the data up, we force the network to prioritize the NLL error 
    # without destroying the gradients of the Bayesian weights.
    data_scale = 1.0 / kl_weight
    
    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0], dim=-1):
        for i in range(len(locs)):
            dist_i = dist.StudentT(dfs[i].squeeze(-1), locs[i].squeeze(-1), scales[i].squeeze(-1))
            target = y_true[:, i] if y_true is not None else None
            
            # Apply scaling ONLY to the observation likelihood
            with pyro.poutine.scale(scale=data_scale):
                pyro.sample(f"obs_section_{i}", dist_i, obs=target)


def get_ll_kl(model_fn, guide, x_g, x_l, y, total_size):
    # Pass kl_weight=1.0 to get the unscaled, true mathematical values for logging
    guide_trace = pyro.poutine.trace(guide).get_trace(x_g, x_l, y_true=y, total_size=total_size, kl_weight=1.0)
    
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model_fn, trace=guide_trace)
    ).get_trace(x_g, x_l, y_true=y, total_size=total_size, kl_weight=1.0)
    
    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()
    
    ll = 0.0
    kl = 0.0
    for name, site in model_trace.nodes.items():
        if site["type"] != "sample": continue
        if site["is_observed"]:
            ll += site["log_prob_sum"]
        else:
            if name not in guide_trace.nodes: continue
            log_p = site["log_prob_sum"]
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            kl += log_q - log_p  
    
    return ll.item(), kl.item()

# ==========================================
# 5. MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    import pyro
    from pyro.optim import PyroLRScheduler
    from pyro.infer import SVI, TraceMeanField_ELBO, Predictive, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    import joblib
    import statistics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_sorted.xlsx"
    x_global_all, x_local_all, y_all, scaler_y = process_raw_data(file_path)
    
    # Chronological split is better, but random is okay for now
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    x_global_train, x_local_train, y_train = x_global_all[train_idx], x_local_all[train_idx], y_all[train_idx]
    x_global_val, x_local_val, y_val = x_global_all[val_idx].to(device), x_local_all[val_idx].to(device), y_all[val_idx].to(device)

    pyro.clear_param_store()

    # CRITICAL FIX 4: Assign to the global variable so model_fn can use it
    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32, device=device).to(device)
    
    # CRITICAL FIX 5: Use the standard guide. Delete guide_fn entirely.
    # init_scale=0.1 helps the model escape the prior faster
    base_guide = AutoDiagonalNormal(model_fn, init_scale=0.1).to(device)
 
    CYCLE_LENGTH = 1000  
    
    def per_param_callable(module_name, param_name):
        if "scale" in param_name:
            return {"lr": 0.001, "weight_decay": 0.0}   
        return {"lr": 0.0002, "weight_decay": 0.01}      

    optimizer_args = {
        "optimizer": torch.optim.AdamW,
        "optim_args": per_param_callable
    }
    
    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=CYCLE_LENGTH, T_mult=1, eta_min=0.0001
        )
    
    scheduler = PyroLRScheduler(scheduler_constructor, optimizer_args)

    # Initialize SVI directly with the standard guide
    svi = SVI(model_fn, base_guide, scheduler, loss=TraceMeanField_ELBO())

    print("\n--- Starting Training ---")
    
    epochs = 4000
    batch_size = 1024
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_size = len(train_dataset)
    print(f"Training dataset size: {total_size}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        relative_epoch = epoch % CYCLE_LENGTH
        ramp_epochs = 500  
        max_beta = 1.0  # Set to 1.0 to fully balance KL and NLL by the end of the ramp
        
        if relative_epoch < ramp_epochs:
            current_kl_weight = max(0.00001, (relative_epoch / ramp_epochs)*max_beta)
        else:
            current_kl_weight = max_beta 
            
        for x_g_batch, x_l_batch, y_batch in train_loader:
            x_g_batch, x_l_batch, y_batch = x_g_batch.to(device), x_l_batch.to(device), y_batch.to(device)
            
            # Step SVI
            raw_loss = svi.step(x_g_batch, x_l_batch, y_batch, total_size=total_size, kl_weight=current_kl_weight)
            
            # CRITICAL FIX 6: Restore the loss scale for accurate logging
            actual_loss = raw_loss * current_kl_weight
            epoch_loss += actual_loss
            
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            current_lr = list(scheduler.optim_objs.values())[0].optimizer.param_groups[0]["lr"] if scheduler.optim_objs else 0.0002
            avg_loss = epoch_loss / len(train_loader)
            
            with torch.no_grad():
                # We pass the standard base_guide here
                ll, kl = get_ll_kl(model_fn, base_guide, x_global_val, x_local_val, y_val, total_size=total_size)
                ratio = abs(kl) / (abs(ll) + 1e-8)
                print(f"Epoch {epoch:05d} | LR: {current_lr:.6f} | KL Wt: {current_kl_weight:.3f} | ELBO Loss: {avg_loss:.2f} | LL: {ll:.2f} | KL: {kl:.2f} | KL/LL: {ratio:.4f}")

    # ==========================================
    # 5. INFERENCE & EVALUATION
    # ==========================================
    print("\n--- Saving Model & Starting Inference ---")
    pyro.get_param_store().save("ghost_bus_model_params_studio.pt")
    joblib.dump(scaler_y, "y_scaler_studio.pkl")
    
    bnn_model.eval()
    base_guide.eval()
    
    # We use the standard base_guide here, no hacks
    predictive = Predictive(model_fn, guide=base_guide, num_samples=50)

    within_bound_count = 0
    section_within_bound_counts = 0 
    error_abs_total = 0
    error_total = 0
    
    list_of_predict = []
    list_of_confidence = []
    list_of_actual = []
    
    print("Running Monte Carlo Sampling on Validation Set...")
    
    # Batch processing is much faster and safer than row-by-row
    for j in range(len(x_global_val)):
        val_x_g = x_global_val[j:j+1]
        val_x_l = x_local_val[j:j+1]
        
        with torch.no_grad():
            samples = predictive(val_x_g, val_x_l)
        
        pred_means_scaled = []
        pred_stds_scaled = []
        actuals_scaled = []
        
        for i in range(num_segment):
            sec_samples = samples[f"obs_section_{i}"].squeeze()
            pred_means_scaled.append(sec_samples.mean().item())
            pred_stds_scaled.append(sec_samples.std().item())
            actuals_scaled.append(y_val[j, i].item())
            
        pred_real = scaler_y.inverse_transform([pred_means_scaled])[0]
        actual_real = scaler_y.inverse_transform([actuals_scaled])[0]
        std_real = np.array(pred_stds_scaled) * scaler_y.scale_
        
        total_pred = pred_real.sum()
        total_act = actual_real.sum()
        # Correctly aggregate independent standard deviations (sqrt of sum of variances)
        total_std = np.sqrt(np.sum(std_real**2)) 
        
        if (total_pred - total_std) <= total_act <= (total_pred + total_std):
            within_bound_count += 1
            
        for i in range(num_segment):
             if (pred_real[i] - std_real[i]) <= actual_real[i] <= (pred_real[i] + std_real[i]):
                section_within_bound_counts += 1
                
        list_of_predict.append(total_pred)
        list_of_confidence.append(total_std)
        list_of_actual.append(total_act)
            
        error_total += (total_act - total_pred)
        error_abs_total += abs(total_act - total_pred)

        # Print detailed logs for a few samples to avoid console spam
        if j % 100 == 0:
            print(f"\n--- Sample {j} ---")
            print(f"Total ETA: {total_pred:.2f}s (Actual: {total_act:.2f}s)")
            print(f"Confidence: +/- {total_std:.2f}s")
            print(f"Within Bound? : {'YES' if (total_pred - total_std) <= total_act <= (total_pred + total_std) else 'NO'}")
            print(f"Current MAE: {error_abs_total/(j+1):.2f}s | Bias: {error_total/(j+1):.2f}s")
    
    print("\n" + "="*40)
    print("FINAL VALIDATION SUMMARY")
    print("="*40)
    print(f"Total Validated Trips: {len(x_global_val)}")
    print(f"Total Coverage (Trip within 1 StdDev): {(within_bound_count/len(x_global_val))*100:.2f}%")
    print(f"Average Section Coverage: {(section_within_bound_counts/(len(x_global_val)*num_segment))*100:.2f}%")
    print(f"Mean Absolute Error (MAE): {error_abs_total/len(x_global_val):.2f} seconds")
    print(f"Overall Bias (Error Tendency): {error_total/len(x_global_val):.2f} seconds")
    print(f"Prediction Diversity (StdDev): {statistics.pstdev(list_of_predict):.2f}s")
    print(f"Actual Diversity (StdDev): {statistics.pstdev(list_of_actual):.2f}s")
    print("="*40)