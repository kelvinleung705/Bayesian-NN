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

    end = 9 + num_segment + 1 + (num_segment * 4) + 2 #indication
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
    print("Total rows loaded:", raw_data_np.shape[0])
    print("Sample y_raw row 0:", y_raw[0])        # Should be travel times, not global features
    print("Sample x_local row 0:", raw_local[0])  # Should be local segment features
    print("y_raw mean per segment:", y_raw.mean(axis=0))
    print("y_raw std per segment:", y_raw.std(axis=0))
    # Return x_global, x_local, the normalized Y, and the Y scaler (for inverse transform later)
    return x_global, x_local, y_scaled, scaler_y


#==========================================
# 2. EMBEDDING LAYER (UNSHARED) LAYERS
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
            df = torch.tensor(10., device=device)
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
        zero = torch.tensor(0.0, device=device)
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
        
        # --- Layer 0: Input Projection (Unique per segment) ---
            
        self.embedding_layer = LocalIsolationLayer(input_dim, hidden_dim, num_sections, device)
        
        # --- Propagation Layers (The Matrix) ---
        # "Different function for each layer each section"
        # We create N layers. Each layer contains N unique networks.
        self.prop_layers = PyroModuleList([
            NeighborMixingLayer(hidden_dim, hidden_dim, num_sections, dropout_rate=0.2, device=device)
            for _ in range(2) # Layer depth = num_segments
        ])
        
        # --- Output Heads ---
        final_dim = hidden_dim
        # yuyu
        self.heads_loc = PyroModuleList([])
        self.heads_scale = PyroModuleList([])
        self.heads_df = PyroModuleList([])
        
        for i in range(self.num_sections):
            loc_std_dev = 2
            # Tensors for Heads
            
            #June
            """
            zero = torch.tensor(0., device=device)
            loc_std = torch.tensor(1.5, device=device)
            loc_bias_mu = torch.tensor(4.0, device=device)
            loc_bias_std = torch.tensor(1., device=device)

            scale_std = torch.tensor(0.1, device=device)
            scale_bias_mu = torch.tensor(-2., device=device)
            scale_bias_std = torch.tensor(0.5, device=device)
            """    
            
            # Mmy 2
            zero = torch.tensor(0., device=device)
            loc_std = torch.tensor(1.0, device=device)
            loc_bias_mu = torch.tensor(0., device=device)
            loc_bias_std = torch.tensor(1., device=device)

            scale_std = torch.tensor(0.3, device=device)
            scale_bias_mu = torch.tensor(0., device=device)
            scale_bias_std = torch.tensor(3.0, device=device)
                        
            three = torch.tensor(3., device=device)
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
        current_time = torch.zeros(batch_size, 1).to(device)
        
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
                    time_i = current_time
                elif i < current_section:
                    time_i = current_time
                else:
                    time_i = torch.zeros(batch_size, 1).to(device)
                
                #if time_i.abs().mean().item() > 15:
                    #time_i = torch.zeros(batch_size, 1).to(device)
                time_i = torch.clamp(time_i, min=-15.0, max=15.0)
                inp = torch.cat([global_features, loc_i, time_i], dim=1)
                inputs_list.append(inp)
                
            # 2. Run the full GNN (Embedding + Mixing) to get context
            h_current = self.embedding_layer(inputs_list)
            for layer in self.prop_layers:
                h_current = layer(h_current)
                
            
            # 3. Extract ONLY the prediction for the current_section
            final_feat = h_current[current_section] 
            
            loc = self.heads_loc[current_section](final_feat)
            scale = torch.nn.functional.softplus(self.heads_scale[current_section](final_feat)) + 1e-3 
            df = torch.nn.functional.softplus(self.heads_df[current_section](final_feat)) + 2.5
            
            # 4. Store the predictions
            all_locs.append(loc)
            all_scales.append(scale)
            all_dfs.append(df)
            # 5. UPDATE THE CLOCK for the next loop iteration
            # We add the predicted travel time of THIS section to the running total.
            
            current_time = current_time + loc
                
            
        return all_locs, all_scales, all_dfs
    
# ==========================================
# 4. EXECUTION
# ==========================================
def model_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0): #kl_weight=1
    with pyro.poutine.scale(scale=kl_weight):
        locs, scales, dfs = bnn_model(x_global, x_local)
    
    if total_size is None:
        total_size = x_global.shape[0]
    
    with pyro.plate("data", size=total_size, subsample_size=x_global.shape[0], dim=-1):
        # --- THE FIX FOR STUCK PREDICTIONS ---
        # We scale the Likelihood by 100.0.
        # This tells the model: "Fitting the data is 100x more important than the Prior."
        # Use kl_weight instead of 100.0!
        for i in range(len(locs)):
            dist_i = dist.StudentT(dfs[i].squeeze(), locs[i].squeeze(), scales[i].squeeze())
            target = y_true[:, i] if y_true is not None else None
            pyro.sample(f"obs_section_{i}", dist_i, obs=target)
            
def get_ll_kl(model_fn, guide, x_g, x_l, y, total_size):
    # Always use kl_weight=1.0 — we want true LL and true KL, not the annealed versions
    guide_trace = pyro.poutine.trace(guide).get_trace(
        x_g, x_l, y, total_size=total_size, kl_weight=1.0
    )
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model_fn, trace=guide_trace)
    ).get_trace(x_g, x_l, y, total_size=total_size, kl_weight=1.0)

    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()

    ll = 0.0
    kl = 0.0

    for name, site in model_trace.nodes.items():
        if site["type"] != "sample":
            continue
        if site["is_observed"]:
            ll += site["log_prob_sum"]
        else:
            if name not in guide_trace.nodes:
                continue
            log_p = site["log_prob_sum"]
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            kl += log_q - log_p

    return ll.item(), kl.item()

if __name__ == "__main__":
    import pyro
    from pyro.optim import PyroLRScheduler
    from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
    #from pyro.nn import AutoDiagonalNormal
    from pyro.infer.autoguide import AutoDiagonalNormal
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split

    # 1. SETUP DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # [Assuming process_raw_data and MatrixGNN are defined above]
    #file_path = "trip_info_9_section_ver2_simplify_ultra.xlsx"
    #file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_sorted.xlsx"
    file_path = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_new.xlsx"
    x_global_all, x_local_all, y_all, scaler_y = process_raw_data(file_path)
    
    idx = np.arange(x_global_all.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.00001, random_state=42)
    
    x_global_train = x_global_all[train_idx]
    x_local_train = x_local_all[train_idx]
    y_train = y_all[train_idx]
    
    x_global_val = x_global_all[val_idx].to(device)
    x_local_val = x_local_all[val_idx].to(device)
    y_val = y_all[val_idx].to(device)

    # 2. CLEAR PARAM STORE BEFORE INITIALIZATION
    pyro.clear_param_store()

    # Initialize The Matrix Model
    #14
    bnn_model = MatrixGNN(num_sections=num_segment, global_dim=9, local_dim=4, hidden_dim=32, device=device).to(device)
    
    # Define Guide (Posterior approximation)
    base_guide = AutoDiagonalNormal(model_fn).to(device)
    
    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
        with pyro.poutine.scale(scale=kl_weight):
            return base_guide(x_global, x_local, y_true, total_size=total_size, kl_weight=kl_weight)
    
    # 3. PYRO OPTIMIZER & SCHEDULER (Fixed)
    CYCLE_LENGTH = 1200  # Sync LR and KL cycles
    
    # Wrap PyTorch optimizer and scheduler the Pyro way
    optimizer_args = {
        "optimizer": torch.optim.AdamW,
        "optim_args": {"lr": 0.001, "weight_decay": 0.01}
    }
    
    def scheduler_constructor(optim):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, 
            T_0=CYCLE_LENGTH,  # Restart every 2500 epochs
            T_mult=1,          # Keep cycle length constant
            eta_min=0.0001     # Prevent LR from dropping to absolute zero
        )
    
    scheduler = PyroLRScheduler(scheduler_constructor, optimizer_args)

    # Initialize SVI with the wrapped scheduler
    svi = SVI(model_fn, guide_fn, scheduler, loss=Trace_ELBO())

    print("\n--- Starting Training ---")
    
    epochs = 4800
    batch_size = 734
    train_dataset = TensorDataset(x_global_train, x_local_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_size = len(train_dataset)
    print(f"Training dataset size: {total_size}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_ll = 0.0
        epoch_kl = 0.0
        
        # 4. PERFECTLY SYNCHRONIZED KL ANNEALING
        # This replaces the external Annealer to ensure exact syncing with the LR Restart
        relative_epoch = epoch % CYCLE_LENGTH
        ramp_epochs = 600
        down_epoch = 1200
        max_beta = 0.9
        if relative_epoch < ramp_epochs:
            current_kl_weight = max(0.00001, (relative_epoch / ramp_epochs)*max_beta)
        elif relative_epoch < down_epoch:
            current_kl_weight = max_beta
        #elif relative_epoch < zero_epoch:
        #    current_kl_weight = max(0.00001, max_beta * (1 - (relative_epoch - down_epoch) / (zero_epoch - down_epoch)))
        else:            
            current_kl_weight = 0.00001
            
        # Training Batch Loop
        for x_g_batch, x_l_batch, y_batch in train_loader:
            x_g_batch = x_g_batch.to(device)
            x_l_batch = x_l_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Step SVI
            loss = svi.step(x_g_batch, x_l_batch, y_batch, total_size=total_size, kl_weight=current_kl_weight)
            epoch_loss += loss
            
        # 5. CRITICAL: Step the scheduler at the epoch level!
        scheduler.step()
        
        

        # Print detailed logging occasionally
        if epoch % 1 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                ll, kl = get_ll_kl(
                    model_fn, guide_fn,
                    x_g_batch, x_l_batch, y_batch,  # use last batch as proxy, or full data
                    total_size=total_size,
                    #kl_weight=current_kl_weight
                )
            epoch_ll += ll
            epoch_kl += kl
            # Safely extract current LR from Pyro's internal dictionary
            current_lr = list(scheduler.optim_objs.values())[0].optimizer.param_groups[0]["lr"] if scheduler.optim_objs else 0.002
            avg_loss = epoch_loss / len(train_loader)
            if epoch == 0 or relative_epoch + 1 == CYCLE_LENGTH or relative_epoch + 1 == ramp_epochs or relative_epoch + 1 == down_epoch or epoch + 1 == epochs or relative_epoch == 0:
                print(f"Epoch {epoch:05d} | LR: {current_lr:.6f} | KL Wt: {current_kl_weight:.3f} | ELBO Loss: {avg_loss:.2f} | LL: {ll:.2f} | Btea KL: {kl:.2f}, | ELBO check: {ll - kl:.2f}, | Original KL: {epoch_kl/current_kl_weight:.2f}")
    
    # 6. SAVE MODEL & SCALER
    # Save parameters for the BNN weights
    pyro.get_param_store().save("ghost_bus_model_cycle_0.9_clamp_6_6_2133.pt")
    # Save the Y-Scaler to convert predictions back to seconds
    joblib.dump(scaler_y, "y_scaler_4_fixed_8000.pkl")
    print("\nModel weights and scaler saved successfully.")
        
        
        
    print("\n--- Final Prediction Test ---")
    # Switch model_fn to inference mode by capturing the posterior samples via Predictive
    #predictive = Predictive(model_fn, guide=guide, num_samples=50)
    #samples = predictive(x_global_val, x_local_val)
    
    # Calculate actuals vs predictions (Assuming 'obs' is your sample site in model_fn)
    # y_pred_mean = samples['obs'].mean(dim=0)
    #total_actual = y_val.sum(dim=1)
    #print("Done!")
    
    
    
# ==========================================
    # 5. INFERENCE 
    # ==========================================
    bnn_model.eval()
    base_guide.eval()
    pyro.get_param_store().save("ghost_bus_model_cycle_KL_9_accu4_fixed_8000_new_encoding.pt")
    print("\n--- Final Prediction Test ---")
    list_of_predict = []
    list_of_confidence = []
    list_of_actual = []
    list_of_predict_sections = [[] for i in range(num_segment)]
    list_of_confidence_sections = [[] for i in range(num_segment)]
    list_of_actual_sections = [[] for i in range(num_segment)]
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=50)

    
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
            
            print(f"  Sec {i}: Pred {pred_real[i]:.1f}s | Actual {actual_real[i]:.1f}s | Conf +/- {std_real[i]:.1f}s | Prediction Dev {section_prediction_std_deviation:.1f}s | Confidence Dev {section_confidence_std_deviation:.1f}s | | Actual Dev {section_actual_std_deviation:.1f}s | Within Bound? {'YES' if (pred_real[i] - std_real[i]) <= actual_real[i] <= (pred_real[i] + std_real[i]) else 'NO'}")
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
        list_of_confidence.append(total_std)
        if len(list_of_confidence) > 1:
            confidence_std_deviation = statistics.pvariance(list_of_confidence) ** 0.5
        else:
            confidence_std_deviation = 0.0
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
        print(f"\nPrediction Std Deviation: {prediction_std_deviation:.2f} , Confidence Std Deviation: {confidence_std_deviation:.2f} , Actual Std Deviation: {actual_std_deviation:.2f})")
        print(f"Error: {error_rate_squared}")
        print(f"Error Tendency: {error_rate}")
        print(f"\n")
    
        print(f"總共 {j + 1} 筆驗證資料中，有 {within_bound_count} 筆落在預測區間內。")
        print(f"平均 {num_segment} Section，有 {section_within_bound_counts/len(x_global_val)} section 落在預測區間內。")
        print(f"平均置信度指標: {number_of_ratio/len(x_global_val)}")
        
        