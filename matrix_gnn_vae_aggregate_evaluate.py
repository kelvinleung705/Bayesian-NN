import statistics
import numpy as np
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
import joblib

from matrix_gnn_vae_aggregate import (
    MatrixGNN_VAE,
    model_fn,
    process_raw_data,
    num_segment,
    LATENT_DIM,
)

# ==========================================
# CONFIG
# ==========================================
PARAMS_PATH  = "ghost_bus_vae_aggregate_0.5.pt"
SCALER_PATH  = "y_scaler_vae_aggregate.pkl"
DATA_PATH    = "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_June.xlsx"
HIDDEN_DIM   = 64
MC_SAMPLES   = 200   # Changed to 200 MC Samples
INFER_BATCH  = 512   # tune up/down depending on your VRAM


# ==========================================
# LOAD MODEL
# ==========================================

def load_model(params_path, device):
    import matrix_gnn_vae_aggregate as _m
    pyro.clear_param_store()

    bnn_model = MatrixGNN_VAE(
        num_sections=num_segment,
        global_dim=9,
        local_dim=4,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        device=device,
    ).to(device)
    _m.bnn_model = bnn_model

    base_guide = AutoDiagonalNormal(model_fn).to(device)

    def guide_fn(x_global, x_local, y_true=None, total_size=None, kl_weight=1.0):
        pass

    WARMUP_BATCH = 734
    TOTAL_SIZE   = 7332
    dummy_g = torch.zeros(WARMUP_BATCH, 9,             device=device)
    dummy_l = torch.zeros(WARMUP_BATCH, num_segment, 4, device=device)
    dummy_y = torch.zeros(WARMUP_BATCH,   device=device)

    optimizer = pyro.optim.Adam({"lr": 0.001})
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi.step(dummy_g, dummy_l, dummy_y,
             total_size=TOTAL_SIZE, kl_weight=0.00001)

    warmup_n = sum(v.numel() for v in pyro.get_param_store().values())
    print(f"Warm-up params: {warmup_n}")

    #pyro.get_param_store().load(params_path, map_location=device)
    bnn_model.load_state_dict(torch.load(params_path, map_location=device))
    loaded_n = sum(v.numel() for v in pyro.get_param_store().values())
    print(f"Checkpoint params: {loaded_n}")

    if warmup_n != loaded_n:
        raise RuntimeError(f"Param mismatch: warm-up={warmup_n} vs checkpoint={loaded_n}.")

    bnn_model.eval()
    return bnn_model


def enable_mc_dropout(model):
    """Turns Dropout back on for MC Uncertainty estimation"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


# ==========================================
# FAST BATCHED INFERENCE (WITH 200 MC SAMPLES)
# ==========================================
def enable_mc_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


def run_inference_fast(x_global_val, x_local_val, y_val, bnn_model, scaler_y, device):
    import time
    n = x_global_val.shape[0]

    # ── Output buffers
    pred_means = np.zeros(n, dtype=np.float32)
    pred_stds  = np.zeros(n, dtype=np.float32)
    actuals    = np.zeros(n, dtype=np.float32)

    print(f"\nRunning batched inference  "
          f"(n={n}, MC={MC_SAMPLES}, batch={INFER_BATCH})...")
    t0 = time.time()

    # Prepare model for MC Dropout Inference
    bnn_model.eval()
    enable_mc_dropout(bnn_model)
    # ── Single pass over all data in large batches
    with torch.no_grad():
        for start in range(0, n, INFER_BATCH):
            end     = min(start + INFER_BATCH, n)
            batch_g = x_global_val[start:end]   # [B, 9]
            batch_l = x_local_val[start:end]    # [B, 9, 4]
            
            mc_predictions = []
            mc_aleatorics = []
            
            # RUN THE BATCH 200 TIMES
            for _ in range(MC_SAMPLES):
                # force_sample=True forces VAE to sample the latent space
                locs, scalers, kl = bnn_model(batch_g, batch_l, force_sample=True)
                
                # Squeeze to shape [B]
                mc_predictions.append(locs.squeeze(-1))
                mc_aleatorics.append(scalers.squeeze(-1))

            # Stack into tensors: [MC_SAMPLES, Batch_Size]
            mc_predictions = torch.stack(mc_predictions)
            mc_aleatorics = torch.stack(mc_aleatorics)
            
            # --- CALCULATE FINAL PREDICTION AND UNCERTAINTY ---
            # 1. Final ETA Prediction (Mean)
            final_eta = mc_predictions.mean(dim=0)
            
            # 2. Epistemic Uncertainty (Spread of the 200 runs)
            epistemic_var = mc_predictions.var(dim=0)
            
            # 3. Aleatoric Uncertainty (Mean of the model's internal scale squared)
            aleatoric_var = (mc_aleatorics ** 2).mean(dim=0)
            
            # 4. Total Uncertainty
            total_std = torch.sqrt(epistemic_var + aleatoric_var)

            # Store in numpy buffers
            pred_means[start:end] = final_eta.cpu().numpy()
            pred_stds[start:end]  = total_std.cpu().numpy()
            actuals[start:end]    = y_val[start:end].cpu().numpy()

            elapsed = time.time() - t0
            rate    = (end) / elapsed
            eta     = (n - end) / rate if rate > 0 else 0
            print(f"  {end:5d}/{n}  |  {rate:.0f} trips/s  |  ETA {eta:.1f}s", end='\r', flush=True)

    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed:.1f}s  ({n/elapsed:.0f} trips/s)")

    # ── Inverse transform all at once
    pred_real   = scaler_y.inverse_transform(pred_means.reshape(-1, 1)).squeeze(-1)   # [n]
    actual_real = scaler_y.inverse_transform(actuals.reshape(-1, 1)).squeeze(-1)      # [n]
    
    # Scale standard deviation (scale_ is the standard deviation from the scaler)
    std_real    = pred_stds * scaler_y.scale_[0]                                      # [n]

    # ── Compute stats
    _print_results(pred_real, actual_real, std_real, n)
    
    
        # After your inference loop, run this:
    with torch.no_grad():
        test_g = x_global_val[:5]
        test_l = x_local_val[:5]
    
        for i in range(5):
            loc, scale, kl_val = bnn_model(test_g[i:i+1], test_l[i:i+1], force_sample=False)
            print(f"Sample {i}: loc={loc.item():.4f}, scale={scale.item():.4f}")
    
        # Also check z directly from the encoder
        from matrix_gnn_vae_aggregate import gaussian_reparameterise
        z_samples = []
        for _ in range(10):
            # You'll need to expose z from the encoder, e.g. bnn_model.encoder(test_g, test_l)
            mu, logvar = bnn_model.encoder(test_g, test_l)
            z, _ = gaussian_reparameterise(mu, logvar, force_sample=True)
            z_samples.append(z)
        z_stack = torch.stack(z_samples)
        print(f"\nz variance across MC calls: {z_stack.var(0).mean().item():.6f}")
        print("Model submodules:", list(bnn_model._modules.keys()))


def _print_results(pred_real, actual_real, std_real, n):
    """Vectorised stats + per-sample print."""
    total_pred = pred_real    
    total_act  = actual_real  
    total_std  = std_real     # No need to sqrt/square again, it's already total std

    trip_within = ((total_act >= total_pred - total_std) &
                   (total_act <= total_pred + total_std))  # [n] bool
    
    pred_running_std = [np.std(total_pred[: j + 1]) for j in range(n)]
    conf_running_std = [np.std(total_std[: j + 1]) for j in range(n)]
    act_running_std = [np.std(total_act[: j + 1]) for j in range(n)]

    for j in range(n):
        print(f"\n--- Sample {j} ---")
        print(f"\n  Total ETA : {total_pred[j]:.1f}s  (Actual: {total_act[j]:.1f}s)")
        print(f"  Conf      : ±{total_std[j]:.1f}s  |  Within? {'YES' if trip_within[j] else 'NO'}")
        print(f"  MAE so far: {np.abs(total_pred[:j+1] - total_act[:j+1]).mean():.1f}s  |  "
              f"Bias: {(total_act[:j+1] - total_pred[:j+1]).mean():.1f}s")
        print(f"\nPrediction Std Deviation: {pred_running_std[j]:.2f} , "
              f"Confidence Std Deviation: {conf_running_std[j]:.2f} , "
              f"Actual Std Deviation: {act_running_std[j]:.2f}")        
        within_so_far = trip_within[:j+1].sum()
        print(f"  總共 {j+1} 筆，{within_so_far} 筆落在區間內")

    mae  = np.abs(total_pred - total_act).mean()
    bias = (total_act - total_pred).mean()
    trip_cov = trip_within.mean() * 100

    print(f"\n{'='*60}")
    print(f"SUMMARY  ({n} trips, {MC_SAMPLES} MC samples)")
    print(f"{'='*60}")
    print(f"  Trip-level within CI : {trip_within.sum()}/{n}  ({trip_cov:.1f}%)")
    print(f"  MAE                  : {mae:.2f}s")
    print(f"  Bias                 : {bias:.2f}s")
    print(f"  CI ratio (mean p/σ)  : {(total_pred / np.where(total_std>0, total_std, 1)).mean():.3f}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params",  default=PARAMS_PATH)
    parser.add_argument("--scaler",  default=SCALER_PATH)
    parser.add_argument("--data",    default=DATA_PATH)
    parser.add_argument("--batch",   type=int, default=INFER_BATCH,
                        help="Inference batch size (default 512, reduce if OOM)")
    parser.add_argument("--mc",      type=int, default=MC_SAMPLES,
                        help="MC posterior samples (default 200)")
    parser.add_argument("--cpu",     action="store_true")
    args = parser.parse_args()

    INFER_BATCH = args.batch
    MC_SAMPLES  = args.mc

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    scaler_y = joblib.load(args.scaler)
    print(f"Scaler: '{args.scaler}'")

    x_global_all, x_local_all, y_all, _ = process_raw_data(args.data)
    x_global_val = x_global_all.to(device)
    x_local_val  = x_local_all.to(device)
    y_val        = y_all.to(device)

    # 1. Load Model
    bnn_model = load_model(args.params, device)
    
    import matrix_gnn_vae_aggregate as _m
    _m.bnn_model = bnn_model

    # 2. Run Inference
    run_inference_fast(x_global_val, x_local_val, y_val, bnn_model, scaler_y, device)
    
    
