import torch

def diagnose_params(pt_file_path):
    print(f"Loading {pt_file_path}...")
    saved_params = torch.load(pt_file_path, map_location="cpu")
    
    if 'params' in saved_params:
        saved_keys = list(saved_params['params'].keys()) # If saved via custom method
    else:
        saved_keys = list(saved_params.keys()) # Standard Pyro save

    print(f"Total Parameters in .pt file: {len(saved_keys)}")
    
    # 1. Group the parameters by layer to see where the counts differ
    groups = {
        "embedding_layer.nets": 0,
        "prop_layers.0": 0,
        "prop_layers.1": 0,
        "prop_layers.2": 0,
        "heads_loc": 0,
        "heads_scale": 0,
        "heads_df": 0
    }
    
    print("\n--- Parameter Counts by Layer Group in .pt file ---")
    for key in saved_keys:
        for group in groups.keys():
            if group in key:
                groups[group] += 1
                
    for group, count in groups.items():
        print(f"  {group}: {count} params")
        
    print("\n--- Detailed Breakdown of Expected Plot.py Parameters ---")
    print("If your plot.py has biases as PyroSample, the expected counts are:")
    print("  embedding_layer.nets: 18 (9 weights, 9 biases)")
    print("  prop_layers.X: 38 (w_self, w_right, 9 nets_1 weights, 9 nets_1 biases, 9 nets_2 weights, 9 nets_2 biases)")
    print("  heads_X: 18 (9 weights, 9 biases)")
    
    print("\nCOMPARE THESE COUNTS to find the 18 missing parameters.")

if __name__ == "__main__":
    diagnose_params("ghost_bus_model_cycle_0.1_2000_df10_KL_Sample.pt")