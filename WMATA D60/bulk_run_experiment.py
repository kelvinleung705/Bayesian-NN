import os
import torch.multiprocessing as mp
# Import the function directly from the file without re-writing the code
from matrix_gnn_vae_ft2_student_t_WMATA_D60_train import train_single_run

def worker_task(config):
    """Unpacks configuration and calls the imported training function."""
    train_single_run(
        run_id=config["run_id"],
        train_file=config["train_file"],
        val_file=config["val_file"],
        output_dir=config["output_dir"],
        epochs=config.get("epochs", 4000),
        device_name=config.get("device", "cuda"),
    )

if __name__ == "__main__":
    # Required for PyTorch multiprocessing across CUDA/CPUs
    mp.set_start_method("spawn", force=True)

    # Define your different dataset runs and dedicated save directories
    tasks = [
        {
            "run_id": "Dataset_Group_Phrase_1",
            "train_file": "samples/train_A.csv",
            "val_file": "samples/val_A.csv",
            "output_dir": "results/experiment_Phrase_1",
            "epochs": 4000,
            "device": "cuda:0", # Change to "cpu" or another GPU ID if needed
        },
        {
            "run_id": "Dataset_Group_Phrase_2",
            "train_file": "samples/train_B.csv",
            "val_file": "samples/val_B.csv",
            "output_dir": "results/experiment_Phrase_2",
            "epochs": 4000,
            "device": "cuda:0",
        },
        {
            "run_id": "Dataset_Group_Phrase_3",
            "train_file": "samples/train_C.xlsx",
            "val_file": "samples/val_C.xlsx",
            "output_dir": "results/experiment_Phrase_3",
            "epochs": 4000,
            "device": "cuda:0",
        },
    ]

    # Set maximum simultaneous workers (e.g. 2 or 3 parallel runs)
    max_parallel_jobs = min(len(tasks), 3)

    print(f"Spawning {len(tasks)} runs using {max_parallel_jobs} workers...")

    with mp.Pool(processes=max_parallel_jobs) as pool:
        pool.map(worker_task, tasks)

    print("All parallel training experiments have finished.")