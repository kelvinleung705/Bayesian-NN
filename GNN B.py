import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal # <--- KEY CHANGE: Guide that learns variance

# --- 1. Synthetic Dataset (Same as before) ---
class SyntheticGraphDataset:
    def __init__(self, num_nodes=5, num_features=3, num_samples=1000):
        self.num_nodes = num_nodes
        self.num_features = num_features # Added this line
        self.edge_index = torch_geometric.utils.sort_edge_index(
            torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        )
        self.data_list = []
        for _ in range(num_samples):
            x = torch.randn(num_nodes, num_features) * 2
            y = (x.sum(dim=1) + torch.randn(num_nodes)) * 5 + torch.randn(num_nodes) * 3
            self.data_list.append(Data(x=x, edge_index=self.edge_index, y=y))

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

# --- 2. Full Bayesian GCN ---
class FullBayesianGCN(PyroModule):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super().__init__()

        # --- LAYER 1: Bayesian GCN ---
        # We wrap GCNConv in PyroModule
        self.conv1 = PyroModule[GCNConv](num_node_features, hidden_dim)

        # KEY STEP: We must manually replace the weights inside GCNConv with PyroSamples (Priors).
        # PyG GCNConv typically has a linear layer named 'lin' and a 'bias'.
        # We set a Standard Normal prior N(0, 1) for the weights.
        self.conv1.lin = PyroModule[nn.Linear](num_node_features, hidden_dim)
        self.conv1.lin.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, num_node_features]).to_event(2))
        self.conv1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))

        # --- LAYER 2: Bayesian GCN ---
        self.conv2 = PyroModule[GCNConv](hidden_dim, hidden_dim)
        self.conv2.lin = PyroModule[nn.Linear](hidden_dim, hidden_dim)
        self.conv2.lin.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, hidden_dim]).to_event(2))
        self.conv2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))

        # --- OUTPUT LAYERS: Bayesian Linear ---
        # Predict Loc (Mean)
        self.loc = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.loc.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.loc.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

        # Predict Scale (Uncertainty)
        self.scale = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.scale.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.scale.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

        # Predict DF (Tail Heaviness)
        self.df = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.df.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.df.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

    def forward(self, x, edge_index, y=None):
        # Forward pass is normal logic; Pyro handles the weight sampling behind the scenes
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Output Heads
        loc = self.loc(x).squeeze(-1)
        scale = torch.exp(self.scale(x)).squeeze(-1) + 1e-3 # Ensure positive
        df = torch.exp(self.df(x)).squeeze(-1) + 2.0        # Ensure > 2

        return loc, scale, df

    def model(self, x, edge_index, y=None):
        # 1. This triggers the sampling of ALL weights (GCN and Linear) from priors
        loc, scale, df = self.forward(x, edge_index)

        # 2. Observation
        with pyro.plate("nodes", x.shape[0]):
            pyro.sample("obs", dist.StudentT(df, loc, scale), obs=y)

# --- 3. Training & Prediction ---

def main():
    # Setup
    torch.manual_seed(42)  # Set seed for reproducibility
    dataset = SyntheticGraphDataset(num_nodes=5, num_samples=200)
    data = dataset[0]

    # --- CRITICAL STEP: NORMALIZE TARGETS ---
    # We record mean/std to reverse this later
    y_mean = data.y.mean()
    y_std = data.y.std()

    # Normalize y to be roughly Mean 0, Std 1
    # This makes the Bayesian Prior (which is 0) compatible with the Data
    normalized_y = (data.y - y_mean) / y_std

    # Instantiate
    model = FullBayesianGCN(dataset.num_features, 16, 1)

    # --- KEY CHANGE: THE GUIDE ---
    # AutoDelta only learns a "best guess" for weights (Point estimate).
    # AutoDiagonalNormal learns a Mean AND StdDev for every weight (Full Bayesian).
    # This means the model learns "how unsure it is about its own weights".
    guide = AutoDiagonalNormal(model)

    # Optimizer
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())

    print("Training Full Bayesian GNN...")
    pyro.clear_param_store()

    # Training Loop (Needs more epochs usually because it's harder to learn)
    for epoch in range(1500):
        loss = svi.step(data.x, data.edge_index, data.y)
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.2f}")

    # --- PREDICTION ---
    print("\nPredicting...")
    # Because weights are distributions, every time we predict, we get a different network!
    predictive = Predictive(model.model, guide=guide, num_samples=500)

    with torch.no_grad():
        samples = predictive(data.x, data.edge_index)

    # The raw samples are in "Normalized Space"
    raw_preds = samples['obs'].squeeze()  # Shape [500, 5]

    # --- CRITICAL STEP: UN-NORMALIZE PREDICTIONS ---
    # Convert back to real world values
    final_preds = raw_preds * y_std + y_mean

    # Analysis for Node 0
    node_idx = 0
    true_val = data.y[node_idx].item()
    mean_pred = final_preds[:, node_idx].mean().item()
    std_pred = final_preds[:, node_idx].std().item()

    print(f"\n--- Results for Node {node_idx} ---")
    print(f"True Value:       {true_val:.4f}")
    print(f"Prediction Mean:  {mean_pred:.4f}")
    print(f"Uncertainty (Std): {std_pred:.4f}")

    # Why is this "Full Bayesian"?
    # Because there are two sources of uncertainty now:
    # 1. Aleatoric: The noise in the data (Student T scale output)
    # 2. Epistemic: The uncertainty in the GCN weights themselves

if __name__ == "__main__":
    main()