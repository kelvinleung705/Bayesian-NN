import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal


# --- 1. Synthetic Dataset ---
class SyntheticGraphDataset:
    def __init__(self, num_nodes=500, num_features=3):
        self.num_nodes = num_nodes
        self.num_features = num_features

        # Random connections
        self.edge_index = torch_geometric.utils.erdos_renyi_graph(num_nodes, edge_prob=0.1)
        self.x = torch.randn(num_nodes, num_features)
        # Static edge index for a small graph
        self.edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                                        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)

        # Generate random features
        self.x = torch.randn(num_nodes, num_features)

        # Generate targets with a known rule: 5 * sum(features) + noise + offset
        # This will produce values roughly in the 5 to 25 range
        self.y = (self.x.sum(dim=1) * 5.0) + 15.0 + (torch.randn(num_nodes) * 2.0)

    def get_data(self):
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)


# --- 2. Full Bayesian GCN ---
class FullBayesianGCN(PyroModule):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super().__init__()

        # Bayesian GCN Layer 1
        self.conv1 = PyroModule[GCNConv](num_node_features, hidden_dim)
        self.conv1.lin = PyroModule[nn.Linear](num_node_features, hidden_dim)
        prior_std = 1.0
        self.conv1.lin.weight = PyroSample(dist.Normal(0., prior_std).expand([hidden_dim, num_node_features]).to_event(2))
        self.conv1.lin.bias = PyroSample(dist.Normal(0., prior_std).expand([hidden_dim]).to_event(1))

        # Bayesian GCN Layer 2
        self.conv2 = PyroModule[GCNConv](hidden_dim, hidden_dim)
        self.conv2.lin = PyroModule[nn.Linear](hidden_dim, hidden_dim)
        self.conv2.lin.weight = PyroSample(dist.Normal(0., prior_std).expand([hidden_dim, hidden_dim]).to_event(2))
        self.conv2.lin.bias = PyroSample(dist.Normal(0., prior_std).expand([hidden_dim]).to_event(1))

        # Head for Mean (loc)
        self.loc_head = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.loc_head.weight = PyroSample(dist.Normal(0., prior_std).expand([output_dim, hidden_dim]).to_event(2))
        self.loc_head.bias = PyroSample(dist.Normal(0., prior_std).expand([output_dim]).to_event(1))

        # Head for Uncertainty (scale)
        self.scale_head = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.scale_head.weight = PyroSample(dist.Normal(0., prior_std).expand([output_dim, hidden_dim]).to_event(2))
        self.scale_head.bias = PyroSample(dist.Normal(0., prior_std).expand([output_dim]).to_event(1))

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index, y=None):
        # relu
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        loc = self.loc_head(x).squeeze(-1)
        # Softplus ensures the uncertainty is ALWAYS positive and starts small
        scale = torch.nn.functional.softplus(self.scale_head(x)).squeeze(-1) + 1e-3

        return loc, scale

    def model(self, x, edge_index, y=None):
        loc, scale = self.forward(x, edge_index)
        with pyro.plate("nodes", x.shape[0]):
            pyro.sample("obs", dist.Normal(loc, scale), obs=y)


# --- 3. Execution ---
def main():
    torch.manual_seed(42)
    pyro.set_rng_seed(42)

    # 1. Create Data
    dataset = SyntheticGraphDataset(num_nodes=500, num_features=3)
    data = dataset.get_data()

    # --- THE CRITICAL FIX: NORMALIZATION ---
    y_mean = data.y.mean()
    y_std = data.y.std()
    y_norm = (data.y - y_mean) / y_std  # Scale to mean 0, std 1
    # ---------------------------------------

    # 2. Setup Model
    model = FullBayesianGCN(num_node_features=3, hidden_dim=64)
    guide = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": 0.005})  # Higher LR is fine now that data is normalized
    svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())

    # 3. Training
    pyro.clear_param_store()
    print("Training Full Bayesian GNN...")
    for epoch in range(10000):
        loss = svi.step(data.x, data.edge_index, y_norm)  # Train on NORMALIZED data
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.2f}")

    # 4. Prediction
    print("\nPredicting...")
    predictive = Predictive(model.model, guide=guide, num_samples=1000)
    with torch.no_grad():
        # Get samples in normalized space
        preds_norm = predictive(data.x, data.edge_index)['obs']

        # --- THE CRITICAL FIX: UN-NORMALIZATION ---
        # Convert back to original scale (15.0 range)
        preds = (preds_norm * y_std) + y_mean

    # 5. Results
    node_idx = 0
    true_val = data.y[node_idx].item()
    mean_pred = preds[:, node_idx].mean().item()
    std_pred = preds[:, node_idx].std().item()

    print(f"\n--- Results for Node {node_idx} ---")
    print(f"True Value:       {true_val:.4f}")
    print(f"Prediction Mean:  {mean_pred:.4f}")
    print(f"Uncertainty (Std): {std_pred:.4f}")
    print(f"Absolute Error:   {abs(true_val - mean_pred):.4f}")


if __name__ == "__main__":
    main()