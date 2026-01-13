import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


# --- 1. Define a Synthetic Graph Dataset ---
# In your real project, this would come from your processed bus route data.
# Let's imagine a small portion of your bus route as a graph.
# Nodes could be stops/intersections, edges are road segments.
class SyntheticGraphDataset:
    def __init__(self, num_nodes=5, num_features=3, num_samples=100):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_samples = num_samples

        # Create a simple linear graph (like a bus route segment)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        # Sort edge_index to avoid warnings in PyG
        self.edge_index = torch_geometric.utils.sort_edge_index(edge_index)

        self.data_list = []
        for _ in range(num_samples):
            # Node features: e.g., [time_of_day_sin, time_of_day_cos, day_of_week]
            # Replicate for all nodes for simplicity, or make node-specific
            x = torch.randn(num_nodes, num_features) * 2

            # Target (e.g., future travel time for each node/segment)
            # Make the target somewhat dependent on node features and random noise
            y_true_base = (x.sum(dim=1) + torch.randn(num_nodes)) * 5

            # For demonstration, let's make it a graph-level target (e.g., total segment time)
            # or a node-level target (e.g., expected delay at this node).
            # Let's stick with node-level prediction: `y` is travel time from this node to the next.
            y = y_true_base + torch.randn(num_nodes) * 3  # Add some noise

            self.data_list.append(Data(x=x, edge_index=self.edge_index, y=y))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_list[idx]


# --- 2. Define the Bayesian GNN Model ---
# This model outputs the parameters for a t-distribution.
class BayesianGCN(PyroModule):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # The output layer is Bayesian to predict the parameters of the t-distribution
        # We predict 3 parameters: loc (mean), log_scale, and log_df (log_degrees_of_freedom)
        # Using PyroSample to define priors over weights
        self.loc_layer = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.loc_layer.weight = PyroSample(dist.Normal(0, 1).expand([output_dim, hidden_dim]).to_event(2))
        self.loc_layer.bias = PyroSample(dist.Normal(0, 1).expand([output_dim]).to_event(1))

        self.scale_layer = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.scale_layer.weight = PyroSample(dist.Normal(0, 1).expand([output_dim, hidden_dim]).to_event(2))
        self.scale_layer.bias = PyroSample(dist.Normal(0, 1).expand([output_dim]).to_event(1))

        # Degrees of freedom must be > 0. We'll predict log(df) and then exp()
        self.df_layer = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.df_layer.weight = PyroSample(dist.Normal(0, 1).expand([output_dim, hidden_dim]).to_event(2))
        self.df_layer.bias = PyroSample(dist.Normal(0, 1).expand([output_dim]).to_event(1))

    def forward(self, x, edge_index):
        # Apply GNN layers
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)  # Output is node embeddings

        # Predict parameters of the t-distribution for each node
        loc = self.loc_layer(x).squeeze(-1)  # Mean of t-distribution
        # Scale must be positive. Use exp()
        scale = torch.exp(self.scale_layer(x)).squeeze(-1)  # Scale of t-distribution
        # Degrees of freedom must be positive. Use exp() and ensure a minimum (e.g., 2 for defined variance)
        df = torch.exp(self.df_layer(x)).squeeze(
            -1) + 2.0  # Degrees of freedom (v > 0, often v > 2 for defined variance)

        return loc, scale, df

    # Pyro model: Defines the generative process and priors
    def model(self, x, edge_index, y=None):
        # Sample GNN weights from their priors (defined in __init__ using PyroSample)
        # The forward method samples them when called within a Pyro context
        loc, scale, df = self.forward(x, edge_index)

        # Observe the true labels (y) distributed according to a Student's t-distribution
        with pyro.plate("nodes", x.shape[0]):  # A plate for each node/prediction
            return pyro.sample("obs", dist.StudentT(df, loc, scale), obs=y)

    # Pyro guide: Defines the approximate posterior
    def guide(self, x, edge_index, y=None):
        # Uses AutoDelta (mean-field approximation) for simplicity,
        # where each parameter's posterior is approximated by a delta function (point estimate).
        # More complex guides use AutoNormal (Gaussian)
        pass  # AutoDelta is implicit with PyroModule if using SVI.


# --- 3. Training Function ---
def train_bayesian_model(model, dataset, num_epochs=200, learning_rate=0.01):
    optimizer = Adam({"lr": learning_rate})
    svi = SVI(model.model, AutoDelta(model), optimizer, loss=Trace_ELBO())

    pyro.clear_param_store()

    # Get a single graph from the dataset for training (assuming batch_size=1 for simplicity)
    # For GNNs, you often train on a single large graph or use batched graphs
    data = dataset[0]

    print("Starting Bayesian GCN training...")
    for epoch in range(num_epochs):
        loss = svi.step(data.x, data.edge_index, data.y)
        if epoch % 20 == 0:
            print(f"Epoch {epoch} Loss: {loss:.4f}")
    print("Training finished.")


# --- 4. Prediction Function ---
def predict_bayesian_model(model, data, num_samples=100):
    # To get a prediction from a BNN, we sample from the learned posterior
    # (or the prior if no training). This involves running the forward pass
    # multiple times with different sampled weights.

    # We use pyro.infer.Predictive to handle the sampling
    predictive = pyro.infer.Predictive(model.model, guide=AutoDelta(model), num_samples=num_samples)

    # Get the samples
    with torch.no_grad():  # Disable gradient calculations for inference
        samples = predictive(data.x, data.edge_index)

    # Extract the 'obs' (observed y) samples
    y_samples = samples['obs'].squeeze(-1)  # Shape: (num_samples, num_nodes)

    # Calculate statistics
    mean_prediction = y_samples.mean(dim=0)
    std_prediction = y_samples.std(dim=0)

    # For confidence intervals, we can use percentiles
    lower_bound = y_samples.quantile(0.025, dim=0)  # 2.5th percentile for 95% CI
    upper_bound = y_samples.quantile(0.975, dim=0)  # 97.5th percentile for 95% CI

    return mean_prediction, std_prediction, lower_bound, upper_bound, y_samples


# --- Main Execution ---
if __name__ == "__main__":

    from pyro.autoguide import AutoDelta  # Import AutoDelta for the guide

    # Hyperparameters
    NUM_NODES = 5
    NUM_NODE_FEATURES = 3  # e.g., time_sin, time_cos, day_of_week
    HIDDEN_DIM = 16
    OUTPUT_DIM = 1  # Predicting one value per node (e.g., travel time to next node)
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.01
    NUM_PREDICT_SAMPLES = 200

    # 1. Generate synthetic data
    dataset = SyntheticGraphDataset(num_nodes=NUM_NODES, num_features=NUM_NODE_FEATURES, num_samples=1)
    train_data = dataset[0]

    # 2. Instantiate the Bayesian GNN model
    bgcn_model = BayesianGCN(NUM_NODE_FEATURES, HIDDEN_DIM, OUTPUT_DIM)

    # 3. Train the model
    train_bayesian_model(bgcn_model, dataset, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    # 4. Make predictions and get uncertainty
    mean_pred, std_pred, lower_ci, upper_ci, y_samples = predict_bayesian_model(bgcn_model, train_data,
                                                                                NUM_PREDICT_SAMPLES)

    print("\n--- Prediction Results (First Node) ---")
    print(f"True Y (Node 0): {train_data.y[0].item():.2f}")
    print(f"Mean Prediction (Node 0): {mean_pred[0].item():.2f}")
    print(f"Std Dev of Prediction (Node 0): {std_pred[0].item():.2f}")
    print(f"95% CI (Node 0): [{lower_ci[0].item():.2f}, {upper_ci[0].item():.2f}]")

    print("\n--- All Node Predictions ---")
    for i in range(NUM_NODES):
        print(
            f"Node {i}: True={train_data.y[i].item():.2f}, Pred_Mean={mean_pred[i].item():.2f}, CI=[{lower_ci[i].item():.2f}, {upper_ci[i].item():.2f}]")

    # Visualize a histogram of the posterior predictive distribution for a node
    import matplotlib.pyplot as plt

    plt.hist(y_samples[:, 0].numpy(), bins=30, density=True, alpha=0.6, color='g', label='Predictive Distribution')
    plt.axvline(train_data.y[0].item(), color='r', linestyle='dashed', linewidth=1,
                label=f'True Y ({train_data.y[0].item():.2f})')
    plt.axvline(mean_pred[0].item(), color='b', linestyle='solid', linewidth=1,
                label=f'Mean Pred ({mean_pred[0].item():.2f})')
    plt.title(f'Posterior Predictive Distribution for Node 0')
    plt.xlabel('Predicted Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()