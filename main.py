import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class BayesianNN(nn.Module):
    """
    The Bayesian Neural Network model.
    It contains the architecture and the BayesianLinear layer definition.
    """

    class BayesianLinear(nn.Module):
        """
        A Bayesian Linear layer with learnable mean and variance for its weights.
        """

        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            # Weight parameters
            self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1))
            self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

            # Bias parameters
            self.b_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.1, 0.1))
            self.b_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

            # Prior distribution for KL divergence calculation
            self.prior_mu = 0.0
            self.prior_sigma = 1.0

            # This will be populated in the forward pass
            self.kl_divergence_loss = 0.0

        def forward(self, x):
            # Calculate standard deviation from rho using the softplus function
            w_sigma = torch.log1p(torch.exp(self.w_rho))
            b_sigma = torch.log1p(torch.exp(self.b_rho))

            # Reparameterization Trick: Sample weights and biases
            epsilon_w = torch.randn_like(w_sigma)
            epsilon_b = torch.randn_like(b_sigma)
            w_sampled = self.w_mu + w_sigma * epsilon_w
            b_sampled = self.b_mu + b_sigma * epsilon_b

            # Calculate the KL divergence for this layer
            kl_w = self._kl_divergence(self.w_mu, w_sigma)
            kl_b = self._kl_divergence(self.b_mu, b_sigma)
            self.kl_divergence_loss = kl_w + kl_b

            # Perform the standard linear operation
            return F.linear(x, w_sampled, b_sampled)

        def _kl_divergence(self, mu, sigma):
            # Analytic KL-divergence between two Gaussians N(mu, sigma) and N(0, 1)
            kl = torch.log(self.prior_sigma / sigma) + (sigma ** 2 + (mu - self.prior_mu) ** 2) / (
                        2 * self.prior_sigma ** 2) - 0.5
            return kl.sum()

    def __init__(self):
        super().__init__()
        self.blinear1 = self.BayesianLinear(1, 100)
        self.blinear2 = self.BayesianLinear(100, 1)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        x = self.blinear2(x)
        return x

    def get_kl_divergence(self):
        # Sum up the KL divergences from all Bayesian layers
        return self.blinear1.kl_divergence_loss + self.blinear2.kl_divergence_loss


class BNNTrainer:
    """
    Handles the training process for the Bayesian Neural Network.
    """

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _elbo_loss(self, y_pred, y_true, kl_divergence, num_batches):
        """
        Calculates the Evidence Lower Bound (ELBO) loss.
        """
        # Negative log-likelihood (assuming Gaussian likelihood, which is MSE)
        reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='sum')

        # ELBO = Likelihood - KL Divergence
        # We want to maximize ELBO, which is equivalent to minimizing this loss
        loss = reconstruction_loss + kl_divergence / num_batches
        return loss

    def train(self, dataloader, epochs=1000):
        """
        Executes the training loop for a given number of epochs.
        """
        num_batches = len(dataloader)
        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()

                y_pred = self.model(x_batch)
                kl = self.model.get_kl_divergence()

                loss = self._elbo_loss(y_pred, y_batch, kl, num_batches)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(dataloader.dataset)
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
        print("Training complete.")


class BNNVisualizer:
    """
    Handles prediction and visualization for a trained BNN model.
    """

    def __init__(self, model):
        self.model = model

    def predict_with_uncertainty(self, x_test, n_samples=100):
        """
        Performs multiple forward passes to get a distribution of predictions.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self.model(x_test).cpu().numpy())

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std_dev = predictions.std(axis=0)

        return mean, std_dev

    def plot_results(self, X_train, y_train, X_test, y_mean, y_std):
        """
        Generates and displays the plot of the BNN's predictions and uncertainty.
        """
        plt.figure(figsize=(12, 6))
        plt.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(), s=10, c='navy', label='Training Data')
        plt.plot(X_test.cpu().numpy(), y_mean, c='red', label='Predictive Mean')
        plt.fill_between(X_test.cpu().numpy().flatten(),
                         (y_mean - 2 * y_std).flatten(),
                         (y_mean + 2 * y_std).flatten(),
                         color='pink', alpha=0.5, label='Uncertainty (±2 std)')
        plt.title('Bayesian Neural Network Regression (OOP Version)')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.ylim(-3, 3)
        plt.show()


def generate_data(n_points):
    """Utility function to generate synthetic data."""
    X = np.concatenate([np.linspace(-5, -1, int(n_points / 2)), np.linspace(1, 5, int(n_points / 2))])
    y = np.sin(X) + np.random.normal(0, 0.2, X.shape)  # Add some noise
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device)


if __name__ == "__main__":
    # 1. Generate Data
    X_train, y_train = generate_data(200)
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Initialize Model
    bnn_model = BayesianNN().to(device)

    # 3. Train the Model
    trainer = BNNTrainer(bnn_model, learning_rate=0.01)
    trainer.train(dataloader, epochs=1000)

    # 4. Visualize the Results
    visualizer = BNNVisualizer(bnn_model)

    # Create test points covering the full range
    X_test = torch.linspace(-8, 8, 200).view(-1, 1).to(device)

    # Get predictions and uncertainty
    y_mean, y_std = visualizer.predict_with_uncertainty(X_test, n_samples=200)

    # Plot everything
    visualizer.plot_results(X_train, y_train, X_test, y_mean, y_std)