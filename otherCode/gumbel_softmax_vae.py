import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# |%%--%%| <LZghsDIrvI|g20WPwZiEM>


# Define the Gumbel-Softmax function
def gumbel_softmax(logits, temperature, hard=False):
    """
    Samples from the Gumbel-Softmax distribution.

    Args:
      logits: [batch_size, num_classes] unnormalized log-probs
      temperature: float, temperature for smoothing
      hard: bool, if True, returns one-hot samples, but differentiable via straight-through estimator.
            If False, returns soft samples.

    Returns:
      [batch_size, num_classes] samples
    """
    y_soft = F.gumbel_softmax(logits, tau=temperature, hard=hard)

    if hard:
        # Straight-through estimator.
        y_hard = torch.max(y_soft, dim=-1, keepdim=True)[1]  # Get one-hot indices
        y_hard = torch.zeros_like(logits).scatter_(
            -1, y_hard, 1.0
        )  # Convert indices to one-hot
        y = (
            y_hard.detach() + y_soft - y_soft.detach()
        )  # y_hard for gradient, y_soft for value
    else:
        y = y_soft
    return y


# Define the VAE model
class VAE(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, num_classes
    ):  # num_classes for discrete latent
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(
            hidden_dim, latent_dim * num_classes
        )  # Changed to output for all classes
        self.fc_logvar = nn.Linear(
            hidden_dim, latent_dim * num_classes
        )  # Changed to output for all classes

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * num_classes, hidden_dim),  # Input adjusted
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # For image reconstruction
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h).view(
            -1, self.latent_dim, self.num_classes
        )  # Reshape for classes
        logvar = self.fc_logvar(h).view(
            -1, self.latent_dim, self.num_classes
        )  # Reshape for classes
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = z.view(-1, self.latent_dim * self.num_classes)  # Flatten before decoding
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# |%%--%%| <g20WPwZiEM|8VCOGkPWM3>


# Hyperparameters
input_dim = 784
hidden_dim = 512
latent_dim = 10
num_classes = 5
learning_rate = 1e-3
temperature = 1.0
anneal_rate = 0.001
min_temp = 0.5
batch_size = 64
epochs = 50  # Increased for better training

# Data loading
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="~/Documents/data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = datasets.MNIST(
    root="~/Documents/data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


# Model and optimizer
model = VAE(input_dim, hidden_dim, latent_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(data)

        z = gumbel_softmax(mu, temperature, hard=True)

        reconstruction_loss = (
            F.binary_cross_entropy(x_hat, data, reduction="sum") / data.shape[0]
        )

        kl_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu**2) / data.shape[0]
        # kl_loss = kl_loss.view(-1, latent_dim, num_classes).sum(dim = (1,2))
        # kl_loss = kl_loss.view(-1, latent_dim * num_classes).sum(dim=1)
        loss = reconstruction_loss + kl_loss

        loss.backward()
        optimizer.step()

        temperature = max(temperature * (1 - anneal_rate), min_temp)

        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Temp: {temperature:.4f}"
            )


# Plotting the results (after training)
with torch.no_grad():
    # Generate samples from the latent space
    num_samples = 10
    # Create random one-hot vectors for the latent variable
    z = torch.randn(num_samples, latent_dim, num_classes)
    z_indices = torch.argmax(z, dim=2)
    z_one_hot = torch.zeros_like(z).scatter_(2, z_indices.unsqueeze(2), 1)

    z = z_one_hot.view(num_samples, -1)  # Flatten for decoding

    samples = model.decode(z).view(num_samples, 1, 28, 28).cpu()  # Reshape for plotting

    # Plot the generated samples
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")
    plt.show()

    # Reconstructions:
    data, _ = next(iter(test_loader))  # Get a batch of test data
    data = data.view(-1, input_dim)
    x_hat, _, _ = model(data)
    n = min(data.size(0), 8)  # Number of images to display
    comparison = torch.cat(
        [data[:n].view(-1, 1, 28, 28), x_hat[:n].view(-1, 1, 28, 28)], dim=0
    )

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(7, 2 * n))
    for i in range(n):
        for j in range(2):
            axes[i, j].imshow(
                comparison[i * 2 + j].view(28, 28).cpu().data, cmap="gray"
            )
            axes[i, j].axis("off")
    plt.show()
