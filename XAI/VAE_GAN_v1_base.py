import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm


# |%%--%%| <0KiVc3DKq9|45FbLi3jh9>
r"""°°°
CNN model
°°°"""
# |%%--%%| <45FbLi3jh9|4cYhBtzvcP>

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Define transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # MNIST is grayscale
)

# Load datasets
train_dataset = torchvision.datasets.MNIST(
    root="~/Documents/data", train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="~/Documents/data", train=False, download=True, transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


# Initialize model
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
            )

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
# |%%--%%| <4cYhBtzvcP|QpPrzqZjIJ>
r"""°°°
Define parameters for the model
°°°"""
# |%%--%%| <QpPrzqZjIJ|k5YWyyqgo7>


class Config:
    batch_size = 100
    latent_dim = 20
    epochs = 10
    num_classes = 10
    img_dim = 28
    output_dim = 28 * 28
    initial_filters = 16
    intermediate_dim = 256
    lamb = 2.5  # 重构损失权重
    sample_std = 0.5  # 采样标准差
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# |%%--%%| <k5YWyyqgo7|lMC32SbEnu>
r"""°°°
Define VAE model
°°°"""
# |%%--%%| <lMC32SbEnu|GLMDo7DbS2>


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Linear layer to expand 10 features to 64*7*7
        self.fc = nn.Linear(input_dim, 64 * 7 * 7)
        # First transposed convolution: 7x7 -> 14x14
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Second transposed convolution: 14x14 -> 28x28
        self.deconv2 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        # Activation
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # Input shape: (batch, 10)
        x = self.fc(x)  # Shape: (batch, 64*7*7)
        x = x.view(-1, 64, 7, 7)  # Reshape to (batch, 64, 7, 7)
        x = self.leaky_relu(self.deconv1(x))  # Shape: (batch, 32, 14, 14)
        x = self.deconv2(x)  # Shape: (batch, 1, 28, 28)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, Config.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ClusterVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder()

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        return {
            "recon": self.decoder(z),
            "z": z,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
        }


# |%%--%%| <GLMDo7DbS2|zRr2XpaDJo>
r"""°°°
define Critic model
°°°"""
# |%%--%%| <zRr2XpaDJo|bKaOFAsJO1>


# Critic
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1, 28, 28)
        return x


# |%%--%%| <bKaOFAsJO1|eqx0LfDGeH>
r"""°°°
loss function
°°°"""
# |%%--%%| <eqx0LfDGeH|83f6l2Dw3G>


def loss_fn_vae(recon_x, x, z_mean, c, data):
    recon_loss = F.mse_loss(recon_x, x)

    gaussian_loss = (z_mean - data) ** 2
    gaussian_loss = c * gaussian_loss
    gaussian_loss = gaussian_loss.sum(dim=(1, 2, 3))
    gaussian_loss = gaussian_loss.mean()

    total_loss = Config.lamb * recon_loss + gaussian_loss
    return total_loss


def loss_fn_critic(x, z_mean, c, data, model):
    model.eval()  # Ensure model is in eval mode (affects dropout, batchnorm, etc.)
    with torch.no_grad():  # Prevent gradients from flowing through the main model
        perturbed_data = c * data
        pred = model(perturbed_data)

    critic_loss = F.mse_loss(pred, x)

    gaussian_loss = (z_mean - data) ** 2
    gaussian_loss = c * gaussian_loss
    gaussian_loss = gaussian_loss.sum(dim=(1, 2, 3))
    gaussian_loss = gaussian_loss.mean()

    cat_loss = c * torch.log(c + 1e-8)
    cat_loss = cat_loss.sum(dim=(1, 2, 3)).mean()

    total_loss = critic_loss + Config.lamb * cat_loss + gaussian_loss
    # total_loss = gaussian_loss

    return total_loss


# |%%--%%| <83f6l2Dw3G|iRScmHTlMW>
r"""°°°
Training
°°°"""
# |%%--%%| <iRScmHTlMW|tQqyQelokQ>

c_vae = ClusterVAE(Config.num_classes).to(Config.device)
critic = Critic(Config.num_classes).to(Config.device)

vae_optimizer = optim.Adam(c_vae.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

model.eval()
model.requires_grad_(False)  # Freeze main model parameters

# c = torch.ones(batch_size, 1, Config.img_dim, Config.img_dim).to(Config.device)

for epoch in range(Config.epochs):
    total_loss_vae = 0
    total_loss_critic = 0
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")
    acc_list = []

    for data, labels in pbar:
        data = data.to(Config.device)
        labels = labels.to(Config.device)

        # train the VAE
        c_vae.train()
        critic.eval()

        vae_optimizer.zero_grad()

        with torch.no_grad():
            x = model(data)
            _, pred_base = x.max(1)
            c = critic(x)
            if torch.isnan(c).any():
                print("NaN detected in critic output c. Stopping training.")
                # break

        output = c_vae(x)
        recon_x, z, z_mean, z_logvar = (
            output["recon"],
            output["z"],
            output["z_mean"],
            output["z_logvar"],
        )
        loss_vae = loss_fn_vae(output["recon"], x, output["z_mean"], c, data)
        total_loss_vae += loss_vae.item()

        loss_vae.backward()
        vae_optimizer.step()

        # train the critic
        c_vae.eval()
        critic.train()

        critic_optimizer.zero_grad()
        # c: output of the critic from x (x was computed earlier using model(data))
        c = critic(x)

        # Detach the VAE output so that gradients do NOT flow into c_vae.
        with torch.no_grad():
            output = c_vae(x)

        # Compute the critic loss using the detached z_mean from c_vae.
        loss_critic = loss_fn_critic(x, output["z_mean"], c, data, model)

        loss_critic.backward()
        critic_optimizer.step()
        total_loss_critic += loss_critic.item()

        with torch.no_grad():
            _, pred_z = model(c * data).max(1)
            acc = (pred_base == pred_z).float().mean().item()
            acc_list.append(acc)

        pbar.set_postfix(loss=total_loss_critic / (pbar.n + 1))

    print(
        f"Epoch {epoch+1}/{Config.epochs}, Loss VAE: {total_loss_vae/len(train_loader):.4f}, Loss Critic: {total_loss_critic/len(train_loader):.4f}, Acc: {sum(acc_list)/len(acc_list):.4f}"
    )

# |%%--%%| <tQqyQelokQ|g1Uac4InPL>
r"""°°°
Plot images from the test set
°°°"""
# |%%--%%| <g1Uac4InPL|Pv1qVZYTpt>

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_results(test_loader, model, c_vae, critic, device):
    # Get test batch
    data, labels = next(iter(test_loader))
    data = data.to(device)

    with torch.no_grad():
        # Get model outputs
        x = model(data)
        output = c_vae(x)
        c = critic(x)

        # Convert to numpy arrays
        # Using "recon" or "z_mean" as your reconstructed image depends on your architecture.
        # Here we follow your original code:
        recon_imgs = output["z_mean"].cpu().numpy()
        true_imgs = data.cpu().numpy()
        # Reshape critic output to (batch, 28*28) then later to (28,28)
        c_values = c.cpu().numpy().reshape(-1, 28 * 28)
        labels = labels.cpu().numpy()

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot a grid of 3x3 images with the critic heatmap overlay
    for idx in range(9):
        ax = plt.subplot(3, 3, idx + 1)
        # Show the reconstructed image (grayscale)
        plt.imshow(recon_imgs[idx].reshape(28, 28), cmap="gray")
        # Overlay the critic's output as a heatmap (using 'jet' colormap)
        plt.imshow(c_values[idx].reshape(28, 28), cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.title(f"True Label: {labels[idx]}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("reconstructed_images.png")
    plt.show()


# Usage
model.eval()
c_vae.eval()
critic.eval()
visualize_results(test_loader, model, c_vae, critic, Config.device)
