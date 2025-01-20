
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os


#|%%--%%| <XkM6XPd3Wp|6pO89jqATn>

# Hyperparameters
batch_size = 100
latent_dim = 20
epochs = 50
num_classes = 10
img_dim = 28
filters = 16
intermediate_dim = 256
lamb = 2.5

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="~/Documents/data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="~/Documents/data", train=False, download=True, transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



#|%%--%%| <6pO89jqATn|nqqFegd7gL>


# Encoder
class Encoder(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(
            (img_dim // 4) * (img_dim // 4) * filters * 2, latent_dim
        )
        self.fc_log_var = nn.Linear(
            (img_dim // 4) * (img_dim // 4) * filters * 2, latent_dim
        )

    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var


# Reparameterization Trick
def reparameterize(z_mean, z_log_var):
    epsilon = torch.randn_like(z_mean)
    return z_mean + epsilon * torch.exp(0.5 * z_log_var)


# Decoder
class Decoder(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, (img_dim // 4) * (img_dim // 4) * filters * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                filters * 2,
                filters,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                filters, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, filters * 2, img_dim // 4, img_dim // 4)
        x_recon = self.deconv(h)
        return x_recon


# Classifier
class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, num_classes)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        y = torch.softmax(self.fc2(h), dim=-1)
        return y


# Gaussian Layer
# NOTE:q(z|y) shape batch_size x num_classes x latent_dim
class GaussianLayer(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(GaussianLayer, self).__init__()
        self.means = nn.Parameter(torch.zeros(num_classes, latent_dim))

    def forward(self, z):
        z = z.unsqueeze(1)
        # print('GaussianLayer:')
        # print(z.shape, self.means.shape)
        return z - self.means.unsqueeze(0)


# VAE Model
class VAE(nn.Module):
    def __init__(self, img_dim, latent_dim, num_classes):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_dim, latent_dim)
        self.decoder = Decoder(img_dim, latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)
        self.gaussian_layer = GaussianLayer(num_classes, latent_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        y = self.classifier(z)
        z_prior_mean = self.gaussian_layer(z)
        return x_recon, z_mean, z_log_var, z_prior_mean, y


# Loss Function
def vae_loss(x, x_recon, z_mean, z_log_var, z_prior_mean, y):
    # Reconstruction Loss
    recon_loss = torch.mean((x - x_recon) ** 2, axis=(1,2,3))
    # print(recon_loss.shape)
    recon_loss = recon_loss.sum()

    # KL Divergence for the latent space
    z_mean = z_mean.unsqueeze(1)  # Shape (batch_size, 1, latent_dim)
    z_log_var = z_log_var.unsqueeze(1)  # Shape (batch_size, 1, latent_dim)
    # kl_divergence = -0.5 * torch.sum(
    #     1 + z_log_var_expanded - z_prior_mean.pow(2) - z_log_var_expanded.exp(),
    #     dim=-1
    # )
    # kl_loss = torch.mean(torch.sum(y * kl_divergence, dim=1))
      # KL Loss
    # print(z_log_var.shape, z_prior_mean.shape)
    kl_loss = -0.5 * (z_log_var - z_prior_mean.pow(2))  # Shape: (batch_size, num_classes, latent_dim)
    # print('kl_loss:')
    # print(kl_loss.shape, y.shape, kl_loss.sum(dim=-1).unsqueeze(-1).shape,y.unsqueeze(-1).shape)
    # print(torch.bmm(y.unsqueeze(-1), kl_loss.sum(dim=-1).unsqueeze(1)).shape)
    # kl_loss = torch.bmm(y.unsqueeze(-1), kl_loss.sum(dim=-1).unsqueeze(1)).mean() # shape batch_size x num_classes
    
    kl_loss = torch.mean(y * kl_loss.sum(dim=-1),axis=1)
    # print(kl_loss.shape)
    kl_loss = kl_loss.sum()

    # Category Loss (Entropy Regularization)
    cat_loss = torch.mean(y * torch.log(y + 1e-8), axis=1)
    # print(cat_loss.shape)
    cat_loss = cat_loss.sum()

    # Weighted total loss
    total_loss = lamb * recon_loss + kl_loss + cat_loss
    return total_loss

lamb =2.5

#|%%--%%| <nqqFegd7gL|wYktO9pGCY>


# Initialize Model
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")
vae = VAE(img_dim, latent_dim, num_classes).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)


# Training Loop
vae.train()
for epoch in range(epochs):
    for x, _ in train_loader:
        x = x.to(device)
        x_recon, z_mean, z_log_var, z_prior_mean, y = vae(x)
        # print(x.shape, x_recon.shape)
        # print(z_mean.shape, z_log_var.shape, z_prior_mean.shape, y.shape)
        loss = vae_loss(x, x_recon, z_mean, z_log_var, z_prior_mean, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# |%%--%%| <wYktO9pGCY|po01dGWU3r>


# Ensure output directory exists
os.makedirs("./samples", exist_ok=True)

def cluster_sample(path, category, x_train, y_train_pred):
    """Visualize samples clustered into a specific category."""
    n = 8  # Grid size
    figure = torch.zeros((n * img_dim, n * img_dim))

    idxs = (y_train_pred == category).nonzero(as_tuple=True)[0]
    if len(idxs) == 0:
        print(f"No samples found for category {category}.")
        return

    for i in range(n):
        for j in range(n):
            index = idxs[torch.randint(len(idxs), (1,)).item()]
            digit = x_train[index].squeeze(0)  # Remove channel dimension
            figure[i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim] = (
                digit
            )

    save_image(figure.unsqueeze(0), path)


def random_sample(path, category, means, std=1.0):
    """Generate random samples conditioned on a specific category."""
    n = 8  # Grid size
    figure = torch.zeros((n * img_dim, n * img_dim))

    for i in range(n):
        for j in range(n):
            noise = torch.randn((1, latent_dim)) * std + means[category]
            noise = noise.to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                generated = vae.decoder(noise).squeeze(0).cpu()
            digit = generated.squeeze(0)  # Remove channel dimension
            figure[i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim] = (
                digit
            )

    save_image(figure.unsqueeze(0), path)


#|%%--%%| <po01dGWU3r|ihaVn7EYVk>

# Get the latent space means from the Gaussian layer
means = vae.gaussian_layer.means.data.cpu()

# Predict cluster labels for training data
vae.eval()
x_train_tensor = torch.stack([data[0] for data in train_dataset]).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(x_train_tensor.shape)
z_mean, _ = vae.encoder(x_train_tensor)
y_train_pred = vae.classifier(z_mean).argmax(dim=1).cpu()

# Generate cluster and random samples
os.makedirs("./samples", exist_ok=True)

for i in range(num_classes):
    cluster_sample(
        f"./samples/cluster_category_{i}.png", i, x_train_tensor.cpu(), y_train_pred
    )
    random_sample(f"./samples/random_category_{i}.png", i, means)


#|%%--%%| <ihaVn7EYVk|NwHHY3FQJz>

# Cluster Sampling
def cluster_sample(vae, category, num_samples=64):
    """
    Generate samples based on the Gaussian mean of a specific category (cluster).
    """
    vae.eval()
    with torch.no_grad():
        # Generate latent vectors around the mean of the specified category
        z_mean = vae.gaussian_layer.means[category].unsqueeze(0).repeat(num_samples, 1)
        z_sample = z_mean + torch.randn_like(z_mean) * 0.1  # Add small noise
        x_recon = vae.decoder(z_sample).cpu()

        # Save the generated images
        save_image(x_recon, f'./samples/cluster_category_{category}.png')
        print(f"Cluster samples for category {category} saved.")

# Random Sampling
def random_sample(vae, loader, num_classes, std=1.0, num_samples=64):
    """
    Generate random samples and calculate train/test accuracy based on classifier predictions.
    """
    vae.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Loop over the dataset to calculate accuracy
        for x, labels in loader:
            x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Forward pass through VAE
            _, _, _, _, y_pred = vae(x)

            # Predicted labels
            predicted = torch.argmax(y_pred, dim=1)

            # Accuracy calculation
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100.0 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

        # Generate samples for each class
        for category in range(num_classes):
            z_mean = vae.gaussian_layer.means[category].unsqueeze(0).repeat(num_samples, 1)
            z_sample = z_mean + torch.randn_like(z_mean) * std
            x_recon = vae.decoder(z_sample).cpu()

            # Save the generated images
            save_image(x_recon, f'./samples/random_category_{category}.png')
            print(f"Random samples for category {category} saved.")

# Usage Example
print("Generating random samples and calculating accuracy on the test set...")
random_sample(vae, test_loader, num_classes)


# |%%--%%| <NwHHY3FQJz|93fODz0go5>

cluster_sample(vae, category=0, num_samples=64)
