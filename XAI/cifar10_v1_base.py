
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm

##########################################
# 1. Download and Prepare CIFAR-10
##########################################
# Define transforms (using normalization values common for CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

# Download training and test sets
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

##########################################
# 2. Load a Pre-trained CIFAR-10 Model
##########################################
print("Loading pre-trained model...")
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Evaluate the pre-trained model on CIFAR-10 test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # outputs are logits of shape (batch, 10)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
test_accuracy = 100.0 * correct / total
print(f"Test Accuracy on CIFAR-10: {test_accuracy:.2f}%")

##########################################
# 3. Configuration
##########################################
class Config:
    batch_size = 128
    epochs = 1000
    num_classes = 10         # Output dimension of the pre-trained model (logits)
    img_dim = 32             # CIFAR-10 images are 32x32
    output_dim = 32 * 32 * 3   # Not used directly, but here for reference
    lamb = 2.5               # Weight for the reconstruction loss
    device = device

##########################################
# 4. Define the Integrated ClusterVAE Model
##########################################
# Encoder: maps a 10-dim vector (pre-trained model output) to a latent image (batch, 3, 32, 32)
class Encoder(nn.Module):
    def __init__(self, input_dim, img_dim=32):
        super().__init__()
        # Map input vector to a feature map of shape (64, 8, 8)
        self.fc = nn.Linear(input_dim, 64 * 8 * 8)
        # Two transposed conv layers to upsample: 8x8 -> 16x16 -> 32x32
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)   # 16x16 -> 32x32
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.fc(x)              # (batch, 64*8*8)
        x = x.view(-1, 64, 8, 8)     # (batch, 64, 8, 8)
        x = self.leaky_relu(self.deconv1(x))  # (batch, 32, 16, 16)
        x = self.deconv2(x)         # (batch, 3, 32, 32)
        return x

# Decoder: maps the latent image (batch, 3, 32, 32) to classification logits (batch, 10)
class Decoder(nn.Module):
    def __init__(self, img_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, Config.num_classes)

    def forward(self, x):
        x = self.conv(x)            # (batch, 64, 8, 8)
        x = self.flatten(x)         # (batch, 4096)
        x = self.fc(x)              # (batch, num_classes)
        return x

# ClusterVAE integrates the Encoder, Decoder, and a Critic module.
class ClusterVAE(nn.Module):
    def __init__(self, input_dim, img_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, img_dim=img_dim)
        self.decoder = Decoder(img_dim=img_dim)
        # Critic: processes the input vector and outputs a single-channel mask (batch, 1, 32, 32)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim * img_dim),  # 32x32 = 1024
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, num_classes) from the pre-trained model
        z = self.encoder(x)         # latent image: (batch, 3, 32, 32)
        recon = self.decoder(z)     # reconstructed logits: (batch, num_classes)
        c = self.critic(x)          # critic output: (batch, 1024)
        c = c.view(-1, 1, Config.img_dim, Config.img_dim)  # reshape to (batch, 1, 32, 32)
        return {"recon": recon, "z": z, "critic": c}

# Combined loss function: reconstruction loss + Gaussian-like penalty + critic loss + regularization terms
def combined_loss_fn(x, data, output, frozen_model):
    """
    x: Pre-trained model output on data (batch, num_classes)
    data: Original CIFAR-10 images (batch, 3, 32, 32)
    output: Dict from ClusterVAE with keys "recon", "z", and "critic"
    frozen_model: Pre-trained model (frozen) for reference predictions
    """
    # Reconstruction loss (MSE between decoder output and pre-trained model logits)
    recon_loss = F.mse_loss(output["recon"], x)

    # Gaussian-like penalty between encoder output and input image, weighted by critic mask
    gaussian_loss = 0.5 * (((output["z"] - data) ** 2 * output["critic"]).sum(dim=(1, 2, 3))).mean()
    vae_loss = Config.lamb * recon_loss + gaussian_loss

    # Critic loss: perturb the image using the critic mask and compare predictions
    with torch.no_grad():
        perturbed_data = output["critic"] * data
        pred = frozen_model(perturbed_data)
    critic_loss = F.mse_loss(pred, x)

    # Additional regularization for the critic
    cat_loss = (output["critic"] * torch.log(output["critic"] + 1e-8)).sum(dim=(1, 2, 3)).mean()
    cat_sum = output["critic"].sum(dim=(1, 2, 3)).mean()

    total_loss = vae_loss + critic_loss + cat_loss + cat_sum
    return total_loss

##########################################
# 5. Training Loop
##########################################
# Instantiate the integrated model (input_dim is num_classes = 10)
c_vae = ClusterVAE(Config.num_classes, img_dim=Config.img_dim).to(Config.device)
optimizer = optim.Adam(c_vae.parameters(), lr=0.001)

# Freeze the pre-trained model so its parameters are not updated
for param in model.parameters():
    param.requires_grad = False

print("Starting training...")
for epoch in range(Config.epochs):
    total_loss = 0.0
    acc_list = []
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")
    for images, labels in pbar:
        images = images.to(Config.device)
        labels = labels.to(Config.device)

        # Get the pre-trained model's output (logits) and prediction baseline
        with torch.no_grad():
            x = model(images)  # x: (batch, 10)
            _, pred_base = x.max(1)

        c_vae.train()
        optimizer.zero_grad()

        # Forward pass: use pre-trained model output as input to ClusterVAE
        output = c_vae(x)
        loss = combined_loss_fn(x, images, output, model)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Optional: compute an accuracy metric comparing frozen model predictions on encoder output vs. baseline
        with torch.no_grad():
            _, pred_z = model(output["z"]).max(1)
            acc = (pred_base == pred_z).float().mean().item()
            acc_list.append(acc)

        pbar.set_postfix(loss=total_loss/(pbar.n+1), acc=sum(acc_list)/len(acc_list))
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = sum(acc_list) / len(acc_list)
    print(f"Epoch {epoch+1}/{Config.epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
