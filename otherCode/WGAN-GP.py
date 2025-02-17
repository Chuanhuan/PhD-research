import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
critic_iterations = 5
lambda_gp = 10
latent_dim = 100
image_size = 28 * 28  # MNIST images are 28x28

# Data loading
transform = transforms.ToTensor()  # Convert to tensor and normalize (0-1)
train_dataset = datasets.MNIST(
    root="~/Documents/data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh(),  # Output should be in the range [-1, 1] for images
        )

    def forward(self, x):
        return self.main(x)


# Critic
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.main(x)


# Initialize models and optimizers
generator = Generator(latent_dim, image_size)
critic = Critic(image_size)

optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_c = optim.Adam(critic.parameters(), lr=learning_rate, betas=(beta1, beta2))


# Gradient Penalty Function (Corrected)
def gradient_penalty(critic, real_data, fake_data):
    batch_size = min(real_data.size(0), fake_data.size(0))  # Get the smaller batch size
    alpha = torch.rand(batch_size, 1).expand_as(
        real_data[:batch_size]
    )  # Use smaller batch size
    interpolated_data = (
        alpha * real_data[:batch_size] + (1 - alpha) * fake_data[:batch_size]
    )

    interpolated_data.requires_grad_(True)
    critic_interpolated = critic(interpolated_data)

    gradients = grad(
        outputs=critic_interpolated,
        inputs=interpolated_data,
        grad_outputs=torch.ones(batch_size, 1),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients**2, dim=1))
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2) * lambda_gp
    return gradient_penalty


# Training loop (Corrected)
for epoch in range(50):
    for i, (real_data, _) in enumerate(train_loader):
        real_data = real_data.view(-1, image_size)

        # Handle last batch (important!)
        if real_data.size(0) < batch_size:  # Skip if the last batch is too small
            continue

        # Train critic
        for _ in range(critic_iterations):
            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)

            # Ensure fake_data and real_data have the same batch size
            min_batch_size = min(real_data.size(0), fake_data.size(0))
            loss_c = -torch.mean(critic(real_data[:min_batch_size])) + torch.mean(
                critic(fake_data[:min_batch_size])
            )
            gp = gradient_penalty(
                critic, real_data, fake_data
            )  # Pass in the potentially smaller batches
            loss_c += gp

            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

        # Train generator
        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        loss_g = -torch.mean(critic(fake_data))

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(
                f"Epoch: {epoch}, Batch: {i}, Critic Loss: {loss_c.item()}, Generator Loss: {loss_g.item()}"
            )

    # Generate and plot images at the end of each epoch
    with torch.no_grad():
        fixed_noise = torch.randn(
            16, latent_dim
        )  # Fixed noise for consistent image generation
        generated_images = generator(fixed_noise).view(
            -1, 1, 28, 28
        )  # Reshape for plotting

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for j, ax in enumerate(axes.flat):
            ax.imshow(
                generated_images[j].squeeze().detach(), cmap="gray"
            )  # Detach from graph
            ax.axis("off")
        plt.suptitle(f"Epoch {epoch}")
        plt.show()  # Or plt.savefig(f"epoch_{epoch}.png") to save images
