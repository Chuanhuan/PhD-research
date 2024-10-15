# %%

# SECTION: Define a CNN model for MNIST dataset and load the model weights

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output


model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = MNIST(
    root="~/Documents//data", train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = MNIST(
    root="~/Documents//data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=1000, shuffle=True)

# %%


# SECTION: load MNIST dataa only 8
class MNIST_8(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.eight_indices = [
            i for i, (img, label) in enumerate(self.mnist_dataset) if label == 8
        ]

    def __getitem__(self, index):
        return self.mnist_dataset[self.eight_indices[index]]

    def __len__(self):
        return len(self.eight_indices)


# Create the dataset for digit 8
testset_8 = MNIST_8(testset)
testloader_8 = DataLoader(testset_8, batch_size=32, shuffle=True)


"""## Load CNN Weights"""

# save the mode weights in .pth format (99.25% accuracy
# torch.save(model.state_dict(), 'CNN_MNSIT.pth')

# NOTE: load the model weights

model.load_state_dict(torch.load("./XAI/CNN_MNSIT.pth", weights_only=True))

"""## Inital image setup"""

img_id = 3
input = testset_8[img_id]
img = input[0].squeeze(0).clone()
true_y = input[1]
# img = transform(img)
plt.imshow(img, cmap="gray")
plt.savefig(f"ID {img_id}-Digit {input[1]} original_image.png")
print(
    f"ID: {img_id}, True y = {input[1]}, probability: {F.softmax(model(input[0].unsqueeze(0)), dim=1).max():.5f}"
)
print(
    f"predicted probability:{F.softmax(model(input[0].unsqueeze(0)), dim=1).max():.5f}"
)
print(f"pixel from {img.max()} to {img.min()}")
# plt.show()
plt.clf()


# %%
class generator(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.channels_img = channels_img
        # self.features_d = features_d
        self.k = 2
        # decoder
        self.decoder = nn.Sequential(
            # NOTE: convTranspose2d output = (input -1)*s -2p + k + op
            # (14-1)*2 + 2 = img 28x28
            nn.ConvTranspose2d(
                channels_img, channels_img, kernel_size=2, stride=2, padding=0
            ),
            nn.Tanh(),
        )

        self.p_layer = nn.Sequential(
            # NOTE: convTranspose2d output = (input -1)*s -2p + k + op
            # (14-1)*2 + 2 = img 28x28
            nn.ConvTranspose2d(
                channels_img, channels_img * self.k, kernel_size=2, stride=2, padding=0
            ),
            nn.Tanh(),
        )

    def forward(self, x, z):
        p = self.p_layer(z)
        p = F.softmax(p, dim=0)
        # x_recon = self.decoder(z) * p[:, 0, :, :] + x * (p[:, 1, :, :])
        x_recon = self.decoder(z)
        return x_recon, p


class learner(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.channels_img = channels_img
        # self.features_d = features_d
        self.k = 2
        # encoder
        # latent mean and variance
        self.mean_layer = nn.Sequential(
            # NOTE: conv2d output = (input + 2p -k)/s +1
            # (28-2)/2 +1 = img 14x14
            nn.Conv2d(channels_img, channels_img, kernel_size=2, stride=2),
            nn.InstanceNorm2d(channels_img, affine=True),
            nn.LeakyReLU(0.2),
        )  # latent mean and variance
        self.logvar_layer = nn.Sequential(
            # NOTE: conv2d output = (input + 2p -k)/s +1
            # (28-2)/2 +1 = img 14x14
            nn.Conv2d(channels_img, channels_img, kernel_size=2, stride=2),
            nn.InstanceNorm2d(channels_img, affine=True),
            nn.LeakyReLU(0.2),
        )

    def reparameterization(self, mean, var):
        # mean = mean.view(mean.size[0], -1)
        # var = var.view(var.size[0], -1)
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def encode(self, x):
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        return z, mean, log_var


G = generator(1).to(device)
L = learner(1).to(device)

x = torch.randn(1, 1, 28, 28).to(device)
z, mean, log_var = L(x)
x_recon, p = G(x, z)
print(f"mean:{mean.shape}, log_var:{log_var.shape}, x_recon:{x_recon.shape}")

# %%


epochs = 1000
leaner_epochs = 10
predicted = true_y
# predicted = 9
G = generator(1).to(device)
L = learner(1).to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=0.005)
opt_L = torch.optim.Adam(L.parameters(), lr=0.005)


def loss_function(x, x_recon, mean, log_var, p):
    alpha = torch.sum(p[:, 0, :, :])
    # reproduction_loss = F.mse_loss(x_recon * p[:, 0, :, :], x * p[:, 0, :, :])
    reproduction_loss = F.mse_loss(x_recon, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


# %%

for epoch in range(epochs + 1):
    for leaner_epoch in range(leaner_epochs + 1):
        opt_L.zero_grad()
        x = img.clone().to(device)
        x = x.view(1, 1, 28, 28)
        z, mean, log_var = L(x)
        x_recon, p = G(x, z)
        # Get the index of the max log-probability
        model.eval()
        critic_fake = F.softmax(model(x_recon), dim=1)[0][predicted]

        loss_L = -(torch.mean(critic_fake))
        # loss = loss_function(x, x_recon, mean, log_var) - torch.log(critic_real + 1e-5) * (
        #     -torch.log(critic_fake + 1e-5)
        # )

        loss_L.backward(retain_graph=True)
        opt_L.step()

    opt_G.zero_grad()
    x = img.clone().to(device)
    x = x.view(1, 1, 28, 28)
    z, mean, log_var = L(x)
    x_recon, p = G(x, z)

    model.eval()
    critic_fake = F.softmax(model(x_recon), dim=1)[0][predicted]
    t1 = -torch.sum(torch.log(critic_fake + 1e-5))
    t2 = loss_function(x, x_recon, mean, log_var, p)

    # NOTE: original loss function
    loss_G = t1 + t2
    # NOTE: alternative loss function, from GAN
    # loss_G = torch.mean(critic_fake)

    loss_G.backward()
    opt_G.step()
    if epoch % 500 == 0:
        print(f"epoch: {epoch}, loss_L: {loss_L}, loss_G: {loss_G}")


# %%
print(f"x.max(): {x.max()}, x.min(): {x.min()}")
print(f"x_recon.max(): {x_recon.max()}, x_recon.min(): {x_recon.min()}")
print(f"mu.max(): {mean.max()}, mu.min(): {mean.min()}")
print(f"log_var.max(): {log_var.max()}, log_var.min(): {log_var.min()}")
print(f"prob: {F.softmax(model(x_recon.view(1, 1, 28, 28)), dim=1)}")
num_pixels = (p[:, 0, :, :] > 0.5).sum()
print(f"num_pixels: {num_pixels}")
high_var_index = torch.exp(log_var) >= torch.exp(log_var).max() * 0.5
torch.sum(high_var_index)


# %%


new_image = x_recon.view(1, 1, 28, 28)
# new_image = F.interpolate(mean, size=(28, 28), mode="nearest")
x_recon_pred = torch.argmax(F.softmax(model(new_image), dim=1))
print(
    f"True y = {true_y}. New image full model prediction: {F.softmax(model(new_image))}"
)
plt.imshow(new_image.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
plt.title(
    f"Digit {x_recon_pred} Surrogate model with prediction: {F.softmax(model(new_image), dim=1).max():.3f}"
)
plt.colorbar()
plt.savefig(f"ID {img_id}-Digit {true_y} pred {x_recon_pred} new_image.png")
plt.show()
plt.clf()

# %%


# high_var_index = torch.exp(log_var) >= torch.exp(log_var).max() * 0.3
# torch.sum(high_var_index)

# NOTE: find the top n_th high variance pixels
for n in range(5, 31, 5):
    flat_tensor = log_var.exp().flatten()
    top_10_indices = torch.topk(flat_tensor, n).indices
    high_var_index = torch.zeros_like(flat_tensor, dtype=torch.bool)
    high_var_index[top_10_indices] = True
    high_var_index = high_var_index.view(log_var.shape[2], log_var.shape[3])

    # Convert boolean tensor to float tensor
    high_var_index = high_var_index.float()
    high_var_index = high_var_index.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and channel dimensions
    # interpolation to 28x28
    c = F.interpolate(high_var_index, size=(28, 28), mode="nearest")
    c = c.squeeze(0).view(1, 1, 28, 28)
    new_image = x * c.view(1, 1, 28, 28)
    x_recon_pred = torch.argmax(F.softmax(model(new_image), dim=1))
    print(f"When n={n}, x_recon_pred: {x_recon_pred}")
    plt.imshow(new_image.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
    # Add a colorbar to show the mapping from colors to values
    plt.title(
        f"Digit {x_recon_pred} Surrogate model with prediction: {F.softmax(model(new_image), dim=1).max():.3f}"
    )
    plt.savefig(f"ID {img_id}-Digit {true_y} pred {x_recon_pred} with n={n}.png")
    plt.show()
    plt.clf()

# %%
