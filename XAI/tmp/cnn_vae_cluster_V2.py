# %%

# NOTE: Define a CNN model for MNIST dataset and load the model weights

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
class VI(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k
        self.c = 14
        self.dim = self.c**2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, 2, stride=2),  # image 14x14
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(self.dim, self.dim),  # Adjusted to match flattened size
            nn.LeakyReLU(0.2),
        )
        self.q_c = nn.Linear(self.dim, self.k * self.dim)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(
                self.k * self.dim, self.k * self.dim
            ),  # Adjusted to match flattened size
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (self.k, self.c, self.c)),
            nn.ConvTranspose2d(self.k, 1, 2, stride=2),
        )

    def encode(self, x):
        x = self.encoder(x)
        phi = self.q_c(x)
        phi = phi**2
        phi = phi.view(self.k, -1)
        phi = F.softmax(phi, dim=0)
        return x, phi

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        z, phi = self.encode(x)
        c = z * phi
        c = c.view(-1, self.k * self.dim)
        x_hat = self.decode(c)
        return x_hat, z, phi


m = VI().to(device)
x = torch.randn(1, 1, 28, 28).to(device)
x_hat, z, phi = m(x)
print(f"phi:{phi.shape}")

# %%
# Define the loss function and optimizer


def loss_elbo(x, phi, x_recon, model, predicted):
    # HACK: use the CNN model predition as the input
    # log_var = log_var + 1e-5
    high_phi_index = phi[0, :] > 0.5
    high_phi_index = high_phi_index.view(1, 1, 14, 14)
    # Convert boolean tensor to float tensor
    high_phi_index = high_phi_index.float()
    # interpolation to 28x28
    c = F.interpolate(high_phi_index, size=(28, 28), mode="nearest")
    c = c.squeeze(0).view(1, 1, 28, 28)
    lamb = (high_phi_index > 0.5).sum()

    # NOTE: Alternative implementation
    t1 = 0.5 * (x * c - x_recon) ** 2
    t1 = -torch.sum(t1)

    # t3 = -torch.log(phi).mean()
    t3 = phi * torch.log(phi + 1e-5)
    t3 = -torch.sum(t3)

    # HACK: use the CNN model predition as the input
    # x_recon = x_recon - 10
    model.eval()
    input = x_recon.view(1, 1, 28, 28)
    # Forward pass
    outputs = model(input)
    outputs = F.softmax(outputs, dim=1)
    outputs = torch.clamp(outputs, 1e-5, 1 - 1e-5)
    t5 = torch.log(outputs[:, predicted])
    return -(t1 + t3 - t5) + lamb


# %%

epochs = 5000
predicted = true_y
m = VI().to(device)
optim = torch.optim.Adam(m.parameters(), lr=0.005)

for epoch in range(epochs + 1):
    x = img.clone().to(device)
    x = x.view(1, 1, 28, 28)
    optim.zero_grad()
    x_recon, z, phi = m(x)
    # Get the index of the max log-probability

    loss = loss_elbo(x, phi, x_recon, model, predicted)

    if epoch % 500 == 0:
        print(f"epoch: {epoch}, loss: {loss}")

    loss.backward(retain_graph=True)
    # loss.backward()
    optim.step()

print(f"x.max(): {x.max()}, x.min(): {x.min()}")
print(f"x_recon.max(): {x_recon.max()}, x_recon.min(): {x_recon.min()}")
print(f"prob: {F.softmax(model(x_recon.view(1, 1, 28, 28)), dim=1)}")

# %%


new_image = x_recon.view(1, 1, 28, 28)
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

high_phi_index = phi[0, :] > 0.5
high_phi_index = high_phi_index.view(1, 1, 14, 14)
# Convert boolean tensor to float tensor
high_phi_index = high_phi_index.float()
# interpolation to 28x28
c = F.interpolate(high_phi_index, size=(28, 28), mode="nearest")
c = c.squeeze(0).view(1, 1, 28, 28)
lamb = (high_phi_index > 0.5).sum()
new_image = x * c.view(1, 1, 28, 28)
x_recon_pred = torch.argmax(F.softmax(model(new_image), dim=1))

plt.imshow(new_image.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
# Add a colorbar to show the mapping from colors to values
plt.title(
    f"Digit {x_recon_pred} Surrogate model with prediction: {F.softmax(model(new_image), dim=1).max():.3f}"
)
plt.savefig(f"ID {img_id}-Digit {true_y} pred {x_recon_pred}.png")
plt.show()
plt.clf()
