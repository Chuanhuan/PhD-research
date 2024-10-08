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

model.load_state_dict(torch.load("./CNN_MNSIT.pth", weights_only=True))

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
    def __init__(self, dim, K=2):
        super().__init__()
        self.K = K
        self.q_dim = dim
        h1_dim = 20
        h2_dim = 10

        self.q_c = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, self.K * self.q_dim),
            # nn.Linear(h2_dim, self.q_dim),
        )
        self.q_mu = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, self.K),
            # nn.Linear(h2_dim, self.q_dim),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, self.K),
            # nn.Linear(h2_dim, self.q_dim),
        )
        self.mu_y = nn.Sequential(
            nn.Linear(self.q_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, self.q_dim),
            # nn.Linear(h2_dim, 1),
        )

    def reparameterize(self, mu, log_var, phi):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        sigma = sigma.unsqueeze(0)
        mu = mu.unsqueeze(0)
        eps = torch.randn_like(phi)
        z = mu + sigma * eps
        z = z * phi
        # return z.sum(dim=1) + 10
        return z.sum(dim=1)

    def forward(self, x):
        phi = self.q_c(x) ** 2
        phi = phi.view(self.q_dim, self.K)

        # NOTE: method1: softmax winner takes all
        phi = F.softmax(phi, dim=1)

        # NOTE: method2: softmax winner takes all
        # phi = phi / phi.sum(dim=1).view(-1, 1)

        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        z = self.reparameterize(mu, log_var, phi)
        mu_y = self.mu_y(z)
        return z, mu, log_var, phi, mu_y


# %%


def loss_elbo(x, mu, log_var, phi, x_recon, model, predicted):
    # HACK: use the CNN model predition as the input
    # log_var = log_var + 1e-5
    phi = phi + 1e-10
    high_mu_index = mu.argmax()
    high_phi_index = phi[:, high_mu_index] > 0.5
    lamb = (high_phi_index > 0.5).sum()

    t1 = -0.5 * (mu.view(1, -1) - mu_y.view(-1, 1)) ** 2
    t1 = phi * t1
    t1 = t1.mean()
    # t1 = t1.sum()

    # t1 = torch.outer(mu_y, mu) - 0.5 * mu.view(1, -1) **2
    # t1 = -0.5  * (log_var.exp() + mu**2).view(1, -1) + t1
    # t1 = phi * t1
    # t1 = t1.sum()

    # NOTE: Alternative implementation
    t2 = 0.5 * (x * high_mu_index - x_recon) ** 2
    t2 = -torch.sum(t2)
    # t2_1 = 0.5 * (x - x_recon) ** 2
    # t2_1 = -torch.mean(t2_1[high_mu_index])

    # NOTE: this is correct
    # t2 = torch.outer(x, mu) - 0.5 * x.view(-1, 1) ** 2
    # t2 = -0.5 * (log_var.exp() + mu**2).view(1, -1) + t2
    # t2 = phi * t2
    # # t2 = torch.mean(t2)
    # t2 = torch.sum(t2)

    t3 = -torch.log(phi).mean()
    # t3 = phi * torch.log(phi)
    # t3 = -torch.sum(t3)

    t4 = 0.5 * log_var.mean()
    # t4 = torch.pi * log_var.sum()

    # HACK: use the CNN model predition as the input
    # x_recon = x_recon - 10
    model.eval()
    input = x_recon.view(1, 1, 28, 28)
    # Forward pass
    outputs = model(input)
    outputs = F.softmax(outputs, dim=1)
    outputs = torch.clamp(outputs, 1e-5, 1 - 1e-5)
    t5 = torch.log(outputs[:, predicted])
    # print(f't1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}, t5: {t5}, lamb: {lamb}')
    return -(t1 + t2 + t3 + t4 - t5) + lamb
    # return (t1 + t2  + t3 + t4) * (t5)


# %%
print(f"t1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}, t5: {t5}, lamb: {lamb}")

# %%


def ll_gaussian(x, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (x - mu) ** 2


def loss_gau_elbo(x, mu, log_var, phi, x_recon, model, predicted):
    phi = phi + 1e-10
    t1 = ll_gaussian(x_recon, torch.tensor(0), torch.tensor(0))
    t1 = phi * t1
    t1 = torch.mean(t1)

    t2 = ll_gaussian(x, x_recon, torch.tensor(0))
    t2 = phi * t2
    t2 = torch.mean(t2)

    t3 = phi * torch.log(phi)
    t3 = -torch.mean(t3)

    t4 = ll_gaussian(x_recon, mu, log_var)
    t4 = torch.mean(t4)
    # HACK: use the CNN model predition as the input
    # x_recon = x_recon - 10
    model.eval()
    input = x_recon.view(1, 1, 28, 28)
    # Forward pass
    outputs = model(input)
    outputs = F.softmax(outputs, dim=1)
    t5 = torch.log(outputs[:, predicted] + 1e-10)
    # print(f"t1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}, t5: {t5}")
    return -(t1 + t2 + t3 + t4 - t5)


# %%
def loss_cluster_elbo(x, mu, log_var, phi, x_recon, model, predicted):
    lamb = 25
    x1 = x.view(-1, 1) * phi
    binary_array = torch.randint(0, 2, (784, 2))
    phi = (phi > 0.5).int()

    # x1 = x1.sum(dim=1)

    # t1 = (x - x_recon) ** 2
    # t1 = -torch.mean(t1)

    # t2 = log_var.exp().view(-1, 1) + x_recon.view(1,-1) ** 2
    # t2 = log_var.exp().view(-1, 1)
    # t2 = phi * t2
    # t2 = torch.mean(t2)
    t2 = 0.5 * (x - x1[:, 0]) ** 2
    t2 = -torch.sum(t2)

    phi = phi + 1e-5
    t3 = phi * torch.log(phi)
    t3 = -torch.mean(t3)
    model.eval()
    # input = x_recon.view(1, 1, 28, 28)
    input = x1[:, 0].view(1, 1, 28, 28)
    # Forward pass
    outputs = model(input)
    outputs = F.softmax(outputs, dim=1)
    outputs = torch.clamp(outputs, 1e-5, 1 - 1e-5)
    t5 = torch.log(outputs[:, predicted])

    # return -(t1 * lamb) * (-t5)
    return -(t2 + t3 - t5)


# %%

mu = None
log_var = None
predicted = true_y
q_dim = 784
epochs = 3000
m = VI(q_dim, 2).to(device)
optim = torch.optim.Adam(m.parameters(), lr=0.005)

for epoch in range(epochs + 1):
    x = img.view(784).clone().to(device)
    optim.zero_grad()
    x_recon, mu, log_var, phi, mu_y = m(x)
    # Get the index of the max log-probability

    loss = loss_elbo(x, mu, log_var, phi, x_recon, model, predicted)
    # loss = loss_gau_elbo(x, mu, log_var, phi, x_recon, model, predicted)
    # loss = loss_cluster_elbo(x, mu, log_var, phi, x_recon, model, predicted)

    if epoch % 500 == 0:
        print(f"epoch: {epoch}, loss: {loss}")

    loss.backward(retain_graph=True)
    # loss.backward()
    optim.step()

print(f"x.max(): {x.max()}, x.min(): {x.min()}")
print(f"mu.max(): {mu.max()}, mu.min(): {mu.min()}")
print(f"mu_y.max(): {mu_y.max()}, mu_y.min(): {mu_y.min()}")
print(f"var.max(): {log_var.exp().max()}, var.min(): {log_var.exp().min()}")
print(f"prob: {F.softmax(model(x_recon.view(1, 1, 28, 28)), dim=1)}")


# %%


new_image = x_recon.view(1, 1, 28, 28)
# new_image = mu_y.view(1, 1, 28, 28)
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
# NOTE: The model in VI evaluation
k = 0.5
high_phi_index = np.where(phi[:, mu.argmax()].view(28, 28) > k)

print(f"number of high_phi_index:{high_phi_index[0].size}")

plt.imshow(img.clone().detach().numpy(), cmap="gray")
plt.colorbar()


# Scatter plot with colors corresponding to exp_values
plt.scatter(
    # high_var_index[1], high_var_index[0], s=10, c=exp_values_flatten, cmap="viridis"
    high_phi_index[1],
    high_phi_index[0],
    s=10,
    # cmap="viridis",
)

# Add a colorbar to show the mapping from colors to values
plt.title(
    f"Digit {x_recon_pred} Surrogate model with prediction: {F.softmax(model(new_image), dim=1).max():.3f}"
)
plt.savefig(
    f"ID {img_id}-Digit {true_y} pred {x_recon_pred} with {epochs} epochs({k}k).png"
)
plt.show()
plt.clf()

# %%

# NOTE: THe cluster in VI evaluation
clusters = phi.argmax(1).view(28, 28)
plt.imshow(clusters.clone().detach().numpy(), cmap="viridis")


plt.savefig(
    f"Cluster-ID {img_id}-Digit {true_y} pred {x_recon_pred} with {epochs} epochs({k}k).png"
)
print(f"mu: {mu}")


def img_mask_func(img, mu, true_y=true_y, x_recon_pred=x_recon_pred):
    for i in range(mu.shape.__getitem__(0)):
        mask_img = img.clone().detach()
        mask_img = (clusters == i) * mask_img
        model.eval()
        output = F.softmax(model(mask_img.view(1, 1, 28, 28)), dim=1)
        print(f"prob: {F.softmax(model(mask_img.view(1, 1, 28, 28)), dim=1)}")
        if output.max() > 0.5:
            print(f"Cluster {i} has a prediction")
            plt.imshow(mask_img.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
            plt.savefig(
                f"ID {img_id}-Digit {true_y} pred {x_recon_pred} cluster {i} mask_image.png"
            )
            plt.show()
            plt.clf()


img_mask_func(img, mu)

# %%

mask_img = img.clone().detach()


# mask_img[i, j] = img.min()
def img_mask_func(img, high_phi_index):
    for k in range(high_phi_index[0].size):
        for i in range(0, 1, 1):
            for j in range(0, 1, 1):
                mask_img[high_phi_index[0][k] + i, high_phi_index[1][k] + j] = img.min()
    return img


mask_img = img_mask_func(mask_img, high_phi_index)
# mask_img[high_phi_index] = img.min()
print(f"prob: {F.softmax(model(mask_img.view(1, 1, 28, 28)), dim=1)}")
plt.imshow(mask_img.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
plt.savefig(f"ID {img_id}-Digit {true_y} mask_image.png")
