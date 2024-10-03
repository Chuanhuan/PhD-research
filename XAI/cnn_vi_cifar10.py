import torch
import torch.nn as nn
import torch.distributions as dist
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# model = torch.hub.load("pytorch/vision", "resnet18", weights=True)
model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
)

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)  # Changing the output layer to have 10 classes for CIFAR-10
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

testset = torchvision.datasets.CIFAR10(
    root="~/Documents/data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    "Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total)
)


def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean


img_idx = 0
img, _ = testset[img_idx]
img = denormalize(img)
plt.imshow(img.permute(1, 2, 0))
plt.savefig("cifar_img.png")


def imshow(img):
    img = img / 2 + 0.5  # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("cifar_img.png")
    plt.show()


# get some random training images
dataiter = iter(testloader)
images, labels = next(dataiter)

# show images
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", " ".join("%5s" % labels[j] for j in range(4)))


image = images[5].view(1, 3, 32, 32)
predicted = model(image).argmax().item()
imshow(image[0])
print("Predicted: ", classes[predicted])


def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (y - mu) ** 2


class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3072),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3072),
        )
        self.q_c = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3072),
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        c = torch.sigmoid(self.q_c(x))
        return self.reparameterize(mu, log_var), mu, log_var, c


def elbo(y_pred, y, mu, log_var, c, model=model, predicted=predicted):
    # HACK: use the CNN model predition as the input
    model.eval()
    input = y_pred.view(1, 1, 28, 28)
    # Forward pass
    outputs = model(input)

    log_p_y = torch.log(F.softmax(outputs, dim=1)[:, predicted])
    sigma = log_var.exp() ** 0.5
    # likelihood of observing y given Variational mu and sigma
    likelihood = c * dist.Normal(mu, sigma).log_prob(y)

    # prior probability of y_pred
    log_prior = dist.Normal(0, 1).log_prob(y_pred)

    # variational probability of y_pred
    log_p_q = dist.Normal(mu, sigma).log_prob(y_pred)

    # by taking the mean we approximate the expectation
    return (log_p_y + likelihood + log_prior - log_p_q).mean()


def det_loss(y_pred, y, mu, log_var, c, model=model, predicted=predicted):
    return -elbo(y_pred, y, mu, log_var, c, model, predicted)


"""### Training VI"""

# NOTE: Train the VI model
# HACK: if not specify "predicted" digits, than will attack to certain become another value
epochs = 50

m = VI()
optim = torch.optim.Adam(m.parameters(), lr=0.005)


for epoch in range(epochs + 1):
    X = img.view(784).clone()
    Y = img.view(784).clone()
    optim.zero_grad()
    y_pred, mu, log_var, c = m(X)
    # Get the index of the max log-probability

    loss = det_loss(y_pred, Y, mu, log_var, c, model, input[1])

    # try view different digit
    # loss = det_loss(y_pred, Y, mu, log_var, model, 3)

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss: {loss}")
        Y = mu.view(784).clone().detach()
        Y.requires_grad = True

    loss.backward(retain_graph=True)
    # loss.backward()
    optim.step()

print(torch.where(c > 0.3))
