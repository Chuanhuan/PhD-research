# %%
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19, densenet201
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch.distributions as dist
import torch.nn.functional as F
import cv2

# %%
# use the ImageNet transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# dataset = datasets.ImageFolder(
#     root=os.path.expanduser("~/Documents/imagenet_images/elephant/"),
#     transform=transform,
# )
# # define a 1 image dataset
# dataset = datasets.ImageFolder(
#     root="~/Documents/imagenet_images/elephant/",
#     transform=transform,
# )
#
# # define the dataloader to load that single image
# dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


# %%


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # get the pretrained VGG19 network
        self.vgg = vgg19(weights=True)

        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]

        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        )

        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


# %%

# initialize the VGG model
vgg = VGG()

# set the evaluation mode
vgg.eval()


# %%
# get the image from the dataloader
# img, _ = next(iter(dataloader))

image_dir = os.path.expanduser("~/Documents/imagenet_images/piggery/")
image_files = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith(".jpg")
]

# Load the ImageNet class names
with open("./imagenet-classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

image_file = image_files[2]
img = Image.open(image_file)
img = transform(img).unsqueeze(0)

# get the most likely prediction of the model
pred_index = vgg(img).argmax(dim=1)

print(f"Prediction: {pred_index}, {class_names[pred_index]}")
# Get the top 5 predictions of the model
top_pred_values, top_pred_indices = vgg(img).topk(5)

# Print the top 5 predictions and their corresponding classes
top_pred_indices = top_pred_indices.detach().cpu().numpy().flatten()
top_pred_values = F.softmax(vgg(img), dim=1)[:, top_pred_indices]
top_pred_values = top_pred_values.detach().cpu().numpy().flatten()
for i in range(5):
    print(
        f"Prediction {i+1}: {top_pred_indices[i]}, {top_pred_values[i]*100 :.2f}%, {class_names[top_pred_indices[i]]}"
    )


# %%
# get the gradient of the output with respect to the parameters of the model
pred = vgg(img)
pred[:, pred_index].backward()

# pull the gradients out of the model
gradients = vgg.get_activations_gradient()
print(gradients.shape)

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
print(pooled_gradients.shape)

# get the activations of the last convolutional layer
activations = vgg.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())
plt.savefig("./grad-cam-heatmap.jpg")

# %%

# Get the base name of the image file (i.e., the name without the directory path)
image_name_with_ext = os.path.basename(image_file)

# Split the base name into name and extension
image_name, image_ext = os.path.splitext(image_name_with_ext)
# img = cv2.imread("./data/Elephant/data/05fig34.jpg")
img = cv2.imread(image_file)
cv2.imwrite(f"./orig-image-{image_name}.jpg", img)

# %%

# Convert the PyTorch tensor to a NumPy array
heatmap_np = heatmap.numpy()

# Resize the heatmap
heatmap_resized = cv2.resize(heatmap_np, (img.shape[1], img.shape[0]))
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

superimposed_img = heatmap_resized * 0.4 + img
print(image_name, image_ext)
cv2.imwrite(f"./grdd-cam-map-{image_name}.jpg", superimposed_img)

# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
# superimposed_img = heatmap * 0.4 + img
# print(image_name, image_ext)
# cv2.imwrite(f"./grdd-cam-map-{image_name}.jpg", superimposed_img)

# %%


image_file = image_files[2]
img = Image.open(image_file)
img = transform(img).unsqueeze(0)

model = vgg19(weights=True)


def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (y - mu) ** 2


class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(3 * 224 * 224, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3 * 224 * 224),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(3 * 224 * 224, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3 * 224 * 224),
        )

    def reparameterize(self, mu, log_var):
        # Define the minimum and maximum values for log_var
        min_log_var = -10
        max_log_var = 10

        # Clip log_var to the specified range
        clipped_log_var = torch.clamp(log_var, min_log_var, max_log_var)

        # Compute sigma
        sigma = torch.exp(0.5 * clipped_log_var) + 1e-5
        # print(torch.sum(torch.isnan(sigma)))
        # std can not be negative, thats why we use log variance
        # sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var


def elbo(x_pred, X, mu, log_var, model=model, predicted=pred_index.item()):
    # HACK: use the CNN model predition as the input
    model.eval()
    input = x_pred.view(1, 3, 224, 224)
    # Forward pass
    outputs = model(input)
    # print(torch.sum(torch.isnan(outputs)))
    eps = 1e-8  # small constant

    tmp = F.softmax(outputs, dim=1)[:, predicted] + eps
    log_p_y = torch.log(tmp)
    # HACK: use eps for variance, as we want
    # Define the minimum and maximum values for log_var
    min_log_var = -10
    max_log_var = 10

    # Clip log_var to the specified range
    clipped_log_var = torch.clamp(log_var, min_log_var, max_log_var)

    # Compute sigma
    sigma = (clipped_log_var.exp() + 1e-8) ** 0.5

    # likelihood of observing y given Variational mu and sigma
    likelihood = dist.Normal(mu, sigma).log_prob(X)
    # sigma = log_var.exp() ** 0.5
    # # likelihood of observing y given Variational mu and sigma
    # likelihood = dist.Normal(mu, sigma).log_prob(X)

    # prior probability of x_pred
    log_prior = dist.Normal(0, 1).log_prob(x_pred)

    # variational probability of x_pred
    log_p_q = dist.Normal(mu, sigma).log_prob(x_pred)

    # by taking the mean we approximate the expectation
    return (log_p_y + likelihood + log_prior - log_p_q).mean() + eps


def det_loss(x_pred, X, mu, log_var, model=model, predicted=pred_index.item()):
    return -elbo(x_pred, X, mu, log_var, model, predicted)


# %%
"""### Training VI"""

# NOTE: Train the VI model
# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg19(weights=True).to(device)  # Move the VGG model to the device

# Move the model to the GPU
m = VI().to(device)

optim = torch.optim.Adam(m.parameters(), lr=0.005)

epochs = 1000
for epoch in range(epochs + 1):
    # Move the data to the GPU
    X = img.view(3 * 224 * 224).clone().to(device)
    optim.zero_grad()
    x_pred, mu, log_var = m(X)

    loss = det_loss(x_pred, X, mu, log_var, model, pred_index.item())

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss: {loss}")

    loss.backward()
    optim.step()

# %%

with torch.no_grad():
    X = img.view(3 * 224 * 224).clone()
    x_pred, mu, log_var = m(X)
    print(torch.abs(x_pred - X).mean())

new_image = mu.view(1, 3, 224, 224)
new_image = (new_image - new_image.min()) / (new_image.max() - new_image.min())
predicted = torch.argmax(F.softmax(model(new_image), dim=1))
print(
    f"True y = {pred_index.item()}, the highest probability index: {predicted.max():.5f}"
)
pred_prob = F.softmax(model(new_image), dim=1)[:, predicted]
print(f"New image full model prediction: {pred_prob.item()}")
new_image = new_image.permute(0, 2, 3, 1)
plt.imshow(new_image.squeeze(0).detach().numpy())
plt.title(
    f"Pred {pred_index.item()} Surrogate model with prediction: {pred_prob.item():.3f}"
)
plt.savefig(f"Surrogate_image-{image_name}.png")
plt.show()
plt.clf()

# %%

X = img.view(3 * 224 * 224).clone().to(device)
X = (X - X.min()) / (X.max() - X.min())

print(f"variance: {log_var.exp().max()}")
highest_var = log_var.exp().max()
k = 0.8
high_var_index = np.where(log_var.view(3, 224, 224).exp() > highest_var * k)

plt.imshow(X.view(3, 224, 224).permute(1, 2, 0).detach().numpy())
# plt.scatter(high_var_index[1], high_var_index[0], s=10, c="red")
# Assume log_var is a tensor and you compute its exponential
exp_values = log_var.view(3, 224, 224).exp()

# Flatten the tensor to 1D for scatter plot
exp_values_flatten = exp_values[high_var_index[0], high_var_index[1], high_var_index[2]]


# Scatter plot with colors corresponding to exp_values
plt.scatter(high_var_index[2], high_var_index[1], s=10, c=exp_values_flatten)

# Add a colorbar to show the mapping from colors to values
plt.title(f"Pred {pred_index.item()} High Variance Index(> {k})")
plt.colorbar(label="exp(log_var)")
plt.savefig(f"High_var_index-{image_name}.png")
plt.show()
plt.clf()


# %%
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        # get the pretrained DenseNet201 network
        self.densenet = densenet201(pretrained=True)

        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features

        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        # get the classifier of the vgg19
        self.classifier = self.densenet.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # don't forget the pooling
        x = self.global_avg_pool(x)
        x = x.view((1, 1920))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)
