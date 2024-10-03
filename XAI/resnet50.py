# %%
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()
print(model)
# Define the transformation
# NOTE: use the same transform as the training data
# image -> 3x256x256 image -> 3x224x224 image -> tensor -> normalize
# transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load and preprocess the images
image_dir = os.path.expanduser("~/Documents/imagenet_images/elephant/")
image_files = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith(".jpg")
]

# Load the ImageNet class names
with open("./imagenet-classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

i = 0
for image_file in image_files:
    print(image_file)
    image = Image.open(image_file)
    image = transform(image).unsqueeze(0)
    # Pass the image through the model and get the top 5 predictions
    output = model(image)
    _, top5_preds = torch.topk(output, 5)
    # Convert the output indices to class names
    top5_preds = [class_names[idx] for idx in top5_preds[0]]
    print(f"Top 5 predictions for {os.path.basename(image_file)}: {top5_preds}")
    i += 1
    if i >= 10:
        break

# %%


def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (y - mu) ** 2


class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 784),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 784),
        )
        self.q_c = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 784),
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
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
