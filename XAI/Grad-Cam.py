import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # Register hook for the gradients
        self.gradients = None
        self.activations = None
        self.conv2.register_backward_hook(self.save_gradients)
        self.conv2.register_forward_hook(self.save_activations)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_input[0]

    def save_activations(self, module, input, output):
        self.activations = output

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Instantiate the model
model = Net()

model.load_state_dict(torch.load("./CNN_MNSIT.pth"))
# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # Train the model
# for epoch in range(5):
#     for images, labels in trainloader:
#         optimizer.zero_grad()
#         output = model(images)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()


# Grad-CAM
def compute_gradcam(model, img):
    model.eval()
    img = img.unsqueeze(0)
    img.requires_grad_()
    scores = model(img)
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]
    score_max.backward()
    gradients = model.gradients.data.numpy()  # Use the gradients saved by the hook
    activations = (
        model.activations.detach().numpy()
    )  # Use the activations saved by the hook
    pooled_gradients = np.mean(gradients, axis=(0, 2, 3))
    for i in range(32):  # 64 is the number of channels in conv2
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = np.mean(activations, axis=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


# Prepare image
img, _ = next(iter(testloader))
img = img[0]

# Compute Grad-CAM
heatmap = compute_gradcam(model, img)

# Display heatmap
plt.matshow(heatmap.squeeze())
plt.savefig("GradCAMheatmap.png")
plt.show()
