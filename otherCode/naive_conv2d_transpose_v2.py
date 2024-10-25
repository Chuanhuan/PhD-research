import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Simple CNN Encoder with MaxPooling
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=2, stride=1, padding=0
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # MaxPooling (2x2) reduces 63x63 to 31x31
        self.fc1 = nn.Linear(32 * 31 * 31, 10)  # Adjust input size based on pooling

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv2D output: (32, 63, 63)
        x = self.pool(x)  # MaxPooling output: (32, 31, 31)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc1(x)  # Output: (batch_size, 10)
        return x


# Transpose CNN Decoder with Upsampling
class TransposeCNN(nn.Module):
    def __init__(self):
        super(TransposeCNN, self).__init__()
        self.fc = nn.Linear(
            10, 32 * 31 * 31
        )  # Reverse fully connected layer to match pooled size
        self.upsample = nn.Upsample(
            size=(64, 64), mode="nearest"
        )  # Upsample from 31x31 to 62x62
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.relu(self.fc(x))  # Shape: (batch_size, 32 * 31 * 31)
        x = x.view(-1, 32, 31, 31)  # Reshape back to (32, 31, 31)
        x = self.upsample(x)  # Upsample to (32, 62, 62)
        x = self.deconv1(x)  # ConvTranspose2d will output (batch_size, 3, 64, 64)
        return x


# Initialize models
simple_cnn = SimpleCNN()
transpose_cnn = TransposeCNN()

# Set up optimizers and MSE loss function
encoder_optimizer = optim.Adam(simple_cnn.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(transpose_cnn.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Generate random input image as dummy data
    input_data = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

    # Forward pass through the encoder and decoder
    encoded_features = simple_cnn(input_data)
    reconstructed_image = transpose_cnn(encoded_features)

    # Compute the loss
    loss = criterion(reconstructed_image, input_data)

    # Backpropagation and optimization
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
