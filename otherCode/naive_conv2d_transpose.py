import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Simple CNN Encoder
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=2, stride=1, padding=0
        )
        self.fc1 = nn.Linear(32 * 63 * 63, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc1(x)  # Output is a 10-dimensional vector
        return x


# Transpose CNN Decoder
class TransposeCNN(nn.Module):
    def __init__(self):
        super(TransposeCNN, self).__init__()
        self.fc = nn.Linear(10, 32 * 63 * 63)  # Reverse the last layer in SimpleCNN
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=2, stride=1, padding=0
        )

    def forward(self, x):
        x = F.relu(self.fc(x))  # Shape: (batch_size, 32 * 63 * 63)
        x = x.view(-1, 32, 63, 63)
        x = self.deconv1(x)  # Reconstruct to (batch_size, 3, 64, 64)
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
