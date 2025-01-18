import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# |%%--%%| <F7CWjipypl|bAaMZ0eaAD>


# Load Iris dataset
iris = load_iris()
X = iris.data  # Features (150 samples, 4 features)
y = iris.target  # Labels (3 classes)

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Normalize features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Define a Fully Connected Network (FCN)
class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Define a ResNet-like model with residual connections
class ResNetLike(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ResNetLike, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.residual_fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        residual = self.residual_fc(x)  # Transform input for residual connection
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  # Add residual connection
        out = self.relu(out)
        out = self.fc3(out)
        return out


# Hyperparameters
input_size = 4  # Number of features
hidden_size = 10  # Number of neurons in hidden layer
num_classes = 3  # Number of classes
learning_rate = 0.01
num_epochs = 100

# Initialize models
fcn_model = FCN(input_size, hidden_size, num_classes)
resnet_model = ResNetLike(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
fcn_optimizer = optim.Adam(fcn_model.parameters(), lr=learning_rate)
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)


# |%%--%%| <bAaMZ0eaAD|5WDNaQM3na>

# Train the FCN model
for epoch in range(num_epochs):
    fcn_model.train()
    outputs = fcn_model(X_train)
    loss = criterion(outputs, y_train)
    fcn_optimizer.zero_grad()
    loss.backward()
    fcn_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"FCN Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Train the ResNet-like model
for epoch in range(num_epochs):
    resnet_model.train()
    outputs = resnet_model(X_train)
    loss = criterion(outputs, y_train)
    resnet_optimizer.zero_grad()
    loss.backward()
    resnet_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"ResNet Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Evaluate the models
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy


fcn_accuracy = evaluate_model(fcn_model, X_test, y_test)
resnet_accuracy = evaluate_model(resnet_model, X_test, y_test)

print(f"FCN Test Accuracy: {fcn_accuracy * 100:.2f}%")
print(f"ResNet Test Accuracy: {resnet_accuracy * 100:.2f}%")
# |%%--%%| <5WDNaQM3na|hdUr7NcZEw>


# Define a Residual Block for Fully Connected Networks
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        residual = x  # Save the input for the residual connection
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  # Add the residual (skip connection)
        out = self.relu(out)
        return out


# Define a Simple ResNet-like Model for Iris Dataset
class SimpleResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.residual_block1 = ResidualBlock(hidden_size, hidden_size)
        self.residual_block2 = ResidualBlock(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.fc2(out)
        return out


# Hyperparameters
input_size = 4  # Number of features in Iris dataset
hidden_size = 10  # Number of neurons in hidden layers
num_classes = 3  # Number of classes in Iris dataset
learning_rate = 0.01
num_epochs = 100

# Initialize model, loss function, and optimizer
model = SimpleResNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy


accuracy = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
