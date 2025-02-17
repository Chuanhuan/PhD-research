import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.backends.mps

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
device = torch.device(device)

# Data loading
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    "~/Documents/data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST("~/Documents/data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
# |%%--%%| <HrDNit0izM|0D2fwCYmSZ>


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32x26x26
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))  # 64x24x24
        x = self.pool(x)  # 64x12x12
        x = self.dropout2(x)
        x = torch.flatten(x, 1)  # 9216
        x = F.relu(self.fc1(x))  # 128
        x = self.fc2(x)  # 10
        return x


class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 14x14
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16x14x14
        x = torch.flatten(x, 1)  # 3136
        x = F.relu(self.fc1(x))  # 64
        x = self.fc2(x)  # 10
        return x


# |%%--%%| <0D2fwCYmSZ|ssZeabjaWx>


def train(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"Accuracy: {accuracy:.2f}%")


teacher = Teacher().to(device)
optimizer = optim.Adam(teacher.parameters(), lr=0.001)

print("Training Teacher:")
for epoch in range(10):
    train(teacher, device, train_loader, optimizer, epoch)
    test(teacher, device, test_loader)


# |%%--%%| <ssZeabjaWx|4KD3oNT8JM>


def train_kd(student, teacher, device, loader, optimizer, epoch, alpha=0.1, T=5):
    student.train()
    teacher.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward passes
        with torch.no_grad():
            teacher_logits = teacher(data)
        student_logits = student(data)

        # Compute losses
        loss_ce = F.cross_entropy(student_logits, target)
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T**2)
        total_loss = alpha * loss_ce + (1 - alpha) * loss_kd

        total_loss.backward()
        optimizer.step()


student = Student().to(device)
optimizer = optim.Adam(student.parameters(), lr=0.001)

print("\nTraining Student with Knowledge Distillation:")
for epoch in range(10):
    train_kd(student, teacher, device, train_loader, optimizer, epoch)
    test(student, device, test_loader)
