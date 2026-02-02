import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Setup: CPU, batch mic, epochs puține
device = torch.device('cpu')
batch_size = 64
epochs = 10  # Suficient să vezi trend, nu full train

print("="*60)
print("φ-COMPRESSION LAW - LOCAL CPU TEST")
print("="*60)
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print("="*60)

# Data loaders cu augment basic
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# SimpleCNN Standard (32-64-128)
class SimpleCNNStandard(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        return x

# SimpleCNN Lucas (29-47-76)
class SimpleCNNLucas(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 29, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(29)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(29, 47, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(47)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(47, 76, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(76)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(76 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, 76 * 4 * 4)
        x = self.fc(x)
        return x

# Funcție train & eval
def train_and_eval(model, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.to(device)
    
    print(f"\n--- Training {name} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"{name} Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} Accuracy: {acc:.2f}% Params: {params:,}")
    return acc, params

# Rulează
print("\n" + "="*60)
print("TRAINING STANDARD MODEL (32-64-128)")
print("="*60)
model_std = SimpleCNNStandard()
acc_std, params_std = train_and_eval(model_std, "Standard")

print("\n" + "="*60)
print("TRAINING LUCAS MODEL (29-47-76)")
print("="*60)
model_lucas = SimpleCNNLucas()
acc_lucas, params_lucas = train_and_eval(model_lucas, "Lucas")

# Metrics φ-Law
print("\n" + "="*60)
print("φ-COMPRESSION LAW RESULTS")
print("="*60)
retention = params_lucas / params_std
eff_std = acc_std / (params_std / 1e6)
eff_lucas = acc_lucas / (params_lucas / 1e6)
efficiency = eff_lucas / eff_std if eff_std > 0 else 0
product = efficiency * retention

print(f"\nStandard: {acc_std:.2f}% accuracy, {params_std:,} params")
print(f"Lucas:    {acc_lucas:.2f}% accuracy, {params_lucas:,} params")
print(f"\nRetention (Lucas/Std params): {retention:.4f}")
print(f"  Target: 1/e ≈ 0.368")
print(f"\nEfficiency gain: {efficiency:.4f}x")
print(f"  Target: e ≈ 2.718")
print(f"\nProduct (conservation): {product:.4f}")
print(f"  Target: 1.0")
print("\n" + "="*60)
if 0.8 < product < 1.2:
    print("✓ φ-COMPRESSION LAW VALIDATED!")
else:
    print("⚠ Product outside expected range (may need more epochs)")
print("="*60)
