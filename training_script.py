
#!/usr/bin/env python3
"""Training script that runs on RunPod GPU"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json

print("="*70)
print("PHI ARCHITECTURE TEST - RUNNING ON RUNPOD")
print("="*70)

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV2 = 1 / PHI**2
PHI_NEG4 = PHI ** (-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Containers
class LogicContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = PHI_INV
    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        compressed = mean + (x - mean) * PHI_NEG4
        return self.weight * compressed + (1 - self.weight) * x

class IntuitionContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = PHI_INV2
    def forward(self, x):
        energy = x.abs().mean(dim=(2, 3), keepdim=True)
        max_e = energy.max(dim=1, keepdim=True)[0] + 1e-8
        boost = 1 + (energy / max_e) * (PHI - 1)
        return self.weight * (x * boost) + (1 - self.weight) * x

class SeriesContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.logic = LogicContainer()
        self.intuition = IntuitionContainer()
    def forward(self, x):
        return self.intuition(self.logic(x))


# Models
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

class LucasCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 47, 3, padding=1), nn.BatchNorm2d(47), nn.ReLU(),
            nn.Conv2d(47, 47, 3, padding=1), nn.BatchNorm2d(47), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(47, 76, 3, padding=1), nn.BatchNorm2d(76), nn.ReLU(),
            nn.Conv2d(76, 76, 3, padding=1), nn.BatchNorm2d(76), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(76, 123, 3, padding=1), nn.BatchNorm2d(123), nn.ReLU(),
            nn.Conv2d(123, 123, 3, padding=1), nn.BatchNorm2d(123), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(123, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

class PhiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 47, 3, padding=1), nn.BatchNorm2d(47), nn.ReLU(),
            nn.Conv2d(47, 47, 3, padding=1), nn.BatchNorm2d(47), nn.ReLU(),
            nn.MaxPool2d(2))
        self.c1 = SeriesContainer()
        self.block2 = nn.Sequential(
            nn.Conv2d(47, 76, 3, padding=1), nn.BatchNorm2d(76), nn.ReLU(),
            nn.Conv2d(76, 76, 3, padding=1), nn.BatchNorm2d(76), nn.ReLU(),
            nn.MaxPool2d(2))
        self.c2 = SeriesContainer()
        self.block3 = nn.Sequential(
            nn.Conv2d(76, 123, 3, padding=1), nn.BatchNorm2d(123), nn.ReLU(),
            nn.Conv2d(123, 123, 3, padding=1), nn.BatchNorm2d(123), nn.ReLU(),
            nn.MaxPool2d(2))
        self.c3 = SeriesContainer()
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(123, 10))
    def forward(self, x):
        x = self.c1(self.block1(x))
        x = self.c2(self.block2(x))
        x = self.c3(self.block3(x))
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, trainloader, testloader, epochs=50, name="Model"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_acc': [], 'test_acc': []}
    
    print(f"\nTraining {name} ({count_params(model):,} params)...")
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_acc = 100. * correct / total
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100. * correct / total
        
        scheduler.step()
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train {train_acc:.2f}% | Test {test_acc:.2f}%")
    
    return history


# Main
print("\nLoading CIFAR-10...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

EPOCHS = 50
results = {}

for name, model in [('Baseline', BaselineCNN()), ('Lucas', LucasCNN()), ('Phi', PhiCNN())]:
    history = train_model(model, trainloader, testloader, epochs=EPOCHS, name=name)
    results[name] = {
        'params': count_params(model),
        'final_acc': history['test_acc'][-1],
        'best_acc': max(history['test_acc']),
        'history': history
    }

print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, r in results.items():
    eff = r['best_acc'] / (r['params'] / 1e6)
    print(f"{name}: {r['params']:,} params | Best: {r['best_acc']:.2f}% | Eff: {eff:.2f}")

with open('/workspace/phi_results.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in results.items()}, f, indent=2)

print("\n✓ Results saved to /workspace/phi_results.json")
print("\nDONE!")
