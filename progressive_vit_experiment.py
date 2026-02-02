"""
Progressive-Capacity ViT Experiment
Tests if φ-scaling benefits hierarchical transformers (varying heads per layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime
import math

# Progressive ViT with variable heads per layer
class ProgressiveViT(nn.Module):
    def __init__(self, d_model=240, heads_schedule=[4,4,4,4,4,4,4,4], 
                 num_classes=10, image_size=32, patch_size=4, mlp_ratio=4.0):
        super().__init__()
        self.d_model = d_model
        self.heads_schedule = heads_schedule
        self.num_layers = len(heads_schedule)
        
        # Patch embedding
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        
        # Transformer layers with varying heads
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, heads_schedule[i], mlp_ratio)
            for i in range(self.num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Count params
        self.total_params = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100 * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return 100 * correct / total


def run_experiment(config, device):
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, num_workers=4, pin_memory=True)
    
    # Model
    torch.manual_seed(config['seed'])
    model = ProgressiveViT(
        d_model=240,
        heads_schedule=config['heads_schedule'],
        num_classes=10
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Train
    best_acc = 0
    history = []
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        history.append({'epoch': epoch+1, 'train_acc': train_acc, 'test_acc': test_acc})
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train {train_acc:.2f}%, Test {test_acc:.2f}%, Best {best_acc:.2f}%")
    
    return {
        'architecture': config['name'],
        'heads_schedule': config['heads_schedule'],
        'lr': config['lr'],
        'seed': config['seed'],
        'params': model.total_params,
        'best_acc': best_acc,
        'final_acc': test_acc,
        'history': history
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define architectures
    architectures = {
        'uniform': [4, 4, 4, 4, 4, 4, 4, 4],
        'progressive_lucas': [2, 2, 3, 3, 5, 5, 8, 8],
        'progressive_standard': [2, 2, 2, 4, 4, 4, 8, 8],
        'progressive_sqrt2': [2, 2, 3, 3, 4, 6, 6, 8],
        'reverse_lucas': [8, 8, 5, 5, 3, 3, 2, 2],
    }
    
    learning_rates = [1e-3, 3e-4]
    seeds = [42, 123, 456]
    epochs = 50
    
    os.makedirs('results/progressive_vit', exist_ok=True)
    
    total = len(architectures) * len(learning_rates) * len(seeds)
    count = 0
    
    for arch_name, heads in architectures.items():
        for lr in learning_rates:
            for seed in seeds:
                count += 1
                print(f"\n[{count}/{total}] {arch_name}, lr={lr}, seed={seed}")
                
                config = {
                    'name': arch_name,
                    'heads_schedule': heads,
                    'lr': lr,
                    'seed': seed,
                    'epochs': epochs
                }
                
                result = run_experiment(config, device)
                
                # Save result
                fname = f"results/progressive_vit/{arch_name}_lr{lr}_seed{seed}.json"
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"  Best: {result['best_acc']:.2f}%, Params: {result['params']:,}")
    
    # Summary
    print("\n" + "="*70)
    print("PROGRESSIVE VIT EXPERIMENT COMPLETE")
    print("="*70)
    
    import glob
    from collections import defaultdict
    
    by_arch = defaultdict(list)
    for f in glob.glob('results/progressive_vit/*.json'):
        with open(f) as fp:
            r = json.load(fp)
            by_arch[r['architecture']].append(r['best_acc'])
    
    print(f"{'Architecture':<25} {'Accuracy':<20} {'N'}")
    print("-"*60)
    for arch in architectures.keys():
        if arch in by_arch:
            accs = by_arch[arch]
            mean = sum(accs) / len(accs)
            std = (sum((x-mean)**2 for x in accs) / len(accs)) ** 0.5
            print(f"{arch:<25} {mean:.2f}% ± {std:.2f}%      {len(accs)}")


if __name__ == '__main__':
    main()
