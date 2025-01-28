import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import wandb
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
import json
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

class FeatureDataset(Dataset):
    """Enhanced dataset class with feature normalization and statistics tracking"""
    def __init__(self, features: np.ndarray, labels: np.ndarray, normalize: bool = True):
        # Store original statistics for debugging
        self.original_stats = {
            "mean": features.mean(),
            "std": features.std(),
            "min": features.min(),
            "max": features.max()
        }
        
        if normalize:
            # Normalize features using robust statistics
            mean = features.mean(0, keepdims=True)
            std = features.std(0, keepdims=True)
            features = (features - mean) / (std + 1e-5)
            
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
        # Store normalized statistics
        self.normalized_stats = {
            "mean": self.features.mean().item(),
            "std": self.features.std().item(),
            "min": self.features.min().item(),
            "max": self.features.max().item()
        }
        
        # Verify label distribution
        self.label_distribution = torch.bincount(self.labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def print_statistics(self):
        """Print comprehensive dataset statistics"""
        console.print("\n[bold cyan]Dataset Statistics:[/bold cyan]")
        console.print("\nOriginal Features:")
        for k, v in self.original_stats.items():
            console.print(f"{k}: {v:.4f}")
        
        console.print("\nNormalized Features:")
        for k, v in self.normalized_stats.items():
            console.print(f"{k}: {v:.4f}")
        
        console.print(f"\nNumber of classes: {len(self.label_distribution)}")
        console.print(f"Samples per class min: {self.label_distribution.min().item()}")
        console.print(f"Samples per class max: {self.label_distribution.max().item()}")
        console.print(f"Samples per class mean: {self.label_distribution.float().mean().item():.2f}")

class LinearProbe(nn.Module):
    """Improved linear probe with better initialization and normalization"""
    def __init__(self, input_dim: int, num_classes: int, normalize_weights: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_weights = normalize_weights
        
        # Improved initialization using Kaiming initialization
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out')
        nn.init.constant_(self.linear.bias, 0)
        
    def forward(self, x):
        if self.normalize_weights:
            w = self.linear.weight
            self.linear.weight = nn.Parameter(nn.functional.normalize(w, dim=1))
        return self.linear(x)

class GradientTracker:
    """Utility class to track gradient statistics during training"""
    def __init__(self):
        self.grad_norms = []
        self.param_norms = []
        
    def update(self, model: nn.Module):
        total_grad_norm = 0
        total_param_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                param_norm = p.data.norm(2).item()
                total_grad_norm += grad_norm
                total_param_norm += param_norm
        
        self.grad_norms.append(total_grad_norm)
        self.param_norms.append(total_param_norm)
        
    def plot_statistics(self, save_path: str):
        """Plot gradient and parameter norm trends"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.grad_norms)
        plt.title('Gradient Norm History')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.param_norms)
        plt.title('Parameter Norm History')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def train_and_evaluate(
    model_name: str,
    features_path: str,
    val_features_path: str,
    batch_size: int = 512,
    lr: float = 1e-4,  # Reduced learning rate
    epochs: int = 100,
    weight_decay: float = 1e-3,  # Increased weight decay
    save_dir: str = 'results',
    wandb_key: Optional[str] = None,
    wandb_project: str = "clip-dino-linear-probe"
):
    """Enhanced training and evaluation function with improved optimization and diagnostics"""
    
    # Setup wandb
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    
    run = wandb.init(
        project=wandb_project,
        name=f"{model_name}-linear-probe-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": model_name,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "weight_decay": weight_decay
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold blue]Using device: {device}[/bold blue]")
    
    # Load and prepare data with diagnostics
    console.print(f"[bold green]Loading {model_name} data...[/bold green]")
    with h5py.File(features_path, 'r') as f:
        train_features = f['last_hidden_cls'][:]
        train_labels = f['targets'][:]
    with h5py.File(val_features_path, 'r') as f:
        val_features = f['last_hidden_cls'][:]
        val_labels = f['targets'][:]
    
    # Create datasets with statistics tracking
    train_dataset = FeatureDataset(train_features, train_labels, normalize=True)
    val_dataset = FeatureDataset(val_features, val_labels, normalize=True)
    
    # Print dataset statistics
    console.print("\n[bold]Training Dataset Statistics:[/bold]")
    train_dataset.print_statistics()
    console.print("\n[bold]Validation Dataset Statistics:[/bold]")
    val_dataset.print_statistics()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model setup
    input_dim = train_features.shape[1]
    num_classes = len(np.unique(train_labels))
    model = LinearProbe(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Improved optimizer setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Initialize gradient tracker
    gradient_tracker = GradientTracker()
    
    # Training with unified progress tracking
    best_acc = 0
    console.print(Panel(f"[bold]{model_name.upper()} - Starting training for {epochs} epochs[/bold]"))
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        epoch_task = progress.add_task(f"[cyan]Training {model_name}", total=epochs)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                loss.backward()
                gradient_tracker.update(model)  # Track gradients
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": gradient_tracker.grad_norms[-1],
                "parameter_norm": gradient_tracker.param_norms[-1]
            })
            
            # Update progress
            progress.update(epoch_task, advance=1)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = Path(save_dir) / model_name
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, save_path / 'best_model.pth')
            
            # Display current results
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim")
            table.add_column("Value")
            table.add_row("Epoch", f"{epoch+1}/{epochs}")
            table.add_row("Train Loss", f"{train_loss:.4f}")
            table.add_row("Train Accuracy", f"{train_acc:.2f}%")
            table.add_row("Val Loss", f"{val_loss:.4f}")
            table.add_row("Val Accuracy", f"{val_acc:.2f}%")
            table.add_row("Best Val Accuracy", f"{best_acc:.2f}%")
            table.add_row("Learning Rate", f"{scheduler.get_last_lr()[0]:.6f}")
            console.print(table)
    
    # Save gradient statistics plot
    gradient_tracker.plot_statistics(str(Path(save_dir) / model_name / 'gradient_stats.png'))
    
    # Save final results
    results = {
        "model": model_name,
        "best_accuracy": float(best_acc),
        "final_train_accuracy": float(train_acc),
        "final_val_accuracy": float(val_acc),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    wandb.finish()
    return best_acc

def main(wandb_key: Optional[str] = None):
    """Main function with improved configuration"""
    save_dir = "linear_probe_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Improved training configuration
    config = {
        "batch_size": 512,
        "lr": 1e-4,  # Reduced learning rate
        "epochs": 100,
        "weight_decay": 1e-3  # Increased weight decay
    }
    
    # Evaluate CLIP
    console.print(Panel("[bold blue]Evaluating CLIP[/bold blue]"))
    clip_acc = train_and_evaluate(
        "clip",
        '/pasteur2/u/yuhuiz/yiming/clip_vs_dino/cached_features/clip_train_features.h5',
        '/pasteur2/u/yuhuiz/yiming/clip_vs_dino/cached_features/clip_val_features.h5',
        wandb_key=wandb_key,
        **config,
        save_dir=save_dir
    )
    
    # Evaluate DINO
    console.print(Panel("[bold blue]Evaluating DINO[/bold blue]"))
    dino_acc = train_and_evaluate(
        "dino",
        '/pasteur2/u/yuhuiz/yiming/clip_vs_dino/cached_features/dino_train_features.h5',
        '/pasteur2/u/yuhuiz/yiming/clip_vs_dino/cached_features/dino_val_features.h5',
        wandb_key=wandb_key,
        **config,
        save_dir=save_dir
    )
    
    # Create and display final comparison with rich formatting
    final_table = Table(
        title="Final Comparison Results",
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue"
    )
    final_table.add_column("Model", style="cyan")
    final_table.add_column("Best Accuracy", justify="right")
    final_table.add_row("CLIP", f"{clip_acc:.2f}%")
    final_table.add_row("DINO", f"{dino_acc:.2f}%")
    console.print("\n")
    console.print(Panel(final_table))
    
    # Save comparison results
    comparison_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clip_accuracy": float(clip_acc),
        "dino_accuracy": float(dino_acc),
        "config": config
    }
    
    with open(Path(save_dir) / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    models = ['CLIP', 'DINO']
    accuracies = [clip_acc, dino_acc]
    
    sns.barplot(x=models, y=accuracies)
    plt.title('CLIP vs DINO Linear Probe Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.savefig(Path(save_dir) / 'accuracy_comparison.png')
    plt.close()
    
    console.print("\n[bold green]Evaluation completed! Results have been saved to the results directory.[/bold green]")
    console.print(f"[blue]Check {save_dir} for detailed results and visualizations.[/blue]")

if __name__ == '__main__':
    # You can replace this with your WandB API key
    WANDB_API_KEY = "f986124e83c452a464d737564954ac48e74264c0"
    
    try:
        main(wandb_key=WANDB_API_KEY)
    except Exception as e:
        console.print(f"[bold red]Error occurred: {str(e)}[/bold red]")
        raise