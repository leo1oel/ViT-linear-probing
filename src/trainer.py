import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from data_utils import FeatureDataset
from models import LinearProbe, MLPProbe
from sklearn.metrics import classification_report, balanced_accuracy_score
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
import json

console = Console()

def create_progress_bar() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )

def print_metrics_table(metrics: Dict, epoch: int):
    table = Table(title=f"Epoch {epoch} Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.2f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)

def assign_learning_rate(param_group: dict, new_lr: float):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr: float, warmup_length: int, step: int) -> float:
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer: optim.Optimizer, base_lrs: float, warmup_length: int, steps: int):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

def find_best_hyperparams(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    device: str,
    learning_rates: List[float],
    weight_decays: List[float],
    quick_epochs: int,
) -> Tuple[float, float]:
    """Find best learning rate and weight decay using grid search"""
    best_acc = 0
    best_lr = learning_rates[0]
    best_wd = weight_decays[0]
    
    # 创建结果表格
    results_table = Table(
        title="Hyperparameter Search Results",
        show_header=True,
        header_style="bold magenta"
    )
    results_table.add_column("Learning Rate", style="cyan")
    results_table.add_column("Weight Decay", style="cyan")
    results_table.add_column("Validation Accuracy", justify="right", style="green")
    
    with Progress() as progress:
        total_combinations = len(learning_rates) * len(weight_decays)
        search_task = progress.add_task(
            "[cyan]Searching hyperparameters...",
            total=total_combinations
        )
        
        for lr in learning_rates:
            for wd in weight_decays:
                console.print(f"\n[yellow]Testing LR={lr:.1e}, WD={wd:.1e}[/yellow]")
                model = LinearProbe(input_dim, num_classes).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                
                # Quick training with progress tracking
                for epoch in range(quick_epochs):
                    train_loss, train_acc = train_epoch(model, train_loader, optimizer, None, device, epoch=epoch)
                    console.print(f"[dim]Quick Epoch {epoch + 1}/{quick_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%[/dim]")
                
                # Validation
                val_metrics = evaluate_model(model, val_loader, device)
                val_acc = val_metrics["Accuracy"]
                results_table.add_row(f"{lr:.1e}", f"{wd:.1e}", f"{val_acc:.2f}%")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_lr = lr
                    best_wd = wd
                    console.print("[green]New best configuration![/green]")
                
                progress.update(search_task, advance=1)
    
    # 打印结果表格
    console.print(results_table)
    console.print(f"\n[bold green]Best configuration: LR={best_lr:.1e}, WD={best_wd:.1e}, Val Acc={best_acc:.2f}%[/bold green]")
    
    return best_lr, best_wd

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    device: str,
    epoch: int,
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress = create_progress_bar()
    train_task = progress.add_task("[cyan]Training...", total=len(train_loader))
    
    with progress:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            optimizer.step()
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                    if batch_idx == 0:  # 对于epoch-based的scheduler
                        scheduler.step()
                else:  # 对于cosine_lr这样的自定义scheduler
                    scheduler(batch_idx)
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            total_loss += loss.item()
            
            progress.update(train_task, advance=1)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total  # 修复：乘以100使其与验证集准确率保持一致
    
    return avg_loss, accuracy

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on the given data loader"""
    model.eval()
    all_preds = []
    all_targets = []
    
    progress = create_progress_bar()
    eval_task = progress.add_task("[cyan]Evaluating...", total=len(data_loader))
    
    with progress:
        with torch.no_grad():
            for features, targets in data_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                progress.update(eval_task, advance=1)
    
    # Convert to numpy arrays for sklearn metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = (all_preds == all_targets).mean() * 100
    balanced_accuracy = balanced_accuracy_score(all_targets, all_preds) * 100
    
    # Get detailed classification report
    report = classification_report(
        all_targets,
        all_preds,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Macro_F1": report["macro avg"]["f1-score"] * 100,
        "Weighted_F1": report["weighted avg"]["f1-score"] * 100
    }
    
    return metrics

def plot_training_history(history: Dict, save_path: str):
    """Create a beautiful training history plot with loss and accuracy metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    # Set the style
    sns.set_theme(style="darkgrid")
    
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot loss on primary y-axis
    color = sns.color_palette()[0]
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(history['epochs'], history['train_loss'], color=color, label='Train Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()
    color = sns.color_palette()[1]
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12)
    ax2.plot(history['epochs'], [acc * 100 for acc in history['train_acc']], 
            color=color, label='Train Accuracy', linestyle='--', marker='s')
    ax2.plot(history['epochs'], [acc * 100 for acc in history['val_acc']], 
            color=sns.color_palette()[2], label='Val Accuracy', linestyle='--', marker='^')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title
    plt.title('Training History', pad=20, fontsize=14, fontweight='bold')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(
    model_name: str,
    features_path: str,
    val_features_path: str,
    config: dict,
):
    """Enhanced training and evaluation function with proper model saving"""
    
    # 设置结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(Panel(f"Using device: {device}", style="bold blue"))
    
    # Load and prepare data with diagnostics
    console.print(Panel("Loading and preparing data...", style="bold green"))
    
    def load_features(file_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:  
        with h5py.File(file_path, 'r') as f:
            features = f['last_hidden_cls'][:]
            labels = f['targets'][:]
        return features, labels
    
    train_features, train_labels = load_features(features_path, "train")
    val_features, val_labels = load_features(val_features_path, "val")
    
    # Create datasets with statistics tracking
    train_dataset = FeatureDataset(train_features, train_labels, normalize=True, is_train=True)
    val_dataset = FeatureDataset(val_features, val_labels, normalize=True, train_stats=train_dataset.train_stats, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1024),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 1024),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    # Initialize model
    input_dim = train_features.shape[1]
    num_classes = len(np.unique(train_labels))
    
    probe_config = config.get("probe", {})
    probe_type = probe_config.get("type", "linear")
    
    if probe_type == "linear":
        model = LinearProbe(input_dim, num_classes)
        console.print(Panel("Using Linear Probe", style="bold blue"))
    elif probe_type == "mlp":
        mlp_config = probe_config.get("mlp", {})
        model = MLPProbe(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=mlp_config.get("hidden_dim", 2048),
            num_layers=mlp_config.get("num_layers", 2),
            dropout=mlp_config.get("dropout", 0.1)
        )
        console.print(Panel(
            f"Using MLP Probe with:\n"
            f"  Hidden Dim: {mlp_config.get('hidden_dim', 2048)}\n"
            f"  Num Layers: {mlp_config.get('num_layers', 2)}\n"
            f"  Dropout: {mlp_config.get('dropout', 0.1)}",
            style="bold blue"
        ))
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    model = model.to(device)
    
    # 超参数搜索和记录
    if config.probe.hyperparameter_search.enabled:
        console.print(Panel("Starting hyperparameter search...", style="bold cyan"))
        best_lr, best_wd = find_best_hyperparams(
            train_loader,
            val_loader,
            train_features.shape[1],
            len(np.unique(train_labels)),
            device,
            config.probe.hyperparameter_search.learning_rates,
            config.probe.hyperparameter_search.weight_decays,
            config.probe.hyperparameter_search.quick_epochs
        )
    else:
        best_lr = config.probe.learning_rate
        best_wd = config.probe.weight_decay
    
    # 记录实际使用的超参数
    actual_hyperparams = {
        "learning_rate": best_lr,
        "weight_decay": best_wd,
        "batch_size": config.get("batch_size", 1024),
        "warmup_epochs": config.probe.warmup_epochs,
        "total_epochs": config.probe.epochs,
    }
    
    # Model setup with recorded hyperparameters
    optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
    
    # Learning rate schedule setup
    steps_per_epoch = len(train_loader)
    total_steps = config.probe.epochs * steps_per_epoch
    warmup_steps = config.probe.warmup_epochs * steps_per_epoch
    scheduler = cosine_lr(optimizer, best_lr, warmup_steps, total_steps)
    
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 训练循环
    best_val_acc = 0
    best_epoch_metrics = None
    best_model = None
    best_model_state = None
    
    for epoch in range(config.probe.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch=epoch,
        )
        
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Update history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics["Accuracy"])
        
        current_metrics = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            **val_metrics
        }
        print_metrics_table(current_metrics, epoch + 1)
        
        # 保存最佳模型及其对应的指标
        if val_metrics["Accuracy"] > best_val_acc:
            best_val_acc = val_metrics["Accuracy"]
            best_epoch_metrics = current_metrics.copy()
            # 保存模型状态
            best_model_state = {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v 
                for k, v in model.state_dict().items()
            }
            best_model = {
                "epoch": epoch + 1,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": {
                    k: v.clone().detach() if isinstance(v, torch.Tensor) else v 
                    for k, v in optimizer.state_dict().items()
                },
                "val_accuracy": best_val_acc,
                "hyperparameters": actual_hyperparams,
                "train_stats": train_dataset.train_stats,
                "config": OmegaConf.to_container(config, resolve=True),
                "metrics": best_epoch_metrics.copy()
            }
    
    # 恢复到最佳模型状态
    model.load_state_dict(best_model_state)
    
    # 保存训练历史图表
    plot_path = os.path.join(save_dir, "training_history.png")
    plot_training_history(history, plot_path)
    
    # 保存最佳模型
    model_save_path = os.path.join(save_dir, "best_model.pth")
    torch.save(best_model, model_save_path)
    
    # 保存完整的评估结果，使用最佳模型的指标
    results = {
        "model_name": model_name,
        "best_epoch": best_model["epoch"],
        "best_val_accuracy": best_val_acc,
        "best_metrics": best_epoch_metrics,
        "final_metrics": current_metrics,
        "hyperparameters": actual_hyperparams,
        "config": OmegaConf.to_container(config, resolve=True),
        "timestamp": timestamp
    }
    
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(Panel(f"[green]Training completed! Results saved to: {save_dir}"))
    return results