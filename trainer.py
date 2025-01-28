import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
import wandb
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from data_utils import FeatureDataset
from models import LinearProbe, GradientTracker
from sklearn.metrics import classification_report, balanced_accuracy_score
from omegaconf import OmegaConf

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
                    train_acc = train_epoch(model, train_loader, optimizer, None, device)
                    console.print(f"[dim]Quick Epoch {epoch + 1}/{quick_epochs}, Train Acc: {train_acc:.2f}%[/dim]")
                
                # Validation
                val_acc, metrics = evaluate_model(model, val_loader, device)
                results_table.add_row(f"{lr:.1e}", f"{wd:.1e}", f"{val_acc:.2f}%")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_lr = lr
                    best_wd = wd
                    console.print("[green]New best configuration![/green]")
                
                progress.update(search_task, advance=1)
    
    # 显示结果表格
    console.print("\n")
    console.print(results_table)
    
    # 显示最佳结果
    console.print(Panel(
        f"[bold green]Best Configuration:[/bold green]\n"
        f"Learning Rate: {best_lr:.1e}\n"
        f"Weight Decay: {best_wd:.1e}\n"
        f"Validation Accuracy: {best_acc:.2f}%",
        title="Search Results",
        border_style="green"
    ))
    
    return best_lr, best_wd

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    device: str,
    grad_tracker: Optional[GradientTracker] = None,
    use_wandb: bool = False,
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
            
            if grad_tracker is not None:
                grad_tracker.update(model)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            total_loss += loss.item()
            
            # 记录到wandb
            if use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": pred.eq(target.view_as(pred)).sum().item() / target.size(0),
                    "learning_rate": optimizer.param_groups[0]["lr"]
                })
            
            progress.update(train_task, advance=1)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> Tuple[float, dict]:
    model.eval()
    correct = 0
    total = 0
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
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                progress.update(eval_task, advance=1)
    
    accuracy = 100. * correct / total
    balanced_acc = 100. * balanced_accuracy_score(all_targets, all_preds)
    
    # 详细的分类报告，处理零除警告
    report = classification_report(
        all_targets, 
        all_preds, 
        output_dict=True,
        zero_division=0 
    )
    
    # 计算每个类的预测统计
    unique_classes = np.unique(all_targets)
    class_stats = {}
    for cls in unique_classes:
        mask = np.array(all_targets) == cls
        pred_mask = np.array(all_preds) == cls
        class_stats[int(cls)] = {
            "total_samples": np.sum(mask),
            "correct_predictions": np.sum(np.logical_and(mask, pred_mask)),
            "predicted_as_this_class": np.sum(pred_mask)
        }
    
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "report": report,
        "class_stats": class_stats
    }
    
    return accuracy, metrics

def train_and_evaluate(
    model_name: str,
    features_path: str,
    val_features_path: str,
    config: dict,
):
    """Enhanced training and evaluation function with improved optimization and diagnostics"""
    use_wandb = config.get("use_wandb", False)
    
    # 设置结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup wandb
    if config.probe.wandb.key:
        os.environ["WANDB_API_KEY"] = config.probe.wandb.key
    
    # 将 OmegaConf 配置转换为普通字典
    wandb_config = OmegaConf.to_container(config, resolve=True)
    
    wandb.init(
        project=config.probe.wandb.project,
        name=f"{model_name}-linear-probe-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=wandb_config
    )
    
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
    val_dataset = FeatureDataset(val_features, val_labels, normalize=True, is_train=False)
    
    # Print dataset statistics
    console.print("\n[bold]Dataset Statistics:[/bold]")
    train_dataset.print_statistics()
    val_dataset.print_statistics()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.probe.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.probe.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 超参数搜索
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
    
    # Model setup with best hyperparameters
    model = LinearProbe(train_features.shape[1], len(np.unique(train_labels))).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
    
    # Learning rate schedule setup
    steps_per_epoch = len(train_loader)
    total_steps = config.probe.epochs * steps_per_epoch
    warmup_steps = config.probe.warmup_epochs * steps_per_epoch
    scheduler = cosine_lr(optimizer, best_lr, warmup_steps, total_steps)
    
    # Gradient tracking
    grad_tracker = GradientTracker()
    
    # 训练循环
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(config.probe.epochs):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, 
            grad_tracker=grad_tracker if config.probe.track_gradients else None,
            use_wandb=use_wandb
        )
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        
        # 打印结果
        metrics = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            **val_metrics
        }
        print_metrics_table(metrics, epoch + 1)
        
        # 记录到wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                **{f"val_{k.lower()}": v for k, v in val_metrics.items()}
            })
        
        # 保存最佳模型
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": OmegaConf.to_container(config, resolve=True),
                "metrics": metrics
            }
    
    # 保存最佳模型和结果
    model_save_path = os.path.join(save_dir, "best_model.pth")
    torch.save(best_model_state, model_save_path)
    
    # 保存完整的评估结果
    results = {
        "model_name": model_name,
        "best_epoch": best_model_state["epoch"],
        "best_val_accuracy": best_val_acc,
        "final_metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
        "timestamp": timestamp
    }
    
    import json
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(Panel(f"[green]Training completed! Results saved to: {save_dir}"))
    
    return results
