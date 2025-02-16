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
from utils.data_utils import FeatureDataset
from models import LinearProbe, MLPProbe
from sklearn.metrics import classification_report, balanced_accuracy_score
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
import orjson
from utils.progress_utils import create_progress_bar, print_metrics_table
from utils.plot_utils import plot_training_history
import arrow
from contextlib import contextmanager, nullcontext
from pathlib import Path
import deepspeed

console = Console()

@contextmanager
def distributed_sync():
    """在分布式训练中同步所有进程的上下文管理器"""
    try:
        yield
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

class NullContextProgress:
    """空的进度条上下文，用于非主进程"""
    def update(self, *args, **kwargs):
        pass
    
    def add_task(self, *args, **kwargs):
        return None

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
    config: dict,
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

                if config.deepspeed.enabled and config.deepspeed.fp16.enabled:
                    model = model.half()

                if config.deepspeed.enabled:
                    steps_per_epoch = len(train_loader)
                    warmup_steps = int(config.probe.warmup_epochs * steps_per_epoch)
                    total_steps = int(config.probe.epochs * steps_per_epoch)
                    
                    # 创建DeepSpeed配置的副本并更新
                    ds_config = OmegaConf.to_container(config.deepspeed, resolve=True)
                    ds_config['train_micro_batch_size_per_gpu'] = config.probe.batch_size
                    ds_config['optimizer']['params']['lr'] = lr
                    ds_config['optimizer']['params']['weight_decay'] = wd
                    ds_config['scheduler']['params']['warmup_num_steps'] = warmup_steps
                    ds_config['scheduler']['params']['total_num_steps'] = total_steps
                    
                    model_engine, _, _, _ = deepspeed.initialize(
                        model=model,
                        config=ds_config
                    )
                    model = model_engine
                else:
                    model_engine = model
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                
                for epoch in range(quick_epochs):
                    train_loss, train_acc = train_epoch(
                        model_engine,
                        train_loader,
                        optimizer if not config.deepspeed.enabled else None,
                        None,
                        device,
                        epoch=epoch,
                        use_deepspeed=config.deepspeed.enabled
                    )
                    console.print(f"[dim]Quick Epoch {epoch + 1}/{quick_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%[/dim]")
                
                val_metrics = evaluate_model(model_engine, val_loader, device, use_deepspeed=config.deepspeed.enabled)
                val_acc = val_metrics["Accuracy"]
                results_table.add_row(f"{lr:.1e}", f"{wd:.1e}", f"{val_acc:.2f}%")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_lr = lr
                    best_wd = wd
                    console.print("[green]New best configuration![/green]")
                
                progress.update(search_task, advance=1)
    
    console.print(results_table)
    console.print(f"\n[bold green]Best configuration: LR={best_lr:.1e}, WD={best_wd:.1e}, Val Acc={best_acc:.2f}%[/bold green]")
    
    return best_lr, best_wd

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[object],
    device: str,
    epoch: int,
    use_deepspeed: bool = False,
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 只在主进程显示进度条
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        progress = create_progress_bar()
        train_task = progress.add_task("[cyan]Training...", total=len(train_loader))
        progress_context = progress
    else:
        progress_context = nullcontext()
        
    with progress_context:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            if use_deepspeed:
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                        if batch_idx == 0:
                            scheduler.step()
                    else:
                        scheduler(batch_idx)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            total_loss += loss.item()
            
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                progress.update(train_task, advance=1)
    
    # 在分布式训练中收集所有进程的指标
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        # 收集损失
        loss_tensor = torch.tensor(total_loss).cuda()
        torch.distributed.all_reduce(loss_tensor)
        total_loss = loss_tensor.item() / world_size
        # 收集正确预测数和总样本数
        correct_tensor = torch.tensor(correct).cuda()
        total_tensor = torch.tensor(total).cuda()
        torch.distributed.all_reduce(correct_tensor)
        torch.distributed.all_reduce(total_tensor)
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    use_deepspeed: bool = False,
) -> Dict[str, float]:
    """Evaluate model on the given data loader with distributed support"""
    model.eval()
    all_preds = []
    all_targets = []
    
    # 只在主进程显示进度条
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        progress = create_progress_bar()
        eval_task = progress.add_task("[cyan]Evaluating...", total=len(data_loader))
        progress_context = progress
    else:
        progress_context = nullcontext()
    
    with progress_context:
        with torch.no_grad():
            for features, targets in data_loader:
                features, targets = features.to(device), targets.to(device)
                if use_deepspeed:
                    outputs = model.module(features)
                else:
                    outputs = model(features)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    progress.update(eval_task, advance=1)
    
    # 在分布式训练中收集所有进程的预测结果
    if torch.distributed.is_initialized():
        # 将预测结果转换为张量，确保使用相同的数据类型
        all_preds_tensor = torch.tensor(all_preds, dtype=torch.float32, device=device)
        all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32, device=device)
        
        # 获取每个进程的预测数量
        local_size = torch.tensor([len(all_preds)], device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(all_sizes, local_size)
        
        # 收集所有进程的预测结果
        max_size = max(size.item() for size in all_sizes)
        
        # 填充到最大长度
        if len(all_preds) < max_size:
            padding_size = max_size - len(all_preds)
            all_preds_tensor = torch.cat([all_preds_tensor, torch.zeros(padding_size, dtype=torch.float32, device=device)])
            all_targets_tensor = torch.cat([all_targets_tensor, torch.zeros(padding_size, dtype=torch.float32, device=device)])
        
        # 收集所有预测结果
        all_preds_gathered = [torch.zeros(max_size, dtype=torch.float32, device=device) for _ in range(torch.distributed.get_world_size())]
        all_targets_gathered = [torch.zeros(max_size, dtype=torch.float32, device=device) for _ in range(torch.distributed.get_world_size())]
        
        torch.distributed.all_gather(all_preds_gathered, all_preds_tensor)
        torch.distributed.all_gather(all_targets_gathered, all_targets_tensor)
        
        # 移除填充并合并结果
        all_preds = []
        all_targets = []
        for pred, target, size in zip(all_preds_gathered, all_targets_gathered, all_sizes):
            # 转换回整数类型用于评估
            all_preds.extend(pred[:size.item()].long().cpu().numpy())
            all_targets.extend(target[:size.item()].long().cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算评估指标
    accuracy = (all_preds == all_targets).mean() * 100
    balanced_accuracy = balanced_accuracy_score(all_targets, all_preds) * 100
    
    # 获取详细的分类报告
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


def save_best_model(model, optimizer, best_val_acc, epoch, actual_hyperparams, 
                    train_stats, config, best_epoch_metrics, save_dir, use_deepspeed):
    best_model = {
        "epoch": epoch + 1,
        "val_accuracy": best_val_acc,
        "hyperparameters": actual_hyperparams,
        "train_stats": train_stats,
        "config": OmegaConf.to_container(config, resolve=True),
        "metrics": best_epoch_metrics.copy()
    }
    
    if use_deepspeed:
        # DeepSpeed 保存
        checkpoint_path = save_dir / f"checkpoint-epoch{epoch+1}"
        model.save_checkpoint(checkpoint_path, client_state=best_model)
    else:
        # PyTorch 保存
        best_model.update({
            "model_state_dict": {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v 
                for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v 
                for k, v in optimizer.state_dict().items()
            }
        })
        torch.save(best_model, save_dir / "best_model.pth")
    
    return best_model

def load_best_model(model, optimizer, save_path, use_deepspeed):
    if use_deepspeed:
        # DeepSpeed 加载
        _, client_state = model.load_checkpoint(save_path)
        return client_state
    else:
        # PyTorch 加载
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

def train_and_evaluate(
    model_name: str,
    features_path: str,
    val_features_path: str,
    config: dict,
):
    """Enhanced training and evaluation function with distributed training support"""
    
    timestamp = arrow.now().format("YYYYMMDD_HHmmss")
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "results" / f"{model_name}_{timestamp}"
    
    # 只在主进程创建目录
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.distributed.get_rank() == 0:
        console.print(Panel(f"Using device: {device}", style="bold blue"))
    
    if torch.distributed.get_rank() == 0:
        console.print(Panel("Loading and preparing data...", style="bold green"))
    
    def load_features(file_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:  
        with h5py.File(file_path, 'r') as f:
            features = f['last_hidden_cls'][:]
            labels = f['targets'][:]
        return features, labels
    
    train_features, train_labels = load_features(features_path, "train")
    val_features, val_labels = load_features(val_features_path, "val")
    
    train_dataset = FeatureDataset(train_features, train_labels, normalize=True, is_train=True)
    val_dataset = FeatureDataset(val_features, val_labels, normalize=True, train_stats=train_dataset.train_stats, is_train=False)
    
    # 添加分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        shuffle=True
    ) if torch.distributed.is_initialized() else None
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        shuffle=False
    ) if torch.distributed.is_initialized() else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1024),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 1024),
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    input_dim = train_features.shape[1]
    num_classes = len(np.unique(train_labels))
    
    probe_config = config.get("probe", {})
    probe_type = probe_config.get("type", "linear")
    
    if probe_type == "linear":
        model = LinearProbe(input_dim, num_classes)
        if config.deepspeed.enabled and config.deepspeed.fp16.enabled:
            model = model.half()
        if torch.distributed.get_rank() == 0:
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
        if config.deepspeed.enabled and config.deepspeed.fp16.enabled:
            model = model.half()
        if torch.distributed.get_rank() == 0:
            console.print(Panel(
                f"Using MLP Probe with:\n"
                f"  Hidden Dim: {mlp_config.get('hidden_dim', 2048)}\n"
                f"  Num Layers: {mlp_config.get('num_layers', 2)}\n"
                f"  Dropout: {mlp_config.get('dropout', 0.1)}",
                style="bold blue"
            ))
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    use_deepspeed = config.get("deepspeed", {}).get("enabled", False)
    if use_deepspeed:
        if torch.distributed.get_rank() == 0:
            console.print(Panel("Initializing DeepSpeed...", style="bold cyan"))
        
        steps_per_epoch = len(train_loader)
        warmup_steps = int(config.probe.warmup_epochs * steps_per_epoch)
        total_steps = int(config.probe.epochs * steps_per_epoch)
        
        # 更新DeepSpeed配置
        ds_config = OmegaConf.to_container(config.deepspeed, resolve=True)
        ds_config['train_micro_batch_size_per_gpu'] = config.probe.batch_size
        ds_config['scheduler']['params']['warmup_num_steps'] = warmup_steps
        ds_config['scheduler']['params']['total_num_steps'] = total_steps
        
        if torch.distributed.get_rank() == 0:
            console.print(Panel(
                f"DeepSpeed Configuration:\n"
                f"  Batch Size per GPU: {config.probe.batch_size}\n"
                f"  Warmup Steps: {warmup_steps}\n"
                f"  Total Steps: {total_steps}",
                style="bold cyan"
            ))
        
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        model = model_engine
    else:
        model = model.to(device)
    
    # 超参数搜索和记录
    if config.probe.hyperparameter_search.enabled:
        if torch.distributed.get_rank() == 0:
            console.print(Panel("Starting hyperparameter search...", style="bold cyan"))
        best_lr, best_wd = find_best_hyperparams(
            train_loader,
            val_loader,
            train_features.shape[1],
            len(np.unique(train_labels)),
            device,
            config.probe.hyperparameter_search.learning_rates,
            config.probe.hyperparameter_search.weight_decays,
            config.probe.hyperparameter_search.quick_epochs,
            config
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
    
    if not use_deepspeed:
        optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
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
    
    for epoch in range(config.probe.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            None if use_deepspeed else optimizer,
            None if use_deepspeed else scheduler,
            device,
            epoch=epoch,
            use_deepspeed=use_deepspeed
        )
        
        # 同步所有进程
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # 只在主进程上进行评估
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                use_deepspeed=use_deepspeed
            )
            
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
                best_model = save_best_model(
                    model=model,
                    optimizer=optimizer if not use_deepspeed else None,
                    best_val_acc=best_val_acc,
                    epoch=epoch,
                    actual_hyperparams=actual_hyperparams,
                    train_stats=train_dataset.train_stats,
                    config=config,
                    best_epoch_metrics=best_epoch_metrics,
                    save_dir=save_dir,
                    use_deepspeed=use_deepspeed
                )
        
        # 同步所有进程
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    # 只在主进程保存结果
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if best_model is not None:
            if use_deepspeed:
                save_path = save_dir / f"checkpoint-epoch{best_model['epoch']}"
            else:
                save_path = save_dir / "best_model.pth"
                
            best_model = load_best_model(
                model=model,
                optimizer=optimizer if not use_deepspeed else None,
                save_path=save_path,
                use_deepspeed=use_deepspeed
            )
        
        # 保存训练历史图表
        plot_path = save_dir / "training_history.png"
        plot_training_history(history, str(plot_path))
        
        # 保存完整的评估结果
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
        
        results_path = save_dir / "results.json"
        with open(results_path, "wb") as f:  
            f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
        
        console.print(Panel(f"[green]Training completed! Results saved to: {save_dir}"))
        return results