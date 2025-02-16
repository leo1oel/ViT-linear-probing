from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from torch.utils.data import DataLoader, Dataset

console = Console()


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class FeatureDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
        train_stats: Dict = None,
    ):
        if normalize:
            assert train_stats is not None, "Need training stats for validation set"
            mean = train_stats["mean"]
            std = train_stats["std"]
            features = (features - mean) / (std + 1e-5)

        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_model_and_stats(
    model_path: str, device: str = "cuda"
) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
        pickle_module=__import__("pickle"),
    )
    state_dict = checkpoint["model_state_dict"]
    train_stats = checkpoint.get("train_stats", None)

    if train_stats is None:
        train_features_path = "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/imagenet/dino_train_features.h5"
        with h5py.File(train_features_path, "r") as f:
            train_features = f["last_hidden_cls"][:]
        train_stats = {
            "mean": train_features.mean(0, keepdims=True),
            "std": train_features.std(0, keepdims=True),
        }

    classifier = LinearProbe(
        state_dict["linear.weight"].shape[1], state_dict["linear.weight"].shape[0]
    )
    classifier.load_state_dict(state_dict)
    return classifier.to(device).eval(), train_stats


def evaluate_classifier(
    classifier: nn.Module, val_loader: DataLoader, device: str = "cuda"
) -> Dict[str, float]:
    correct = 0
    total = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    )

    with torch.no_grad():
        with progress:
            task = progress.add_task("Evaluating", total=len(val_loader))
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress.update(task, advance=1)

    accuracy = 100.0 * correct / total
    return {"accuracy": accuracy}


def evaluate_dataset(
    classifier: nn.Module, features_path: str, train_stats: Dict, device: str
) -> float:
    with h5py.File(features_path, "r") as f:
        features = f["last_hidden_cls"][:]
        labels = f["targets"][:]

    dataset = FeatureDataset(
        features=features, labels=labels, normalize=True, train_stats=train_stats
    )

    loader = DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
    )
    results = evaluate_classifier(classifier, loader, device)
    return results["accuracy"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(Panel(f"Using device: {device}"))

    model_path = "/pasteur2/u/yuhuiz/yiming/experiments/src/results/CLIP-ROBERTA4-DATACOMP_20250212_001708/best_model.pth"
    imagenet_path = "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/cifar10/clip-roberta4-datacomp_val_features.h5"
    imagenetv2_path = "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/cifar10.1/clip-roberta4-datacomp_val_features.h5"

    try:
        classifier, train_stats = load_model_and_stats(model_path, device)
        console.print("[green]Model loaded successfully")

        # Evaluate on ImageNet
        imagenet_acc = evaluate_dataset(classifier, imagenet_path, train_stats, device)
        console.print(f"\n[cyan]ImageNet Accuracy: {imagenet_acc:.4f}%")

        # Evaluate on ImageNetV2
        imagenetv2_acc = evaluate_dataset(
            classifier, imagenetv2_path, train_stats, device
        )
        console.print(f"[cyan]ImageNetV2 Accuracy: {imagenetv2_acc:.4f}%")

        # Print accuracy drop
        acc_drop = imagenet_acc - imagenetv2_acc
        console.print(f"[yellow]Accuracy Drop: {acc_drop:.4f}%")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


if __name__ == "__main__":
    main()
