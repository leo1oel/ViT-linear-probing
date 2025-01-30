import torch
import torch.nn as nn
import h5py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Dict

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def load_classifier(model_path: str, device: str = "cuda") -> nn.Module:
    """加载训练好的线性分类器"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取模型状态字典
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # 以防是直接保存的状态字典
        
    weight = state_dict["linear.weight"]
    bias = state_dict["linear.bias"]

    classifier = LinearProbe(
        input_dim=weight.shape[1],
        num_classes=weight.shape[0]
    )
    
    # 加载状态字典
    classifier.load_state_dict(state_dict)
    
    return classifier.to(device).eval()

def load_features(feature_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载特征和标签"""
    with h5py.File(feature_path, 'r') as f:
        features = f['last_hidden_cls'][:]  # 假设这是特征的key
        targets = f['targets'][:]
    return features, targets

def evaluate_classifier(classifier: nn.Module, 
                       features: np.ndarray, 
                       targets: np.ndarray,
                       device: str = "cuda") -> Dict[str, float]:
    """评估分类器性能"""
    # 将特征转换为tensor并移到正确的设备上
    features_tensor = torch.FloatTensor(features).to(device)
    
    # 分批次处理以避免内存溢出
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features_tensor[i:i + batch_size]
            batch_preds = classifier(batch_features)
            predictions.append(batch_preds.cpu().numpy())
    
    # 合并所有预测结果
    predictions = np.concatenate(predictions, axis=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    # 计算性能指标
    accuracy = accuracy_score(targets, pred_labels)
    report = classification_report(targets, pred_labels)
    
    return {
        "accuracy": accuracy,
        "report": report
    }

def main():
    # 设置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/pasteur2/u/yuhuiz/yiming/experiments/results/CLIP_20250128_203844/best_model.pth"
    feature_paths = [
        "/pasteur2/u/yuhuiz/yiming/experiments/cached_features/imagenetv2/clip_train_features.h5",
        "/pasteur2/u/yuhuiz/yiming/clip_vs_dino/cached_features_new/clip_train_features.h5"
    ]
    
    # 加载分类器
    classifier = load_classifier(model_path, device)
    print(f"Successfully loaded classifier from {model_path}")
    
    # 在每个特征集上评估
    for i, feature_path in enumerate(feature_paths, 1):
        print(f"\nEvaluating on feature set {i}: {feature_path}")
        try:
            # 加载特征和标签
            features, targets = load_features(feature_path)
            print(f"Loaded features with shape: {features.shape}")
            print(f"Loaded targets with shape: {targets.shape}")
            
            # 评估性能
            results = evaluate_classifier(classifier, features, targets, device)
            
            print(f"\nAccuracy: {results['accuracy']:.4f}")
            print("\nDetailed Classification Report:")
            print(results['report'])
            
        except Exception as e:
            print(f"Error processing feature set {i}: {str(e)}")

if __name__ == "__main__":
    main()