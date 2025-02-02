import torch
import torch.nn as nn
import h5py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Dict
import os
import json
from datetime import datetime

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def load_classifier(model_path: str, device: str = "cuda") -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 打印检查点的内容结构
    print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "直接是状态字典")
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    # 打印权重的形状
    print("Weight shape:", state_dict["linear.weight"].shape)
    print("Bias shape:", state_dict["linear.bias"].shape)
    
    weight = state_dict["linear.weight"]

    classifier = LinearProbe(
        input_dim=weight.shape[1],
        num_classes=weight.shape[0]
    )
    
    classifier.load_state_dict(state_dict)
    return classifier.to(device).eval()

def evaluate_classifier(classifier: nn.Module, 
                       features: np.ndarray, 
                       targets: np.ndarray,
                       device: str = "cuda") -> Dict[str, float]:
    features_tensor = torch.FloatTensor(features).to(device)
    
    # 添加特征统计信息
    print(f"Feature statistics: mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"Feature min: {features.min():.4f}, max: {features.max():.4f}")
    
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features_tensor[i:i + batch_size]
            batch_preds = classifier(batch_features)
            # 检查每个批次的预测
            if i == 0:
                print("First batch prediction example:")
                print(batch_preds[0])  # 打印第一个预测的logits
            predictions.append(batch_preds.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    # 打印预测和目标的分布
    print(f"Prediction distribution: {np.bincount(pred_labels)}")
    print(f"Target distribution: {np.bincount(targets)}")
    
    accuracy = accuracy_score(targets, pred_labels)
    report = classification_report(targets, pred_labels)
    
    return {
        "accuracy": accuracy,
        "report": report
    }

def load_features(feature_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载特征和标签，并进行归一化"""
    with h5py.File(feature_path, 'r') as f:
        features = f['last_hidden_cls'][:]
        targets = f['targets'][:]
        
    # 添加特征归一化
    features_mean = features.mean(axis=0, keepdims=True)
    features_std = features.std(axis=0, keepdims=True)
    features = (features - features_mean) / (features_std + 1e-8)
    
    return features, targets

def save_results(results: Dict, feature_path: str, save_dir: str = "evaluation_results"):
    """保存评估结果到文件"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 从特征路径中提取数据集名称
    dataset_name = os.path.basename(os.path.dirname(feature_path))
    
    # 创建基础文件名
    base_filename = f"{dataset_name}_{timestamp}"
    
    json_filename = os.path.join(save_dir, f"{base_filename}_results.json")
    with open(json_filename, 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'feature_path': feature_path,
            'timestamp': timestamp
        }, f, indent=4)
    
    
    print(f"Results saved to {json_filename}")
    return json_filename

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_path = "/pasteur2/u/yuhuiz/yiming/experiments/src/results/CLIP_20250131_001844/best_model.pth"
    feature_paths = [
        "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/cifar10.1/clip_val_features.h5",
        "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/cifar10/clip_val_features.h5"
    ]
    
    # 存储所有结果
    all_results = {}
    
    try:
        classifier = load_classifier(model_path, device)
        print("Model architecture:")
        print(classifier)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
        
    for i, feature_path in enumerate(feature_paths, 1):
        print(f"\nEvaluating on feature set {i}: {feature_path}")
        try:
            # 检查文件是否存在
            if not os.path.exists(feature_path):
                print(f"Feature file not found: {feature_path}")
                continue
                
            features, targets = load_features(feature_path)
            print(f"Features shape: {features.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Targets unique values: {np.unique(targets)}")
            
            results = evaluate_classifier(classifier, features, targets, device)
            
            print(f"\nAccuracy: {results['accuracy']:.4f}")
            print("\nDetailed Classification Report:")
            print(results['report'])
            
            # 保存结果
            json_file = save_results(results, feature_path)
            all_results[feature_path] = {
                'accuracy': results['accuracy'],
                'json_file': json_file
            }
            
        except Exception as e:
            print(f"Error processing feature set {i}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # 保存所有结果的总结
    summary_file = os.path.join("evaluation_results", f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSummary of all results saved to {summary_file}")

if __name__ == "__main__":
    main()