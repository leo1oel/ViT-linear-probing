import h5py
import numpy as np
from tqdm import tqdm
import csv
import faiss
import random
import torch


def load_features_batch(file_path, start_idx, batch_size):
    """分批加载特征和路径"""
    with h5py.File(file_path, "r") as f:
        features = f["last_hidden_cls"][start_idx : start_idx + batch_size]
        paths = f["paths"][start_idx : start_idx + batch_size]
    return features, paths


def build_gpu_index(features, gpu_id=0):
    """构建GPU加速的FAISS索引"""
    d = features.shape[1]

    # 创建CPU索引
    cpu_index = faiss.IndexFlatIP(d)

    # 转换为GPU索引
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)

    # 归一化特征并添加到索引
    features = features.astype(np.float32)  # 确保数据类型正确
    faiss.normalize_L2(features)
    gpu_index.add(features)

    return gpu_index


def normalize_features(features):
    """归一化特征向量"""
    return features / torch.norm(features, dim=1, keepdim=True)


def find_contrasting_pairs_batch(
    clip_features,
    dino_features,
    paths,
    target_pairs=100,
    batch_size=100000,
    clip_high_thresh=0.9,
    clip_low_thresh=0.2,
    dino_high_thresh=0.95,
    dino_low_thresh=0.5,
    k=2048,
    gpu_id=0,
):
    """使用GPU在批次数据中找到对比的图片对"""
    print("Building GPU indices...")
    clip_index = build_gpu_index(clip_features, gpu_id)
    dino_index = build_gpu_index(dino_features, gpu_id)

    clip_high_dino_low = []
    dino_high_clip_low = []

    n_queries = min(batch_size // 2, len(clip_features))
    query_indices = random.sample(range(len(clip_features)), n_queries)

    print(f"Processing {n_queries} query points, k={k}")

    # 预先归一化所有特征
    clip_features_gpu = normalize_features(torch.from_numpy(clip_features).cuda(gpu_id))
    dino_features_gpu = normalize_features(torch.from_numpy(dino_features).cuda(gpu_id))

    for idx in tqdm(query_indices):
        # 准备查询向量
        query_clip = clip_features[idx : idx + 1].copy()
        query_dino = dino_features[idx : idx + 1].copy()

        # CLIP空间搜索
        _, clip_neighbors = clip_index.search(query_clip, k)

        # 手动计算余弦相似度
        query_clip_gpu = normalize_features(torch.from_numpy(query_clip).cuda(gpu_id))
        neighbor_clips = clip_features_gpu[clip_neighbors[0]]
        clip_sim = torch.matmul(query_clip_gpu, neighbor_clips.T).cpu().numpy()

        query_dino_gpu = normalize_features(torch.from_numpy(query_dino).cuda(gpu_id))
        neighbor_dinos = dino_features_gpu[clip_neighbors[0]]
        dino_sim_for_clip = torch.matmul(query_dino_gpu, neighbor_dinos.T).cpu().numpy()

        # DINO空间搜索
        _, dino_neighbors = dino_index.search(query_dino, k)

        neighbor_clips = clip_features_gpu[dino_neighbors[0]]
        clip_sim_for_dino = torch.matmul(query_clip_gpu, neighbor_clips.T).cpu().numpy()

        neighbor_dinos = dino_features_gpu[dino_neighbors[0]]
        dino_sim = torch.matmul(query_dino_gpu, neighbor_dinos.T).cpu().numpy()

        # 打印调试信息
        if idx % 100 == 0:
            print(f"\nQuery {idx}:")
            print(
                f"CLIP similarities range: {clip_sim[0].min():.3f} to {clip_sim[0].max():.3f}"
            )
            print(
                f"DINO similarities range: {dino_sim[0].min():.3f} to {dino_sim[0].max():.3f}"
            )

        # 查找对比对
        for j in range(k):
            if (
                clip_sim[0][j] > clip_high_thresh
                and dino_sim_for_clip[0][j] < dino_low_thresh
            ):
                clip_high_dino_low.append(
                    (
                        paths[idx].decode("utf-8"),
                        paths[clip_neighbors[0][j]].decode("utf-8"),
                        float(clip_sim[0][j]),
                        float(dino_sim_for_clip[0][j]),
                    )
                )

            if (
                dino_sim[0][j] > dino_high_thresh
                and clip_sim_for_dino[0][j] < clip_low_thresh
            ):
                dino_high_clip_low.append(
                    (
                        paths[idx].decode("utf-8"),
                        paths[dino_neighbors[0][j]].decode("utf-8"),
                        float(clip_sim_for_dino[0][j]),
                        float(dino_sim[0][j]),
                    )
                )

        if idx % 100 == 0:
            print(
                f"Found {len(clip_high_dino_low)} CLIP-high pairs and {len(dino_high_clip_low)} DINO-high pairs"
            )

        if (
            len(clip_high_dino_low) >= target_pairs
            and len(dino_high_clip_low) >= target_pairs
        ):
            break

    # 清理GPU内存
    torch.cuda.empty_cache()

    return clip_high_dino_low[:target_pairs], dino_high_clip_low[:target_pairs]


def save_results(clip_high_pairs, dino_high_pairs, output_file):
    """将结果保存到CSV文件"""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["type", "image1_path", "image2_path", "clip_similarity", "dino_similarity"]
        )

        for pair in clip_high_pairs:
            writer.writerow(["clip_high_dino_low"] + list(pair))

        for pair in dino_high_pairs:
            writer.writerow(["dino_high_clip_low"] + list(pair))


def main():
    clip_file = "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/datacomp12m/clip-datacomp_train_features.h5"
    dino_file = "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/datacomp12m/dino-datacomp_train_features.h5"
    output_file = "contrasting_pairs.csv"

    batch_size = 100000
    target_pairs = 1000
    gpu_id = 0  # 使用第一个GPU

    # 检查GPU是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Please check your CUDA installation.")

    print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")

    with h5py.File(clip_file, "r") as f:
        total_size = len(f["last_hidden_cls"])

    start_idx = random.randint(0, max(0, total_size - batch_size))

    print(f"Loading features from index {start_idx} to {start_idx + batch_size}...")
    clip_features, paths = load_features_batch(clip_file, start_idx, batch_size)
    dino_features, _ = load_features_batch(dino_file, start_idx, batch_size)

    print(f"Feature shapes: CLIP {clip_features.shape}, DINO {dino_features.shape}")

    print("Finding contrasting pairs...")
    clip_high_pairs, dino_high_pairs = find_contrasting_pairs_batch(
        clip_features,
        dino_features,
        paths,
        target_pairs=target_pairs,
        batch_size=batch_size,
        gpu_id=gpu_id,
    )

    print(f"Saving results to {output_file}...")
    save_results(clip_high_pairs, dino_high_pairs, output_file)

    print(
        f"Found {len(clip_high_pairs)} CLIP-high pairs and {len(dino_high_pairs)} DINO-high pairs"
    )


if __name__ == "__main__":
    main()
