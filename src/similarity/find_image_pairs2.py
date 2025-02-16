import h5py
from tqdm import tqdm
import csv
from pathlib import Path
import torch
from torch.nn import functional as F
from collections import defaultdict

def normalize_features_in_chunks(features, chunk_size=1000000):
    """分批对特征进行归一化"""
    total_size = features.shape[0]
    for i in range(0, total_size, chunk_size):
        end_idx = min(i + chunk_size, total_size)
        features[i:end_idx] = F.normalize(features[i:end_idx], p=2, dim=1)
    return features

def load_features_to_gpu(file_path, device='cuda', load_batch_size=1000000, norm_batch_size=1000000):
    """分批将特征加载到GPU并归一化"""
    with h5py.File(file_path, 'r') as f:
        total_size = len(f['last_hidden_cls'])
        feature_dim = f['last_hidden_cls'].shape[1]
        
        features = torch.empty((total_size, feature_dim), dtype=torch.float32, device=device)
        paths = []
        
        for i in tqdm(range(0, total_size, load_batch_size), desc="Loading features"):
            end_idx = min(i + load_batch_size, total_size)
            batch_features = torch.from_numpy(f['last_hidden_cls'][i:end_idx]).to(device)
            features[i:end_idx] = batch_features
            
            batch_paths = [p.decode('utf-8') for p in f['paths'][i:end_idx]]
            paths.extend(batch_paths)
        
        print("Normalizing features...")
        features = normalize_features_in_chunks(features, norm_batch_size)
        
    return features, paths

def find_contrasting_pairs_streamed(clip_features, dino_features, paths, idx1, idx2, batch_size=10000):
    """流式处理相似度计算和对比对查找"""
    results = []
    n = len(idx1)
    
    for i in tqdm(range(0, n, batch_size), desc="Processing batches"):
        end_i = min(i + batch_size, n)
        clip_batch1 = clip_features[idx1[i:end_i]]
        dino_batch1 = dino_features[idx1[i:end_i]]
        
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            clip_batch2 = clip_features[idx2[j:end_j]]
            dino_batch2 = dino_features[idx2[j:end_j]]
            
            # 计算这个批次的相似度
            clip_sims = torch.mm(clip_batch1, clip_batch2.t())
            dino_sims = torch.mm(dino_batch1, dino_batch2.t())
            
            # 寻找对比对
            mask1 = (clip_sims > 0.7) & (dino_sims < 0.5)
            mask2 = (dino_sims > 0.8) & (clip_sims < 0.5)
            
            for mask, pair_type in [(mask1, 'high_clip_low_dino'), 
                                  (mask2, 'high_dino_low_clip')]:
                pairs = torch.nonzero(mask, as_tuple=False).cpu().numpy()
                for pi, pj in pairs:
                    global_i = idx1[i + pi].item()
                    global_j = idx2[j + pj].item()
                    
                    # 确保path1总是字典序较小的路径，保证一致性
                    path1 = paths[global_i]
                    path2 = paths[global_j]
                    if path1 > path2:
                        path1, path2 = path2, path1
                        # 交换相似度计算的顺序
                        clip_sim = clip_sims[pi, pj].item()
                        dino_sim = dino_sims[pi, pj].item()
                    else:
                        clip_sim = clip_sims[pi, pj].item()
                        dino_sim = dino_sims[pi, pj].item()
                    
                    results.append({
                        'path1': path1,
                        'path2': path2,
                        'clip_sim': clip_sim,
                        'dino_sim': dino_sim,
                        'type': pair_type
                    })
            
            del clip_sims, dino_sims
            torch.cuda.empty_cache()
    
    return results

def find_contrasting_pairs_gpu(clip_file, dino_file, output_file, search_size=50000, 
                             target_pairs_clip=500, target_pairs_dino=1000,
                             device='cuda', continue_search=True):
    """在GPU上进行大规模相似度搜索"""
    print("Loading CLIP features to GPU...")
    clip_features, paths = load_features_to_gpu(clip_file, device)
    
    print("Loading DINO features to GPU...")
    dino_features, _ = load_features_to_gpu(dino_file, device)
    
    total_samples = len(clip_features)
    print(f"Total samples: {total_samples}")
    
    # 使用字典存储找到的pairs和它们的相似度
    found_pairs_info = {}  # (path1, path2) -> {'clip_sim': X, 'dino_sim': Y, 'type': Z}
    found_pairs_by_type = defaultdict(set)
    
    # 加载已有的结果
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 5:  # path1, path2, clip_sim, dino_sim, type
                    # 确保path1是字典序较小的路径
                    path1, path2 = sorted([row[0], row[1]])
                    pair_key = (path1, path2)
                    found_pairs_info[pair_key] = {
                        'clip_sim': float(row[2]),
                        'dino_sim': float(row[3]),
                        'type': row[4]
                    }
                    found_pairs_by_type[row[4]].add(pair_key)
    
    print(f"Loaded {len(found_pairs_info)} existing pairs")
    print(f"High CLIP, Low DINO pairs: {len(found_pairs_by_type['high_clip_low_dino'])}")
    print(f"High DINO, Low CLIP pairs: {len(found_pairs_by_type['high_dino_low_clip'])}")
    
    # 如果是新文件，先写入头
    if not Path(output_file).exists():
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image1_path', 'image2_path', 'clip_similarity', 
                           'dino_similarity', 'pair_type'])
    
    iteration = 0
    while (continue_search or 
           len(found_pairs_by_type['high_clip_low_dino']) < target_pairs_clip or 
           len(found_pairs_by_type['high_dino_low_clip']) < target_pairs_dino):
        
        iteration += 1
        print(f"\nIteration {iteration}, processing batch of size {search_size}")
        print(f"Current counts - CLIP pairs: {len(found_pairs_by_type['high_clip_low_dino'])}, "
              f"DINO pairs: {len(found_pairs_by_type['high_dino_low_clip'])}")
        
        # 随机采样索引
        idx1 = torch.randint(0, total_samples, (search_size,), device=device)
        idx2 = torch.randint(0, total_samples, (search_size,), device=device)
        
        # 流式处理查找对比对
        print("Finding contrasting pairs...")
        new_pairs = find_contrasting_pairs_streamed(
            clip_features, dino_features, paths, 
            idx1, idx2, batch_size=10000
        )
        
        # 处理找到的对比对
        pairs_added = defaultdict(int)
        
        # 批量写入新找到的pairs
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for pair in new_pairs:
                pair_key = (pair['path1'], pair['path2'])
                
                # 如果这个pair已经存在，跳过
                if pair_key in found_pairs_info:
                    continue
                
                # 如果达到目标数且不继续搜索，则跳过该类型的对
                if not continue_search:
                    if (pair['type'] == 'high_clip_low_dino' and 
                        len(found_pairs_by_type['high_clip_low_dino']) >= target_pairs_clip):
                        continue
                    if (pair['type'] == 'high_dino_low_clip' and 
                        len(found_pairs_by_type['high_dino_low_clip']) >= target_pairs_dino):
                        continue
                
                # 保存pair信息
                writer.writerow([
                    pair['path1'], pair['path2'],
                    pair['clip_sim'], pair['dino_sim'],
                    pair['type']
                ])
                
                found_pairs_info[pair_key] = {
                    'clip_sim': pair['clip_sim'],
                    'dino_sim': pair['dino_sim'],
                    'type': pair['type']
                }
                found_pairs_by_type[pair['type']].add(pair_key)
                pairs_added[pair['type']] += 1
        
        print("New pairs in this iteration:")
        print(f"  High CLIP, Low DINO: {pairs_added['high_clip_low_dino']}")
        print(f"  High DINO, Low CLIP: {pairs_added['high_dino_low_clip']}")
        
        if not continue_search:
            clip_target_reached = len(found_pairs_by_type['high_clip_low_dino']) >= target_pairs_clip
            dino_target_reached = len(found_pairs_by_type['high_dino_low_clip']) >= target_pairs_dino
            if clip_target_reached and dino_target_reached:
                print("\nReached both target numbers. Stopping search.")
                break

if __name__ == "__main__":
    clip_file = "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/datacomp12m/clip-datacomp_train_features.h5"
    dino_file = "/pasteur2/u/yuhuiz/yiming/experiments/src/cached_features/datacomp12m/dino-datacomp-new_train_features.h5"
    output_file = "/pasteur2/u/yuhuiz/yiming/experiments/src/similarity/contrasting_pairs_loose.csv"
    
    find_contrasting_pairs_gpu(
        clip_file=clip_file,
        dino_file=dino_file,
        output_file=output_file,
        search_size=100000,        # 增加搜索批次大小
        target_pairs_clip=100,    # CLIP相似但DINO不相似的目标对数
        target_pairs_dino=100,   # DINO相似但CLIP不相似的目标对数
        continue_search=False,     # 即使达到目标数量也继续搜索
        device='cuda'
    )