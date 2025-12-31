import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import faiss
from tqdm import tqdm

# ----------------- 配置 -----------------
DATASET_DIR = r"F:\computer_vision\0" 
DELG_MODEL_DIR = r"F:\computer_vision\local_and_global"
# 特征保存的文件名
CACHE_FILE = "features_cache.npz"

EMBEDDING_DIM = 2048
NUM_IMAGES = 1000
TOP_K = 3

# ----------------- 1. 加载或计算特征库 -----------------

# 检查是否存在“缓存文件”
if os.path.exists(CACHE_FILE):
    print(f"发现缓存文件 {CACHE_FILE}，正在直接加载...")
    # 直接读取，跳过漫长的模型计算
    data = np.load(CACHE_FILE, allow_pickle=True)
    embedding_matrix = data['embeddings']
    valid_paths = data['paths']
    print(f"加载成功！库中共有 {len(valid_paths)} 张图片的特征。")

else:
    print("未发现缓存，开始加载模型进行计算（仅需执行一次）...")
    
    # === 加载模型 ===
    model = tf.saved_model.load(DELG_MODEL_DIR)
    signature_fn = model.signatures['serving_default']
    
    # === 准备图片列表 ===
    print(f"正在搜索图片...")
    all_image_paths = list(Path(DATASET_DIR).rglob("*.jpg"))
    if len(all_image_paths) > NUM_IMAGES:
        image_paths = random.sample(all_image_paths, NUM_IMAGES)
    else:
        image_paths = all_image_paths
        
    # === 提取特征函数 (优化版) ===
    def extract_global_descriptor(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_arr = np.array(img)
        except:
            return None
        
        img_tensor = tf.convert_to_tensor(img_arr, tf.uint8)
        
        result = signature_fn(
            input_image=img_tensor,
            # [加速点] 改为只用 1.0 原尺度，速度提升 3 倍
            input_scales=tf.constant([1.0], tf.float32), 
            input_abs_thres=tf.constant(0.0, tf.float32),
            input_max_feature_num=tf.constant(1000, tf.int32)
        )
        embedding = tf.nn.l2_normalize(tf.reduce_sum(result['global_descriptors'], axis=0), axis=0)
        return embedding.numpy()

    # === 循环提取 ===
    valid_embeddings = []
    valid_paths = []
    
    print("开始提取特征（已启用单尺度加速）...")
    for path in tqdm(image_paths):
        emb = extract_global_descriptor(str(path))
        if emb is not None:
            valid_embeddings.append(emb)
            valid_paths.append(str(path))
            
    embedding_matrix = np.array(valid_embeddings).astype('float32')
    
    # === 保存结果 ===
    print(f"正在保存特征到 {CACHE_FILE} ...")
    np.savez(CACHE_FILE, embeddings=embedding_matrix, paths=valid_paths)
    print("保存完毕！下次运行将瞬间完成。")

# ----------------- 2. 构建索引与查询 -----------------

print("构建索引...")
index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(embedding_matrix)

# 随机选一张图测试
query_path = str(random.choice(valid_paths))
print(f"\n查询图: {query_path}")

# 注意：查询时也需要提取特征（如果之前没加载模型，这里需要临时加载一下，或者直接从矩阵里取）
# 为演示简单，我们直接从刚才加载的矩阵里找这张图对应的向量
query_index = list(valid_paths).index(query_path)
query_vec = embedding_matrix[query_index].reshape(1, -1)

distances, indices = index.search(query_vec, TOP_K)

print("\n" + "="*30)
print(f"最相似的 {TOP_K} 张图：")
for dist, idx in zip(distances[0], indices[0]):
    score = 1 - dist / 4
    print(f"-> [相似度: {score:.4f}] {valid_paths[idx]}")