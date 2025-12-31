import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import random
import numpy as np
from pathlib import Path
import tensorflow as tf
import faiss
import threading

# ================= 配置区域 =================
# 确保路径与你的实际环境一致
DATASET_DIR = r"F:\computer_vision\0" 
DELG_MODEL_DIR = r"F:\computer_vision\local_and_global"
CACHE_FILE = "features_cache.npz"
EMBEDDING_DIM = 2048
NUM_IMAGES = 1000 
TOP_K = 3

class GeoLocateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Google Landmark Retrieval System (Han Shixuan - CS Class 1)")
        self.root.geometry("1100x650") # 稍微调大一点窗口以适应内容
        
        # 数据存储变量
        self.embedding_matrix = None
        self.valid_paths = []
        self.index = None
        self.model = None
        self.signature_fn = None
        self.image_refs = [] # 防止图片被垃圾回收机制清除

        # --- 1. 顶部控制栏 ---
        control_frame = tk.Frame(root, pady=15, bg="#f0f0f0")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttk.Button(control_frame, text="1. 初始化/加载引擎", command=self.start_loading_thread)
        self.btn_load.pack(side=tk.LEFT, padx=20)

        self.btn_search = ttk.Button(control_frame, text="2. 随机抽取并检索", command=self.perform_search, state=tk.DISABLED)
        self.btn_search.pack(side=tk.LEFT, padx=20)

        self.status_label = tk.Label(control_frame, text="状态: 等待初始化...", fg="gray", bg="#f0f0f0", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=20)

        # --- 2. 图片展示区域 ---
        display_frame = tk.Frame(root)
        display_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # 左侧：Query 图片
        self.query_frame = tk.LabelFrame(display_frame, text="查询图片 (Query)", width=350, font=("Arial", 11, "bold"))
        self.query_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.lbl_query_img = tk.Label(self.query_frame, text="暂无图片\n请先点击初始化", bg="#e0e0e0")
        self.lbl_query_img.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.lbl_query_name = tk.Label(self.query_frame, text="", wraplength=300, fg="#333")
        self.lbl_query_name.pack(side=tk.BOTTOM, pady=10)

        # 右侧：结果图片容器
        self.results_frame = tk.LabelFrame(display_frame, text=f"Top-{TOP_K} 相似结果 ", width=700, font=("Arial", 11, "bold"))
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 创建3个结果展示位
        self.result_widgets = []
        for i in range(TOP_K):
            frame = tk.Frame(self.results_frame)
            frame.pack(side=tk.LEFT, expand=True, padx=5, fill=tk.BOTH)
            
            # 排名标签
            lbl_rank = tk.Label(frame, text=f"Rank {i+1}", font=("Arial", 10, "bold"), fg="#555")
            lbl_rank.pack(pady=5)
            
            # 图片标签
            lbl_img = tk.Label(frame, text="等待检索...", bg="#e0e0e0", width=20, height=10)
            lbl_img.pack(expand=True, fill=tk.BOTH, padx=5)
            
            # 分数和文件名标签
            lbl_score = tk.Label(frame, text="", fg="blue", justify=tk.CENTER, wraplength=180)
            lbl_score.pack(side=tk.BOTTOM, pady=10)
            
            self.result_widgets.append((lbl_img, lbl_score))

    def update_status(self, text, color="black"):
        self.status_label.config(text=f"状态: {text}", fg=color)

    def start_loading_thread(self):
        """使用线程加载，防止界面卡死"""
        self.btn_load.config(state=tk.DISABLED)
        self.update_status("正在加载数据，请稍候...", "orange")
        threading.Thread(target=self.load_engine, daemon=True).start()

    def load_engine(self):
        try:
            # 1. 尝试加载缓存
            if os.path.exists(CACHE_FILE):
                self.root.after(0, lambda: self.update_status("发现缓存 features_cache.npz，正在读取...", "blue"))
                data = np.load(CACHE_FILE, allow_pickle=True)
                self.embedding_matrix = data['embeddings']
                self.valid_paths = data['paths']
            else:
                self.root.after(0, lambda: self.update_status("未发现缓存，请先运行 geolocate_fast.py 生成缓存！", "red"))
                messagebox.showerror("缺少缓存", f"请先运行 geolocate_fast.py 生成 {CACHE_FILE} 文件！\n为了保证GUI流畅，本程序只读取缓存文件。")
                self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))
                return

            # 2. 构建索引
            self.root.after(0, lambda: self.update_status("正在构建 Faiss 索引...", "blue"))
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.index.add(self.embedding_matrix)

            # 3. 完成
            msg = f"系统就绪！库中包含 {len(self.valid_paths)} 张图像"
            self.root.after(0, lambda: self.update_status(msg, "green"))
            self.root.after(0, lambda: self.btn_search.config(state=tk.NORMAL))

        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"错误: {str(e)}", "red"))
            print(f"Error details: {e}")

    def resize_image(self, path, max_size=(400, 400)):
        """调整图片大小以适应窗口"""
        try:
            img = Image.open(path)
            # 使用缩略图模式保持比例
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"无法加载图片 {path}: {e}")
            return None

    def perform_search(self):
        if not self.index:
            return

        self.image_refs = [] # 清空引用，防止图片不显示

        # 1. 随机选图
        query_path = str(random.choice(self.valid_paths))
        query_idx = list(self.valid_paths).index(query_path)
        
        # 2. 显示 Query 图
        tk_img = self.resize_image(query_path, max_size=(300, 300))
        if tk_img:
            self.lbl_query_img.config(image=tk_img, text="", bg="white")
            self.lbl_query_name.config(text=os.path.basename(query_path))
            self.image_refs.append(tk_img)

        # 3. 搜索 (核心修改：搜索 TOP_K + 1 张，为了有空间排除自己)
        query_vec = self.embedding_matrix[query_idx].reshape(1, -1)
        # 搜索 4 张，因为第 1 张肯定是它自己
        distances, indices = self.index.search(query_vec, TOP_K + 1)

        # 4. 过滤并显示结果
        display_count = 0
        
        for dist, idx in zip(distances[0], indices[0]):
            match_path = self.valid_paths[idx]
            
            # --- 核心逻辑：如果路径和 Query 一样，直接跳过 ---
            if match_path == query_path:
                continue
            
            # 如果已经显示够了 TOP_K 个，就停止
            if display_count >= TOP_K:
                break

            # 计算分数
            score = 1 - dist / 4
            
            # 获取对应的 UI 组件
            lbl_img, lbl_score = self.result_widgets[display_count]
            
            # 显示图片
            res_tk_img = self.resize_image(match_path)
            if res_tk_img:
                lbl_img.config(image=res_tk_img, text="", bg="white")
                self.image_refs.append(res_tk_img)
            else:
                lbl_img.config(text="无法加载", bg="#e0e0e0")

            # 显示分数 (根据相似度改变颜色)
            # 蓝色表示高相似，红色表示低相似
            text_color = "blue" if score > 0.6 else "red"
            info_text = f"相似度: {score:.4f}\n{os.path.basename(match_path)}"
            lbl_score.config(text=info_text, fg=text_color)
            
            display_count += 1

# 启动应用
if __name__ == "__main__":
    try:
        root = tk.Tk()
        # 设置高DPI支持，防止在高分屏上模糊 (Windows 10/11)
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
            
        app = GeoLocateApp(root)
        root.mainloop()
    except Exception as e:
        print(f"启动失败: {e}")
        input("按回车键退出...")