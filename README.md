
# Intelligent Landmark Recognition System (智能地标识别系统)

Course: Computer Vision (CS460D1), Macau University of Science and Technology  
Date: December 2025
Team Members
Zhao Longyu (5250001412)
Wen Haoyu (1220008351)
## Project Overview
This project implements a robust image retrieval system capable of identifying landmarks from "noisy" user-generated content. It integrates:
* **DELG (Deep Local and Global features):** For attention-based feature extraction.
* **FAISS:** For high-speed similarity search.
* **GUI:** A Tkinter-based desktop application for real-time demonstration.
# Introduction of main codes 
geolocate_fast(1).py文件用于启动图片数据集预处理的工作。
geolocate_fast(1).py运行后生成features_cache(1).npz后缀文件以保存模型对于图片的识别数据；
geolocate_gui(1).py运行后启动图形化页面，在本地目录中寻找features_cache(1).npz文件并生成地标相似度成果。
