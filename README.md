# Intelligent Landmark Recognition System (智能地标识别系统)

**Course:** Computer Vision (CS460D1), Macau University of Science and Technology  
**Date:** December 2025  

##  Team Members
* **Zhao Longyu** (5250001412)
* **Wen Haoyu** (1220008351)

##  Project Overview
This project implements a robust image retrieval system capable of identifying landmarks from "noisy" user-generated content. It integrates:
* **DELG (Deep Local and Global features):** For attention-based feature extraction.
* **FAISS:** For high-speed similarity search.
* **GUI:** A Tkinter-based desktop application for real-time demonstration.

---

## 1) Requirements: Software
Please ensure you have **Python 3.8+** installed. 
You can install the necessary dependencies using the provided `requirements.txt` file.

**Command:**
```bash
pip install -r requirements.txt

Core Dependencies:

numpy

tensorflow>=2.5.0

faiss-cpu

pillow

tqdm

2) Pretrained Models
This system requires the DELG model and a generated feature database.

DELG Model: The code utilizes the DELG model (ResNet-50 based) for feature extraction.

Feature Database (features_cache.npz): * This file contains the pre-computed feature vectors for the landmark dataset.

Note: We have uploaded a pre-generated features_cache.npz. The system will automatically detect and load this file to skip the time-consuming feature extraction process.

3) Preparation for Testing
Please follow the steps below to run the code.

Note: Please rename the downloaded files to remove (1) if present (e.g., rename geolocate_fast(1).py to geolocate_fast.py).

Step 1: Feature Extraction (Optional if cache exists)
Run this script to process the dataset and generate the model parameters (features_cache.npz).

Command:

Bash

python geolocate_fast.py
Step 2: Run the GUI Demo
Start the graphical interface to perform landmark retrieval.

Command:

Bash

python geolocate_gui.py
Operation: 1. Click button "1. 初始化/加载引擎" (Initialize/Load Engine). 2. Click button "2. 随机抽取并检索" (Random Query) to see the Top-3 results.
