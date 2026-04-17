# **Nailoong-ResNet50-Classifier**
## 奶龙分类器

## 中文/English/日本語

## **中文**

基于 PyTorch 和 ResNet50 架构实现的图像分类器，用于快速鉴定图片中是否包含大家都喜欢的小奶龙。项目包含了从模型训练，支持混合精度与余弦退火，批量推理的完整流程。

### **✨ 主要功能**

* **自定义训练**: 支持动态计算类别权重，自带数据增强（翻转、亮度/对比度调整）。  
* **高性能**: 使用 autocast 混合精度训练，大幅降低显存占用并提升速度。  
* **批量鉴定**: 提供开箱即用的推理脚本，一键扫描文件夹内的所有图片并输出置信度。

### **🚀 快速开始**

**1\. 安装依赖**

pip install torch torchvision pillow tqdm

**2\. 准备数据**

在项目根目录创建 dataset 文件夹，结构如下：

dataset/  
├── train/  
│   ├── class\_1/ (例如：不是奶龙的图片)  
│   └── class\_0/ (例如：是奶龙的图片)  
└── val/  
    ├── class\_1/  
    └── class\_0/

**3\. 训练模型**

python train_nailoong.py

*训练完成后，最佳模型将保存在 out/ 目录下。*

**4\. 批量测试**

在根目录创建 try/ 文件夹，放入需要测试的图片，然后运行：

python nailong_predict.py

## **English**

An image classifier based on PyTorch and the ResNet50 architecture, designed to quickly identify whether an image contains "Nailoong". This repository includes the complete pipeline from model training (supporting mixed precision and cosine annealing) to batch inference.

### **✨ Features**

* **Custom Training**: Supports dynamic class weight calculation with built-in data augmentation.  
* **High Performance**: Utilizes autocast mixed precision training to reduce VRAM usage and boost training speed.  
* **Batch Inference**: Ready-to-use inference script to scan all images in a directory and output confidence scores.

### **🚀 Quick Start**

**1\. Install Dependencies**

pip install torch torchvision pillow tqdm

**2\. Prepare Data**

Create a dataset folder in the root directory with the following structure:

dataset/  
├── train/  
│   ├── class\_1/ (e.g., Non-Nailoong images)  
│   └── class\_0/ (e.g., Nailoong images)  
└── val/  
    ├── class\_1/  
    └── class\_0/

**3\. Train the Model**

python train_nailoong.py

*After training, the best model will be saved in the out/ directory.*

**4\. Batch Testing**

Create a try/ folder, place the images you want to test inside, and run:

python nailong_predict.py

## **日本語**

PyTorch と ResNet50 アーキテクチャをベースにした画像分類器で、画像に「乳龍（ミルクドラゴン）」が含まれているかを迅速に判定します。モデルの学習（混合精度とコサインアニーリングをサポート）からバッチ推論までの完全なパイプラインが含まれています。

### **✨ 主な機能**

* **カスタム学習**: 動的なクラスター重み計算をサポートし、データ拡張機能を内蔵。  
* **高性能**: autocast 混合精度学習を使用し、VRAM の使用量を削減して学習速度を向上。  
* **バッチ判定**: ディレクトリ内のすべての画像をスキャンし、信頼度を出力する推論スクリプトを提供。

### **🚀 クイックスタート**

**1\. 依存関係のインストール**

pip install torch torchvision pillow tqdm

**2\. データの準備**

プロジェクトのルートディレクトリに dataset フォルダを作成し、以下の構造にします：

dataset/  
├── train/  
│   ├── class\_1/ (例：乳龍ではない画像)  
│   └── class\_0/ (例：乳龍の画像)  
└── val/  
    ├── class\_1/  
    └── class\_0/

**3\. モデルの学習**

python train_nailoong.py

*学習完了後、最適なモデルは out/ ディレクトリに保存されます。*

**4\. バッチテスト**

ルートディレクトリに try/ フォルダを作成し、テストしたい画像を入れて以下を実行します：

python nailong_predict.py  
