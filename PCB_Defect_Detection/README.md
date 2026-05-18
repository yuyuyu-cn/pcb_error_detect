# PCB 缺陷检测系统

> 数字图像处理 · 期末项目  
> 难度系数 1.5  
> 传统图像处理 + YOLOv8 深度学习

---

## 项目结构

```
PCB_Defect_Detection/
├── main.py                     # Streamlit 主入口
├── requirements.txt
├── core/                       # 传统图像处理模块
│   ├── filters.py              #   均值/高斯/中值/双边滤波
│   ├── enhancement.py          #   直方图均衡/CLAHE/伽马/锐化
│   ├── edge_detection.py       #   Sobel/Canny/Laplacian
│   ├── morphology.py           #   腐蚀/膨胀/开闭运算/梯度/顶帽黑帽
│   └── transforms.py           #   缩放/旋转/翻转/仿射/透视
├── detection/
│   ├── yolo_detector.py        # YOLOv8 检测引擎
│   └── best.pt                 # 预训练权重（需自行放置）
├── applications/
│   ├── defect_report.py        # 功能一：缺陷分类报告
│   └── pass_fail.py            # 功能二：PASS/FAIL 判定
├── utils/
│   └── visualization.py        # 画框、热力图、对比图
├── samples/                    # 测试图片
└── outputs/                    # 检测结果输出
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 放置 YOLO 权重文件
# 从 mehulnaik16/PCB-2.0-YOLOV8-STREAMLIT 下载 best.pt
# 放到 detection/ 目录下

# 3. 启动应用
streamlit run main.py
```

## 功能说明

### 1. 图像预处理（难度 1.0）
传统数字图像处理方法，提供滤波降噪、图像增强、边缘检测、形态学操作、几何变换五大类共 20+ 种算法。

### 2. AI 缺陷检测（难度 1.3）
基于 YOLOv8 的目标分割与定位，检测 PCB 上 6 类缺陷：Mouse Bite、Spur、Spurious Copper、Short、Missing Hole、Open Circuit。

### 3. 检测报告（难度 1.5）
- **功能一**：缺陷分类报告 — 统计各类型缺陷数量、严重程度、位置信息，支持 JSON 导出
- **功能二**：PASS/FAIL 自动判定 — 基于规则引擎模拟产线质检，自动判定良品/不良品

## 技术栈

| 模块 | 技术 |
|------|------|
| 传统图像处理 | OpenCV 4.x + NumPy |
| 深度学习 | YOLOv8 (ultralytics) + PyTorch |
| GUI | Streamlit |
| 可视化 | Pillow + Matplotlib |
