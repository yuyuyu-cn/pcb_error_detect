"""YOLOv8 缺陷检测引擎。

使用方法:
    detector = YOLODetector("best.pt")
    boxes, labels, scores = detector.detect(image)
"""

import numpy as np
import cv2


class YOLODetector:
    """YOLOv8 PCB 缺陷检测器。

    支持两种后端：
    - ultralytics: 原生 YOLOv8（推荐，需 pip install ultralytics）
    - opencv_dnn: OpenCV DNN 模块（仅支持 ONNX 导出模型）
    """

    # 缺陷类别映射（与 DeepPCB 数据集一致）
    CLASS_NAMES = {
        0: "Mouse Bite",
        1: "Spur",
        2: "Spurious Copper",
        3: "Short",
        4: "Missing Hole",
        5: "Open Circuit",
    }

    def __init__(self, model_path="best.pt", backend="ultralytics",
                 conf_threshold=0.25, iou_threshold=0.45):
        """初始化检测器。

        Args:
            model_path: 模型权重路径 (.pt 或 .onnx)
            backend: "ultralytics" 或 "opencv_dnn"
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
        """
        self.model_path = model_path
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型。"""
        if self.backend == "ultralytics":
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self._predict = self._predict_ultralytics
                print(f"[YOLO] 已加载 ultralytics 模型: {self.model_path}")
            except ImportError:
                print("[YOLO] ultralytics 未安装，尝试 OpenCV DNN 后端...")
                self.backend = "opencv_dnn"
                self._load_model_opencv()
        elif self.backend == "opencv_dnn":
            self._load_model_opencv()
        else:
            raise ValueError(f"未知后端: {self.backend}")

    def _load_model_opencv(self):
        """使用 OpenCV DNN 加载 ONNX 模型。"""
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self._predict = self._predict_opencv
        print(f"[YOLO] 已加载 OpenCV DNN 模型: {self.model_path}")

    def detect(self, image: np.ndarray):
        """对图像进行缺陷检测。

        Args:
            image: BGR 或 RGB numpy 图像

        Returns:
            (boxes, labels, scores)
            boxes:  [[x1, y1, x2, y2], ...]  单位：像素
            labels: ["Short", "Open Circuit", ...]
            scores: [0.95, 0.87, ...]
        """
        return self._predict(image)

    def _predict_ultralytics(self, image):
        results = self.model(image, conf=self.conf_threshold,
                             iou=self.iou_threshold, verbose=False)
        boxes, labels, scores = [], [], []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf >= self.conf_threshold:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.CLASS_NAMES.get(cls_id, f"Class_{cls_id}"))
                    scores.append(conf)
        return boxes, labels, scores

    def _predict_opencv(self, image):
        """OpenCV DNN 推理（需图像预处理和输出解析）。"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640),
                                      swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        # 简易输出解析（实际使用时需根据模型输出格式调整）
        boxes, labels, scores = [], [], []
        for det in output[0]:
            conf = float(det[4])
            if conf < self.conf_threshold:
                continue
            x, y, bw, bh = det[:4]
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            cls_id = int(det[5])
            boxes.append([x1, y1, x2, y2])
            labels.append(self.CLASS_NAMES.get(cls_id, f"Class_{cls_id}"))
            scores.append(conf)
        return boxes, labels, scores

    def get_defect_summary(self, labels):
        """统计各类缺陷数量。

        Returns:
            dict: {"Short": 3, "Open Circuit": 1, ...}
        """
        summary = {}
        for name in labels:
            summary[name] = summary.get(name, 0) + 1
        return dict(sorted(summary.items(), key=lambda x: -x[1]))
