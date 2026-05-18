"""边缘检测模块 — Sobel, Canny, Laplacian。"""

import cv2
import numpy as np


def to_gray(image):
    return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sobel_edge(image, dx=1, dy=1, ksize=3):
    """Sobel 边缘检测 — 一阶梯度"""
    gray = to_gray(image)
    grad = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
    grad = np.absolute(grad)
    return np.uint8(grad)


def sobel_combined(image, ksize=3):
    """Sobel 综合梯度（X+Y方向）"""
    gx = sobel_edge(image, 1, 0, ksize)
    gy = sobel_edge(image, 0, 1, ksize)
    return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)


def canny_edge(image, low=50, high=150):
    """Canny 边缘检测 — 完整的边缘提取流程"""
    gray = to_gray(image)
    return cv2.Canny(gray, low, high)


def laplacian_edge(image, ksize=3):
    """Laplacian 边缘检测 — 二阶导数"""
    gray = to_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return np.uint8(np.absolute(lap))


EDGES = {
    "Sobel X": lambda img: sobel_edge(img, 1, 0),
    "Sobel Y": lambda img: sobel_edge(img, 0, 1),
    "Sobel 综合": sobel_combined,
    "Canny": canny_edge,
    "Laplacian": laplacian_edge,
}
