"""图像增强模块 — 直方图均衡化、CLAHE、伽马校正、锐化。"""

import cv2
import numpy as np


def histogram_equalization(image):
    """全局直方图均衡化 — 拉伸对比度"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return cv2.equalizeHist(image)


def clahe(image, clip_limit=2.0, tile_size=8):
    """自适应直方图均衡化 (CLAHE) — 局部对比度增强，PCB 检测必用"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=(tile_size, tile_size))
        l = clahe_obj.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit,
                                 tileGridSize=(tile_size, tile_size))
    return clahe_obj.apply(image)


def gamma_correction(image, gamma=1.0):
    """伽马校正 — gamma<1 提亮暗部，gamma>1 压暗亮部"""
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def sharpen(image, strength=1.0):
    """Unsharp Masking 锐化"""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def laplacian_sharpen(image):
    """Laplacian 锐化"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else lap


ENHANCEMENTS = {
    "直方图均衡化": histogram_equalization,
    "CLAHE 自适应": clahe,
    "伽马校正": gamma_correction,
    "Unsharp 锐化": sharpen,
    "Laplacian 锐化": laplacian_sharpen,
}
