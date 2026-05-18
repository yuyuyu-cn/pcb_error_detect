"""图像滤波模块 — 均值、高斯、中值、双边滤波。"""

import cv2
import numpy as np


def mean_filter(image, ksize=5):
    """均值滤波 — 适用于随机噪声"""
    return cv2.blur(image, (ksize, ksize))


def gaussian_filter(image, ksize=5, sigma=0):
    """高斯滤波 — 适用于高斯噪声，保留边缘优于均值"""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_filter(image, ksize=5):
    """中值滤波 — 适用于椒盐噪声，PCB 图像常用"""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """双边滤波 — 保边去噪，保护焊点和线路边缘"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


FILTERS = {
    "均值滤波": mean_filter,
    "高斯滤波": gaussian_filter,
    "中值滤波": median_filter,
    "双边滤波": bilateral_filter,
}
