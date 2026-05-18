"""形态学处理模块 — 腐蚀、膨胀、开闭运算、梯度、顶帽黑帽。"""

import cv2
import numpy as np


def _kernel(shape, size):
    return cv2.getStructuringElement(shape, (size, size))


def erode(image, ksize=3, shape=cv2.MORPH_RECT):
    """腐蚀 — 消除细小噪点，收缩前景区域"""
    return cv2.erode(image, _kernel(shape, ksize), iterations=1)


def dilate(image, ksize=3, shape=cv2.MORPH_RECT):
    """膨胀 — 填充细小空洞，扩展前景区域"""
    return cv2.dilate(image, _kernel(shape, ksize), iterations=1)


def opening(image, ksize=3):
    """开运算 = 腐蚀→膨胀 — 去噪点、断开细连接"""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _kernel(cv2.MORPH_RECT, ksize))


def closing(image, ksize=3):
    """闭运算 = 膨胀→腐蚀 — 填空洞、连接断线，PCB 线路修复常用"""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _kernel(cv2.MORPH_RECT, ksize))


def morph_gradient(image, ksize=3):
    """形态学梯度 = 膨胀 - 腐蚀 — 提取物体轮廓"""
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, _kernel(cv2.MORPH_RECT, ksize))


def top_hat(image, ksize=9):
    """顶帽 = 原图 - 开运算 — 提取亮背景上的暗细节"""
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, _kernel(cv2.MORPH_RECT, ksize))


def black_hat(image, ksize=9):
    """黑帽 = 闭运算 - 原图 — 提取暗背景上的亮细节"""
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, _kernel(cv2.MORPH_RECT, ksize))


MORPHS = {
    "腐蚀 (Erode)": erode,
    "膨胀 (Dilate)": dilate,
    "开运算 (Open)": opening,
    "闭运算 (Close)": closing,
    "形态学梯度": morph_gradient,
    "顶帽 (TopHat)": top_hat,
    "黑帽 (BlackHat)": black_hat,
}
