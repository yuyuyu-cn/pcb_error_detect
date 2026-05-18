"""几何变换模块 — 缩放、旋转、仿射、透视变换。"""

import cv2
import numpy as np


def resize_image(image, width=None, height=None, scale=1.0):
    """缩放图像"""
    if width and height:
        return cv2.resize(image, (width, height))
    if scale != 1.0:
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w*scale), int(h*scale)))
    return image


def rotate_image(image, angle, center=None, scale=1.0):
    """旋转图像（自动扩展边界）"""
    h, w = image.shape[:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int(h*sin + w*cos)
    nH = int(h*cos + w*sin)
    M[0, 2] += nW/2 - center[0]
    M[1, 2] += nH/2 - center[1]
    return cv2.warpAffine(image, M, (nW, nH))


def affine_transform(image, src_pts, dst_pts):
    """仿射变换 — 3 个对应点"""
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h))


def perspective_transform(image, src_pts, dst_pts):
    """透视变换 — 4 个对应点，用于校正 PCB 拍摄角度"""
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, M, (w, h))


def flip_image(image, mode=0):
    """翻转：0-垂直, 1-水平, -1-同时"""
    return cv2.flip(image, mode)
