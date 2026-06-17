"""可视化工具：画框、标注、热力图、对比图。"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFECT_COLORS = {
    "Mouse Bite":       (255, 140, 50),
    "Spur":             (250, 70,  90),
    "Spurious Copper":  (245, 180, 40),
    "Short":            (230, 50,  50),
    "Missing Hole":     (60,  200, 200),
    "Open Circuit":     (180, 100, 255),
    "default":          (200, 200, 200),
}

try:
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 14)
except Exception:
    try:
        _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        _FONT = ImageFont.load_default()


def draw_boxes_pillow(image, boxes, labels=None, scores=None, line_width=2):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    if labels is None:
        labels = ["Defect"] * len(boxes)
    if scores is None:
        scores = [None] * len(boxes)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = labels[i] if i < len(labels) else "Defect"
        score = scores[i] if i < len(scores) else None
        color = DEFECT_COLORS.get(label, DEFECT_COLORS["default"])
        for offset in range(line_width):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)
        text = f"{label}" if score is None else f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=_FONT)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        label_y = y1 - th - 6
        if label_y < 0:
            label_y = y1 + 2
        draw.rectangle([x1-1, label_y-2, x1+tw+5, label_y+th+2], fill=color)
        draw.text((x1+2, label_y), text, fill=(255, 255, 255), font=_FONT)
    return np.array(pil_img)


def draw_heatmap(image, mask, alpha=0.45):
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    if len(mask.shape) == 2:
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    else:
        heatmap = mask
    return cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)


def generate_defect_density_map(image, boxes, labels=None, blur_sigma=45):
    """根据缺陷框生成密度热力图，叠加到原图上。

    原理：将每个缺陷框的中心或区域映射到一张空白的灰度密度图上，
    用高斯模糊扩散成连续热力场，再通过 JET colormap 渲染并叠加。

    Args:
        image: BGR 图像
        boxes: [[x1,y1,x2,y2], ...]
        labels: 缺陷类型列表，用于加权
        blur_sigma: 高斯模糊 sigma，越大越扩散

    Returns:
        BGR 叠加图
    """
    h, w = image.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    severity_weight = {
        "Short": 3.0, "Open Circuit": 3.0,
        "Missing Hole": 2.0, "Mouse Bite": 1.5,
        "Spur": 1.0, "Spurious Copper": 1.0,
    }

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        weight = severity_weight.get(labels[i], 1.0) if labels else 1.0
        # 在框的中心区域写入权重
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max((x2 - x1) // 2, (y2 - y1) // 2, 15)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        density[mask] += weight

    if density.max() > 0:
        density = (density / density.max() * 255).astype(np.uint8)
    density = cv2.GaussianBlur(density, (0, 0), blur_sigma)

    heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)

    # 叠加原缺陷框（细线）
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = labels[i] if labels else "defect"
        color = DEFECT_COLORS.get(label, DEFECT_COLORS["default"])
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return result


def create_comparison(original, processed, labels=("Original", "Processed"), vertical=False):
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    if vertical:
        target_w = max(w1, w2)
        from cv2 import resize as cv_resize
        original = cv_resize(original, (target_w, int(h1*target_w/w1))) if w1 != target_w else original
        processed = cv_resize(processed, (target_w, int(h2*target_w/w2))) if w2 != target_w else processed
    else:
        target_h = max(h1, h2)
        from cv2 import resize as cv_resize
        original = cv_resize(original, (int(w1*target_h/h1), target_h)) if h1 != target_h else original
        processed = cv_resize(processed, (int(w2*target_h/h2), target_h)) if h2 != target_h else processed
    axis = 0 if vertical else 1
    return np.concatenate([original, processed], axis=axis)
