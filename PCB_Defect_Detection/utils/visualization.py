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
