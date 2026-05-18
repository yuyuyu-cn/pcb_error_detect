"""应用功能一：缺陷分类报告。

基于 YOLO 检测结果，生成结构化的缺陷分析报告。
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class DefectInstance:
    """单个缺陷实例。"""
    defect_type: str
    confidence: float
    bbox: List[float]   # [x1, y1, x2, y2]
    area_px: int = 0
    severity: str = "未知"   # 轻微 / 中等 / 严重

    def __post_init__(self):
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.area_px = int(w * h)
        self.severity = self._estimate_severity()

    def _estimate_severity(self):
        """根据缺陷类型和面积估算严重程度。"""
        # 短路和开路无论大小都算严重
        if self.defect_type in ("Short", "Open Circuit"):
            return "严重"
        # 缺孔面积越大越严重
        if self.defect_type == "Missing Hole":
            return "严重" if self.area_px > 5000 else "中等"
        # 其余按面积判定
        if self.area_px > 8000:
            return "严重"
        elif self.area_px > 3000:
            return "中等"
        return "轻微"


@dataclass
class DefectReport:
    """完整检测报告。"""
    image_name: str = ""
    total_defects: int = 0
    instances: List[DefectInstance] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_detection(cls, image_name, boxes, labels, scores):
        """从 YOLO 检测结果构建报告。"""
        instances = []
        summary = {}
        for box, label, score in zip(boxes, labels, scores):
            inst = DefectInstance(
                defect_type=label,
                confidence=float(score),
                bbox=box,
            )
            instances.append(inst)
            summary[label] = summary.get(label, 0) + 1
        return cls(
            image_name=image_name,
            total_defects=len(instances),
            instances=instances,
            summary=dict(sorted(summary.items(), key=lambda x: -x[1])),
        )

    def to_dict(self):
        """转为字典，供前端展示。"""
        return {
            "image_name": self.image_name,
            "total_defects": self.total_defects,
            "summary": self.summary,
            "instances": [
                {
                    "type": i.defect_type,
                    "confidence": round(i.confidence, 3),
                    "bbox": [round(v, 1) for v in i.bbox],
                    "area_px": i.area_px,
                    "severity": i.severity,
                }
                for i in self.instances
            ],
        }

    def get_severity_summary(self):
        """严重程度分布统计。"""
        sev = {"严重": 0, "中等": 0, "轻微": 0}
        for inst in self.instances:
            sev[inst.severity] += 1
        return sev
