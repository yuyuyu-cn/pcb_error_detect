"""应用功能二：良品/不良品自动判定。

模拟产线质检流程，基于检测结果自动判定 PASS 或 FAIL。
"""

from dataclasses import dataclass, field
from typing import List
from applications.defect_report import DefectReport


# ── 判定规则定义 ──
# 出现以下任一类型缺陷，直接判 FAIL
CRITICAL_DEFECTS = {"Short", "Open Circuit"}
# 单类缺陷超过此数量阈值，判 FAIL
DEFECT_COUNT_THRESHOLDS = {
    "Short": 0,           # 一旦出现就 FAIL
    "Open Circuit": 0,    # 一旦出现就 FAIL
    "Missing Hole": 2,    # 超过 2 个缺孔判 FAIL
    "Mouse Bite": 3,      # 超过 3 个鼠咬判 FAIL
    "Spur": 4,            # 超过 4 个毛刺判 FAIL
    "Spurious Copper": 3, # 超过 3 个杂铜判 FAIL
}
# 缺陷总数超过此阈值，判 FAIL
TOTAL_DEFECT_THRESHOLD = 6


@dataclass
class PassFailResult:
    """PASS/FAIL 判定结果。"""
    verdict: str           # "PASS" 或 "FAIL"
    reason: str            # 判定理由
    critical_count: int = 0
    total_defects: int = 0
    details: List[str] = field(default_factory=list)

    @classmethod
    def evaluate(cls, report: DefectReport) -> "PassFailResult":
        """根据检测报告执行 PASS/FAIL 判定。

        Args:
            report: DefectReport 实例

        Returns:
            PassFailResult
        """
        total = report.total_defects
        summary = report.summary
        critical_count = 0
        reasons = []

        # 规则 1: 严重缺陷检查
        for defect_type in CRITICAL_DEFECTS:
            count = summary.get(defect_type, 0)
            if count > 0:
                critical_count += count
                reasons.append(f"检测到 {count} 个 {defect_type}（严重缺陷）")

        # 规则 2: 单类缺陷超阈值
        for defect_type, threshold in DEFECT_COUNT_THRESHOLDS.items():
            count = summary.get(defect_type, 0)
            if threshold > 0 and count > threshold:
                reasons.append(
                    f"{defect_type} 数量 {count} 超过阈值 {threshold}"
                )

        # 规则 3: 总数超阈值
        if total > TOTAL_DEFECT_THRESHOLD:
            reasons.append(
                f"缺陷总数 {total} 超过阈值 {TOTAL_DEFECT_THRESHOLD}"
            )

        # 判定结果
        if reasons:
            verdict = "FAIL"
            reason = "；".join(reasons)
        else:
            verdict = "PASS"
            if total == 0:
                reason = "未检测到任何缺陷，PCB 质量合格"
            else:
                reason = (
                    f"检测到 {total} 个缺陷，均在允许范围内，判定合格"
                )

        return cls(
            verdict=verdict,
            reason=reason,
            critical_count=critical_count,
            total_defects=total,
            details=reasons if reasons else [reason],
        )

    def to_dict(self):
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "critical_count": self.critical_count,
            "total_defects": self.total_defects,
            "details": self.details,
        }
