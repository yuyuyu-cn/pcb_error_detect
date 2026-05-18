"""PCB 缺陷检测系统 — 主入口。

运行方式:
    streamlit run main.py

功能:
    1. 图像预处理（传统数字图像处理方法）
    2. AI 缺陷检测（YOLOv8）
    3. 缺陷分类报告 + PASS/FAIL 自动判定
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import sys
import time
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent))

from core.filters import FILTERS
from core.enhancement import ENHANCEMENTS
from core.edge_detection import EDGES
from core.morphology import MORPHS
from core.transforms import (
    resize_image, rotate_image, flip_image,
)
from detection.yolo_detector import YOLODetector
from applications.defect_report import DefectReport
from applications.pass_fail import PassFailResult
from utils.visualization import draw_boxes_pillow, create_comparison


# ═══════════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# 暗色调 CSS
# ═══════════════════════════════════════════════════════════════
DARK_CSS = """
<style>
    /* ── 全局暗色背景 ── */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }

    /* ── 侧边栏 ── */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #c9d1d9;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #8b949e;
    }

    /* ── 主要文字 ── */
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important;
    }
    p, li, span, div {
        color: #c9d1d9;
    }

    /* ── 卡片容器 ── */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .card-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 12px;
    }

    /* ── 指标数字 ── */
    .metric-value {
        font-size: 2.4em;
        font-weight: 700;
        color: #e6edf3;
    }
    .metric-label {
        font-size: 0.85em;
        color: #8b949e;
    }

    /* ── PASS / FAIL 标签 ── */
    .verdict-pass {
        background: #0d3320;
        border: 2px solid #3fb950;
        border-radius: 10px;
        padding: 16px 24px;
        text-align: center;
    }
    .verdict-pass .big {
        font-size: 2em;
        font-weight: 800;
        color: #3fb950;
    }
    .verdict-fail {
        background: #3d0d0d;
        border: 2px solid #f85149;
        border-radius: 10px;
        padding: 16px 24px;
        text-align: center;
    }
    .verdict-fail .big {
        font-size: 2em;
        font-weight: 800;
        color: #f85149;
    }

    /* ── 按钮 ── */
    .stButton > button {
        background: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        border-color: #58a6ff;
        color: #e6edf3;
    }

    /* ── 主要操作按钮 ── */
    .primary-btn > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .primary-btn > button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    }

    /* ── 分割线 ── */
    hr {
        border-color: #21262d !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 会话状态初始化
# ═══════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "original_image": None,     # 原始上传图像 (BGR numpy)
        "processed_image": None,    # 预处理后的图像
        "detection_result": None,   # YOLO 检测结果 (boxes, labels, scores)
        "defect_report": None,      # DefectReport 对象
        "pass_fail": None,          # PassFailResult 对象
        "yolo_model": None,         # YOLODetector 实例
        "history": [],              # 处理历史记录
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════
def load_image(uploaded_file) -> np.ndarray | None:
    """从上传文件读取 BGR 图像。"""
    if uploaded_file is None:
        return None
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def get_display_image(image: np.ndarray) -> np.ndarray:
    """将 BGR 转 RGB 用于 Streamlit 显示。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_yolo_model(model_path: str):
    """懒加载 YOLO 模型。"""
    if st.session_state.yolo_model is None:
        with st.spinner("正在加载 YOLO 模型..."):
            try:
                st.session_state.yolo_model = YOLODetector(model_path)
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                st.info(
                    "请确保已安装 ultralytics: `pip install ultralytics`\n"
                    "并将 best.pt 放到 detection/ 目录下"
                )
                return None
    return st.session_state.yolo_model


# ═══════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h1 style='color:#58a6ff; font-size:1.5em;'>🔬 PCB 缺陷检测</h1>"
        "<p style='color:#8b949e; font-size:0.85em;'>"
        "数字图像处理 · 期末项目</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── 模型配置 ──
    st.markdown("<p style='color:#8b949e; font-size:0.8em;'>模型配置</p>",
                unsafe_allow_html=True)
    model_path = st.text_input(
        "权重路径",
        value="detection/best.pt",
        label_visibility="collapsed",
    )
    conf_thresh = st.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)

    st.divider()

    # ── 导航 ──
    st.markdown("<p style='color:#8b949e; font-size:0.8em;'>功能区</p>",
                unsafe_allow_html=True)
    tab_choice = st.radio(
        "",
        ["📷 图像预处理", "🤖 AI 缺陷检测", "📊 检测报告"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── 图片上传（全局） ──
    st.markdown("<p style='color:#8b949e; font-size:0.8em;'>上传图片</p>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.session_state.original_image = load_image(uploaded_file)
        # 同步处理图像
        if st.session_state.processed_image is None:
            st.session_state.processed_image = \
                st.session_state.original_image.copy()

    st.divider()

    # ── 信息 ──
    if st.session_state.original_image is not None:
        img = st.session_state.original_image
        h, w = img.shape[:2]
        st.caption(f"分辨率: {w} × {h}")
        st.caption(f"通道数: {img.shape[2] if len(img.shape)==3 else 1}")


# ═══════════════════════════════════════════════════════════════
# 主内容区
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Tab 1: 图像预处理（传统方法）
# ═══════════════════════════════════════════════════════════════
def render_traditional_tab():
    """渲染图像预处理标签页。"""
    st.markdown(
        "<h2>📷 图像预处理 <span style='color:#8b949e; font-size:0.6em;'>"
        "— 传统数字图像处理方法</span></h2>",
        unsafe_allow_html=True,
    )

    if st.session_state.original_image is None:
        st.info("👈 请先在左侧上传 PCB 图片")
        return

    img = st.session_state.processed_image.copy()

    # 子标签：分类展示
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "🔧 滤波降噪", "☀️ 图像增强", "📐 边缘检测",
        "🔲 形态学", "🔄 几何变换"
    ])

    # ── 滤波 ──
    with subtab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">滤波降噪</p>',
                    unsafe_allow_html=True)
        st.caption("PCB 图像常见噪声：椒盐噪声（中值滤波）、高斯噪声（高斯滤波）")

        col1, col2 = st.columns(2)
        with col1:
            filter_name = st.selectbox("滤波器", list(FILTERS.keys()))
            if filter_name in ("均值滤波", "高斯滤波"):
                ksize = st.slider("核大小", 3, 15, 5, 2)
            elif filter_name == "中值滤波":
                ksize = st.slider("核大小", 3, 15, 5, 2)
            elif filter_name == "双边滤波":
                d = st.slider("邻域直径", 3, 15, 9, 2)
                sigma_c = st.slider("颜色空间 σ", 10, 150, 75, 5)
                sigma_s = st.slider("坐标空间 σ", 10, 150, 75, 5)
        with col2:
            if st.button("应用滤波", use_container_width=True):
                st.session_state._before_op = img.copy()
                if filter_name == "双边滤波":
                    result = FILTERS[filter_name](img, d, sigma_c, sigma_s)
                else:
                    result = FILTERS[filter_name](img, ksize)
                st.session_state.processed_image = result
                st.success(f"已应用 {filter_name}")

        before_img = st.session_state.get("_before_op", img)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("处理前")
            st.image(get_display_image(before_img), use_container_width=True)
        with col_b:
            st.caption("处理后")
            st.image(
                get_display_image(st.session_state.processed_image),
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 增强 ──
    with subtab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">图像增强</p>',
                    unsafe_allow_html=True)
        st.caption("提升对比度、突出缺陷区域。CLAHE 自适应均衡化在 PCB 检测中效果最佳。")

        col1, col2 = st.columns(2)
        with col1:
            enhance_name = st.selectbox("增强方法", list(ENHANCEMENTS.keys()))
            if enhance_name == "CLAHE 自适应":
                clip = st.slider("Clip Limit", 1.0, 5.0, 2.0, 0.5)
                tile = st.slider("Tile Size", 4, 16, 8, 2)
            elif enhance_name == "伽马校正":
                gamma = st.slider("Gamma", 0.2, 3.0, 1.0, 0.1)
            elif enhance_name == "Unsharp 锐化":
                strength = st.slider("强度", 0.1, 3.0, 1.0, 0.1)
        with col2:
            if st.button("应用增强", use_container_width=True):
                st.session_state._before_op = img.copy()
                if enhance_name == "CLAHE 自适应":
                    result = ENHANCEMENTS[enhance_name](img, clip, tile)
                elif enhance_name == "伽马校正":
                    result = ENHANCEMENTS[enhance_name](img, gamma)
                elif enhance_name == "Unsharp 锐化":
                    result = ENHANCEMENTS[enhance_name](img, strength)
                else:
                    result = ENHANCEMENTS[enhance_name](img)
                st.session_state.processed_image = result
                st.success(f"已应用 {enhance_name}")

        before_img = st.session_state.get("_before_op", img)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("处理前")
            st.image(get_display_image(before_img), use_container_width=True)
        with col_b:
            st.caption("处理后")
            st.image(
                get_display_image(st.session_state.processed_image),
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 边缘检测 ──
    with subtab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">边缘检测</p>',
                    unsafe_allow_html=True)
        st.caption("提取 PCB 电路走线轮廓，Canny 是最成熟的通用方案。")

        col1, col2 = st.columns(2)
        with col1:
            edge_name = st.selectbox("检测方法", list(EDGES.keys()))
            if edge_name == "Canny":
                low = st.slider("低阈值", 10, 200, 50, 5)
                high = st.slider("高阈值", 50, 300, 150, 5)
                ksize = 3
            else:
                ksize = st.slider("核大小", 3, 7, 3, 2)
        with col2:
            if st.button("检测边缘", use_container_width=True):
                if edge_name == "Canny":
                    result = EDGES[edge_name](img, low, high)
                elif edge_name == "Sobel 综合":
                    result = EDGES[edge_name](img, ksize)
                else:
                    result = EDGES[edge_name](img)
                # 边缘图保持灰度显示
                st.session_state._edge_result = result
                st.success(f"已应用 {edge_name}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("原图")
            st.image(get_display_image(img), use_container_width=True)
        with col_b:
            st.caption("边缘检测结果")
            if st.session_state.get("_edge_result") is not None:
                st.image(
                    st.session_state._edge_result,
                    use_container_width=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 形态学 ──
    with subtab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">形态学操作</p>',
                    unsafe_allow_html=True)
        st.caption(
            "闭运算连接断线、开运算去除噪点——PCB 线路修复的核心操作。"
        )

        col1, col2 = st.columns(2)
        with col1:
            morph_name = st.selectbox("形态学操作", list(MORPHS.keys()))
            ksize = st.slider("结构元素大小", 3, 15, 3, 2)
        with col2:
            if st.button("应用形态学", use_container_width=True):
                st.session_state._before_op = img.copy()
                result = MORPHS[morph_name](img, ksize)
                st.session_state.processed_image = result
                st.success(f"已应用 {morph_name}")

        before_img = st.session_state.get("_before_op", img)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("处理前")
            st.image(get_display_image(before_img), use_container_width=True)
        with col_b:
            st.caption("处理后")
            st.image(
                get_display_image(st.session_state.processed_image),
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 几何变换 ──
    with subtab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">几何变换</p>',
                    unsafe_allow_html=True)
        st.caption("旋转、缩放、翻转——用于校正 PCB 拍摄角度。")

        col1, col2 = st.columns(2)
        with col1:
            transform_type = st.selectbox(
                "变换类型",
                ["旋转", "缩放", "水平翻转", "垂直翻转"],
            )
            if transform_type == "旋转":
                angle = st.slider("角度", -180, 180, 0, 1)
            elif transform_type == "缩放":
                scale = st.slider("缩放比例", 0.1, 3.0, 1.0, 0.05)
        with col2:
            if st.button("应用变换", use_container_width=True):
                st.session_state._before_op = img.copy()
                if transform_type == "旋转":
                    result = rotate_image(img, angle)
                elif transform_type == "缩放":
                    result = resize_image(img, scale=scale)
                elif transform_type == "水平翻转":
                    result = flip_image(img, 1)
                elif transform_type == "垂直翻转":
                    result = flip_image(img, 0)
                st.session_state.processed_image = result
                st.success(f"已应用 {transform_type}")

        before_img = st.session_state.get("_before_op", img)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("处理前")
            st.image(get_display_image(before_img), use_container_width=True)
        with col_b:
            st.caption("处理后")
            st.image(
                get_display_image(st.session_state.processed_image),
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 重置按钮 ──
    st.divider()
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("🔄 重置为原图"):
            st.session_state.processed_image =                 st.session_state.original_image.copy()
            st.session_state._edge_result = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# Tab 2: AI 缺陷检测（YOLO）
# ═══════════════════════════════════════════════════════════════
def render_detection_tab(model_path, conf_thresh):
    """渲染 AI 缺陷检测标签页。"""
    st.markdown(
        "<h2>🤖 AI 缺陷检测 <span style='color:#8b949e; font-size:0.6em;'>"
        "— YOLOv8 目标检测</span></h2>",
        unsafe_allow_html=True,
    )

    if st.session_state.original_image is None:
        st.info("👈 请先在左侧上传 PCB 图片")
        return

    img = st.session_state.processed_image

    # 显示当前图像
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_img, col_ctrl = st.columns([3, 1])

    with col_img:
        st.image(get_display_image(img), use_container_width=True,
                 caption="待检测图像")

    with col_ctrl:
        st.markdown('<p class="card-header">检测控制</p>',
                    unsafe_allow_html=True)
        st.caption(f"模型: {model_path}")
        st.caption(f"置信度阈值: {conf_thresh}")

        detect_btn = st.button(
            "🚀 开始检测", use_container_width=True,
        )

        if st.button("🧹 清除结果", use_container_width=True):
            st.session_state.detection_result = None
            st.session_state.defect_report = None
            st.session_state.pass_fail = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 执行检测 ──
    if detect_btn:
        detector = load_yolo_model(model_path)
        if detector is None:
            return
        detector.conf_threshold = conf_thresh

        with st.spinner("YOLO 推理中..."):
            t0 = time.time()
            boxes, labels, scores = detector.detect(img)
            elapsed = time.time() - t0

        st.session_state.detection_result = (boxes, labels, scores)

        if boxes:
            report = DefectReport.from_detection(
                "uploaded_pcb", boxes, labels, scores,
            )
            st.session_state.defect_report = report
            st.session_state.pass_fail = PassFailResult.evaluate(report)

        st.success(f"检测完成 · 耗时 {elapsed:.2f}s · "
                   f"发现 {len(boxes)} 个缺陷")

    # ── 展示检测结果 ──
    if st.session_state.detection_result is not None:
        boxes, labels, scores = st.session_state.detection_result

        if boxes:
            # 标注后的图像
            annotated = draw_boxes_pillow(img, boxes, labels, scores)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="card-header">检测结果</p>',
                        unsafe_allow_html=True)
            st.image(annotated, use_container_width=True,
                     caption=f"检测到 {len(boxes)} 个缺陷")

            # 缺陷列表
            with st.expander(f"📋 缺陷详情 ({len(boxes)})"):
                for i, (box, label, score) in enumerate(
                        zip(boxes, labels, scores)):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    w, h = x2 - x1, y2 - y1
                    st.markdown(
                        f"**#{i+1}** `{label}` "
                        f"置信度: **{score:.2%}** "
                        f"位置: ({x1},{y1}) 尺寸: {w}×{h}px"
                    )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("✅ 未检测到缺陷")


# ═══════════════════════════════════════════════════════════════
# Tab 3: 检测报告
# ═══════════════════════════════════════════════════════════════
def render_report_tab():
    """渲染检测报告标签页。"""
    st.markdown(
        "<h2>📊 检测报告 <span style='color:#8b949e; font-size:0.6em;'>"
        "— 缺陷分类 & PASS/FAIL 判定</span></h2>",
        unsafe_allow_html=True,
    )

    if st.session_state.defect_report is None:
        st.info("请先在「AI 缺陷检测」中执行检测")
        return

    report = st.session_state.defect_report
    pf = st.session_state.pass_fail

    # ── 顶部指标卡片 ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{report.total_defects}</div>'
            f'<div class="metric-label">缺陷总数</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        sev = report.get_severity_summary()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value" style="color:#f85149;">'
            f'{sev["严重"]}</div>'
            f'<div class="metric-label">严重缺陷</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{len(report.summary)}</div>'
            f'<div class="metric-label">缺陷类型数</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        # PASS / FAIL 判定
        if pf and pf.verdict == "PASS":
            verdict_class = "verdict-pass"
        else:
            verdict_class = "verdict-fail"
        verdict_text = pf.verdict if pf else "—"
        st.markdown(
            f'<div class="{verdict_class}">'
            f'<div class="big">{verdict_text}</div></div>',
            unsafe_allow_html=True,
        )

    # ── 判定理由 ──
    if pf:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<p class="card-header">'
                    f'{"✅" if pf.verdict=="PASS" else "❌"} 判定详情</p>',
                    unsafe_allow_html=True)
        st.markdown(pf.reason)
        if pf.details:
            for d in pf.details:
                st.markdown(f"- {d}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 功能一：缺陷分类报告 ──
    st.divider()
    st.markdown(
        "<h3>📋 功能一：缺陷分类报告</h3>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # 缺陷分类汇总表
        if report.summary:
            import pandas as pd
            df = pd.DataFrame(
                list(report.summary.items()),
                columns=["缺陷类型", "数量"],
            )
            df["占比"] = df["数量"] / df["数量"].sum()
            df["占比"] = df["占比"].apply(lambda x: f"{x:.1%}")
            st.dataframe(
                df, use_container_width=True, hide_index=True,
            )

            # 简易柱状图
            st.bar_chart(
                df.set_index("缺陷类型")["数量"],
                use_container_width=True,
            )

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-header">严重程度分布</p>',
                    unsafe_allow_html=True)
        sev = report.get_severity_summary()
        for level, count in sev.items():
            color = {"严重": "#f85149", "中等": "#d29922",
                      "轻微": "#8b949e"}[level]
            st.markdown(
                f'<span style="color:{color};">●</span> '
                f'{level}: <b>{count}</b>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # 导出按钮
        if st.button("📥 导出报告 JSON", use_container_width=True):
            import json
            report_dict = report.to_dict()
            if pf:
                report_dict["pass_fail"] = pf.to_dict()
            st.download_button(
                "⬇ 下载 JSON",
                json.dumps(report_dict, ensure_ascii=False, indent=2),
                "pcb_defect_report.json",
                "application/json",
                use_container_width=True,
            )

    # ── 功能二：PASS/FAIL 判定 ──
    st.divider()
    st.markdown(
        "<h3>⚖️ 功能二：PASS/FAIL 自动判定</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "模拟产线质检逻辑：出现 Short/Open Circuit 直接判 FAIL；"
        "单类缺陷超阈值判 FAIL；总缺陷数超阈值判 FAIL。"
    )

    if pf:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="card-header">判定规则</p>',
                        unsafe_allow_html=True)
            st.markdown(
                "• **严重缺陷** (Short/Open Circuit): 出现即 FAIL\n"
                "• **缺孔** > 2 个: FAIL\n"
                "• **鼠咬** > 3 个: FAIL\n"
                "• **毛刺** > 4 个: FAIL\n"
                "• **杂铜** > 3 个: FAIL\n"
                "• **总数** > 6 个: FAIL"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_v2:
            verdict_html = (
                f'<div class="{verdict_class}">'
                f'<p style="margin:0; color:#8b949e;">判定结果</p>'
                f'<div class="big">{pf.verdict}</div>'
                f'<p style="margin:0; color:#8b949e; font-size:0.85em;">'
                f'{pf.reason}</p></div>'
            )
            st.markdown(verdict_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════

if tab_choice == "📷 图像预处理":
    render_traditional_tab()

elif tab_choice == "🤖 AI 缺陷检测":
    render_detection_tab(model_path, conf_thresh)

elif tab_choice == "📊 检测报告":
    render_report_tab()

# 启动入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Streamlit 通过 `streamlit run main.py` 启动，
    # 自动执行文件顶层代码。此 __main__ 块仅在
    # `python main.py` 时执行，提供提示。
    print("请使用: streamlit run main.py")
