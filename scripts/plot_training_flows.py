#!/usr/bin/env python3
"""
Draws a side-by-side flow diagram comparing the Original AlphaZero
training loop vs. an augmented version with a Teacher module
(Reanalyze with teacher MCTS + optional EMA teacher consistency losses).

Outputs: PNG and SVG under scripts/ by default.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _setup_fonts():
    # Try common CJK fonts first; gracefully fall back if unavailable.
    try:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
            'Source Han Sans SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans'
        ]
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


def add_box(ax, center, width, height, text, fc="#E6F0FF", ec="#2B6CB0", lw=1.5, text_color="#1A202C"):
    x, y = center
    rect = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=6",
                          linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, color=text_color, wrap=True)
    return rect


def add_arrow(ax, start, end, color="#4A5568"):
    x0, y0 = start
    x1, y1 = end
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))


def draw_flows(save_dir: Path):
    _setup_fonts()

    fig_w, fig_h = 14, 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Column centers
    x_left = 0.28
    x_right = 0.72

    # Vertical layout (top to bottom)
    ys_left = [0.85, 0.65, 0.45, 0.25]
    ys_right = [0.85, 0.65, 0.48, 0.31, 0.14]

    box_w = 0.38
    box_h = 0.12

    # Colors
    common_fc = "#E6F0FF"  # light blue
    common_ec = "#2B6CB0"
    teacher_fc = "#FFF4E5"  # light orange
    teacher_ec = "#C05621"

    # Titles
    ax.text(x_left, 0.95, "原始 AlphaZero", ha='center', va='center', fontsize=14, fontweight='bold', color="#2D3748")
    ax.text(x_right, 0.95, "AlphaZero + 教师模块", ha='center', va='center', fontsize=14, fontweight='bold', color="#2D3748")

    # Left: Original AlphaZero
    left_texts = [
        "自对弈\n(MCTS 使用当前网络)",
        "收集训练样本\n(s, π, z)",
        "训练新网络\nL = L_pi + L_v",
        "评估 (新 vs 旧) 并更新",
    ]
    left_boxes = []
    for y, t in zip(ys_left, left_texts):
        left_boxes.append(add_box(ax, (x_left, y), box_w, box_h, t, fc=common_fc, ec=common_ec))
    for i in range(len(left_boxes) - 1):
        add_arrow(ax, (x_left, ys_left[i] - box_h / 2 - 0.01), (x_left, ys_left[i + 1] + box_h / 2 + 0.01))

    # Right: With Teacher module
    right_texts = [
        "自对弈\n(MCTS 使用当前网络)",
        "收集训练样本\n(s, π, z)",
        (
            "重分析 Reanalyze (新增)\n"
            "教师 = 冻结旧网络 (pnet)\n"
            "教师MCTS 更强: 模拟数×k, 无根噪, 对称集成\n"
            "生成 π* (可与原 π 按 α 混合), 值可按 λ 混合"
        ),
        (
            "训练学生网络\n"
            "L = L_pi + L_v"
            " + λ_pi KL(π_s || π_EMA)"
            " + λ_v MSE(v_s, v_EMA) (可选)"
        ),
        "评估 (新 vs 旧) 并更新",
    ]
    right_colors = [
        (common_fc, common_ec),
        (common_fc, common_ec),
        (teacher_fc, teacher_ec),  # Reanalyze is new
        (teacher_fc, teacher_ec),  # EMA consistency is new
        (common_fc, common_ec),
    ]
    right_boxes = []
    for y, t, (fc, ec) in zip(ys_right, right_texts, right_colors):
        right_boxes.append(add_box(ax, (x_right, y), box_w, box_h, t, fc=fc, ec=ec))
    for i in range(len(right_boxes) - 1):
        add_arrow(ax, (x_right, ys_right[i] - box_h / 2 - 0.01), (x_right, ys_right[i + 1] + box_h / 2 + 0.01))

    # Legend
    legend_x = 0.5
    legend_y = 0.05
    add_box(ax, (legend_x - 0.12, legend_y), 0.08, 0.06, "共同步骤", fc=common_fc, ec=common_ec)
    add_box(ax, (legend_x + 0.12, legend_y), 0.08, 0.06, "教师模块新增/改动", fc=teacher_fc, ec=teacher_ec)

    # Title
    ax.text(0.5, 0.995, "AlphaZero 训练流程对比（原始 vs 添加教师）", ha='center', va='top', fontsize=16, fontweight='bold')

    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / 'training_flows_cn.png'
    svg_path = save_dir / 'training_flows_cn.svg'
    fig.tight_layout(pad=0.5)
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(svg_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def main():
    here = Path(__file__).resolve().parent
    draw_flows(here)


if __name__ == '__main__':
    main()

