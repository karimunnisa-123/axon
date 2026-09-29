"""
Generate the AXON architecture diagram as a PNG for the project gallery.
Uses matplotlib — no external diagram tools needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, label, color, fontsize=10, text_color="white"):
    """Draw a rounded rectangle with centered text."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, color=text_color,
        fontweight="bold",
    )


def arrow(ax, x1, y1, x2, y2, label=None):
    """Draw an arrow between two points."""
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=20,
        color="#444444", linewidth=1.6,
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 0.05, label,
            ha="center", va="bottom", fontsize=8, color="#666666",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Color palette
    C_INPUT = "#475569"       # slate
    C_PERCEPTION = "#0ea5e9"  # sky blue
    C_AGENT = "#7c3aed"       # purple
    C_TOOL = "#16a34a"        # green
    C_ACTION = "#ea580c"      # orange
    C_AWS = "#ff9900"         # AWS orange

    # Title
    ax.text(7, 8.6, "AXON — Agentic Visual Inspection",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="#0f172a")
    ax.text(7, 8.25, "perception that acts",
            ha="center", va="center", fontsize=11, style="italic",
            color="#64748b")

    # Layer 1 — Input
    box(ax, 0.5, 7.0, 3.0, 0.7, "Video / Camera Feed", C_INPUT, 11)
    box(ax, 4.0, 7.0, 3.0, 0.7, "sample.mp4", C_INPUT, 11)
    box(ax, 7.5, 7.0, 3.0, 0.7, "S3 / Local Storage", C_INPUT, 11)

    arrow(ax, 3.5, 7.35, 4.0, 7.35)
    arrow(ax, 7.0, 7.35, 7.5, 7.35)

    # Layer 2 — Perception (OpenCV 5)
    box(ax, 0.5, 5.5, 10.0, 0.9,
        "PERCEPTION  —  OpenCV 5  •  Canny  •  Adaptive Threshold  •  Contour Analysis  •  Severity Scoring",
        C_PERCEPTION, 12)

    arrow(ax, 5.5, 7.0, 5.5, 6.4)

    # Layer 3 — Tool Layer
    box(ax, 11.0, 5.5, 2.5, 0.9, "TOOL LAYER\n(MCP)", C_TOOL, 11)
    arrow(ax, 10.5, 5.95, 11.0, 5.95)

    # Five tools
    tools = [
        ("detect_defects", 0.5),
        ("zoom_roi", 2.4),
        ("rescan", 4.3),
        ("compare_frames", 6.2),
        ("escalate_to_human", 8.1),
    ]
    for label, x in tools:
        box(ax, x, 4.4, 1.7, 0.6, label, C_TOOL, 9)

    arrow(ax, 5.5, 5.5, 5.5, 5.0)

    # Layer 4 — Agent
    box(ax, 1.0, 3.1, 12.0, 1.0,
        "REASONING  —  LLM Agent (Gemini 2.x flash)  •  5-Model Fallback Chain  •  Deterministic Policy Fallback",
        C_AGENT, 12)

    arrow(ax, 5.5, 4.4, 5.5, 4.1)

    # Layer 5 — Decision outputs
    box(ax, 1.0, 1.8, 5.5, 0.8,
        "DECISION  →  next tool call", C_AGENT, 11)
    box(ax, 7.5, 1.8, 5.5, 0.8,
        "HUMAN CONTROL  →  escalation ticket", C_ACTION, 11)

    arrow(ax, 3.75, 3.1, 3.75, 2.6)
    arrow(ax, 10.25, 3.1, 10.25, 2.6)

    # Layer 6 — Evidence / AWS
    box(ax, 1.0, 0.5, 5.5, 0.8,
        "OBSERVABILITY  →  agent_trace.json  •  CloudWatch", C_AWS, 10)
    box(ax, 7.5, 0.5, 5.5, 0.8,
        "AWS  →  S3  •  Lambda  •  SNS  •  Graviton (COOL)", C_AWS, 10)

    arrow(ax, 3.75, 1.8, 3.75, 1.3)
    arrow(ax, 10.25, 1.8, 10.25, 1.3)

    # Feedback loop
    a = FancyArrowPatch(
        (13.4, 2.2), (13.4, 5.5),
        arrowstyle="-|>", mutation_scale=18,
        color="#7c3aed", linewidth=1.6,
        connectionstyle="arc3,rad=-0.3",
    )
    ax.add_patch(a)
    ax.text(13.7, 3.85, "loop", ha="left", va="center",
            fontsize=9, color="#7c3aed", style="italic")

    out = Path("docs") / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[AXON] Architecture diagram written to {out}")


if __name__ == "__main__":
    main()