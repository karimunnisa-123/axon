"""
Generate a synthetic test video with known 'defects' for AXON pipeline testing.

Creates a moving textured surface (concrete-like) with injected dark blobs,
scratch lines, and rust-colored patches that move across frames. Ground truth
defect locations are written to a JSON sidecar so we can evaluate detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def make_surface(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Create a concrete-like base texture."""
    base = rng.integers(110, 150, size=(height, width), dtype=np.uint8)
    noise = rng.normal(0, 18, size=(height, width))
    surface = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    surface = cv2.GaussianBlur(surface, (3, 3), 0)
    return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)


def draw_defect(frame: np.ndarray, kind: str, x: int, y: int,
                w: int, h: int, rng: np.random.Generator) -> dict:
    """Draw a defect and return its bbox."""
    if kind == "crack":
        pts = np.array([
            [x, y],
            [x + w // 3, y + h // 2],
            [x + 2 * w // 3, y + h // 3],
            [x + w, y + h],
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], False, (30, 30, 30), 3, cv2.LINE_AA)
    elif kind == "rust":
        overlay = frame.copy()
        cv2.ellipse(overlay, (x + w // 2, y + h // 2),
                    (w // 2, h // 2), 0, 0, 360, (30, 70, 140), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    elif kind == "blob":
        cv2.circle(frame, (x + w // 2, y + h // 2), min(w, h) // 2,
                   (40, 40, 40), -1)
    return {"kind": kind, "bbox": [x, y, w, h]}


def main() -> None:
    out_path = Path("sample.mp4")
    gt_path = Path("sample_ground_truth.json")

    width, height = 960, 540
    fps = 30
    duration_s = 6
    total_frames = fps * duration_s

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    rng = np.random.default_rng(seed=42)
    all_defects = []

    for i in range(total_frames):
        frame = make_surface(width, height, rng)

        # Rolling "conveyor" texture shift
        shift = int(i * 4)
        frame = np.roll(frame, shift, axis=1)

        defects_this_frame = []

        # Three defects that appear, persist, then leave the frame
        if 20 <= i <= 140:
            defects_this_frame.append(
                draw_defect(frame, "crack", 200 + shift % 200, 180, 80, 120, rng)
            )
        if 40 <= i <= 160:
            defects_this_frame.append(
                draw_defect(frame, "rust", 520 + shift % 180, 300, 100, 90, rng)
            )
        if 60 <= i <= 120:
            defects_this_frame.append(
                draw_defect(frame, "blob", 720 + shift % 150, 100, 70, 70, rng)
            )

        writer.write(frame)
        all_defects.append({"frame": i, "defects": defects_this_frame})

    writer.release()

    gt_path.write_text(json.dumps({
        "video": str(out_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": total_frames,
        "ground_truth": all_defects,
    }, indent=2))

    print(f"[AXON] Wrote {out_path} ({total_frames} frames, {width}x{height} @ {fps}fps)")
    print(f"[AXON] Wrote {gt_path}")


if __name__ == "__main__":
    main()