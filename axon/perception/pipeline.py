"""
AXON Perception Pipeline
========================
Core OpenCV 5 defect-detection pipeline.

This module turns raw video frames into structured defect detections
that the agentic layer can reason over. It is intentionally modular:
each stage can be swapped or benchmarked independently, which matters
for the COOL/Graviton comparison harness.

Run `python -m axon.perception.pipeline --help` for CLI usage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Defect:
    """A single detected defect in a frame."""
    frame_index: int
    timestamp_s: float
    bbox: tuple[int, int, int, int]      # x, y, w, h
    area_px: int
    confidence: float                    # 0.0 – 1.0
    severity: str                        # "low" | "medium" | "high"
    contour: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameResult:
    """All defects found in one frame, plus frame-level metadata."""
    frame_index: int
    timestamp_s: float
    frame_shape: tuple[int, int, int]
    defects: list[Defect] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "frame_shape": list(self.frame_shape),
            "defects": [d.to_dict() for d in self.defects],
        }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, target_width: int = 960) -> np.ndarray:
    """
    Normalize a frame for detection:
      - resize to a consistent width (keeps aspect ratio)
      - convert to grayscale
      - apply bilateral filter (denoise while preserving edges)
      - apply CLAHE (adaptive contrast for uneven lighting)
    """
    h, w = frame.shape[:2]
    if w != target_width:
        scale = target_width / w
        frame = cv2.resize(frame, (target_width, int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_defects(
    gray: np.ndarray,
    min_area: int = 60,
    max_area_ratio: float = 0.20,
    canny_low: int = 30,
    canny_high: int = 100,
) -> list[tuple[tuple[int, int, int, int], int, float, list[list[int]]]]:
    """
    Classical CV defect detection.

    Strategy:
      1. Canny edges -> structural boundaries
      2. Adaptive threshold -> local anomaly mask (texture deviations)
      3. Combine with morphological cleanup
      4. Contour analysis -> candidate defects, filtered by area
      5. Score confidence from edge density + area + solidity

    Returns a list of (bbox, area_px, confidence, contour_points).
    """
    h, w = gray.shape[:2]
    frame_area = h * w
    max_area = int(frame_area * max_area_ratio)

    # 1. Edge map
    edges = cv2.Canny(gray, canny_low, canny_high)

    # 2. Adaptive threshold mask (local texture anomaly)
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=10,
    )

    # 3. Combine: defect candidates live where edges and anomaly mask agree
    combined = cv2.bitwise_and(mask, cv2.dilate(edges, np.ones((3, 3), np.uint8)))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: list[tuple[tuple[int, int, int, int], int, float, list[list[int]]]] = []

    for c in contours:
        area = int(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw == 0 or bh == 0:
            continue

        # Solidity: area / convex hull area. Low solidity => irregular shape => more defect-like.
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 0 else 1.0
        irregularity = 1.0 - solidity

        # Edge density inside bbox — how "edgy" the region is
        roi_edges = edges[y:y + bh, x:x + bw]
        edge_density = float(np.count_nonzero(roi_edges)) / max(bw * bh, 1)

        # Confidence: blend of area, irregularity, and edge density
        area_score = min(area / (max_area + 1e-6), 1.0)
        confidence = float(
            0.40 * area_score
            + 0.35 * irregularity
            + 0.25 * min(edge_density * 4.0, 1.0)
        )
        confidence = max(0.0, min(1.0, confidence))

        contour_pts = c.reshape(-1, 2).tolist()
        results.append(((x, y, bw, bh), area, confidence, contour_pts))

    return results

def classify_severity(confidence: float, area_px: int) -> str:
    """Map confidence + area to a coarse severity label."""
    if confidence >= 0.55 or area_px >= 800:
        return "high"
    if confidence >= 0.30 or area_px >= 250:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------

def iter_frames(video_path: Path, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, frame) for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: Path,
    stride: int = 5,
    min_area: int = 40,
    annotate_dir: Path | None = None,
) -> list[FrameResult]:
    """
    Run the full perception pipeline over a video.

    Args:
        video_path: input video
        stride: process every Nth frame (5 = every 5th frame)
        min_area: minimum defect area in pixels
        annotate_dir: if provided, write annotated frames here as PNGs

    Returns:
        list of FrameResult, one per processed frame.
    """
    if annotate_dir is not None:
        annotate_dir.mkdir(parents=True, exist_ok=True)

    results: list[FrameResult] = []

    for frame_idx, ts, frame in iter_frames(video_path, stride=stride):
        gray = preprocess(frame)
        raw_defects = detect_defects(gray, min_area=min_area)

        defects: list[Defect] = []
        for (bbox, area, conf, contour_pts) in raw_defects:
            defects.append(Defect(
                frame_index=frame_idx,
                timestamp_s=round(ts, 3),
                bbox=bbox,
                area_px=area,
                confidence=round(conf, 3),
                severity=classify_severity(conf, area),
                contour=contour_pts,
            ))

        result = FrameResult(
            frame_index=frame_idx,
            timestamp_s=round(ts, 3),
            frame_shape=frame.shape,
            defects=defects,
        )
        results.append(result)

        if annotate_dir is not None and defects:
            annotated = frame.copy()
            for d in defects:
                x, y, w, h = d.bbox
                color = {"low": (0, 200, 0), "medium": (0, 165, 255), "high": (0, 0, 255)}[d.severity]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label = f"{d.severity} {d.confidence:.2f}"
                cv2.putText(annotated, label, (x, max(y - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            out_path = annotate_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), annotated)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AXON perception pipeline")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("--stride", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--min-area", type=int, default=40, help="Minimum defect area (px)")
    parser.add_argument("--annotate", type=Path, default=None,
                        help="Directory to write annotated frames")
    parser.add_argument("--out", type=Path, default=Path("perception_output.json"),
                        help="Path for JSON results")
    args = parser.parse_args()

    results = run_pipeline(
        video_path=args.video,
        stride=args.stride,
        min_area=args.min_area,
        annotate_dir=args.annotate,
    )

    total_defects = sum(len(r.defects) for r in results)
    payload = {
        "video": str(args.video),
        "frames_processed": len(results),
        "defects_found": total_defects,
        "frames": [r.to_dict() for r in results],
    }
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"[AXON] Frames processed : {len(results)}")
    print(f"[AXON] Defects found    : {total_defects}")
    print(f"[AXON] JSON written to  : {args.out}")
    if args.annotate:
        print(f"[AXON] Annotated frames : {args.annotate}")


if __name__ == "__main__":
    main()