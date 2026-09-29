"""
AXON MCP Tool Server
====================
Exposes OpenCV 5 perception operations as agent-callable tools.

Every tool returns structured output. The agent uses that output to
decide the NEXT tool call. This is what makes AXON agentic rather than
a detector with a chat interface.

All tool inputs pass through defensive normalization: the LLM is
treated as untrusted input, and we repair/reject invalid values.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from axon.perception.pipeline import (
    detect_defects,
    preprocess,
    iter_frames,
    classify_severity,
)
from axon.tools.schemas import (
    BBox,
    CompareFramesInput,
    CompareFramesOutput,
    DefectModel,
    DetectDefectsInput,
    DetectDefectsOutput,
    EscalateInput,
    EscalateOutput,
    RescanInput,
    RescanOutput,
    Severity,
    ZoomROIInput,
    ZoomROIOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUT_DIR = Path("out")
_OUT_DIR.mkdir(exist_ok=True)


def _resolve_video(video_path: str) -> str:
    """
    Defensive video path resolution.

    The LLM is untrusted input — it occasionally hallucinates filenames
    like 'current_video' or 'inspection_stream.mp4'. If the requested
    path doesn't exist, fall back to sample.mp4. This prevents a bad
    LLM output from breaking inspection entirely.
    """
    if video_path and Path(video_path).exists():
        return video_path
    fallback = "sample.mp4"
    if Path(fallback).exists():
        return fallback
    raise RuntimeError(
        f"Video not found: {video_path!r} (no fallback available)"
    )


def _read_frame(video_path: str, frame_index: int) -> np.ndarray:
    """Read a single frame by index."""
    video_path = _resolve_video(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        return frame
    finally:
        cap.release()


def _to_defect_model(
    frame_idx: int, ts: float, bbox, area: int, conf: float
) -> DefectModel:
    x, y, w, h = bbox
    return DefectModel(
        frame_index=frame_idx,
        timestamp_s=round(ts, 3),
        bbox=BBox(x=int(x), y=int(y), w=int(w), h=int(h)),
        area_px=int(area),
        confidence=float(conf),
        severity=Severity(classify_severity(conf, area)),
    )


# ---------------------------------------------------------------------------
# Tool 1 — detect_defects
# ---------------------------------------------------------------------------

def tool_detect_defects(inp: DetectDefectsInput) -> DetectDefectsOutput:
    """Run the OpenCV 5 perception pipeline over a video."""
    video_path = _resolve_video(inp.video_path)
    defects_out: list[DefectModel] = []
    frames_processed = 0

    for frame_idx, ts, frame in iter_frames(Path(video_path), stride=inp.stride):
        if inp.frame_range is not None:
            start, end = inp.frame_range
            if not (start <= frame_idx <= end):
                continue

        gray = preprocess(frame)
        raw = detect_defects(gray, min_area=inp.min_area)

        for bbox, area, conf, _contour in raw:
            defects_out.append(_to_defect_model(frame_idx, ts, bbox, area, conf))

        frames_processed += 1

    return DetectDefectsOutput(
        video_path=video_path,
        frames_processed=frames_processed,
        defects_found=len(defects_out),
        defects=defects_out,
    )


# ---------------------------------------------------------------------------
# Tool 2 — zoom_roi
# ---------------------------------------------------------------------------

def tool_zoom_roi(inp: ZoomROIInput) -> ZoomROIOutput:
    """Crop and upscale a region of interest."""
    video_path = _resolve_video(inp.video_path)
    frame = _read_frame(video_path, inp.frame_index)
    x, y, w, h = inp.bbox.as_tuple()

    fh, fw = frame.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))

    roi = frame[y:y + h, x:x + w]
    new_w = int(w * inp.scale)
    new_h = int(h * inp.scale)
    zoomed = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    out_path = _OUT_DIR / f"zoom_f{inp.frame_index}_{uuid.uuid4().hex[:8]}.png"
    cv2.imwrite(str(out_path), zoomed)

    return ZoomROIOutput(
        frame_index=inp.frame_index,
        original_bbox=inp.bbox,
        zoomed_image_path=str(out_path),
        zoomed_shape=(zoomed.shape[0], zoomed.shape[1], zoomed.shape[2]),
    )


# ---------------------------------------------------------------------------
# Tool 3 — rescan
# ---------------------------------------------------------------------------

def tool_rescan(inp: RescanInput) -> RescanOutput:
    """
    Re-run detection on a specific frame with a sensitivity preset.
    Used by the agent when the first pass was low-confidence.
    """
    presets = {
        "low":    {"area_frac": 0.02,  "canny_low": 50, "canny_high": 150},
        "normal": {"area_frac": 0.005, "canny_low": 40, "canny_high": 120},
        "high":   {"area_frac": 0.001, "canny_low": 20, "canny_high": 80},
    }
    # Defensive: LLM sometimes sends a numeric sensitivity
    if inp.sensitivity not in presets:
        inp = inp.model_copy(update={"sensitivity": "high"})
    cfg = presets[inp.sensitivity]

    video_path = _resolve_video(inp.video_path)
    frame = _read_frame(video_path, inp.frame_index)
    gray = preprocess(frame)

    x, y, w, h = inp.bbox.as_tuple()
    fh, fw = gray.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))
    crop = gray[y:y + h, x:x + w]

    roi_area = crop.shape[0] * crop.shape[1]
    min_area = max(2, int(roi_area * cfg["area_frac"]))

    raw = detect_defects(
        crop,
        min_area=min_area,
        canny_low=cfg["canny_low"],
        canny_high=cfg["canny_high"],
    )

    defects: list[DefectModel] = []
    for (bx, by, bw, bh), area, conf, _ in raw:
        global_bbox = (bx + x, by + y, bw, bh)
        defects.append(_to_defect_model(inp.frame_index, 0.0, global_bbox, area, conf))

    return RescanOutput(
        frame_index=inp.frame_index,
        region=inp.bbox,
        sensitivity_used=inp.sensitivity,
        defects_found=len(defects),
        defects=defects,
        changed_from_previous=len(defects) > 0,
    )


# ---------------------------------------------------------------------------
# Tool 4 — compare_frames
# ---------------------------------------------------------------------------

def tool_compare_frames(inp: CompareFramesInput) -> CompareFramesOutput:
    """Compare two frames to determine if a defect persisted or was transient."""
    video_path = _resolve_video(inp.video_path)
    fa = _read_frame(video_path, inp.frame_a)
    fb = _read_frame(video_path, inp.frame_b)

    if fa.shape != fb.shape:
        raise RuntimeError("Frames have different shapes; cannot compare.")

    if inp.bbox is not None:
        x, y, w, h = inp.bbox.as_tuple()
        fa = fa[y:y + h, x:x + w]
        fb = fb[y:y + h, x:x + w]

    diff = cv2.absdiff(fa, fb)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mean_abs = float(gray_diff.mean())

    changed = int(np.count_nonzero(gray_diff > 25))
    changed_ratio = changed / gray_diff.size
    persisted = changed_ratio < 0.20

    note = (
        "Defect region appears stable across frames (persisted)."
        if persisted else
        "Region changed significantly between frames (possibly transient)."
    )

    return CompareFramesOutput(
        frame_a=inp.frame_a,
        frame_b=inp.frame_b,
        mean_abs_diff=round(mean_abs, 3),
        changed_pixels_ratio=round(changed_ratio, 4),
        persisted=persisted,
        note=note,
    )


# ---------------------------------------------------------------------------
# Tool 5 — escalate_to_human
# ---------------------------------------------------------------------------

def tool_escalate_to_human(inp: EscalateInput) -> EscalateOutput:
    """
    Record an escalation for human review. Logs to out/escalations.jsonl.
    """
    ticket_id = f"AXON-{uuid.uuid4().hex[:10].upper()}"
    logged_at = datetime.now(timezone.utc).isoformat()

    log_path = _OUT_DIR / "escalations.jsonl"
    record = {
        "ticket_id": ticket_id,
        "logged_at": logged_at,
        "reason": inp.reason,
        "defect": inp.defect.model_dump(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return EscalateOutput(
        escalated=True,
        ticket_id=ticket_id,
        message=f"Escalation logged: {inp.reason}",
        logged_at=logged_at,
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "detect_defects":    tool_detect_defects,
    "zoom_roi":          tool_zoom_roi,
    "rescan":            tool_rescan,
    "compare_frames":    tool_compare_frames,
    "escalate_to_human": tool_escalate_to_human,
}

_TOOL_INPUT_MODELS = {
    "detect_defects":    DetectDefectsInput,
    "zoom_roi":          ZoomROIInput,
    "rescan":            RescanInput,
    "compare_frames":    CompareFramesInput,
    "escalate_to_human": EscalateInput,
}


def call_tool(name: str, payload: dict):
    """Dispatch a tool call by name with a raw dict payload."""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    input_model = _TOOL_INPUT_MODELS[name]
    validated = input_model(**payload)
    return TOOL_REGISTRY[name](validated)