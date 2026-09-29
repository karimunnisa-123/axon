"""
AXON Tool Schemas
=================
Pydantic models that define the contract between the perception layer
and the agent. Every MCP tool takes one of these as input and returns
one of these as output. Structured I/O is what makes the agent able
to reason over vision results instead of just narrating them.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(str, Enum):
    """Actions the agent can request after perception."""
    RESCAN = "rescan"
    ZOOM = "zoom"
    COMPARE = "compare"
    ESCALATE = "escalate"
    LOG_AND_CONTINUE = "log_and_continue"


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    x: int = Field(..., ge=0, description="Left edge in pixels")
    y: int = Field(..., ge=0, description="Top edge in pixels")
    w: int = Field(..., gt=0, description="Width in pixels")
    h: int = Field(..., gt=0, description="Height in pixels")

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


# ---------------------------------------------------------------------------
# Defect
# ---------------------------------------------------------------------------

class DefectModel(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_s: float = Field(..., ge=0.0)
    bbox: BBox
    area_px: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: Severity


# ---------------------------------------------------------------------------
# Tool inputs
# ---------------------------------------------------------------------------

class DetectDefectsInput(BaseModel):
    video_path: str = Field(..., description="Path to the video file")
    stride: int = Field(10, ge=1, le=60, description="Process every Nth frame")
    min_area: int = Field(40, ge=1, description="Minimum defect area in pixels")
    frame_range: tuple[int, int] | None = Field(
        None, description="Optional (start, end) frame range to process"
    )


class ZoomROIInput(BaseModel):
    video_path: str
    frame_index: int = Field(..., ge=0)
    bbox: BBox
    scale: float = Field(2.0, gt=0.0, le=10.0, description="Upscale factor")


class RescanInput(BaseModel):
    video_path: str
    frame_index: int = Field(..., ge=0)
    bbox: BBox
    sensitivity: Literal["low", "normal", "high"] = "high"


class CompareFramesInput(BaseModel):
    video_path: str
    frame_a: int = Field(..., ge=0)
    frame_b: int = Field(..., ge=0)
    bbox: BBox | None = Field(None, description="Optional ROI to compare")


class EscalateInput(BaseModel):
    defect: DefectModel
    reason: str = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Tool outputs
# ---------------------------------------------------------------------------

class DetectDefectsOutput(BaseModel):
    video_path: str
    frames_processed: int
    defects_found: int
    defects: list[DefectModel]


class ZoomROIOutput(BaseModel):
    frame_index: int
    original_bbox: BBox
    zoomed_image_path: str
    zoomed_shape: tuple[int, int, int]
    note: str = Field(
        default="Zoomed ROI saved. Agent can re-run detection on this crop."
    )


class RescanOutput(BaseModel):
    frame_index: int
    region: BBox
    sensitivity_used: str
    defects_found: int
    defects: list[DefectModel]
    changed_from_previous: bool = Field(
        ..., description="True if the rescan found defects the first pass missed"
    )


class CompareFramesOutput(BaseModel):
    frame_a: int
    frame_b: int
    mean_abs_diff: float = Field(..., ge=0.0)
    changed_pixels_ratio: float = Field(..., ge=0.0, le=1.0)
    persisted: bool = Field(
        ..., description="True if the change is small => defect persisted"
    )
    note: str


class EscalateOutput(BaseModel):
    escalated: bool
    ticket_id: str
    message: str
    logged_at: str


# ---------------------------------------------------------------------------
# Agent decision record (for tracing)
# ---------------------------------------------------------------------------

class AgentDecision(BaseModel):
    """One record in the agent's reasoning trace."""
    step: int
    observation: str
    tool_called: str
    tool_input: dict
    tool_output_summary: str
    next_action: ActionType
    reasoning: str