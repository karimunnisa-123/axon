"""
AXON Agent Policy
=================
Deterministic guardrails and fallback logic that run regardless of the LLM.
Ensures AXON never loops forever, never escalates without evidence, and
degrades gracefully when the LLM is unavailable.
"""

from __future__ import annotations

from axon.tools.schemas import DefectModel, Severity

MAX_STEPS = 8
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_SEVERITY_ESCALATE = True


def prioritize_defects(defects: list[DefectModel]) -> list[DefectModel]:
    """Order defects by severity then confidence (highest first)."""
    order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
    return sorted(defects, key=lambda d: (order[d.severity], -d.confidence))


def should_rescan(defect: DefectModel) -> bool:
    return defect.confidence < LOW_CONFIDENCE_THRESHOLD


def should_escalate(defect: DefectModel, persisted: bool) -> bool:
    if not HIGH_SEVERITY_ESCALATE:
        return False
    return defect.severity == Severity.HIGH and persisted


def summarize_defects(defects: list[DefectModel]) -> str:
    """Human-readable summary the LLM receives."""
    if not defects:
        return "No defects detected."

    lines = [f"{len(defects)} defect(s) detected:"]
    for i, d in enumerate(defects[:10]):
        lines.append(
            f"  [{i}] frame={d.frame_index} bbox={d.bbox.as_tuple()} "
            f"severity={d.severity.value} confidence={d.confidence:.3f} "
            f"area={d.area_px}px"
        )
    if len(defects) > 10:
        lines.append(f"  ... and {len(defects) - 10} more")
    return "\n".join(lines)


def deterministic_decision(defects: list[DefectModel]) -> dict:
    """
    Fallback decision if the LLM is unavailable. Uses only policy rules.
    Guarantees AXON still functions when Gemini is down.
    """
    if not defects:
        return {
            "tool": "stop",
            "tool_input": {"reason": "No defects to act on."},
            "next_action": "stop",
            "reasoning": "Deterministic fallback: empty defect list.",
        }

    top = prioritize_defects(defects)[0]

    if should_rescan(top):
        return {
            "tool": "rescan",
            "tool_input": {
                "frame_index": top.frame_index,
                "bbox": top.bbox.model_dump(),
                "sensitivity": "high",
            },
            "next_action": "rescan",
            "reasoning": f"Fallback: low confidence ({top.confidence:.2f}).",
        }

    return {
        "tool": "compare_frames",
        "tool_input": {
            "frame_a": top.frame_index,
            "frame_b": top.frame_index + 10,
            "bbox": top.bbox.model_dump(),
        },
        "next_action": "compare",
        "reasoning": "Fallback: confirm persistence before escalation.",
    }