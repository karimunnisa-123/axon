"""
AXON Agent Loop
===============
The perception-decision-action loop. Given a video, AXON:
  1. Runs the perception pipeline (detect_defects tool)
  2. Feeds structured output to the LLM
  3. Executes whatever tool the LLM chooses
  4. Loops until stop / max steps

Every step is logged to a trace for evidence and observability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from axon.agent.llm_client import LLMClient
from axon.agent.policy import (
    MAX_STEPS,
    deterministic_decision,
    prioritize_defects,
    summarize_defects,
)
from axon.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from axon.tools import call_tool
from axon.tools.schemas import BBox


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_agent(video_path: str, out_dir: str = "out") -> dict:
    """Run the AXON agent loop end-to-end on a video. Returns the full trace."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    trace = {
        "video_path": video_path,
        "started_at": _now(),
        "steps": [],
        "final_summary": {},
    }

    # --- Step 0: initial perception ---
    perception = call_tool("detect_defects", {
        "video_path": video_path,
        "stride": 10,
        "min_area": 40,
    })

    trace["initial_perception"] = {
        "frames_processed": perception.frames_processed,
        "defects_found": perception.defects_found,
    }

    defects = perception.defects
    summary = summarize_defects(defects)

    print(f"[AXON] Initial perception: {perception.defects_found} defects "
          f"across {perception.frames_processed} frames")

    # --- LLM setup ---
    try:
        llm = LLMClient()
        llm_available = True
    except Exception as e:
        print(f"[AXON] LLM unavailable ({e}). Falling back to deterministic policy.")
        llm = None
        llm_available = False

    history: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        print(f"[AXON] Step {step}: deciding next action...")

        # Decide
        if llm_available and defects:
            try:
                user_prompt = build_user_prompt(step, summary, history)
                decision, model_used = llm.decide_json(SYSTEM_PROMPT, user_prompt)
                decision["_model"] = model_used
                decision["_source"] = "llm"
            except Exception as e:
                print(f"[AXON] LLM failed ({e}); using deterministic fallback.")
                decision = deterministic_decision(defects)
                decision["_source"] = "fallback"
                decision["_model"] = "policy"
        else:
            decision = deterministic_decision(defects)
            decision["_source"] = "fallback" if not llm_available else "policy"
            decision["_model"] = "policy"

        tool_name = decision.get("tool", "stop")
        tool_input = decision.get("tool_input", {})

        print(f"[AXON] Step {step}: -> {tool_name} "
              f"({decision.get('reasoning', '')[:80]})")

        step_record = {
            "step": step,
            "timestamp": _now(),
            "observation": decision.get("observation", summary),
            "reasoning": decision.get("reasoning", ""),
            "tool": tool_name,
            "tool_input": tool_input,
            "source": decision.get("_source"),
            "model_used": decision.get("_model"),
        }

        # Execute tool (unless stopping)
        if tool_name == "stop":
            step_record["tool_output"] = {"stopped": True}
            trace["steps"].append(step_record)
            history.append(f"step {step}: stop - {decision.get('reasoning', '')}")
            break

        try:
            payload = dict(tool_input)

            # Inject video_path if the LLM forgot it
            if "video_path" not in payload and tool_name != "escalate_to_human":
                payload["video_path"] = video_path

                        # Normalize bbox: accept dict OR list [x,y,w,h]
            if "bbox" in payload and payload["bbox"] is not None:
                b = payload["bbox"]
                if isinstance(b, dict):
                    payload["bbox"] = BBox(**b).model_dump()
                elif isinstance(b, (list, tuple)) and len(b) == 4:
                    payload["bbox"] = BBox(
                        x=int(b[0]), y=int(b[1]), w=int(b[2]), h=int(b[3])
                    ).model_dump()

            # Normalize sensitivity: LLM sometimes sends a number
            if "sensitivity" in payload and not isinstance(payload["sensitivity"], str):
                s = payload["sensitivity"]
                if isinstance(s, (int, float)):
                    payload["sensitivity"] = "high" if s >= 0.7 else ("normal" if s >= 0.4 else "low")
                else:
                    payload["sensitivity"] = "high"

                        # Normalize escalate_to_human payload — the LLM sometimes uses
            # field aliases. Map them back to the schema and fill in missing
            # fields from the perception result if needed.
            if tool_name == "escalate_to_human":
                d = payload.get("defect") or {}
                if isinstance(d, dict):
                    # Field alias mapping
                    if "frame" in d and "frame_index" not in d:
                        d["frame_index"] = d.pop("frame")
                    if "area" in d and "area_px" not in d:
                        d["area_px"] = d.pop("area")

                    # Fill missing required fields from the top defect
                    if defects:
                        top = prioritize_defects(defects)[0]
                        d.setdefault("frame_index", top.frame_index)
                        d.setdefault("timestamp_s", top.timestamp_s)
                        d.setdefault("area_px", top.area_px)
                        d.setdefault("confidence", top.confidence)
                        d.setdefault("bbox", top.bbox.model_dump())
                        d.setdefault("severity", top.severity.value)

                    payload["defect"] = d
                elif defects:
                    # LLM sent no defect at all — use the top one
                    payload["defect"] = prioritize_defects(defects)[0].model_dump()

            # Escalation fallback: fill defect if the LLM omitted it
            if tool_name == "escalate_to_human" and "defect" not in payload and defects:
                payload["defect"] = prioritize_defects(defects)[0].model_dump()

            result = call_tool(tool_name, payload)
            output = result.model_dump() if hasattr(result, "model_dump") else result
            step_record["tool_output"] = output
            print(f"[AXON] Step {step}: tool executed successfully")

        except Exception as e:
            step_record["tool_output"] = {"error": str(e)}
            print(f"[AXON] Step {step}: tool error - {e}")

        trace["steps"].append(step_record)
        history.append(
            f"step {step}: called {tool_name} -> "
            f"{str(step_record['tool_output'])[:120]}"
        )

    # --- Final summary ---
    trace["finished_at"] = _now()
    trace["final_summary"] = {
        "total_steps": len(trace["steps"]),
        "defects_acted_on": len(defects),
        "escalations": sum(
            1 for s in trace["steps"]
            if s["tool"] == "escalate_to_human"
            and isinstance(s.get("tool_output"), dict)
            and "ticket_id" in s["tool_output"]
        ),
        "llm_used": llm_available,
    }

    # Write trace
    trace_path = out / "agent_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2, default=str))
    print(f"[AXON] Trace written to {trace_path}")

    return trace