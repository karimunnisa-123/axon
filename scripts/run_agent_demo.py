"""
AXON Agent Demo
===============
Runs the full perception-decision-action loop on a video and prints the trace.
"""

import json
from pathlib import Path

from axon.agent import run_agent


def main() -> None:
    video = "sample.mp4"
    if not Path(video).exists():
        raise SystemExit(
            f"Missing {video}. Run: python scripts\\make_test_video.py"
        )

    print("=" * 70)
    print("AXON AGENT DEMO")
    print(f"Video: {video}")
    print("=" * 70)

    trace = run_agent(video, out_dir="out")

    print()
    print("=" * 70)
    print("AGENT TRACE")
    print("=" * 70)
    for step in trace["steps"]:
        print(f"\nStep {step['step']}  [source={step.get('source')} "
              f"model={step.get('model_used')}]")
        print(f"  Reasoning : {step.get('reasoning', '')}")
        print(f"  Tool      : {step.get('tool')}")
        print(f"  Output    : {json.dumps(step.get('tool_output'), default=str)[:200]}")

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for k, v in trace["final_summary"].items():
        print(f"  {k}: {v}")
    print()
    print("Full trace: out/agent_trace.json")


if __name__ == "__main__":
    main()