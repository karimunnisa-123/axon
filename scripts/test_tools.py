"""Smoke test for the AXON MCP tool layer."""

from axon.tools import call_tool


def main() -> None:
    video = "sample.mp4"

    print("=" * 60)
    print("TOOL 1 — detect_defects")
    print("=" * 60)
    out = call_tool("detect_defects", {
        "video_path": video,
        "stride": 30,          # fewer frames for speed
        "min_area": 40,
    })
    print(f"Frames processed : {out.frames_processed}")
    print(f"Defects found    : {out.defects_found}")
    if out.defects:
        first = out.defects[0]
        print(f"First defect     : frame {first.frame_index}, "
              f"bbox={first.bbox.as_tuple()}, "
              f"severity={first.severity.value}, "
              f"confidence={first.confidence}")

    if not out.defects:
        print("No defects found — cannot test downstream tools.")
        return

    target = out.defects[0]

    print()
    print("=" * 60)
    print("TOOL 2 — zoom_roi")
    print("=" * 60)
    zoom = call_tool("zoom_roi", {
        "video_path": video,
        "frame_index": target.frame_index,
        "bbox": target.bbox.model_dump(),
        "scale": 3.0,
    })
    print(f"Zoomed image saved : {zoom.zoomed_image_path}")
    print(f"Zoomed shape       : {zoom.zoomed_shape}")

    print()
    print("=" * 60)
    print("TOOL 3 — rescan")
    print("=" * 60)
    rescan = call_tool("rescan", {
        "video_path": video,
        "frame_index": target.frame_index,
        "bbox": target.bbox.model_dump(),
        "sensitivity": "high",
    })
    print(f"Sensitivity used   : {rescan.sensitivity_used}")
    print(f"Defects on rescan  : {rescan.defects_found}")

    print()
    print("=" * 60)
    print("TOOL 4 — compare_frames")
    print("=" * 60)
    compare = call_tool("compare_frames", {
        "video_path": video,
        "frame_a": target.frame_index,
        "frame_b": target.frame_index + 10,
        "bbox": target.bbox.model_dump(),
    })
    print(f"Mean abs diff      : {compare.mean_abs_diff}")
    print(f"Changed pixel ratio: {compare.changed_pixels_ratio}")
    print(f"Persisted          : {compare.persisted}")
    print(f"Note               : {compare.note}")

    print()
    print("=" * 60)
    print("TOOL 5 — escalate_to_human")
    print("=" * 60)
    esc = call_tool("escalate_to_human", {
        "defect": target.model_dump(),
        "reason": "High-confidence defect persisted across frames.",
    })
    print(f"Escalated  : {esc.escalated}")
    print(f"Ticket ID  : {esc.ticket_id}")
    print(f"Logged at  : {esc.logged_at}")

    print()
    print("=" * 60)
    print("ALL TOOLS OK")
    print("=" * 60)


if __name__ == "__main__":
    main()