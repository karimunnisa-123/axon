"""Compare AXON detection output against ground truth."""
import json
from pathlib import Path

gt = json.loads(Path("sample_ground_truth.json").read_text())
det = json.loads(Path("perception_output.json").read_text())

STRIDE = 10

gt_defects = sum(
    len(f["defects"])
    for f in gt["ground_truth"]
    if f["frame"] % STRIDE == 0
)

detected = det["defects_found"]
ratio = detected / max(gt_defects, 1)

print(f"Ground truth defects (stride {STRIDE}): {gt_defects}")
print(f"Detected defects                       : {detected}")
print(f"Ratio (detected / ground truth)        : {ratio:.2f}")

# Per-frame breakdown
frames_with_gt = sum(1 for f in gt["ground_truth"]
                     if f["frame"] % STRIDE == 0 and f["defects"])
frames_with_det = sum(1 for f in det["frames"] if f["defects"])
print(f"Frames with ground-truth defects       : {frames_with_gt}")
print(f"Frames with detected defects           : {frames_with_det}")

# Severity distribution
from collections import Counter
sev = Counter(
    d["severity"]
    for f in det["frames"]
    for d in f["defects"]
)
print(f"Severity distribution                  : {dict(sev)}")