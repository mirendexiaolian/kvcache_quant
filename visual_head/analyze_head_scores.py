import json
import numpy as np
import os


input_file = "./visual_head/head_score/qwen2-vl.json"
output_file = "./visual_head/head_score/qwen2-vl_stats.json"

assert os.path.exists(input_file), f"File not found: {input_file}"

with open(input_file, 'r') as f:
    head_scores = json.load(f)

head_stats = {}

for head, scores in head_scores.items():
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 0:
        continue
    head_stats[head] = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "non_zero_ratio": float(np.count_nonzero(scores) / len(scores)),
        "num_samples": int(len(scores))
    }


with open(output_file, 'w') as f:
    json.dump(head_stats, f, indent=2)

print(f"Head score stats saved to: {output_file}")