"""Discover tasks whose output is a LOCAL function of the input.

For radius r, output[y,x] must be a deterministic function of the input patch
input[y-r:y+r, x-r:x+r] across ALL official examples (train/test/arc-gen). If a
small r works, the task is r-local and compilable as a per-patch lookup (a conv
+ table), whose minimal cost we estimate. Tasks where the incumbent net costs
MUCH more than the local minimum are guaranteed algorithmic wins.

This does NOT fit anything; it only flags candidates + the consistency check.
Pure discovery, no temp files.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ngu  # noqa: E402


def examples(task_num):
    task = json.loads((ROOT / "data" / f"task{task_num:03d}.json").read_text())
    out = []
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            a = np.array(ex["input"]); b = np.array(ex["output"])
            if a.shape == b.shape:
                out.append((a, b))
    return out


def is_local(exs, r):
    """Return True if a patch of radius r determines the output pixel,
    consistently across all examples. Uses a dict from patch-bytes -> out color;
    a collision (same patch, different out) means r is too small."""
    table = {}
    for a, b in exs:
        H, W = a.shape
        ap = np.pad(a, r, constant_values=-1)
        for y in range(H):
            for x in range(W):
                patch = ap[y:y + 2 * r + 1, x:x + 2 * r + 1].tobytes()
                o = int(b[y, x])
                if patch in table:
                    if table[patch] != o:
                        return False, len(table)
                else:
                    table[patch] = o
    return True, len(table)


def scan(task_num):
    exs = examples(task_num)
    if not exs:
        return {"task": task_num, "local": False, "reason": "no same-shape examples"}
    # only same-shape tasks are candidates for pixelwise-local compile
    for r in (0, 1, 2, 3):
        ok, ncls = is_local(exs, r)
        if ok:
            return {"task": task_num, "local": True, "radius": r, "table_entries": ncls}
    return {"task": task_num, "local": False, "radius": ">3"}


if __name__ == "__main__":
    print(json.dumps(scan(int(sys.argv[1]))))
