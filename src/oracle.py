"""Local scoring oracle: replicates the official NeuroGolf grader.

Wraps the competition's neurogolf_utils to (a) verify a network on all
train/test/arc-gen examples and (b) compute (memory, params, points)
exactly as the grader does.
"""
import json
import math
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(DATA / "neurogolf_utils"))
import neurogolf_utils as ngu  # noqa: E402


def load_task(task_num: int) -> dict:
    with open(DATA / f"task{task_num:03d}.json") as f:
        return json.load(f)


def examples_as_numpy(task: dict):
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            bench = ngu.convert_to_numpy(ex)
            if bench is not None:
                yield split, bench


def evaluate(model_path: str, task_num: int, profile_prefix: str | None = None):
    """Returns dict with pass/fail counts, memory, params, points."""
    raw = onnx.load(model_path)
    sanitized = ngu.sanitize_model(raw)
    if sanitized is None:
        return {"error": "sanitize failed"}
    size = pathlib.Path(model_path).stat().st_size
    if size > ngu._FILESIZE_LIMIT_IN_BYTES:
        return {"error": f"file too large: {size}"}

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = profile_prefix or f"prof{task_num:03d}"
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load failed: {e}"}

    task = load_task(task_num)
    right = wrong = 0
    fail_splits = set()
    try:
        for split, bench in examples_as_numpy(task):
            out = session.run(["output"], {"input": bench["input"]})[0]
            if np.array_equal((out > 0.0).astype(float), bench["output"]):
                right += 1
            else:
                wrong += 1
                fail_splits.add(split)
    except Exception as e:  # noqa: BLE001
        session.end_profiling()
        return {"error": f"run failed: {e}"}

    trace = session.end_profiling()
    memory, params = ngu.score_network(sanitized, trace)
    pathlib.Path(trace).unlink(missing_ok=True)
    if memory is None or params is None or memory < 0 or params < 0:
        return {"error": "cost could not be measured", "right": right, "wrong": wrong}
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return {
        "right": right,
        "wrong": wrong,
        "fail_splits": sorted(fail_splits),
        "memory": memory,
        "params": params,
        "cost": memory + params,
        "points": points if wrong == 0 else 0.0,
        "filesize": size,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("task_num", type=int)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.model, args.task_num), indent=1))
