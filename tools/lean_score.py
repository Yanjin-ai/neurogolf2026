"""Profiler-free scorer for STATIC-shape nets (no temp files, fast).

For fully static graphs, the grader's profiler observes exactly the static
shapes, so cost = static params + static intermediate-tensor bytes. Correctness
is checked with a plain ORT run (profiling disabled -> no trace files).

Returns {right, wrong, cost, points} or {error}.
"""
import json
import math
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ngu  # noqa: E402

DT_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}


def static_cost(model):
    params = 0
    for init in model.graph.initializer:
        params += math.prod(init.dims) if init.dims else 1
    for si in model.graph.sparse_initializer:
        params += math.prod(si.values.dims) if si.values.dims else 1
    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    params += math.prod(a.t.dims) if a.t.dims else 1
                elif a.name == "value_floats":
                    params += len(a.floats)
                elif a.name == "value_ints":
                    params += len(a.ints)
    try:
        inf = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:  # noqa: BLE001
        return None, None
    mem = 0
    for vi in list(inf.graph.value_info):
        if vi.name in ("input", "output"):
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None, None
        numel = 1
        for d in tt.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                return None, None
            numel *= d.dim_value
        mem += numel * DT_BYTES.get(tt.elem_type, 4)
    return mem, params


def score(model_path, task_num):
    try:
        model = onnx.load(model_path)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load {e}"}
    if pathlib.Path(model_path).stat().st_size > ngu._FILESIZE_LIMIT_IN_BYTES:
        return {"error": "too large"}
    for node in model.graph.node:
        if node.op_type.upper() in ("LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT",
                                    "FUNCTION", "COMPRESS") or "Sequence" in node.op_type:
            return {"error": f"banned {node.op_type}"}
    mem, params = static_cost(model)
    if mem is None:
        return {"error": "cost/shape"}
    try:
        san = ngu.sanitize_model(model)
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = onnxruntime.InferenceSession(san.SerializeToString(), opt)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load-ort {e}"}
    task = json.loads((ROOT / "data" / f"task{task_num:03d}.json").read_text())
    right = wrong = 0
    try:
        for split in ("train", "test", "arc-gen"):
            for ex in task.get(split, []):
                b = ngu.convert_to_numpy(ex)
                if b is None:
                    continue
                out = sess.run(["output"], {"input": b["input"]})[0]
                if np.array_equal((out > 0.0).astype(float), b["output"]):
                    right += 1
                else:
                    wrong += 1
    except Exception as e:  # noqa: BLE001
        return {"error": f"run {e}"}
    cost = mem + params
    return {"right": right, "wrong": wrong, "cost": cost,
            "points": max(1.0, 25.0 - math.log(max(1.0, cost))) if wrong == 0 else 0.0}


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], int(sys.argv[2]))))
