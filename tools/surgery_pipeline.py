"""Apply the public graph-surgery techniques (seddiktrk catalog) to a net.

Chains the safe, function-preserving cost reductions:
  generic simplify -> lossless cleanup -> constant->initializer rescue ->
  index surgery -> broadcast compression -> int32 downcast ->
  structural micro-rewrites -> FP16 surgery.

After each stage we re-check EXACT correctness on official examples and keep the
stage only if it is still correct AND strictly cheaper. Function is preserved,
so a kept result is Kaggle-safe on any set the input was correct on.
"""
import copy
import pathlib
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
from lean_score import score, static_cost  # noqa: E402

try:
    import onnxsim  # noqa: F401
except Exception:  # noqa: BLE001
    onnxsim = None
try:
    from onnxsim import simplify as _onnxsim_simplify  # noqa: F401
except Exception:  # noqa: BLE001
    _onnxsim_simplify = None


class Config:
    use_onnx_simplification = True
    use_safe_cleanup = True
    use_index_surgery = True
    use_broadcast_compression = True
    use_int32_downcasting = True
    use_micro_rewrites = True
    use_fp16_surgery = True


# load the extracted surgery function definitions
_defs = (ROOT / "study" / "surgery_defs.py").read_text()
# neutralise notebook-only imports that pull heavy/absent deps
_defs = _defs.replace("from tqdm.auto import tqdm", "def tqdm(x, *a, **k):\n    return x")
import re as _re
_defs = _re.sub(r"^import pandas.*$", "", _defs, flags=_re.M)
_defs = _re.sub(r"^import pandas as pd.*$", "", _defs, flags=_re.M)
exec(compile(_defs, "surgery_defs", "exec"), globals())


def _cost(model, path_tmp):
    onnx.save(model, path_tmp)
    m, p = static_cost(onnx.load(path_tmp))
    return None if m is None else m + p


STAGES = [
    ("simplify", lambda m: apply_generic_onnx_simplification(m)),  # noqa: F821
    ("cleanup", lambda m: apply_safe_lossless_cleanup(m)),  # noqa: F821
    ("const_rescue", lambda m: apply_constant_to_initializer_rescue(m)),  # noqa: F821
    ("index", lambda m: apply_index_surgery(m)),  # noqa: F821
    ("broadcast", lambda m: apply_broadcast_compression(m)),  # noqa: F821
    ("int32", lambda m: apply_int32_downcasting(m)),  # noqa: F821
    ("micro", lambda m: apply_structural_micro_rewrites(m)),  # noqa: F821
    ("fp16", lambda m: apply_fp16_surgery_v2(m)),  # noqa: F821
]


def _unwrap(res):
    return res[0] if isinstance(res, tuple) else res


def surger(in_path, task_num, out_path):
    tmp = str(ROOT / ".tmp" / f"_surg_{task_num}.onnx")
    base = score(in_path, task_num)
    if base.get("wrong") != 0 or "error" in base:
        return {"skip": "input incorrect"}
    best_cost = base["cost"]
    model = onnx.load(in_path)
    applied = []
    for name, fn in STAGES:
        try:
            cand = _unwrap(fn(copy.deepcopy(model)))
        except Exception:  # noqa: BLE001
            continue
        try:
            r = score_model(cand, task_num, tmp)
        except Exception:  # noqa: BLE001
            continue
        if r.get("wrong") == 0 and "error" not in r and r["cost"] < best_cost:
            model = cand
            applied.append((name, best_cost, r["cost"]))
            best_cost = r["cost"]
    if best_cost < base["cost"]:
        onnx.save(model, out_path)
        import math
        return {"improved": True, "old": base["cost"], "new": best_cost,
                "gain": round((25 - math.log(best_cost)) - (25 - math.log(base["cost"])), 4),
                "applied": [a[0] for a in applied]}
    return {"improved": False, "cost": base["cost"]}


def score_model(model, task_num, tmp_path):
    onnx.save(model, tmp_path)
    return score(tmp_path, task_num)


if __name__ == "__main__":
    import json
    print(json.dumps(surger(sys.argv[1], int(sys.argv[2]), sys.argv[3])))
