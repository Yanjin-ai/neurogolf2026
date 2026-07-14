"""Function-preserving cost reduction (graph surgery) for a single net.

Tries onnxslim + a few safe rewrites, keeps the result only if it is still
EXACTLY correct on official examples AND cheaper than the input. Pure cost
golf: the function is unchanged, so it is Kaggle-safe on any set the input
was correct on.
"""
import json
import pathlib
import sys

import onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from lean_score import score, static_cost  # noqa: E402


def try_onnxslim(model):
    try:
        import onnxslim
        return onnxslim.slim(model)
    except Exception:  # noqa: BLE001
        return None


def strip_metadata(model):
    """Remove doc_strings / metadata_props / producer info (file-size only, but
    harmless)."""
    model.doc_string = ""
    del model.metadata_props[:]
    model.producer_name = ""
    model.producer_version = ""
    for node in model.graph.node:
        node.doc_string = ""
    return model


def surgeon(in_path, task_num, out_path):
    orig = onnx.load(in_path)
    base = score(in_path, task_num)
    if base.get("wrong") != 0 or "error" in base:
        return {"skip": "input not correct", "cost": base.get("cost")}
    base_cost = base["cost"]

    candidates = []
    slim = try_onnxslim(onnx.load(in_path))
    if slim is not None:
        candidates.append(strip_metadata(slim))
    candidates.append(strip_metadata(onnx.load(in_path)))  # metadata-only

    best = None
    for cand in candidates:
        try:
            onnx.save(cand, out_path)
        except Exception:  # noqa: BLE001
            continue
        r = score(out_path, task_num)
        if r.get("wrong") == 0 and "error" not in r and r["cost"] < base_cost:
            if best is None or r["cost"] < best[0]:
                best = (r["cost"], r["points"])
                onnx.save(cand, out_path)  # keep the good one on disk
    if best:
        return {"improved": True, "old_cost": base_cost, "new_cost": best[0],
                "gain": round(best[1] - (25 - __import__("math").log(max(1, base_cost))), 3)}
    pathlib.Path(out_path).unlink(missing_ok=True)
    return {"improved": False, "cost": base_cost}


if __name__ == "__main__":
    in_path, task_num, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    print(json.dumps(surgeon(in_path, task_num, out_path)))
