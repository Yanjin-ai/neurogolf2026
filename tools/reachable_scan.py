"""Reachable-lower-bound scan (Phase 1).

The merge already picks the cheapest CORRECT net across 18 bundles, so for most
tasks the current-best is already an expert-compact circuit (near its structural
floor -- e.g. task064 = 10358, using ArgMax-extent + single-Where output). Those
have NO cost-golf headroom.

A real opportunity = current-best is EXPENSIVE *and* still structurally NAIVE
(the cheapest bundle for this task never applied the compact patterns). We detect
naive-ness from the net itself:
  - float_frac : fraction of intermediate bytes still in float32 (compact nets use
                 bool/uint8/int8)
  - has_10ch   : a multi-channel [1,>=8,H,W] intermediate (naive mask/output; the
                 expert single-Where trick avoids it)
  - n_tensors  : op/tensor count (expert nets fuse into few tensors)
  - technique  : op fingerprint (bitpacked/quantized/einsum = expert-ish;
                 dense/conv/gather-heavy at high cost = likely naive)

opportunity = high cost AND naive signals. Those are where B11 techniques can
plausibly cut cost; the rest are already near floor -> skip.
"""
import json
import math
import pathlib
from collections import Counter

import onnx
from onnx import shape_inference

ROOT = pathlib.Path(__file__).resolve().parent.parent
best = json.load(open(ROOT / "experiments" / "merge_best.json"))
DT = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}
FLOATS = {1, 10, 11}  # f32, f16, f64


def analyze(path):
    m = shape_inference.infer_shapes(onnx.load(path))
    g = m.graph
    io = {v.name for v in list(g.input) + list(g.output)}
    tot = fbytes = 0
    has_10ch = False
    n_big = 0
    for vi in g.value_info:
        if vi.name in io:
            continue
        tt = vi.type.tensor_type
        ne = 1
        ok = True
        dims = []
        for d in tt.shape.dim:
            if not d.HasField("dim_value"):
                ok = False
                break
            ne *= d.dim_value
            dims.append(d.dim_value)
        if not ok:
            continue
        by = ne * DT.get(tt.elem_type, 4)
        tot += by
        if tt.elem_type in FLOATS:
            fbytes += by
        if len(dims) >= 2 and sorted(dims, reverse=True)[1] >= 8 and ne >= 8 * 100:
            has_10ch = True  # a real multi-channel spatial tensor
        if ne >= 900:
            n_big += 1
    oc = Counter(nd.op_type for nd in g.node)
    bit = oc["BitShift"] + oc["BitwiseAnd"] + oc["BitwiseOr"]
    if bit > 20:
        tech = "bitpacked"
    elif oc.get("QLinearConv", 0):
        tech = "quantized"
    elif oc.get("Einsum", 0) and oc.get("ArgMax", 0):
        tech = "expert-einsum"
    elif oc.get("Einsum", 0):
        tech = "einsum"
    elif oc.get("Conv", 0):
        tech = "conv"
    else:
        tech = "dense"
    return {"tot": tot, "float_frac": fbytes / max(1, tot), "has_10ch": has_10ch,
            "n_tensors": len(g.value_info), "n_big": n_big, "tech": tech,
            "n_nodes": len(g.node)}


def main():
    rows = []
    for n in range(1, 401):
        cost = best[str(n)]["cost"]
        try:
            a = analyze(best[str(n)]["path"])
        except Exception as e:  # noqa: BLE001
            continue
        # naive multiplier: how far from expert-compact does the net look?
        naive = (1.0 + 2.0 * a["float_frac"]
                 + (1.5 if a["has_10ch"] else 0.0)
                 + (1.0 if a["n_big"] > 12 else 0.0)
                 + (0.8 if a["tech"] in ("dense", "conv") else 0.0)
                 - (0.5 if a["tech"] in ("bitpacked", "expert-einsum", "quantized") else 0.0))
        # opportunity: log-points recoverable if cost dropped to cost/naive-ish,
        # weighted by how naive it looks. Only meaningful where cost is high.
        opp = math.log(max(1, cost)) * naive if cost > 3000 else 0.0
        rows.append((opp, n, cost, a["float_frac"], a["has_10ch"], a["n_big"],
                     a["tech"], a["src"] if "src" in a else best[str(n)]["path"].split("/")[-2]))
    rows.sort(reverse=True)
    print(f"{'opp':>5} {'task':>4} {'cost':>7} {'flt%':>5} {'10ch':>4} {'nbig':>4} "
          f"{'tech':>13}  source")
    for opp, n, cost, ff, h10, nb, tech, src in rows[:40]:
        print(f"{opp:5.1f} {n:>4} {cost:>7} {ff*100:4.0f}% {'Y' if h10 else '.':>4} "
              f"{nb:>4} {tech:>13}  {src}")
    # summary
    naive_expensive = [r for r in rows if r[2] > 5000 and (r[3] > 0.4 or r[4] or r[6] in ("dense", "conv"))]
    print(f"\n{len(naive_expensive)} tasks are BOTH expensive(>5000) AND naive-looking "
          f"(real cost-golf candidates)")
    json.dump([{"task": n, "cost": c, "float_frac": round(ff, 2), "has_10ch": h10,
                "n_big": nb, "tech": tech, "opp": round(opp, 1)}
               for opp, n, c, ff, h10, nb, tech, src in rows],
              open(ROOT / "experiments" / "reachable_scan.json", "w"), indent=1)


if __name__ == "__main__":
    main()
