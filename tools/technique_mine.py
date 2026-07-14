"""Mine a technique library across all bundles + flag transfer targets.

Two fingerprints per task:
  1. STRUCTURE  = the set of arc-gen generator primitives (common.* calls) ->
                  the computational shape of the true rule.
  2. TECHNIQUE  = the op-class fingerprint of every bundle's net for the task
                  (bitpacked / quantized / maxpool-prop / gather / dense / conv)
                  + that net's cost.

Then cluster tasks by STRUCTURE. Within a cluster, a task whose CHEAPEST known
net is far more expensive than a structurally-similar sibling is a transfer
target: a cheaper technique provably exists for that structure, but no bundle
applied it here. Those are where a hand port has real hope (vs blind attack).
"""
import json
import math
import pathlib
import re
import sys
from collections import Counter, defaultdict

import onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_examples as g  # noqa: E402

IDS = g.task_ids()
TASKS = ROOT / "third_party" / "arc-gen" / "tasks"
best = json.load(open(ROOT / "experiments" / "merge_best.json"))


def structure_sig(task_num):
    h = IDS[task_num - 1]
    f = TASKS / f"task_{h}.py"
    if not f.exists():
        return ("?",)
    s = f.read_text()
    prims = sorted(set(re.findall(r"common\.(\w+)", s)))
    # keep the structural (non-color) primitives -> the "shape" of the rule
    color = {"black", "blue", "red", "green", "yellow", "gray", "grey", "cyan",
             "magenta", "orange", "maroon", "pink", "random_color", "random_colors"}
    struct = tuple(p for p in prims if p not in color
                   and p not in ("grid", "grids", "randint", "sample", "get_pixel",
                                 "all_pixels", "shuffle"))
    return struct or ("plain",)


def technique(model_path):
    try:
        m = onnx.load(model_path)
    except Exception:  # noqa: BLE001
        return "?"
    oc = Counter(n.op_type for n in m.graph.node)
    bit = oc["BitShift"] + oc["BitwiseAnd"] + oc["BitwiseOr"]
    if bit > 20:
        return "bitpacked"
    if oc.get("QLinearConv", 0) > 0:
        return "quantized-conv"
    if oc.get("MaxPool", 0) >= 4:
        return "maxpool-prop"
    if oc.get("Conv", 0) > 0:
        return "conv"
    if oc.get("Einsum", 0) > 0:
        return "einsum"
    if oc.get("Gather", 0) >= max(1, len(m.graph.node) // 4):
        return "gather-heavy"
    return "dense"


def main():
    struct = {n: structure_sig(n) for n in range(1, 401)}
    tech, cost = {}, {}
    for n in range(1, 401):
        p = best[str(n)]["path"]
        tech[n] = technique(p)
        cost[n] = best[str(n)]["cost"]

    # cluster by structure signature
    clusters = defaultdict(list)
    for n in range(1, 401):
        clusters[struct[n]].append(n)

    # within each multi-member cluster, flag high-cost outliers
    print("=== technique distribution (current incumbents) ===")
    for t, c in Counter(tech.values()).most_common():
        print(f"  {t:>15}: {c} tasks")

    print("\n=== STRUCTURE clusters with cost spread (transfer targets) ===")
    targets = []
    for sig, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        costs = sorted((cost[n], n) for n in members)
        lo_c, lo_n = costs[0]
        hi_c, hi_n = costs[-1]
        if hi_c > lo_c * 2 and hi_c - lo_c > 3000:
            # expensive members whose cheap sibling proves a cheaper circuit exists
            cheap_tech = tech[lo_n]
            for c, n in costs:
                if c > lo_c * 2 and c - lo_c > 3000 and tech[n] != cheap_tech:
                    gain = (25 - math.log(max(1, lo_c))) - (25 - math.log(max(1, c)))
                    targets.append((abs(gain), n, c, tech[n], lo_n, lo_c,
                                    cheap_tech, sig))
    targets.sort(reverse=True)
    print(f"{'task':>5}{'cost':>7} {'its-tech':>14} | cheap sibling {'task':>5}{'cost':>7} {'tech':>14}  potential  structure")
    for gain, n, c, tn, ln, lc, lt, sig in targets[:25]:
        sigs = ",".join(sig)[:34]
        print(f"{n:>5}{c:>7} {tn:>14} |          {ln:>5}{lc:>7} {lt:>14}  +{gain:>6.2f}  [{sigs}]")
    json.dump([{"task": n, "cost": c, "tech": tn, "cheap_sibling": ln,
                "sibling_cost": lc, "sibling_tech": lt, "potential": round(gain, 3),
                "structure": list(sig)}
               for gain, n, c, tn, ln, lc, lt, sig in targets],
              open(ROOT / "experiments" / "transfer_targets.json", "w"), indent=1)
    print(f"\n{len(targets)} transfer targets written to experiments/transfer_targets.json")


if __name__ == "__main__":
    main()
