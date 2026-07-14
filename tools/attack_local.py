"""B-line: for each LOCAL candidate task, LP-fit a single Conv, verify on fresh
arc-gen, keep into candidates/ only if EXACT on fresh AND cheaper than incumbent.

Floor-safe: a kept candidate is picked by the merge only if strictly cheaper and
correct; a miss changes nothing. Overfit-guarded by the fresh gate.
"""
import concurrent.futures as cf
import json
import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
CAND = ROOT / "candidates"
best = json.load(open(ROOT / "experiments" / "merge_best.json"))
loc = json.load(open(ROOT / "experiments" / "locality.json"))


def incumbent_cost(t):
    return best[str(t)]["cost"]


def attack(t):
    info = loc.get(str(t)) or loc.get(t)
    if not info or not info.get("local"):
        return None
    r = info["radius"]
    if not isinstance(r, int):
        return None
    inc = incumbent_cost(t)
    # fit single conv at radius r (try r and r+1 if allowed)
    for rr in sorted({r, min(r + 1, 3)}):
        fit = subprocess.run([str(PY), str(ROOT / "tools" / "conv_fitter.py"),
                              str(t), str(rr)],
                             capture_output=True, text=True,
                             cwd=str(ROOT / ".tmp"), timeout=1800)
        try:
            fr = json.loads(fit.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            continue
        if fr.get("fit") == "infeasible" or not fr.get("saved"):
            continue
        out = fr["saved"]
        # validate: exact on all official + 3000 fresh, measure cost
        val = subprocess.run([str(PY), str(ROOT / "src" / "validate.py"),
                              out, str(t), "--fresh", "3000"],
                             capture_output=True, text=True,
                             cwd=str(ROOT / ".tmp"), timeout=1800)
        try:
            vr = json.loads(val.stdout)
        except Exception:  # noqa: BLE001
            continue
        if vr.get("ok") and vr.get("cost", 1e9) < inc:
            g = (25 - math.log(vr["cost"])) - (25 - math.log(inc))
            return {"task": t, "radius": rr, "old": inc, "new": vr["cost"],
                    "gain": round(g, 4), "path": out}
        pathlib.Path(out).unlink(missing_ok=True)
    return None


# target: local tasks, richest headroom first (highest incumbent cost)
targets = sorted((t for t in range(1, 401)
                  if (loc.get(str(t)) or {}).get("local")),
                 key=incumbent_cost, reverse=True)
print(f"attacking {len(targets)} local tasks (floor-safe)...", flush=True)
wins = []
with cf.ThreadPoolExecutor(max_workers=6) as pool:
    done = 0
    for res in pool.map(attack, targets):
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(targets)}  wins so far: {len(wins)}", flush=True)
        if res:
            wins.append(res)
            print(f"  WIN task{res['task']:03d}: {res['old']}->{res['new']} (+{res['gain']:.3f})", flush=True)

wins.sort(key=lambda w: -w["gain"])
tot = sum(w["gain"] for w in wins)
print(f"\nB-line: {len(wins)} wins, total +{tot:.3f}")
json.dump(wins, open(ROOT / "experiments" / "local_wins.json", "w"), indent=1)
