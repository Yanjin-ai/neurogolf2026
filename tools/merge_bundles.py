"""Best-of-all-public-bundles + our candidates merge (two-phase, disk-safe).

Phase 1 (in-process, instant): compute STATIC cost of every candidate net.
Phase 2 (subprocess-isolated ORT): for each task, walk candidates cheapest-first
and keep the first that is EXACTLY correct on official examples. Only a few
verifications per task, so no 4600-subprocess blowup and no profiler temp files.

Every source net is a public/legal Kaggle submission component, correct on the
public examples, so the min-cost-correct merge is Kaggle-safe on the public LB.
"""
import concurrent.futures as cf
import json
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = pathlib.Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
BUNDLES = ROOT / "bundles"
CAND = ROOT / "candidates"
WORK = ROOT / "merge_work"

import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "tools"))
from lean_score import static_cost  # noqa: E402
import onnx  # noqa: E402


def extract_all():
    WORK.mkdir(exist_ok=True)
    sources = {"udit": ROOT / "baselines" / "udit" / "nets"}
    for bdir in sorted(BUNDLES.glob("*")):
        if not bdir.is_dir():
            continue
        dest = WORK / bdir.name
        dest.mkdir(exist_ok=True)
        z = bdir / "submission.zip"
        if z.exists():
            try:
                with zipfile.ZipFile(z) as zf:
                    for nm in zf.namelist():
                        bn = pathlib.Path(nm).name
                        if bn.endswith(".onnx") and bn.startswith("task") and not bn.startswith("._"):
                            (dest / bn).write_bytes(zf.read(nm))
            except Exception as e:  # noqa: BLE001
                print(f"  {bdir.name}: {e}")
        for o in bdir.rglob("task*.onnx"):
            if not o.name.startswith("._") and not (dest / o.name).exists():
                (dest / o.name).write_bytes(o.read_bytes())
        n = len(list(dest.glob("task*.onnx")))
        if n:
            sources[bdir.name] = dest
            print(f"  {bdir.name}: {n} nets")
    return sources


def cost_of(path):
    try:
        mem, params = static_cost(onnx.load(path))
        return None if mem is None else mem + params
    except Exception:  # noqa: BLE001
        return None


def verify(path, task_num):
    try:
        proc = subprocess.run(
            [str(PY), str(ROOT / "tools" / "lean_score.py"), path, str(task_num)],
            capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=300)
        r = json.loads(proc.stdout)
        return r.get("wrong") == 0 and "error" not in r, r.get("cost")
    except Exception:  # noqa: BLE001
        return False, None


def main():
    sources = extract_all()
    # gather candidate paths per task
    per_task = {n: [] for n in range(1, 401)}
    for src_dir in sources.values():
        for n in range(1, 401):
            p = src_dir / f"task{n:03d}.onnx"
            if p.exists():
                per_task[n].append(str(p))
    for n in range(1, 401):
        cdir = CAND / f"task{n:03d}"
        if cdir.is_dir():
            per_task[n] += [str(o) for o in cdir.glob("*.onnx") if not o.name.startswith("._")]

    # phase 1: static cost, sort cheapest first (in-process, fast)
    print("phase 1: static costs...")
    ranked = {}
    for n in range(1, 401):
        scored = [(cost_of(p), p) for p in per_task[n]]
        scored = [(c, p) for c, p in scored if c is not None]
        scored.sort()
        ranked[n] = scored

    # phase 2: verify cheapest-first per task (parallel across tasks)
    print("phase 2: correctness verification (cheapest-first)...")

    def resolve(n):
        for cost, path in ranked[n][:8]:  # cheapest 8 candidates max
            ok, real_cost = verify(path, n)
            if ok:
                import math
                pts = max(1.0, 25.0 - math.log(max(1.0, real_cost)))
                return n, {"cost": real_cost, "path": path, "points": pts}
        return n, None

    best = {}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        done = 0
        for n, res in pool.map(resolve, range(1, 401)):
            done += 1
            if done % 100 == 0:
                print(f"  resolved {done}/400", flush=True)
            if res:
                best[n] = res

    total = sum(v["points"] for v in best.values())
    json.dump({str(k): v for k, v in best.items()},
              open(ROOT / "experiments" / "merge_best.json", "w"), indent=1)
    # per-source contribution
    from collections import Counter
    src_count = Counter(pathlib.Path(v["path"]).parent.name for v in best.values())
    print(f"\nmerged: {len(best)}/400 tasks, expected local total = {total:.2f}")
    print("source contribution:", dict(src_count.most_common()))
    missing = [n for n in range(1, 401) if n not in best]
    if missing:
        print("MISSING:", missing)
    return best


if __name__ == "__main__":
    main()
