"""Score many models with per-model process isolation.

Required on this machine: ORT 1.24.4 (macOS) reuses the first session's
compiled graph for later same-shaped models in one process, silently returning
the wrong function. One process per model makes results trustworthy.
"""
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = pathlib.Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"


def score_one(args):
    model_path, task_num = args
    proc = subprocess.run(
        [str(PY), str(ROOT / "src" / "oracle.py"), str(model_path), str(task_num)],
        capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=600)
    if proc.returncode != 0:
        return task_num, {"error": f"subprocess failed: {proc.stderr[-300:]}"}
    try:
        return task_num, json.loads(proc.stdout)
    except json.JSONDecodeError:
        return task_num, {"error": f"bad output: {proc.stdout[-300:]}"}


def main(model_dir: str, out_path: str, workers: int = 6):
    jobs = [(pathlib.Path(model_dir).resolve() / f"task{n:03d}.onnx", n)
            for n in range(1, 401)]
    jobs = [(p, n) for p, n in jobs if p.exists()]
    results = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for task_num, res in pool.map(score_one, jobs):
            results[task_num] = res
            if len(results) % 50 == 0:
                print(f"{len(results)} scored", flush=True)
    json.dump(results, open(out_path, "w"), indent=1)
    total = sum(v.get("points", 0) for v in results.values())
    errs = [n for n, v in results.items() if "error" in v]
    fails = [n for n, v in results.items() if v.get("wrong")]
    print(f"TOTAL: {total:.2f} | errors: {errs} | failing: {sorted(fails)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
