"""Package submission.zip from merge_best.json (best-per-task across bundles).

Falls back to the baseline net for any task missing from the merge.
Optionally re-validates each selected net on fresh arc-gen examples (--safe)
and, for any that fail, swaps in the baseline net (private-set safety).
"""
import json
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = pathlib.Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
BASELINE = ROOT / "baselines" / "udit" / "nets"
OUT = ROOT / "submissions"


def fresh_ok(path, task_num, fresh=2000):
    proc = subprocess.run(
        [str(PY), str(ROOT / "src" / "validate.py"), path, str(task_num),
         "--fresh", str(fresh)],
        capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=900)
    try:
        return json.loads(proc.stdout).get("ok", False)
    except Exception:  # noqa: BLE001
        return False


def build(tag, safe=False):
    OUT.mkdir(exist_ok=True)
    best = json.loads((ROOT / "experiments" / "merge_best.json").read_text())
    zpath = OUT / f"submission_{tag}.zip"
    manifest, total, swaps = {}, 0.0, []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in range(1, 401):
            name = f"task{n:03d}.onnx"
            pick = best.get(str(n))
            if pick and safe and not fresh_ok(pick["path"], n):
                swaps.append(n)
                pick = None  # fall back to baseline
            if pick:
                zf.write(pick["path"], name)
                manifest[name] = {"cost": pick["cost"], "points": pick["points"],
                                  "src": pick["path"].split("/")[-2]}
                total += pick["points"]
            else:
                bp = BASELINE / name
                zf.write(bp, name)
                manifest[name] = {"src": "baseline_fallback"}
    (OUT / f"manifest_{tag}.json").write_text(json.dumps(
        {"expected_local_total": round(total, 2), "safe_swaps": swaps,
         "tasks": manifest}, indent=1))
    print(f"{zpath.name}: expected local total = {total:.2f}"
          + (f" | safe-swapped {len(swaps)} tasks: {swaps}" if safe else ""))
    return zpath


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "merge"
    safe = "--safe" in sys.argv
    build(tag, safe=safe)
