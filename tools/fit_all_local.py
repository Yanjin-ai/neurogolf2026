"""Run conv_fitter over all proven-local tasks; keep candidates that beat
baseline. Each fit+validate runs in its own subprocess (ORT isolation)."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = pathlib.Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"


def run(task_num: int, radius: int, baseline_pts: float) -> str:
    fit = subprocess.run(
        [str(PY), str(ROOT / "tools" / "conv_fitter.py"), str(task_num), str(radius)],
        capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=3600)
    if fit.returncode != 0:
        return f"fit crashed: {fit.stderr[-200:]}"
    info = json.loads(fit.stdout)
    if "saved" not in info:
        return "LP infeasible"
    model = ROOT / "candidates" / f"task{task_num:03d}" / "lpconv.onnx"
    val = subprocess.run(
        [str(PY), str(ROOT / "src" / "validate.py"), str(model), str(task_num),
         "--fresh", "2000"],
        capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=3600)
    if val.returncode != 0:
        return f"validate crashed: {val.stderr[-200:]}"
    report = json.loads(val.stdout)
    if not report.get("ok"):
        model.unlink(missing_ok=True)
        return f"validation failed: {report.get('fail')}"
    if report["points"] <= baseline_pts + 1e-9:
        model.unlink(missing_ok=True)
        return f"no gain ({report['points']:.2f} <= baseline {baseline_pts:.2f})"
    model.with_suffix(".json").write_text(json.dumps(report, indent=1))
    return (f"OK {report['points']:.2f} (baseline {baseline_pts:.2f}, "
            f"+{report['points']-baseline_pts:.2f}) cost={report['cost']} "
            f"kernel={info['kernel']} params={info['params']}")


def main(hit_files):
    hits = []
    for hf in hit_files:
        hits += [tuple(h) for h in json.loads(pathlib.Path(hf).read_text())]
    total_gain = 0.0
    for hit in hits:
        n, r, pts = hit[0], hit[1], hit[2]
        msg = run(int(n), int(r), float(pts))
        print(f"task{int(n):03d} (r={r}): {msg}", flush=True)
        if msg.startswith("OK"):
            total_gain += float(msg.split("+")[1].split(")")[0])
    print(f"TOTAL GAIN: +{total_gain:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
