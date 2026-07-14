"""Run refit_conv over strict-local tasks; keep only genuine cost wins.
Each fit+validate in its own subprocess (ORT isolation). Validates on 3000
fresh examples to catch overfit fits."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = pathlib.Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"


def process(n, r, baseline_pts, max_k):
    fit = subprocess.run(
        [str(PY), str(ROOT / "tools" / "refit_conv.py"), str(n), str(max_k)],
        capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=5400)
    if fit.returncode != 0:
        return f"refit crashed: {fit.stderr[-160:]}"
    info = json.loads(fit.stdout)
    if "saved" not in info:
        return info.get("result", "no fit")
    model = ROOT / "candidates" / f"task{n:03d}" / "refit.onnx"
    val = subprocess.run(
        [str(PY), str(ROOT / "src" / "validate.py"), str(model), str(n),
         "--fresh", "3000"],
        capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=5400)
    if val.returncode != 0:
        model.unlink(missing_ok=True)
        return f"validate crashed: {val.stderr[-160:]}"
    rep = json.loads(val.stdout)
    if not rep.get("ok"):
        model.unlink(missing_ok=True)
        return f"FAIL {rep.get('fail')}"
    if rep["points"] <= baseline_pts + 1e-9:
        model.unlink(missing_ok=True)
        return f"no gain ({rep['points']:.2f} <= {baseline_pts:.2f})"
    model.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    return (f"WIN {rep['points']:.2f} (+{rep['points']-baseline_pts:.2f}) "
            f"cost={rep['cost']} kernel={info['kernel']}")


def main(hits_file, max_k=7):
    hits = json.loads(pathlib.Path(hits_file).read_text())
    gain = 0.0
    for hit in hits:
        n, r, pts = int(hit[0]), int(hit[1]), float(hit[2])
        msg = process(n, r, pts, max_k)
        print(f"task{n:03d} (r={r}): {msg}", flush=True)
        if msg.startswith("WIN"):
            gain += float(msg.split("+")[1].split(")")[0])
    print(f"TOTAL GENUINE GAIN: +{gain:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7)
