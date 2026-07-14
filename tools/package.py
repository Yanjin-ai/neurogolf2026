"""Build submission.zip from the candidate pool.

Pool layout: candidates/taskNNN/<name>.onnx plus <name>.json (a validate.py
report with "ok": true). For each task pick the validated candidate with the
lowest cost; fall back to the baseline net when no candidate exists.
"""
import json
import pathlib
import shutil
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "candidates"
BASELINE = ROOT / "baselines" / "udit" / "nets"
BASELINE_SCORES = ROOT / "baselines" / "udit" / "scores_isolated.json"
OUT = ROOT / "submissions"


def best_candidate(task_num: int):
    pool = CANDIDATES / f"task{task_num:03d}"
    best = None
    if pool.is_dir():
        for report_path in pool.glob("*.json"):
            if report_path.name.startswith("._"):  # exFAT AppleDouble junk
                continue
            report = json.loads(report_path.read_text())
            if not report.get("ok"):
                continue
            model = report_path.with_suffix(".onnx")
            if not model.exists():
                continue
            if best is None or report["cost"] < best[1]:
                best = (model, report["cost"], report["points"])
    return best


def build(tag: str) -> pathlib.Path:
    OUT.mkdir(exist_ok=True)
    baseline_scores = json.loads(BASELINE_SCORES.read_text())
    zpath = OUT / f"submission_{tag}.zip"
    manifest, total = {}, 0.0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in range(1, 401):
            name = f"task{n:03d}.onnx"
            cand = best_candidate(n)
            base_pts = baseline_scores.get(str(n), {}).get("points", 0)
            if cand:
                zf.write(cand[0], name)
                manifest[name] = {"source": str(cand[0].relative_to(ROOT)),
                                  "cost": cand[1], "points": cand[2]}
                total += cand[2]
            else:
                zf.write(BASELINE / name, name)
                manifest[name] = {"source": "baseline", "points": base_pts}
                total += base_pts
    (OUT / f"manifest_{tag}.json").write_text(json.dumps(
        {"expected_local_total": round(total, 2), "tasks": manifest}, indent=1))
    print(f"{zpath.name}: expected local total = {total:.2f}")
    return zpath


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    args = ap.parse_args()
    build(args.tag)
