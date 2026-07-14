"""Final-day harvest loop: re-scan public kernels, pull fresh ones, re-merge,
submit only if local total improves by >= 0.1 over the last submitted local.

Usage: harvest.py [--submit]   (without --submit it only reports)
State: experiments/harvest_state.json  {last_submitted_local, seen: {ref: lastRunTime}}
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
STATE = ROOT / "experiments" / "harvest_state.json"

state = json.loads(STATE.read_text()) if STATE.exists() else {
    "last_submitted_local": 7270.19, "seen": {}}


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(ROOT))


def main(submit=False):
    # 1. list kernels, newest first
    r = sh([str(KAGGLE), "kernels", "list", "--competition", "neurogolf-2026",
            "--sort-by", "dateRun", "--page-size", "40"])
    fresh = []
    for line in r.stdout.splitlines()[2:]:
        m = re.match(r"^(\S+)\s+.*?(\d{4}-\d{2}-\d{2} [\d:.]+)\s+\d+\s*$", line)
        if not m:
            continue
        ref, ts = m.group(1), m.group(2)
        if state["seen"].get(ref) != ts:
            fresh.append((ref, ts))
    print(f"fresh/updated kernels: {len(fresh)}")

    # 2. pull fresh ones (top 10 newest)
    for ref, ts in fresh[:10]:
        name = ref.split("/")[1]
        dest = ROOT / "bundles" / name
        r = sh([str(KAGGLE), "kernels", "output", ref, "-p", str(dest)],
               timeout=900)
        ok = (dest / "submission.zip").exists() or list(dest.rglob("task*.onnx"))
        print(f"  pull {name}: {'ok' if ok else 'no-onnx'}")
        state["seen"][ref] = ts

    # 3. re-merge
    subprocess.run(["rm", "-rf", str(ROOT / "merge_work")])
    r = sh([str(PY), str(ROOT / "tools" / "merge_bundles.py")], timeout=3600)
    m = re.search(r"expected local total = ([\d.]+)", r.stdout)
    if not m:
        print("MERGE FAILED", r.stdout[-300:], r.stderr[-300:])
        return
    total = float(m.group(1))
    prev = state["last_submitted_local"]
    print(f"merged local total = {total:.2f} (last submitted {prev:.2f})")

    # 4. submit if improved
    if total >= prev + 0.1:
        tag = f"harvest{int(total*100)}"
        sh([str(PY), str(ROOT / "tools" / "package_merge.py"), tag], timeout=1200)
        z = ROOT / "submissions" / f"submission_{tag}.zip"
        subprocess.run(["cp", str(z), str(ROOT / "submissions" / "submission.zip")])
        if submit:
            r = sh([str(KAGGLE), "competitions", "submit", "neurogolf-2026",
                    "-f", str(ROOT / "submissions" / "submission.zip"),
                    "-m", f"{tag}: harvest re-merge (local {total:.2f})"],
                   timeout=600)
            print("SUBMITTED:", r.stdout.strip()[-80:] or r.stderr.strip()[-80:])
            state["last_submitted_local"] = total
        else:
            print(f"IMPROVEMENT +{total-prev:.2f} — packaged, rerun with --submit")
    else:
        print("no improvement worth submitting")
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main(submit="--submit" in sys.argv)
