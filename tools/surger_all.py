"""Apply surgery_pipeline to all merge-best nets; report total realizable gain."""
import concurrent.futures as cf
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
OUT = ROOT / "surgered"
OUT.mkdir(exist_ok=True)
best = json.load(open(ROOT / "experiments" / "merge_best.json"))


def one(item):
    t, v = item
    n = int(t)
    out = str(OUT / f"task{n:03d}.onnx")
    try:
        r = subprocess.run([str(PY), str(ROOT / "tools" / "surgery_pipeline.py"),
                            v["path"], str(n), out],
                           capture_output=True, text=True, timeout=600)
        d = json.loads(r.stdout)
        d["task"] = n
        return d
    except Exception as e:  # noqa: BLE001
        return {"task": n, "error": str(e)[:80]}


imp = []
tot = 0.0
with cf.ThreadPoolExecutor(max_workers=8) as pool:
    done = 0
    for d in pool.map(one, best.items()):
        done += 1
        if done % 50 == 0:
            print(f"  {done}/400", flush=True)
        if d.get("improved"):
            imp.append(d)
            tot += d.get("gain", 0)

imp.sort(key=lambda d: -d.get("gain", 0))
print(f"\nsurgery improved {len(imp)}/400 tasks, total gain = +{tot:.4f}")
for d in imp[:20]:
    print(f"  task{d['task']:03d}: {d['old']}->{d['new']} (+{d['gain']:.3f}) {d.get('applied')}")
json.dump(imp, open(ROOT / "experiments" / "surgery_gains.json", "w"), indent=1)
