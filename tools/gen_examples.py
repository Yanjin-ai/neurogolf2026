"""Generate fresh private-like validation examples via arc-gen generators.

Official visible arc-gen examples use random.seed(2025+i) for i in 0..~261;
we generate from a far seed offset so every example is new. Results are cached
to gen_cache/taskNNN.json (same {input, output} pair format as the task files).
"""
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARC_GEN = ROOT / "third_party" / "arc-gen"
CACHE = ROOT / "gen_cache"
SEED_OFFSET = 1_000_000

sys.path.insert(0, str(ARC_GEN))
sys.path.insert(0, str(ARC_GEN / "tasks"))


def task_ids() -> list[str]:
    """taskNNN (1-based) -> ARC id, via hex-sorted solver ids (verified)."""
    solvers = (ROOT / "third_party" / "arc-dsl" / "solvers.py").read_text()
    return sorted(set(re.findall(r"solve_([0-9a-f]{8})", solvers)))


def generate(task_num: int, count: int = 2000, seed_offset: int = SEED_OFFSET,
             force: bool = False) -> list[dict]:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"task{task_num:03d}.json"
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if len(cached) >= count:
            return cached[:count]
    tid = task_ids()[task_num - 1]
    module = __import__(f"task_{tid}")
    examples = []
    for i in range(count):
        random.seed(seed_offset + i)
        try:
            ex = module.generate()
        except Exception:
            continue  # some generators occasionally reject a sample; skip
        gi, go = ex["input"], ex["output"]
        if max(len(gi), len(gi[0]), len(go), len(go[0])) > 30:
            continue  # grader skips >30x30 examples
        examples.append(ex)
    path.write_text(json.dumps(examples))
    return examples


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("task_num", type=int)
    ap.add_argument("--count", type=int, default=2000)
    args = ap.parse_args()
    exs = generate(args.task_num, args.count)
    print(f"task{args.task_num:03d}: {len(exs)} fresh examples cached")
