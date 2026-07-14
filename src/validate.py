"""Three-gate validation for candidate networks (PLAN.md §5).

Gate 1: exact pass on ALL official examples (train/test/arc-gen) + cost measure
        via the official scoring stack (oracle.py).
Gate 2: exact pass on >=N freshly generated arc-gen examples (private-set
        proxy) + color-permutation perturbation screen where applicable.
Gate 3: constraint audit: banned ops, static shapes, single I/O, file size.

A candidate enters the pool only if all gates pass.
"""
import json
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))

import neurogolf_utils as ngu  # noqa: E402
from oracle import evaluate  # noqa: E402
import gen_examples  # noqa: E402

BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def _session(model_path: str) -> onnxruntime.InferenceSession:
    sanitized = ngu.sanitize_model(onnx.load(model_path))
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return onnxruntime.InferenceSession(sanitized.SerializeToString(), opt)


def _run_examples(session, examples) -> tuple[int, int]:
    right = wrong = 0
    for ex in examples:
        bench = ngu.convert_to_numpy(ex)
        if bench is None:
            continue
        out = session.run(["output"], {"input": bench["input"]})[0]
        if np.array_equal((out > 0.0).astype(float), bench["output"]):
            right += 1
        else:
            wrong += 1
    return right, wrong


def constraint_audit(model_path: str) -> list[str]:
    problems = []
    model = onnx.load(model_path)
    if pathlib.Path(model_path).stat().st_size > ngu._FILESIZE_LIMIT_IN_BYTES:
        problems.append("file too large")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        problems.append("must have exactly one input and one output")
    if model.functions:
        problems.append("functions not allowed")
    for opset in model.opset_import:
        if opset.domain not in ("", "ai.onnx"):
            problems.append(f"bad domain {opset.domain}")
    if model.graph.sparse_initializer:
        problems.append("sparse_initializer breaks official sanitizer")
    for node in model.graph.node:
        if node.op_type.upper() in BANNED or "Sequence" in node.op_type:
            problems.append(f"banned op {node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                problems.append(f"subgraph attribute on {node.op_type}")
    return problems


def validate(model_path: str, task_num: int, fresh_count: int = 2000) -> dict:
    report = {"model": str(model_path), "task": task_num}

    problems = constraint_audit(model_path)
    if problems:
        report["fail"] = f"gate3: {problems}"
        return report

    official = evaluate(model_path, task_num)
    report["official"] = official
    if "error" in official or official.get("wrong"):
        report["fail"] = "gate1: official examples / scoring"
        return report

    session = _session(model_path)
    fresh = gen_examples.generate(task_num, fresh_count)
    right, wrong = _run_examples(session, fresh)
    report["fresh"] = {"right": right, "wrong": wrong}
    if wrong:
        report["fail"] = f"gate2: {wrong}/{right + wrong} fresh arc-gen examples"
        return report
    if right < min(fresh_count, 200):
        report["fail"] = f"gate2: only {right} fresh examples available/ran"
        return report

    report["points"] = official["points"]
    report["cost"] = official["cost"]
    report["ok"] = True
    return report


def validate_isolated(model_path: str, task_num: int, fresh_count: int = 2000) -> dict:
    """Run validate() in a fresh subprocess.

    REQUIRED when validating several models from one process: macOS ORT 1.24.4
    reuses the first session's compiled graph for later same-shaped models,
    silently evaluating the wrong function.
    """
    import subprocess

    py = pathlib.Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    proc = subprocess.run(
        [str(py), str(ROOT / "src" / "validate.py"),
         str(pathlib.Path(model_path).resolve()),
         str(task_num), "--fresh", str(fresh_count)],
        capture_output=True, text=True, cwd=str(ROOT / ".tmp"), timeout=1200)
    if proc.returncode != 0:
        return {"error": f"subprocess failed: {proc.stderr[-300:]}"}
    return json.loads(proc.stdout)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("task_num", type=int)
    ap.add_argument("--fresh", type=int, default=2000)
    args = ap.parse_args()
    print(json.dumps(validate(args.model, args.task_num, args.fresh), indent=1))
