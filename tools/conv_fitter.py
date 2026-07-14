"""Automatic single-Conv compiler for tasks with proven k-local rules.

For a task whose output cell is a function of the (2r+1)^2 input neighborhood,
fit integer conv weights (one linear threshold per output channel) so that
correct channels score >= +1 and all others <= -1 on every training patch.
Solved per channel as an LP with box bounds, then rounded to integers and
re-verified exactly. Kernel is cropped to the bounding box of nonzero weights
(smaller weight tensors = fewer params).

Patch features: 10 one-hot channels; padding cells are all-zero, so the
in-grid indicator (sum of channels) is linear — the LP sees raw one-hots and
can synthesize it. Optional per-channel bias (adds 10 params) only if the
no-bias LP is infeasible.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import gen_examples  # noqa: E402


def collect_patches(task_num: int, r: int, fresh: int = 400):
    """Unique (patch_onehot, out_color) pairs from official + fresh examples."""
    data = json.loads((ROOT / "data" / f"task{task_num:03d}.json").read_text())
    exs = [e for k in ("train", "test", "arc-gen") for e in data[k]]
    exs += gen_examples.generate(task_num, fresh)
    k = 2 * r + 1
    seen = {}
    for e in exs:
        gi = np.array(e["input"], np.int8)
        go = np.array(e["output"], np.int8)
        if max(gi.shape + go.shape) > 30 or gi.shape != go.shape:
            continue
        H, W = gi.shape
        # embed into 30x30 one-hot with sentinel padding beyond the grid
        pad = np.full((H + 2 * r, W + 2 * r), -1, np.int8)
        pad[r:r + H, r:r + W] = gi
        for rr in range(H):
            for cc in range(W):
                patch = pad[rr:rr + k, cc:cc + k]
                key = patch.tobytes()
                v = int(go[rr, cc])
                if seen.setdefault(key, v) != v:
                    raise ValueError(f"not {r}-local at official examples")
    # also constrain PADDING output cells near the grid border: for every
    # example the tensor rows/cols beyond (H,W) must stay <= 0 for all
    # channels. Their patches see partial grid content. Collect them too
    # with out_color = -1 (meaning: every channel must be <= -1).
    for e in exs[:600]:
        gi = np.array(e["input"], np.int8)
        go = np.array(e["output"], np.int8)
        if max(gi.shape + go.shape) > 30 or gi.shape != go.shape:
            continue
        H, W = gi.shape
        big = np.full((30 + 2 * r, 30 + 2 * r), -1, np.int8)
        big[r:r + H, r:r + W] = gi
        for rr in range(min(H + r, 30)):
            for cc in range(min(W + r, 30)):
                if rr < H and cc < W:
                    continue
                patch = big[rr:rr + k, cc:cc + k]
                key = patch.tobytes()
                if key not in seen:
                    seen[key] = -1
    # deep-padding constraint: the all-zero patch must map every channel <= 0
    # (forces bias <= -1 whenever a bias is used)
    seen.setdefault(np.full((k, k), -1, np.int8).tobytes(), -1)
    k2 = k * k
    X, y = [], []
    for key, v in seen.items():
        patch = np.frombuffer(key, np.int8).reshape(k, k)
        feat = np.zeros((10, k, k), np.float64)
        for ch in range(10):
            feat[ch] = (patch == ch)
        X.append(feat.reshape(10 * k2))
        y.append(v)
    return np.array(X), np.array(y), k


def fit_channel(X, pos_mask, bound=9.0, use_bias=False):
    """LP: find w with X_pos @ w >= 1, X_neg @ w <= -1, |w| <= bound."""
    from scipy.optimize import linprog

    Xa = np.hstack([X, np.ones((len(X), 1))]) if use_bias else X
    n = Xa.shape[1]
    # constraints: -Xa_pos w <= -1 ; Xa_neg w <= -1
    A = np.vstack([-Xa[pos_mask], Xa[~pos_mask]])
    b = -np.ones(len(A))
    # minimize L1 via split variables w = p - q, p,q >= 0
    c = np.ones(2 * n)
    A2 = np.hstack([A, -A])
    res = linprog(c, A_ub=A2, b_ub=b, bounds=[(0, bound)] * (2 * n),
                  method="highs")
    if not res.success:
        return None
    w = res.x[:n] - res.x[n:]
    return w


def round_verify(w, X, pos_mask):
    for scale in (1, 2, 3, 4, 8):
        wi = np.round(w * scale)
        s = X @ wi[:X.shape[1]] + (wi[-1] if len(wi) > X.shape[1] else 0.0)
        if (s[pos_mask] > 0).all() and (s[~pos_mask] <= 0).all():
            return wi
    return None


def fit_task(task_num: int, r: int, fresh: int = 400):
    X, y, k = collect_patches(task_num, r, fresh)
    nfeat = 10 * k * k
    weights = np.zeros((10, nfeat))
    biases = np.zeros(10)
    used_bias = False
    for ch in range(10):
        pos = (y == ch)
        if not pos.any():
            # channel never appears: keep all-zero row (never positive: X>=0
            # entries with zero weights give 0 <= 0, fine)
            continue
        w = fit_channel(X, pos, use_bias=False)
        bias_here = False
        if w is None:
            w = fit_channel(X, pos, use_bias=True)
            bias_here = True
        if w is None:
            return None
        wi = round_verify(w, X, pos)
        if wi is None:
            return None
        if bias_here:
            weights[ch] = wi[:nfeat]
            biases[ch] = wi[-1]
            used_bias = True
        else:
            weights[ch] = wi
    W = weights.reshape(10, 10, k, k)
    # crop kernel to bounding box of nonzeros (params = element count)
    nz = np.argwhere(np.abs(W).sum(axis=(0, 1)) > 0)
    if len(nz) == 0:
        r0, r1, c0, c1 = r, r, r, r
    else:
        r0, r1 = nz[:, 0].min(), nz[:, 0].max()
        c0, c1 = nz[:, 1].min(), nz[:, 1].max()
    Wc = W[:, :, r0:r1 + 1, c0:c1 + 1]
    # ONNX pads = [top, left, bottom, right]; negative end-pads trim to 30x30
    pads = [int(r - r0), int(r - c0), int(r1 - r), int(c1 - r)]
    return {"W": Wc, "B": biases if used_bias else None, "pads": pads,
            "kernel": [int(Wc.shape[2]), int(Wc.shape[3])]}


def build_model(fit) -> "onnx.ModelProto":
    import onnx
    from onnx import helper, TensorProto

    W = fit["W"].astype(np.float32)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    inits = [helper.make_tensor("W", TensorProto.FLOAT, list(W.shape),
                                W.flatten().tolist())]
    inputs = ["input", "W"]
    if fit["B"] is not None:
        inits.append(helper.make_tensor("B", TensorProto.FLOAT, [10],
                                        fit["B"].astype(np.float32).tolist()))
        inputs.append("B")
    node = helper.make_node("Conv", inputs, ["output"],
                            kernel_shape=fit["kernel"], pads=fit["pads"])
    graph = helper.make_graph([node], "g", [x], [y], inits)
    model = helper.make_model(graph, ir_version=10,
                              opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    import argparse
    import onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("task_num", type=int)
    ap.add_argument("radius", type=int)
    args = ap.parse_args()
    fit = fit_task(args.task_num, args.radius)
    if fit is None:
        print(json.dumps({"task": args.task_num, "fit": "infeasible"}))
        sys.exit(0)
    model = build_model(fit)
    out_dir = ROOT / "candidates" / f"task{args.task_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lpconv.onnx"
    onnx.save(model, path)
    params = int(np.prod(fit["W"].shape)) + (10 if fit["B"] is not None else 0)
    print(json.dumps({"task": args.task_num, "saved": str(path),
                      "kernel": fit["kernel"], "pads": fit["pads"],
                      "params": params}))
