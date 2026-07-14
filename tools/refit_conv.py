"""Re-fit a task ALREADY solved by a single Conv, searching for the smallest
kernel (and thus fewest params) whose integer-weight linear thresholds are
still exactly correct. Linear separability is guaranteed (baseline is a single
conv), so the only question is how small the kernel can get.

Tries every kernel window (kh in 1..KH, kw in 1..KW) centered/anchored to cover
the receptive field, LP-fits per output channel, rounds to integers, verifies
exactly on all official + fresh patches, and returns the cheapest that works.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_examples  # noqa: E402


def all_patches(task_num, kh, kw, fresh=3000):
    """Collect (feat, out_color) for every in-grid cell with a kh x kw window,
    plus border cells that must stay <=0. Anchor: window top-left offset so the
    center cell is at (kh//2, kw//2). Returns None if not (kh,kw)-local."""
    data = json.loads((ROOT / "data" / f"task{task_num:03d}.json").read_text())
    exs = [e for k in ("train", "test", "arc-gen") for e in data[k]]
    exs += gen_examples.generate(task_num, fresh)
    ph, pw = kh // 2, kw // 2
    seen = {}
    for e in exs:
        gi = np.array(e["input"], np.int8)
        go = np.array(e["output"], np.int8)
        if max(gi.shape + go.shape) > 30 or gi.shape != go.shape:
            return None
        H, W = gi.shape
        big = np.full((30 + kh, 30 + kw), -1, np.int8)
        big[ph:ph + H, pw:pw + W] = gi
        for rr in range(min(H + ph, 30)):
            for cc in range(min(W + pw, 30)):
                patch = big[rr:rr + kh, cc:cc + kw]
                key = patch.tobytes()
                v = int(go[rr, cc]) if (rr < H and cc < W) else -1
                if key in seen:
                    if seen[key] != v:
                        return None
                else:
                    seen[key] = v
    seen.setdefault(np.full((kh, kw), -1, np.int8).tobytes(), -1)
    X, y = [], []
    for key, v in seen.items():
        patch = np.frombuffer(key, np.int8).reshape(kh, kw)
        feat = np.stack([(patch == ch) for ch in range(10)]).astype(np.float64)
        X.append(feat.reshape(10 * kh * kw))
        y.append(v)
    return np.array(X), np.array(y)


def fit_channel(X, pos, bound=200.0, bias=False):
    from scipy.optimize import linprog
    Xa = np.hstack([X, np.ones((len(X), 1))]) if bias else X
    n = Xa.shape[1]
    A = np.vstack([-Xa[pos], Xa[~pos]])
    b = -np.ones(len(A))
    res = linprog(np.ones(2 * n), A_ub=np.hstack([A, -A]), b_ub=b,
                  bounds=[(0, bound)] * (2 * n), method="highs")
    if not res.success:
        return None
    return res.x[:n] - res.x[n:]


def round_verify(w, X, pos, nfeat):
    for s in (1, 2, 3, 4, 6, 8, 16, 32):
        wi = np.round(w * s)
        sc = X @ wi[:nfeat] + (wi[-1] if len(wi) > nfeat else 0.0)
        if (sc[pos] > 0).all() and (sc[~pos] <= 0).all():
            return wi
    return None


def try_kernel(task_num, kh, kw):
    data = all_patches(task_num, kh, kw)
    if data is None:
        return None
    X, y = data
    nfeat = 10 * kh * kw
    W = np.zeros((10, nfeat))
    bias = np.zeros(10)
    used_bias = False
    for ch in range(10):
        pos = (y == ch)
        if not pos.any():
            continue
        w = fit_channel(X, pos)
        bh = False
        if w is None:
            w = fit_channel(X, pos, bias=True)
            bh = True
        if w is None:
            return None
        wi = round_verify(w, X, pos, nfeat)
        if wi is None:
            return None
        if bh:
            W[ch] = wi[:nfeat]
            bias[ch] = wi[-1]
            used_bias = True
        else:
            W[ch] = wi
    return {"W": W.reshape(10, 10, kh, kw), "B": bias if used_bias else None,
            "kh": kh, "kw": kw, "params": nfeat + (10 if used_bias else 0)}


def build(fit):
    import onnx
    from onnx import helper, TensorProto
    W = fit["W"].astype(np.float32)
    kh, kw = fit["kh"], fit["kw"]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    inits = [helper.make_tensor("W", TensorProto.FLOAT, list(W.shape), W.flatten().tolist())]
    inputs = ["input", "W"]
    if fit["B"] is not None:
        inits.append(helper.make_tensor("B", TensorProto.FLOAT, [10], fit["B"].astype(np.float32).tolist()))
        inputs.append("B")
    node = helper.make_node("Conv", inputs, ["output"], kernel_shape=[kh, kw],
                            pads=[kh // 2, kw // 2, kh // 2, kw // 2])
    g = helper.make_graph([node], "g", [x], [y], inits)
    m = helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(m, full_check=True)
    return m


def refit(task_num, max_kh=9, max_kw=9):
    """Search kernels small->large; return cheapest working fit."""
    best = None
    cands = sorted([(kh, kw) for kh in range(1, max_kh + 1, 2)
                    for kw in range(1, max_kw + 1, 2)],
                   key=lambda p: p[0] * p[1])
    for kh, kw in cands:
        fit = try_kernel(task_num, kh, kw)
        if fit is not None:
            return fit  # first (smallest) that works
    return best


if __name__ == "__main__":
    import onnx
    task_num = int(sys.argv[1])
    max_k = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    fit = refit(task_num, max_k, max_k)
    if fit is None:
        print(json.dumps({"task": task_num, "result": "no kernel fits"}))
        sys.exit(0)
    m = build(fit)
    out = ROOT / "candidates" / f"task{task_num:03d}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "refit.onnx"
    onnx.save(m, p)
    print(json.dumps({"task": task_num, "saved": str(p),
                      "kernel": [fit["kh"], fit["kw"]], "params": fit["params"]}))
