"""Two-layer integer circuit for k-local rules that a single Conv can't express.

Architecture (all static, no banned ops):
    input[1,10,30,30]
      -> Conv(W1 [H,10,k,k], b1)         # hidden pre-activation, float
      -> Relu                            # hidden = max(0, .)  (bounded ints)
      -> Cast to bool? no: keep as the counted intermediate
      -> Conv(W2 [10,H,1,1], b2)         # combine hidden -> output logits
      == output                          # sign-thresholded by the grader

Cost = memory(hidden [1,H,30,30]) + params(W1,b1,W2,b2). To minimize memory we
push H as low as possible and, when the fit is integer-valued and bounded, cast
the hidden tensor to a 1-byte dtype (uint8) so memory = H*900 instead of H*3600.

We fit with a small ReLU MLP (torch) on the (patch -> output_color) table, using
a margin/hinge loss so the final sign pattern is correct, then round weights to
integers and re-verify EXACTLY on all patches (official + fresh). If integer
rounding breaks correctness we widen the margin and retrain. Hidden width H is
searched from small to large; first H that yields an exactly-correct integer net
wins.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gen_examples  # noqa: E402


def local_table(task_num, r, fresh=3000):
    """Collect the exact (patch -> out_color) table; None if not r-local.
    out_color == -1 marks border/padding cells that must stay all <= 0."""
    data = json.loads((ROOT / "data" / f"task{task_num:03d}.json").read_text())
    exs = [e for k in ("train", "test", "arc-gen") for e in data[k]]
    exs += gen_examples.generate(task_num, fresh)
    k = 2 * r + 1
    seen = {}
    for e in exs:
        gi = np.array(e["input"], np.int8)
        go = np.array(e["output"], np.int8)
        if max(gi.shape + go.shape) > 30 or gi.shape != go.shape:
            return None
        H, W = gi.shape
        big = np.full((30 + 2 * r, 30 + 2 * r), -1, np.int8)
        big[r:r + H, r:r + W] = gi
        for rr in range(min(H + r, 30)):
            for cc in range(min(W + r, 30)):
                patch = big[rr:rr + k, cc:cc + k]
                v = int(go[rr, cc]) if (rr < H and cc < W) else -1
                key = patch.tobytes()
                if key in seen:
                    if seen[key] != v:
                        return None
                else:
                    seen[key] = v
    seen.setdefault(np.full((k, k), -1, np.int8).tobytes(), -1)
    return seen, k


def table_to_arrays(seen, k):
    X, y = [], []
    for key, v in seen.items():
        patch = np.frombuffer(key, np.int8).reshape(k, k)
        feat = np.stack([(patch == ch) for ch in range(10)]).astype(np.float32)
        X.append(feat)
        y.append(v)
    return np.array(X), np.array(y)  # X:[N,10,k,k], y:[N]


def fit(task_num, r, H_list=(2, 3, 4, 6, 8), fresh=3000):
    import torch

    tab = local_table(task_num, r, fresh)
    if tab is None:
        return {"result": f"not {r}-local on {fresh} fresh"}
    seen, k = tab
    X, y = table_to_arrays(seen, k)
    Xt = torch.tensor(X)
    N = len(y)
    # target sign matrix S[N,10]: +1 for the correct channel, -1 otherwise;
    # border cells (y==-1): all -1.
    S = -torch.ones(N, 10)
    for i, v in enumerate(y):
        if v >= 0:
            S[i, v] = 1.0

    for H in H_list:
        torch.manual_seed(0)
        W1 = torch.zeros(H, 10, k, k, requires_grad=True)
        b1 = torch.zeros(H, requires_grad=True)
        W2 = torch.zeros(10, H, requires_grad=True)
        b2 = torch.zeros(10, requires_grad=True)
        torch.nn.init.normal_(W1, std=0.3)
        torch.nn.init.normal_(W2, std=0.3)
        opt = torch.optim.Adam([W1, b1, W2, b2], lr=0.05)
        for step in range(4000):
            opt.zero_grad()
            hid = torch.relu(torch.einsum("nikl,hikl->nh", Xt, W1) + b1)
            logit = hid @ W2.t() + b2                      # [N,10]
            # hinge: want S*logit >= margin
            loss = torch.relu(2.0 - S * logit).mean()
            loss = loss + 1e-3 * (W1.abs().mean() + W2.abs().mean())
            loss.backward()
            opt.step()
            if loss.item() < 1e-6:
                break
        # snap to integers at several scales, verify exactly
        for scale in (2, 4, 8, 16, 32):
            w1 = torch.round(W1.detach() * scale)
            bb1 = torch.round(b1.detach() * scale)
            w2 = torch.round(W2.detach() * scale)
            bb2 = torch.round(b2.detach() * scale * scale)
            hid = torch.relu(torch.einsum("nikl,hikl->nh", Xt, w1) + bb1)
            logit = hid @ w2.t() + bb2
            pred = logit > 0
            gt = S > 0
            # every correct channel > 0, every other <= 0
            ok = bool(((pred == gt) | (S < 0) & (logit <= 0)).all()
                      and (logit[gt] > 0).all() and (logit[~gt] <= 0).all())
            if ok:
                hidmax = float(hid.max())
                return {"W1": w1.numpy(), "b1": bb1.numpy(),
                        "W2": w2.numpy(), "b2": bb2.numpy(),
                        "H": H, "k": k, "scale": scale, "hidmax": hidmax}
    return {"result": "no integer 2-layer fit found"}


def build(fit, uint8_hidden=True):
    import onnx
    from onnx import helper, TensorProto

    H, k = fit["H"], fit["k"]
    W1 = fit["W1"].astype(np.float32)
    b1 = fit["b1"].astype(np.float32)
    W2 = fit["W2"].reshape(10, H, 1, 1).astype(np.float32)
    b2 = fit["b2"].astype(np.float32)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    yv = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    inits = [
        helper.make_tensor("W1", TensorProto.FLOAT, list(W1.shape), W1.flatten().tolist()),
        helper.make_tensor("b1", TensorProto.FLOAT, [H], b1.tolist()),
        helper.make_tensor("W2", TensorProto.FLOAT, list(W2.shape), W2.flatten().tolist()),
        helper.make_tensor("b2", TensorProto.FLOAT, [10], b2.tolist()),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "W1", "b1"], ["pre"],
                         kernel_shape=[k, k], pads=[k // 2] * 4),
        helper.make_node("Relu", ["pre"], ["hid"]),
    ]
    if uint8_hidden:
        nodes.append(helper.make_node("Cast", ["hid"], ["hidq"], to=TensorProto.UINT8))
        nodes.append(helper.make_node("Cast", ["hidq"], ["hidf"], to=TensorProto.FLOAT))
        conv2_in = "hidf"
    else:
        conv2_in = "hid"
    nodes.append(helper.make_node("Conv", [conv2_in, "W2", "b2"], ["output"],
                                  kernel_shape=[1, 1]))
    graph = helper.make_graph(nodes, "g", [x], [yv], inits)
    m = helper.make_model(graph, ir_version=10,
                          opset_imports=[helper.make_opsetid("", 10)])
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    import onnx

    task_num = int(sys.argv[1])
    r = int(sys.argv[2])
    f = fit(task_num, r)
    if "W1" not in f:
        print(json.dumps({"task": task_num, **f}))
        sys.exit(0)
    # uint8 hidden requires hidmax <= 255
    uint8 = f["hidmax"] <= 255
    m = build(f, uint8_hidden=uint8)
    out = ROOT / "candidates" / f"task{task_num:03d}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fit2l.onnx"
    onnx.save(m, p)
    print(json.dumps({"task": task_num, "saved": str(p), "H": f["H"],
                      "k": f["k"], "uint8": uint8, "hidmax": f["hidmax"]}))
