"""Hand-compiled integer-weight conv circuits for the 4 broken baseline tasks.

All four rules are purely local 3x3 patterns (read from arc-gen sources).
Weights are integers with margin >=1 on both sides of the >0 threshold, so
they are immune to the float-edge fragility that broke the baseline nets.

Kernel index convention: W[out_ch][in_ch][1+dr][1+dc] maps input at (r+dr,c+dc)
to output at (r,c).
"""
import pathlib
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from graphlib import GRID, FLOAT  # noqa: E402

N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def conv_model(w: np.ndarray, bias: np.ndarray | None = None,
               group: int = 1) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", FLOAT, GRID)
    y = helper.make_tensor_value_info("output", FLOAT, GRID)
    inits = [helper.make_tensor("W", FLOAT, list(w.shape), w.flatten().tolist())]
    inputs = ["input", "W"]
    if bias is not None:
        inits.append(helper.make_tensor("B", FLOAT, [10], bias.tolist()))
        inputs.append("B")
    node = helper.make_node("Conv", inputs, ["output"],
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=group)
    graph = helper.make_graph([node], "g", [x], [y], inits)
    model = helper.make_model(graph, ir_version=10,
                              opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


def build_task220() -> onnx.ModelProto:
    """913fb3ed: halo around isolated pixels; 2->1, 3->6, 8->4 halo colors."""
    w = np.zeros((10, 10, 3, 3), np.float32)
    for c in range(10):
        w[c, c, 1, 1] = 1.0  # identity passthrough
    for src, halo in ((2, 1), (3, 6), (8, 4)):
        for dr, dc in N8:
            w[halo, src, 1 + dr, 1 + dc] += 1.0   # halo color on
            w[0, src, 1 + dr, 1 + dc] -= 3.0      # black off at halo cells
    return conv_model(w)


def build_task230() -> onnx.ModelProto:
    """95990924: 2x2 gray squares get colors 1/2/3/4 outside their corners.

    Cell x is diagonally outside corner (dr,dc) of a square iff gray at the
    inward diagonal but not gray at the two inward orthogonal neighbors.
    """
    w = np.zeros((10, 10, 3, 3), np.float32)
    for c in range(10):
        w[c, c, 1, 1] = 1.0
    # (dr,dc) of the inward diagonal for each corner color 1..4
    corners = {1: (1, 1), 2: (1, -1), 3: (-1, 1), 4: (-1, -1)}
    for color, (dr, dc) in corners.items():
        w[color, 5, 1 + dr, 1 + dc] += 1.0
        w[color, 5, 1 + dr, 1] -= 1.0
        w[color, 5, 1, 1 + dc] -= 1.0
        # black channel: out0 = in0 - 9*g(center) - 2*sum(indicators).
        # Sum(ind) = +1 exactly at corner-diag cells, <=0 at other black cells,
        # and = -3 at every gray cell (2x2 blocks), so 9*g(center) dominates.
        w[0, 5, 1 + dr, 1 + dc] -= 2.0
        w[0, 5, 1 + dr, 1] += 2.0
        w[0, 5, 1, 1 + dc] += 2.0
    w[0, 5, 1, 1] -= 9.0
    return conv_model(w)


def build_task294() -> onnx.ModelProto:
    """bb43febb: interiors of gray rectangles turn red (border stays gray)."""
    w = np.zeros((10, 10, 3, 3), np.float32)
    b = np.zeros(10, np.float32)
    w[0, 0, 1, 1] = 1.0                      # black identity
    # red = center gray AND all 8 neighbours gray: sum8 + center - 8 > 0
    for dr, dc in N8:
        w[2, 5, 1 + dr, 1 + dc] = 1.0
    w[2, 5, 1, 1] = 1.0
    b[2] = -8.0
    # gray survives on border: 9*center - sum8 - 1 > 0 iff gray and not interior
    for dr, dc in N8:
        w[5, 5, 1 + dr, 1 + dc] = -1.0
    w[5, 5, 1, 1] = 9.0
    b[5] = -1.0
    return conv_model(w, b)


def build_task352() -> onnx.ModelProto:
    """dc1df850: red pixels get a blue 8-neighbour halo (clipped at grid edge).

    All dependencies live in channels 0-4, so a group=2 conv halves params.
    Blue rule anchors on in-grid black (in0) to keep padding cells zero-hot.
    """
    w = np.zeros((10, 5, 3, 3), np.float32)
    b = np.zeros(10, np.float32)
    for c in range(10):
        w[c, c % 5, 1, 1] = 1.0              # identity within each group
    for dr, dc in N8:
        w[1, 2, 1 + dr, 1 + dc] += 1.0       # blue on around red
        w[0, 2, 1 + dr, 1 + dc] -= 3.0       # black off at halo cells
    w[1, 0, 1, 1] += 4.0                     # halo only where black (in-grid)
    b[1] = -4.0
    # ORT 1.24.4 group-conv bug: group 2 reuses group 1's FILTER ROWS AND BIAS
    # (out5..9 = w[0..4] applied to in5..9, plus b[0..4]). Coefficient 5 on the
    # x1 slot of both row 1 and row 6 makes every output sign-correct under all
    # three semantics: spec, bias-tiled, and filter+bias-tiled. Signals that
    # keep this sound: input never contains blue(1); pixels are >=2 apart.
    w[1, 1, 1, 1] = 5.0
    w[6, 1, 1, 1] = 5.0
    return conv_model(w, b, group=2)


BUILDERS = {220: build_task220, 230: build_task230,
            294: build_task294, 352: build_task352}

if __name__ == "__main__":
    out_root = pathlib.Path(__file__).resolve().parent.parent.parent / "candidates"
    for n, builder in BUILDERS.items():
        d = out_root / f"task{n:03d}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / "conv_int.onnx"
        onnx.save(builder(), path)
        print("built", path)
