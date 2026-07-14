"""Hand-built circuit for task064 (2c608aff): box + dots, each orthogonally-
aligned dot shoots a same-color ray to the box.

First-principles design (no copying the incumbent):
- identify color roles by frequency (bg=most, box=2nd, dot=least) -- all dynamic
- a background cell fills with dotcolor iff, on its row, box is on one side AND a
  dot is on the other (prefix/suffix presence) -- likewise on its column
- prefix/suffix presence = CumSum along the axis (one tensor per direction)

Correctness first; cost measured empirically afterwards.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

C, H, W = 10, 30, 30
DT = TensorProto.FLOAT


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


def consti(name, arr, dt=np.int64):
    return numpy_helper.from_array(np.asarray(arr, dtype=dt), name)


def build():
    n = []
    inits = []

    # ---- color counts over spatial dims -> [1,10] ----
    inits.append(consti("ax023", [0, 2, 3]))
    n.append(helper.make_node("ReduceSum", ["input", "ax023"], ["cnt"], keepdims=0))  # [10]

    # bg = argmax count
    n.append(helper.make_node("ArgMax", ["cnt"], ["bg_idx"], axis=0, keepdims=0))
    # present = cnt>0
    inits.append(const("zero", 0.0))
    n.append(helper.make_node("Greater", ["cnt", "zero"], ["present"]))
    # dot = argmin over present (mask absent with +inf)
    inits.append(const("big", 1e9))
    n.append(helper.make_node("Where", ["present", "cnt", "big_b"], ["cnt_min"]))
    inits.append(const("big_b", [1e9] * 10))
    n.append(helper.make_node("ArgMin", ["cnt_min"], ["dot_idx"], axis=0, keepdims=0))

    # box = present, not bg, not dot, argmax
    inits.append(consti("arange10", list(range(10))))
    n.append(helper.make_node("Equal", ["arange10", "bg_idx"], ["is_bg"]))
    n.append(helper.make_node("Equal", ["arange10", "dot_idx"], ["is_dot"]))
    n.append(helper.make_node("Or", ["is_bg", "is_dot"], ["is_bgdot"]))
    n.append(helper.make_node("Not", ["is_bgdot"], ["not_bgdot"]))
    n.append(helper.make_node("And", ["present", "not_bgdot"], ["box_elig"]))
    inits.append(const("negone_b", [-1.0] * 10))
    n.append(helper.make_node("Where", ["box_elig", "cnt", "negone_b"], ["cnt_box"]))
    n.append(helper.make_node("ArgMax", ["cnt_box"], ["box_idx"], axis=0, keepdims=0))

    # ---- one-hot channel selectors [1,10,1,1] ----
    inits.append(consti("arange10_c", np.arange(10).reshape(1, 10, 1, 1)))
    for role in ("bg", "box", "dot"):
        n.append(helper.make_node("Equal", ["arange10_c", f"{role}_idx"], [f"e_{role}_b"]))
        n.append(helper.make_node("Cast", [f"e_{role}_b"], [f"e_{role}"], to=DT))

    # ---- masks [1,1,30,30] via Gather of the dynamic channel (no [1,10,30,30]) ----
    for role in ("bg", "box", "dot"):
        # Gather along channel axis -> [1,30,30]; unsqueeze -> [1,1,30,30]
        n.append(helper.make_node("Gather", ["input", f"{role}_idx"], [f"g_{role}"], axis=1))
        n.append(helper.make_node("Unsqueeze", [f"g_{role}", "ax1"], [f"m_{role}"]))
    inits.append(consti("ax1", [1]))

    # ---- prefix/suffix presence via CumSum, thresholded to 0/1 ----
    inits.append(consti("axW", 3))
    inits.append(consti("axH", 2))

    def presence(mask, axis_const, reverse, out):
        cs = out + "_cs"
        n.append(helper.make_node("CumSum", [mask, axis_const], [cs], reverse=reverse))
        n.append(helper.make_node("Greater", [cs, "zero"], [out + "_b"]))
        n.append(helper.make_node("Cast", [out + "_b"], [out], to=DT))

    presence("m_box", "axW", 0, "boxPreW")
    presence("m_box", "axW", 1, "boxSufW")
    presence("m_dot", "axW", 0, "dotPreW")
    presence("m_dot", "axW", 1, "dotSufW")
    presence("m_box", "axH", 0, "boxPreH")
    presence("m_box", "axH", 1, "boxSufH")
    presence("m_dot", "axH", 0, "dotPreH")
    presence("m_dot", "axH", 1, "dotSufH")

    # ---- fills (AND = Mul), gated by bg ----
    def fill(a, b, out):
        n.append(helper.make_node("Mul", [a, b], [out + "_ab"]))
        n.append(helper.make_node("Mul", [out + "_ab", "m_bg"], [out]))

    fill("boxPreW", "dotSufW", "right")
    fill("boxSufW", "dotPreW", "left")
    fill("boxPreH", "dotSufH", "down")
    fill("boxSufH", "dotPreH", "up")
    n.append(helper.make_node("Max", ["right", "left"], ["fh"]))
    n.append(helper.make_node("Max", ["up", "down"], ["fv"]))
    n.append(helper.make_node("Max", ["fh", "fv"], ["fill"]))  # [1,1,30,30]

    # ---- output = input + fill*(e_dot - e_bg) ----
    n.append(helper.make_node("Sub", ["e_dot", "e_bg"], ["delta"]))
    n.append(helper.make_node("Mul", ["fill", "delta"], ["addend"]))
    n.append(helper.make_node("Add", ["input", "addend"], ["output"]))

    g = helper.make_graph(
        n, "task064",
        [helper.make_tensor_value_info("input", DT, [1, C, H, W])],
        [helper.make_tensor_value_info("output", DT, [1, C, H, W])],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 10
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "candidates" / "task064"
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(build(), out / "handbuilt.onnx")
    print("saved", out / "handbuilt.onnx")
