"""task064 v3 -- full cost-golf stack (COST_GOLF_PLAYBOOK B1-B6).

- Slice 30x30 -> 24x24 (actual max grid), Pad back at the end.
- masks via Einsum('bchw,ec->behw') -- channel-select fused, no Gather chains.
- extents in 1D via ReduceMax/Min (uint8, no CumSum).
- output as a single-channel colour-INDEX grid M -> OneHot -> free 10ch output.
  padding cells get index 99 (OneHot >= depth -> all-zero).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

C, H, W, S = 10, 30, 30, 30
F, U8, I8, I64, B = (TensorProto.FLOAT, TensorProto.UINT8, TensorProto.INT8,
                     TensorProto.INT64, TensorProto.BOOL)


def cf(name, arr, dt=np.float32):
    return numpy_helper.from_array(np.asarray(arr, dtype=dt), name)


def build():
    n, I = [], []

    # ---- colour roles: bg=most, dot=least present, box=remaining ----
    I += [cf("ax023", [0, 2, 3], np.int64)]
    n += [helper.make_node("ReduceSum", ["input", "ax023"], ["cnt"], keepdims=0)]  # [10]
    I += [cf("f0", 0.0), cf("big10", [1e9] * 10), cf("neg10", [-1.0] * 10),
          cf("ar10", list(range(10)), np.int64)]
    n += [
        helper.make_node("ArgMax", ["cnt"], ["bg"], axis=0, keepdims=0),
        helper.make_node("Greater", ["cnt", "f0"], ["pres"]),
        helper.make_node("Where", ["pres", "cnt", "big10"], ["clo"]),
        helper.make_node("ArgMin", ["clo"], ["dot"], axis=0, keepdims=0),
        helper.make_node("Equal", ["ar10", "bg"], ["isbg"]),
        helper.make_node("Equal", ["ar10", "dot"], ["isdot"]),
        helper.make_node("Or", ["isbg", "isdot"], ["bd"]),
        helper.make_node("Not", ["bd"], ["nbd"]),
        helper.make_node("And", ["pres", "nbd"], ["belig"]),
        helper.make_node("Where", ["belig", "cnt", "neg10"], ["cbx"]),
        helper.make_node("ArgMax", ["cbx"], ["box"], axis=0, keepdims=0),
    ]

    # ---- one-hot selectors [1,10] for einsum channel-select ----
    I += [cf("ar10c", np.arange(10).reshape(1, 10), np.float32)]
    for r in ("box", "dot"):
        n += [helper.make_node("Cast", [r], [r + "f"], to=F),
              helper.make_node("Equal", ["ar10c", r + "f"], [r + "_b"]),
              helper.make_node("Cast", [r + "_b"], [r + "_s"], to=F)]  # [1,10] 0/1

    # ---- masks via Einsum('bchw,ec->behw') -> [1,1,24,24], to uint8 ----
    for m, sel in (("md", "dot_s"), ("mb", "box_s")):
        n += [helper.make_node("Einsum", ["input", sel], [m + "f"], equation="bchw,ec->behw"),
              helper.make_node("Cast", [m + "f"], [m], to=U8)]

    # ---- ramps ----
    I += [cf("cc", (np.arange(S) + 1).reshape(1, 1, 1, S), np.uint8),
          cf("rr", (np.arange(S) + 1).reshape(1, 1, S, 1), np.uint8),
          cf("cc0", np.arange(S).reshape(1, 1, 1, S), np.uint8),
          cf("rr0", np.arange(S).reshape(1, 1, S, 1), np.uint8),
          cf("HI", 99, np.uint8), cf("axW", [3], np.int64), cf("axH", [2], np.int64)]
    n += [helper.make_node("Cast", ["md"], ["mdb"], to=B)]

    # ---- dot extents (per row / per col) ----
    n += [helper.make_node("Mul", ["md", "cc"], ["dxc"]),
          helper.make_node("ReduceMax", ["dxc", "axW"], ["rmd"], keepdims=1),   # [1,1,24,1]
          helper.make_node("Mul", ["md", "rr"], ["dxr"]),
          helper.make_node("ReduceMax", ["dxr", "axH"], ["bmd"], keepdims=1),   # [1,1,1,24]
          helper.make_node("Where", ["mdb", "cc0", "HI"], ["lcw"]),
          helper.make_node("ReduceMin", ["lcw", "axW"], ["lmd"], keepdims=1),
          helper.make_node("Where", ["mdb", "rr0", "HI"], ["trw"]),
          helper.make_node("ReduceMin", ["trw", "axH"], ["tmd"], keepdims=1)]

    # ---- box extents ----
    n += [helper.make_node("ReduceMax", ["mb", "axH"], ["colBox"], keepdims=1),
          helper.make_node("ReduceMax", ["mb", "axW"], ["rowBox"], keepdims=1),
          helper.make_node("Cast", ["colBox"], ["colBoxb"], to=B),
          helper.make_node("Cast", ["rowBox"], ["rowBoxb"], to=B),
          helper.make_node("Mul", ["colBox", "cc"], ["cbxp"]),
          helper.make_node("ReduceMax", ["cbxp", "axW"], ["C1p"], keepdims=1),
          helper.make_node("Where", ["colBoxb", "cc0", "HI"], ["cbn"]),
          helper.make_node("ReduceMin", ["cbn", "axW"], ["C0"], keepdims=1),
          helper.make_node("Mul", ["rowBox", "rr"], ["rbx"]),
          helper.make_node("ReduceMax", ["rbx", "axH"], ["R1p"], keepdims=1),
          helper.make_node("Where", ["rowBoxb", "rr0", "HI"], ["rbn"]),
          helper.make_node("ReduceMin", ["rbn", "axH"], ["R0"], keepdims=1)]

    # ---- bands (1D) ----
    n += [helper.make_node("GreaterOrEqual", ["rr0", "R0"], ["ge_r0"]),
          helper.make_node("Less", ["rr0", "R1p"], ["lt_r1"]),
          helper.make_node("And", ["ge_r0", "lt_r1"], ["rowBand"]),
          helper.make_node("GreaterOrEqual", ["cc0", "C0"], ["ge_c0"]),
          helper.make_node("Less", ["cc0", "C1p"], ["lt_c1"]),
          helper.make_node("And", ["ge_c0", "lt_c1"], ["colBand"])]

    # ---- fills (2D only here) ----
    n += [# right: col>=C1p & col+1<=rmd
          helper.make_node("GreaterOrEqual", ["cc0", "C1p"], ["r_gt"]),
          helper.make_node("LessOrEqual", ["cc", "rmd"], ["r_le"]),
          helper.make_node("And", ["r_gt", "r_le"], ["rM"]),
          # left: col<C0 & col>=lmd
          helper.make_node("Less", ["cc0", "C0"], ["l_lt"]),
          helper.make_node("GreaterOrEqual", ["cc0", "lmd"], ["l_ge"]),
          helper.make_node("And", ["l_lt", "l_ge"], ["lM"]),
          helper.make_node("Or", ["rM", "lM"], ["hM0"]),
          helper.make_node("And", ["hM0", "rowBand"], ["hM"]),
          # down: row>=R1p & row+1<=bmd
          helper.make_node("GreaterOrEqual", ["rr0", "R1p"], ["d_gt"]),
          helper.make_node("LessOrEqual", ["rr", "bmd"], ["d_le"]),
          helper.make_node("And", ["d_gt", "d_le"], ["dM"]),
          # up: row<R0 & row>=tmd
          helper.make_node("Less", ["rr0", "R0"], ["u_lt"]),
          helper.make_node("GreaterOrEqual", ["rr0", "tmd"], ["u_ge"]),
          helper.make_node("And", ["u_lt", "u_ge"], ["uM"]),
          helper.make_node("Or", ["dM", "uM"], ["vM0"]),
          helper.make_node("And", ["vM0", "colBand"], ["vM"]),
          helper.make_node("Or", ["hM", "vM"], ["fill"])]  # [1,1,30,30] bool

    # ---- output: colour-index grid M -> OneHot -> free 10ch, then Pad ----
    # orig colour index = sum_c c * s   (s is one-hot) -> [1,1,24,24] float
    I += [cf("arC", np.arange(10).reshape(1, 10), np.float32)]
    I += [cf("ax1c", [1], np.int64)]
    n += [helper.make_node("Einsum", ["input", "arC"], ["orig"], equation="bchw,ec->behw")]
    # valid = any channel present
    n += [helper.make_node("ReduceMax", ["input", "ax1c"], ["vmax"], keepdims=1),
          helper.make_node("Cast", ["vmax"], ["valid"], to=B)]
    # M = fill ? dot : orig ; then if not valid -> 99
    I += [cf("hi_u", 99, np.uint8)]
    n += [helper.make_node("Cast", ["orig"], ["origu"], to=U8),
          helper.make_node("Cast", ["dot"], ["dotiv"], to=U8),
          helper.make_node("Where", ["fill", "dotiv", "origu"], ["M1"]),
          helper.make_node("Where", ["valid", "M1", "hi_u"], ["M2"]),
          helper.make_node("Squeeze", ["M2", "ax1c"], ["Msq"]),
          helper.make_node("Cast", ["Msq"], ["Mi"], to=I64)]  # [1,30,30] int64
    I += [cf("depth", 10, np.int64), cf("ohv", [0, 1], np.float32)]
    n += [helper.make_node("OneHot", ["Mi", "depth", "ohv"], ["output"], axis=1)]  # free 10ch output

    g = helper.make_graph(n, "task064v3",
                          [helper.make_tensor_value_info("input", F, [1, C, H, W])],
                          [helper.make_tensor_value_info("output", F, [1, C, H, W])], I)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 10
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "candidates" / "task064"
    onnx.save(build(), out / "v4.onnx")
    print("saved v4.onnx")
