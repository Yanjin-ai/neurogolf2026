"""task064 v2 -- extent method (no CumSum, no bit-packing).

Exploit that the box is a solid rectangle and each ray reaches the FARTHEST dot:
  rightmostDot[r] = max column of a dot on row r  (ReduceMax of mask*col)
  rightFill = (row in box-band) & (col > box.right) & (col <= rightmostDot) & bg
and symmetrically for left / up / down. Replaces 8 int32 CumSums (28800 floor)
with uint8 ReduceMax + broadcast comparisons.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

C, H, W = 10, 30, 30
F = TensorProto.FLOAT
U8 = TensorProto.UINT8


def ci(name, arr, dt=np.int64):
    return numpy_helper.from_array(np.asarray(arr, dtype=dt), name)


def build():
    n, inits = [], []

    # ---------- color roles (bg=most, box=2nd, dot=least frequent) ----------
    inits.append(ci("ax023", [0, 2, 3]))
    n += [helper.make_node("ReduceSum", ["input", "ax023"], ["cnt"], keepdims=0)]
    inits.append(numpy_helper.from_array(np.float32(0), "f0"))
    inits.append(numpy_helper.from_array(np.array([1e9] * 10, np.float32), "big10"))
    inits.append(numpy_helper.from_array(np.array([-1.0] * 10, np.float32), "neg10"))
    inits.append(ci("ar10", list(range(10))))
    n += [
        helper.make_node("ArgMax", ["cnt"], ["bg"], axis=0, keepdims=0),
        helper.make_node("Greater", ["cnt", "f0"], ["pres"]),
        helper.make_node("Where", ["pres", "cnt", "big10"], ["cnt_lo"]),
        helper.make_node("ArgMin", ["cnt_lo"], ["dot"], axis=0, keepdims=0),
        helper.make_node("Equal", ["ar10", "bg"], ["isbg"]),
        helper.make_node("Equal", ["ar10", "dot"], ["isdot"]),
        helper.make_node("Or", ["isbg", "isdot"], ["bd"]),
        helper.make_node("Not", ["bd"], ["nbd"]),
        helper.make_node("And", ["pres", "nbd"], ["belig"]),
        helper.make_node("Where", ["belig", "cnt", "neg10"], ["cnt_bx"]),
        helper.make_node("ArgMax", ["cnt_bx"], ["box"], axis=0, keepdims=0),
    ]

    # ---------- cast input to int8 ONCE (reused for masks + output) ----------
    I8 = TensorProto.INT8
    n += [helper.make_node("Cast", ["input"], ["in8"], to=I8)]  # [1,10,30,30] int8
    inits.append(ci("ax1", [1]))
    for nm, idx in (("mb", "box"), ("md", "dot"), ("mg", "bg")):
        n += [
            helper.make_node("Gather", ["in8", idx], [nm + "g"], axis=1),   # [1,30,30] int8
            helper.make_node("Unsqueeze", [nm + "g", "ax1"], [nm + "i"]),    # [1,1,30,30] int8
            helper.make_node("Cast", [nm + "i"], [nm], to=U8),
        ]

    # ---------- coordinate ramps (uint8) ----------
    inits.append(ci("cc", (np.arange(W) + 1).reshape(1, 1, 1, W), np.uint8))   # col+1: 1..W
    inits.append(ci("rr", (np.arange(H) + 1).reshape(1, 1, H, 1), np.uint8))   # row+1
    inits.append(ci("cc0", np.arange(W).reshape(1, 1, 1, W), np.uint8))        # col: 0..W-1
    inits.append(ci("rr0", np.arange(H).reshape(1, 1, H, 1), np.uint8))        # row
    inits.append(ci("HI", np.uint8(99), np.uint8))

    # ---------- dot extents ----------
    inits.append(ci("axW", [3]))
    inits.append(ci("axH", [2]))
    # rightmost dot col per row: max(md*(col+1)) -> [1,1,30,1]
    n += [helper.make_node("Mul", ["md", "cc"], ["dxc"]),
          helper.make_node("ReduceMax", ["dxc", "axW"], ["rmd"], keepdims=1)]
    # bottommost dot row per col: max(md*(row+1)) -> [1,1,1,30]
    n += [helper.make_node("Mul", ["md", "rr"], ["dxr"]),
          helper.make_node("ReduceMax", ["dxr", "axH"], ["bmd"], keepdims=1)]
    # leftmost dot col per row: min col where dot (absent->99)
    n += [helper.make_node("Where", ["mdb", "cc0", "HI"], ["lc"]) if False else
          helper.make_node("Cast", ["md"], ["mdb"], to=TensorProto.BOOL),
          helper.make_node("Where", ["mdb", "cc0", "HI"], ["lcw"]),
          helper.make_node("ReduceMin", ["lcw", "axW"], ["lmd"], keepdims=1)]
    # topmost dot row per col
    n += [helper.make_node("Where", ["mdb", "rr0", "HI"], ["trw"]),
          helper.make_node("ReduceMin", ["trw", "axH"], ["tmd"], keepdims=1)]

    # ---------- box extents ----------
    n += [helper.make_node("ReduceMax", ["mb", "axH"], ["colBox"], keepdims=1),  # [1,1,1,30]
          helper.make_node("ReduceMax", ["mb", "axW"], ["rowBox"], keepdims=1)]  # [1,1,30,1]
    n += [helper.make_node("Cast", ["colBox"], ["colBoxb"], to=TensorProto.BOOL),
          helper.make_node("Cast", ["rowBox"], ["rowBoxb"], to=TensorProto.BOOL)]
    # c1=max box col (0-based) via max(colBox*(col+1))-1 ; keep C1p=max(col+1)=c1+1
    n += [helper.make_node("Mul", ["colBox", "cc"], ["cbx"]),
          helper.make_node("ReduceMax", ["cbx", "axW"], ["C1p"], keepdims=1),   # c1+1
          helper.make_node("Where", ["colBoxb", "cc0", "HI"], ["cbn"]),
          helper.make_node("ReduceMin", ["cbn", "axW"], ["C0"], keepdims=1),    # c0
          helper.make_node("Mul", ["rowBox", "rr"], ["rbx"]),
          helper.make_node("ReduceMax", ["rbx", "axH"], ["R1p"], keepdims=1),   # r1+1
          helper.make_node("Where", ["rowBoxb", "rr0", "HI"], ["rbn"]),
          helper.make_node("ReduceMin", ["rbn", "axH"], ["R0"], keepdims=1)]    # r0

    # ---------- bands ----------
    n += [helper.make_node("GreaterOrEqual", ["rr0", "R0"], ["ge_r0"]),
          helper.make_node("Less", ["rr0", "R1p"], ["lt_r1"]),
          helper.make_node("And", ["ge_r0", "lt_r1"], ["rowBand"]),           # [1,1,30,1]
          helper.make_node("GreaterOrEqual", ["cc0", "C0"], ["ge_c0"]),
          helper.make_node("Less", ["cc0", "C1p"], ["lt_c1"]),
          helper.make_node("And", ["ge_c0", "lt_c1"], ["colBand"])]           # [1,1,1,30]

    # ---------- fills ----------
    n += [helper.make_node("Cast", ["mg"], ["bgb"], to=TensorProto.BOOL)]
    # right: rowBand & col>=C1p & col+1<=rmd  (col+1 = cc; cc<=rmd)
    n += [helper.make_node("GreaterOrEqual", ["cc0", "C1p"], ["r_gt"]),
          helper.make_node("LessOrEqual", ["cc", "rmd"], ["r_le"]),
          helper.make_node("And", ["rowBand", "r_gt"], ["r_a"]),
          helper.make_node("And", ["r_a", "r_le"], ["right"])]
    # left: rowBand & col<C0 & col>=lmd
    n += [helper.make_node("Less", ["cc0", "C0"], ["l_lt"]),
          helper.make_node("GreaterOrEqual", ["cc0", "lmd"], ["l_ge"]),
          helper.make_node("And", ["rowBand", "l_lt"], ["l_a"]),
          helper.make_node("And", ["l_a", "l_ge"], ["left"])]
    # down: colBand & row>=R1p & row+1<=bmd
    n += [helper.make_node("GreaterOrEqual", ["rr0", "R1p"], ["d_gt"]),
          helper.make_node("LessOrEqual", ["rr", "bmd"], ["d_le"]),
          helper.make_node("And", ["colBand", "d_gt"], ["d_a"]),
          helper.make_node("And", ["d_a", "d_le"], ["down"])]
    # up: colBand & row<R0 & row>=tmd
    n += [helper.make_node("Less", ["rr0", "R0"], ["u_lt"]),
          helper.make_node("GreaterOrEqual", ["rr0", "tmd"], ["u_ge"]),
          helper.make_node("And", ["colBand", "u_lt"], ["u_a"]),
          helper.make_node("And", ["u_a", "u_ge"], ["up"])]
    # combine (broadcast row/col bands to full grid via Or) & gate bg
    n += [helper.make_node("Or", ["right", "left"], ["fh"]),
          helper.make_node("Or", ["up", "down"], ["fv"]),
          helper.make_node("Or", ["fh", "fv"], ["f_any"]),
          helper.make_node("And", ["f_any", "bgb"], ["fillb"]),
          helper.make_node("Cast", ["fillb"], ["fill"], to=I8)]  # [1,1,30,30] int8

    # ---------- output = in8 + fill*(e_dot - e_bg), all int8 ----------
    inits.append(ci("ar10c", np.arange(10).reshape(1, 10, 1, 1)))
    n += [helper.make_node("Equal", ["ar10c", "dot"], ["edb"]),
          helper.make_node("Equal", ["ar10c", "bg"], ["ebb"]),
          helper.make_node("Cast", ["edb"], ["ed"], to=I8),
          helper.make_node("Cast", ["ebb"], ["eb"], to=I8),
          helper.make_node("Sub", ["ed", "eb"], ["delta"]),          # [1,10,1,1] int8
          helper.make_node("Mul", ["fill", "delta"], ["addend"]),    # [1,10,30,30] int8
          helper.make_node("Add", ["in8", "addend"], ["out8"]),
          helper.make_node("Cast", ["out8"], ["output"], to=F)]

    g = helper.make_graph(n, "task064v2",
                          [helper.make_tensor_value_info("input", F, [1, C, H, W])],
                          [helper.make_tensor_value_info("output", F, [1, C, H, W])],
                          inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 10
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "candidates" / "task064"
    onnx.save(build(), out / "v2.onnx")
    print("saved v2.onnx")
