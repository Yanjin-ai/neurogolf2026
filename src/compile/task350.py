"""task350 (dbc1a6ce): connect collinear blue pixels with cyan.

TRUE rule (verified 0 mismatches vs generator on 2000 fresh):
  cyan at (r,c) iff not-blue AND ( (blue left-inclusive AND blue right-inclusive
  in row r) OR (blue up AND blue down in col c) ).

Circuit keeps everything in fp16 grid tensors. "blue to the right" is derived
as (row_total - prefix) > 0 to avoid a second cumsum per axis.
All constants are initializers; input & output tensors are free.
"""
import pathlib
import numpy as np
import onnx
from onnx import helper, TensorProto as TP

F16 = TP.FLOAT16


def build():
    g = helper.make_graph
    nodes, inits = [], []

    def C(name, arr, dt=F16):
        inits.append(helper.make_tensor(name, dt, list(np.asarray(arr).shape),
                                        np.asarray(arr).flatten().tolist()))
        return name

    # blue channel (index 1) as fp16 [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["input"], ["xf"], to=F16))
    # slice channel 1
    nodes.append(helper.make_node("Slice", ["xf", C("s1", [1], TP.INT64),
                 C("e1", [2], TP.INT64), C("a1", [1], TP.INT64)], ["blue"]))

    # ---- row axis (W=3) ----
    nodes.append(helper.make_node("CumSum", ["blue", C("axW", 3, TP.INT64)], ["csL"]))
    nodes.append(helper.make_node("ReduceSum", ["blue", C("rax", [3], TP.INT64)],
                 ["rowtot"], keepdims=1))
    # left_inc = csL>0 ; right_inc = (rowtot - csL + blue)>0
    nodes.append(helper.make_node("Sub", ["rowtot", "csL"], ["tmpR"]))
    nodes.append(helper.make_node("Add", ["tmpR", "blue"], ["csR"]))
    nodes.append(helper.make_node("Greater", ["csL", C("z", 0.0)], ["Lb"]))
    nodes.append(helper.make_node("Greater", ["csR", "z"], ["Rb"]))
    nodes.append(helper.make_node("And", ["Lb", "Rb"], ["hrow"]))

    # ---- col axis (H=2) ----
    nodes.append(helper.make_node("CumSum", ["blue", C("axH", 2, TP.INT64)], ["csU"]))
    nodes.append(helper.make_node("ReduceSum", ["blue", C("cax", [2], TP.INT64)],
                 ["coltot"], keepdims=1))
    nodes.append(helper.make_node("Sub", ["coltot", "csU"], ["tmpD"]))
    nodes.append(helper.make_node("Add", ["tmpD", "blue"], ["csD"]))
    nodes.append(helper.make_node("Greater", ["csU", "z"], ["Ub"]))
    nodes.append(helper.make_node("Greater", ["csD", "z"], ["Db"]))
    nodes.append(helper.make_node("And", ["Ub", "Db"], ["hcol"]))

    # cyan = (hrow OR hcol) AND not blue
    nodes.append(helper.make_node("Or", ["hrow", "hcol"], ["hany"]))
    nodes.append(helper.make_node("Cast", ["blue"], ["blueb"], to=TP.BOOL))
    nodes.append(helper.make_node("Not", ["blueb"], ["nblue"]))
    nodes.append(helper.make_node("And", ["hany", "nblue"], ["cyanb"]))
    nodes.append(helper.make_node("Cast", ["cyanb"], ["cyan"], to=F16))  # [1,1,30,30]

    # delta over channels: +cyan at ch8, -cyan at ch0. kernel [1,10,1,1]
    kern = np.zeros((1, 10, 1, 1), np.float16); kern[0, 8] = 1.0; kern[0, 0] = -1.0
    nodes.append(helper.make_node("Mul", ["cyan", C("kern", kern)], ["delta"]))
    nodes.append(helper.make_node("Add", ["xf", "delta"], ["outf"]))
    nodes.append(helper.make_node("Cast", ["outf"], ["output"], to=TP.FLOAT))

    x = helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])
    graph = g(nodes, "g", [x], [y], inits)
    m = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent.parent.parent / "candidates" / "task350"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "cumsum.onnx"
    onnx.save(build(), p)
    print("saved", p)
