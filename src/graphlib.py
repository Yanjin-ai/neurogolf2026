"""Minimal ONNX graph builder for NeuroGolf circuits.

Design rules baked in (see PLAN.md §0):
- build with onnx.helper directly, ir_version=10, per-model opset choice
- initializers over Constant nodes; attributes over inputs when the opset allows
- value_info is auto-derived via strict shape inference at build() time
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

GRID = [1, 10, 30, 30]
FLOAT = TensorProto.FLOAT

NP2ONNX = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float16): TensorProto.FLOAT16,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.int8): TensorProto.INT8,
    np.dtype(np.bool_): TensorProto.BOOL,
}


class GraphBuilder:
    """Accumulates nodes/initializers; names are auto-assigned (t0, t1, ...)."""

    def __init__(self, opset: int = 10, input_shape=None, output_shape=None):
        self.opset = opset
        self.nodes = []
        self.inits = []
        self._n = 0
        self.input_shape = input_shape or GRID
        self.output_shape = output_shape or GRID

    def _name(self) -> str:
        self._n += 1
        return f"t{self._n}"

    def init(self, array, name=None) -> str:
        """Add an initializer; returns its tensor name. Reuse names to share."""
        arr = np.asarray(array)
        name = name or f"w{len(self.inits)}"
        for existing in self.inits:
            if existing.name == name:
                return name
        self.inits.append(
            helper.make_tensor(name, NP2ONNX[arr.dtype], arr.shape,
                               arr.flatten().tolist())
        )
        return name

    def op(self, op_type: str, inputs, out: str | None = None, **attrs) -> str:
        """Append a node; returns output tensor name ('output' finishes graph)."""
        out = out or self._name()
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    def build(self, check: bool = True) -> onnx.ModelProto:
        x = helper.make_tensor_value_info("input", FLOAT, self.input_shape)
        y = helper.make_tensor_value_info("output", FLOAT, self.output_shape)
        graph = helper.make_graph(self.nodes, "g", [x], [y], self.inits)
        model = helper.make_model(
            graph, ir_version=10,
            opset_imports=[helper.make_opsetid("", self.opset)])
        # strict inference both fills value_info and proves static shapes
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        if check:
            onnx.checker.check_model(model, full_check=True)
        return model


def static_cost(model: onnx.ModelProto) -> tuple[int, int]:
    """(memory, params) from static shapes only — matches the grader unless the
    profiler observes larger runtime shapes (rare for fully static graphs)."""
    import math

    params = 0
    for init in model.graph.initializer:
        params += math.prod(init.dims) if init.dims else 1
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    params += math.prod(attr.t.dims) if attr.t.dims else 1
    memory = 0
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    vi = {t.name: t for t in list(inferred.graph.value_info) + list(inferred.graph.output)}
    for node in inferred.graph.node:
        for out in node.output:
            if out in ("input", "output") or out not in vi:
                continue
            tt = vi[out].type.tensor_type
            n = 1
            for d in tt.shape.dim:
                n *= d.dim_value
            memory += n * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)).itemsize
    return memory, params


# ---- common single-node circuit factories -------------------------------

def identity() -> onnx.ModelProto:
    b = GraphBuilder()
    b.op("Identity", ["input"], out="output")
    return b.build()


def transpose_hw() -> onnx.ModelProto:
    b = GraphBuilder()
    b.op("Transpose", ["input"], out="output", perm=[0, 1, 3, 2])
    return b.build()


def flip(axis: int) -> onnx.ModelProto:
    """Mirror along H (axis=2) or W (axis=3): single Slice with negative step
    (opset 10 Slice takes inputs; 4 scalar int64 initializers => cost 4)."""
    b = GraphBuilder(opset=10)
    starts = b.init(np.array([29], np.int64), "s")
    ends = b.init(np.array([-31], np.int64), "e")
    axes = b.init(np.array([axis], np.int64), "a")
    steps = b.init(np.array([-1], np.int64), "st")
    b.op("Slice", ["input", starts, ends, axes, steps], out="output")
    return b.build()


def color_remap(mapping: list[int]) -> onnx.ModelProto:
    """mapping[new_channel] = old_channel; single Gather on axis=1 (cost 10)."""
    b = GraphBuilder(opset=13)
    idx = b.init(np.array(mapping, np.int32), "m")
    b.op("Gather", ["input", idx], out="output", axis=1)
    return b.build()
