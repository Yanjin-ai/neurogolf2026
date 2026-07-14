"""task243 (9edfc990): flood-fill blue(1) through black(0), 4-connected.

Architecture: bit-packed rows (uint32 scalar per row) + Gauss-Seidel serpentine
sweeps. Each row visit ORs in the neighbor row's bits, then does a masked
doubling row-fill with shifts {1,2} in both directions (reach 3/visit/dir).
10 half-sweeps (gate worst=6, 100k-sample worst=8, +2 margin).

memory ~14.1KB + ~429 params -> ~15.4 pts (baseline 15.21 @ 17871).
"""
import sys
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = pathlib.Path("/Volumes/SANDISK ELE/ARCAGIneurogolf")
sys.path.insert(0, str(ROOT / "src"))
import graphlib  # noqa: E402
graphlib.NP2ONNX[np.dtype(np.uint32)] = TensorProto.UINT32
from graphlib import GraphBuilder  # noqa: E402

N_ROWS = 18       # max grid size
N_HALVES = 10     # serpentine half-sweeps (down, up, down, ...)


def build():
    b = GraphBuilder(opset=18)

    # ---- pack: rows -> uint32 bitmasks -------------------------------
    # conv kernel [2,10,1,18] stride (1,18): ch0 = open (black|blue), ch1 = seed
    w = np.zeros((2, 10, 1, 18), np.float32)
    for j in range(18):
        w[0, 0, 0, j] = 2.0 ** j   # open: black
        w[0, 1, 0, j] = 2.0 ** j   # open: blue
        w[1, 1, 0, j] = 2.0 ** j   # seed: blue
    convw = b.init(w, "convw")
    cv0 = b.op("Conv", ["input", convw], kernel_shape=[1, 18],
               strides=[1, 18], pads=[0, 0, 0, 0])          # [1,2,30,1] 240B

    st = b.init(np.array([0], np.int64), "sl_s")
    en = b.init(np.array([N_ROWS], np.int64), "sl_e")
    ax = b.init(np.array([2], np.int64), "sl_a")
    cv1 = b.op("Slice", [cv0, st, en, ax])                   # [1,2,18,1] 144B
    cvu = b.op("Cast", [cv1], to=TensorProto.UINT32)         # 72B
    shp3 = b.init(np.array([1, 2, N_ROWS], np.int64), "shp3")
    cvf = b.op("Reshape", [cvu, shp3])                       # [1,2,18] 72B

    idx = [b.init(np.array(i, np.int32), f"i{i}") for i in range(N_ROWS)]
    mfull = b.op("Gather", [cvf, idx[0]], axis=1)            # [1,18] open  72B
    sfull = b.op("Gather", [cvf, idx[1]], axis=1)            # [1,18] seeds 72B

    m = [b.op("Gather", [mfull, idx[i]], axis=1) for i in range(N_ROWS)]  # [1]
    s = [b.op("Gather", [sfull, idx[i]], axis=1) for i in range(N_ROWS)]  # [1]

    one = b.init(np.array(1, np.uint32), "one")
    two = b.init(np.array(2, np.uint32), "two")

    # ---- precompute run masks: c2 = m & (m<<1), d2 = c2>>1 -----------
    c2, d2 = [], []
    for i in range(N_ROWS):
        t = b.op("BitShift", [m[i], one], direction="LEFT")
        c = b.op("BitwiseAnd", [m[i], t])
        c2.append(c)
        d2.append(b.op("BitShift", [c, one], direction="RIGHT"))

    def rowfill(x, i):
        """masked doubling fill, shifts {1,2}, both directions."""
        f = b.op("BitwiseAnd", [x, m[i]])
        for k, up_mask, dn_mask in ((one, m[i], m[i]), (two, c2[i], d2[i])):
            t = b.op("BitShift", [f, k], direction="LEFT")
            a = b.op("BitwiseAnd", [t, up_mask])
            f = b.op("BitwiseOr", [f, a])
        for k, up_mask, dn_mask in ((one, m[i], m[i]), (two, c2[i], d2[i])):
            t = b.op("BitShift", [f, k], direction="RIGHT")
            a = b.op("BitwiseAnd", [t, dn_mask])
            f = b.op("BitwiseOr", [f, a])
        return f

    # ---- serpentine sweeps -------------------------------------------
    r = list(s)
    for half in range(N_HALVES):
        order = range(N_ROWS) if half % 2 == 0 else range(N_ROWS - 1, -1, -1)
        first = True
        for i in order:
            nb = i - 1 if half % 2 == 0 else i + 1
            x = r[i] if first else b.op("BitwiseOr", [r[i], r[nb]])
            r[i] = rowfill(x, i)
            first = False

    # ---- unpack newly-blue + paint -----------------------------------
    rall = b.op("Concat", r, axis=0)                          # [18] uint32 72B
    shpr = b.init(np.array([1, N_ROWS], np.int64), "shpr")
    rall2 = b.op("Reshape", [rall, shpr])                     # [1,18] 72B
    nb = b.op("BitwiseXor", [rall2, sfull])                   # [1,18] newly blue
    shpc = b.init(np.array([1, N_ROWS, 1], np.int64), "shpc")
    nb2 = b.op("Reshape", [nb, shpc])                         # [1,18,1] 72B
    pow2 = b.init((2 ** np.arange(18, dtype=np.uint32)).reshape(1, 1, 18), "pow2")
    ub = b.op("BitwiseAnd", [nb2, pow2])                      # [1,18,18] 1296B
    cnd = b.op("Cast", [ub], to=TensorProto.BOOL)             # 324B
    shp4 = b.init(np.array([1, 1, N_ROWS, N_ROWS], np.int64), "shp4")
    cnd4 = b.op("Reshape", [cnd, shp4])                       # [1,1,18,18] 324B
    pads = b.init(np.array([0, 0, 0, 0, 0, 0, 12, 12], np.int64), "pads")
    cnd30 = b.op("Pad", [cnd4, pads])                         # [1,1,30,30] 900B
    paint = b.init(np.array([-1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                            np.float32).reshape(1, 10, 1, 1), "paint")
    b.op("Where", [cnd30, paint, "input"], out="output")
    return b.build()


if __name__ == "__main__":
    model = build()
    from graphlib import static_cost
    mem, par = static_cost(model)
    print(f"memory={mem} params={par} cost={mem + par}")
    out = ROOT / "candidates" / "task243" / "bitflood.onnx"
    out.parent.mkdir(exist_ok=True)
    onnx.save(model, str(out))
    print("saved", out)
