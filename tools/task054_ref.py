"""task054 (264363fd) reference implementation in numpy.

Rule (from generator):
- 2 solid boxes (boxcolor) + 1-2 single-pixel star-markers (c0) inside each box
- one reference star OUTSIDE boxes: 3x3 fill c2 (may be absent) + center c0 +
  arm tips c1 at +-1,+-2 (vert and/or horiz)
- OUT: erase everything outside boxes (-> bg); for each marker draw a full line
  through ITS box (c1) along the arm direction(s); stamp small star at marker
  (3x3 c2 + center c0 + 1-cell arms c1).  flip/xpose variants exist.

Priority: c0 center > c1 arms > c2 fill > lines(c1) > box > bg.
"""
import numpy as np


def solve(a):
    H, W = a.shape
    vals, cnts = np.unique(a, return_counts=True)
    order = vals[np.argsort(-cnts)]
    top2 = order[:2]
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    b0 = np.sum(border == top2[0])
    b1 = np.sum(border == top2[1])
    bg, box = (top2[0], top2[1]) if b0 >= b1 else (top2[1], top2[0])
    boxm = a == box

    # markers: cells of a non-bg/box colour whose 3x3 neighbourhood is mostly box
    c0 = None
    markers = []
    for c in vals:
        if c in (bg, box):
            continue
        for y, x in zip(*np.where(a == c)):
            nb = a[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            if np.sum(nb == box) >= 5:
                markers.append((y, x))
                c0 = c
    # reference star centre: c0 cell not a marker, with same-colour orthogonal
    # neighbours (arms) -- pick the c0-cell outside boxes whose 4-orth
    # neighbourhood has a repeated non-bg colour
    # reference star centre: c0 cell (not a marker) with a TRUE arm direction.
    # true arm: cells at +-1 AND +-2 along that axis are all the same colour
    # (!= bg) -- the 3x3 fill only reaches +-1, so it can't fake this.
    ref_cells = [(y, x) for y, x in zip(*np.where(a == c0)) if (y, x) not in markers]

    def arm_colour(y, x, dy, dx):
        cells = []
        for k in (1, 2):
            yy, xx = y + dy * k, x + dx * k
            if not (0 <= yy < H and 0 <= xx < W):
                return None
            cells.append(int(a[yy, xx]))
        yy, xx = y - dy, x - dx
        y2, x2 = y - 2 * dy, x - 2 * dx
        if not (0 <= yy < H and 0 <= xx < W and 0 <= y2 < H and 0 <= x2 < W):
            return None
        cells += [int(a[yy, xx]), int(a[y2, x2])]
        if len(set(cells)) == 1 and cells[0] != int(bg):
            return cells[0]
        return None

    ctr, c1, vert, horiz = None, None, False, False
    for y, x in ref_cells:
        cv = arm_colour(y, x, 1, 0)
        ch = arm_colour(y, x, 0, 1)
        if cv or ch:
            ctr, c1 = (y, x), cv or ch
            vert, horiz = cv is not None, ch is not None
            break
    y0, x0 = ctr
    # c2: corner of the 3x3 (absent -> stays bg)
    cc = int(a[y0 - 1, x0 - 1])
    c2 = cc if cc not in (int(bg),) else -1

    out = np.where(boxm, box, bg)

    mk = np.zeros_like(boxm)
    for y, x in markers:
        mk[y, x] = True

    # lines: same-run test via cumsum of non-box along each axis
    # (markers sit INSIDE boxes but aren't box-coloured -- count them as box,
    #  else the marker pixel splits its own run)
    solid = boxm | mk
    runV = np.cumsum(~solid, axis=0)  # column runs
    runH = np.cumsum(~solid, axis=1)  # row runs
    lineV = np.zeros_like(boxm)
    lineH = np.zeros_like(boxm)
    for y, x in markers:
        if vert:
            lineV[:, x] |= boxm[:, x] & (runV[:, x] == runV[y, x])
        if horiz:
            lineH[y, :] |= boxm[y, :] & (runH[y, :] == runH[y, x])
    line = lineV | lineH

    # stamp masks
    def dil(m, ky, kx):
        r = np.zeros_like(m)
        for dy in range(-(ky // 2), ky // 2 + 1):
            for dx in range(-(kx // 2), kx // 2 + 1):
                r |= np.roll(np.roll(m, dy, 0), dx, 1)
        return r
    m33 = dil(mk, 3, 3)
    marm = (dil(mk, 3, 1) if vert else np.zeros_like(mk)) | \
           (dil(mk, 1, 3) if horiz else np.zeros_like(mk))

    out = np.where(line, c1, out)
    if c2 != -1:
        out = np.where(m33, c2, out)
    out = np.where(marm & ~mk, c1, out)
    out = np.where(mk, c0, out)
    return out


if __name__ == "__main__":
    import json
    import pathlib
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    d = json.load(open(ROOT / "data" / "task054.json"))
    right = wrong = 0
    fails = []
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(d.get(split, [])):
            a = np.array(ex["input"])
            want = np.array(ex["output"])
            try:
                got = solve(a)
                if np.array_equal(got, want):
                    right += 1
                else:
                    wrong += 1
                    fails.append((split, i, "mismatch"))
            except Exception as e:  # noqa: BLE001
                wrong += 1
                fails.append((split, i, str(e)[:40]))
    print(f"right={right} wrong={wrong}")
    for f in fails[:8]:
        print("  FAIL", f)
