"""把 raw.csv 变成两张东西：

  1. `results/curve_N{nranks}_{transport}.svg` —— 消息大小 vs 时间曲线，
     一眼看出"小消息谁赢、大消息谁赢、交叉点在哪"
  2. 终端打印的 algorithm selection 表 —— 每个消息大小上的最优算法

只依赖标准库，手写 SVG（本机没有 matplotlib，也不该为一张图引入依赖）。
"""

from __future__ import annotations

import csv
import math
import os
import sys

COLORS = {
    "ring": "#d94a4a",
    "tree": "#3b7dd8",
    "recdouble": "#2f9e44",
    "rabenseifner": "#e08b00",
}


def load(path: str):
    with open(path) as f:
        return list(csv.DictReader(f))


def agg_min(rows):
    """(nranks, transport, algo, count) -> 单 rank 最小耗时（ms）。"""
    out = {}
    for r in rows:
        k = (int(r["nranks"]), r["transport"], r["algo"], int(r["count"]))
        t = float(r["elapsed_ms"])
        if k not in out or t < out[k]:
            out[k] = t
    return out


def make_svg(agg, nranks, transport, out_path):
    algos = sorted({k[2] for k in agg if k[0] == nranks and k[1] == transport})
    sizes = sorted({k[3] for k in agg if k[0] == nranks and k[1] == transport})
    if not algos:
        return None

    xs = {s: s * 4 / 1024 for s in sizes}                      # KB
    ts = {k: v for k, v in agg.items() if k[0] == nranks and k[1] == transport}

    lx = [math.log10(xs[s]) for s in sizes if xs[s] > 0]
    ly = [math.log10(t) for t in ts.values() if t > 0]
    x0, x1 = min(lx), max(lx)
    y0, y1 = min(ly), max(ly)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1

    W, H, PAD_L, PAD_R, PAD_T, PAD_B = 760, 440, 78, 150, 46, 52

    def px(v):
        return PAD_L + (math.log10(v) - x0) / (x1 - x0) * (W - PAD_L - PAD_R)

    def py(v):
        return H - PAD_B - (math.log10(v) - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="PingFang SC, Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="24" font-size="15" font-weight="600">'
        f'AllReduce 时间 vs 消息大小（N={nranks}, transport={transport}）</text>',
        f'<text x="{PAD_L}" y="40" font-size="11" fill="#777">'
        f'单 rank 最小耗时（3 次取 min）· 双向对数轴 · ccl-lab</text>',
    ]

    # 网格 + 刻度
    for s in sizes:
        if xs[s] <= 0:
            continue
        x = px(xs[s])
        parts.append(f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{H - PAD_B}" stroke="#eee"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - PAD_B + 16}" font-size="10" fill="#666" '
                     f'text-anchor="middle">{xs[s]:g}KB</text>')
    decades = []
    d = math.floor(y0)
    while d <= math.ceil(y1):
        decades.append(10 ** d)
        d += 1
    for v in decades:
        y = py(v)
        if PAD_T <= y <= H - PAD_B:
            parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="#eee"/>')
            parts.append(f'<text x="{PAD_L - 8}" y="{y + 3:.1f}" font-size="10" fill="#666" '
                         f'text-anchor="end">{v:g}ms</text>')

    parts.append(f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" stroke="#999"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" stroke="#999"/>')
    parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 12}" font-size="11" '
                 f'fill="#444" text-anchor="middle">输入数据量（每个 rank）</text>')
    parts.append(f'<text x="16" y="{(PAD_T + H - PAD_B) / 2:.0f}" font-size="11" fill="#444" '
                 f'text-anchor="middle" transform="rotate(-90 16 {(PAD_T + H - PAD_B) / 2:.0f})">'
                 f'时间</text>')

    # 曲线
    for i, algo in enumerate(algos):
        pts = [(s, ts[(nranks, transport, algo, s)]) for s in sizes
               if (nranks, transport, algo, s) in ts and ts[(nranks, transport, algo, s)] > 0]
        if not pts:
            continue
        color = COLORS.get(algo, "#888")
        d_attr = " ".join(f'{"M" if j == 0 else "L"}{px(xs[s]):.1f},{py(t):.1f}'
                          for j, (s, t) in enumerate(pts))
        parts.append(f'<path d="{d_attr}" fill="none" stroke="{color}" stroke-width="2"/>')
        for s, t in pts:
            parts.append(f'<circle cx="{px(xs[s]):.1f}" cy="{py(t):.1f}" r="2.6" fill="{color}"/>')
        ly_ = PAD_T + 18 + i * 17
        parts.append(f'<rect x="{W - PAD_R + 18}" y="{ly_ - 9}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{W - PAD_R + 34}" y="{ly_}" font-size="11" fill="#333">{algo}</text>')

    parts.append("</svg>")
    svg = "\n".join(parts)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(svg)
    return out_path


def main(rows_paths, out_dir: str) -> int:
    rows = []
    for p in rows_paths:
        rows.extend(load(p))
    agg = agg_min(rows)
    groups = sorted({(k[0], k[1]) for k in agg})

    print("=== Algorithm Selection 表（每个消息大小上的最优算法）===")
    header = f"{'N':>2} {'xport':>6} {'size':>9} " + " ".join(f"{a:>12}" for a in
                                                              ["ring", "tree", "recdouble", "rabenseifner"]) + "   best"
    print(header)
    print("-" * len(header))
    for nranks, transport in groups:
        sizes = sorted({k[3] for k in agg if k[0] == nranks and k[1] == transport})
        for s in sizes:
            vals = {}
            for a in ("ring", "tree", "recdouble", "rabenseifner"):
                v = agg.get((nranks, transport, a, s))
                if v is not None:
                    vals[a] = v
            if not vals:
                continue
            best = min(vals, key=vals.get)
            cells = " ".join(f"{vals.get(a, float('nan')):>12.3f}" for a in
                             ("ring", "tree", "recdouble", "rabenseifner"))
            print(f"{nranks:>2} {transport:>6} {s * 4 / 1024:>7.1f}KB {cells}   {best}")
        print()

    made = []
    for nranks, transport in groups:
        p = os.path.join(out_dir, f"curve_N{nranks}_{transport}.svg")
        if make_svg(agg, nranks, transport, p):
            made.append(p)
    print("=== 图 ===")
    for p in made:
        print(" ", p)
    return 0


if __name__ == "__main__":
    # 用法： python3 -m cclab.plots <csv...> <输出目录>
    args = sys.argv[1:]
    if not args:
        args = ["results/raw.csv", "results"]
    raise SystemExit(main(args[:-1] or ["results/raw.csv"], args[-1]))
