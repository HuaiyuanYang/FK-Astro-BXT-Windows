#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trace_params.py —— 从 ONNX 里沿 `parameters` 输入往下追, 把参数通路和权重全部挖出来。

目的:
    我们已经知道"参数动起来会把图毁掉", 而不是"锐化不够"。这不像标定问题,
    像接线问题。所以要看:
      1. `parameters` 在 ONNX 里被哪些节点消费
      2. 每个消费点的权重张量长什么样(形状/数值量级)
      3. 参数取不同值时, 经过这些权重算出来的调制量变化多大

    如果调制量的变化幅度离谱(比如差 100 倍), 就能反推出正确的缩放。

用法:
    python trace_params.py bxt5.onnx [--depth 4]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERT = os.path.dirname(HERE)
for p in (HERE, os.path.join(CONVERT, "vendor")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from onnx_verify import parse_fields, get, one, parse_tensor, parse_node  # noqa: E402


def load_graph(path):
    """-> (nodes, inits{name:(dims, arr)})"""
    blob = open(path, "rb").read()
    print(f"读取 {os.path.basename(path)}  {len(blob)/1048576:.1f} MB")
    root = parse_fields(blob)
    graph = one(root, 7, 2)                 # ModelProto.graph
    if graph is None:
        raise SystemExit("找不到 graph(字段 7)")
    g = parse_fields(graph)
    nodes = [parse_node(b) for b in get(g, 1, 2)]
    inits = {}
    for b in get(g, 5, 2):
        t = parse_tensor(b)
        name, dtname, dims, arr, has_raw = t
        inits[name] = (dims, arr)
    print(f"  节点 {len(nodes)} 个, 初始化张量 {len(inits)} 个")
    return nodes, inits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()

    nodes, inits = load_graph(a.onnx)

    prod = {}
    for n in nodes:
        for o in n["out"]:
            prod[o] = n

    # ---- 找出直接消费 parameters 的节点 ----
    print("\n" + "=" * 84)
    print("直接消费 `parameters` 的节点")
    print("=" * 84)
    roots = []
    for n in nodes:
        if "parameters" in n["in"]:
            roots.append(n)
            ws = []
            for i in n["in"]:
                if i in inits:
                    dims, arr = inits[i]
                    ws.append(f"{i} dims={dims}" +
                              (f" 量级 min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g} "
                               f"absmean={np.nanmean(np.abs(arr)):.4g}" if arr is not None else ""))
            print(f"  {n['op']:<10} name={n['name'][:48]}")
            print(f"     in  = {n['in']}")
            print(f"     out = {n['out']}")
            for w in ws:
                print(f"     权重 {w}")
    if not roots:
        print("  ** 没有任何节点直接消费 parameters **")

    # ---- BFS 往下追 depth 层 ----
    print("\n" + "=" * 84)
    print(f"从 parameters 往下追 {a.depth} 层")
    print("=" * 84)
    frontier = [o for n in roots for o in n["out"]]
    seen = set(roots and [id(x) for x in roots])
    for d in range(a.depth):
        nxt = []
        print(f"\n--- 第 {d+1} 层 ({len(frontier)} 个张量) ---")
        shown = 0
        for t in frontier:
            for n in nodes:
                if t in n["in"]:
                    key = (n["op"], n["name"])
                    if key in seen:
                        continue
                    seen.add(key)
                    print(f"  {n['op']:<12} {n['name'][:56]}")
                    print(f"     in={[x[:26] for x in n['in']]}  out={[x[:26] for x in n['out']]}")
                    for x in n["in"]:
                        if x in inits:
                            dims, arr = inits[x]
                            s = f"      权重 {x} dims={dims}"
                            if arr is not None and arr.size:
                                s += (f"  量级 [{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]"
                                      f"  absmean {np.nanmean(np.abs(arr)):.4g}")
                            print(s)
                    shown += 1
                    if shown > 30:
                        break
                    nxt.extend(n["out"])
            if shown > 30:
                print("  ...(截断)")
                break
        frontier = nxt
        if not frontier:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
