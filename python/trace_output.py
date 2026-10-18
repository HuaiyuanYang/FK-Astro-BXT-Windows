#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trace_output.py —— 从 ONNX 的 output 节点往回追, 打印末端几层和其中的常量。

动机:
    测出我们的模型在星点处的增量恒为 +0.14 左右, 而与输入电平无关;
    官方是 +0.014~0.038 且随电平收缩。放大 4~9 倍, 且是"常数式"。
    正常的残差网络: output = input_center + delta, delta 应随结构变化。
    嫌疑是末端某处缩放/残差相加的常量不对。

用法:
    python trace_output.py bxt5.onnx [--depth 8]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from onnx_verify import parse_fields, get, one, parse_tensor, parse_node  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--from", dest="seed", default=None, help="从指定张量名开始追")
    ap.add_argument("--branch", type=int, default=0, help="取第几个输入继续追")
    a = ap.parse_args()

    blob = open(a.onnx, "rb").read()
    root = parse_fields(blob)
    g = parse_fields(one(root, 7, 2))
    nodes = [parse_node(x) for x in get(g, 1, 2)]
    inits = {}
    for x in get(g, 5, 2):
        n, dt, dims, arr, hr = parse_tensor(x)
        inits[n] = (dims, arr)

    # 图的输出名
    outs = []
    for vb in get(g, 12, 2):
        fs = parse_fields(vb)
        nm = one(fs, 1, 2, b"")
        if nm:
            outs.append(nm.decode("utf-8", "replace"))
    print(f"图输出: {outs}")

    prod = {}
    for n in nodes:
        for o in n["out"]:
            prod[o] = n

    def desc_const(name):
        if name not in inits:
            return None
        dims, arr = inits[name]
        if arr is None or arr.size == 0:
            return f"dims={dims}"
        if arr.size == 1:
            return f"**标量 {arr.ravel()[0]:.8g}**"
        if arr.size <= 8:
            return f"dims={dims} = {arr.ravel().tolist()}"
        return (f"dims={dims} 量级[{np.nanmin(arr):.4g},{np.nanmax(arr):.4g}] "
                f"absmean={np.nanmean(np.abs(arr)):.4g}")

    cur = a.seed if a.seed else (outs[0] if outs else None)
    print(f"\n{'='*92}")
    print(f"从 {cur} 往回追 {a.depth} 步")
    print(f"{'='*92}")
    for step in range(a.depth):
        n = prod.get(cur)
        if n is None:
            print(f"\n[{step}] {cur}  <- 图输入或常量, 停止")
            break
        print(f"\n[{step}] {n['op']}   name={n['name'][-56:]}")
        for i in n["in"]:
            c = desc_const(i)
            tag = "  (常量)" if c else ""
            print(f"      in  {i[-52:]:<54}{tag}")
            if c:
                print(f"          {c}")
        for o in n["out"]:
            print(f"      out {o[-52:]}")
        if n["attrs"]:
            print(f"      attrs {n['attrs']}")
        # 取第一个非常量输入作为下一步
        nxt = None
        cand = [i for i in n["in"] if i in prod]
        if cand:
            nxt = cand[min(a.branch, len(cand) - 1)]
        if nxt is None:
            print("      上一个输入是常量/图输入, 停止")
            break
        cur = nxt
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
