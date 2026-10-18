# -*- coding: utf-8 -*-
"""
verify_shapes.py —— 逐节点对比「ONNX 实算形状」与「MIL 声明形状」。

这是定位形状 bug 最有效的方法：
    MIL 的每个 Operation 都带 ValueType（声明输出形状）。
    转换后的 ONNX 跑出来应该和它一致。第一个不一致的节点就是 bug 所在。

用法:
    python verify_shapes.py <mlpackage 目录> <model.onnx> [输入边长]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mil_graph          # noqa: E402
import onnx_run as R      # noqa: E402


def main():
    pkg = sys.argv[1] if len(sys.argv) > 1 else \
        "../mlpackage/BlurXTerminator.4.mlpackage"
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else "bxt5.onnx"
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 768
    max_report = 40

    # MIL 声明的形状（以输出名为键）
    g = mil_graph.load(pkg)
    mil = {}
    for o in g.ops:
        if o.dims:
            mil[o.name] = tuple(int(d) for d in o.dims)

    m = R.Model(onnx_path)
    env = dict(m.initializers)
    for k, v in enumerate(m.inputs):
        shp = [d if d is not None else 1 for d in v["shape"]]
        if k == 0:
            shp = [1, size, size, shp[-1]]
            env[v["name"]] = np.random.RandomState(0).rand(*shp).astype(np.float32)
        else:
            env[v["name"]] = np.zeros(shp, np.float32)

    print("=" * 96)
    print(f"MIL 声明 vs ONNX 实算   输入 {size}x{size}")
    print("=" * 96)
    print(f"{'ONNX#':>5} {'op':<11} {'输出名':<44} {'实算':<22} {'MIL 声明':<22} 判定")
    print("-" * 96)

    bad = []
    checked = 0
    for i, n in enumerate(m.nodes):
        fn = R.DISPATCH.get(n["op"])
        ins = [env.get(x) if x else None for x in n["in"]]
        try:
            r = fn(ins, n["attrs"], {"_n_out": n["out"], "node": n})
        except Exception as e:                              # noqa: BLE001
            print(f"{i:>5} {n['op']:<11} ** 执行失败 ** {type(e).__name__}: {e}")
            print(f"      输入: {[(x[-28:], None if a is None else a.shape) for x,a in zip(n['in'], ins)]}")
            break
        outs = [y for y in n["out"] if y]
        rs = r if isinstance(r, (list, tuple)) else [r]
        for y, v in zip(outs, rs):
            env[y] = v

        y = outs[0] if outs else None
        got = tuple(v.shape) if (outs and rs) else None
        want = mil.get(y)
        if want is None:
            continue
        checked += 1
        ok = (got == want)
        if not ok:
            bad.append((i, n["op"], y, got, want))
        if len(bad) <= max_report and not ok:
            print(f"{i:>5} {n['op']:<11} {str(y)[:44]:<44} {str(got):<22} {str(want):<22} ✗")
        elif ok and len(bad) == 0 and checked <= 60:
            print(f"{i:>5} {n['op']:<11} {str(y)[:44]:<44} {str(got):<22} {str(want):<22} ok")

    print("-" * 96)
    print(f"对比了 {checked} 个有 MIL 声明形状的节点，不符 {len(bad)} 个")
    if bad:
        i, op, y, got, want = bad[0]
        print()
        print("  ** 第一个不符的节点 **")
        print(f"     ONNX 序号 : {i}")
        print(f"     算子      : {op}")
        print(f"     输出名    : {y}")
        print(f"     实算      : {got}")
        print(f"     MIL 声明  : {want}")
        print()
        print("  差异分析:")
        if got is not None and want is not None and len(got) == len(want):
            for k, (a, b) in enumerate(zip(got, want)):
                if a != b:
                    print(f"     第 {k} 维: 实算 {a}  vs  声明 {b}   （比值 {b/a if a else '?'}）")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
