# -*- coding: utf-8 -*-
"""
trace_shapes.py —— 逐节点跑一遍 ONNX 模型并打印形状变化。

调试形状问题的利器：只看最终报错往往看不出是哪一步把维度搞歪的，
逐节点打印就能一眼看到"从哪一层开始 shape 不对"。

用法:
    python trace_shapes.py <model.onnx> [输入尺寸] [从第几个节点开始]
    python trace_shapes.py bxt5.onnx 32 0
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnx_run as R      # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bxt5.onnx"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    m = R.Model(path)
    env = dict(m.initializers)
    for k, v in enumerate(m.inputs):
        shp = [d if d is not None else 1 for d in v["shape"]]
        if k == 0:
            shp = [1, size, size, shp[-1]]
            env[v["name"]] = np.random.RandomState(0).rand(*shp).astype(np.float32)
        else:
            env[v["name"]] = np.zeros(shp, np.float32)

    print(f"模型 {Path(path).name}: {len(m.nodes)} 节点，输入尺寸 {size}")
    print("=" * 100)
    for i, n in enumerate(m.nodes):
        op = n["op"]
        fn = R.DISPATCH.get(op)
        ins = [env.get(x) if x else None for x in n["in"]]
        try:
            r = fn(ins, n["attrs"], {"_n_out": n["out"], "node": n})
        except Exception as e:                              # noqa: BLE001
            print(f"[{i:>4}] {op:<12} ** 失败 ** {type(e).__name__}: {e}")
            print(f"        输入:")
            for nm, a in zip(n["in"], ins):
                print(f"          {nm[:60]:<60} {None if a is None else a.shape}")
            print(f"        属性: {n['attrs']}")
            return 1
        outs = [y for y in n["out"] if y]
        if len(outs) == 1:
            env[outs[0]] = r
            rs = [r]
        else:
            for y, v in zip(outs, r):
                env[y] = v
            rs = list(r)

        if i >= start:
            ins_s = ", ".join(
                f"{nm.split('_')[-1][:14]}:{None if a is None else 'x'.join(map(str,a.shape))}"
                for nm, a in zip(n["in"], ins))
            out_s = ", ".join(
                f"{y.split('_')[-1][:14]}:{None if v is None else 'x'.join(map(str,v.shape))}"
                for y, v in zip(outs, rs))
            print(f"[{i:>4}] {op:<12} {ins_s:<62} -> {out_s}")

    print("=" * 100)
    print("全部节点执行完毕，没有形状错误。")
    for o in m.outputs:
        v = env.get(o["name"])
        print(f"  输出 {o['name']}: "
              f"{None if v is None else v.shape}  "
              f"min={None if v is None else np.nanmin(v):+.6f} "
              f"max={None if v is None else np.nanmax(v):+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
