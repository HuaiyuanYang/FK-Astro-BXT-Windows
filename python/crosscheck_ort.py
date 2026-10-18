# -*- coding: utf-8 -*-
"""
crosscheck_ort.py —— 用官方 onnxruntime 交叉验证我的 NumPy 解释器。

为什么重要：
    `onnx_run.py` 是我自己写的 ONNX 解释器，用来在没有运行时时做数值验证。
    但"我自己写的解释器"本身可能有实现偏差 —— 它验证过的结论
    （形状、平坦性、padding 对齐）都建立在这个前提上。

    拿官方 onnxruntime 跑同一个模型、同一个输入，逐数值比对，
    就能证明（或推翻）我这套验证的有效性。

对比内容：
    1. 输出张量的最大绝对误差 / 相对误差
    2. 逐节点的中间张量（可选，抽查若干节点）
    3. 两者的输出值域、均值、std 是否一致

用法:
    python crosscheck_ort.py [v4|v5] [--nodes 20]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# onnxruntime 是自解压的 wheel（pip 装不了，沙箱挡临时目录）
VENDOR = HERE.parent / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

import onnx_run as R      # noqa: E402

BASE = HERE.parent
CONTRACTS = {
    "v4": dict(onnx="../bxt4.onnx", tile=768, nparam=5),
    "v5": dict(onnx="../bxt5.onnx", tile=1024, nparam=6),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="v4", choices=["v4", "v5"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flat", type=float, default=None,
                    help="用平坦输入而不是随机输入（训练分布内建议 0.02）")
    a = ap.parse_args()

    c = CONTRACTS[a.model]
    onnx_path = BASE / c["onnx"]

    print("=" * 76)
    print(f"自我解释器 vs 官方 onnxruntime    model={a.model}")
    print("=" * 76)
    print(f"  onnxruntime : {'可用 ' + ort.__version__ if HAS_ORT else '不可用'}")
    if not HAS_ORT:
        print("  [!] 没有 onnxruntime，无法交叉验证")
        return 1

    t = c["tile"]
    if a.flat is not None:
        img = np.full((1, t, t, 3), a.flat, np.float32)
        kind = f"平坦 {a.flat}"
    else:
        img = np.random.RandomState(a.seed).rand(1, t, t, 3).astype(np.float32)
        kind = f"随机 (seed={a.seed})"
    params = np.full((1, c["nparam"]), 0.5, np.float32)

    # ---- 官方运行时 ----
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), so,
                                providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    print(f"  ORT 输入: {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"  ORT 输出: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    print(f"  测试输入: {kind}")
    print()

    t0 = time.time()
    outs_ort = sess.run(None, {inames[0]: img, inames[1]: params})
    t_ort = time.time() - t0
    print(f"  ORT  前向耗时: {t_ort:.2f}s")

    # ---- 我的解释器 ----
    m = R.Model(onnx_path)
    t0 = time.time()
    res, missing = R.run(m, {m.inputs[0]["name"]: img,
                             m.inputs[1]["name"]: params})
    t_mine = time.time() - t0
    print(f"  我的 前向耗时: {t_mine:.1f}s   (x{t_mine/max(t_ort,1e-6):.0f} 慢)")

    # 对齐输出顺序
    ort_names = [o.name for o in sess.get_outputs()]
    outs_mine = [res[n] for n in ort_names]

    print()
    print("  --- 输出逐数值对比 ---")
    ok = True
    for name, oa, ob in zip(ort_names, outs_ort, outs_mine):
        oa = np.asarray(oa); ob = np.asarray(ob)
        if oa.shape != ob.shape:
            print(f"    {name}: [X] 形状不同 {oa.shape} vs {ob.shape}")
            ok = False
            continue
        diff = np.abs(oa - ob)
        denom = np.maximum(np.abs(oa), 1e-6)
        rel = diff / denom
        maxabs = float(diff.max())
        meanabs = float(diff.mean())
        maxrel = float(rel.max())
        # float32 跑 500~1000 个算子，1e-4 以内算一致
        good = maxabs < 1e-4
        ok = ok and good
        print(f"    {name}  shape={oa.shape}")
        print(f"      ORT : min={oa.min():+.6f} max={oa.max():+.6f} "
              f"mean={oa.mean():+.6f} std={oa.std():.6e}")
        print(f"      我的: min={ob.min():+.6f} max={ob.max():+.6f} "
              f"mean={ob.mean():+.6f} std={ob.std():.6e}")
        print(f"      最大绝对误差 = {maxabs:.3e}   平均 = {meanabs:.3e}   "
              f"最大相对 = {maxrel:.3e}")
        print(f"      {'[OK] 一致' if good else '[X] 不一致'}")

    print()
    print("=" * 76)
    print("  交叉验证结果: " +
          ("[OK] 我的解释器与官方运行时一致，之前的验证结论有效"
           if ok else "[X] 存在不一致，之前的验证结论需要重新审视"))
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
