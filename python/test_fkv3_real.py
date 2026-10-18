#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fkv3_real.py —— 在真实 XISF 上跑 FKv3,与官方 BXT 参考输出逐像素比对。

注意: 归一化状态依赖**整图统计量**, 所以必须整图跑, 不能裁剪后再比
      (裁剪会改变 low/median/x)。
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

import xisf_io                     # noqa: E402
import fkv3                        # noqa: E402
import onnxruntime as ort          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True, help="官方 BXT 参考输出 (xisf)")
    ap.add_argument("--onnx", default="bxt5.onnx")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--correct-only", type=int, default=1)
    ap.add_argument("--sharpen", type=float, default=0.5)
    ap.add_argument("--sharpen-nonstellar", type=float, default=0.5)
    a = ap.parse_args()

    print("读输入 ...", flush=True)
    src, _ = xisf_io.read_image(a.src)
    print("读参考 ...", flush=True)
    ref, _ = xisf_io.read_image(a.ref)
    hwc = np.moveaxis(np.asarray(src, np.float32), 0, -1).astype(np.float64)
    rhwc = np.moveaxis(np.asarray(ref, np.float32), 0, -1).astype(np.float64)
    print(f"  输入 {hwc.shape}   参考 {rhwc.shape}", flush=True)

    nrm = fkv3.ArcsinhNormalizer().fit(hwc)
    pv = nrm.parameters(stars=a.sharpen, nonstellar=a.sharpen_nonstellar,
                        correct_only=bool(a.correct_only))

    onnx = os.path.join(CONVERT, os.pardir, a.onnx)
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
    def predict(x):
        return sess.run(None, {"input": np.ascontiguousarray(x, np.float32),
                               "parameters": pv.reshape(1, 6)})[0]

    print(f"归一化: {nrm.describe()}", flush=True)
    print(f"参数: {pv.tolist()}   overlap={a.overlap}", flush=True)
    print("跑 FKv3 ...", flush=True)
    out, diag = fkv3.run_fkv3(hwc, predict, nrm, tile_out=512, overlap=a.overlap,
                              tile_in=1024, want_diag=True)

    d = (out - rhwc).astype(np.float64)
    ad = np.abs(d)
    print("\n" + "=" * 88)
    print(f"FKv3 vs 官方 BXT    {hwc.shape[1]}x{hwc.shape[0]}   overlap={a.overlap}   "
          f"correct_only={a.correct_only}")
    print("=" * 88)
    print(f"  mean |d| = {ad.mean():.3e}    RMS = {np.sqrt((d**2).mean()):.3e}")
    print(f"  P50 = {np.percentile(ad,50):.3e}   P99 = {np.percentile(ad,99):.3e}   "
          f"P99.9 = {np.percentile(ad,99.9):.3e}   max = {ad.max():.3e}")
    bright = rhwc.mean(axis=2) > 0.03
    if bright.any():
        print(f"  亮区(a>0.03)相对误差 = "
              f"{ad[bright].mean()/max(rhwc[bright].mean(),1e-12)*100:.2f}%"
              f"   ({bright.sum()} px)")
    print(f"  |d|>0.3 占比 = {(ad>0.3).mean()*100:.4f}%")
    print(f"  相关系数 = {np.corrcoef(out.ravel(), rhwc.ravel())[0,1]:.7f}")
    print(f"  分块 {diag['tiles']}   wsum [{diag['wsum_min']:.4f}, {diag['wsum_max']:.4f}]")
    np.save(os.path.join(CONVERT, os.pardir, "_fkv3_real.npy"),
            out.astype(np.float32))
    print("  结果已存 _fkv3_real.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
