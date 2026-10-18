#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fkv3.py —— 用 FKv3 方案跑测试图并与官方 BXT 逐像素比对。

用法:
    python test_fkv3.py --dir _peak  --bank peak_bank.fits  --bxt bxt_peak.fits
    python test_fkv3.py --dir _scale --bank scale_bank.fits --bxt bxt_scale.fits
"""
import argparse
import json
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

import fits_io                     # noqa: E402
import fkv3                        # noqa: E402
import onnxruntime as ort          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_peak")
    ap.add_argument("--bank", default="peak_bank.fits")
    ap.add_argument("--bxt", default="bxt_peak.fits")
    ap.add_argument("--man", default="manifest.json")
    ap.add_argument("--onnx", default="bxt5.onnx")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--correct-only", type=int, default=1)
    a = ap.parse_args()

    D = os.path.join(CONVERT, os.pardir, a.dir)
    src, _ = fits_io.read_fits(os.path.join(D, a.bank))
    bxt, _ = fits_io.read_fits(os.path.join(D, a.bxt))
    hwc = np.moveaxis(np.ascontiguousarray(src, np.float32), 0, -1).astype(np.float64)
    ghwc = np.moveaxis(np.ascontiguousarray(bxt, np.float32), 0, -1).astype(np.float64)

    nrm = fkv3.ArcsinhNormalizer().fit(hwc)
    pv = nrm.parameters(correct_only=bool(a.correct_only))

    onnx = os.path.join(CONVERT, os.pardir, a.onnx)
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
    def predict(x):
        return sess.run(None, {"input": np.ascontiguousarray(x, np.float32),
                               "parameters": pv.reshape(1, 6)})[0]

    out, diag = fkv3.run_fkv3(hwc, predict, nrm, tile_out=512,
                              overlap=a.overlap, tile_in=1024, want_diag=True)

    print("=" * 96)
    print(f"FKv3 方案   图 = {a.bank}   分辨率 {hwc.shape[1]}x{hwc.shape[0]}")
    print("=" * 96)
    print("归一化状态: " + nrm.describe())
    print(f"参数向量: {pv.tolist()}")
    print(f"分块: {diag['tiles']}  wsum 范围 [{diag['wsum_min']:.4f}, {diag['wsum_max']:.4f}]")

    d = (out - ghwc).astype(np.float64)
    print("\n--- 与官方 BXT 的逐像素差([0,1] 域) ---")
    print(f"  mean |d| = {np.abs(d).mean():.3e}    RMS = {np.sqrt((d**2).mean()):.3e}")
    print(f"  P50 = {np.percentile(np.abs(d),50):.3e}   "
          f"P99 = {np.percentile(np.abs(d),99):.3e}   "
          f"P99.9 = {np.percentile(np.abs(d),99.996):.3e}   max = {np.abs(d).max():.3e}")
    bright = ghwc.mean(axis=2) > 0.03
    if bright.any():
        rel = np.abs(d[bright]).mean() / max(ghwc[bright].mean(), 1e-12)
        print(f"  亮区(a>0.03)相对误差 = {rel*100:.2f}%")
    print(f"  |d|>0.3 的像素占比 = {(np.abs(d)>0.3).mean()*100:.4f}%")
    print(f"  相关系数 = {np.corrcoef(out.ravel(), ghwc.ravel())[0,1]:.6f}")

    # 逐星点
    mp = os.path.join(D, a.man)
    if os.path.isfile(mp):
        blobs = json.load(open(mp, encoding="utf-8"))["blobs"]
        go = out.mean(axis=2); gb = ghwc.mean(axis=2); gi = hwc.mean(axis=2)
        print(f"\n{'对象':>8} {'输入':>9} {'官方':>9} {'FKv3':>9} {'差':>10}")
        print("-" * 52)
        for p in blobs:
            y, x = p["y"], p["x"]
            tag = f"σ{p['sigma']}" if "sigma" in p and "peak" not in p else f"P{p['peak']:.2f}"
            print(f"{tag:>8} {gi[y,x]:>9.4f} {gb[y,x]:>9.4f} {go[y,x]:>9.4f} {go[y,x]-gb[y,x]:>+10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
