#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_scale_bank.py —— 造"尺度阶梯"测试图: 同一张图上放 σ 从 0.6 到 5.0 的星点。

为什么这样做:
    现象是"参数全零时, 单颗孤立星我们和 BXT 几乎精确吻合(差 0.0007),
    但真实密集星场上差 7 倍"。同样的参数、同样的模型 -> 差异来自**输入的性质**。
    最可能的解释是多尺度(金字塔)路径有偏差: 只涉及少数尺度的简单输入不受影响,
    涉及全尺度的复杂输入偏差累积。

    σ 阶梯就是把"尺度"这个变量单独拎出来测:
      * 若某个 σ 区间偏差明显更大 -> 就是那一层的问题
      * 若所有 σ 都差不多        -> 不是多尺度问题, 要换方向

图: 1024x1024x3, FITS BITPIX=-32, 背景 0.004, 星点峰值 0.30(不饱和)。
    7 个 σ, 4x2 网格排布, 彼此 >=240 px, 互不干扰。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_synth_stars import write_fits_f32           # noqa: E402

SIGMAS = [0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 5.0]
GRID_X = [160, 400, 640, 880]
GRID_Y = [300, 700]
BG = 0.004
PEAK = 0.30
N = 1024


def main():
    conv = os.path.dirname(HERE)
    out = os.path.join(conv, os.pardir, "_scale")
    os.makedirs(out, exist_ok=True)

    img = np.full((N, N), BG, dtype=np.float64)
    yy, xx = np.mgrid[0:N, 0:N]
    pts = []
    idx = 0
    for sy in GRID_Y:
        for sx in GRID_X:
            if idx >= len(SIGMAS):
                break
            s = SIGMAS[idx]
            img += PEAK * np.exp(-((yy-sy)**2 + (xx-sx)**2) / (2*s*s))
            pts.append({"sigma": s, "y": sy, "x": sx, "peak": PEAK})
            idx += 1

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    cube = np.repeat(img[None, :, :], 3, axis=0)
    name = "scale_bank.fits"
    write_fits_f32(os.path.join(out, name), cube,
                   {"OBJECT": "SCALEBANK", "BGVAL": f"{BG:.6f}"})
    json.dump({"n": N, "bg": BG, "peak": PEAK, "blobs": pts},
              open(os.path.join(out, "manifest.json"), "w"), indent=2)

    print(f"  写出 {name}   背景 {BG}   峰值 {PEAK}   {len(pts)} 个星点")
    for p in pts:
        print(f"    sigma={p['sigma']:<4}  位置 ({p['y']},{p['x']})"
              f"   FWHM={2.3548*p['sigma']:.2f} px")
    print(f"  输出目录 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
