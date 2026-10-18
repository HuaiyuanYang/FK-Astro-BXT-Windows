#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_synth_stars.py —— 造"平坦背景 + 已知高斯星"的合成测试图(FITS float32)。

为什么要合成图:
    真实 NGC7331 场里 460 万像素挤在背景一个箱、星点样本只有几百个,
    统计上分不清"曲线差异"和"结构性差异"。合成图可以精确控制:
      * 背景电平 bg      -> 直接检验 wp = 0.505*bg 这个系数
      * 星点 FWHM 与峰值  -> 直接量峰值、半宽、翼部
      * 完全没有星云/星系结构 -> 没有混淆

    扫多个 bg, 看偏差是否随 bg 变:
      * 若偏差随 bg 变  -> 背景统计口径或 wp 系数的问题, 可反解
      * 若偏差与 bg 无关 -> 结构性问题(权重/转换/分块)

图:1024x1024x3(三通道内容相同), FITS BITPIX=-32。
    星点:4 颗, 峰值 0.05 / 0.15 / 0.40 / 0.75, sigma=1.5 px(FWHM≈3.53),
          位置避开边缘(离边 ≥160 px, 避免反射填充干扰)。

用法:
    python make_synth_stars.py [--outdir DIR]
"""
import argparse
import json
import os

import numpy as np


def write_fits_f32(path, cube, extra=None):
    """cube (C,H,W) float32 -> FITS BITPIX=-32"""
    C, H, W = cube.shape
    cards = []

    def card(s):
        cards.append(s.ljust(80))

    card(f"{'SIMPLE':<8}= {'T':>20}")
    card(f"{'BITPIX':<8}= {-32:>20}")
    card(f"{'NAXIS':<8}= {3:>20}")
    card(f"{'NAXIS1':<8}= {W:>20}")
    card(f"{'NAXIS2':<8}= {H:>20}")
    card(f"{'NAXIS3':<8}= {C:>20}")
    card(f"{'BSCALE':<8}= {1.0:>20}")
    card(f"{'BZERO':<8}= {0.0:>20}")
    for k, v in (extra or {}).items():
        s = f"{k:<8}= {str(v):>20}"
        if len(s) > 80:
            s = s[:80]
        card(s)
    card("END")
    blk = "".join(cards)
    blk += " " * ((2880 - len(blk) % 2880) % 2880)
    data = np.ascontiguousarray(cube, dtype=">f4").tobytes()
    pad = (2880 - len(data) % 2880) % 2880
    with open(path, "wb") as f:
        f.write(blk.encode("ascii"))
        f.write(data)
        f.write(b"\0" * pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = a.outdir or os.path.join(os.path.dirname(here), os.pardir, "_synth")
    os.makedirs(outdir, exist_ok=True)

    N = 1024
    SIGMA = 1.5
    # (y, x, 峰值)—— 都在离边 >= 200 px 处, 彼此 >= 300 px, 互不干扰
    stars = [(260, 260, 0.75), (260, 720, 0.40), (720, 260, 0.15), (720, 720, 0.05)]
    bgs = [0.002, 0.004, 0.008, 0.016]

    yy, xx = np.mgrid[0:N, 0:N]
    profile = np.zeros((N, N), dtype=np.float64)
    for (sy, sx, amp) in stars:
        profile += amp * np.exp(-((yy - sy) ** 2 + (xx - sx) ** 2) / (2 * SIGMA * SIGMA))

    manifest = {"n": N, "sigma": SIGMA, "stars": stars, "cases": []}
    for bg in bgs:
        rng = np.random.RandomState(int(bg * 1e6) & 0x7FFFFFFF)
        # 背景噪声: 幅度取 bg 的 1.5%, 让不同 bg 的相对噪声一致
        noise = rng.normal(0.0, bg * 0.015, (N, N))
        img = np.clip(bg + noise + profile, 0.0, 1.0).astype(np.float32)
        cube = np.repeat(img[None, :, :], 3, axis=0)          # (3,H,W)
        name = f"synth_bg{bg:.4f}.fits"
        path = os.path.join(outdir, name)
        write_fits_f32(path, cube, {"OBJECT": "SYNTH", "BGVAL": f"{bg:.6f}"})
        manifest["cases"].append({"bg": bg, "file": name})
        print(f"  {name}   bg={bg:.4f}   峰值 {img.max():.4f}   中位 {np.median(img):.6f}")

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n输出目录: {outdir}")
    print(f"星点: {stars}")
    print(f"sigma={SIGMA} px  ->  FWHM = {2.3548*SIGMA:.3f} px")
    print(f"已写 manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
