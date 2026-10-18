#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_fkv3.py —— FKv3 与官方 BXT 的多场景逐像素验证。

每个场景: (输入图, 官方 BXT 参考, 参数, overlap) -> 精度指标

用法:
    python verify_fkv3.py --case o2co
    python verify_fkv3.py --all
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
import fits_io                     # noqa: E402
import fkv3                        # noqa: E402
import star_metrics as SM          # noqa: E402
import onnxruntime as ort          # noqa: E402


# ---- 路径解析 ---------------------------------------------------------------
# 模型( bxt5.onnx )由使用者自备, 放在工作区根目录; 也兼容放在项目内。
WORKSPACE = os.path.dirname(CONVERT)
def find_model():
    for c in (os.path.join(WORKSPACE, "bxt5.onnx"),
              os.path.join(CONVERT, "bxt5.onnx"),
              os.path.join(os.path.dirname(CONVERT), "bxt5.onnx")):
        if os.path.isfile(c):
            return c
    return None

# 测试图与官方参考输出都放 samples/ 下(不随仓库分发)。
SAMPLES = os.path.join(CONVERT, "samples")


def need(path, what):
    if not os.path.isfile(path):
        raise SystemExit(f"缺少{what}: {path}\n"
                         f"请把文件放到 samples\\ 目录, 或修改脚本里的路径。")
    return path


NGC = os.path.join(SAMPLES, "masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_FILTER-NoFilter_RGB_autocrop.xisf")
M31 = os.path.join(SAMPLES, "Light_M31_300s_0002.fit")

CASES = {
    "o0co":  dict(src=NGC, ref="bxt_ref_zero.xisf", overlap=0.0, correct_only=True,
                  stars=0, halos=0, nonstellar=0, label="NGC7331 ov=0  correct-only"),
    "o0sh":  dict(src=NGC, ref="bxt_ref_ov0.xisf",  overlap=0.0, correct_only=False,
                  stars=0.5, halos=0, nonstellar=0.5, label="NGC7331 ov=0  sharpen .5/.5"),
    "o2co":  dict(src=NGC, ref="bxt_o2_co.xisf",    overlap=0.2, correct_only=True,
                  stars=0, halos=0, nonstellar=0, label="NGC7331 ov=0.2 correct-only"),
    "o2sh":  dict(src=NGC, ref="bxt_o2_sh.xisf",    overlap=0.2, correct_only=False,
                  stars=0.5, halos=0, nonstellar=0.5, label="NGC7331 ov=0.2 sharpen .5/.5"),
    "halo":  dict(src=NGC, ref="bxt_halo.xisf",     overlap=0.0, correct_only=False,
                  stars=0.7, halos=0.2, nonstellar=0.3, label="NGC7331 ov=0  ss=.7 halo=.2 ns=.3"),
    "m31":   dict(src=M31, ref="bxt_m31.xisf",      overlap=0.0, correct_only=True,
                  stars=0, halos=0, nonstellar=0, label="M31 单张(单色) correct-only"),
}


def read_any(path):
    """-> (C,H,W) float32 [0,1]。

    注意: fits_io 已经应用了 BSCALE/BZERO(返回 0~65535 的物理值),
    所以这里**不能**再按 BITPIX 除一次,只能做一次统一标定。
    """
    if path.lower().endswith(".xisf"):
        a, _ = xisf_io.read_image(path)
    else:
        a, hdr = fits_io.read_fits(path)
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[None, :, :]
    mx = float(a.max()) if a.size else 0.0
    if mx > 1.5:                     # 整数存成浮点(ADU 值域) -> 归一到 [0,1]
        a = a / (65535.0 if mx > 300.0 else 255.0)
    return a

def orient_to(a, ref):
    """把 a 的 H/W 调成和 ref 一致(FITS 与 XISF 轴序常相反)"""
    if a.shape[1:3] == ref.shape[1:3]:
        return a
    if a.shape[1] == ref.shape[2] and a.shape[2] == ref.shape[1]:
        return np.transpose(a, (0, 2, 1))
    return a


def run_case(name, sess, verify=True):
    c = CASES[name]
    print("=" * 92)
    print(f"[{name}] {c['label']}")
    print("=" * 92)
    src = read_any(c["src"])
    ref = read_any(os.path.join(SAMPLES, c["ref"]))
    src = orient_to(src, ref)
    hwc = np.moveaxis(src, 0, -1).astype(np.float64)
    if hwc.ndim == 2:
        hwc = hwc[:, :, None]
    rhwc = np.moveaxis(ref, 0, -1).astype(np.float64)
    if rhwc.ndim == 2:
        rhwc = rhwc[:, :, None]
    print(f"  输入 {hwc.shape}   参考 {rhwc.shape}")

    nrm = fkv3.fit_image(hwc)
    pv = nrm.parameters(stars=c["stars"], halos=c["halos"],
                        nonstellar=c["nonstellar"], correct_only=c["correct_only"])
    print(f"  归一化 {nrm.describe()}")
    print(f"  参数 {pv.tolist()}  overlap={c['overlap']}")

    def predict(x):
        return sess.run(None, {"input": np.ascontiguousarray(x, np.float32),
                               "parameters": pv.reshape(1, 6)})[0]

    out, diag = fkv3.run_fkv3(hwc, predict, nrm, tile_out=512,
                              overlap=c["overlap"], tile_in=1024, want_diag=True)
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    if rhwc.shape[2] == 1:
        rhwc = rhwc[:, :, 0]

    d = out - rhwc
    ad = np.abs(d)
    print(f"  分块 {diag['tiles']}  wsum [{diag['wsum_min']:.4f}, {diag['wsum_max']:.4f}]")
    print(f"  P50   = {np.percentile(ad,50):.3e}")
    print(f"  P99   = {np.percentile(ad,99):.3e}")
    print(f"  P99.9 = {np.percentile(ad,99.9):.3e}")
    print(f"  max   = {ad.max():.3e}    mean = {ad.mean():.3e}")
    bright = rhwc.mean(axis=2) if rhwc.ndim == 3 else rhwc > 0.03
    if rhwc.ndim == 3:
        bright = rhwc.mean(axis=2) > 0.03
    if bright.any():
        base = (rhwc.mean(axis=2) if rhwc.ndim == 3 else rhwc)
        print(f"  亮区相对误差 = {ad[bright].mean()/max(base[bright].mean(),1e-12)*100:.2f}%"
              f"  ({bright.sum()} px)")
    print(f"  |d|>0.3 = {(ad>0.3).mean()*100:.5f}%")
    print(f"  相关系数 = {np.corrcoef(out.ravel(), rhwc.ravel())[0,1]:.8f}")

    if verify:      # FWHM 对比(需要能找星)
        gi = hwc.mean(axis=2) if hwc.ndim == 3 else hwc
        go = out if out.ndim == 2 else out.mean(axis=2)
        gr = rhwc if rhwc.ndim == 2 else rhwc.mean(axis=2)
        try:
            stars = SM.find_stars(gi, n=200)
            fi, fb, fo, hb, ho = [], [], [], 0, 0
            for (y, x) in stars:
                fa = SM.fwhm_from_profile(SM.radial_profile(gi, y, x, rad=16))
                if fa is None or not np.isfinite(fa) or fa > 10.0 or gi[y, x] > 0.97:
                    continue
                a1 = SM.fwhm_from_profile(SM.radial_profile(gr, y, x, rad=16))
                a2 = SM.fwhm_from_profile(SM.radial_profile(go, y, x, rad=16))
                if a1 is None or a2 is None or not np.isfinite(a1) or not np.isfinite(a2):
                    continue
                fi.append(fa); fb.append(a1); fo.append(a2)
            if len(fi) > 20:
                fi = np.array(fi); fb = np.array(fb); fo = np.array(fo)
                print(f"  FWHM 输入 {fi.mean():.3f}  BXT {fb.mean():.3f} ({fb.mean()/fi.mean():.4f})"
                      f"  FKv3 {fo.mean():.3f} ({fo.mean()/fi.mean():.4f})"
                      f"  差 {abs(fo.mean()-fb.mean()):.4f}  ({len(fi)} 颗)")
        except Exception as e:
            print(f"  (FWHM 跳过: {e})")
    print()
    return dict(name=name, p99=float(np.percentile(ad, 99)),
                p999=float(np.percentile(ad, 99.9)),
                corr=float(np.corrcoef(out.ravel(), rhwc.ravel())[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--onnx", default="bxt5.onnx")
    ap.add_argument("--no-fwhm", action="store_true")
    a = ap.parse_args()

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(a.onnx if os.path.isabs(a.onnx) else (find_model() or ""), so,
                                providers=["CPUExecutionProvider"])

    names = list(CASES) if a.all else [a.case]
    if not names or names == [None]:
        raise SystemExit("给 --case 或 --all")

    res = []
    for n in names:
        try:
            res.append(run_case(n, sess, verify=not a.no_fwhm))
        except Exception as e:
            print(f"[{n}] 失败: {e}\n")

    if len(res) > 1:
        print("=" * 92)
        print(f"{'场景':<8} {'P99':>12} {'P99.9':>12} {'相关':>12}")
        print("-" * 92)
        for r in res:
            print(f"{r['name']:<8} {r['p99']:>12.3e} {r['p999']:>12.3e} {r['corr']:>12.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
