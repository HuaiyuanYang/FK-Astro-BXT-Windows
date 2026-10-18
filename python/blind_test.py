#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
blind_test.py —— 盲测:对工作区里新增的 5 张实拍图,逐一比较 FKv3 与官方 BXT。

不需要事先知道哪张是窄带、哪张是宽窄混合 —— 只按顺序处理并报告:
  * 归一化状态(bg / median / x / gain)—— 反映图的亮度与噪声水平
  * 通道统计(各通道均值、通道间相关)—— 窄带 H+O 的通道差异通常明显
  * 与官方 BXT 的逐像素差

判定标准(事先定好, 不事后调整):
  P99 < 2e-4 且 相关 > 0.999  ->  通过
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


IMGS = [
    "masterLight_BIN-1_3008x3008_EXPOSURE-300.00s_FILTER-NoFilter_RGB_autocrop.xisf",
    "masterLight_BIN-1_3008x3008_EXPOSURE-300.00s_FILTER-NoFilter_RGB_autocrop(1).xisf",
    "masterLight_BIN-1_3008x3008_EXPOSURE-30.00s_FILTER-NoFilter_RGB_autocrop.xisf",
    "masterLight_BIN-1_3008x3008_EXPOSURE-300.00s_FILTER-NoFilter_RGB_autocrop(2).xisf",
    "masterLight_BIN-1_4948x3280_EXPOSURE-300.00s_FILTER-NoFilter_RGB_autocrop.xisf",
]


def stats_of(hwc):
    """通道统计 —— 用来客观描述图的性质, 不做猜测"""
    c = hwc.shape[2]
    bg = float(np.percentile(hwc[hwc > 0], 35)) if (hwc > 0).any() else 0.0
    means = [float(hwc[:, :, k].mean()) for k in range(min(c, 3))]
    cors = {}
    if c >= 3:
        a, b, d = (hwc[:, :, 0].ravel(), hwc[:, :, 1].ravel(), hwc[:, :, 2].ravel())
        cors["R-G"] = float(np.corrcoef(a, b)[0, 1])
        cors["R-B"] = float(np.corrcoef(a, d)[0, 1])
        cors["G-B"] = float(np.corrcoef(b, d)[0, 1])
    return bg, means, cors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--onnx", default="bxt5.onnx")
    ap.add_argument("--only", type=int, default=0, help="只跑第 N 张(1 起), 0 = 全部")
    a = ap.parse_args()

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(os.path.join(MODELS, a.onnx), so,
                                providers=["CPUExecutionProvider"])

    rows = []
    for idx, fn in enumerate(IMGS, start=1):
        if a.only and idx != a.only:
            continue
        tag = f"b{idx}"
        ref_path = os.path.join(SAMPLES, f"bxt_{tag}.xisf")
        if not os.path.isfile(ref_path):
            print(f"[{tag}] 缺参考 {ref_path}, 跳过")
            continue
        src, _ = xisf_io.read_image(need(os.path.join(SAMPLES, fn), "测试图"))
        ref, _ = xisf_io.read_image(ref_path)
        hwc = np.moveaxis(np.asarray(src, np.float32), 0, -1).astype(np.float64)
        rhwc = np.moveaxis(np.asarray(ref, np.float32), 0, -1).astype(np.float64)

        nrm = fkv3.fit_image(hwc)
        pv = nrm.parameters(correct_only=True)
        bg, means, cors = stats_of(hwc)

        def predict(x):
            return sess.run(None, {"input": np.ascontiguousarray(x, np.float32),
                                   "parameters": pv.reshape(1, 6)})[0]

        out = fkv3.run_fkv3(hwc, predict, nrm, tile_out=512,
                            overlap=a.overlap, tile_in=1024)

        d = out - rhwc
        ad = np.abs(d)
        p99 = float(np.percentile(ad, 99))
        corr = float(np.corrcoef(out.ravel(), rhwc.ravel())[0, 1])
        ok = (p99 < 2e-4) and (corr > 0.999)

        print("=" * 96)
        print(f"[{tag}] {fn}")
        print("=" * 96)
        print(f"  尺寸 {hwc.shape[1]}x{hwc.shape[0]}x{hwc.shape[2]}   "
              f"bg(P35)={bg:.6g}   通道均值 {[round(m,6) for m in means]}")
        if cors:
            print("  通道相关 " + "  ".join(f"{k}={v:.4f}" for k, v in cors.items()))
        print(f"  归一化 {nrm.describe()}")
        print(f"  P50={np.percentile(ad,50):.3e}  P99={p99:.3e}  "
              f"P99.9={np.percentile(ad,99.9):.3e}  max={ad.max():.3e}")
        bright = rhwc.mean(axis=2) > 0.03
        if bright.any():
            print(f"  亮区相对误差 = "
                  f"{ad[bright].mean()/max(rhwc.mean(axis=2)[bright].mean(),1e-12)*100:.2f}%"
                  f"  ({bright.sum()} px)")
        print(f"  相关 = {corr:.8f}    判定 {'通过' if ok else '** 未通过 **'}")
        rows.append((tag, p99, corr, ok, nrm.gain, bg, hwc.shape))

    if len(rows) > 1:
        print("\n" + "=" * 96)
        print(f"{'图':<5} {'尺寸':<16} {'bg':>10} {'gain':>8} {'P99':>11} {'相关':>12}  判定")
        print("-" * 96)
        for tag, p99, corr, ok, gain, bg, shp in rows:
            dim = f"{shp[1]}x{shp[0]}x{shp[2]}"
            print(f"{tag:<5} {dim:<16} {bg:>10.5f} {gain:>8.4f} {p99:>11.3e} {corr:>12.8f}  "
                  f"{'通过' if ok else '** 未通过 **'}")
        n_ok = sum(1 for r in rows if r[3])
        print(f"\n{n_ok}/{len(rows)} 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
