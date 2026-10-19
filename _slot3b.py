# -*- coding: utf-8 -*-
"""量化 slot3 = 1(已验证) 与 slot3 = 0(模块当前行为) 的差异。"""
import os
import sys

import numpy as np

PROJ = r"C:\Users\yhy20\Desktop\Deepseek Daily\FK-Astro-BXT-Windows"
sys.path.insert(0, os.path.join(PROJ, "python"))

import fkv3            # noqa: E402
import fits_io         # noqa: E402
import onnxruntime as ort   # noqa: E402

FITS = r"E:\素材\asker 106\light\Light_M 31_300.0s_Bin1_ARGB_20260913-003744_31deg_0002.fit"
MODEL = r"C:\Users\yhy20\Desktop\Deepseek Daily\bxt5.onnx"

print("  读取 FITS ...")
a, hdr = fits_io.read_fits(FITS)
a = np.asarray(a, np.float32)
mx = float(a.max()) if a.size else 0.0
if mx > 1.5:
    a = a / (65535.0 if mx > 300.0 else 255.0)
img = a[:, :, None].astype(np.float64) if a.ndim == 2 else a.astype(np.float64)
print("  形状:", img.shape, " 值域: %.4g ~ %.4g" % (img.min(), img.max()))

nrm = fkv3.fit_image(img)
print("  归一化:", nrm.describe())

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])


def run(slot3):
    pv = np.array([0.0, 0.0, 0.0, float(slot3), 0.0, nrm.gain], dtype=np.float32)

    def predict(b):
        return sess.run(None, {
            "input": np.ascontiguousarray(b, np.float32),
            "parameters": pv.reshape(1, 6),
        })[0]

    out = fkv3.run_fkv3(img, predict, nrm, tile_out=512, overlap=0.0, tile_in=1024)
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    return out


print("\n  跑 slot3 = 1 (已验证配置) ...")
x1 = run(1)
print("  跑 slot3 = 0 (模块当前) ...")
x0 = run(0)

d = np.abs(x1.astype(np.float64) - x0.astype(np.float64))
print("\n  === slot3=1  vs  slot3=0 ===")
print("   max    %.4e   (满量程的 %.3f%%)" % (d.max(), d.max() * 100))
print("   P99.9  %.4e" % np.percentile(d, 99.9))
print("   P99    %.4e" % np.percentile(d, 99))
print("   mean   %.4e" % d.mean())
print("   相关   %.9f" % np.corrcoef(x1.ravel().astype(np.float64),
                                    x0.ravel().astype(np.float64))[0, 1])
print("\n  两者的整体亮度:")
print("   slot3=1  mean %.6f  max %.6f" % (x1.mean(), x1.max()))
print("   slot3=0  mean %.6f  max %.6f" % (x0.mean(), x0.max()))
