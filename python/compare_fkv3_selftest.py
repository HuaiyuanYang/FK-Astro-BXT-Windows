#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compare_fkv3_selftest.py —— 把 C++ 的 FKv3 输出与 Python 参考实现逐像素对比。

用法:
    python compare_fkv3_selftest.py <前缀> [--model PATH] [--overlap 0.2] [--mode co|sh]

约定:
    <前缀>_in.f32    输入, 平面布局 (C,H,W), float32
    <前缀>_out.f32   C++ 的输出, 同上
    Python 读同一个输入跑 run_fkv3, 结果与 C++ 比。

判定: 两边跑的是同一套数学, 差异应当只来自浮点舍入。
      P99 < 1e-5 视为移植正确。
"""
import argparse
import os
import sys

import numpy as np


def find_model(explicit=None):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    ws = os.path.dirname(proj)
    for c in (os.path.join(ws, "bxt5.onnx"),
              os.path.join(proj, "bxt5.onnx"),
              os.path.join(proj, "models", "bxt5.onnx")):
        if os.path.isfile(c):
            return c
    return None


def read_planar(path, C, H, W):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != C * H * W:
        raise SystemExit(f"{path}: 期望 {C*H*W} 个样本, 实得 {a.size}")
    return a.reshape(C, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--model", default=None)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--tile-out", type=int, default=512)
    ap.add_argument("--tile-in", type=int, default=1024)
    ap.add_argument("--mode", default="co", choices=["co", "sh"])
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import onnxruntime as ort
    import fkv3

    model = find_model(a.model)
    if not model:
        raise SystemExit("找不到 bxt5.onnx, 用 --model 指定")

    cpp_out = a.prefix + "_out.f32"
    src = a.prefix + "_in.f32"
    if not os.path.isfile(src) or not os.path.isfile(cpp_out):
        raise SystemExit(f"缺文件: {src} / {cpp_out}")

    # 从文件名拿不到形状 -> 用 C++ 侧打印的样本数反推不了, 所以按约定:
    # 由 --shape 推导。这里用一个简单办法: 先读 C++ 输出, 由总数与通道数求解。
    raw_in = np.fromfile(src, dtype=np.float32)
    raw_out = np.fromfile(cpp_out, dtype=np.float32)
    if raw_out.size % raw_in.size != 0:
        raise SystemExit("输入/输出样本数不成比例")

    # 形状从命令行补: 约定用环境变量或额外参数太啰嗦, 这里按 (C,H,W) 由
    # 调用方通过 --shape 给出; 未给出时尝试 C=3 / C=1 的常见情形。
    C = int(os.environ.get("FKV3_C", "0"))
    H = int(os.environ.get("FKV3_H", "0"))
    W = int(os.environ.get("FKV3_W", "0"))
    if not (C and H and W):
        # 退化推断: 由样本数找最接近的方形尺寸
        n = raw_in.size
        for cand in (3, 1):
            if n % cand == 0:
                hw = n // cand
                h = int(round(hw ** 0.5))
                if h * h == hw:
                    C, H, W = cand, h, h
                    break
        if not (C and H and W):
            raise SystemExit("无法推断形状, 请设置 FKV3_C / FKV3_H / FKV3_W")

    img = read_planar(src, C, H, W).transpose(1, 2, 0)     # (H,W,C)
    cpp = read_planar(cpp_out, C, H, W).transpose(1, 2, 0) # (H,W,C)

    # ---- ORT 会话 ----
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]

    # 归一化与参数必须由同一份拟合状态产生 —— 第 6 列就是 gain,
    # 传错会让模型行为完全改变(这条曾把对账结果推到 3e-4)。
    nrm = fkv3.fit_image(img)
    co = (a.mode == "co")
    pv = nrm.parameters(stars=0.0 if co else 0.5,
                        halos=0.0,
                        nonstellar=0.0 if co else 0.5,
                        correct_only=co)
    print(f"  normalizer low={nrm.low:.9g} median={nrm.median:.9g} x={nrm.x:.9g} gain={nrm.gain:.9g}")

    def predict(batch_nhwc):
        feed = {in_names[0]: np.ascontiguousarray(batch_nhwc, dtype=np.float32)}
        if len(in_names) > 1:
            feed[in_names[1]] = np.asarray(pv, dtype=np.float32).reshape(1, -1)
        return sess.run(None, feed)[0]

    py = fkv3.run_fkv3(img, predict, nrm,
                       tile_out=a.tile_out, overlap=a.overlap, tile_in=a.tile_in)

    # mono 时 run_fkv3 返回 (H,W), C++ 侧是 (H,W,1) —— 先对齐
    if py.ndim == 2 and cpp.ndim == 3:
        cpp = cpp[:, :, 0]
    elif cpp.ndim == 2 and py.ndim == 3:
        py = py[:, :, 0]

    # ---- 对比 ----
    d = py.astype(np.float64) - cpp.astype(np.float64)
    ad = np.abs(d)
    flat = ad.ravel()
    p99 = float(np.percentile(flat, 99))
    p999 = float(np.percentile(flat, 99.9))
    mx = float(flat.max())
    # 相关系数
    x1 = py.astype(np.float64).ravel()
    x2 = cpp.astype(np.float64).ravel()
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    corr = float((x1 * x2).sum() / (np.sqrt((x1 * x1).sum() * (x2 * x2).sum()) + 1e-300))

    print(f"  shape {C}x{H}x{W}  overlap={a.overlap}  tiles={-(-H//a.tile_out)}x{-(-W//a.tile_out)}")
    print(f"  max {mx:.3e}   P99.9 {p999:.3e}   P99 {p99:.3e}   corr {corr:.9f}")
    print(f"  通过阈值 P99 < 1e-5 : {'OK' if p99 < 1e-5 else 'FAIL'}")

    # 分通道看
    for c in range(C):
        cc = np.abs(py[..., c].astype(np.float64) - cpp[..., c].astype(np.float64))
        print(f"    ch{c}: max {cc.max():.3e}  P99 {np.percentile(cc, 99):.3e}")

    return 0 if p99 < 1e-5 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
