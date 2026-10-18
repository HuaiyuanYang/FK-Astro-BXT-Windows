# -*- coding: utf-8 -*-
"""
star_metrics.py —— 星点形态指标（去卷积效果的直接量化）。

全图相关系数不是好指标 —— 它被大量的背景噪声主导，
0.5 左右很正常，看不出模型到底做了什么。

真正有意义的指标是**星点的形态**：
    * FWHM（半高全宽）—— 去卷积应该让它变小
    * 峰值/总能量比（集中度）—— 去卷积应该让它变大
    * 背景 RMS —— 不应显著恶化

本模块在整幅图上找亮星，逐个测 FWHM：
    1. 用 5x5 中值滤波后的图找局部极大值
    2. 取亮度最高的 N 颗，每颗周围取 21x21 窗口
    3. 在窗口里算径向平均轮廓，找半高处的半径 -> FWHM

用法:
    python star_metrics.py <输入图> <输出图> [--n 60]
    两个参数都支持 .png（16 位灰度）和 .fits
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_gray(path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".fits", ".fit", ".fts"):
        import fits_io
        d, _ = fits_io.read_fits(p)
        d = np.nan_to_num(d.astype(np.float64))
        return np.clip((d - d.min()) / max(d.max() - d.min(), 1e-9), 0, 1)
    from PIL import Image
    a = np.asarray(Image.open(p))
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float64) / 65535.0


def median2d(a, k=5):
    """小核中值滤波（纯 numpy，用 stride tricks 加速）"""
    from numpy.lib.stride_tricks import as_strided
    h, w = a.shape
    r = k // 2
    pad = np.pad(a, r, mode="reflect")
    shp = (h, w, k, k)
    st = (pad.strides[0], pad.strides[1], pad.strides[0], pad.strides[1])
    win = as_strided(pad, shape=shp, strides=st, writeable=False)
    return np.median(win, axis=(2, 3))


def find_stars(img, n=60, edge=24):
    """找最亮的 n 颗星的中心坐标"""
    h, w = img.shape
    med = median2d(img, 5)
    # 局部极大值：比 3x3 邻域都亮
    from numpy.lib.stride_tricks import as_strided
    pad = np.pad(med, 1, mode="reflect")
    shp = (h, w, 3, 3)
    st = (pad.strides[0], pad.strides[1], pad.strides[0], pad.strides[1])
    win = as_strided(pad, shape=shp, strides=st, writeable=False)
    is_max = med >= win.max(axis=(2, 3))
    is_max[:edge] = False
    is_max[-edge:] = False
    is_max[:, :edge] = False
    is_max[:, -edge:] = False
    ys, xs = np.nonzero(is_max)
    if len(ys) == 0:
        return np.zeros((0, 2), int)
    vals = med[ys, xs]
    order = np.argsort(vals)[::-1]
    # 去掉挨得太近的（避免同一颗星被选多次）
    picked = []
    taken = np.zeros((h, w), bool)
    for i in order:
        y, x = ys[i], xs[i]
        if taken[max(0, y - 6):y + 7, max(0, x - 6):x + 7].any():
            continue
        picked.append((y, x))
        taken[y, x] = True
        if len(picked) >= n:
            break
    return np.array(picked, int)


def radial_profile(img, y, x, rad=16, bg_sub=True):
    """算某点周围的径向平均轮廓（按 0.5 像素分箱）

    rad 取 16 而不是 10：去卷积会把星点峰值推高（可能到饱和），
    半高值随之变高，如果搜索半径太小就找不到半高交叉点，
    FWHM 会算成 nan —— 看着像"变糊"，其实是"变得更集中"。
    """
    y0, y1 = y - rad - 1, y + rad + 2
    x0, x1 = x - rad - 1, x + rad + 2
    h, w = img.shape
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    win = img[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(yy - y, xx - x)
    if bg_sub:
        # 用 r 在 [rad*0.7, rad] 的环形中位数当背景
        bgm = (r > rad * 0.7) & (r <= rad)
        if bgm.sum() > 10:
            win = win - np.median(win[bgm])
    nb = int(rad / 0.5) + 1
    prof = np.full(nb, np.nan)
    for b in range(nb):
        m = (r >= b * 0.5) & (r < (b + 1) * 0.5)
        if m.sum() >= 3:
            prof[b] = win[m].mean()
    return prof


def fwhm_from_profile(prof):
    """从径向轮廓算 FWHM（像素）"""
    if prof is None or not np.isfinite(prof).any():
        return None
    p = prof.copy()
    pk = np.nanmax(p)
    if pk <= 0:
        return None
    half = pk / 2.0
    ipk = int(np.nanargmax(p))
    rs = np.arange(len(p)) * 0.5
    tail = p[ipk:]
    below = np.where(tail < half)[0]
    if len(below) == 0:
        return None                       # 半径内没降到半高以下
    j = ipk + below[0]
    if j == 0:
        return None
    p1, p0 = p[j], p[j - 1]
    if p1 == p0 or not np.isfinite(p0) or not np.isfinite(p1):
        r_half = rs[j]
    else:
        r_half = rs[j - 1] + 0.5 * (half - p0) / (p1 - p0)
    return 2.0 * r_half


def concentration(img, y, x, rad=4, bg_sub=True):
    """集中度 = 半径 rad 内的通量 / 半径 3*rad 内的通量"""
    h, w = img.shape
    R = 3 * rad
    y0, y1 = y - R - 1, y + R + 2
    x0, x1 = x - R - 1, x + R + 2
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    win = img[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(yy - y, xx - x)
    if bg_sub:
        bgm = r > rad * 2.5
        if bgm.sum() > 10:
            win = win - np.median(win[bgm])
    win = np.maximum(win, 0)
    inner = win[r <= rad].sum()
    outer = win[r <= R].sum()
    return float(inner / outer) if outer > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-size", type=int, default=0)
    a = ap.parse_args()

    def prep(p):
        g = load_gray(p)
        if a.max_size and max(g.shape) > a.max_size:
            n = a.max_size
            y0 = (g.shape[0] - n) // 2
            x0 = (g.shape[1] - n) // 2
            g = g[y0:y0 + n, x0:x0 + n]
        return g

    A = prep(a.src)
    B = prep(a.dst)
    if A.shape != B.shape:
        raise SystemExit(f"尺寸不同: {A.shape} vs {B.shape}")

    print("=" * 72)
    print("星点形态指标")
    print("=" * 72)
    print(f"  输入: {Path(a.src).name}   {A.shape}")
    print(f"  输出: {Path(a.dst).name}")

    # 在**输入**图里找星（保证两图测的是同一批星）
    stars = find_stars(A, n=a.n)
    print(f"  找到 {len(stars)} 颗候选星")
    if len(stars) == 0:
        print("  [!] 没找到星，无法测量")
        return 1

    fw_a, fw_b, cc_a, cc_b, pk_a, pk_b = [], [], [], [], [], []
    skipped = 0
    for (y, x) in stars:
        pa = radial_profile(A, y, x)
        pb = radial_profile(B, y, x)
        fa = fwhm_from_profile(pa)
        fb = fwhm_from_profile(pb)
        if fa is None or fb is None:
            skipped += 1
            continue
        # 排除明显不是星点（FWHM 太大）和饱和的
        if fa > 10.0:
            skipped += 1
            continue
        if A[y, x] > 0.97:
            skipped += 1
            continue
        fw_a.append(fa); fw_b.append(fb)
        ca = concentration(A, y, x)
        cb = concentration(B, y, x)
        if ca is not None and cb is not None:
            cc_a.append(ca); cc_b.append(cb)
        pk_a.append(float(A[y, x])); pk_b.append(float(B[y, x]))

    if not fw_a:
        print(f"  [!] 没有可用的星点样本（跳过 {skipped}）")
        return 1

    fw_a = np.array(fw_a); fw_b = np.array(fw_b)
    print(f"  有效样本: {len(fw_a)} 颗（跳过 {skipped}）")
    print()
    print(f"  FWHM        : 输入 {fw_a.mean():.3f} ± {fw_a.std():.3f} px"
          f"  ->  输出 {fw_b.mean():.3f} ± {fw_b.std():.3f} px")
    ratio = fw_b.mean() / fw_a.mean()
    print(f"                比值 {ratio:.4f}"
          + ("   [OK] 星点变锐" if ratio < 0.97 else
             ("   [i] 基本不变" if ratio < 1.03 else "   [!] 变糊了")))
    # 逐星配对比较（更严格，排除选择偏差）
    shrink = fw_b < fw_a
    print(f"                逐星比较: {int(shrink.sum())}/{len(fw_a)} 颗变锐"
          f"  ({float(shrink.mean()):.1%})")
    if cc_a:
        cc_a = np.array(cc_a); cc_b = np.array(cc_b)
        print(f"  集中度      : 输入 {cc_a.mean():.4f}  ->  输出 {cc_b.mean():.4f}"
              f"   比值 {cc_b.mean()/cc_a.mean():.4f}"
              + ("   [OK]" if cc_b.mean() > cc_a.mean() else ""))
    pk_a = np.array(pk_a); pk_b = np.array(pk_b)
    print(f"  星点峰值    : 输入 {pk_a.mean():.5f}  ->  输出 {pk_b.mean():.5f}"
          f"   比值 {pk_b.mean()/pk_a.mean():.3f}")

    # 背景 RMS
    def bg_rms(g):
        k = 32
        hh = (g.shape[0] // k) * k
        ww = (g.shape[1] // k) * k
        blk = g[:hh, :ww].reshape(hh // k, k, ww // k, k).transpose(0, 2, 1, 3)
        blk = blk.reshape(-1, k * k)
        s = blk.std(axis=1)
        sel = blk[s <= np.percentile(s, 20)]
        return float(sel.std())
    ra, rb = bg_rms(A), bg_rms(B)
    print(f"  背景 RMS    : 输入 {ra:.6f}  ->  输出 {rb:.6f}"
          f"   比值 {rb/max(ra,1e-12):.3f}"
          + ("   [OK] 噪声未恶化" if rb / max(ra, 1e-12) < 1.5 else "   [!] 噪声升高"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
