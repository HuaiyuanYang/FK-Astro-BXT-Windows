#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fkv3.py —— FKv3 宿主处理方案(Python 实现)。

来源:https://github.com/wfzylcl878-creator/fk-astro
       src/FKv3Pipeline.h 与 docs/FKv3.md(逐行对齐)
       该方案据官方 RCAstroSuite-pxm.dylib(ARM64)静态分析恢复。

⚠️ 为什么要有这个文件:
    我们此前用的是 **FKv1**(log + MTF + 仿射,w=0.505*bg),
    而 FKv1 已被上游仓库移除。实测它对官方 BXT 的偏差达 4~9 倍。
    FKv3 才是当前口径,归一化是 **整图 arcsinh**。

三个组件:
    ArcsinhNormalizer —— 整图拟合一次, 输入 / 第六列 / 逆变换共用同一状态
    AxisGrid          —— 官方分块原点与余弦过渡带
    blend_tile        —— 双精度直接累加, 不除以权重和

关键约定(照抄上游):
  * Fit 四边各裁 int(size*0.05), 重采样长边 128, 取点 floor(idx*src/target)
  * low = (n-1)*0.001 线性插值分位; high 固定 1(来自 arcsinh,wp=1)
  * |high-low| < 0.001 时重置 low=0, high=1
  * 正样本映射到 (p-low)/(high-low) 后取**上中位数**(下标 n//2)
  * x = clamp(0.2/(m*(m+0.1)), 1, 1e6);  a = asinh(x);  gain = log10(x/a)
  * Forward(p) = asinh((p-low)/(high-low)*x)/a      —— **不 clamp、不裁负值**
  * Inverse(y) = low + sinh(y*a)/x*(high-low)
  * 越界块区域直接填模型域 0, 不是 Forward(0)
"""
import numpy as np

PI = 3.14159265358979323846


class ArcsinhNormalizer:
    """与上游 fka::fkv3::ArcsinhNormalizer 对应。"""

    def __init__(self):
        self.low = 0.0
        self.high = 1.0
        self.median = 0.0
        self.x = 1.0
        self.asinh_x = float(np.arcsinh(1.0))
        self.gain = float(np.log10(1.0 / np.arcsinh(1.0)))
        self.crop_x = self.crop_y = self.crop_w = self.crop_h = 0
        self.grid_w = self.grid_h = 0
        self.positive_samples = 0

    def fit(self, img, lunar=False):
        """img: (C,H,W) 或 (H,W)。整图拟合一次。"""
        a = np.asarray(img, dtype=np.float64)
        if a.ndim == 2:
            a = a[None, :, :]
        c, h, w = a.shape
        if w < 1 or h < 1 or c < 1:
            raise ValueError("FKv3 归一化要求非空图像")

        self.crop_x = int(w * 0.05)
        self.crop_y = int(h * 0.05)
        self.crop_w = w - 2 * self.crop_x
        self.crop_h = h - 2 * self.crop_y
        if self.crop_w < 1 or self.crop_h < 1:
            self.crop_x = self.crop_y = 0
            self.crop_w, self.crop_h = w, h

        scale = float(max(self.crop_w, self.crop_h)) / 128.0
        self.grid_w = max(1, int(round(self.crop_w / scale)))   # llround
        self.grid_h = max(1, int(round(self.crop_h / scale)))
        step_x = float(self.crop_w) / self.grid_w
        step_y = float(self.crop_h) / self.grid_h

        sx = np.minimum(self.crop_w - 1,
                        np.floor(np.arange(self.grid_w) * step_x).astype(np.int64))
        sy = np.minimum(self.crop_h - 1,
                        np.floor(np.arange(self.grid_h) * step_y).astype(np.int64))
        sx += self.crop_x
        sy += self.crop_y
        # (C, gridH, gridW)
        sub = a[:, sy[:, None], sx[None, :]]
        vals = sub.ravel()
        vals = vals[vals > 0.0]
        self.positive_samples = int(vals.size)

        if vals.size:
            vals.sort()
            q = float(vals.size - 1) * 0.001
            i = int(q)
            j = min(i + 1, vals.size - 1)
            # 与 std::fma(q-i, s[j]-s[i], s[i]) 等价
            self.low = (q - i) * (vals[j] - vals[i]) + vals[i]

        if abs(self.high - self.low) < 0.001:
            self.low, self.high = 0.0, 1.0

        inv_span = 1.0 / (self.high - self.low)
        if vals.size:
            z = (vals - self.low) * inv_span
            mid = z.size // 2
            # 上中位数: nth_element 后取下标 size/2
            self.median = float(np.partition(z, mid)[mid])
            if (not lunar) and self.median > 1.0e-6:
                x = 0.2 / (self.median * (self.median + 0.1) + 1.0e-12)
                self.x = float(min(max(x, 1.0), 1.0e6))

        self.asinh_x = float(np.arcsinh(self.x))
        self.gain = float(np.log10(self.x / self.asinh_x))
        return self

    def forward(self, values):
        v = np.asarray(values, dtype=np.float64)
        return np.arcsinh((v - self.low) * (1.0 / (self.high - self.low)) * self.x) / self.asinh_x

    def inverse(self, values):
        v = np.asarray(values, dtype=np.float64)
        return self.low + (np.sinh(v * self.asinh_x) / self.x) * (self.high - self.low)

    def parameters(self, stars=0.0, halos=0.0, diameter=0.0, auto_psf=True,
                   nonstellar=0.0, correct_only=False, lunar=False):
        """-> np.float32[6]。规则与上游 Settings::Effective() 一致。"""
        if correct_only:
            stars = halos = diameter = nonstellar = 0.0
            auto_psf = True
            lunar = False
        if lunar:
            stars = halos = 0.0
            auto_psf = False
        if auto_psf:
            diameter = 0.0
        return np.array([stars, halos, diameter * 0.5, 1.0 if auto_psf else 0.0,
                         nonstellar, self.gain], dtype=np.float32)

    def describe(self):
        return (f"low={self.low:.9g} high={self.high:.6g} median={self.median:.6g} "
                f"x={self.x:.6g} asinh_x={self.asinh_x:.6g} gain={self.gain:.6f} "
                f"crop=({self.crop_x},{self.crop_y},{self.crop_w},{self.crop_h}) "
                f"grid=({self.grid_w},{self.grid_h}) pos={self.positive_samples}")


def fit_image(hwc, lunar=False):
    """按 (H,W,C) 布局拟合 —— 这是 run_fkv3 / 各验证脚本使用的布局。

    存在的理由: ArcsinhNormalizer.fit() 按 (C,H,W) 解释数组, 两者形状
    (h,w,3) 与 (c,h,w) 在 h==w 时完全无法区分; 曾经因此把 (H,W,C) 直接
    喂给 fit(), 导致裁边算成 3x462 而不是 462x462。统一走这个入口。
    """
    a = np.asarray(hwc, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    return ArcsinhNormalizer().fit(np.ascontiguousarray(a.transpose(2, 0, 1)), lunar=lunar)


def _f32(v):
    return float(np.float32(v))


def _llround(v):
    """C 的 std::llround: 四舍五入, 恰好 .5 时远离零。

    不能用 np.rint / Python 的 round —— 那两个是"五取偶",
    在 pitch*k - (T-pitch)/2 恰好落在 .5 时会把原点算偏 1 像素。
    规格要求的是 llround。
    """
    return int(np.floor(v + 0.5)) if v >= 0 else int(np.ceil(v - 0.5))


class AxisGrid:
    """与上游 fka::fkv3::AxisGrid 对应。"""

    def __init__(self, size, tile, fraction):
        self.size = int(size)
        self.tile = int(tile)
        f = _f32(min(max(float(fraction), 0.0), 0.5))
        # overlap = 2 * floor(float32(tile) * f * 0.5f)
        self.overlap = 2 * int(np.floor(_f32(_f32(self.tile) * f) * _f32(0.5)))
        if size <= tile:
            count = 1
        else:
            count = int(np.ceil(float(size) / float(tile - self.overlap)))
        self.count = count
        self.pitch = _f32(float(size) / float(count))
        self._d = count

        origins = []
        for k in range(count):
            if size < tile:
                origin = int((size - tile) / 2)     # C++ 整数除法向零截断
            elif k == 0:
                origin = 0
            elif k == count - 1:
                origin = size - tile
            else:
                a = _f32(self.pitch * _f32(k))
                b = _f32(_f32(tile) - self.pitch)
                origin = int(_llround(float(a) - 0.5 * float(b)))
            origins.append(int(origin))
        self.origins = origins

    def window(self, index):
        """-> np.float64[tile], 余弦过渡带 + 带外置零。"""
        t = self.tile
        w = np.ones(t, dtype=np.float64)
        ov = self.overlap
        origin = self.origins[index]

        def falling(j):
            return (np.cos((j + 1) * PI / (ov + 1)) + 1.0) * 0.5

        if index > 0:
            left = int(_llround(float(_f32(self.pitch * _f32(index))))) - origin - ov // 2
            for i in range(t):
                if i < left:
                    w[i] = 0.0
                elif i < left + ov:
                    w[i] *= 1.0 - falling(i - left)
        if index + 1 < len(self.origins):
            right = int(_llround(float(_f32(self.pitch * _f32(index + 1))))) - origin - ov // 2
            for i in range(t):
                if i >= right + ov:
                    w[i] = 0.0
                elif i >= right:
                    w[i] *= falling(i - right)
        return w

    def describe(self):
        return f"tile={self.tile} overlap={self.overlap} count={self.count} pitch={self.pitch} origins={self.origins}"


def run_fkv3(img, predict, normalizer, tile_out=512, overlap=0.2,
             tile_in=1024, lunar=False, want_diag=False):
    """完整 FKv3 宿主流程。

    参数
    ----
    img       : (H,W,C) 或 (H,W) 原始像素(float, 浮点图直接使用)
    predict   : fn(float32[1,tile_in,tile_in,C]) -> float32[1,tile_out,tile_out,3]
    normalizer: 已经 fit() 过的 ArcsinhNormalizer(或 None -> 现算)

    流程(严格照上游):
      1. 整图 Fit 一次 -> 得到 low/high/x/a/gain
      2. **整图执行一次 Forward**, 越界窗区域填模型域 0(不是 Forward(0))
      3. 逐块 predict, 用余弦带权重加权, **double 直接累加, 不除以权重和**
      4. 全图融合后**只做一次 Inverse**
      5. clamp 到 [0,1]
    """
    a = np.asarray(img, dtype=np.float64)
    mono = (a.ndim == 2) or (a.ndim == 3 and a.shape[2] == 1)
    if a.ndim == 2:
        a = a[:, :, None]
    H, W, C = a.shape

    if normalizer is None:
        normalizer = fit_image(a, lunar=lunar)

    # 2. 整图 Forward
    fwd = normalizer.forward(a)                      # (H,W,C) double
    # 单色图复制到三个模型输入通道
    if C == 1:
        fwd = np.repeat(fwd, 3, axis=2)
        feed = fwd
    else:
        feed = fwd

    Cfeed = feed.shape[2]
    gx = AxisGrid(W, tile_out, overlap)
    gy = AxisGrid(H, tile_out, overlap)
    acc = np.zeros((H, W, 3), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)

    wx_cache = {i: gx.window(i) for i in range(gx.count)}
    wy_cache = {j: gy.window(j) for j in range(gy.count)}
    ctx = (tile_in - tile_out) // 2

    for j in range(gy.count):
        y0 = gy.origins[j]
        wy = wy_cache[j]
        sy0, sy1 = max(0, y0), min(H, y0 + tile_out)
        cy0 = y0 - ctx
        iy0, iy1 = max(0, cy0), min(H, cy0 + tile_in)
        for i in range(gx.count):
            x0 = gx.origins[i]
            wx = wx_cache[i]
            sx0, sx1 = max(0, x0), min(W, x0 + tile_out)
            cx0 = x0 - ctx
            ix0, ix1 = max(0, cx0), min(W, cx0 + tile_in)

            win = np.zeros((tile_in, tile_in, Cfeed), dtype=np.float64)   # 越界 = 模型域 0
            if iy1 > iy0 and ix1 > ix0:
                win[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0, :] = \
                    feed[iy0:iy1, ix0:ix1, :]

            out = predict(np.ascontiguousarray(win[None, ...], dtype=np.float32))[0]
            out = np.asarray(out, dtype=np.float64)                      # (512,512,3)

            oy = slice(sy0 - y0, sy0 - y0 + (sy1 - sy0))
            ox = slice(sx0 - x0, sx0 - x0 + (sx1 - sx0))
            wgt = wy[sy0 - y0:sy0 - y0 + (sy1 - sy0), None] * \
                  wx[None, sx0 - x0:sx0 - x0 + (sx1 - sx0)]
            acc[sy0:sy1, sx0:sx1, :] += out[oy, ox, :] * wgt[:, :, None]
            wsum[sy0:sy1, sx0:sx1] += wgt

    # 4. 融合后整图逆变换一次(不除权重和)
    if mono:
        pair = acc[:, :, 0] + acc[:, :, 1]
        third = 1.0 / 3.0
        merged = pair * third + acc[:, :, 2] * third
        res = normalizer.inverse(merged)
    else:
        res = normalizer.inverse(acc)

    # 5. clamp
    res = np.clip(res, 0.0, 1.0)
    if want_diag:
        return res, {"gain": normalizer.gain, "x": normalizer.x, "low": normalizer.low,
                     "high": normalizer.high, "tiles": (gx.count, gy.count),
                     "wsum_min": float(wsum.min()), "wsum_max": float(wsum.max())}
    return res
