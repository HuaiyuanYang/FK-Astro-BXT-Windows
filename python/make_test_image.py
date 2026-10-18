# -*- coding: utf-8 -*-
"""
make_test_image.py —— 生成一张"像天文图"的合成测试图。

为什么需要：
    之前用平滑渐变图（0.15~0.85）测端到端，那不是天文图 ——
    对 FKv1 归一化来说等于"整幅图都极亮"，会 83% 饱和。
    归一化在**背景 ~0.02** 的真实量级上才是有意义的测试。

合成模型（大致模仿校准后的线性天文帧）：
    背景      ~0.02 (带轻微梯度)
    噪声      ~0.0015 高斯
    星点      多个，不同亮度，高斯轮廓（sigma 1.5~3 px）
    几个亮星  峰值接近 1.0
    一点暗结构  模拟星云/星系

用法:
    python make_test_image.py [输出路径] [尺寸]
"""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def make_starfield(H=1024, W=1024, seed=42, bg=0.02, noise=0.0015):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    # 背景：轻微梯度 + 噪声
    img = bg * (1.0 + 0.06 * (xx / W) + 0.04 * (yy / H))

    # 一个弥散的暗结构（模拟星系/星云）
    img += 0.012 * np.exp(-(((yy - 0.6 * H) ** 2 + (xx - 0.45 * W) ** 2)
                            / (2 * (0.12 * H) ** 2)))
    img += 0.006 * np.exp(-(((yy - 0.3 * H) ** 2 + (xx - 0.7 * W) ** 2)
                            / (2 * (0.09 * H) ** 2)))

    # 星点：幂律分布的亮度，高斯轮廓
    n_stars = 220
    amps = (0.8 * rng.power(2.6, n_stars)).astype(np.float32)
    amps[:6] = [0.95, 0.8, 0.7, 0.55, 0.45, 0.35]     # 几颗亮星
    for i in range(n_stars):
        cy = rng.uniform(8, H - 8)
        cx = rng.uniform(8, W - 8)
        sig = rng.uniform(1.2, 3.0)
        a = float(amps[i])
        y0, y1 = max(0, int(cy - 5 * sig)), min(H, int(cy + 5 * sig) + 1)
        x0, x1 = max(0, int(cx - 5 * sig)), min(W, int(cx + 5 * sig) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        sy = yy[y0:y1, x0:x1] - cy
        sx = xx[y0:y1, x0:x1] - cx
        img[y0:y1, x0:x1] += a * np.exp(-(sy ** 2 + sx ** 2) / (2 * sig ** 2))

    # 噪声（泊松近似：亮处噪声大一点）
    img += rng.normal(0, 1.0, (H, W)) * (noise + 0.01 * np.sqrt(np.maximum(img, 0)))
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return np.repeat(img[:, :, None], 3, axis=2)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "test_astro.png")
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    img = make_starfield(size, size)
    print(f"合成天文图 {size}x{size}x3")
    print(f"  值域      [{img.min():.5f}, {img.max():.5f}]")
    print(f"  均值      {img.mean():.5f}")
    print(f"  中位数    {float(np.median(img)):.5f}")
    print(f"  正样本P35 {float(np.percentile(img[img > 0], 35)):.5f}   <- 归一化用的背景")
    print(f"  99.9 分位 {float(np.percentile(img, 99.9)):.5f}")
    print(f"  饱和(>=1) 的比例 {float(np.mean(img >= 1.0)):.5%}")
    try:
        from PIL import Image
        out.parent.mkdir(parents=True, exist_ok=True)
        # 存成 16 位 PNG 以保留动态范围
        arr16 = np.clip(img[:, :, 0] * 65535.0, 0, 65535).astype(np.uint16)
        Image.fromarray(arr16, mode="I;16").save(out)
        print(f"  已写出: {out}  ({out.stat().st_size/1024:.0f} KB, 16 位灰度)")
    except ImportError:
        print("  (没有 PIL，未写文件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
