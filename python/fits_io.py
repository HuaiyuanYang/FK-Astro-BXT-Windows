# -*- coding: utf-8 -*-
"""
fits_io.py —— 纯标准库的 FITS 读取（不依赖 astropy）。

FITS 结构很简单，不需要装 astropy：
    头：若干个 2880 字节的块，每块 36 张 80 字节的卡片
        卡片格式：KEYWORD = VALUE / COMMENT
        END 卡片结束
    数据：紧跟头部，按 BITPIX 定宽，补齐到 2880 字节的整数倍
        BITPIX  8=uint8 16=int16 32=int32 -32=float32 -64=float64
        维度：NAXIS 后面的 NAXIS1..NAXISn，**FITS 是列优先（Fortran 序）**，
              所以 reshape 之后要转置才能得到常规的 (rows, cols)

用法:
    python fits_io.py <file.fits>            # 打印头部摘要和数据统计
"""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BITPIX_DTYPE = {8: ">u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}
BLOCK = 2880


def read_header(fh):
    """读 FITS 头 -> dict（值是 python 原生类型）"""
    hdr = {}
    while True:
        raw = fh.read(BLOCK)
        if len(raw) < BLOCK:
            break
        for i in range(0, BLOCK, 80):
            card = raw[i:i + 80]
            key = card[:8].decode("ascii", "replace").strip()
            if key == "END":
                return hdr
            if not key or key == "COMMENT" or key == "HISTORY":
                continue
            # VALUE 段：跳过 "= "
            val = card[10:].split(b"/")[0].decode("ascii", "replace").strip()
            if val.startswith("'"):
                v = val.strip("'").rstrip()
            else:
                try:
                    v = int(val)
                except ValueError:
                    try:
                        v = float(val)
                    except ValueError:
                        v = val
            if key in hdr:                       # 重复键合成列表
                if not isinstance(hdr[key], list):
                    hdr[key] = [hdr[key]]
                hdr[key].append(v)
            else:
                hdr[key] = v
    return hdr


def read_fits(path):
    """-> (data ndarray, header dict)

    data 的形状：一维数据给 (ny,)，二维给 (ny, nx)，三维给 (nz, ny, nx)
    （已从 FITS 的列优先序转成常规行优先）
    """
    p = Path(path)
    with open(p, "rb") as fh:
        hdr = read_header(fh)
        bitpix = int(hdr.get("BITPIX", 16))
        naxis = int(hdr.get("NAXIS", 0))
        dims = [int(hdr.get(f"NAXIS{i}", 0)) for i in range(1, naxis + 1)]
        if "END" not in hdr and naxis == 0:
            raise ValueError("不是有效的 FITS（没有 END 卡片）")

        dtype = BITPIX_DTYPE.get(bitpix)
        if dtype is None:
            raise ValueError(f"不支持的 BITPIX: {bitpix}")

        count = int(np.prod(dims)) if dims else 0
        if count == 0:
            return np.zeros(0), hdr
        buf = fh.read(count * np.dtype(dtype).itemsize)

    arr = np.frombuffer(buf, dtype=dtype)
    # FITS 是列优先：NAXIS1 是最快变化的轴（列）
    shape_fortran = tuple(dims)
    arr = arr.reshape(shape_fortran, order="F")
    if len(dims) >= 2:
        arr = arr.T                                # -> (ny, nx ...)

    bscale = float(hdr.get("BSCALE", 1.0))
    bzero = float(hdr.get("BZERO", 0.0))
    if bscale != 1.0 or bzero != 0.0:
        arr = arr.astype(np.float32) * bscale + bzero
    return arr, hdr


def describe(path):
    arr, hdr = read_fits(path)
    p = Path(path)
    print("=" * 76)
    print(f"文件: {p.name}   ({p.stat().st_size/1048576:.1f} MB)")
    print("=" * 76)

    keys = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
            "BSCALE", "BZERO", "EXPTIME", "EXPOSURE", "CCD-TEMP", "GAIN",
            "OFFSET", "XBINNING", "YBINNING", "INSTRUME", "TELESCOP",
            "OBJECT", "FILTER", "DATE-OBS", "IMAGETYP", "BAYERPAT",
            "ROWORDER", "PEDESTAL", "STACKCNT"]
    print("  头部关键字段:")
    for k in keys:
        if k in hdr:
            print(f"    {k:<10} = {hdr[k]}")

    extra = [k for k in hdr if k not in keys and
             k not in ("COMMENT", "HISTORY", "END")]
    other = [k for k in extra
             if any(t in k for t in ("TEMP", "GAIN", "OFF", "EXPO", "OBJ",
                                     "DATE", "FILT", "BIN", "PIX", "SAT",
                                     "STACK", "CAMERA", "SITE", "TELE"))]
    if other:
        print("  其他相关字段:")
        for k in sorted(other)[:25]:
            print(f"    {k:<14} = {hdr[k]}")

    print()
    print(f"  数据: shape={arr.shape} dtype={arr.dtype}")
    a = arr.astype(np.float64)
    print(f"    min={np.nanmin(a):.6g}  max={np.nanmax(a):.6g}")
    print(f"    mean={np.nanmean(a):.6g}  median={np.nanmedian(a):.6g}")
    pos = a[a > 0]
    if pos.size:
        print(f"    正样本 P35 = {np.percentile(pos, 35):.6g}   "
              f"<- 归一化要用的背景 bg")
    print(f"    0 值比例 = {float(np.mean(a == 0)):.4%}   "
          f"负值比例 = {float(np.mean(a < 0)):.4%}")
    sat = int(hdr.get("SATLEVEL", 65535))
    print(f"    饱和(>={sat}) 比例 = {float(np.mean(a >= sat)):.6%}")
    return arr, hdr


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    describe(sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
