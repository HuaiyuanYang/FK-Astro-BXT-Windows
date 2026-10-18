# -*- coding: utf-8 -*-
"""
xisf_io.py —— 纯标准库 + numpy 的 XISF 读取(不需要 PixInsight)。

为什么写这个:
    PixInsight 的原生格式是 .xisf。要在不启动 PI 的前提下处理它, 就得自己读。
    XISF 结构本身很简单, 不需要任何第三方库:

        offset 0    8 字节  签名 "XISF0100"
        offset 8    4 字节  头部长度(uint32, 小端)
        offset 12   N 字节  XML 头部(UTF-8)
        之后        ...     数据块(由 <Image location="attachment:off:len"> 指定)

关键事实(来自本机 XISF 1.0 规范文档, 已逐字核对):
    * "For Image elements without a pixelStorage attribute, the **default planar**
      pixel storage model shall be assumed."
      -> 通道顺序是 (C, H, W) 平面, 与 bxt_bridge.py 的契约一致, 无需转置
    * sampleFormat="Float32" 的数据是小端 IEEE754 单精度
    * bounds 属性给出样本的标称区间(如 "0:1"), 即 PI 内部的 [0,1] 浮点空间

支持范围:
    * sampleFormat: Float32 / Float64 / UInt8 / UInt16 / UInt32
    * compression: 无压缩; 以及 zlib/deflate(标准库可直接解)
      其余(lz4/lz4hc/zstd)会抛出明确错误而不是给出错误数据
    * location="attachment:<offset>:<length>"

用法:
    import xisf_io
    img, meta = xisf_io.read_image("master.xisf")   # img: (C,H,W) float32
"""
import xml.etree.ElementTree as ET
import zlib

import numpy as np

XISF_MAGIC = b"XISF0100"
_SIG_LEN = 8
_LEN_LEN = 4
_RESERVED = 4        # 长度字段之后还有 4 字节保留区(必须为 0), XML 从偏移 16 才开始

# XISF sampleFormat -> numpy dtype(小端)
SAMPLE_DTYPES = {
    "UInt8":   "<u1",
    "UInt16":  "<u2",
    "UInt32":  "<u4",
    "UInt64":  "<u8",
    "Float32": "<f4",
    "Float64": "<f8",
}

# 这些压缩算法标准库解不了 —— 明确报错, 绝不返回可能错误的数据
_UNSUPPORTED_CODECS = {"lz4", "lz4hc", "zstd", "zstd1"}


class XisfError(Exception):
    pass


def _strip_ns(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find(elem, name):
    for c in elem.iter():
        if _strip_ns(c.tag) == name:
            return c
    return None


def read_header(path):
    """只读头部(不解数据)-> (xml_root, header_len, data_start)"""
    with open(path, "rb") as fh:
        sig = fh.read(_SIG_LEN)
        if sig != XISF_MAGIC:
            raise XisfError(
                f"不是 XISF 文件(签名 {sig!r}, 期望 {XISF_MAGIC!r})")
        raw = fh.read(_LEN_LEN)
        if len(raw) != _LEN_LEN:
            raise XisfError("文件过短, 读不到头部长度")
        hlen = int(np.frombuffer(raw, dtype="<u4")[0])
        # 偏移 12 起是 4 字节保留区, XML 实际从偏移 16 开始。
        # 实测校验: 本机那张 master 的长度字段 = 458546, 而 XML 结束于 458562,
        # 458562 - 16 = 458546, 与长度字段逐字节吻合。
        fh.seek(_SIG_LEN + _LEN_LEN + _RESERVED)
        hb = fh.read(hlen)
        if len(hb) != hlen:
            raise XisfError(f"头部不完整: 期望 {hlen} 字节, 实得 {len(hb)}")
    try:
        root = ET.fromstring(hb.decode("utf-8"))
    except Exception as e:
        raise XisfError(f"头部 XML 解析失败: {e}") from e
    return root, hlen, _SIG_LEN + _LEN_LEN + _RESERVED + hlen


def _props(elem):
    """收集 FITSKeyword / Property 之类的元数据 -> dict"""
    out = {}
    if elem is None:
        return out
    for c in elem.iter():
        t = _strip_ns(c.tag)
        if t == "FITSKeyword":
            out[c.get("name", "")] = c.get("value", "")
        elif t == "Property":
            out[c.get("id", c.get("name", ""))] = c.get("value", "")
    return out


def read_image(path, dtype=np.float32):
    """读第一个 <Image> -> (data (C,H,W) dtype, meta dict)

    meta 含: geometry, sampleFormat, bounds, colorSpace, imageType,
             location, compression, pixelStorage, keywords
    """
    root, _hlen, _dstart = read_header(path)
    img = _find(root, "Image")
    if img is None:
        raise XisfError("头部里没有 <Image> 元素")

    geom = img.get("geometry", "")
    try:
        w, h, c = (int(v) for v in geom.split(":"))
    except Exception as e:
        raise XisfError(f"geometry 解析失败: {geom!r}") from e

    sf = img.get("sampleFormat", "Float32")
    if sf not in SAMPLE_DTYPES:
        raise XisfError(f"不支持的 sampleFormat: {sf!r}")
    npdt = SAMPLE_DTYPES[sf]

    loc = img.get("location", "")
    if not loc.startswith("attachment:"):
        raise XisfError(
            f"只支持 attachment 形式的数据块, 实得 {loc!r}"
            "(inline / 外部文件引用未实现)")
    parts = loc.split(":")
    if len(parts) != 3:
        raise XisfError(f"location 格式异常: {loc!r}")
    off, ln = int(parts[1]), int(parts[2])

    comp = (img.get("compression") or "").strip().lower()
    codec = comp.split(":", 1)[0] if comp else ""
    if codec in _UNSUPPORTED_CODECS:
        raise XisfError(
            f"该文件用了 {codec} 压缩, Python 标准库解不了。"
            "请先用 PixInsight 另存为无压缩 XISF, 或转成 FITS。")

    expect = w * h * c * np.dtype(npdt).itemsize
    if codec in ("", "none"):
        if ln != expect:
            raise XisfError(
                f"未压缩但长度不符: attachment {ln} != 期望 {expect}"
                f"({w}x{h}x{c} {sf})")
        with open(path, "rb") as fh:
            fh.seek(off)
            buf = fh.read(ln)
        if len(buf) != ln:
            raise XisfError(f"数据块不完整: 期望 {ln}, 实得 {len(buf)}")
        flat = np.frombuffer(buf, dtype=npdt)
        real = (w, h, c)
    elif codec in ("zlib", "deflate"):
        with open(path, "rb") as fh:
            fh.seek(off)
            buf = fh.read(ln)
        try:
            raw = zlib.decompress(buf)
        except Exception as e:
            raise XisfError(f"zlib 解压失败: {e}") from e
        if len(raw) != expect:
            raise XisfError(
                f"解压后长度不符: {len(raw)} != 期望 {expect}")
        flat = np.frombuffer(raw, dtype=npdt)
        real = (w, h, c)
    else:
        raise XisfError(f"未支持的压缩编码: {codec!r}")

    # XISF 默认 planar: 每通道连续 -> 直接 reshape 成 (C,H,W)
    data = flat.reshape((c, h, w)).astype(dtype, copy=False)

    meta = {
        "geometry": geom, "width": w, "height": h, "channels": c,
        "sampleFormat": sf, "bounds": img.get("bounds", ""),
        "colorSpace": img.get("colorSpace", ""),
        "imageType": img.get("imageType", ""),
        "location": loc, "compression": comp or "none",
        "pixelStorage": img.get("pixelStorage", "planar(默认)"),
        "id": img.get("id", ""),
        "keywords": _props(img),
    }
    return data, meta


def describe(path):
    data, meta = read_image(path)
    print("=" * 76)
    print(f"XISF: {path}")
    print("=" * 76)
    print(f"  几何      {meta['geometry']}  (W={meta['width']} H={meta['height']} C={meta['channels']})")
    print(f"  样本格式  {meta['sampleFormat']}   存储模型 {meta['pixelStorage']}")
    print(f"  色彩空间  {meta['colorSpace']}   类型 {meta['imageType']}")
    print(f"  标称区间  bounds={meta['bounds']}")
    print(f"  压缩      {meta['compression']}")
    print(f"  实际形状  {data.shape}  {data.dtype}   {data.nbytes/1048576:.1f} MB")
    for i in range(data.shape[0]):
        p = data[i]
        print(f"    通道 {i}: 最小值 {p.min():.6g}  最大值 {p.max():.6g}  "
              f"均值 {p.mean():.6g}  中位 {np.median(p):.6g}")
    kw = meta["keywords"]
    for k in ("OBJECT", "INSTRUME", "TELESCOP", "EXPTIME", "DATE-OBS", "DATE-END",
              "BAYERPAT", "EGAIN", "FOCALLEN"):
        if k in kw:
            print(f"    {k:<10} {kw[k]}")
    return data, meta


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    describe(sys.argv[1])
