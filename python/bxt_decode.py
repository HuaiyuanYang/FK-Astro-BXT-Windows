# -*- coding: utf-8 -*-
"""
bxt_decode.py —— 把 .mlpackage 里的 MIL 图 + weight.bin 权重完整解出来。

这是自研 MIL -> ONNX 翻译器的第一步（也是地基）。
coremltools 没有 CoreML -> ONNX 的能力，所以只能自己写。

weight.bin 布局（逐字节实测破译，286 个张量全部走通）：
    [0:64]                    全局头
                                 u32 @0  = 张量个数
                                 u32 @4  = 版本 (2)
    然后每个张量:
        [pos    : pos+64]     64 字节块头
                                 +0  u32  magic = 0xDEADBEEF
                                 +4  u32  version = 2
                                 +8  u32  dataSize   <- 真实数据字节数
                                 +12 u32  reserved / flags
                                 +16 u32  allocSize  <- 分配大小（不是数据大小）
                                 +20 u32  reserved
                                 +24 u32  校验值 (类 FNV-1a)
                                 +28 u32  校验值
                                 +32 u32  校验值
                                 +36 u32  校验值
        [pos+64 : pos+64+dataSize]   裸张量数据 (FLOAT16 / FLOAT32 ...)
        然后补齐到下一个 64 字节边界

    实测：286 个张量恰好走到文件末尾（最后一块的补齐会越过 EOF，属正常）。

用法:
    python bxt_decode.py <mlpackage 目录>                # 概览 + 校验
    python bxt_decode.py <mlpackage 目录> --dump 5       # 附前 5 个张量的数值统计
"""
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from opscan import parse_fields, first_bytes, as_text   # noqa: E402

# ---------------------------------------------------------------------------
# MIL DataType —— 逐条抄自 apple/coremltools mlmodel/format/MIL.proto
# ---------------------------------------------------------------------------
MIL_DTYPE = {
    0: "UNUSED_TYPE", 1: "BOOL", 2: "STRING",
    10: "FLOAT16", 11: "FLOAT32", 12: "FLOAT64", 13: "BFLOAT16",
    21: "INT8", 22: "INT16", 23: "INT32", 24: "INT64", 25: "INT4",
    31: "UINT8", 32: "UINT16", 33: "UINT32", 34: "UINT64",
    35: "UINT4", 36: "UINT2", 37: "UINT1", 38: "UINT6", 39: "UINT3",
    40: "FLOAT8E4M3FN", 41: "FLOAT8E5M2",
}
NP_OF = {
    "FLOAT16": "float16", "FLOAT32": "float32", "FLOAT64": "float64",
    "INT8": "int8", "INT16": "int16", "INT32": "int32", "INT64": "int64",
    "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32",
    "UINT64": "uint64", "BOOL": "bool",
}
ITEMSIZE = {
    "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8, "bool": 1,
}

W_MAGIC = 0xDEADBEEF
W_HDR = 64


def align64(x):
    return (x + 63) & ~63


# ---------------------------------------------------------------------------
# weight.bin
# ---------------------------------------------------------------------------
def read_weight_index(raw):
    """-> [(block_pos, data_size, data_start), ...]，按文件顺序"""
    if len(raw) < W_HDR:
        raise ValueError("weight.bin 太小")
    n = struct.unpack_from("<I", raw, 0)[0]
    out = []
    pos = W_HDR
    for i in range(n):
        if pos + W_HDR > len(raw):
            raise ValueError(f"第 {i} 个张量越界: pos={pos} size={len(raw)}")
        magic, ver, dsize = struct.unpack_from("<3I", raw, pos)
        if magic != W_MAGIC:
            raise ValueError(f"第 {i} 个张量魔数错误 @{pos}: 0x{magic:08X}")
        out.append((pos, dsize, pos + W_HDR))
        pos = align64(pos + W_HDR + dsize)
    return out


# ---------------------------------------------------------------------------
# MIL 类型
# ---------------------------------------------------------------------------
def parse_dimension(blob):
    for f, wt, v in parse_fields(blob):
        if f == 1 and wt == 2:
            for g, gwt, gv in parse_fields(v):
                if g == 1:
                    return int(gv)
        elif f == 2 and wt == 2:
            return -1
    return -1


def parse_tensor_type(blob):
    dtype = None
    dims = []
    for f, wt, v in parse_fields(blob):
        if f == 1 and wt == 0:
            dtype = MIL_DTYPE.get(int(v), f"enum{int(v)}")
        elif f == 3 and wt == 2:
            dims.append(parse_dimension(v))
    return dtype, dims


def parse_value_type(blob):
    tt = first_bytes(parse_fields(blob), 1)
    if tt is None:
        return None, []
    return parse_tensor_type(tt)


def parse_value(blob):
    """MIL Value -> dict"""
    out = {"dtype": None, "dims": [], "offset": None, "filename": None}
    for f, wt, v in parse_fields(blob):
        if f == 2 and wt == 2:
            out["dtype"], out["dims"] = parse_value_type(v)
        elif f == 5 and wt == 2:
            for g, gwt, gv in parse_fields(v):
                if g == 1 and gwt == 2:
                    out["filename"] = as_text(gv)
                elif g == 2:
                    out["offset"] = int(gv)
    return out


# ---------------------------------------------------------------------------
# 把 const 算子（= 权重）按 offset 收集起来
# ---------------------------------------------------------------------------
def find_blobfile(blob, depth=0, max_depth=24):
    """递归找 BlobFileValue。

    结构链（实测）：
        Operation.f5  = 单个 map entry { f1:'name'/'val', f2:Value }
        Value.f2      = ValueType   -> f1 TensorType -> f1 dataType, f3 dims
        Value.f3      = ImmediateValue（小常量内联）
        Value.f5      = BlobFileValue { f1: fileName, f2: offset }

    踩过的坑：一开始把 Operation.f5 当成"map 本身"按普通消息遍历，
    结果 g==2 永远匹配不到，命中 0。实际上它是**单个 entry**，
    key 在 f1、value 在 f2。
    """
    hits = []
    if depth > max_depth:
        return hits
    for f, wt, v in parse_fields(blob):
        if wt != 2:
            continue
        if f == 5:
            fn = None
            off = None
            for g, gwt, gv in parse_fields(v):
                if g == 1 and gwt == 2:
                    fn = as_text(gv)
                elif g == 2 and gwt == 0:
                    off = gv
            if fn and fn.endswith(".bin") and off is not None:
                hits.append((fn, int(off)))
                continue
        hits.extend(find_blobfile(v, depth + 1, max_depth))
    return hits


def collect_consts(raw_model):
    """-> [(offset, dtype, dims, const_name), ...]（286 条，offset 唯一）"""
    top = parse_fields(raw_model)
    ml = first_bytes(top, 502)
    if ml is None:
        raise ValueError("顶层没有 f502 (MLProgram)")
    funcmap = first_bytes(parse_fields(ml), 2)
    fblob = None
    for f, wt, v in parse_fields(funcmap):
        if f == 2 and wt == 2:
            fblob = v
    blockmap = first_bytes(parse_fields(fblob), 3)
    block = None
    for f, wt, v in parse_fields(blockmap):
        if f == 2 and wt == 2:
            block = v

    found = []

    def deep_types(blob, out, depth=0):
        """收集子树里所有的 (dtype, dims)"""
        if depth > 24:
            return
        for f, wt, v in parse_fields(blob):
            if wt != 2:
                continue
            for g, gwt, gv in parse_fields(v):
                if g == 2 and gwt == 2:
                    dt, dm = parse_value_type(gv)
                    if dt or dm:
                        out.append((dt, dm))
            deep_types(v, out, depth + 1)

    for f, wt, v in parse_fields(block):
        if f != 3 or wt != 2:
            continue
        opf = parse_fields(v)
        if as_text(first_bytes(opf, 1)) != "const":
            continue
        # 输出名 = Operation.f3 -> NamedValueType.f1
        outname = "?"
        o3 = first_bytes(opf, 3)
        if o3:
            nm = first_bytes(parse_fields(o3), 1)
            if nm:
                outname = as_text(nm) or "?"
        refs = find_blobfile(v)
        if not refs:
            continue
        types = []
        deep_types(v, types)
        dt, dm = (types[0] if types else (None, []))
        for _fn, off in refs:
            found.append((off, dt, dm, outname))
    return found


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump_n = 0
    if "--dump" in sys.argv:
        i = sys.argv.index("--dump")
        if i + 1 < len(sys.argv):
            dump_n = int(sys.argv[i + 1])

    p = Path(args[0]) if args else Path("../mlpackage")
    if p.is_dir():
        c = sorted(p.rglob("model.mlmodel"))
        if not c:
            print(f"[X] {p} 里找不到 model.mlmodel")
            return 1
        p = c[0]
    pkg = p.parent.parent.parent
    wbpath = pkg / "Data/com.apple.CoreML/weights/weight.bin"

    print("=" * 78)
    print(f"包目录: {pkg.name}")
    print("=" * 78)

    wraw = wbpath.read_bytes()
    index = read_weight_index(wraw)
    print(f"  weight.bin      {len(wraw):,} 字节")
    print(f"  解析出张量       {len(index)} 个")

    consts = collect_consts(p.read_bytes())
    print(f"  MIL 里 const 引用 {len(consts)} 个")

    # offset -> 在文件中的位置
    by_off = {blk: (dsz, dstart) for blk, dsz, dstart in index}

    # 用 shape*dtype 独立校验 dataSize
    print()
    print("  --- 校验：形状 x dtype 算出的字节数 vs 元数据 dataSize ---")
    checked = mismatch = 0
    dtype_hist = Counter()
    for off, dt, dims, nm in consts:
        dtype_hist[dt] += 1
        if off not in by_off:
            continue
        dsz, dstart = by_off[off]
        if dt not in NP_OF:
            continue
        pos_dims = [d for d in dims if d and d > 0]
        if not pos_dims:
            continue
        n = 1
        for d in pos_dims:
            n *= d
        want = n * ITEMSIZE[NP_OF[dt]]
        checked += 1
        if want != dsz:
            mismatch += 1
            if mismatch <= 10:
                print(f"    [差异] {nm:<28} {dt:<8} {dims} 期望 {want:>9} 实际 {dsz:>9}")
    print(f"    检查 {checked} 个，完全吻合 {checked-mismatch} 个，"
          f"不吻合 {mismatch} 个")

    print()
    print(f"  --- const 的 dtype 分布 ---")
    for k, v in dtype_hist.most_common():
        print(f"    {str(k):<12} x{v}")

    print()
    print(f"  --- 形状示例（前 15 个能算出形状的）---")
    shown = 0
    for off, dt, dims, nm in consts:
        if not dims or shown >= 15:
            continue
        print(f"    {nm:<30} {str(dt):<8} {dims}")
        shown += 1

    if dump_n:
        print()
        print(f"  --- 前 {dump_n} 个张量的数值统计 ---")
        try:
            import numpy as np
        except ImportError:
            print("    (需要 numpy)")
            return 0
        for off, dt, dims, nm in consts[:dump_n]:
            if off not in by_off or dt not in NP_OF:
                continue
            dsz, dstart = by_off[off]
            buf = wraw[dstart:dstart + dsz]
            arr = np.frombuffer(buf, dtype=NP_OF[dt])
            print(f"    {nm:<30} {NP_OF[dt]:<8} n={arr.size:<8} "
                  f"min={arr.min():+.5f} max={arr.max():+.5f} "
                  f"mean={arr.mean():+.5f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
