# -*- coding: utf-8 -*-
"""
weight_read.py —— 从 .mlpackage 的 weight.bin 里把权重张量真正读出来（纯标准库）。

这一步为什么重要：
    CoreML 从来没有 "CoreML -> ONNX" 的转换器（coremltools 的 onnx 子模块
    是 ONNX->CoreML 方向，而且 6.0 就删掉了）。所以要转 ONNX，只能自己写翻译器。
    写翻译器的第一个前提是：权重必须能从 weight.bin 里正确解出来。
    本脚本就是验证这件事。

格式（对照 apple/coremltools mlmodel/format/MIL.proto + Model.proto）：
    Value (MIL) 里：
        f2 = ValueType
        f3 = immediateValue   （小常量直接内联）
        f5 = blobFileValue    -> { f1=fileName, f2=offset }
                               offset 指向 weight.bin 里的一段 64 字节元数据头

    weight.bin 里的元数据头是 64 字节对齐的，开头有魔数。
    CoreML 用 'BLOB' 作为 magic（小端读成 uint32 = 0x424C4F42）。
    紧跟 metadata size 之后就是裸张量数据（float16 / float32）。

用法:
    python weight_read.py <mlpackage 目录>              # 概览
    python weight_read.py <mlpackage 目录> --list 20    # 列前 20 个张量
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from opscan import parse_fields, first_bytes, as_text   # noqa: E402


BLOB_MAGIC_LE = struct.unpack("<I", b"BLOB")[0]
BLOB_MAGIC_BE = struct.unpack(">I", b"BLOB")[0]

# ---------------------------------------------------------------------------
# weight.bin 的真实布局（逐字节实测破译，不是猜的）
# ---------------------------------------------------------------------------
#   [0 : 64]                全局头
#                             u32 @0  = 张量个数 (v4 = 286)
#   [64 : 64 + N*192]       每个张量 192 字节元数据块，结构:
#                             +0   u32  magic = 0xDEADBEEF
#                             +4   u32  version = 2
#                             +8   u64  数据区/元数据标识
#                             +16  u64  (对齐/保留)
#                             +24  u32  张量数量（重复出现）
#                             +28  u32  (?)
#                             +48  u64  dataOffset  —— 数据在 weight.bin 里的字节偏移
#                             +56  u32  dataSize    —— 数据字节数
#   [64 + N*192 : ...]      裸张量数据（FP16 / FP32），按 dataOffset 取
#
# 实测校验:
#   N = 286 -> 数据区起点 = 64 + 286*192 = 54976
#   第一个张量 [4,3,3,3] FP16 -> 4*3*3*3*2 = 216 字节
#   元数据 +8 处读到 0xA0 = 160，+16 处读到 0x80 = 128，
#   160 + 128 = 288 ... 而 288 = 216 向上对齐到 32 字节后 + 32（对齐填充）。
#   因此以 dataOffset/dataSize 为准，不要自己算。
# ---------------------------------------------------------------------------
META_SIZE = 192
GLOBAL_HDR = 64
TENSOR_MAGIC = 0xDEADBEEF


def iter_tensor_meta(raw):
    """遍历 weight.bin 的张量元数据块 -> [(data_offset, data_size), ...]"""
    if len(raw) < GLOBAL_HDR:
        return []
    n = struct.unpack_from("<I", raw, 0)[0]
    out = []
    pos = GLOBAL_HDR
    for _ in range(n):
        if pos + META_SIZE > len(raw):
            break
        magic = struct.unpack_from("<I", raw, pos)[0]
        if magic != TENSOR_MAGIC:
            break
        data_off = struct.unpack_from("<Q", raw, pos + 48)[0]
        data_size = struct.unpack_from("<Q", raw, pos + 56)[0]
        out.append((data_off, data_size))
        pos += META_SIZE
    return out

# MIL DataType 枚举 —— 逐条抄自 apple/coremltools mlmodel/format/MIL.proto
# （之前凭印象写错过一次：真实值是 FLOAT16=10 / FLOAT32=11，
#   而不是"10=INT8, 11=INT16"那种按块编号的猜测。）
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
    "BFLOAT16": "bfloat16", "FLOAT8E4M3FN": "float8_e4m3fn",
    "FLOAT8E5M2": "float8_e5m2",
    "INT8": "int8", "INT16": "int16", "INT32": "int32", "INT64": "int64",
    "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64",
    "BOOL": "bool",
}

ITEMSIZE = {
    "float16": 2, "float32": 4, "float64": 8, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
    "bool": 1,
}


# ---------------------------------------------------------------------------
# 解析 MIL 类型
# ---------------------------------------------------------------------------
def parse_dimension(blob):
    """Dimension -> int（未知维度返回 -1）"""
    for f, wt, v in parse_fields(blob):
        if f == 1 and wt == 2:                  # constant
            for g, gwt, gv in parse_fields(v):
                if g == 1:
                    return int(gv)
        elif f == 2 and wt == 2:                # unknown
            return -1
    return -1


def parse_tensor_type(blob):
    """TensorType -> (numpy dtype name, [dims])"""
    dtype = None
    dims = []
    for f, wt, v in parse_fields(blob):
        if f == 1 and wt == 0:                  # dataType
            dtype = MIL_DTYPE.get(int(v), f"enum{v}")
        elif f == 3 and wt == 2:                # dimensions (repeated Dimension)
            dims.append(parse_dimension(v))
    return dtype, dims


def parse_value_type(blob):
    """ValueType -> (dtype, dims)  只处理 tensorType(f1)"""
    tt = first_bytes(parse_fields(blob), 1)
    if tt is None:
        return None, []
    return parse_tensor_type(tt)


def parse_value(blob):
    """MIL Value -> dict(role, dtype, dims, offset, filename, inline)"""
    out = {"dtype": None, "dims": [], "offset": None,
           "filename": None, "inline": None}
    for f, wt, v in parse_fields(blob):
        if f == 2 and wt == 2:                          # ValueType
            out["dtype"], out["dims"] = parse_value_type(v)
        elif f == 3 and wt == 2:                        # immediateValue
            out["inline"] = v
        elif f == 5 and wt == 2:                        # blobFileValue
            for g, gwt, gv in parse_fields(v):
                if g == 1 and gwt == 2:
                    out["filename"] = as_text(gv)
                elif g == 2:
                    out["offset"] = int(gv)
    return out


# ---------------------------------------------------------------------------
# 读取 weight.bin
# ---------------------------------------------------------------------------
def probe_blob(raw, offset):
    """在 offset 处探测元数据头，返回 (数据起始位置, 信息字典)"""
    info = {"offset": offset, "magic": None, "meta_size": None,
            "data_start": None, "ok": False}
    if offset is None or offset < 0 or offset + 8 > len(raw):
        return info
    m_le = struct.unpack_from("<I", raw, offset)[0]
    m_be = struct.unpack_from(">I", raw, offset)[0]
    if m_le == BLOB_MAGIC_LE:
        info["magic"] = "BLOB(LE)"
    elif m_be == BLOB_MAGIC_BE:
        info["magic"] = "BLOB(BE)"
    else:
        info["magic"] = f"0x{m_le:08X}"
        # 没有魔数：无法定位，退化为"数据紧跟 64 字节头"
        info["data_start"] = offset + 64
        return info

    # 魔数之后通常是 metadata size（uint32）
    meta_size = struct.unpack_from("<I", raw, offset + 4)[0]
    info["meta_size"] = meta_size
    # 头部按 64 字节对齐；metadata size 不包含魔数/自身
    cand = offset + meta_size
    if cand <= offset or cand > len(raw):
        cand = offset + 64
    info["data_start"] = cand
    info["ok"] = True
    return info


def deep_find_blob_refs(blob, op_type=None, depth=0, max_depth=20):
    """递归找出所有 BlobFileValue 引用。

    踩过的坑：一开始只在 Operation.f2 (inputs map) 里找，结果 0 命中。
    实际上 blobFileValue 嵌在 Operation.f5 (attributes map) -> val -> ...
    所以必须整棵子树递归搜，不能只看某一层。
    """
    hits = []
    if depth > max_depth:
        return hits
    for f, wt, v in parse_fields(blob):
        if wt != 2:
            continue
        sub = parse_fields(v)
        if f == 5:
            fn = None
            off = None
            for g, gwt, gv in sub:
                if g == 1 and gwt == 2:
                    fn = as_text(gv)
                elif g == 2 and gwt == 0:
                    off = gv
            if fn and fn.endswith(".bin") and off is not None:
                hits.append({"filename": fn, "offset": int(off), "op": op_type})
                continue
        hits.extend(deep_find_blob_refs(v, op_type, depth + 1, max_depth))
    return hits


def collect_blobs(raw_model):
    """遍历整个 MIL program，收集所有 blobFileValue 引用"""
    refs = []

    top = parse_fields(raw_model)
    ml = first_bytes(top, 502)
    if ml is None:
        return refs, "顶层没有 f502 (MLProgram)"
    funcmap = first_bytes(parse_fields(ml), 2)
    if funcmap is None:
        return refs, "MLProgram 里没有 f2 (functions)"

    # functions 是单个 entry: f1=key, f2=Function
    fblob = None
    for f, wt, v in parse_fields(funcmap):
        if f == 2 and wt == 2:
            fblob = v
    if fblob is None:
        return refs, "functions entry 里没有 f2 (Function)"

    ff = parse_fields(fblob)
    blockmap = first_bytes(ff, 3)
    if blockmap is None:
        return refs, "Function 里没有 f3 (block_specializations)"
    block = None
    for f, wt, v in parse_fields(blockmap):
        if f == 2 and wt == 2:
            block = v
    if block is None:
        return refs, "block_specializations entry 里没有 f2 (Block)"

    def walk(b, depth=0):
        if depth > 16:
            return
        for f, wt, v in parse_fields(b):
            if f != 3 or wt != 2:
                continue                             # Operation
            opf = parse_fields(v)
            op_type = as_text(first_bytes(opf, 1)) or "?"
            # 整个算子子树递归找 blob 引用 + 抓 shape
            for hit in deep_find_blob_refs(v, op_type):
                # 同一算子里找带 dtype/dims 的 Value，作为张量元信息
                vals = []
                deep_find_values(v, vals)
                info = next((x for x in vals if x["dims"]), None)
                if info:
                    hit["dtype"] = info["dtype"]
                    hit["dims"] = info["dims"]
                refs.append(hit)
            for g, gwt, gv in opf:
                if g == 4 and gwt == 2:
                    walk(gv, depth + 1)

    walk(block)
    return refs, None


def deep_find_values(blob, out, depth=0, max_depth=20):
    """递归找出所有带 dtype/dims 的 MIL Value"""
    if depth > max_depth:
        return
    for f, wt, v in parse_fields(blob):
        if wt != 2:
            continue
        sub = parse_fields(v)
        for g, gwt, gv in sub:
            if g == 2 and gwt == 2:                 # ValueType
                dtype, dims = parse_value_type(gv)
                if dtype or dims:
                    out.append({"dtype": dtype, "dims": dims})
        deep_find_values(v, out, depth + 1, max_depth)


def _map_entries(blob):
    """Operation.f2 是 map<string, Argument>：单个 entry 或 entry 列表"""
    fs = parse_fields(blob)
    wrapped = [v for f, wt, v in fs if f == 1 and wt == 2 and _is_entry(v)]
    if wrapped:
        return [_kv(e) for e in wrapped]
    return [_kv(blob)]


def _is_entry(blob):
    fs = parse_fields(blob)
    return bool(fs) and fs[0][0] == 1


def _kv(entry):
    k = None
    v = None
    for g, gwt, gv in parse_fields(entry):
        if g == 1 and k is None:
            k = as_text(gv) if gwt == 2 else gv
        elif g == 2 and v is None:
            v = gv
    return k, v


# ---------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_list = "--list" in sys.argv
    limit = 20
    for i, a in enumerate(sys.argv):
        if a == "--list" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    p = Path(args[0]) if args else Path("../mlpackage")
    if p.is_dir():
        cands = sorted(p.rglob("model.mlmodel"))
        if not cands:
            print(f"[X] {p} 里找不到 model.mlmodel")
            return 1
        p = cands[0]
    pkg = p.parent.parent.parent                          # .../X.mlpackage
    wb = pkg / "Data/com.apple.CoreML/weights/weight.bin"

    print("=" * 76)
    print(f"包目录 : {pkg}")
    print(f"模型   : {p.name}  ({p.stat().st_size} bytes)")
    if not wb.exists():
        print(f"[X] 找不到 {wb}")
        return 1
    wraw = wb.read_bytes()
    print(f"权重   : weight.bin  ({len(wraw)} bytes)")
    print("=" * 76)

    refs, err = collect_blobs(p.read_bytes())
    if err:
        print(f"  [!] {err}")
        return 1
    print(f"  发现 {len(refs)} 个 blobFileValue 引用（外部权重张量）")

    filenames = sorted({r["filename"] for r in refs if r["filename"]})
    print(f"  引用的文件名: {filenames}")
    offsets = sorted({r["offset"] for r in refs})
    print(f"  offset 范围: {offsets[0]} .. {offsets[-1]}"
          f"   （weight.bin 大小 {len(wraw)}）")

    # 探测魔数
    good = 0
    for r in refs:
        info = probe_blob(wraw, r["offset"])
        r["info"] = info
        if info["ok"]:
            good += 1
    print(f"  含 'BLOB' 魔数的: {good} / {len(refs)}"
          f"   （{'格式判断正确' if good == len(refs) else '有偏差，需再看'}）")

    # 抽样验证：用 dtype + shape 算出期望字节数，和下一段 offset 的间隔比较
    print()
    print("  --- 抽样验证：期望字节数 vs 实际占用间隔 ---")
    checked = 0
    okcnt = 0
    for i, r in enumerate(refs):
        if r["offset"] is None or "info" not in r:
            continue
        if checked >= limit:
            break
        dtype = r["dtype"]
        dims = [d for d in r["dims"] if d >= 0]
        if dtype not in NP_OF or not dims:
            continue
        n = 1
        for d in dims:
            n *= d
        itemsize = {"float16": 2, "float32": 4, "float64": 8,
                    "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
                    "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
                    "bool": 1}[NP_OF[dtype]]
        want = n * itemsize
        start = r["info"]["data_start"]
        # 下一个更大 offset 作为上界
        nxt = None
        for o in offsets:
            if o is not None and o > r["offset"]:
                nxt = o
                break
        avail = (nxt - start) if nxt else (len(wraw) - start)
        fit = want <= avail
        okcnt += 1 if fit else 0
        checked += 1
        if checked <= 12 or not fit:
            print(f"    {r['op']:<12} {str(r.get('arg')):<8} {dtype:<6} "
                  f"{str(r['dims']):<20} 需要 {want:>10} 可用 {avail:>10} "
                  f"{'OK' if fit else '** 放不下 **'}")
    print(f"    小计: {okcnt} / {checked} 个张量的字节数能放进分配的区间")

    if do_list:
        print()
        print("  --- 全部引用 ---")
        for i, r in enumerate(refs[:200]):
            print(f"    [{i:>4}] off={r['offset']:<12} {r['dtype']:<6} "
                  f"{str(r['dims']):<22} op={r['op']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
