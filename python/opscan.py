# -*- coding: utf-8 -*-
"""
opscan.py —— 自包含的 CoreML ML Program 算子扫描器（纯标准库，Windows / Linux 通用）。

为什么单独写一份：
    之前复用 spec_read.py 的解析器时，protobuf 字段号靠猜，走了很多弯路。
    这一份不 import 任何本地模块，字段号全部对照 apple/coremltools 的
    mlmodel/format/MIL.proto 并逐层实测校准过。

用途：
    在动手装 WSL2 之前，先确认模型里有没有 ONNX 无法表达的算子。
    ML Program -> ONNX 本质是逐算子翻译；只要有一个自定义算子，
    无论在 macOS 还是 Linux 转，结果都不可用。

实测结构（v4 / v5 均一致）：
    Model
      f1   = spec version (varint) = 6
      f2   = ModelDescription
      f502 = MLProgram
               f1 = version (varint)
               f2 = map entry * N      每个 entry: f1=key(str), f2=Function
               f4 = metadata
    Function
      f1 = inputs (repeated NamedValueType)
      f2 = opset  (str, "CoreML5")
      f3 = map entry * N      每个 entry: f1=key(str), f2=Block
    Block
      f1 = inputs
      f2 = outputs (repeated string)
      f3 = Operation * N
      f4 = attributes
    Operation
      f1 = type  (str)
      f2 = inputs (map)
      f3 = outputs
      f4 = blocks (repeated Block)
      f5 = attributes

用法:
    python opscan.py <model.mlmodel 或 .mlpackage 目录> [...]
    python opscan.py --dump-block <model.mlmodel>     # 打印前 20 个算子明细
"""
import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ===========================================================================
# protobuf 最小读取器
# ===========================================================================
def read_varint(b, p):
    """返回 (值, 新位置)；失败返回 (None, p)"""
    result = 0
    shift = 0
    while True:
        if p >= len(b):
            return None, p
        x = b[p]
        p += 1
        result |= (x & 0x7F) << shift
        if not (x & 0x80):
            return result, p
        shift += 7
        if shift > 70:
            return None, p


def parse_fields(b):
    """把一个消息拆成 [(field_number, wire_type, value), ...]

    wire_type 0 -> value 是 int
    wire_type 2 -> value 是 bytes
    """
    out = []
    p = 0
    n = len(b)
    while p < n:
        key, p2 = read_varint(b, p)
        if key is None:
            break
        p = p2
        fnum = key >> 3
        wtype = key & 7
        if fnum == 0:
            break
        if wtype == 0:
            val, p2 = read_varint(b, p)
            if val is None:
                break
            out.append((fnum, 0, val))
            p = p2
        elif wtype == 2:
            ln, p2 = read_varint(b, p)
            if ln is None:
                break
            p = p2
            if p + ln > n:
                break
            out.append((fnum, 2, b[p:p + ln]))
            p += ln
        elif wtype == 5:
            if p + 4 > n:
                break
            out.append((fnum, 5, b[p:p + 4]))
            p += 4
        elif wtype == 1:
            if p + 8 > n:
                break
            out.append((fnum, 1, b[p:p + 8]))
            p += 8
        else:
            break
    return out


def as_text(v):
    """bytes -> str，不行就返回 None"""
    if not isinstance(v, (bytes, bytearray)):
        return None
    try:
        return v.decode("utf-8")
    except Exception:
        return None


def first_bytes(fields, fnum):
    for f, wt, v in fields:
        if f == fnum and wt == 2:
            return v
    return None


def all_bytes(fields, fnum):
    return [v for f, wt, v in fields if f == fnum and wt == 2]


def _entry_kv(entry):
    """一个 map entry 消息 -> (key, value)。f1=key(str), f2=value(bytes)"""
    key = None
    val = None
    for g, gwt, gv in parse_fields(entry):
        if g == 1 and key is None:
            key = as_text(gv) if gwt == 2 else gv
        elif g == 2 and val is None:
            val = gv
    return key, val


def map_entries(blob):
    """解析一个 map 字段的载荷 -> [(key, value), ...]

    关键点（之前写错的地方）：
        protobuf 的 map 在字节流里是"重复的 entry 消息"，
        每个 entry 自身是一个小消息 { f1=key, f2=value }。
        所以要先看 blob 里 field 2 出现了几次：

          * 出现多次  -> blob 是 entry 列表，逐个解析
          * 一个都没有 -> blob 本身就是单个 entry 的内容
                          （此时 f1=key, f2=value 已经直接在 blob 里）

        实测：MLProgram.f2 (functions) 和 Function.f3 (block_specializations)
        都是「单个 entry」，即 f1='main' / f1='CoreML5' + f2=<消息>。
        早期版本误把 f1/f2 当成两个 entry，导致 key 解析成
        '\\n\\nparameters...' 这种二进制垃圾。
    """
    fields = parse_fields(blob)
    wrapped = [v for f, wt, v in fields if f == 1 and wt == 2 and _looks_like_entry(v)]
    if wrapped:
        return [_entry_kv(e) for e in wrapped]
    return [_entry_kv(blob)]


def _looks_like_entry(blob):
    """entry 消息：第一个字段应该是 f1 (key)。"""
    fs = parse_fields(blob)
    return bool(fs) and fs[0][0] == 1


# ===========================================================================
# 遍历模型
# ===========================================================================
def walk_block(blob, ops, depth=0, max_depth=16):
    """Block.f3 = repeated Operation；Operation.f4 = 子 Block"""
    if depth > max_depth:
        return
    for f, wt, v in parse_fields(blob):
        if f != 3 or wt != 2:
            continue                                   # 只要 Operation
        op_type = None
        sub_blocks = []
        for g, gwt, gv in parse_fields(v):
            if g == 1 and gwt == 2 and op_type is None:
                op_type = as_text(gv)
            elif g == 4 and gwt == 2:
                sub_blocks.append(gv)
        if op_type:
            ops[op_type] += 1
        for sb in sub_blocks:
            walk_block(sb, ops, depth + 1, max_depth)


def collect(raw):
    """-> (ops Counter, opsets set, funcs list, error or None)"""
    ops = Counter()
    opsets = set()
    funcs = []

    top = parse_fields(raw)
    ml = first_bytes(top, 502)
    if ml is None:
        return ops, opsets, funcs, "顶层没有 f502 (MLProgram) —— 可能是旧版 NeuralNetwork 模型"

    mlf = parse_fields(ml)
    funcmap = first_bytes(mlf, 2)
    if funcmap is None:
        return ops, opsets, funcs, "MLProgram 里没有 f2 (functions map)"

    entries = map_entries(funcmap)
    if not entries:
        return ops, opsets, funcs, "functions map 解析出 0 个条目"

    for fname, fblob in entries:
        funcs.append(fname if fname else "?")
        if fblob is None:
            continue
        ff = parse_fields(fblob)
        opset = as_text(first_bytes(ff, 2))
        if opset:
            opsets.add(opset)
        blockmap = first_bytes(ff, 3)
        if blockmap is None:
            continue
        for spec, block in map_entries(blockmap):
            if block is not None:
                walk_block(block, ops)

    if not ops:
        return ops, opsets, funcs, "找到了 function 但没有解析出任何算子"
    return ops, opsets, funcs, None


# ===========================================================================
# ONNX 对应判断
# ===========================================================================
HAVE_ONNX = {
    "add", "sub", "mul", "real_div", "div", "pow", "sqrt", "rsqrt", "exp", "log",
    "abs", "neg", "sign", "floor", "ceil", "round", "clip", "threshold",
    "maximum", "minimum", "square", "erf", "tanh", "sigmoid", "softplus",
    "softsign", "elu", "gelu", "relu", "relu6", "leaky_relu", "prelu", "selu",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
    "logical_not", "logical_and", "logical_or", "logical_xor",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "select", "where", "cast", "identity",
    "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod",
    "reduce_l1_norm", "reduce_l2_norm", "reduce_log_sum", "reduce_log_sum_exp",
    "reduce_argmax", "reduce_argmin", "argsort", "argmax", "argmin",
    "reshape", "transpose", "expand_dims", "squeeze", "flatten", "concat",
    "split", "slice_by_index", "slice_by_size", "gather", "gather_nd",
    "scatter", "scatter_nd", "tile", "pad", "reverse", "shape",
    "conv", "conv_transpose", "pool", "max_pool", "avg_pool", "l2_pool",
    "lrn", "batch_norm", "instance_norm", "layer_norm", "group_norm",
    "local_response_norm", "softmax", "matmul", "einsum", "topk",
    "non_maximum_suppression", "resize_bilinear", "upsample_bilinear",
    "resize_nearest_neighbor", "crop", "crop_resize", "upsample",
    "linear", "stack", "range", "cumsum", "one_hot", "mod", "iota",
    "broadcast_to", "stop_gradient", "clamped_relu", "scaled_tanh",
    "log_softmax", "slice_by_index", "sliding_window",
}

RISKY = {
    "custom":  "自定义算子：ONNX 无对应，转换会失败或退化成占位节点",
    "xattn":   "cross-attention 融合算子：官方转换器不认，需用 MatMul/Softmax 手工重写",
    "attn":    "attention 融合算子：需手工重写",
    "mlp_glu": "GLU 融合 MLP：需拆成 MatMul + Sigmoid + Mul",
    "rfft":    "实数 FFT：ONNX 支持有限",
    "stft":    "短时傅里叶：ONNX 无标准算子",
    "fft":     "FFT：ONNX 支持有限",
    "random":  "随机数：语义不同",
    "sample":  "采样：无对应",
    "resize":  "缩放：ONNX 有 Resize，但坐标变换语义要对齐",
    "affine":  "仿射变换：需用 GridSample 或手工重写",
}


def classify(name):
    n = (name or "").lower()
    # 常量字面量：MLProgram 把它也编码成一个 Operation，
    # 每个权重张量一个，所以数量巨大（v4 有 1263 个）。
    # 转 ONNX 时会被内联成 initializer，不是真算子。
    if n == "const":
        return "CONST"
    if n in RISKY:
        return "RISKY"
    if n in HAVE_ONNX:
        return "OK"
    for k in RISKY:
        if k in n:
            return "RISKY"
    # 已知的官方算子前缀
    for k in ("conv", "reduce", "slice", "resize", "scatter", "gather",
              "batch", "layer", "instance", "group", "pool", "upsample",
              "crop", "soft", "log", "elu", "relu", "prelu", "clip", "cast"):
        if n.startswith(k):
            return "LIKELY"
    return "UNKNOWN"


def dump_block(raw, limit=20):
    top = parse_fields(raw)
    ml = first_bytes(top, 502)
    funcmap = first_bytes(parse_fields(ml), 2)
    for fname, fblob in map_entries(funcmap):
        ff = parse_fields(fblob)
        blockmap = first_bytes(ff, 3)
        for spec, block in map_entries(blockmap):
            print(f"  function={fname} opset={spec}")
            cnt = 0
            for f, wt, v in parse_fields(block):
                if f != 3 or wt != 2:
                    continue
                cnt += 1
                if cnt > limit:
                    break
                op = as_text(first_bytes(parse_fields(v), 1))
                print(f"    [{cnt:>4}] {op}")
            print(f"    ... 共 {cnt}+ 个算子")


def scan(path):
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.rglob("model.mlmodel"))
        if not cands:
            print(f"[X] {p} 里找不到 model.mlmodel")
            return None
        p = cands[0]
    raw = p.read_bytes()
    ops, opsets, funcs, err = collect(raw)

    print("=" * 76)
    print(f"模型: {path}")
    print("=" * 76)
    if err:
        print(f"  [!] {err}")
        return None
    print(f"  函数      : {', '.join(funcs)}")
    print(f"  算子集    : {', '.join(sorted(opsets)) or '?'}")
    print(f"  算子种类  : {len(ops)}      算子实例: {sum(ops.values())}")
    print()

    buckets = {"OK": [], "LIKELY": [], "CONST": [], "UNKNOWN": [], "RISKY": []}
    for name, cnt in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0])):
        buckets[classify(name)].append((name, cnt))

    labels = [
        ("OK",      "确认有 ONNX 对应"),
        ("LIKELY",  "大概率有对应（前缀匹配，需实测确认）"),
        ("CONST",   "常量字面量（会被内联成 ONNX initializer，不是算子）"),
        ("UNKNOWN", "不认识，需要人工查 ONNX 算子库"),
        ("RISKY",   "有风险 —— 需要手工重写或可能不支持"),
    ]
    for tag, label in labels:
        items = buckets[tag]
        if not items:
            continue
        print(f"  --- {tag} : {label}  ({len(items)} 种) ---")
        for name, cnt in items:
            note = RISKY.get((name or "").lower(), "")
            if not note:
                for k, v in RISKY.items():
                    if k in (name or "").lower():
                        note = v
                        break
            print(f"      {name:<34} x{cnt:<6}" + (f"  <- {note}" if note else ""))
        print()

    print("-" * 76)
    problem = buckets["RISKY"] + buckets["UNKNOWN"]
    if buckets["RISKY"]:
        print(f"  [结论] 有 {len(buckets['RISKY'])} 种高风险算子，转换不能一键完成。")
    elif buckets["UNKNOWN"]:
        print(f"  [结论] 有 {len(buckets['UNKNOWN'])} 种未知算子，先查清再动手。")
    else:
        print("  [结论] 全部算子都有 ONNX 对应，转换可行性高。")
    print()
    return {"ops": dict(ops), "opsets": sorted(opsets), "funcs": funcs,
            "risky": [n for n, _ in buckets["RISKY"]],
            "unknown": [n for n, _ in buckets["UNKNOWN"]]}


def main():
    args = sys.argv[1:]
    if args and args[0] == "--dump-block":
        raw = Path(args[1]).read_bytes()
        dump_block(raw)
        return 0
    if not args:
        args = ["../mlpackage"]
    results = {}
    for a in args:
        r = scan(a)
        if r:
            results[a] = r
    if len(results) > 1:
        print("=" * 76)
        print("汇总：需要人工处理的算子")
        for k, r in results.items():
            print(f"  {Path(k).name}")
            print(f"      高风险: {r['risky'] or '无'}")
            print(f"      未知  : {r['unknown'] or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
