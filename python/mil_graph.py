# -*- coding: utf-8 -*-
"""
mil_graph.py —— 把 MIL program 解析成可用的图结构（纯标准库）。

输出一个 Graph 对象：
    ops     : [Op, ...]        按 MIL 里的拓扑顺序
    Op.type : 算子名
    Op.name : 输出名（MIL 是 SSA，每个算子一个输出）
    Op.in_  : {参数名: [绑定, ...]}   绑定是 str(值名) 或 {"int":..}/{"float":..}/...
    Op.dtype, Op.dims : 输出的类型信息
    consts  : {输出名: (offset, dtype, dims)}   外部权重引用

这是 MIL -> ONNX 翻译器的输入层。

用法:
    python mil_graph.py <mlpackage 目录>                 # 概览
    python mil_graph.py <mlpackage 目录> --attrs conv    # 打印某算子的属性样例
    python mil_graph.py <mlpackage 目录> --sample 3      # 每种算子打印 3 个
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from opscan import parse_fields, first_bytes, as_text          # noqa: E402
from bxt_decode import (MIL_DTYPE, parse_value_type,           # noqa: E402
                        find_blobfile)
from spec_read import unpack_varints                           # noqa: E402

# ModelDescription 里 multiArray 的 dataType 枚举（和 MIL 的 DataType 不同！）
MLMULTI_DTYPE = {
    65568: "FLOAT32", 65552: "FLOAT16", 65600: "FLOAT64",
    131104: "INT32", 131080: "INT64",
}


def parse_feature_type(blob):
    """FeatureType -> (dtype, dims)；只处理 multiArrayType(f5)"""
    ma = first_bytes(parse_fields(blob), 5)
    if ma is None:
        return None, []
    dtype = None
    dims = []
    for f, wt, v in parse_fields(ma):
        if f == 1 and wt == 2:                      # Shape（打包 varint）
            dims = unpack_varints(v)
        elif f == 2 and wt == 0:                    # dataType
            dtype = MLMULTI_DTYPE.get(int(v), f"enum{int(v)}")
    return dtype, dims


# ---------------------------------------------------------------------------
# 从 protobuf 值里取出"立即值"（整数 / 浮点 / 字符串 / 张量）
# ---------------------------------------------------------------------------
def decode_immediate(blob, depth=0):
    """ImmediateValue -> python 值

    实测路径（transpose 的 perm=[0,3,1,2]）:
        Value.f3 -> ImmediateValue.f1 -> TensorValue.f2 -> RepeatedInts.f1 -> RAW
        最后 4 字节 00 03 01 02 是 **packed varint**，解出来就是 [0,3,1,2]

    注意 TensorValue 的字段号（对照 MIL.proto）：
        f1 floats / f2 ints / f3 bools / f4 strings / f5 longInts / f7 bytes
    而 ints(f2) 里面 f1 才是 packed 的 varint 数据。
    所以 TensorValue.f2 本身是 **消息**，不是 packed 数组 ——
    直接把 f2 的载荷按 int32 解会得到垃圾（曾解出 33620736）。
    """
    if depth > 6:
        return None
    for f, wt, v in parse_fields(blob):
        if wt != 2:
            continue
        if f == 1:                                   # TensorValue
            r = decode_tensor_value(v, depth + 1)
            if r is not None:
                return r
        # 有些常量外面还多包了一层，往下钻
        if depth < 6:
            fs = parse_fields(v)
            for g, gwt, gv in fs:
                if g == 1 and gwt == 2:
                    r = decode_tensor_value(gv, depth + 1)
                    if r is not None:
                        return r
    return None


_FIELD_KIND = {1: "floats", 2: "ints", 3: "bools", 4: "strings",
               5: "longInts", 6: "doubles", 7: "bytes"}


def decode_tensor_value(blob, depth=0):
    """TensorValue -> dict(kind, values)

    每个字段（f1/f2/...）本身是一个小消息，它的 f1 才是 packed 数据。
    实测：f2(ints) -> {f1: packed varint}
    """
    if depth > 4:
        return None
    for f, wt, v in parse_fields(blob):
        if wt != 2:
            continue
        kind = _FIELD_KIND.get(f)
        if kind is None:
            continue
        vals = _unpack_field(v, kind)
        if vals is not None:
            return {"kind": kind, "values": vals}
    return None


def _unpack_field(blob, kind):
    """RepeatedX 消息 -> python 列表

    里面可能直接就是 packed 裸数据，也可能还包一层 {f1: packed}。
    """
    # 先看里面有没有 f1
    fs = parse_fields(blob)
    payload = None
    for g, gwt, gv in fs:
        if g == 1 and gwt == 2:
            payload = gv
            break
    if payload is None:
        payload = blob                            # 裸 packed 数据

    if kind == "floats":
        if len(payload) % 4:
            return None
        import struct as _s
        return list(_s.unpack("<%df" % (len(payload) // 4), payload))
    if kind == "doubles":
        if len(payload) % 8:
            return None
        import struct as _s
        return list(_s.unpack("<%dd" % (len(payload) // 8), payload))
    if kind in ("ints", "longInts"):
        bits = 32 if kind == "ints" else 64
        return _unpack_varints(payload, bits)
    if kind == "bools":
        return [bool(b) for b in payload]
    if kind == "strings":
        return [b.decode("utf-8", "replace") for b in
                ([payload] if isinstance(payload, (bytes, bytearray)) else [])]
    return None


def _unpack_varints(blob, bits=32):
    """解 packed varint，并按位宽做补码还原。

    prototype 的 int32/int64 负数是用**补码 varint** 编码的，
    也就是先转成无符号再 varint（所以 -1 是 10 个字节 0xFF...0x01）。
    直接读会得到 18446744073709551615 这种巨数，
    塞进 numpy int64 / ONNX 会报
    "OverflowError: Python int too large to convert to C long"。

    实测：central_crop 的 begin=[0,128,128,0] 没问题，
    但 size 因为 CoreML 用 -1 表示"到末尾"，解出来就是 2^64-1。
    """
    out = []
    p = 0
    n = len(blob)
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    while p < n:
        r = 0
        s = 0
        while True:
            if p >= n:
                return out
            x = blob[p]
            p += 1
            r |= (x & 0x7F) << s
            if not (x & 0x80):
                break
            s += 7
            if s > 70:
                return out
        r &= mask
        if r & sign:                              # 负数：补码还原
            r -= (1 << bits)
        out.append(r)
    return out


def decode_value(blob):
    """MIL Value -> python 值

    Value = { f2 ValueType, f3 ImmediateValue, f5 BlobFileValue }
    """
    dtype = None
    dims = []
    for f, wt, v in parse_fields(blob):
        if f == 2 and wt == 2:
            dtype, dims = parse_value_type(v)
        elif f == 3 and wt == 2:
            iv = decode_immediate(v)
            if iv is not None:
                return iv
        elif f == 5 and wt == 2:
            fn = None
            off = None
            for g, gwt, gv in parse_fields(v):
                if g == 1 and gwt == 2:
                    fn = as_text(gv)
                elif g == 2:
                    off = int(gv)
            return {"kind": "blob", "file": fn, "offset": off}
    return None


def decode_argument(blob):
    """Argument -> [绑定, ...]

    实测字节（transpose 的 perm 参数）：
        0a 14  0a 12  "transpose_1_perm_0"
        ^      ^     ^
        |      |     +-- Binding.f1 = 名字（这才是值名）
        |      +-------- Binding { f1: name | f2: Value }
        +--------------- Argument.f1 = repeated Binding

    踩过的坑：以为 Argument.f1 里直接就是字符串，
    于是把 Binding.f1 的内层 blob 当名字，拿到的是一串二进制而不是
    'transpose_1_perm_0'，导致所有 perm/shape 解析失败、Transpose 没有 perm 属性。
    """
    out = []
    for f, wt, v in parse_fields(blob):
        if f != 1 or wt != 2:
            continue
        # v 是 Binding
        bf = parse_fields(v)
        name = None
        val = None
        for g, gwt, gv in bf:
            if g == 1 and gwt == 2:
                name = _deep_text(gv)
            elif g == 2 and gwt == 2:
                val = decode_value(gv)
        if name is not None:
            out.append(name)
        elif val is not None:
            out.append(val)
    return out


def _deep_text(blob, depth=0):
    """Binding.f1 里可能还包了一层才到真正的字符串。

    实测：Binding.f1(18B) -> f1(16B) -> 'transpose_1_perm_0'
    所以这里往下钻，直到拿到可打印字符串；钻不动就返回 None。
    """
    if depth > 4:
        return None
    s = as_text(blob)
    if s is not None and s.isprintable() and s:
        return s
    fs = parse_fields(blob)
    if len(fs) == 1 and fs[0][1] == 2:
        return _deep_text(fs[0][2], depth + 1)
    return None


def decode_arg_map(blob):
    """Operation.f2 = repeated map entry，每个 entry 是一个算子参数

    结构（实测）：
        Operation.f2 (repeated)     一个参数一个 entry
          entry.f1 = 参数名 (str)   例如 'x' / 'weight' / 'strides' / 'pad'
          entry.f2 = Argument
            Argument.f1 (repeated) = Binding
              Binding.f1 = 值名 (str)     引用前一个算子的输出
              Binding.f2 = Value          立即值 / 权重 blob 引用

    踩过的坑：一开始用 "_is_entry" 判断 entry 是否符合预期，
    结果 key 只有 1 字节时判定失败，8 个参数只解出 1 个。
    map entry 不需要启发式判断，直接每个 f2 都当 entry 解就行。
    """
    return _decode_map_field(blob, 2)


def decode_attr_map(blob):
    """Operation.f5 = attribute map（同样是 repeated map entry）

    结构：
        entry.f1 = 属性名 ('val' / 'name' / ...)
        entry.f2 = Value
          Value.f2 = ValueType
          Value.f3 = ImmediateValue  -> TensorValue -> floats/ints/...
          Value.f5 = BlobFileValue

    const 算子的内联小常量（transpose 的 perm、reshape 的 shape 等）
    全在这里。之前只解 f2 不解 f5，导致所有 perm 变成 None。
    """
    return _decode_map_field(blob, 5)


def _decode_map_field(blob, field):
    """解 Operation 的 map 字段（f2=inputs, f5=attributes）

    两种 entry 的载荷形状不同：
      * f2 (inputs)   entry.f2 = Argument { f1: repeated Binding }
                      Binding 里 f1=名字 或 f2=Value
      * f5 (attrs)    entry.f2 = Value 直接就是值本身
                      （Value.f2=ValueType, f3=ImmediateValue, f5=BlobFileValue）
                      没有 Argument/Binding 那层包装

    判定顺序很重要（这里翻过车）：
        先试 Argument（输入）——因为它的载荷里有 Binding.f1 名字/ f2 Value
        再试 Value（属性）
        最后兜底裸名字
    如果反过来先试 Value，名字绑定会被 decode_value 的模糊匹配误判成值，
    结果是输入参数全变成空列表（节点数从 509 掉到 497，13 处悬空）。
    """
    params = []
    for f, wt, v in parse_fields(blob):
        if f != field or wt != 2:
            continue
        k, payload = _kv(v)
        if payload is None:
            continue

        # 1) 输入：载荷是 Argument（里面是 repeated Binding）
        args = decode_argument(payload)
        if args:
            params.append((k, args))
            continue

        # 2) 属性：载荷直接是 Value
        dv = decode_value(payload)
        if dv is not None:
            params.append((k, [dv]))
            continue

        # 3) 兜底：裸名字
        nm = _deep_text(payload)
        params.append((k, [nm] if nm else []))
    return params


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
class Op:
    __slots__ = ("type", "name", "names", "in_", "dtype", "dims", "order")

    def __init__(self, otype, name, in_, dtype, dims, order, names=None):
        self.type = otype
        self.name = name
        self.names = names or ([name] if name else [])
        self.in_ = in_
        self.dtype = dtype
        self.dims = dims
        self.order = order

    def __repr__(self):
        return f"<Op {self.type} {self.name} in={ {k: v for k, v in self.in_.items()} }>"


class Graph:
    def __init__(self):
        self.ops = []
        self.inputs = []          # [(name, dtype, dims)]
        self.outputs = []         # [(name, dtype, dims)]
        self.intermediates = []   # [(name, dtype, dims)]
        self.consts = {}          # name -> (offset, dtype, dims)


def load(path):
    """加载 .mlpackage -> Graph"""
    p = Path(path)
    if p.is_dir():
        c = sorted(p.rglob("model.mlmodel"))
        if not c:
            raise FileNotFoundError(f"{p} 里找不到 model.mlmodel")
        p = c[0]
    raw = p.read_bytes()

    top = parse_fields(raw)
    ml = first_bytes(top, 502)
    if ml is None:
        raise ValueError("不是 ML Program 模型")

    g = Graph()

    # 输入 / 输出来自 ModelDescription (Model.f2)
    desc = first_bytes(top, 2)
    if desc:
        for f, wt, v in parse_fields(desc):
            if wt != 2:
                continue
            if f == 1:                                    # 输入
                fl = parse_fields(v)
                nm = first_bytes(fl, 1)
                ty = first_bytes(fl, 3)
                dt, dm = parse_feature_type(ty) if ty else (None, [])
                g.inputs.append((as_text(nm) if nm else "?", dt, dm))
            elif f == 2:                                  # 输出
                fl = parse_fields(v)
                nm = first_bytes(fl, 1)
                ty = first_bytes(fl, 3)
                dt, dm = parse_feature_type(ty) if ty else (None, [])
                g.outputs.append((as_text(nm) if nm else "?", dt, dm))
            elif f == 10:                                 # 中间张量
                fl = parse_fields(v)
                nm = first_bytes(fl, 1)
                ty = first_bytes(fl, 3)
                dt, dm = parse_feature_type(ty) if ty else (None, [])
                g.intermediates.append((as_text(nm) if nm else "?", dt, dm))

    # 图
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

    order = 0

    def walk(bl, depth=0):
        nonlocal order
        if depth > 16:
            return
        for f, wt, v in parse_fields(bl):
            if f != 3 or wt != 2:
                continue
            opf = parse_fields(v)
            otype = as_text(first_bytes(opf, 1)) or "?"
            # 输出名：Operation.f3 是 **repeated** NamedValueType
            # （split 之类的多输出算子会有多个，例如 xxx_split_1）
            names = []
            dtype, dims = (None, [])
            for a, b, c in opf:
                if a != 3 or b != 2:
                    continue
                fl = parse_fields(c)
                nm = first_bytes(fl, 1)
                nm_s = as_text(nm) if nm else None
                if nm_s:
                    names.append(nm_s)
                if dtype is None:
                    t2 = first_bytes(fl, 2)
                    if t2:
                        dtype, dims = parse_value_type(t2)
            name = names[0] if names else "?"
            # 输入参数
            # 注意：Operation.f2 是 **repeated** 的（一个算子 8 个参数 = 8 个 f2），
            # 所以必须把整个 op blob 交给 decode_arg_map 去遍历，
            # 不能用 first_bytes(opf, 2) —— 那只会拿到第一个参数。
            #
            # 另外 Operation.f5 是 attribute map，也是 repeated map entry。
            # const 算子的内联常量（例如 transpose 的 perm=[0,2,3,1]）
            # 就存在 f5 的 'val' 属性里，不解码 f5 会丢掉所有 perm/shape 之类的小常量。
            in_ = {}
            for k, vals in decode_arg_map(v):
                if k:
                    in_[k] = vals
            for k, vals in decode_attr_map(v):
                if k and k not in in_:
                    in_[k] = vals
            # 权重引用
            for _fn, off in find_blobfile(v):
                g.consts[name] = (off, dtype, dims)

            g.ops.append(Op(otype, name, in_, dtype, dims, order, names))
            order += 1

            for gg, gwt, gv in opf:
                if gg == 4 and gwt == 2:
                    walk(gv, depth + 1)

    walk(block)
    return g


def fmt_binding(b):
    if isinstance(b, str):
        return b
    if isinstance(b, dict):
        k = b.get("kind")
        if k == "blob":
            return f"<blob @{b['offset']}>"
        vs = b.get("values") or []
        if len(vs) <= 6:
            return f"<{k}:{vs}>"
        return f"<{k}[{len(vs)}]>"
    return repr(b)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = None
    if "--attrs" in sys.argv:
        i = sys.argv.index("--attrs")
        if i + 1 < len(sys.argv):
            only = sys.argv[i + 1]
    sample = 1
    if "--sample" in sys.argv:
        i = sys.argv.index("--sample")
        if i + 1 < len(sys.argv):
            sample = int(sys.argv[i + 1])

    path = args[0] if args else "../mlpackage"
    g = load(path)

    print("=" * 78)
    print(f"模型: {path}")
    print("=" * 78)
    print(f"  MIL 声明输入 : {[(n, d, s) for n, d, s in g.inputs]}")
    print(f"  MIL 声明输出 : {[(n, d, s) for n, d, s in g.outputs]}")
    if g.intermediates:
        print(f"  中间张量     : {[(n, d, s) for n, d, s in g.intermediates]}")
    print(f"  算子总数     : {len(g.ops)}")
    print(f"  外部权重引用 : {len(g.consts)}")
    print()

    hist = Counter(o.type for o in g.ops)
    print("  --- 算子种类 ---")
    for t, n in hist.most_common():
        print(f"    {t:<34} x{n}")

    print()
    print("  --- 每种算子的属性样例 ---")
    by_type = defaultdict(list)
    for o in g.ops:
        by_type[o.type].append(o)
    types = sorted(by_type) if only is None else [only]
    for t in types:
        group = by_type.get(t, [])
        if not group:
            print(f"    {t}: (没有)")
            continue
        print(f"    [{t}]  x{len(group)}")
        for o in group[:sample]:
            print(f"        out   : {o.name}   {o.dtype} {o.dims}")
            for k, vals in o.in_.items():
                shown = ", ".join(fmt_binding(b) for b in vals[:4])
                more = f" ...(+{len(vals)-4})" if len(vals) > 4 else ""
                print(f"        in.{k:<12}: {shown}{more}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
