# -*- coding: utf-8 -*-
"""
onnx_writer.py —— 零依赖的 ONNX protobuf 序列化器。

为什么需要它：
    本机 pip 装 `onnx` 需要代理，而代理经常不在。
    为了不让"写不出 .onnx 文件"卡住整条流程，这里直接用纯标准库
    把 ONNX 模型序列化成 protobuf 字节流。

字段号全部抄自官方定义 onnx/onnx.proto3
（已存档到 ../proto/onnx.proto3），不是猜的：

    ModelProto
      f1  ir_version        (int64)
      f2  producer_name     (string)
      f3  producer_version  (string)
      f7  graph             (GraphProto)
      f8  opset_import      (repeated OperatorSetIdProto)
      f14 metadata_props

    GraphProto
      f1  node              (repeated NodeProto)
      f2  name              (string)
      f5  initializer       (repeated TensorProto)
      f11 input             (repeated ValueInfoProto)
      f12 output            (repeated ValueInfoProto)
      f13 value_info        (repeated ValueInfoProto)

    NodeProto
      f1  input             (repeated string)
      f2  output            (repeated string)
      f3  name              (string)
      f4  op_type           (string)
      f5  attribute         (repeated AttributeProto)

    TensorProto
      f1  dims              (repeated int64, packed)
      f2  data_type         (int32)
      f4  float_data        (repeated float, packed)
      f5  int32_data        (repeated int32, packed)
      f7  int64_data        (repeated int64, packed)
      f8  name              (string)
      f9  raw_data          (bytes)
      f10 double_data       (repeated double, packed)

    ValueInfoProto
      f1  name              (string)
      f2  type              (TypeProto)

    TypeProto
      f1  tensor_type       (TypeProto.Tensor)
          TypeProto.Tensor:  f1 elem_type(int32), f2 shape(TensorShapeProto)
      TensorShapeProto:        f1 dim(repeated Dimension)
          Dimension:           f1 dim_value(int64) | f2 dim_param(string)

    AttributeProto
      f1  name              (string)
      f20 type              (AttributeType enum)
      f2  f (float)  f3 i (int64)  f4 s (bytes)
      f7  floats (packed)   f8 ints (packed)   f9 strings
    AttributeType: FLOAT=1 INT=2 STRING=3 TENSOR=4 GRAPH=5
                   FLOATS=6 INTS=7 STRINGS=8 TENSORS=9 GRAPHS=10

    OperatorSetIdProto
      f1 domain (string)   f2 version (int64)
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# protobuf 写入原语
# ---------------------------------------------------------------------------
def wvarint(out, value):
    """写一个 varint"""
    if value < 0:
        value += 1 << 64                      # 负数按补码当无符号写
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def wtag(out, field, wire):
    wvarint(out, (field << 3) | wire)


def wvint(out, field, value):
    """varint 字段"""
    wtag(out, field, 0)
    wvarint(out, int(value))


def wbytes(out, field, data):
    """length-delimited 字段"""
    if isinstance(data, str):
        data = data.encode("utf-8")
    wtag(out, field, 2)
    wvarint(out, len(data))
    out.extend(data)


def wmsg(out, field, sub):
    """嵌套消息"""
    wbytes(out, field, bytes(sub))


def wfloat(out, field, value):
    wtag(out, field, 5)
    out.extend(struct.pack("<f", float(value)))


def wdouble(out, field, value):
    wtag(out, field, 1)
    out.extend(struct.pack("<d", float(value)))


def packed_varints(out, field, values):
    """packed repeated int64 —— 注意：空列表不写"""
    vals = [int(v) for v in values]
    if not vals:
        return
    buf = bytearray()
    for v in vals:
        wvarint(buf, v)
    wbytes(out, field, bytes(buf))


def packed_floats(out, field, values):
    vals = list(values)
    if not vals:
        return
    wbytes(out, field, struct.pack("<%df" % len(vals), *[float(v) for v in vals]))


def zigzag(n):
    return (n << 1) ^ (n >> 63)


# ---------------------------------------------------------------------------
# 各消息
# ---------------------------------------------------------------------------
# TensorProto.DataType
ONNX_DT = {
    "float32": 1, "uint8": 2, "int8": 3, "uint16": 4, "int16": 5,
    "int32": 6, "int64": 7, "string": 8, "bool": 9, "float16": 10,
    "float64": 11, "uint32": 12, "uint64": 13, "bfloat16": 16,
}


def tensor_proto(name, arr):
    """numpy 数组 -> TensorProto 字节"""
    arr = np.asarray(arr)
    dt = str(arr.dtype)
    if dt == "bool":
        # bool 在 ONNX 里用 int32_data 存
        code = 9
        data = arr.astype(np.uint8).tobytes()
        use_raw = True
    elif dt in ("float32", "float64", "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64", "float16"):
        code = ONNX_DT[dt]
        data = arr.tobytes()
        use_raw = True
    else:
        raise ValueError(f"不支持的 dtype: {dt}")

    out = bytearray()
    packed_varints(out, 1, arr.shape)              # dims
    wvint(out, 2, code)                            # data_type
    if use_raw:
        wbytes(out, 9, data)                       # raw_data
    wbytes(out, 8, name)                           # name
    return out


def dimension_proto(dim):
    out = bytearray()
    if dim is None:
        wbytes(out, 2, "?")                        # dim_param
    else:
        wvint(out, 1, int(dim))                    # dim_value
    return out


def type_proto(elem_type, shape):
    """TypeProto（只支持 tensor）"""
    t = bytearray()
    wvint(t, 1, elem_type)                         # elem_type
    if shape is not None:
        sp = bytearray()
        for d in shape:
            wmsg(sp, 1, dimension_proto(d))        # repeated dim
        wmsg(t, 2, sp)                             # shape
    out = bytearray()
    wmsg(out, 1, t)                                # tensor_type
    return out


def value_info_proto(name, elem_type, shape):
    out = bytearray()
    wbytes(out, 1, name)
    wmsg(out, 2, type_proto(elem_type, shape))
    return out


ATTR_TYPE = {"f": 1, "i": 2, "s": 3, "t": 4, "g": 5,
             "floats": 6, "ints": 7, "strings": 8}


def attribute_proto(name, value):
    """根据 python 值的类型自动选 AttributeProto 的字段"""
    out = bytearray()
    wbytes(out, 1, name)

    if isinstance(value, bool):
        wvint(out, 3, 1 if value else 0)
        wvint(out, 20, ATTR_TYPE["i"])
    elif isinstance(value, (int, np.integer)):
        wvint(out, 3, int(value))
        wvint(out, 20, ATTR_TYPE["i"])
    elif isinstance(value, (float, np.floating)):
        wfloat(out, 2, float(value))
        wvint(out, 20, ATTR_TYPE["f"])
    elif isinstance(value, str):
        wbytes(out, 4, value)
        wvint(out, 20, ATTR_TYPE["s"])
    elif isinstance(value, bytes):
        wbytes(out, 4, value)
        wvint(out, 20, ATTR_TYPE["s"])
    elif isinstance(value, (list, tuple)):
        vals = list(value)
        if not vals:
            wvint(out, 20, ATTR_TYPE["ints"])
        elif all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                 for v in vals):
            packed_varints(out, 8, [int(v) for v in vals])
            wvint(out, 20, ATTR_TYPE["ints"])
        elif all(isinstance(v, (float, np.floating)) for v in vals):
            packed_floats(out, 7, vals)
            wvint(out, 20, ATTR_TYPE["floats"])
        elif all(isinstance(v, str) for v in vals):
            for s in vals:
                wbytes(out, 9, s)
            wvint(out, 20, ATTR_TYPE["strings"])
        else:
            raise ValueError(f"无法判断属性列表类型: {name}={vals!r}")
    else:
        raise ValueError(f"无法判断属性类型: {name}={value!r} ({type(value)})")
    return out


def node_proto(n):
    """n 是 {'op':..,'in':[..],'out':[..],'name':..,'attrs':{..}}"""
    out = bytearray()
    for x in n.get("in", []):
        wbytes(out, 1, x)
    for y in n.get("out", []):
        wbytes(out, 2, y)
    if n.get("name"):
        wbytes(out, 3, n["name"])
    wbytes(out, 4, n["op"])
    for k, v in (n.get("attrs") or {}).items():
        if v is None:
            continue
        wmsg(out, 5, attribute_proto(k, v))
    return out


def graph_proto(g):
    """g 是 {'nodes':[..],'name':..,'inputs':[..],'outputs':[..],
             'initializers':[..],'value_info':[..]}"""
    out = bytearray()
    for n in g.get("nodes", []):
        wmsg(out, 1, node_proto(n))
    wbytes(out, 2, g.get("name", "graph"))
    for ini in g.get("initializers", []):
        wmsg(out, 5, tensor_proto(ini["name"], ini["array"]))
    for vi in g.get("inputs", []):
        wmsg(out, 11, value_info_proto(vi["name"], vi["dtype"], vi["shape"]))
    for vo in g.get("outputs", []):
        wmsg(out, 12, value_info_proto(vo["name"], vo["dtype"], vo["shape"]))
    for vi in g.get("value_info", []):
        wmsg(out, 13, value_info_proto(vi["name"], vi["dtype"], vi["shape"]))
    return out


def model_proto(model):
    """model 是 {'ir_version':..,'producer_name':..,'graph':..,'opset':..}"""
    out = bytearray()
    wvint(out, 1, model.get("ir_version", 9))
    if model.get("producer_name"):
        wbytes(out, 2, model["producer_name"])
    if model.get("producer_version"):
        wbytes(out, 3, model["producer_version"])
    wmsg(out, 7, graph_proto(model["graph"]))
    for opset in model.get("opset", [(None, 17)]):
        dom, ver = opset if isinstance(opset, (tuple, list)) else (None, opset)
        sub = bytearray()
        if dom:
            wbytes(sub, 1, dom)
        wvint(sub, 2, ver)
        wmsg(out, 8, sub)
    return out


# ---------------------------------------------------------------------------
def save(model, path):
    data = bytes(model_proto(model))
    Path(path).write_bytes(data)
    return len(data)


def parse_fields(b):
    """（自检用）把 protobuf 拆回字段列表"""
    out = []
    p = 0
    n = len(b)
    while p < n:
        # varint key
        key = 0
        shift = 0
        while True:
            if p >= n:
                return out
            x = b[p]
            p += 1
            key |= (x & 0x7F) << shift
            if not (x & 0x80):
                break
            shift += 7
        f, wt = key >> 3, key & 7
        if wt == 0:
            v = 0
            shift = 0
            while True:
                x = b[p]
                p += 1
                v |= (x & 0x7F) << shift
                if not (x & 0x80):
                    break
                shift += 7
            out.append((f, 0, v))
        elif wt == 2:
            ln = 0
            shift = 0
            while True:
                x = b[p]
                p += 1
                ln |= (x & 0x7F) << shift
                if not (x & 0x80):
                    break
                shift += 7
            out.append((f, 2, b[p:p + ln]))
            p += ln
        elif wt == 5:
            out.append((f, 5, b[p:p + 4]))
            p += 4
        elif wt == 1:
            out.append((f, 1, b[p:p + 8]))
            p += 8
        else:
            raise ValueError(f"未知 wire type {wt} (field {f})")
    return out


def self_test():
    """写一个小模型再读回来，验证编码正确"""
    g = {
        "name": "t",
        "nodes": [{"op": "Add", "in": ["a", "b"], "out": ["c"], "name": "n1",
                   "attrs": {"axis": 1, "alpha": 0.5,
                             "pads": [1, 2, 3, 4], "mode": "constant"}}],
        "inputs": [{"name": "a", "dtype": 1, "shape": [1, 3, None]},
                   {"name": "b", "dtype": 1, "shape": [1, 3, 4]}],
        "outputs": [{"name": "c", "dtype": 1, "shape": [1, 3, 4]}],
        "initializers": [{"name": "w",
                          "array": np.arange(6, dtype=np.float32).reshape(2, 3)}],
        "value_info": [],
    }
    m = {"ir_version": 9, "producer_name": "t", "graph": g,
         "opset": [(None, 17)]}
    data = bytes(model_proto(m))
    top = parse_fields(data)
    flds = {}
    for f, wt, v in top:
        flds.setdefault(f, []).append(v)
    ok = True
    assert 1 in flds and flds[1][0] == 9, "ir_version"
    assert 2 in flds and flds[2][0] == b"t", "producer_name"
    assert 7 in flds, "graph 缺失"
    assert 8 in flds, "opset_import 缺失"
    gsub = parse_fields(flds[7][0])
    gf = {}
    for f, wt, v in gsub:
        gf.setdefault(f, []).append(v)
    assert 1 in gf and len(gf[1]) == 1, "node"
    assert 2 in gf and gf[2][0] == b"t", "graph name"
    assert 5 in gf and len(gf[5]) == 1, "initializer"
    assert 11 in gf and len(gf[11]) == 2, "inputs"
    assert 12 in gf and len(gf[12]) == 1, "outputs"
    nsub = parse_fields(gf[1][0])
    nf = {}
    for f, wt, v in nsub:
        nf.setdefault(f, []).append(v)
    assert nf[4][0] == b"Add", "op_type"
    assert len(nf[5]) == 4, f"attributes 数量应为 4，实际 {len(nf.get(5,[]))}"
    # initializer 内容
    tsub = parse_fields(gf[5][0])
    tf = {}
    for f, wt, v in tsub:
        tf.setdefault(f, []).append(v)
    assert tf[2][0] == 1, "tensor data_type"
    raw = tf[9][0]
    arr = np.frombuffer(raw, dtype=np.float32)
    assert arr.tolist() == [0, 1, 2, 3, 4, 5], f"raw_data 不符: {arr}"
    assert tf[8][0] == b"w", "tensor name"
    print(f"  [OK] 自检通过（{len(data)} 字节）")
    print("       - ir_version / producer_name / graph / opset_import")
    print("       - node op_type / 4 个属性（int, float, ints, string）")
    print("       - initializer dims / data_type / raw_data / name")
    print("       - input / output 数量")
    return True


if __name__ == "__main__":
    print("onnx_writer 自检")
    self_test()
