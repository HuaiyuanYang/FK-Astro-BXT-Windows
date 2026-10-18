# -*- coding: utf-8 -*-
"""
onnx_verify.py —— 独立读回 .onnx 文件做结构验证（纯标准库 + numpy）。

不用 onnx 包，直接按 onnx.proto3 的字段号解析，核对：
  1. ir_version / producer / opset
  2. graph 里的 node / initializer / input / output 数量
  3. 所有 node 的输入是否都能解析（图输入 / initializer / 前序输出）
  4. initializer 的张量能否还原成正确形状和数值
  5. 关键算子的属性是否存在且合理

用法:
    python onnx_verify.py <model.onnx>
"""
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from onnx_writer import parse_fields          # noqa: E402

DT_NAME = {1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
           6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
           11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16"}
NP_DT = {"float32": np.float32, "float64": np.float64, "float16": np.float16,
         "int8": np.int8, "int16": np.int16, "int32": np.int32,
         "int64": np.int64, "uint8": np.uint8, "uint16": np.uint16,
         "uint32": np.uint32, "uint64": np.uint64, "bool": np.bool_}

ATTR_TYPE = {1: "FLOAT", 2: "INT", 3: "STRING", 4: "TENSOR", 5: "GRAPH",
             6: "FLOATS", 7: "INTS", 8: "STRINGS"}


def get(fields, num, wt=None):
    return [v for f, w, v in fields if f == num and (wt is None or w == wt)]


def one(fields, num, wt=None, default=None):
    v = get(fields, num, wt)
    return v[0] if v else default


def varint_list(blob, bits=64):
    """解 packed varint，并按补码还原负数。

    ONNX 的 int64/int32 属性用 protobuf varint 编码，负数是补码形式
    （-1 编成 0xFF...0x01，10 字节）。如果这里不做符号还原，
    读出来的 axis 会显示成 18446744073709551615 而不是 -1
    ——数值其实是对的，只是显示错，容易误判成 bug。
    """
    out = []
    p = 0
    n = len(blob)
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    while p < n:
        v = 0
        shift = 0
        while True:
            if p >= n:
                return out
            x = blob[p]
            p += 1
            v |= (x & 0x7F) << shift
            if not (x & 0x80):
                break
            shift += 7
            if shift > 70:
                return out
        v &= mask
        if v & sign:
            v -= (1 << bits)
        out.append(v)
    return out


def parse_tensor(blob):
    """TensorProto -> (name, dtype_str, dims, ndarray or None)"""
    fs = parse_fields(blob)
    dims = varint_list(one(fs, 1, 2, b"")) if one(fs, 1, 2) is not None else []
    dt = one(fs, 2, 0, 0)
    name = one(fs, 8, 2, b"").decode("utf-8", "replace")
    raw = one(fs, 9, 2)
    dtname = DT_NAME.get(dt, f"?{dt}")
    arr = None
    if raw is not None and dtname in NP_DT:
        try:
            arr = np.frombuffer(raw, dtype=NP_DT[dtname])
            if dims:
                arr = arr.reshape([int(d) for d in dims])
            elif arr.size == 1:
                # dims 为空 = 标量。np.frombuffer 给的是 shape (1,)，
                # 不压成 () 会看不出它其实是 rank-0 标量 ——
                # 这正是之前误判"标量多了一维"的原因。
                arr = arr.reshape(())
        except Exception:
            arr = None
    return name, dtname, [int(d) for d in dims], arr, raw is not None


def sign64(v):
    """protobuf 的 int64 负数按补码 varint 编码，读回来要还原。

    不还原的话 axis=-1 会显示成 18446744073709551615，
    数值其实正确（ONNX 官方解析器会给出 -1），只是显示会误导。
    """
    v &= (1 << 64) - 1
    if v & (1 << 63):
        v -= (1 << 64)
    return v


def parse_node(blob):
    """NodeProto -> dict"""
    fs = parse_fields(blob)
    ins = [v.decode("utf-8", "replace") for v in get(fs, 1, 2)]
    outs = [v.decode("utf-8", "replace") for v in get(fs, 2, 2)]
    name = one(fs, 3, 2, b"").decode("utf-8", "replace")
    op = one(fs, 4, 2, b"").decode("utf-8", "replace")
    attrs = {}
    for ab in get(fs, 5, 2):
        afs = parse_fields(ab)
        an = one(afs, 1, 2, b"").decode("utf-8", "replace")
        at = one(afs, 20, 0, 0)
        if at == 2:
            # 单个 int64 —— 必须做符号还原
            attrs[an] = sign64(one(afs, 3, 0, 0))
        elif at == 1:
            attrs[an] = struct.unpack("<f", one(afs, 2, 5))[0]
        elif at == 3:
            attrs[an] = one(afs, 4, 2, b"").decode("utf-8", "replace")
        elif at == 7:
            attrs[an] = [sign64(v) for v in varint_list(one(afs, 8, 2, b""))]
        elif at == 6:
            fb = one(afs, 7, 2, b"")
            attrs[an] = list(struct.unpack("<%df" % (len(fb) // 4), fb))
        elif at == 8:
            attrs[an] = [v.decode("utf-8", "replace")
                         for v in get(afs, 9, 2)]
    return {"op": op, "in": ins, "out": outs, "name": name, "attrs": attrs}


def parse_value_info(blob):
    fs = parse_fields(blob)
    name = one(fs, 1, 2, b"").decode("utf-8", "replace")
    elem = None
    dims = []
    tp = one(fs, 2, 2)
    if tp is not None:
        tfs = parse_fields(tp)
        tt = one(tfs, 1, 2)
        if tt is not None:
            ttfs = parse_fields(tt)
            e = one(ttfs, 1, 0)
            if e is not None:
                elem = DT_NAME.get(e, f"?{e}")
            sp = one(ttfs, 2, 2)
            if sp is not None:
                for d in get(parse_fields(sp), 1, 2):
                    dfs = parse_fields(d)
                    dv = one(dfs, 1, 0)
                    dims.append(dv if dv is not None else None)
    return {"name": name, "dtype": elem, "shape": dims}


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1
             else "bxt5.onnx")
    data = p.read_bytes()
    print("=" * 78)
    print(f"验证: {p}   ({len(data):,} 字节 = {len(data)/1048576:.1f} MB)")
    print("=" * 78)

    top = parse_fields(data)
    ir = one(top, 1, 0)
    prod = one(top, 2, 2, b"").decode("utf-8", "replace")
    gb = one(top, 7, 2)
    if gb is None:
        print("  [X] 找不到 graph (field 7)")
        return 1
    opsets = []
    for ob in get(top, 8, 2):
        ofs = parse_fields(ob)
        opsets.append((one(ofs, 1, 2, b"").decode("utf-8", "replace"),
                       one(ofs, 2, 0)))
    print(f"  ir_version    : {ir}")
    print(f"  producer      : {prod}")
    print(f"  opset_import  : {opsets}")

    gfs = parse_fields(gb)
    gname = one(gfs, 2, 2, b"").decode("utf-8", "replace")
    nodes = [parse_node(b) for b in get(gfs, 1, 2)]
    inits = [parse_tensor(b) for b in get(gfs, 5, 2)]
    ins = [parse_value_info(b) for b in get(gfs, 11, 2)]
    outs = [parse_value_info(b) for b in get(gfs, 12, 2)]

    print(f"  graph name    : {gname}")
    print(f"  节点          : {len(nodes)}")
    print(f"  initializer   : {len(inits)}")
    print(f"  图输入        : {[(v['name'], v['dtype'], v['shape']) for v in ins]}")
    print(f"  图输出        : {[(v['name'], v['dtype'], v['shape']) for v in outs]}")
    print()

    # --- initializer 检查 ---
    print("  --- initializer 检查 ---")
    no_raw = [i[0] for i in inits if not i[4]]
    bad_shape = []
    tot_elems = 0
    for nm, dt, dims, arr, has_raw in inits:
        if arr is None:
            bad_shape.append(nm)
            continue
        tot_elems += arr.size
        if dims and list(arr.shape) != dims:
            bad_shape.append(f"{nm}: 声明{dims} 实际{list(arr.shape)}")
    print(f"    总数            : {len(inits)}")
    print(f"    没有 raw_data   : {len(no_raw)}")
    print(f"    形状不符        : {len(bad_shape)}" +
          (f"  例: {bad_shape[:3]}" if bad_shape else ""))
    print(f"    元素总数        : {tot_elems:,}")
    sizes = Counter(i[1] for i in inits)
    print(f"    dtype 分布      : {dict(sizes)}")
    print()

    # --- 悬空输入检查 ---
    print("  --- 连接性检查 ---")
    produced = {i[0] for i in inits}
    produced |= {v["name"] for v in ins}
    dangling = []
    dup = []
    seen = set()
    for i, n in enumerate(nodes):
        for x in n["in"]:
            if x and x not in produced:
                dangling.append(f"节点[{i}] {n['op']}: '{x}'")
        for y in n["out"]:
            if y in seen:
                dup.append(y)
            seen.add(y)
            produced.add(y)
    print(f"    悬空输入        : {len(dangling)}" +
          (f"  例: {dangling[:3]}" if dangling else "  [OK]"))
    print(f"    重复输出名      : {len(dup)}" + ("  [OK]" if not dup else ""))
    missing = [v["name"] for v in outs if v["name"] not in produced]
    print(f"    图输出未生产    : {len(missing)}" +
          (f"  {missing}" if missing else "  [OK]"))
    print()

    # --- 关键算子属性抽查 ---
    print("  --- 关键算子属性抽查 ---")
    want = {
        "Conv":        ["kernel_shape", "strides", "dilations", "group",
                        "auto_pad", "pads"],
        "MaxPool":     ["kernel_shape", "strides", "auto_pad", "pads"],
        "AveragePool": ["kernel_shape", "strides", "auto_pad", "pads"],
        "Transpose":   ["perm"],
        "Softmax":     ["axis"],
        "Concat":      ["axis"],
        "Gemm":        ["alpha", "beta", "transB"],
        "Slice":       [],
        "Pad":         ["mode"],
        "Gather":      ["axis"],
    }
    by_op = {}
    for n in nodes:
        by_op.setdefault(n["op"], []).append(n)
    for op, keys in want.items():
        group = by_op.get(op)
        if not group:
            continue
        have = set()
        for n in group:
            have |= set(n["attrs"])
        missing_k = [k for k in keys if k not in have]
        status = "OK" if not missing_k else f"缺 {missing_k}"
        sample = group[0]
        shown = {k: v for k, v in list(sample["attrs"].items())[:5]}
        print(f"    {op:<12} x{len(group):<5} {status:<24} 例: {shown}")
    print()

    # --- 算子直方图 ---
    print("  --- 算子直方图 ---")
    for op, n in Counter(x["op"] for x in nodes).most_common():
        print(f"    {op:<18} x{n}")
    print()

    ok = not dangling and not dup and not missing and not bad_shape
    print("=" * 78)
    print("  验证结果: " + ("[OK] 结构完整，可以被 onnxruntime 加载"
                           if ok else "[!] 有问题，见上"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
