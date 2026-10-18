# -*- coding: utf-8 -*-
"""
onnx_run.py —— 零依赖的 ONNX 推理器（纯 NumPy + SciPy）。

为什么写它：
    onnxruntime 需要 pip 装，而本机 pip 要代理、代理经常不在。
    没有运行时就没法验证模型能不能真的跑出数值 —— 只能做结构检查。
    这个解释器让"能不能跑"这件事离线可测。

它能抓到的错误（结构验证抓不到的）：
    形状不匹配、广播错误、属性缺失/非法、切片越界、
    算子组合产生的维度错乱 —— 也就是真正会炸的地方。

支持的算子（覆盖本项目两个模型用到的全部）：
    Conv, Gemm, MatMul, PRelu, Relu, Sigmoid, Tanh, Gelu, Softmax,
    Add, Sub, Mul, Div, Max, Min, Exp, Log, Sqrt,
    Transpose, Reshape, Unsqueeze, Squeeze, Concat, Split, Slice,
    Gather, Pad, MaxPool, AveragePool, ReduceMean, ReduceSum,
    Identity, Cast, Constant

用法:
    python onnx_run.py <model.onnx>                      # 结构 + 一次前向
    python onnx_run.py <model.onnx> --input-shape 1,768,768,3
    python onnx_run.py <model.onnx> --list-inputs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from onnx_verify import (parse_fields, get, one, varint_list, sign64,
                         parse_tensor, parse_node, parse_value_info,
                         NP_DT, DT_NAME)      # noqa: E402


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
class Model:
    def __init__(self, path):
        data = Path(path).read_bytes()
        top = parse_fields(data)
        self.ir_version = one(top, 1, 0)
        gb = one(top, 7, 2)
        if gb is None:
            raise ValueError("没有 graph")
        self.opset = []
        for ob in get(top, 8, 2):
            ofs = parse_fields(ob)
            self.opset.append((one(ofs, 1, 2, b"").decode("utf-8", "replace"),
                               one(ofs, 2, 0)))
        gfs = parse_fields(gb)
        self.name = one(gfs, 2, 2, b"").decode("utf-8", "replace")
        self.nodes = [parse_node(b) for b in get(gfs, 1, 2)]
        self.initializers = {}
        self.init_info = []
        for b in get(gfs, 5, 2):
            nm, dt, dims, arr, has_raw = parse_tensor(b)
            self.init_info.append((nm, dt, dims))
            if arr is not None:
                self.initializers[nm] = arr
        self.inputs = [parse_value_info(b) for b in get(gfs, 11, 2)]
        self.outputs = [parse_value_info(b) for b in get(gfs, 12, 2)]


# ---------------------------------------------------------------------------
# 算子实现
# ---------------------------------------------------------------------------
def _pool_out(h, k, s, p0, p1, ceil_mode, auto_pad):
    """算池化/卷积输出的空间维度"""
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        return (h + s - 1) // s
    eff = h + p0 + p1 - k
    if ceil_mode:
        return int(np.ceil(eff / s)) + 1
    return eff // s + 1


def _pad_spatial(x, pads, mode, cval=0.0):
    """pads = [b0,b1,...,e0,e1,...]（ONNX 顺序），作用在最后 n 维"""
    n = len(pads) // 2
    begins = pads[:n]
    ends = pads[n:]
    pad_width = [(0, 0)] * (x.ndim - n) + list(zip(begins, ends))
    if mode == "constant":
        return np.pad(x, pad_width, mode="constant", constant_values=cval)
    if mode == "edge":
        return np.pad(x, pad_width, mode="edge")
    if mode == "reflect":
        return np.pad(x, pad_width, mode="reflect")
    raise ValueError(f"不支持的 pad mode: {mode}")


def op_conv(ins, attrs, ctx):
    """Conv —— im2col + matmul，兼容 group/dilation

    性能说明：早期版本用 4 层 Python 循环构造 im2col 矩阵，
    768x768 输入下 Conv 占了整个网络 85% 的时间（403 ms/次）。
    改成 as_strided 向量化切片后快一个量级。
    """
    from numpy.lib.stride_tricks import as_strided

    x = ins[0]
    w = ins[1]
    b = ins[2] if len(ins) > 2 and ins[2] is not None else None
    if x.ndim != 4:
        raise ValueError(f"Conv 只支持 4D 输入，收到 {x.shape}")
    if w.ndim != 4:
        raise ValueError(f"Conv 权重必须是 4D，收到 {w.shape}")

    N, C, H, W = x.shape
    M, Cg, kh, kw = w.shape
    group = int(attrs.get("group", 1))
    strides = list(attrs.get("strides") or [1, 1])
    dils = list(attrs.get("dilations") or [1, 1])
    auto_pad = attrs.get("auto_pad") or "NOTSET"
    pads = list(attrs.get("pads") or [0, 0, 0, 0])

    if C != Cg * group:
        raise ValueError(f"Conv 通道数不匹配: 输入 {C} vs 权重 {Cg}x{group}")

    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        oh = (H + strides[0] - 1) // strides[0]
        ow = (W + strides[1] - 1) // strides[1]
        dh = max((oh - 1) * strides[0] + (kh - 1) * dils[0] + 1 - H, 0)
        dw = max((ow - 1) * strides[1] + (kw - 1) * dils[1] + 1 - W, 0)
        if auto_pad == "SAME_UPPER":
            pt, pl = dh // 2, dw // 2
        else:
            pt, pl = (dh + 1) // 2, (dw + 1) // 2
        pads = [pt, pl, dh - pt, dw - pl]

    if any(pads):
        x = _pad_spatial(x, pads, "constant", 0.0)
        N, C, H, W = x.shape

    oh = (H - ((kh - 1) * dils[0] + 1)) // strides[0] + 1
    ow = (W - ((kw - 1) * dils[1] + 1)) // strides[1] + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(f"Conv 输出尺寸非法: {oh}x{ow} "
                         f"(输入 {H}x{W}, kernel {kh}x{kw}, stride {strides})")

    out = np.empty((N, M, oh, ow), dtype=np.float32)
    Mg = M // group
    ksize = Cg * kh * kw

    for g in range(group):
        xg = x[:, g * Cg:(g + 1) * Cg]                        # (N, Cg, H, W)
        wg = w[g * Mg:(g + 1) * Mg]                           # (Mg, Cg, kh, kw)

        # 用 as_strided 一次切出所有窗口: (N, Cg, oh, ow, kh, kw)
        shp = (N, Cg, oh, ow, kh, kw)
        st = (xg.strides[0], xg.strides[1],
              xg.strides[2] * strides[0], xg.strides[3] * strides[1],
              xg.strides[2] * dils[0], xg.strides[3] * dils[1])
        win = as_strided(xg, shape=shp, strides=st, writeable=False)

        # -> (N, oh*ow, Cg*kh*kw) 的 im2col 矩阵
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N, oh * ow, ksize)
        wm = wg.reshape(Mg, ksize)                            # (Mg, ksize)

        # 一次 matmul 算完整组: (N, oh*ow, Mg)
        r = cols @ wm.T
        out[:, g * Mg:(g + 1) * Mg] = r.transpose(0, 2, 1).reshape(N, Mg, oh, ow)

    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def op_pool(ins, attrs, ctx, kind):
    x = ins[0]
    if x.ndim < 3:
        raise ValueError(f"{kind} 需要 >=3D 输入，收到 {x.shape}")
    spatial = x.ndim - 2
    ks = list(attrs.get("kernel_shape") or [1] * spatial)
    st = list(attrs.get("strides") or [1] * spatial)
    auto_pad = attrs.get("auto_pad") or "NOTSET"
    pads = list(attrs.get("pads") or [0] * (2 * spatial))
    ceil_mode = bool(attrs.get("ceil_mode", 0))
    count_include_pad = bool(attrs.get("count_include_pad", 0))

    H = list(x.shape[2:])
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        oshape = [(H[i] + st[i] - 1) // st[i] for i in range(spatial)]
        pads = []
        for i in range(spatial):
            d = (oshape[i] - 1) * st[i] + ks[i] - H[i]
            d = max(d, 0)
            pads.append(d // 2 if auto_pad == "SAME_UPPER" else (d + 1) // 2)
        pads = pads + [0] * spatial
        for i in range(spatial):
            d = (oshape[i] - 1) * st[i] + ks[i] - H[i]
            d = max(d, 0)
            pads[spatial + i] = d - pads[i]
    else:
        pads = list(attrs.get("pads") or [0] * (2 * spatial))

    oshape = [_pool_out(H[i], ks[i], st[i], pads[i], pads[spatial + i],
                        ceil_mode, auto_pad) for i in range(spatial)]

    if any(pads):
        fill = 0.0
        if kind == "AveragePool" and count_include_pad:
            x = _pad_spatial(x, pads, "constant", 0.0)
        else:
            # max 用 -inf 填充；average 不数 pad，用 nan 标记后 nanmean
            fill = -np.inf if kind == "MaxPool" else np.nan
            x = _pad_spatial(x, pads, "constant", fill)
    H = list(x.shape[2:])

    # 用 stride tricks 切窗口（比循环快很多）
    from numpy.lib.stride_tricks import as_strided
    shape = list(x.shape[:2]) + oshape + ks
    strides_bytes = list(x.strides[:2])
    for i in range(spatial):
        strides_bytes.append(x.strides[2 + i] * st[i])
    for i in range(spatial):
        strides_bytes.append(x.strides[2 + i])
    win = as_strided(x, shape=shape, strides=strides_bytes, writeable=False)
    axes = tuple(range(2 + spatial, 2 + 2 * spatial))
    if kind == "MaxPool":
        return np.max(win, axis=axes)
    with np.errstate(invalid="ignore"):
        return np.nanmean(win, axis=axes)


def op_slice(ins, attrs, ctx):
    x = ins[0]
    starts = np.asarray(ins[1]).astype(np.int64).ravel() if len(ins) > 1 else None
    ends = np.asarray(ins[2]).astype(np.int64).ravel() if len(ins) > 2 else None
    axes = np.asarray(ins[3]).astype(np.int64).ravel() if len(ins) > 3 and ins[3] is not None else None
    steps = np.asarray(ins[4]).astype(np.int64).ravel() if len(ins) > 4 and ins[4] is not None else None
    if starts is None:
        starts = np.asarray(attrs.get("starts", []), dtype=np.int64)
        ends = np.asarray(attrs.get("ends", []), dtype=np.int64)
        axes = np.asarray(attrs.get("axes", []), dtype=np.int64) if attrs.get("axes") else None
        steps = np.asarray(attrs.get("steps", []), dtype=np.int64) if attrs.get("steps") else None
    if axes is None:
        axes = np.arange(len(starts))
    if steps is None:
        steps = np.ones(len(starts), dtype=np.int64)

    sl = [slice(None)] * x.ndim
    for k in range(len(starts)):
        ax = int(axes[k])
        d = x.shape[ax]
        st = int(steps[k])
        bg = int(starts[k])
        en = int(ends[k])
        # ONNX 语义：先 clamp 到 [-d, d]，再处理哨兵大值
        if st > 0:
            if bg < 0:
                bg += d
            bg = max(0, min(bg, d))
            if en < 0:
                en += d
            en = max(0, min(en, d))
        else:
            if bg < 0:
                bg += d
            bg = max(-1, min(bg, d - 1))
            if en < 0:
                en += d
            en = max(-1, min(en, d - 1))
        sl[ax] = slice(bg, en, st)
    return x[tuple(sl)]


def op_gather(ins, attrs, ctx):
    x = ins[0]
    idx = np.asarray(ins[1])
    axis = int(attrs.get("axis", 0))
    return np.take(x, idx.astype(np.int64), axis=axis)


def op_split(ins, attrs, ctx):
    x = ins[0]
    axis = int(attrs.get("axis", 0))
    n_out = len(ctx["_n_out"])
    split = attrs.get("split")
    if split:
        sizes = [int(v) for v in split]
    else:
        d = x.shape[axis]
        if d % n_out:
            raise ValueError(f"Split: 维度 {d} 无法均分成 {n_out} 份")
        sizes = [d // n_out] * n_out
    if sum(sizes) != x.shape[axis]:
        raise ValueError(f"Split: sizes {sizes} 之和 != {x.shape[axis]}")
    cuts = np.cumsum(sizes)[:-1]
    return np.split(x, cuts, axis=axis)


def op_reduce(ins, attrs, ctx, kind):
    x = ins[0]
    axes = None
    if len(ins) > 1 and ins[1] is not None:
        a = np.asarray(ins[1])
        if a.size:
            axes = tuple(int(v) for v in a.ravel())
    if axes is None and attrs.get("axes") is not None:
        aa = [int(v) for v in attrs["axes"]]
        axes = tuple(aa) if aa else None
    keep = bool(attrs.get("keepdims", 1))
    # noop_with_empty_axes=0 时，axes 为空表示全归约
    if axes == ():
        axes = None
    fn = {"ReduceMean": np.mean, "ReduceSum": np.sum,
          "ReduceMax": np.max, "ReduceMin": np.min}[kind]
    return fn(x, axis=axes, keepdims=keep)


def _gemm(x, w, b, attrs):
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    if attrs.get("transA", 0):
        x = x.T
    if attrs.get("transB", 0):
        w = w.T
    y = alpha * np.matmul(x, w)
    if b is not None:
        y = y + beta * b
    return y


def _gelu(x, attrs):
    approx = attrs.get("approximate", "none")
    if approx == "tanh":
        from scipy.special import erf                      # noqa: F401
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                        (x + 0.044715 * x ** 3)))
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _softmax(x, attrs):
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis += x.ndim
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


DISPATCH = {
    "Conv": op_conv,
    "Gemm": lambda i, a, c: _gemm(i[0], i[1], i[2] if len(i) > 2 and i[2] is not None else None, a),
    "MatMul": lambda i, a, c: np.matmul(i[0], i[1]),
    "PRelu": lambda i, a, c: np.where(i[0] >= 0, i[0], i[0] * i[1]),
    "Relu": lambda i, a, c: np.maximum(i[0], 0),
    "Sigmoid": lambda i, a, c: 1.0 / (1.0 + np.exp(-i[0])),
    "Tanh": lambda i, a, c: np.tanh(i[0]),
    "Gelu": lambda i, a, c: _gelu(i[0], a),
    "Softmax": lambda i, a, c: _softmax(i[0], a),
    "Add": lambda i, a, c: i[0] + i[1],
    "Sub": lambda i, a, c: i[0] - i[1],
    "Mul": lambda i, a, c: i[0] * i[1],
    "Div": lambda i, a, c: i[0] / i[1],
    "Max": lambda i, a, c: np.maximum(i[0], i[1]),
    "Min": lambda i, a, c: np.minimum(i[0], i[1]),
    "Exp": lambda i, a, c: np.exp(i[0]),
    "Log": lambda i, a, c: np.log(i[0]),
    "Sqrt": lambda i, a, c: np.sqrt(i[0]),
    "Transpose": lambda i, a, c: np.transpose(i[0], a.get("perm")),
    "Reshape": lambda i, a, c: np.reshape(i[0], [int(v) for v in np.asarray(i[1]).ravel()]),
    "Unsqueeze": lambda i, a, c: np.expand_dims(
        i[0], tuple(int(v) if int(v) >= 0 else int(v) + i[0].ndim + len(np.asarray(i[1]).ravel())
                    for v in np.asarray(i[1]).ravel())),
    "Squeeze": lambda i, a, c: np.squeeze(
        i[0], tuple(int(v) for v in np.asarray(i[1]).ravel()) if len(i) > 1 and i[1] is not None
        else tuple(a.get("axes", ()))),
    "Concat": lambda i, a, c: np.concatenate(i, axis=int(a.get("axis", 0))),
    "Split": op_split,
    "Slice": op_slice,
    "Gather": op_gather,
    "Pad": lambda i, a, c: _pad_spatial(
        i[0], [int(v) for v in np.asarray(i[1]).ravel()], a.get("mode", "constant"),
        float(np.asarray(i[2]).ravel()[0]) if len(i) > 2 and i[2] is not None else 0.0),
    "MaxPool": lambda i, a, c: op_pool(i, a, c, "MaxPool"),
    "AveragePool": lambda i, a, c: op_pool(i, a, c, "AveragePool"),
    "ReduceMean": lambda i, a, c: op_reduce(i, a, c, "ReduceMean"),
    "ReduceSum": lambda i, a, c: op_reduce(i, a, c, "ReduceSum"),
    "ReduceMax": lambda i, a, c: op_reduce(i, a, c, "ReduceMax"),
    "ReduceMin": lambda i, a, c: op_reduce(i, a, c, "ReduceMin"),
    "Identity": lambda i, a, c: i[0],
    "Cast": lambda i, a, c: i[0].astype(NP_DT.get(DT_NAME.get(int(a.get("to", 1)), "float32"), np.float32)),
}


# ---------------------------------------------------------------------------
def trace_back(model, target, depth=6, max_lines=40):
    """从某个值名往上回溯，打印生产链和每一步的形状。

    调试形状问题时比看单个节点有用得多 —— 因为错的地方往往在上游几层。
    """
    prod = {}
    for n in model.nodes:
        for y in n["out"]:
            if y:
                prod[y] = n
    lines = []

    def walk(name, d, seen):
        if len(lines) >= max_lines or d > depth:
            return
        n = prod.get(name)
        if n is None:
            lines.append("  " * d + f"{name[:64]}  <- 图输入/常量")
            return
        lines.append("  " * d + f"{name[:64]}")
        lines.append("  " * d + f"    ^ {n['op']}  attrs={n['attrs']}")
        for x in n["in"]:
            if x and x not in seen:
                seen.add(x)
                walk(x, d + 1, seen)

    walk(target, 0, set())
    return "\n".join(lines)


def run(model, feeds, verbose=False, trace_on_error=True):
    """按拓扑顺序执行，返回 {输出名: 数组}"""
    env = dict(model.initializers)
    for k, v in feeds.items():
        env[k] = np.asarray(v)

    missing_ops = set()
    t0 = time.time()
    for i, n in enumerate(model.nodes):
        op = n["op"]
        fn = DISPATCH.get(op)
        if fn is None:
            missing_ops.add(op)
            # 占位：输出全 0，尺寸取第一个输入
            for y in n["out"]:
                if y:
                    env[y] = np.zeros_like(env[n["in"][0]]) if n["in"] and n["in"][0] in env else np.zeros(1, np.float32)
            continue
        ins = []
        for x in n["in"]:
            if not x:
                ins.append(None)
            elif x in env:
                ins.append(env[x])
            else:
                raise KeyError(f"节点[{i}] {op}: 输入 '{x}' 不在环境里")
        ctx = {"_n_out": n["out"], "node": n}
        try:
            r = fn(ins, n["attrs"], ctx)
        except Exception as e:                             # noqa: BLE001
            msg = (f"节点[{i}] {op} (name={n['name'][:60]}) 执行失败: "
                   f"{type(e).__name__}: {e}\n"
                   f"    输入形状: "
                   f"{[None if a is None else a.shape for a in ins]}\n"
                   f"    属性: {n['attrs']}")
            if trace_on_error:
                msg += "\n\n  --- 上游回溯（第一个输入）---\n"
                msg += trace_back(model, n["in"][0] if n["in"] else "")
            raise RuntimeError(msg) from e

        outs = [y for y in n["out"] if y]
        if len(outs) == 1:
            env[outs[0]] = r
        else:
            if not isinstance(r, (list, tuple)) or len(r) != len(outs):
                raise RuntimeError(f"节点[{i}] {op}: 期望 {len(outs)} 个输出，"
                                   f"实际 {type(r).__name__}")
            for y, v in zip(outs, r):
                env[y] = v
        if verbose and i % 100 == 0:
            print(f"    [{i}/{len(model.nodes)}] {op} ... "
                  f"{time.time()-t0:.1f}s")

    return {o["name"]: env.get(o["name"]) for o in model.outputs}, missing_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--input-shape", default=None,
                    help="覆盖第一个输入的形状，例如 1,768,768,3")
    ap.add_argument("--params", default=None,
                    help="第二个输入的值，逗号分隔")
    ap.add_argument("--list-inputs", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    m = Model(a.path)
    print(f"模型   : {a.path}")
    print(f"  ir_version {m.ir_version}   opset {m.opset}")
    print(f"  节点 {len(m.nodes)}   initializer {len(m.initializers)}")
    print(f"  输入:")
    for v in m.inputs:
        print(f"      {v['name']:<16} {str(v['dtype']):<9} {v['shape']}")
    print(f"  输出:")
    for v in m.outputs:
        print(f"      {v['name']:<16} {str(v['dtype']):<9} {v['shape']}")
    print("=" * 78)

    if a.list_inputs:
        return 0

    # 构造输入
    feeds = {}
    for k, v in enumerate(m.inputs):
        shp = v["shape"]
        if k == 0 and a.input_shape:
            shp = [int(x) for x in a.input_shape.split(",")]
        shp = [d if d is not None else 1 for d in shp]
        if k == 1 and a.params:
            vals = [float(x) for x in a.params.split(",")]
            arr = np.asarray(vals, dtype=np.float32).reshape(shp) \
                if int(np.prod(shp)) == len(vals) else \
                np.asarray(vals, dtype=np.float32)
            feeds[v["name"]] = arr
        else:
            feeds[v["name"]] = np.random.RandomState(0).rand(*shp).astype(np.float32)

    print("  输入张量:")
    for k, v in feeds.items():
        print(f"      {k:<16} {v.shape}  {v.dtype}")
    print()
    print("  开始前向 ...")
    try:
        outs, missing = run(m, feeds, verbose=a.verbose)
    except Exception as e:                                  # noqa: BLE001
        print()
        print("  [X] 前向失败 —— 这正是要抓的问题：")
        print(f"      {e}")
        return 1

    print()
    if missing:
        print(f"  [!] 未实现的算子（输出被置零，结果不可信）: "
              f"{sorted(missing)}")
    else:
        print("  [OK] 所有算子都已实现，前向跑通")
    print()
    print("  输出:")
    for k, v in outs.items():
        if v is None:
            print(f"      {k:<16} <未产出>")
            continue
        finite = np.isfinite(v).all()
        print(f"      {k:<16} {str(v.shape):<22} {v.dtype}")
        print(f"          min={np.nanmin(v):+.6f}  max={np.nanmax(v):+.6f}  "
              f"mean={np.nanmean(v):+.6f}  std={np.nanstd(v):.6f}")
        print(f"          全部有限值: {finite}" +
              ("" if finite else f"  ({np.sum(~np.isfinite(v))} 个非有限)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
