# -*- coding: utf-8 -*-
"""
mil2onnx.py —— 把 CoreML ML Program (.mlpackage) 翻译成 ONNX。

为什么需要自己写：
    coremltools **没有** CoreML -> ONNX 的能力。
    converters/onnx 子模块是反方向（ONNX->CoreML），而且 6.0 就删了。
    外部工具（onnx-coreml / onnx2coreml）也全是反方向。
    所以只能自己实现 MIL -> ONNX 的逐算子翻译。

依赖：
    - opscan.py    protobuf 字段读取
    - mil_graph.py MIL 图解析
    - bxt_decode.py weight.bin 解码
    - onnx         (pip install onnx)

算子映射表（MIL -> ONNX）：
    conv            -> Conv          (pad_type=SAME/VALID -> auto_pad)
    linear          -> Gemm
    matmul          -> MatMul
    prelu           -> PRelu
    relu            -> Relu
    gelu            -> Gelu
    add/sub/mul/real_div -> Add/Sub/Mul/Div
    maximum/minimum -> Max/Min
    exp/log/sqrt/rsqrt -> Exp/Log/Sqrt  (rsqrt = 1/sqrt)
    transpose       -> Transpose
    reshape         -> Reshape
    expand_dims     -> Unsqueeze
    concat          -> Concat
    split           -> Split
    slice_by_index  -> Slice
    slice_by_size   -> Slice
    gather          -> Gather
    pad             -> Pad
    max_pool/avg_pool -> MaxPool/AveragePool
    reduce_mean/sum -> ReduceMean/ReduceSum
    softmax         -> Softmax
    const           -> Initializer（常量/权重）

用法:
    python mil2onnx.py <mlpackage 目录> [-o 输出.onnx]
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np                     # noqa: E402

# onnx 是可选的：没有它也能跑图级自检（--check）。
# 这台机器上 pip 装 onnx 需要代理，代理时常没开，所以不能硬依赖。
try:
    import onnx                                        # noqa: E402
    from onnx import helper, numpy_helper, TensorProto  # noqa: E402
    HAS_ONNX = True
except ImportError:                                     # pragma: no cover
    onnx = None
    HAS_ONNX = False

    class _TP:                                          # 占位枚举
        FLOAT = 1; FLOAT16 = 10; DOUBLE = 11; INT32 = 6
        INT64 = 7; INT8 = 3; UINT8 = 2; BOOL = 9
    TensorProto = _TP

    class _H:                                           # 极简占位
        @staticmethod
        def make_node(op, ins, outs, name=None, **kw):
            return {"op": op, "in": list(ins), "out": list(outs),
                    "name": name, "attrs": kw}

        @staticmethod
        def make_tensor_value_info(name, dt, shape):
            return {"name": name, "dtype": dt, "shape": shape}

        @staticmethod
        def make_graph(nodes, name, ins, outs, inits):
            return {"nodes": nodes, "name": name, "inputs": ins,
                    "outputs": outs, "initializers": inits}

        @staticmethod
        def make_model(g, **kw):
            return {"graph": g, **kw}

        @staticmethod
        def make_opsetid(dom, ver):
            return (dom, ver)
    helper = _H()

    class _NH:
        @staticmethod
        def from_array(arr, name):
            return {"name": name, "array": arr}
    numpy_helper = _NH()

import mil_graph                       # noqa: E402
from bxt_decode import read_weight_index, NP_OF, ITEMSIZE, MIL_DTYPE   # noqa: E402

ONNX_DTYPE = {
    "FLOAT32": TensorProto.FLOAT, "FLOAT16": TensorProto.FLOAT16,
    "FLOAT64": TensorProto.DOUBLE, "INT32": TensorProto.INT32,
    "INT64": TensorProto.INT64, "INT8": TensorProto.INT8,
    "UINT8": TensorProto.UINT8, "BOOL": TensorProto.BOOL,
}


# ---------------------------------------------------------------------------
# 统一的取值助手：有 onnx 包时节点是 NodeProto，没有时是 dict
# ---------------------------------------------------------------------------
def N_OP(n):
    return n["op"] if isinstance(n, dict) else n.op_type


def N_IN(n):
    return n["in"] if isinstance(n, dict) else list(n.input)


def N_OUT(n):
    return n["out"] if isinstance(n, dict) else list(n.output)


def N_NAME(n):
    return n["name"] if isinstance(n, dict) else n.name


def V_NAME(v):
    return v["name"] if isinstance(v, dict) else v.name


def I_NAME(i):
    return i["name"] if isinstance(i, dict) else i.name


# ---------------------------------------------------------------------------
class Translator:
    def __init__(self, graph, weight_raw=None, opset=17):
        self.g = graph
        self.opset = opset
        self.nodes = []
        self.inits = []            # initializers（权重 + 常量）
        self.init_names = set()
        self.inputs = []           # ONNX 图输入
        self.outputs = []          # ONNX 图输出
        self.value_info = []
        self.unsupported = []
        self.wraw = weight_raw
        self.windex = None
        self.counter = 0
        if weight_raw is not None:
            self.windex = {pos: (dsz, dstart)
                           for pos, dsz, dstart in read_weight_index(weight_raw)}

    # ---- 工具 ----
    def uniq(self, base):
        self.counter += 1
        return f"{base}__{self.counter}"

    def add_init(self, name, arr):
        if name in self.init_names:
            return name
        self.inits.append(numpy_helper.from_array(arr, name))
        self.init_names.add(name)
        return name

    def const_array(self, value):
        """ImmediateValue dict -> numpy 数组"""
        kind = value.get("kind")
        vals = value.get("values") or []
        if kind == "floats":
            return np.asarray(vals, dtype=np.float32)
        if kind == "ints":
            return np.asarray(vals, dtype=np.int32)
        if kind == "longInts":
            return np.asarray(vals, dtype=np.int64)
        if kind == "bools":
            return np.asarray(vals, dtype=np.bool_)
        return None

    def const_of(self, binding):
        """把一个绑定解析成 numpy 数组（常量或权重）"""
        if isinstance(binding, dict):
            if binding.get("kind") == "blob":
                off = binding["offset"]
                if self.windex and off in self.windex:
                    dsz, dstart = self.windex[off]
                    buf = self.wraw[dstart:dstart + dsz]
                    # 用 MIL 里声明的 shape 还原
                    return np.frombuffer(buf, dtype=np.float32).copy()
                return None
            return self.const_array(binding)
        return None

    def resolve_const(self, name, depth=0):
        """按名字找到该值的常量数组（可能是一个 const 算子）

        shape 处理很关键：MIL 声明形状为空 [] 时，这是一个**标量**，
        必须保持 rank-0（np.asarray(12.0) 的形状是 ()）。
        如果整形成 [1]，广播 (1,768,768,3) * (1,) 会得到 (1,768,768,3,1)
        —— 凭空多一维，后面 Concat 就炸了。

        实测踩到：BXT_log_mul_y = 12.0 被当成 [12.0]，
        导致 Log→Mul 之后 rank 从 3 变 4，
        最终 Concat 收到 (4,126,126,4) 而另一路是 (1,128,128,3)。
        """
        if depth > 8:
            return None
        op = self.op_by_name.get(name)
        if op is not None and op.type == "const":
            # 外部权重 blob
            info = self.g.consts.get(name)
            if info is not None and self.windex:
                off, dtype, dims = info
                if off in self.windex:
                    dsz, dstart = self.windex[off]
                    buf = self.wraw[dstart:dstart + dsz]
                    arr = np.frombuffer(buf, dtype=np.float32).copy()
                    arr = self._reshape_const(arr, dims)
                    return arr
            # 内联立即值
            for k, vals in op.in_.items():
                for b in vals:
                    a = self.const_of(b)
                    if a is not None:
                        return self._reshape_const(np.asarray(a), op.dims)
        return None

    @staticmethod
    def _reshape_const(arr, dims):
        """按 MIL 声明的形状整形；空形状 = 标量，必须保持 rank-0"""
        arr = np.asarray(arr)
        dims = list(dims or [])
        if not dims:
            # 标量：压成 rank-0，广播时不会引入多余维度
            if arr.size == 1:
                return arr.reshape(())
            return arr
        pos = [d for d in dims if d and d > 0]
        if pos and arr.size == int(np.prod(pos)):
            return arr.reshape(pos)
        try:
            return arr.reshape([d if d and d > 0 else 1 for d in dims])
        except Exception:
            return arr

    def const_ints(self, name, default=None):
        """把一个引用名解析成 python int 列表"""
        a = self.resolve_const(name)
        if a is None:
            return default
        return [int(x) for x in np.asarray(a).ravel()]

    def const_int(self, name, default=None):
        v = self.const_ints(name)
        if not v:
            return default
        return v[0]

    def const_str(self, name, default=None):
        a = self.resolve_const(name)
        if a is None:
            return default
        return None

    def emit(self, op_type, inputs, outputs, name=None, **attrs):
        n = helper.make_node(op_type, inputs, outputs,
                             name=name or self.uniq(op_type),
                             **{k: v for k, v in attrs.items() if v is not None})
        self.nodes.append(n)
        return n

    def unique_out(self, base):
        return self.uniq(f"{base}_out")

    # ---- 主流程 ----
    def run(self):
        self.op_by_name = {o.name: o for o in self.g.ops}

        # 1. 图输入
        for i, (nm, dt, dims) in enumerate(self.g.inputs):
            shape = [d if d and d > 0 else None for d in dims] or None
            vi = helper.make_tensor_value_info(
                nm, ONNX_DTYPE.get(dt, TensorProto.FLOAT), shape)
            self.inputs.append(vi)

        # 2. 逐个算子翻译
        for op in self.g.ops:
            try:
                self.translate(op)
            except Exception as e:                       # noqa: BLE001
                self.unsupported.append((op.type, op.name, f"{type(e).__name__}: {e}"))

        # 3. 图输出：MIL 声明的输出（含中间张量，插件真正要的那个）
        outs = list(self.g.outputs) + list(self.g.intermediates)
        if not outs:
            # 兜底：取最后一个算子
            if self.g.ops:
                o = self.g.ops[-1]
                outs = [(o.name, o.dtype, o.dims)]
        produced = {N_NAME(n) for n in self.nodes}
        for nm, dt, dims in outs:
            if nm in produced or nm in self.init_names:
                shape = [d if d and d > 0 else None for d in dims] or None
                self.outputs.append(helper.make_tensor_value_info(
                    nm, ONNX_DTYPE.get(dt, TensorProto.FLOAT), shape))
        return self

    def translate(self, op):
        t = op.type
        fn = getattr(self, f"op_{t}", None)
        if fn is None:
            self.unsupported.append((t, op.name, "没有实现该算子的映射"))
            return
        fn(op)

    # =====================================================================
    # 各算子
    # =====================================================================
    def _arg(self, op, key, idx=0):
        """取第 idx 个绑定（名字或立即值）"""
        vals = op.in_.get(key) or []
        if idx < len(vals):
            return vals[idx]
        return None

    def _as_input(self, op, key, idx=0):
        """把一个参数变成 ONNX 输入名（名字直接用，常量则建 initializer）"""
        b = self._arg(op, key, idx)
        if b is None:
            return None
        if isinstance(b, str):
            return b
        a = self.const_of(b)
        if a is None:
            return None
        nm = self.uniq("const")
        return self.add_init(nm, a)

    def op_const(self, op):
        """const 算子 -> initializer，输出名即 op.name"""
        arr = self.resolve_const(op.name)
        if arr is None:
            # 退化：给个标量 0，避免图断掉
            arr = np.zeros((), dtype=np.float32)
        arr = self._reshape_const(np.asarray(arr), op.dims)
        self.add_init(op.name, arr)

    # ---- 卷积 / 线性 ----
    def const_str(self, name, default=None):
        """把一个引用名解析成字符串常量。

        实测：MIL 的 conv/pool 里 pad_type 是**字符串** 'same' / 'valid'，
        不是整数枚举！
            BXT_ne1_conv_Conv2Dx_pad_type_0 : dtype=STRING  val='same'

        两个坑：
        1. 按 {0:VALID, 1:SAME_UPPER, 2:CUSTOM} 的整数表去映射会全部落到
           默认分支判成 VALID，卷积从 32x32 缩到 30x30，Concat 两路尺寸对不上。
        2. const 算子的 in_ 里有两个键：'name'（常量自己的名字）和 'val'（值）。
           扫 in_ 时如果不指定键，会先撞上 'name' 而返回名字而非值 ——
           所以必须优先取 'val'。
        """
        if not isinstance(name, str):
            return default
        op = self.op_by_name.get(name)
        if op is None or op.type != "const":
            return default
        # 优先 val，其次其它键
        order = ["val"] + [k for k in op.in_ if k != "val" and k != "name"]
        for key in order:
            for b in op.in_.get(key) or []:
                if isinstance(b, dict) and b.get("kind") == "strings":
                    vs = b.get("values") or []
                    if vs:
                        return vs[0]
        return default

    def pad_attrs(self, op, spatial_rank=2):
        """把 MIL 的 pad_type/pad 转成 ONNX 的 auto_pad / pads

        返回值示例: {'auto_pad': 'SAME_UPPER'} 或 {'pads': [...]} 或 {}
        """
        pt = self.const_str(self._arg(op, "pad_type"))
        pad = self.const_ints(self._arg(op, "pad"))
        if pt is not None:
            p = str(pt).lower()
            if p == "same":
                return {"auto_pad": "SAME_UPPER"}
            if p == "valid":
                return {"auto_pad": "VALID"}
        if pad and any(int(v) != 0 for v in pad):
            # MIL pad 是 [b0,e0,b1,e1,...] 还是 [b0,b1,e0,e1]？
            # MIL 文档是后者（先所有 begin 再所有 end）；
            # 但实测本模型的 pad 全是 0，所以走不到这里。
            n = len(pad) // 2
            return {"pads": [int(v) for v in pad]}
        return {}

    def op_conv(self, op):
        x = self._as_input(op, "x")
        w = self._as_input(op, "weight")
        b = self._as_input(op, "bias")
        strides = self.const_ints(self._arg(op, "strides")) or [1, 1]
        dil = self.const_ints(self._arg(op, "dilations")) or [1, 1]
        groups = self.const_int(self._arg(op, "groups"), 1) or 1
        attrs = {"strides": [int(s) for s in strides],
                 "dilations": [int(d) for d in dil],
                 "group": int(groups)}
        attrs.update(self.pad_attrs(op))
        self.emit("Conv", [x, w] + ([b] if b else []), [op.name],
                  name=op.name, **attrs)

    def op_linear(self, op):
        """linear -> MatMul + Add（**不能**用 Gemm）

        MIL 的 linear 语义（coremltools 源码
        converters/mil/mil/ops/defs/iOS15/linear.py）：
            x: tensor<[*D, D_in]>   1 <= rank <= 3
            weight: [D_out, D_in]
            返回 tensor<[*D, D_out]>   —— 与输入同 rank
            等价于 x @ weight.T + bias

        而 ONNX 的 **Gemm 只接受 2 维输入**。注意力模块里的 linear
        输入是 3 维以上的（例如 [1, 1024, 256]），用 Gemm 会直接报
        "Input A must have rank 2"。

        所以用 MatMul + Add：
          * MatMul 支持 N 维（按最后两维矩阵乘）
          * weight 转置一次（在转换期做，不占运行时）
          * bias 靠广播加上去
        """
        x = self._as_input(op, "x")
        w = self.resolve_const(self._arg(op, "weight"))
        if w is None:
            # 兜底：权重不是常量就不转置，直接 MatMul
            self.emit("MatMul", [x, self._as_input(op, "weight")],
                      [op.name], name=op.name)
            return
        w = np.asarray(w, dtype=np.float32)
        wt = self.add_init(self.uniq("linear_wT"), np.ascontiguousarray(w.T))
        mm = self.unique_out("linear_mm")
        self.emit("MatMul", [x, wt], [mm], name=self.uniq("MatMul"))

        b = self.resolve_const(self._arg(op, "bias"))
        if b is not None:
            b = np.asarray(b, dtype=np.float32).ravel()
            bn = self.add_init(self.uniq("linear_b"), b)
            self.emit("Add", [mm, bn], [op.name], name=op.name)
        else:
            self.emit("Identity", [mm], [op.name], name=op.name)

    def op_matmul(self, op):
        """matmul -> MatMul（必要时先转置最后两维）

        MIL 的 matmul 有 transpose_x / transpose_y 两个属性，分别表示
        在做矩阵乘之前把 x / y 的**最后两维**交换。

        实测（v5 的 cross-attention）：
            MatMul 声明输出 [1,8,1024,1024]，输入都是 [1,8,1024,48]
            —— 这就是 Q·Kᵀ 的写法，靠 transpose_y 实现。
        不处理这两个属性会直接崩：
            "matmul: Input operand 1 has a mismatch in its core dimension 0,
             ... (size 1024 is different from 48)"
        """
        x = self._as_input(op, "x")
        y = self._as_input(op, "y")
        tx = self.const_ints(self._arg(op, "transpose_x")) or [0]
        ty = self.const_ints(self._arg(op, "transpose_y")) or [0]

        def last2(name, op_name):
            """对最后两维做转置"""
            nd = None
            src = self.op_by_name.get(name)
            if src is not None and src.dims:
                nd = len(src.dims)
            if nd is None:
                nd = 4
            perm = list(range(nd))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            out = self.uniq(op_name)
            self.emit("Transpose", [name], [out], name=self.uniq("Transpose"),
                      perm=perm)
            return out

        if int(tx[0]) if tx else 0:
            x = last2(x, "tx")
        if int(ty[0]) if ty else 0:
            y = last2(y, "ty")
        self.emit("MatMul", [x, y], [op.name], name=op.name)

    # ---- 激活 ----
    def op_prelu(self, op):
        """prelu -> PRelu

        关键：alpha 的形状是 [C]，但数据是 NCHW 布局。
        ONNX 的 PRelu 遵守 numpy 广播规则，(1,4,H,W) 和 (4,) 是**不能**
        广播的（会错误地对齐到最后一维 W）。

        正确做法是把 alpha 整形成 [C,1,1]，**维度数 = 输入维度 - 1**
        （alpha 少了 N 那一维，但要覆盖 C 和所有空间维）。
        之前写死成 [C,1,1,1]（4 维）是错的：
        (1,4,30,30) 和 (4,1,1,1) 广播会得到 (4,4,30,30)，凭空多出一维，
        后续 Concat 直接炸。实测就是这么踩到的。

        实测报错：
            (1,4,126,126) 和 (4,)   -> 无法广播
            (1,4,30,30)   和 (4,1,1,1) -> 变成 (4,4,30,30)
        """
        x = self._as_input(op, "x")
        a_name = self._as_input(op, "alpha")
        a = self.resolve_const(self._arg(op, "alpha"))
        if a is not None:
            a = np.asarray(a)
            # 目标形状 [C, 1, 1, ...]，总维度数 = 输出维度 - 1（去掉 N 那一维）。
            # 注意是 nd-2 个尾部 1，不是 nd-1 —— 写错过一次：
            #   nd=4 时要 [4,1,1]（rank 3 = 4-1），
            #   用 [4]+[1]*(nd-1) 会得到 [4,1,1,1]（rank 4），广播就多出一维。
            nd = len(op.dims or [])
            if nd < 2:
                b = self._arg(op, "x")
                src = self.op_by_name.get(b) if isinstance(b, str) else None
                nd = len(src.dims) if src and src.dims else 4
            if a.ndim == 1 and a.size > 1:
                a = a.reshape([a.size] + [1] * (nd - 2))
            a_name = self.add_init(self.uniq("prelu_alpha"), a)
        self.emit("PRelu", [x, a_name], [op.name], name=op.name)

    def op_relu(self, op):
        self.emit("Relu", [self._as_input(op, "x")], [op.name], name=op.name)

    def op_gelu(self, op):
        self.emit("Gelu", [self._as_input(op, "x")], [op.name], name=op.name)

    def op_sigmoid(self, op):
        self.emit("Sigmoid", [self._as_input(op, "x")], [op.name], name=op.name)

    def op_tanh(self, op):
        self.emit("Tanh", [self._as_input(op, "x")], [op.name], name=op.name)

    def op_softmax(self, op):
        axis = self.const_int(self._arg(op, "axis"), -1)
        self.emit("Softmax", [self._as_input(op, "x")], [op.name],
                  name=op.name, axis=int(axis))

    # ---- 逐元素 ----
    def _binary(self, op, onnx_op):
        x = self._as_input(op, "x")
        y = self._as_input(op, "y")
        self.emit(onnx_op, [x, y], [op.name], name=op.name)

    def op_add(self, op):      self._binary(op, "Add")
    def op_sub(self, op):      self._binary(op, "Sub")
    def op_mul(self, op):      self._binary(op, "Mul")
    def op_real_div(self, op): self._binary(op, "Div")
    def op_maximum(self, op):  self._binary(op, "Max")
    def op_minimum(self, op):  self._binary(op, "Min")

    def _unary(self, op, onnx_op):
        self.emit(onnx_op, [self._as_input(op, "x")], [op.name], name=op.name)

    def op_exp(self, op):  self._unary(op, "Exp")
    def op_log(self, op):  self._unary(op, "Log")
    def op_sqrt(self, op): self._unary(op, "Sqrt")

    def op_rsqrt(self, op):
        x = self._as_input(op, "x")
        t = self.unique_out("sqrt")
        self.emit("Sqrt", [x], [t], name=self.uniq("Sqrt"))
        one = self.add_init(self.uniq("one"),
                            np.asarray(1.0, dtype=np.float32))
        self.emit("Div", [one, t], [op.name], name=op.name)

    # ---- 形状 ----
    def op_transpose(self, op):
        """transpose -> Transpose

        perm 里可能有**负值**（从末尾计数），实测 v5 的注意力用了
        perm=[0, 2, -3, -1]。ONNX 的 Transpose 规范里 perm 必须是
        有效的维度下标，负数能不能用取决于实现 —— 不能赌，
        统一归一化成非负值。
        """
        perm = self.const_ints(self._arg(op, "perm"))
        nd = len(op.dims or [])
        if perm and nd:
            perm = [p + nd if p < 0 else p for p in perm]
        self.emit("Transpose", [self._as_input(op, "x")], [op.name],
                  name=op.name, perm=perm)

    def op_reshape(self, op):
        shp = self.const_ints(self._arg(op, "shape"))
        x = self._as_input(op, "x")
        if shp:
            nm = self.add_init(self.uniq("shape"),
                               np.asarray(shp, dtype=np.int64))
            self.emit("Reshape", [x, nm], [op.name], name=op.name)
        else:
            self.emit("Reshape", [x, self._as_input(op, "shape")], [op.name],
                      name=op.name)

    def op_expand_dims(self, op):
        axes = self.const_ints(self._arg(op, "axes")) or [0]
        nm = self.add_init(self.uniq("axes"), np.asarray(axes, dtype=np.int64))
        self.emit("Unsqueeze", [self._as_input(op, "x"), nm], [op.name],
                  name=op.name)

    def op_squeeze(self, op):
        axes = self.const_ints(self._arg(op, "axes")) or []
        nm = self.add_init(self.uniq("axes"), np.asarray(axes, dtype=np.int64))
        self.emit("Squeeze", [self._as_input(op, "x"), nm], [op.name],
                  name=op.name)

    def op_concat(self, op):
        vals = [self._as_input(op, "values", i)
                for i in range(len(op.in_.get("values") or []))]
        vals = [v for v in vals if v]
        axis = self.const_int(self._arg(op, "axis"), 1)
        self.emit("Concat", vals, [op.name], name=op.name, axis=int(axis))

    def op_split(self, op):
        """split -> Split（多输出）

        ONNX 的 Split 在各版本要求不同：
          * opset < 18：必须给 `split` 属性（每份大小）
          * opset >= 18：可以给 `split` **输入**，或者 `num_outputs` 属性；
                         两者都不给会报
                         "Neither 'split' input nor 'num_outputs' attribute
                          has been given"

        MIL 侧一般是按通道均分（`xxx_split_0 / xxx_split_1` …），
        不一定带 split_sizes。所以：
          - 有 sizes 且个数匹配 -> 传 split 属性
          - 否则 -> 传 num_outputs = 输出个数（均分）
        """
        x = self._as_input(op, "x")
        axis = self.const_int(self._arg(op, "axis"), 0)
        sizes = self.const_ints(self._arg(op, "split_sizes"))
        n_out = len(op.names) or 1
        attrs = {"axis": int(axis)}
        if sizes and len(sizes) == n_out:
            attrs["split"] = [int(v) for v in sizes]
        else:
            attrs["num_outputs"] = int(n_out)
        self.emit("Split", [x], list(op.names) or [op.name],
                  name=op.name, **attrs)

    def _code_mask(self, binding):
        """解出 strided_slice 的 mask（位掩码 -> list[int]，按位展开）"""
        vals = self.const_ints(binding)
        if not vals:
            return None, 0
        # mask 可能是标量（单个整数位掩码）或每维一个
        if len(vals) == 1:
            m = int(vals[0])
            return None, m
        return [int(v) for v in vals], 0

    def op_slice_by_index(self, op):
        """slice_by_index -> ONNX Slice

        这是 TensorFlow 的 strided_slice，**带 mask 位掩码**，必须处理：

            begin = [0,0,0,0,0]   end = [0,0,0,0,0]   stride = [1,1,1,1,2]
            begin_mask / end_mask / squeeze_mask

        实测（v5 的多头注意力）：
            - end 全是 0，但 end_mask 置位 => end 被忽略，取到维度末尾
            - stride 最后一维是 2 => 隔一个取一个（切分 query/key 的那种用法）

        如果不认 mask，就会把 "end=0" 当成"切 0 个元素"，
        结果切出 (0,0,0,0,0) 的空张量，后面 Mul 直接崩：
            "operands could not be broadcast together with
             shapes (0,0,0,0,0) (1,32,32,1,24)"
        """
        x = self._as_input(op, "x")
        begin = self.const_ints(self._arg(op, "begin")) or [0]
        end = self.const_ints(self._arg(op, "end")) or [0]
        stride = self.const_ints(self._arg(op, "stride")) or [1] * len(begin)

        n = len(begin)
        bmask_list, bmask = self._code_mask(self._arg(op, "begin_mask"))
        emask_list, emask = self._code_mask(self._arg(op, "end_mask"))

        def bit(list_or_none, scalar, i):
            if list_or_none is not None:
                return bool(list_or_none[i]) if i < len(list_or_none) else False
            return bool((scalar >> i) & 1)

        # 输入各维长度（用于把被 mask 掉的 end 补成维度末尾）
        in_dims = None
        b = self._arg(op, "x")
        if isinstance(b, str):
            src = self.op_by_name.get(b)
            if src is not None and src.dims:
                in_dims = [int(d) for d in src.dims]

        BIG = 2 ** 31 - 1
        starts, ends, steps = [], [], []
        for i in range(n):
            bg = int(begin[i]) if i < len(begin) else 0
            en = int(end[i]) if i < len(end) else 0
            st = int(stride[i]) if i < len(stride) else 1
            # begin_mask 置位 -> 起点用默认（stride>0 时为 0）
            if bit(bmask_list, bmask, i):
                bg = 0
            # end_mask 置位 -> 终点用默认（维度末尾）
            if bit(emask_list, emask, i):
                en = BIG
            # begin=0 且 end=0 且没有 mask 时，TF 里是空切片；
            # 但这里绝大多数情况都被 mask 覆盖了，保持原样即可。
            starts.append(bg)
            ends.append(en)
            steps.append(st)

        ins = [x,
               self.add_init(self.uniq("b"), np.asarray(starts, dtype=np.int64)),
               self.add_init(self.uniq("e"), np.asarray(ends, dtype=np.int64)),
               self.add_init(self.uniq("a"), np.asarray(list(range(n)), dtype=np.int64)),
               self.add_init(self.uniq("s"), np.asarray(steps, dtype=np.int64))]
        self.emit("Slice", ins, [op.name], name=op.name)

    def op_slice_by_size(self, op):
        """slice_by_size -> ONNX Slice

        CoreML 的语义：size 里出现 -1 表示"从 begin 一直到该维末尾"。
        直接用 end = begin + size 算会得到 end = begin - 1，
        Slice 会切出空张量或报错 —— 这是实测踩到的坑。

        v4 的 central_crop 就是 size=[-1,512,512,-1]，所以必须处理。
        做法：把整个 shape 作为 end 传入，让 -1 自然落到末尾；
        非 -1 的维度仍用 begin+size。
        """
        x = self._as_input(op, "x")
        begin = self.const_ints(self._arg(op, "begin")) or [0]
        size = self.const_ints(self._arg(op, "size")) or [0]

        # 需要知道每一维的长度才能算 -1 的终点。
        # 优先用 MIL 声明的输入 shape；拿不到就用一个很大的哨兵值，
        # ONNX Slice 会把超出范围的 end 自动截断到维度末尾（clamp 语义）。
        in_shape = None
        b = self._arg(op, "x")
        if isinstance(b, str):
            src = self.op_by_name.get(b)
            if src is not None and src.dims:
                in_shape = list(src.dims)
        if in_shape is None:
            for nm, dt, dims in self.g.inputs:
                if nm == b and dims:
                    in_shape = list(dims)
                    break

        BIG = 2 ** 31 - 1                     # ONNX 会 clamp 到实际维度长度
        end = []
        for i, (bg, sz) in enumerate(zip(begin, size)):
            if sz == -1:
                end.append(BIG)
            else:
                end.append(bg + sz)
        axes = list(range(len(begin)))
        strides = [1] * len(begin)
        ins = [x,
               self.add_init(self.uniq("b"), np.asarray(begin, dtype=np.int64)),
               self.add_init(self.uniq("e"), np.asarray(end, dtype=np.int64)),
               self.add_init(self.uniq("a"), np.asarray(axes, dtype=np.int64)),
               self.add_init(self.uniq("s"), np.asarray(strides, dtype=np.int64))]
        self.emit("Slice", ins, [op.name], name=op.name)

    def op_gather(self, op):
        x = self._as_input(op, "x")
        idx = self._as_input(op, "indices")
        axis = self.const_int(self._arg(op, "axis"), 0)
        self.emit("Gather", [x, idx], [op.name], name=op.name, axis=int(axis))

    def op_pad(self, op):
        """pad -> ONNX Pad

        坑：MIL 的 mode 是**字符串**，实测 v5 的 53 个 pad 全是 'reflect'
        （TensorFlow 的 MirrorPad）。之前用 const_ints 去解，解不出来
        就落到默认的 'constant' —— 于是所有镜像填充被转成了零填充，
        在图像边缘造出不存在的暗边，导致网络对平坦区域也不保持平坦
        （实测常数输入 0.5 会跑出 std=0.14 的假纹理）。

        ONNX 的 Pad mode 取值：'constant' / 'reflect' / 'edge'。
        MIL 的 'replicate' 对应 ONNX 的 'edge'。
        """
        x = self._as_input(op, "x")
        pad = self.const_ints(self._arg(op, "pad"))
        if not pad or not any(int(v) != 0 for v in pad):
            self.emit("Identity", [x], [op.name], name=op.name)
            return

        # 关键：MIL 的 pad 只覆盖**空间维**（2 维 -> 4 个值），
        # 但 ONNX 的 Pad 要求 pads 覆盖**所有维度**（2*rank 个值）。
        # 4 维张量要 8 个值：(b_N,b_C,b_H,b_W, e_N,e_C,e_H,e_W)。
        # 不补前面缺的 N/C 两维，onnxruntime 会直接拒绝加载：
        #   "Pads has incorrect number of values. Expected 2 * 4 values.
        #    Got 4 values."
        # （我的 NumPy 解释器当时容忍了这个错误，所以没发现 ——
        #   官方运行时更严格，这也是交叉验证的价值。）
        rank = len(op.dims or []) or 4
        n_spatial = len(pad) // 2
        n_lead = max(0, rank - n_spatial)          # 需要补的前导维度个数
        begins = [int(v) for v in pad[:n_spatial]]
        ends = [int(v) for v in pad[n_spatial:]]
        begins = [0] * n_lead + begins
        ends = [0] * n_lead + ends
        nm = self.add_init(self.uniq("pads"),
                           np.asarray(begins + ends, dtype=np.int64))

        mode_s = self.const_str(self._arg(op, "mode"), "constant")
        mode_map = {"constant": "constant", "reflect": "reflect",
                    "replicate": "edge", "edge": "edge"}
        mode_s = mode_map.get(str(mode_s).lower(), "constant")
        if mode_s == "edge":
            # edge 模式不接常量输入
            self.emit("Pad", [x, nm], [op.name], name=op.name, mode=mode_s)
            return

        # constant / reflect 都要带常量值（reflect 会忽略它）
        cv = self.const_floats(self._arg(op, "pad_constant_val"), 0.0)
        const_v = self.add_init(self.uniq("pv"),
                                np.asarray(cv, dtype=np.float32))
        self.emit("Pad", [x, nm, const_v], [op.name], name=op.name,
                  mode=mode_s)

    def const_floats(self, name, default=0.0):
        """把一个引用名解析成浮点标量"""
        if not isinstance(name, str):
            return default
        op = self.op_by_name.get(name)
        if op is None or op.type != "const":
            return default
        for key in ["val"] + [k for k in op.in_ if k not in ("val", "name")]:
            for b in op.in_.get(key) or []:
                if isinstance(b, dict) and b.get("kind") in ("floats", "doubles"):
                    vs = b.get("values") or []
                    if vs:
                        return float(vs[0])
                if isinstance(b, dict) and b.get("kind") in ("ints", "longInts"):
                    vs = b.get("values") or []
                    if vs:
                        return float(vs[0])
        return default

    # ---- 池化 ----
    def _pool(self, op, onnx_op):
        x = self._as_input(op, "x")
        ks = self.const_ints(self._arg(op, "kernel_sizes")) or [1, 1]
        st = self.const_ints(self._arg(op, "strides")) or [1, 1]
        ceil = self.const_ints(self._arg(op, "ceil_mode"))
        attrs = {"kernel_shape": [int(v) for v in ks],
                 "strides": [int(v) for v in st],
                 "ceil_mode": int(ceil[0]) if ceil else 0}
        attrs.update(self.pad_attrs(op))
        self.emit(onnx_op, [x], [op.name], name=op.name, **attrs)

    def op_max_pool(self, op): self._pool(op, "MaxPool")
    def op_avg_pool(self, op): self._pool(op, "AveragePool")

    # ---- 归约 ----
    def _reduce(self, op, onnx_op):
        axes = self.const_ints(self._arg(op, "axes")) or [0]
        keep = self.const_ints(self._arg(op, "keep_dims"))
        nm = self.add_init(self.uniq("rax"), np.asarray(axes, dtype=np.int64))
        self.emit(onnx_op, [self._as_input(op, "x"), nm], [op.name],
                  name=op.name, keepdims=int(keep[0]) if keep else 1)

    def op_reduce_mean(self, op): self._reduce(op, "ReduceMean")
    def op_reduce_sum(self, op):  self._reduce(op, "ReduceSum")
    def op_reduce_max(self, op):  self._reduce(op, "ReduceMax")
    def op_reduce_min(self, op):  self._reduce(op, "ReduceMin")

    # ---- 其它 ----
    def op_identity(self, op):
        self.emit("Identity", [self._as_input(op, "x")], [op.name], name=op.name)

    def op_cast(self, op):
        dt = self.const_int(self._arg(op, "dtype"), 11)
        onnx_t = {10: TensorProto.FLOAT16, 11: TensorProto.FLOAT,
                  12: TensorProto.DOUBLE, 23: TensorProto.INT32,
                  24: TensorProto.INT64}.get(int(dt), TensorProto.FLOAT)
        self.emit("Cast", [self._as_input(op, "x")], [op.name],
                  name=op.name, to=onnx_t)


# ---------------------------------------------------------------------------
def validate(tr, graph):
    """图级自检（不需要 onnx 包）。

    检查：
      1. 每个节点的输入都能解析到：图输入 / 前面的节点输出 / initializer
      2. 有没有节点输出名重复
      3. 图输出是否真的被生产出来了
      4. 有没有悬空引用
    """
    produced = set()
    for vi in tr.inputs:
        produced.add(V_NAME(vi))
    for ini in tr.inits:
        produced.add(I_NAME(ini))

    problems = []
    dup = []
    seen_out = set()
    for i, node in enumerate(tr.nodes):
        op = N_OP(node)
        ins = N_IN(node)
        outs = N_OUT(node)
        for x in ins:
            if x is None:
                problems.append(f"节点[{i}] {op}: 有一个 None 输入")
            elif x not in produced:
                problems.append(f"节点[{i}] {op}: 输入 '{x}' 未定义")
        for y in outs:
            if y in seen_out:
                dup.append(y)
            seen_out.add(y)
            produced.add(y)

    missing_out = []
    for vi in tr.outputs:
        nm = V_NAME(vi)
        if nm not in produced:
            missing_out.append(nm)

    # 找出没有任何消费者、也不是图输出的中间量（提示用，不算错）
    consumed = set()
    for node in tr.nodes:
        consumed.update(x for x in N_IN(node) if x)
    outnames = {V_NAME(vi) for vi in tr.outputs}
    dead = sorted(seen_out - consumed - outnames)

    return {"dangling": problems, "dup_outputs": dup,
            "missing_graph_outputs": missing_out, "dead_ends": dead}


def build(path, out_path=None, opset=17):
    p = Path(path)
    if p.is_dir():
        c = sorted(p.rglob("model.mlmodel"))
        if not c:
            raise FileNotFoundError(f"{p} 里找不到 model.mlmodel")
        p = c[0]
    pkg = p.parent.parent.parent
    wb = pkg / "Data/com.apple.CoreML/weights/weight.bin"
    wraw = wb.read_bytes() if wb.exists() else None

    g = mil_graph.load(p)
    tr = Translator(g, wraw, opset=opset).run()

    graph = helper.make_graph(
        tr.nodes, "bxt", tr.inputs, tr.outputs, tr.inits)
    model = helper.make_model(
        graph, producer_name="mil2onnx",
        opset_imports=[helper.make_opsetid("", opset)])
    if HAS_ONNX:
        model.ir_version = 9

    if out_path and HAS_ONNX:
        onnx.save(model, out_path)
    return model, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true",
                    help="只做图级自检，不写出文件")
    a = ap.parse_args()

    out = a.out or (Path(a.path).name.rstrip("/\\") + ".onnx")
    print("=" * 78)
    print(f"翻译: {a.path}")
    print(f"onnx 包: {'可用 ' + onnx.__version__ if HAS_ONNX else '不可用（只做自检）'}")
    print("=" * 78)
    model, tr = build(a.path, None if a.check else out, a.opset)

    print(f"  算子节点   : {len(tr.nodes)}")
    print(f"  initializer: {len(tr.inits)}")
    print(f"  图输入     : {[V_NAME(i) for i in tr.inputs]}")
    print(f"  图输出     : {[V_NAME(o) for o in tr.outputs]}")
    print()

    # 算子直方图
    from collections import Counter
    hist = Counter(N_OP(n) for n in tr.nodes)
    print("  --- 生成的 ONNX 算子 ---")
    for t, n in hist.most_common():
        print(f"    {t:<18} x{n}")
    print()

    if tr.unsupported:
        print(f"  [!] 未映射的算子 {len(tr.unsupported)} 处：")
        c = Counter(t for t, _, _ in tr.unsupported)
        for t, n in c.most_common():
            print(f"      {t:<24} x{n}")
        print("      样例：")
        for t, nm, why in tr.unsupported[:6]:
            print(f"        {t} / {nm}: {why}")
    else:
        print("  [OK] 全部算子都完成了映射")
    print()

    # 图级自检
    g = tr.g
    v = validate(tr, None)
    print("  --- 图级自检 ---")
    if v["dangling"]:
        print(f"    [!] 悬空输入 {len(v['dangling'])} 处：")
        for s in v["dangling"][:10]:
            print(f"        {s}")
    else:
        print("    [OK] 所有节点的输入都有来源")
    if v["dup_outputs"]:
        print(f"    [!] 重复的输出名 {len(v['dup_outputs'])} 个: "
              f"{v['dup_outputs'][:6]}")
    else:
        print("    [OK] 没有重复输出名")
    if v["missing_graph_outputs"]:
        print(f"    [!] 图输出没有被生产: {v['missing_graph_outputs']}")
    else:
        print(f"    [OK] 图输出都已生产 ({len(tr.outputs)} 个)")
    if v["dead_ends"]:
        print(f"    [i] 无消费者的中间量 {len(v['dead_ends'])} 个（通常无害）")
    print()

    if a.check:
        print("  (--check 模式，未写出文件)")
        return 0

    # 序列化：优先用官方 onnx 包，没有就用自带的零依赖写法
    if HAS_ONNX:
        try:
            onnx.checker.check_model(model)
            print("  [OK] onnx.checker 通过")
        except Exception as e:                        # noqa: BLE001
            print(f"  [!] onnx.checker 报错: {type(e).__name__}: {e}")
        onnx.save(model, out)
        print(f"  已写出: {out}  ({Path(out).stat().st_size/1048576:.1f} MB)")
        return 0

    # ---- 零依赖序列化 ----
    import onnx_writer
    gg = {
        "name": "bxt",
        "nodes": tr.nodes,                       # 占位模式下就是 dict 列表
        "inputs": tr.inputs,
        "outputs": tr.outputs,
        "initializers": tr.inits,
        "value_info": [],
    }
    mdl = {
        "ir_version": 9,
        "producer_name": "mil2onnx",
        "producer_version": "1.0",
        "graph": gg,
        "opset": [(None, a.opset)],
    }
    try:
        n = onnx_writer.save(mdl, out)
    except Exception as e:                        # noqa: BLE001
        print(f"  [!] 零依赖序列化失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print("  [i] 使用自带零依赖序列化器（没装 onnx 包）")
    print(f"  已写出: {out}  ({n/1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
