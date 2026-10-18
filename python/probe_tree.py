# -*- coding: utf-8 -*-
"""
probe_tree.py —— 递归展开 protobuf 消息树，用来找出"操作列表"到底在哪一层。

调试工具，不参与转换流程。用法:
    python probe_tree.py <model.mlmodel> [最大深度] [路径前缀]
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spec_read import parse_msg, rd_varint, unpack_varints


def maybe_string(blob):
    """如果这块字节像可打印字符串就返回它，否则 None。"""
    try:
        t = blob.decode("utf-8")
    except Exception:
        return None
    if t and t.isprintable():
        return t
    return None


def looks_like_shape(blob):
    """小整数序列 -> 可能是 shape"""
    vals = unpack_varints(blob)
    if not vals:
        return None
    if len(vals) > 8:
        return None
    if all(0 <= v <= 8192 for v in vals):
        return vals
    return None


def dump(blob, label, out, depth, max_depth, path):
    ind = "  " * depth
    line = f"{ind}{label}  len={len(blob)}"

    s = maybe_string(blob)
    if s is not None and len(s) < 100:
        out.append(line + f"   str={s!r}")
        return

    fields = parse_msg(blob)
    if not fields:
        out.append(line + "   (no fields)")
        return

    fc = Counter(f for f, _, _ in fields)
    summary = " ".join(f"f{k}x{v}" for k, v in sorted(fc.items()))
    out.append(line + f"   [{summary}]")

    if depth >= max_depth:
        return

    # group repeated message fields and recurse into a sample
    for f in sorted(fc):
        vals = [v for g, wt, v in fields if g == f and wt == 2]
        if not vals:
            continue
        if fc[f] == 1:
            dump(vals[0], f"f{f}", out, depth + 1, max_depth, path + [f])
        else:
            out.append(f"{ind}  f{f} (repeated x{fc[f]}):")
            for i, v in enumerate(vals[:3]):
                dump(v, f"f{f}[{i}]", out, depth + 2, max_depth, path + [f, i])
            if fc[f] > 3:
                out.append(f"{ind}    ... +{fc[f]-3} more")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    p = Path(args[0])
    max_depth = int(args[1]) if len(args) > 1 else 4
    if p.is_dir():
        cands = sorted(p.rglob("model.mlmodel"))
        if not cands:
            print("no model.mlmodel")
            return 1
        p = cands[0]
    raw = p.read_bytes()
    top = parse_msg(raw)
    out = [f"{p}", f"total {len(raw)} bytes", ""]

    # 只展开我们关心的顶层字段，避免刷屏
    for f, wt, v in top:
        if wt == 0:
            out.append(f"f{f} (varint) = {v}")
        elif wt == 2:
            s = maybe_string(v)
            if s is not None and len(s) < 100:
                out.append(f"f{f} (string) = {s!r}")
            else:
                dump(v, f"f{f}", out, 1, max_depth, [f])
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
