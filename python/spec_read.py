# -*- coding: utf-8 -*-
"""
spec_read.py —— 读取 CoreML .mlpackage 的输入/输出规格（纯 Python，Windows 可用）。

背景：coremltools 只有 macOS / Linux 版，Windows 装不了；但规格文件
     model.mlmodel 是明文 protobuf，可以手工解析。本脚本只读，不执行模型。

实测 v4 的结构（这就是 CoreML 的写法）：
    f1 = 规格版本
    f2 ModelDescription
        f1 = 输入(可重复) / f2 = 输出(可重复) / f10 = 中间张量 / f100 = 元数据
            每个 FeatureDescription:
              f1 = 名字(string)
              f3 = 类型(FeatureType)
                   f5 = multiArrayType
                        f1 = Shape  (打包的重复 varint，需解码)
                        f2 = 数据类型 (65568=float32)

用法:  python spec_read.py [mlpackage 目录 ...]
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def rd_varint(b, p):
    r = 0
    s = 0
    while True:
        if p >= len(b):
            return None
        x = b[p]
        p += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80):
            return r, p
        s += 7
        if s > 70:
            return None


def parse_msg(b, start=0, end=None):
    if end is None:
        end = len(b)
    out = []
    p = start
    while p < end:
        v = rd_varint(b, p)
        if not v:
            break
        key, p = v
        f, wt = key >> 3, key & 7
        if f == 0:
            break
        if wt == 0:
            r = rd_varint(b, p)
            if not r:
                break
            out.append((f, 0, r[0]))
            p = r[1]
        elif wt == 2:
            r = rd_varint(b, p)
            if not r:
                break
            ln, p = r
            if p + ln > end:
                break
            out.append((f, 2, b[p:p + ln]))
            p += ln
        elif wt == 5:
            out.append((f, 5, b[p:p + 4]))
            p += 4
        elif wt == 1:
            out.append((f, 1, b[p:p + 8]))
            p += 8
        else:
            break
    return out


def unpack_varints(blob):
    """解码"打包的重复 varint"（Shape 就是这种）。"""
    vals = []
    p = 0
    while p < len(blob):
        r = rd_varint(blob, p)
        if not r:
            break
        vals.append(r[0])
        p = r[1]
    return vals


DT = {65568: "float32", 65552: "float16", 65600: "float64",
      131104: "int32", 131080: "int64", 131072: "int32"}


def parse_feature(blob):
    """FeatureDescription -> {name, kind, shape, dtype}"""
    name, kind, shape, dtype = None, "?", None, None
    for f, wt, v in parse_msg(blob):
        if f == 1 and wt == 2:
            name = v.decode("utf-8", "replace")
        elif f == 3 and wt == 2:                 # 类型
            for g, gwt, gv in parse_msg(v):
                if gwt != 2:
                    continue
                if g == 5:                       # multiArrayType
                    kind = "multiArray"
                    for h, hwt, hv in parse_msg(gv):
                        if h == 1 and hwt == 2:  # Shape（打包 varint）
                            shape = unpack_varints(hv)
                        elif h == 2 and hwt == 0:
                            dtype = DT.get(hv, f"enum{hv}")
                elif g == 4:
                    kind = "image"
                elif g == 2:
                    kind = "int64"
                elif g == 3:
                    kind = "double"
    return {"name": name, "kind": kind, "shape": shape or None, "dtype": dtype}


def parse_metadata(blob):
    md = {}
    for f, wt, v in parse_msg(blob):
        if wt != 2:
            continue
        if f == 100:
            sub = parse_msg(v)
            k = val = None
            for h, hwt, hv in sub:
                if hwt == 2 and h == 1:
                    k = hv.decode("utf-8", "replace")
                elif hwt == 2 and h == 2:
                    val = hv.decode("utf-8", "replace")
            if k:
                md[k] = val or ""
        elif f in (1, 2, 3, 4):
            s = v.decode("utf-8", "replace")
            label = {1: "描述", 2: "版本", 3: "作者", 4: "版权"}.get(f, f"field{f}")
            if s.isprintable():
                md[label] = s
    return md


def layout_of(shape):
    if not shape or len(shape) != 4:
        return None
    n, a, b, c = shape
    if c in (1, 3, 4) and a not in (1, 3, 4):
        return "NHWC", c, a, b
    return "NCHW", a, b, c


def analyse(pkg):
    pkg = Path(pkg)
    print("=" * 74)
    print(f"模型: {pkg}")
    print("=" * 74)
    cands = sorted(pkg.rglob("model.mlmodel"))
    if not cands:
        print("  [X] 找不到 model.mlmodel")
        return None
    raw = cands[0].read_bytes()
    top = parse_msg(raw)

    spec_version = next((v for f, wt, v in top if f == 1 and wt == 0), None)
    desc = next((v for f, wt, v in top if f == 2 and wt == 2), None)

    total = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
    wb = sorted(pkg.rglob("weight.bin"))
    print(f"  规格版本 {spec_version}    目录体积 {total/1048576:.1f} MB"
          + (f"    权重 {wb[0].stat().st_size/1048576:.1f} MB" if wb else ""))

    ins, outs, mids, md = [], [], [], {}
    if desc:
        for f, wt, v in parse_msg(desc):
            if wt != 2:
                continue
            if f == 1:
                ins.append(parse_feature(v))
            elif f == 2:
                outs.append(parse_feature(v))
            elif f == 10:
                mids.append(parse_feature(v))
            elif f == 100:
                md = parse_metadata(v)

    def show(tag, lst):
        for io in lst:
            print(f"    {tag}  {str(io['name']):<14} {io['kind']:<11} "
                  f"形状={io['shape']}  {io['dtype']}")

    print("\n  --- 模型的输入 ---")
    show("输入", ins)
    print("  --- 模型的输出 ---")
    show("输出", outs)
    if mids:
        print("  --- 中间量（插件可能用不到）---")
        show("中间", mids)

    main = params = None
    for io in ins:
        s = io["shape"]
        if s and len(s) == 4 and max(s[1:]) > 32:
            main = io
        elif s:
            cnt = 1
            for x in s:
                cnt *= max(int(x), 1)
            if cnt <= 4096:
                params = io
    # 这个模型把真正的输出声明成了"中间量"（f10），所以两边都要看
    out = outs[0] if outs else (mids[0] if mids else None)

    print("\n  --- 判读（不用懂技术也能看）---")
    gm = layout_of(main["shape"]) if main and main["shape"] else None
    if gm:
        _, c, h, w = gm
        print(f"    ① 每次要把图切成 {w} x {h} 像素、{c} 通道的小块喂进去")
        print(f"       通道排布 {gm[0]}" + ("（通道在最后）" if gm[0] == "NHWC" else "（通道在最前）"))
    else:
        print("    ① 未认出主输入")

    if params and params["shape"]:
        cnt = 1
        for x in params["shape"]:
            cnt *= max(int(x), 1)
        print(f"    ② 还要额外告诉它 {cnt} 个数字（形状 {params['shape']}）")
    else:
        print("    ② 不需要额外参数")

    go = layout_of(out["shape"]) if out and out["shape"] else None
    if go:
        _, c, h, w = go
        print(f"    ③ 每块吐回 {w} x {h} 像素、{c} 通道")
        if gm and gm[2] > h:
            m = (gm[2] - h) // 2
            print(f"    ④ 每块四周要多切 {m} 像素当上下文（边距 margin）")
            print(f"       每步实际前进 {h} - 2x{m} = {h - 2*m} 像素")

    if md:
        print("\n  --- 文件自带的说明信息 ---")
        for k, v in md.items():
            if v:
                print(f"    {k}: {v}")
    print()
    return True


def main():
    args = sys.argv[1:]
    if args:
        targets = [Path(a) for a in args]
    else:
        found = []
        for dp, dns, _ in os.walk("../mlpackage"):
            for d in dns:
                if d.endswith(".mlpackage"):
                    found.append(Path(dp) / d)
        found.sort()
        uniq = []
        for t in found:
            if not any(str(t).startswith(str(u) + os.sep) for u in uniq):
                uniq.append(t)
        targets = uniq
        if not targets:
            print("没找到 .mlpackage，请把路径作为参数传入。")
            return 1
        print(f"自动找到 {len(targets)} 个模型\n")
    n = sum(1 for t in targets if analyse(t))
    print("=" * 74)
    print(f"完成: 成功读取 {n} / {len(targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
