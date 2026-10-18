#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
md2html.py —— 把手册 Markdown 转成自包含的单文件 HTML(中文排版优化)。

支持本项目手册用到的全部语法:
  标题 / 段落 / 表格 / 围栏代码块 / 引用 / 有序无序列表 / 分隔线 /
  行内 **粗体** `代码` [链接](url)

输出:单个 HTML 文件,内嵌 CSS,无外部依赖,带自动生成的目录。
"""
import argparse
import html
import os
import re
import sys

CSS = """
:root{
  --fg:#1f2328; --bg:#ffffff; --muted:#59636e; --line:#d8dee4;
  --accent:#0969da; --accent-soft:#ddf4ff;
  --code-bg:#f6f8fa; --th-bg:#eef2f6; --zebra:#fafbfc;
  --warn-bg:#fff8e6; --warn-line:#e3b341;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; padding:0 20px 80px;
  background:var(--bg); color:var(--fg);
  font-family:"Microsoft YaHei UI","Microsoft YaHei","PingFang SC",
              "Hiragino Sans GB","Source Han Sans SC","Noto Sans CJK SC",
              -apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:16.5px; line-height:1.8;
  text-rendering:optimizeLegibility;
}
.wrap{max-width:920px; margin:0 auto}
h1,h2,h3,h4{line-height:1.35; font-weight:700; margin:2.1em 0 .7em}
h1{
  font-size:1.95em; margin-top:1.1em; padding-bottom:.45em;
  border-bottom:3px solid var(--accent);
}
h2{
  font-size:1.42em; padding-left:.6em; border-left:5px solid var(--accent);
  margin-top:2.4em;
}
h3{font-size:1.16em; color:#243b53; margin-top:1.9em}
h4{font-size:1.02em; color:#334e68}
p{margin:.85em 0}
a{color:var(--accent); text-decoration:none}
a:hover{text-decoration:underline}
hr{border:0; border-top:1px solid var(--line); margin:2.6em 0}

/* 表格 —— 手册里信息最密集的部分, 重点优化 */
.tbl{overflow-x:auto; margin:1.25em 0; border-radius:8px}
table{
  border-collapse:separate; border-spacing:0; width:100%;
  font-size:.95em; display:table;
  border:1px solid var(--line); border-radius:8px; overflow:hidden;
}
thead th{
  background:var(--th-bg); font-weight:700; text-align:left;
  border-bottom:1px solid var(--line); padding:.62em .85em;
  white-space:nowrap;
}
thead th + th{border-left:1px solid var(--line)}
tbody td{padding:.58em .85em; vertical-align:top; border-top:1px solid var(--line)}
tbody td + td{border-left:1px solid var(--line)}
tbody tr:first-child td{border-top:0}
tbody tr:nth-child(even){background:var(--zebra)}

/* 代码 / 示意块 */
pre{
  background:var(--code-bg); border:1px solid var(--line); border-radius:8px;
  padding:.9em 1.1em; overflow-x:auto; margin:1.1em 0;
  font-size:.92em; line-height:1.65;
}
pre code{
  font-family:"Cascadia Mono",Consolas,"Courier New",monospace;
  background:none; border:0; padding:0; white-space:pre;
}
code{
  font-family:"Cascadia Mono",Consolas,"Courier New",monospace;
  background:var(--code-bg); border:1px solid var(--line);
  border-radius:4px; padding:.1em .38em; font-size:.92em;
  color:#b02a37;
}

/* 引用(用于"提醒/警示") */
blockquote{
  margin:1.3em 0; padding:.85em 1.1em;
  background:var(--accent-soft); border-left:5px solid var(--accent);
  border-radius:0 8px 8px 0; color:#0a3069;
}
blockquote p{margin:.35em 0}

ul,ol{margin:.8em 0; padding-left:1.7em}
li{margin:.35em 0}

strong{font-weight:700}

/* 目录 */
.toc{
  background:var(--code-bg); border:1px solid var(--line); border-radius:10px;
  padding:1.1em 1.5em 1.1em 1.2em; margin:1.8em 0 2.4em;
}
.toc-title{font-weight:700; font-size:1.06em; margin:0 0 .6em}
.toc ul{list-style:none; margin:0; padding:0}
.toc ul.lv2>li{margin:.3em 0; font-weight:600}
.toc ul.lv3{margin:.25em 0 .5em; padding-left:1.35em}
.toc ul.lv3>li{margin:.16em 0; font-weight:400; font-size:.93em}
.toc ul.lv3 a{color:var(--muted)}
.toc ul.lv3 a:hover{color:var(--accent)}
.toc li.flat{font-weight:400}
.toc a{color:var(--fg)}
.toc a:hover{color:var(--accent); text-decoration:none}

/* 表头锚点 */
h2[id],h3[id]{scroll-margin-top:14px}

@media (max-width:680px){
  body{font-size:15.5px; padding:0 14px 60px}
  h1{font-size:1.6em}
  h2{font-size:1.25em}
}
@media print{
  body{font-size:11pt; color:#000; padding:0}
  .toc{page-break-after:always}
  h1,h2,h3{page-break-after:avoid}
  table,pre,blockquote{page-break-inside:avoid}
  a{color:#000; text-decoration:none}
  pre{background:#f5f5f5}
}
"""

INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"`([^`]+?)`"), r"<code>\1</code>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
]


def inline(s):
    """行内标记。先转义, 再套标记(标记是我们自己插入的, 不会被转义)。"""
    out = html.escape(s, quote=False)
    for pat, rep in INLINE:
        out = pat.sub(rep, out)
    return out


def slug(text, used):
    """生成稳定的锚点 id。"""
    t = re.sub(r"[`*\[\]()#>]", "", text)
    t = re.sub(r"[^\w\u4e00-\u9fff]+", "-", t).strip("-").lower()
    if not t:
        t = "sec"
    base, i = t, 2
    while t in used:
        t = f"{base}-{i}"
        i += 1
    used.add(t)
    return t


def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_sep(line):
    s = line.strip()
    if not s.startswith("|") or "-" not in s:
        return False
    cells = split_row(s)
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells) and bool(cells)


def convert(md):
    lines = md.split("\n")
    out, toc = [], []
    used_ids = set()
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        s = line.rstrip()

        # 围栏代码块
        if s.startswith("```"):
            body = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(body)) + "</code></pre>")
            continue

        # 分隔线
        if re.fullmatch(r"-{3,}|\*{3,}", s.strip()):
            out.append("<hr>")
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if lvl == 1:
                out.append(f"<h1>{inline(txt)}</h1>")
            else:
                sid = slug(txt, used_ids)
                out.append(f'<h{lvl} id="{sid}">{inline(txt)}</h{lvl}>')
                if lvl in (2, 3):
                    toc.append((lvl, sid, txt))
            i += 1
            continue

        # 表格
        if s.strip().startswith("|") and i + 1 < n and is_sep(lines[i + 1]):
            head = split_row(s)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            t = ["<table><thead><tr>"]
            t += [f"<th>{inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in rows:
                r = (r + [""] * len(head))[:len(head)]
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append('<div class="tbl">' + "".join(t) + "</div>")
            continue

        # 引用
        if s.strip().startswith(">"):
            body = []
            while i < n and lines[i].strip().startswith(">"):
                body.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            txt = " ".join(x.strip() for x in body if x.strip())
            out.append(f"<blockquote><p>{inline(txt)}</p></blockquote>")
            continue

        # 列表
        m = re.match(r"^\s*([-*])\s+(.*)$", line)
        if m:
            items = []
            while i < n:
                mm = re.match(r"^\s*([-*])\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append(mm.group(2))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue

        m = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if m:
            items = []
            while i < n:
                mm = re.match(r"^\s*(\d+)\.\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append(mm.group(2))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>")
            continue

        # 空行
        if not s.strip():
            i += 1
            continue

        # 段落
        para = []
        while i < n and lines[i].strip() and not re.match(
                r"^\s*(#{1,4}\s|```|>|\||[-*]\s|\d+\.\s)", lines[i]) and not re.fullmatch(
                r"-{3,}", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
        else:
            i += 1

    return out, toc


def build(md_path, out_path, title):
    md = open(md_path, encoding="utf-8-sig").read()
    body, toc = convert(md)

    # 目录:h2 为顶层, h3 嵌套在所属 h2 之下, 不编号
    toc_html = ['<nav class="toc"><div class="toc-title">目录</div><ul class="lv2">']
    open_h2 = False
    sub_open = False
    for lvl, sid, txt in toc:
        if lvl == 2:
            if sub_open:
                toc_html.append("</ul>")
                sub_open = False
            if open_h2:
                toc_html.append("</li>")
            toc_html.append(f'<li><a href="#{sid}">{inline(txt)}</a>')
            open_h2 = True
        else:
            if not open_h2:
                toc_html.append(f'<li class="flat"><a href="#{sid}">{inline(txt)}</a></li>')
            else:
                if not sub_open:
                    toc_html.append('<ul class="lv3">')
                    sub_open = True
                toc_html.append(f'<li><a href="#{sid}">{inline(txt)}</a></li>')
    if sub_open:
        toc_html.append("</ul>")
    if open_h2:
        toc_html.append("</li>")
    toc_html.append("</ul></nav>")
    # 目录插在第一个 h2 之前
    first = next((k for k, x in enumerate(body) if x.startswith("<h2")), None)
    if first is not None:
        body.insert(first, "".join(toc_html))
    else:
        body.append("".join(toc_html))

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{chr(10).join(body)}
</div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return len(doc), len(toc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.src)[0] + ".html"
    title = a.title or os.path.splitext(os.path.basename(a.src))[0]
    size, n = build(a.src, out, title)
    print(f"  {os.path.basename(out)}  {size/1024:.1f} KB   目录条目 {n} 个")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
