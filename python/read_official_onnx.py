"""Read-only I/O contract probe: official RC-Astro ONNX models vs our converted ones.

REA has no ONNX capability, so this uses the vendored onnxruntime purely to READ
model signatures. Nothing is executed on the models beyond session construction,
and no file is modified.

Usage:
    python python/read_official_onnx.py
"""

import os
import sys

# 项目根目录 = 本文件的上一级
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PROJ)          # 再上一级(放 bxt5.onnx 的地方)

import onnxruntime as ort  # noqa: E402

# BlurXTerminator 把模型缓存在这里(Windows)
RC = os.path.join(os.environ.get("APPDATA", ""), "RC-Astro")

TARGETS = [
    (os.path.join(RC, "BlurXTerminator.4.onnx"), "official-bxt4"),
    (os.path.join(RC, "BlurXTerminator.5.onnx"), "official-bxt5"),
    (os.path.join(WS, "bxt4.onnx"), "ours-bxt4"),
    (os.path.join(WS, "bxt5.onnx"), "ours-bxt5"),
]

try:
    import onnx  # optional
except Exception:
    onnx = None


def graph_facts(path):
    """Cheap structural facts that do not need a full checker pass."""
    if onnx is None:
        return None
    try:
        m = onnx.load(path, load_external_data=False)
    except Exception as e:
        return f"onnx.load failed: {e}"
    opsets = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
    init_bytes = 0
    for t in m.graph.initializer:
        n = 1
        for d in t.dims:
            n *= int(d)
        init_bytes += n * 4  # all float32 in these graphs
    return {
        "ir_version": m.ir_version,
        "opsets": opsets,
        "nodes": len(m.graph.node),
        "initializers": len(m.graph.initializer),
        "initializer_bytes_f32": init_bytes,
        "doc_string": (m.doc_string or "")[:200],
        "metadata": {p.key: p.value for p in m.metadata_props},
    }


def main():
    print("onnxruntime:", ort.__version__, "| onnx:", getattr(onnx, "__version__", "NOT INSTALLED"))
    for path, tag in TARGETS:
        print("=" * 78)
        if not os.path.exists(path):
            print(f"[{tag}] MISSING: {path}")
            continue
        print(f"[{tag}] {os.path.basename(path)}   {os.path.getsize(path):,} bytes")

        facts = graph_facts(path)
        if isinstance(facts, dict):
            print(f"  ir_version={facts['ir_version']} opsets={facts['opsets']} "
                  f"nodes={facts['nodes']} inits={facts['initializers']} "
                  f"init_bytes={facts['initializer_bytes_f32']:,}")
            if facts["doc_string"]:
                print("  doc:", facts["doc_string"])
            if facts["metadata"]:
                print("  metadata:", facts["metadata"])
        elif facts:
            print("  ", facts)

        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"  !! ORT LOAD FAILED: {type(e).__name__}: {str(e)[:400]}")
            continue

        print("  inputs:")
        for i in sess.get_inputs():
            print(f"    - {i.name!r:24} shape={i.shape}  type={i.type}")
        print("  outputs:")
        for o in sess.get_outputs():
            print(f"    - {o.name!r:24} shape={o.shape}  type={o.type}")
        md = sess.get_modelmeta()
        print(f"  producer={getattr(md, 'producer_name', '?')!r} "
              f"version={getattr(md, 'producer_version', '?')!r} "
              f"domain={getattr(md, 'domain', '?')!r}")
        if md.custom_metadata_map:
            print("  custom_metadata:", dict(md.custom_metadata_map))
        print("  providers:", sess.get_providers())


if __name__ == "__main__":
    main()
