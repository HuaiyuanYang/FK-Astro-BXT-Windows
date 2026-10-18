# FK-Astro:BXT

PixInsight 的神经网络去卷积模块 —— 采用 BlurXTerminator 的参数体系,独立实现。

**用户请看 → [`docs/FK-Astro-BXT-使用手册.html`](docs/FK-Astro-BXT-使用手册.html)**

---

## 需要什么

| 依赖 | 从哪来 |
|---|---|
| Windows x64 + PixInsight 1.9+ | 你已有 |
| **ONNX Runtime** | **PixInsight 1.9 自带**(1.25+),无需另装 |
| `bxt5.onnx`(约 185 MB) | **自备**,放在本目录的**上一级** |

```
Deepseek Daily/
├── bxt5.onnx          ← 放这里
└── FK-Astro-BXT-Windows/      ← 本项目
```

脚本会自动在 `../bxt5.onnx`、`./bxt5.onnx` 两处查找;模块本身通过 `Model path` 参数指定,位置随意。

---

## 状态

| 部分 | 状态 |
|---|---|
| 模型转换(CoreML `.mlpackage` → ONNX) | ✅ 权重逐字节无损 |
| 宿主归一化方案 **FKv3** | ✅ 与官方 BXT 对比验证,P99 < 1e-4 |
| Python 参考实现 | ✅ `python/fkv3.py` |
| **C++ 模块(FKv3)** | ✅ **已移植,编译通过** |
| 可安装的 `.dll` | ✅ `src/deploy/FKAstro-pxm.dll`(659 KB,待签名) |

> 需要签名后才能装进 PixInsight,见 [`docs/FK-Astro-BXT-使用手册.html`](docs/FK-Astro-BXT-使用手册.html) §4。

---

## 目录

| 目录 | 内容 |
|---|---|
| `docs/` | 用户手册(HTML) |
| `src/` | C++ 模块源码、构建脚本、ORT 头文件、部署脚本 |
| `python/` | FKv3 参考实现、验证工具、模型转换器 |

| `samples/` | 测试图放置处。**默认不存在,需要复验时自行新建** |

> `MIL.proto` / `onnx.proto3` 是模型转换的字段号依据,与运行无关,已放到工作区根目录的 `proto\`。

---

## 已完成:FKv3 移植

`src/FKAstroProcess.cpp` 的归一化/分块/融合已全部换成 FKv3,数学放在
`src/FKv3Pipeline.h`(与离线自检共用同一份实现)。

**验证**:编译 `.\build.ps1 -Step v3test` 后得到 `src/_engine_test/fkv3_selftest.exe`,
再配 `python/compare_fkv3_selftest.py` 与 Python 参考实现对账,
7 个用例(单块/多块/小图/单色/overlap 0/大图)与 Python 参考实现的
最大偏差 **2.3e-08**,相关 1.000000000。

**对照依据**:

- [`src/FKv3Pipeline.h`](src/FKv3Pipeline.h) —— FKv3 实现(归一化 / 分块 / 融合),离线自检与模块共用同一份
- [`python/fkv3.py`](python/fkv3.py) —— Python 参考实现(已与官方 BXT 对比)
- 公式速查:工作区根目录 `notes/HANDOVER-交接清单.md` §7

- [`python/fkv3.py`](python/fkv3.py) —— 已验证的 Python 版

---

## 构建

```powershell
cd src
.\bootstrap-toolchain.ps1      # 首次:装 MSVC Build Tools + Windows SDK(需联网)
.\build.ps1 -Step all
```

产物:`src\build\FKAstro-pxm.dll`,签名后放入 `PixInsight\bin\`。
详见 [`src/BUILD.md`](src/BUILD.md)。

---

## Python 侧

```powershell
pip install numpy onnxruntime
python python\fkv3.py
```

验证方法见 [`python/README.md`](python/README.md)。

---

## 许可与声明

**源代码与文档采用 MIT**,见 [`LICENSE`](LICENSE)。

MIT 只覆盖本仓库的代码与文档,不包括 PixInsight / PCL、BlurXTerminator,
也不包括任何 ONNX 模型权重 —— 模型由使用者自备,其使用方式需自行确认合规。

- 本模块基于上游开源项目 **fk-astro** 的调用约定独立实现,与 **RC Astro** 无隶属关系,未获其背书。
- BlurXTerminator 是 RC Astro 的产品。需要官方支持与保证请使用正版。
- **模型文件不随本仓库分发。**
- 分发模块二进制还需 PixInsight 签名:自用走 Local Signing Identity,对外需 CPD。
---

## 致谢

感谢 [企鹅](https://github.com/wfzylcl878-creator) 的前期工作。
