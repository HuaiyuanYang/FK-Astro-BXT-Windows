# src 构建手册

> 目标:`bxt5.onnx` / `bxt4.onnx` 在 PixInsight 1.9.5 (Windows) 上作为**原生模块**运行。
>
> ⚠️ **当前状态:一行都还没编译过。** 源码与构建配方已就绪,但本机缺编译器。
> 本文件里的每条结论都标了来源;标【待验证】的表示尚未用编译器证实。

---

## 0. 目录结构

```
FK-Astro-BXT-Windows\src\
├─ src\
│  ├─ InferenceEngine.h/.cpp      本项目的 ONNX Runtime 引擎(新写)
│  ├─ FKAstroProcess.h/.cpp       ← 自 fk-astro 复制, 仅改 1 行 include
│  ├─ FKAstroInterface.h/.cpp     ← 自 fk-astro 复制, 未改
│  └─ FKAstroModule.cpp           ← 自 fk-astro 复制, 仅改 Description 文案
├─ third_party\onnxruntime\       ORT 1.25.0 官方 C API 头(5 个)
├─ py\bxt_bridge.py               A 路线:Python 桥接(已端到端验证)
└─ py\selfcheck.py                上述自检
```

`src\` 下**没有** `CoreMLEngine.h/.mm` —— 那份是 macOS/CoreML 实现,不参与编译。
`InferenceEngine.h` 末尾有 `using CoreMLEngine = InferenceEngine;`,所以
`FKAstroProcess.cpp` 里原有的 `fka::CoreMLEngine` 写法全部继续有效。

---

## 1. 工具链

需要 **VS 2022 Build Tools** + 两个组件:

| 组件 ID | 作用 | 来源 |
|---|---|---|
| `Microsoft.VisualStudio.Component.VC.Tools.x86.x64` | MSVC 编译器 / 链接器 / CRT | 从布局自带 `Catalog.json` 读出 |
| `Microsoft.VisualStudio.Component.Windows11SDK.26100` | Windows SDK(头文件 + 库) | 同上 |

引导器已下载并**通过签名验证**:

```
D:\PI-Plugin-Dev\vs_BuildTools.exe
  size   4,474,800 bytes
  sha256 2a6aa7d94e38d457a4eb3930fc60056ecab06865b2eb925e5275b8473a017583
  sig    Valid — CN=Microsoft Corporation, issuer=Microsoft Code Signing PCA 2024
```

命令行安装(需管理员):

```cmd
"D:\PI-Plugin-Dev\vs_BuildTools.exe" --installPath "D:\VSBuildTools" ^
  --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
  --add Microsoft.VisualStudio.Component.Windows11SDK.26100 ^
  --includeRecommended --quiet --wait --norestart
```

> 本机实测:从**沙箱内的后台任务**跑 `--layout` 会卡死或静默退出(5 次尝试,每次停在同一处)。
> 该引导器在交互式桌面下正常运行;建议在**自己的终端**里执行,或直接图形界面点选。

---

## 2. 环境变量(⚠️ 与原路线图不同)

```cmd
set PCLDIR      = D:\PixInsight\src\pcl
set PCLBINDIR64 = %PCLDIR%\bin\bin64
set PCLBINDIR   = %PCLBINDIR64%
set PCLLIBDIR64 = %PCLDIR%\lib\lib64
set PCLLIBDIR   = %PCLLIBDIR64%
set PCLINCDIR   = D:\PixInsight\include
set PCLSRCDIR   = D:\PixInsight\src
```

**`PCLSRCDIR` 是修正过的值。** 原路线图写 `…\src\pcl`,但 `PCL.vcxproj` 里是
`$(PCLSRCDIR)\3rdparty`(见 AdditionalIncludeDirectories),而
`D:\PixInsight\src\pcl\3rdparty` **不存在** —— 真实位置是 `D:\PixInsight\src\3rdparty`。
按原值设会导致 3rdparty 头文件全部找不到。

另需先建好输出目录(当前只有占位文件 `1.txt`):

```
D:\PixInsight\src\pcl\bin\bin64
D:\PixInsight\src\pcl\lib\lib64
```

---

## 3. 构建顺序

### 3.1 先编五个 3rdparty 库

每个库都自带 `.vcxproj`(已实测:`lcms` / `lz4` / `RFC6234` / `zlib` / `zstd` 各一个)。
先列出来确认路径:

```powershell
Get-ChildItem "D:\PixInsight\src\3rdparty" -Recurse -File -Filter *.vcxproj |
    Select-Object -ExpandProperty FullName
```

然后逐个:

```cmd
msbuild <某个.vcxproj> /p:Configuration=Release /p:Platform=x64
```

### 3.2 再编 PCL

```cmd
msbuild "D:\PixInsight\src\pcl\windows\vc17\PCL.vcxproj" ^
        /p:Configuration=Release /p:Platform=x64
```

工程属性(已实测,非推测):

| 项 | 值 |
|---|---|
| `ConfigurationType` | `StaticLibrary` |
| `PlatformToolset` | `v143`(VS 2022) |
| 源文件引用 | **相对路径** `..\..\Xxx.cpp`(共 196 个) |
| 附加包含目录 | `$(PCLINCDIR);$(PCLSRCDIR)\3rdparty` |
| 是否依赖 Qt | **否** —— 299 个头文件 + 198 个源文件里 `#include <Q...>` 行数为 **0** |

> 原路线图把「PCL 是否需要 Qt6 开发头文件」列为最大风险点(失败则整张图作废)。
> **该风险实测不存在。**

### 3.3 最后编模块

需要编译的源文件:

```
src\FKAstroModule.cpp
src\FKAstroInterface.cpp
src\FKAstroProcess.cpp
src\InferenceEngine.cpp
```

| 项 | 值 |
|---|---|
| 附加包含目录 | `D:\PixInsight\include` · `FK-Astro-BXT-Windows\src\third_party\onnxruntime` |
| 输出 | `<Name>-pxm.dll`(与签名文件 `.xsgn` 同放 PI 的 `bin\`) |
| 链接 | `PCL.lib` + 五个 3rdparty 库 |
| **不要**链接 | `onnxruntime.lib` —— 引擎是运行时 `LoadLibraryW` 动态加载的 |

`onnxruntime.dll` 的定位顺序(见 `InferenceEngine.cpp`):
1. 环境变量 `BXT_ORT_DLL`(便于脱离 PI 单测)
2. 本模块所在目录下的 `onnxruntime.dll`(即 PI 的 `bin\`,本机实测 1.25.20260427,`ORT_API_VERSION 25`)

---

## 4. 签名

PI 1.9.x **强制模块签名**。自用走本地签名身份(`Edit > Local Signing Identity…`,
或 `Script > Development > SigningKeys`),会生成 `.xssk` 密钥文件。

```cmd
"D:\PixInsight\bin\PixInsight.exe" ^
  --sign-module-file="<你的>-pxm.dll" ^
  --xssk-file="<你的>.xssk" --xssk-password="<密码>"
```

生成同名 `.xsgn`。两者一起放进 `D:\PixInsight\bin\`,重启 PI。

> ⚠️ 本地签名身份**绑定本机 PI 许可证**。要部署到工控机,需把 `.xssk` 一并带过去
> 并在那台机器上注册(勾选 persistent),否则模块会被拒。
>
> 未经签名的模块需要 **CPD 身份**才能放行(见 PixInsight 官方公告)。

---

## 5. 验证顺序(建议)

1. **最小加载验证**:先编一个**空的** `MetaModule`(不建 process、不跑推理),签名、装到 PI,
   确认能被加载 —— 先打掉「ABI/签名」两个未知数,再谈算法。
2. **引擎单测**:脱离 PI,用 `BXT_ORT_DLL` 指向任一 ORT 1.25 的 DLL,
   对 `bxt4.onnx` 跑一次 `Load()` + 一次 `Predict()`,确认形状识别与 NHWC↔CHW 转置正确。
   预期:`InputShape() = {3,768,768}`、`OutputShape() = {3,512,512}`、`ParamsSize() = 5`。
3. **整图验证**:与 `py\bxt_bridge.py` 的输出逐像素比对 —— 两条路线应给出**逐位相同**的结果
   (都走 ORT + 同一份 FKv1 + 同一套 margin 坐标)。

---

## 6. 仍未验证的事项(诚实清单)

| # | 事项 | 为什么还没验 |
|---|---|---|
| 1 | 所有 C++ 能否编译 | 本机无编译器,一行都没编过 |
| 2 | PCL 在 v143 下是否零警告/零错误编过 | 同上(工程文件存在,但未跑过) |
| 3 | 五个 3rdparty 库之间的依赖顺序 | 未跑过 msbuild |
| 4 | 本地签名身份能否被 1.9.5 接受 | 需在 PI 里操作,而本机 pi 配置禁止改动 |
| 5 | GPU EP(CUDA / DirectML)接线 | 当前实现走 ORT 默认 EP;加 EP 需 `onnxruntime_ep_c_api.h` 里的通用 EP API |
| 6 | FKv1 曲线常数是否与官方 BXT 一致 | 参数**语义**已由官方手册确认;曲线**常数**仍是 fk-astro 的 |

---

## 7. 来源与合规

* `FKAstro*` 四个文件来自 `github.com/wfzylcl878-creator/fk-astro`(本地副本经 tree hash
  校验与上游逐字节一致)。**该仓库没有 LICENSE 文件** —— 自用无碍,若要分发编译产物须先解决授权。
* ORT 头文件来自 `github.com/microsoft/onnxruntime` tag `v1.25.0`,MIT 许可。
* 本模块**不读取、不修改** `RCAstroSuite-pxm.dll` 及其模型文件。
