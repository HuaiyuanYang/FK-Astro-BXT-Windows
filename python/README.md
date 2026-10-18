# python

FKv3 的 **Python 参考实现**,以及验证 / 转换工具。

C++ 模块移植时以 `fkv3.py` 为准 —— 它是唯一经过与官方 BXT 对比验证的实现。

---

## 核心文件

| 文件 | 作用 |
|---|---|
| **`fkv3.py`** | ★ FKv3 完整实现:整图 arcsinh 归一化 + 分块 + 余弦带融合 |
| `xisf_io.py` | 纯标准库的 XISF 读取器(不需要启动 PixInsight) |
| `fits_io.py` | FITS 读取(已处理 `BSCALE` / `BZERO`) |
| `star_metrics.py` | 找星、径向剖面、FWHM 测量 |
| `verify_fkv3.py` | 与官方 BXT 的逐像素对比(6 个场景) |
| `blind_test.py` | 5 张实拍图的盲测 |

## 工具

| 文件 | 作用 |
|---|---|
| `mil2onnx.py` | CoreML MIL → ONNX 转换器 |
| `mil_graph.py` / `onnx_writer.py` / `onnx_verify.py` | 转换器组件 |
| `onnx_run.py` / `weight_read.py` / `bxt_decode.py` | 推理封装、权重索引 |
| `make_synth_stars.py` / `make_scale_bank.py` | 合成测试图 |
| `md2html.py` | 用户手册的 Markdown → HTML 转换器 |

---

## 路径约定

| 东西 | 位置 |
|---|---|
| 模型 `bxt5.onnx` | 工作区根目录(与项目文件夹同级) |
| 测试图 / 官方参考输出 | 本项目的 `samples/` 目录(默认不存在,需自行新建) |

`verify_fkv3.py` 与 `blind_test.py` 会自动按上面两处查找,
缺失时会明确提示缺哪个文件。

---

## 怎么验证

验证脚本需要两样东西:**输入图** 和 **官方 BXT 处理同一张图的输出**。

### 1. 准备输入图

新建 `samples/` 目录,把测试图放进去。脚本内置的场景用到这些文件名:

| 文件名 | 说明 |
|---|---|
| `masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_..._autocrop.xisf` | 主验证图(深空叠加) |
| `Light_M31_300s_0002.fit` | 单色 16 位样本 |
| 其余见 `blind_test.py` 里的 `IMGS` | 盲测用 |

自定义路径请直接改脚本开头的 `SAMPLES` / `NGC` / `M31` 常量。

### 2. 生成官方参考输出

用 PixInsight 无头模式调用官方 BXT:

```javascript
// C:/tmp/bxtref.js   (路径不能有空格, 文件须纯 ASCII)
var w = ImageWindow.open("C:/path/to/input.xisf")[0];
var bxt = new BlurXTerminator;
bxt.overlap = 0.2;
bxt.correct_only = true;          // 或按需设 sharpen_* 等
bxt.executeOn(w.mainView, false);
w.saveAs("C:/path/to/samples/bxt_o2_co.xisf", false);
w.forceClose();
```

```powershell
D:\PixInsight\bin\PixInsight.exe -n --automation-mode ^
  --no-startup-scripts -r=C:\tmp\bxtref.js --force-exit
```

参考输出按场景命名(`bxt_ref_zero.xisf`、`bxt_o2_co.xisf` 等),
同样放进 `samples/`。

### 3. 跑对比

```powershell
python verify_fkv3.py --case o2co
```

判定标准(事先定死,不事后调整):**P99 < 2e-4 且相关系数 > 0.999**。

当前 Python 版实测 P99 = 7.5e-6 ~ 1.7e-4,相关 0.9995 ~ 0.9999999。

---

## 依赖

```powershell
pip install numpy onnxruntime
```

注意:endor/ 已移除。ONNX Runtime 请用 pip install onnxruntime 安装。
