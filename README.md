# FK-Astro:BXT

Windows 版 PixInsight 神经网络去卷积模块。

用 AI 模型收紧星点、恢复星系与星云的细节,代替需要手工给 PSF、反复调参数的
传统反卷积。参数体系与 BlurXTerminator 一致,习惯可以直接迁移。

---

## 它能做什么

- 星点更紧实,顺带修正轻微的光学像差(球差、彗差)
- 星系旋臂、行星状星云的边缘轮廓更明确
- 把「估 PSF + 迭代次数 + 正则化强度」三四个互相牵制的参数压成一两个强度值

**做不到**:不降噪、不去星、不修拖线、不做色彩校正或背景提取。

---

## 运行要求

| 项目 | 要求 |
|---|---|
| 操作系统 | **Windows x64**(仅此一种) |
| PixInsight | **1.9 或更新** |
| 内存 | 建议 16 GB 以上;6000 万像素级图像建议 32 GB |

**不需要另装任何运行库。** 推理引擎(PixInsight 1.9 自带的 ONNX Runtime)
和 C++ 运行库 PixInsight 都已包含。

**关于 GPU**:PixInsight 自带的那份 ONNX Runtime 是 **CPU-only 构建**,所以默认跑在 CPU 上
(约每百万像素 8 秒)。想启用 GPU 需要另外准备一份**同为 1.25.x**、带 CUDA 或 DirectML
执行器的 ONNX Runtime,用环境变量 `BXT_ORT_DLL` 指过去即可,**不必替换 PixInsight 的文件**;
它依赖的 provider DLL 要和它放在同一目录。详见手册 §5.2。

---

## 准备模型文件

本模块**不附带模型权重**。你需要一个 ONNX 模型文件(约 185 MB)。

放在哪里都可以,装好模块后在 `Model path` 里指过去即可。建议单独建一个目录:

```
D:\Astro\Models\bxt5.onnx
```

模型的获取方式与授权请自行确认。

---

## 安装

PixInsight 对第三方模块有**三道关卡**,少一步都不会出现在菜单里:

| 步骤 | 缺了会怎样 |
|---|---|
| **签名** | 未签名的模块直接拒绝加载 |
| **拷贝** | `.dll` 和 `.xsgn` 必须都在 `bin\` 下 |
| **登记** | PI 1.9 **不再自动扫描 `bin\`** —— 文件在,菜单里却没有,而且**不报错** |

全部做完约 3 分钟。

### 第 1 步:生成签名密钥(只需一次)

菜单 `Script → Development → SigningKeys`,在弹出的窗口里:

- ☑ **Generate Signing Keys**
- **Developer** — **留空**(那是给认证开发者 CPD 用的)
- ☑ **Local signing identity** ← 关键
- **Output → Keys file** — 必须以 `.xssk` 结尾,例如 `C:\PI\me.xssk`
- **Output → Password** — 设一个保护密码,**记牢**
- 点 **Run**

### 第 2 步:把密钥注册给 PixInsight(只需一次)

菜单 `Edit → Local Signing Identity...`,选中上一步的 `.xssk`,输入密码,**勾上「Make the local signing identity persistent」**,确定。

> ⚠️ 这个对话框只能**选择**已有文件,不能新建 —— 所以顺序不能反。
>
> ⚠️ 不勾持久化的话,模块加载时会报 `Unknown code signing identity`。

### 第 3 步:签名 + 安装 + 登记

**先把 PixInsight 完全退出**,然后在解压目录下执行:

```
cd <解压目录>\src\deploy
.\install-to-pi.ps1 -DllPath .\FKAstro-pxm.dll `
                    -XsskPath C:\PI\me.xssk `
                    -XsskPassword "你的密码" `
                    -Sign -Install -Register
```

`-Register` 会把模块写进 PixInsight 的模块清单(会自动备份设置文件)。

> 不习惯用脚本?手册 §4.2 有完整的手动步骤。

### 第 4 步:启动 PixInsight

菜单里出现 `Process → Preprocessing → FKAstroInference` 就是装好了。

---

## 使用

选中一张图像 → 打开 `FKAstroInference` → 在 **Model path** 里指到你的模型文件 → `Apply`。

**必须在线性(未拉伸)数据上运行。** 已经拉伸过的图,去卷积的数学前提不成立。

参数保持默认即可。第一次使用建议先勾上 **Correct only**,只修光学像差、不动锐化,
确认效果没问题再放开。

完整的参数说明、处理顺序和常见问题见 `docs/manual.html`。

在 GitHub 页面上直接点开这个文件只会看到源码,**下载压缩包后用浏览器打开**即可正常阅读。

---

## 常见问题

**菜单里没有 FKAstroInference,启动也不报错**

漏了**登记**这一步。PixInsight 1.9 不会自动扫描 `bin\` 里的新 DLL,文件放对了、签名也对了,没写进模块清单就不会出现 —— 而且不提示。跑一次 `install-to-pi.ps1 -Register` 即可。

**提示 `Unknown code signing identity`**

签名身份没有持久化。回到第 2 步,勾上 persistent,然后重新签一次(第 3 步)。

**勾了 Correct only,跑完看不出变化**

正常。这个模式只修正星点形状,改动量在万分之一量级,屏幕上几乎不可见。想看明显效果请关掉它、把 `Sharpen stars` 调到 0.5 以上。

**签名时脚本卡住,或提示要先关闭 PixInsight**

签名必须在 PI 完全退出时做。PI 已经在运行时,新进程只会把请求转交给它并立刻返回,签名变成异步的,脚本无从判断何时结束。

**提示找不到模型,或输出全黑**

`Model path` 没指对,或指到的不是本模块对应的模型文件。

**能不能拷给别人用?**

可以,但对方必须用**自己的**签名身份重新签一次(第 3 步)。
签名与 PixInsight 许可证绑定,已签好的文件拷到别人机器上不认。

**处理到一半内存不足**

本模块按三通道双精度累加融合结果,大图会额外占用约每百万像素 24 MB。
6000 万像素级图像建议 32 GB 内存,或先裁切再处理。

---

## 目录说明

日常使用只需要 `FKAstro-pxm.dll` 和手册。其余目录是源码、构建脚本和模型转换工具,
供需要自行编译或转换模型的人使用,普通使用不需要碰。

---

## 许可

源代码与文档采用 **MIT**,见 [LICENSE](LICENSE)。

MIT 只覆盖本仓库的代码与文档,不包括 PixInsight 及其 PCL、BlurXTerminator,
也不包括任何模型权重。模型由使用者自备,其使用方式需自行确认合规。

本模块基于开源项目 fk-astro 的调用约定独立实现,与 RC Astro、
Pleiades Astrophoto 均无隶属关系,未获其背书。

---

## 致谢

感谢 [企鹅](https://github.com/wfzylcl878-creator) 的前期工作。
