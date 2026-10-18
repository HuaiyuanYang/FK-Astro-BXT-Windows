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

PixInsight 要求模块签名,**未签名的模块无法加载**。整个过程约 1 分钟。

### 第 1 步:创建签名身份(只需一次)

PixInsight 菜单 `Edit → Local Signing Identity...`,设定密码,会生成一个 `.xssk` 密钥文件。

> ⚠️ **务必勾选 "Make the local signing identity persistent"**
>
> 不勾选的话,签好的模块在加载时会报 `Unknown code signing identity`。
> 这是 PixInsight 1.9 上最常见的坑。

### 第 2 步:签名

把 `FKAstro-pxm.dll` 从压缩包里解出来,然后执行:

```
"D:\PixInsight\bin\PixInsight.exe" ^
  --sign-module-file="C:\路径\FKAstro-pxm.dll" ^
  --xssk-file="C:\路径\你的密钥.xssk" ^
  --xssk-password="你的密码"
```

成功后在 DLL 旁边会多出一个同名的 `FKAstro-pxm.xsgn` 文件。

### 第 3 步:拷贝并重启

把 **两个文件**一起放进 PixInsight 的模块目录:

```
<PixInsight 安装目录>\bin\
├── FKAstro-pxm.dll
└── FKAstro-pxm.xsgn
```

重启 PixInsight。菜单里出现 `Process → Preprocessing → FKAstroInference` 就是装好了。

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

**提示 `Unknown code signing identity`**

签名身份没有持久化。回到第 1 步,勾上 persistent 后重新签一次。

**提示找不到模型,或输出全黑**

`Model path` 没指对,或指到的不是本模块对应的模型文件。

**能不能拷给别人用?**

可以,但对方必须用**自己的**签名身份重新签一次(第 2 步)。
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
