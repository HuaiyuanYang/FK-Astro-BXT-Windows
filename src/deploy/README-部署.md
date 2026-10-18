# 部署说明

## 现状

`FKAstro-pxm.dll`(659 KB)已经编译好,放在本目录。**它没有签名,直接拷进
PixInsight 是装不上的** —— 必须先签名。

## 一步到位(签名 + 安装)

先按使用手册 §4.2 创建本地签名身份(记得勾 **Make the local signing identity persistent**),然后:

```powershell
cd <解压目录>\src\deploy
.\install-to-pi.ps1 -DllPath .\FKAstro-pxm.dll `
                    -XsskPath C:\path\to\你的密钥.xssk `
                    -XsskPassword "你的密码" `
                    -Sign -Install
```

| 参数 | 说明 |
|---|---|
| `-DllPath` | **必填**,要安装的 DLL |
| `-XsskPath` / `-XsskPassword` | 签名密钥与密码 |
| `-PixInsightRoot` | PixInsight 安装目录,默认 `D:\PixInsight` |
| `-Sign` | 执行签名 |
| `-Install` | 把 DLL 与 `.xsgn` 拷进 `bin\` |
| `-Force` | 目标已存在同名文件时也覆盖 |

## 只签名,不安装

```powershell
.\install-to-pi.ps1 -DllPath .\FKAstro-pxm.dll -XsskPath <密钥> -XsskPassword <密码> -Sign
```

会生成同目录下的 `FKAstro-pxm.xsgn`。之后把 `.dll` 和 `.xsgn` 一起手动拷进
`<PixInsight>\bin\` 也可以。

## 重新编译

```powershell
cd ..
.\bootstrap-toolchain.ps1      # 首次: 装 MSVC Build Tools + Windows SDK(需联网)
.\build.ps1 -Step pcl          # 编译 PCL(约 10 分钟,只需一次)
.\build.ps1 -Step module       # 编译模块 -> _module_build\FKAstro-pxm.dll
.\build.ps1 -Step verify       # 检查导出表与依赖表
```

编译完成后把 `_module_build\FKAstro-pxm.dll` 拷到本目录覆盖旧文件。

## 注意

- 未签名模块在 PixInsight 1.9 上**无法加载**
- 用 `Local Signing Identity` 签的模块只有**同一个 PixInsight 许可证**认得;
  给别人用,让对方用自己的身份重签一次
- 对外发布免签名加载需申请 **CPD**(Certified PixInsight Developer)
