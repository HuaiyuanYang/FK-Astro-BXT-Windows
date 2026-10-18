# build.ps1 —— 构建 bxt 的 PixInsight 原生模块(Windows)
#
# 本脚本里的每一个参数都是**实测跑通过的**,不是从文档抄的:
#   2026-10-17  196/196 PCL 源文件编译成功, 4/4 模块源文件编译成功,
#               FKAstro-pxm.dll 链接成功, 导出表与依赖表经 dumpbin 验证,
#               引擎与 numpy 参考实现逐位一致(0/786432 差异)。
#
# 关键参数为什么是这些(全部来自 D:\PixInsight\src\pcl\windows\vc17\PCL.vcxproj):
#   /MD             RuntimeLibrary = MultiThreadedDLL。**必须**与 PI 一致:
#                   pcl::Error 会从本模块抛进 PI, 两侧 CRT 不同会出问题。
#   /std:c++20      LanguageStandard = stdcpp20(不是 c++17)
#   /EHa            ExceptionHandling = Async
#   /Zc:__cplusplus /permissive-    AdditionalOptions
#   /utf-8          源码是 UTF-8, 不加会被按 GBK 解读(报 C4819)
#   __PCL_* 宏      不定义会在 pcl/Defs.h 直接 #error
#
# 依赖:
#   * 便携工具链 —— 由 bootstrap-toolchain.ps1 从 VS 离线布局自建, 无需管理员
#   * Windows SDK —— 同上
#   * 不需要 MSBuild, 不需要 Qt, 不需要那五个 3rdparty 库(已实测可省)
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Step env      只查环境
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Step pcl      编 PCL(~4 分钟)
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Step module   编模块并链接
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Step test     编引擎测试程序
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Step verify   dumpbin 验证产物
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Step all      全部
#   ... -Clean                                                        先清空中间产物

param(
    [ValidateSet('all', 'env', 'pcl', 'module', 'test', 'v3test', 'verify')]
    [string]$Step = 'all',

    [string]$ToolchainRoot  = 'D:\PI-Plugin-Dev\toolchain',
    [string]$SdkRoot        = 'D:\PI-Plugin-Dev\winsdk\Windows Kits\10',
    [string]$PixInsightRoot = 'D:\PixInsight',
    [string]$SdkVersion     = '10.0.26100.0',

    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

function Say  ([string]$m) { Write-Host "[build ] $m" -ForegroundColor Cyan }
function Warn ([string]$m) { Write-Host "[warn  ] $m" -ForegroundColor Yellow }
function Die  ([string]$m) { Write-Host "[FAIL  ] $m" -ForegroundColor Red; exit 1 }

$ModuleRoot = $PSScriptRoot
$SrcDir     = $ModuleRoot          # 源码与 build.ps1 同级
$OrtInc     = Join-Path $ModuleRoot 'third_party\onnxruntime'
$PclBuild   = Join-Path $ModuleRoot '_pcl_build'
$ModBuild   = Join-Path $ModuleRoot '_module_build'
$TestBuild  = Join-Path $ModuleRoot '_engine_test'

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
function Set-BuildEnv {
    # 注意: PowerShell 变量名大小写不敏感, 所以这里刻意用 $msvcDir 而不是 $msvc,
    # 否则 $Msvc 会命中局部变量(DirectoryInfo), 而 .NET Framework 的
    # DirectoryInfo.ToString() 返回 Name 而非 FullName, 路径会塌成 "14.44.35207"。
    $msvcDir = Get-ChildItem (Join-Path $ToolchainRoot 'VC\Tools\MSVC') -Directory -ErrorAction SilentlyContinue |
               Sort-Object Name -Descending | Select-Object -First 1
    if (-not $msvcDir) {
        Die @"
找不到 MSVC 工具链: $ToolchainRoot\VC\Tools\MSVC

  请先运行 bootstrap-toolchain.ps1, 它会从 VS 离线布局里解出便携工具链
  (不需要管理员权限, 也不需要运行 VS 安装器)。
"@
    }
    $script:Msvc = $msvcDir.FullName
    $binDir = Join-Path $script:Msvc 'bin\Hostx64\x64'

    foreach ($exe in @('cl.exe', 'lib.exe', 'link.exe')) {
        if (-not (Test-Path (Join-Path $binDir $exe))) { Die "工具链缺少 $exe : $binDir" }
    }

    $sdkInc = Join-Path $SdkRoot "Include\$SdkVersion"
    $sdkLib = Join-Path $SdkRoot "Lib\$SdkVersion"
    foreach ($d in @("$sdkInc\ucrt", "$sdkInc\shared", "$sdkInc\um", "$sdkLib\ucrt\x64", "$sdkLib\um\x64")) {
        if (-not (Test-Path $d)) { Die "Windows SDK 不完整, 缺: $d`n  请先运行 bootstrap-toolchain.ps1" }
    }

    $pclInc = Join-Path $PixInsightRoot 'include'
    $tpInc  = Join-Path $PixInsightRoot 'src\3rdparty'
    if (-not (Test-Path (Join-Path $pclInc 'pcl\Defs.h'))) { Die "找不到 PCL 头文件: $pclInc\pcl\Defs.h" }

    $env:INCLUDE = @(
        (Join-Path $Msvc 'include'), $pclInc, $tpInc,
        "$sdkInc\ucrt", "$sdkInc\shared", "$sdkInc\um", "$sdkInc\winrt"
    ) -join ';'

    # onecore\x64 里有 msvcrt.lib / msvcprt.lib / vcruntime.lib —— 桌面 lib\x64 里没有这几个
    $env:LIB = @(
        (Join-Path $Msvc 'lib\x64'), (Join-Path $Msvc 'lib\onecore\x64'),
        "$sdkLib\ucrt\x64", "$sdkLib\um\x64"
    ) -join ';'

    $env:Path = "$binDir;$env:Path"

    Say "MSVC      $(Split-Path $Msvc -Leaf)"
    Say "SDK       $SdkVersion"
    Say "PCL 头    $pclInc"
    $n = (Get-ChildItem (Join-Path $pclInc 'pcl') -File -Filter *.h).Count
    Say "PCL 头数  $n"
}

# 实测跑通的编译/链接参数
$Defs = @(
    '/DWIN32', '/DWIN64', '/D_WINDOWS', '/DUNICODE',
    '/D__PCL_WINDOWS', '/D__PCL_NO_WIN32_MINIMUM_VERSIONS',
    '/D__PCL_AVX2', '/D__PCL_FMA', '/DNDEBUG'
)
$Common = @(
    '/nologo', '/c', '/EHa', '/O2', '/MD', '/std:c++20', '/utf-8',
    '/Zc:__cplusplus', '/permissive-', '/W3'
)

# ---------------------------------------------------------------------------
# PCL
# ---------------------------------------------------------------------------
function Build-Pcl {
    $pclSrc = Join-Path $PixInsightRoot 'src\pcl'
    $srcs = Get-ChildItem $pclSrc -File -Filter *.cpp | Select-Object -ExpandProperty FullName
    if (-not $srcs) { Die "在 $pclSrc 下没找到 .cpp" }

    New-Item -ItemType Directory -Force -Path $PclBuild | Out-Null
    if ($Clean) { Remove-Item "$PclBuild\*.obj", "$PclBuild\PCL.lib" -ErrorAction SilentlyContinue }

    $incs = @("/I`"$(Join-Path $PixInsightRoot 'include')`"", "/I`"$(Join-Path $PixInsightRoot 'src\3rdparty')`"")
    Say "编译 PCL: $($srcs.Count) 个源文件(cl /MP 并行, 约 4 分钟) ..."
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $log = Join-Path $PclBuild 'build.log'
    & cl.exe @Common '/MP' @Defs @incs /Fo"$PclBuild\\" @srcs > $log 2>&1
    $rc = $LASTEXITCODE
    $sw.Stop()

    $objs = Get-ChildItem $PclBuild -File -Filter *.obj -ErrorAction SilentlyContinue
    Say "cl exit=$rc  $($objs.Count)/$($srcs.Count) 个 .obj  $([math]::Round($sw.Elapsed.TotalSeconds,1)) 秒"
    if ($objs.Count -lt $srcs.Count) {
        Warn "有源文件未编译成功:"
        Select-String -Path $log -Pattern ': error ' | Select-Object -First 12 |
            ForEach-Object { Write-Host ("        " + $_.Line.Trim()) -ForegroundColor DarkYellow }
        Die "PCL 构建不完整"
    }

    $lib = Join-Path $PclBuild 'PCL.lib'
    Remove-Item $lib -ErrorAction SilentlyContinue
    & lib.exe /nologo /OUT:"$lib" ($objs | Select-Object -ExpandProperty FullName) 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $lib)) { Die "lib.exe 打包失败" }
    Say "PCL.lib = $([math]::Round((Get-Item $lib).Length/1MB,2)) MB"
}

# ---------------------------------------------------------------------------
# 模块
# ---------------------------------------------------------------------------
function Build-Module {
    if (-not (Test-Path (Join-Path $PclBuild 'PCL.lib'))) { Die "还没有 PCL.lib, 先跑 -Step pcl" }
    New-Item -ItemType Directory -Force -Path $ModBuild | Out-Null
    if ($Clean) { Remove-Item "$ModBuild\*.obj", "$ModBuild\*.dll", "$ModBuild\*.lib", "$ModBuild\*.exp" -ErrorAction SilentlyContinue }

    $srcs = @('FKAstroModule.cpp', 'FKAstroInterface.cpp', 'FKAstroProcess.cpp', 'InferenceEngine.cpp') |
            ForEach-Object { Join-Path $SrcDir $_ }
    foreach ($s in $srcs) { if (-not (Test-Path $s)) { Die "缺源文件: $s" } }
    if (-not (Test-Path $OrtInc)) { Die "缺 ORT 头文件目录: $OrtInc" }

    $incs = @(
        "/I`"$(Join-Path $PixInsightRoot 'include')`"",
        "/I`"$(Join-Path $PixInsightRoot 'src\3rdparty')`"",
        "/I`"$SrcDir`"", "/I`"$OrtInc`""
    )

    Say "编译模块: $($srcs.Count) 个源文件 ..."
    & cl.exe @Common @Defs @incs /Fo"$ModBuild\\" @srcs 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) { Die "模块编译失败" }

    $objs = Get-ChildItem $ModBuild -File -Filter *.obj | Select-Object -ExpandProperty FullName
    if ($objs.Count -ne 4) { Die "期望 4 个 .obj, 实得 $($objs.Count)" }

    $dll = Join-Path $ModBuild 'FKAstro-pxm.dll'
    Say "链接 -> $dll"
    # 只链 PCL.lib。ORT 是运行时 LoadLibrary 动态加载的, 不链 onnxruntime.lib。
    & link.exe /nologo /DLL /OUT:"$dll" /LIBPATH:"$PclBuild" @objs "$PclBuild\PCL.lib" 2>&1 |
        ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $dll)) { Die "链接失败" }
    Say "FKAstro-pxm.dll = $([math]::Round((Get-Item $dll).Length/1KB,1)) KB"
}

# ---------------------------------------------------------------------------
# 引擎测试程序
# ---------------------------------------------------------------------------
function Build-Test {
    New-Item -ItemType Directory -Force -Path $TestBuild | Out-Null
    $srcs = @((Join-Path $SrcDir 'engine_test.cpp'), (Join-Path $SrcDir 'InferenceEngine.cpp'))
    $incs = @("/I`"$SrcDir`"", "/I`"$OrtInc`"")
    Say "编译 engine_test.exe ..."
    & cl.exe /nologo /EHsc /O2 /MD /std:c++17 /utf-8 /DNDEBUG /DWIN32 /D_WINDOWS @incs `
        /Fo"$TestBuild\\" /Fe"$TestBuild\engine_test.exe" @srcs 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) { Die "engine_test 编译失败" }
    Say "engine_test.exe 就绪"
    Say "跑等价性对账(先 python 生成输入, 再 C++ 推理, 再 python 比对):"
    Say "    `$env:BXT_ORT_DLL='$PixInsightRoot\bin\onnxruntime.dll'"
    Say "    python py\engine_equiv_test.py v4"
    Say "    _engine_test\engine_test.exe ..\..\bxt5.onnx _engine_test\in.f32 _engine_test\cxx_out.f32 0 0 0 1 0"
    Say "    python py\engine_equiv_test.py v4"
}

# ---------------------------------------------------------------------------
# FKv3 离线自检(不需要 PCL)
# ---------------------------------------------------------------------------
function Build-V3Test {
    New-Item -ItemType Directory -Force -Path $TestBuild | Out-Null
    $srcs = @((Join-Path $SrcDir 'fkv3_selftest.cpp'), (Join-Path $SrcDir 'InferenceEngine.cpp'))
    foreach ($s in $srcs) { if (-not (Test-Path $s)) { Die "缺源文件: $s" } }
    $incs = @("/I`"$SrcDir`"", "/I`"$OrtInc`"")
    Say "编译 fkv3_selftest.exe ..."
    & cl.exe /nologo /EHsc /O2 /MD /std:c++17 /utf-8 /DNDEBUG /DWIN32 /D_WINDOWS @incs `
        /Fo"$TestBuild\\" /Fe"$TestBuild\fkv3_selftest.exe" @srcs 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) { Die "fkv3_selftest 编译失败" }
    Say "fkv3_selftest.exe = $([math]::Round((Get-Item (Join-Path $TestBuild 'fkv3_selftest.exe')).Length/1KB,1)) KB"
    Say ""
    Say "用法(把 ORT 指到 PI 的 bin):"
    Say "    `$env:BXT_ORT_DLL='$PixInsightRoot\bin\onnxruntime.dll'"
    Say "    _engine_test\fkv3_selftest.exe ..\..\bxt5.onnx $env:TEMP\v3 512 512 3 0.2 co"
    Say "    python python\compare_fkv3_selftest.py $env:TEMP\v3"
}
# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
function Verify-Artifacts {
    $dll = Join-Path $ModBuild 'FKAstro-pxm.dll'
    if (-not (Test-Path $dll)) { Die "没有产物: $dll" }

    Say "=== PE 头 ==="
    & dumpbin.exe /headers $dll 2>&1 |
        Select-String -Pattern 'machine|magic|File Type' | ForEach-Object { Say ("  " + $_.Line.Trim()) }

    Say "=== 导出表(必须有这三个) ==="
    $exp = & dumpbin.exe /exports $dll 2>&1
    foreach ($n in @('InstallPixInsightModule', 'InitializePixInsightModule', 'IdentifyPixInsightModule')) {
        if ($exp | Select-String -Pattern $n -SimpleMatch) { Say "  OK   $n" } else { Die "缺少导出: $n" }
    }

    Say "=== 依赖表(不应出现 onnxruntime.dll) ==="
    $dep = & dumpbin.exe /dependents $dll 2>&1
    $dep | Select-String -Pattern '\.dll' | ForEach-Object { Say ("  " + $_.Line.Trim()) }
    if ($dep | Select-String -Pattern 'onnxruntime' -SimpleMatch) {
        Warn "依赖里出现了 onnxruntime.dll —— 动态加载设计被破坏, 会与 PI 的 ORT 版本耦合"
    } else {
        Say "  OK   未链接 onnxruntime.dll(动态加载生效)"
    }
    Say "验证通过"
}

# ---------------------------------------------------------------------------
Say "步骤: $Step"
switch ($Step) {
    'env'    { Set-BuildEnv }
    'pcl'    { Set-BuildEnv; Build-Pcl }
    'module' { Set-BuildEnv; Build-Module }
    'test'   { Set-BuildEnv; Build-Test }
    'v3test' { Set-BuildEnv; Build-V3Test }
    'verify' { Set-BuildEnv; Verify-Artifacts }
    'all'    { Set-BuildEnv; Build-Pcl; Build-Module; Verify-Artifacts }
}
Say "完成: $Step"
