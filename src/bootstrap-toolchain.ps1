# bootstrap-toolchain.ps1 —— 从 Visual Studio 离线布局自建便携 MSVC 工具链
#
# 为什么需要这个:
#   VS 的安装器(vs_BuildTools.exe --layout / --install)在无人值守环境里并不可靠,
#   而且正式安装需要管理员权限。但离线布局本身只是一堆 ZIP(vsix)和 MSI ——
#   完全可以手工解包成一套**免安装、免管理员**的工具链。
#
# 两个关键手法:
#   1) `.vsix` 本质是 ZIP。VS 的每个组件包目录下都有一个 payload.vsix,
#      其中 `Contents/` 前缀之下就是最终会落进安装目录的**真实文件**。
#      解包时剥掉 `Contents/` 前缀即可。
#   2) Windows SDK 是 78 个 MSI + 149 个哈希命名的 .cab(cab 内文件名也是哈希),
#      直接 expand 出来没有正确路径。但 `msiexec /a`(**管理安装**)会读取
#      MSI 里的映射表, 把文件按正确目录结构解到 TARGETDIR ——
#      这是**纯解包**, 不写注册表、不改系统、不需要管理员。
#
# 实测:2026-10-17 用本方法在本机构建出
#   toolchain\  450 文件 / 280 MB   MSVC 14.44.35207 (cl.exe / link.exe / lib.exe)
#   winsdk\   9488 文件 / 1.79 GB   Windows SDK 10.0.26100
# 并用它成功编译了 196 个 PCL 源文件与整个 PixInsight 模块。
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File bootstrap-toolchain.ps1 -Part msvc
#   powershell -ExecutionPolicy Bypass -File bootstrap-toolchain.ps1 -Part sdk
#   powershell -ExecutionPolicy Bypass -File bootstrap-toolchain.ps1 -Part all
#
# 前置:先用引导器把布局下载下来(只需 --layout, 不需要安装):
#   vs_BuildTools.exe --layout D:\PI-Plugin-Dev\vs-layout --lang en-US ^
#       --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
#       --add Microsoft.VisualStudio.Component.Windows11SDK.26100

param(
    [ValidateSet('all', 'msvc', 'sdk', 'verify')]
    [string]$Part = 'all',

    [string]$LayoutDir    = 'D:\PI-Plugin-Dev\vs-layout',
    [string]$ToolchainRoot= 'D:\PI-Plugin-Dev\toolchain',
    [string]$SdkRoot      = 'D:\PI-Plugin-Dev\winsdk'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue

function Say  ([string]$m) { Write-Host "[boot  ] $m" -ForegroundColor Cyan }
function Warn ([string]$m) { Write-Host "[warn  ] $m" -ForegroundColor Yellow }
function Die  ([string]$m) { Write-Host "[FAIL  ] $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $LayoutDir)) { Die "找不到离线布局: $LayoutDir" }

# ---------------------------------------------------------------------------
# 1) MSVC:把 payload.vsix 当 ZIP 解, 剥掉 Contents/ 前缀
# ---------------------------------------------------------------------------
# 这些包合起来构成一套完整的 x64 工具链。
# 注意 x64 的**动态 CRT 导入库**(msvcrt.lib / msvcprt.lib / vcruntime.lib /
# oldnames.lib)不在 CRT.x64.Desktop 里, 而在 CRT.x64.OneCore.Desktop 的
# lib\onecore\x64\ 下 —— 少这一个包就编不出 /MD 的产物。
$MsvcPackages = @(
    @{ Pattern = 'Microsoft.VC.14.*.Tools.HostX64.TargetX64.base*';        What = 'cl.exe / link.exe / c1xx.dll / c2.dll' }
    @{ Pattern = 'Microsoft.VC.14.*.Tools.HostX64.TargetX64.Res.base*';    What = '工具集 en-US 资源' }
    @{ Pattern = 'Microsoft.VC.14.*.CRT.Headers.base*';                    What = 'CRT + STL 头文件(<vector> 等)' }
    @{ Pattern = 'Microsoft.VC.14.*.CRT.Headers.Resources.base*';          What = '头文件 en-US 资源' }
    @{ Pattern = 'Microsoft.VC.14.*.CRT.x64.Desktop.base*';                What = 'x64 静态 CRT 库(libcmt 等)' }
    @{ Pattern = 'Microsoft.VC.14.*.CRT.x64.OneCore.Desktop.base*';        What = 'x64 动态 CRT 导入库(msvcrt 等)' }
    @{ Pattern = 'Microsoft.VisualCpp.Tools.HostX64.TargetX64*';           What = 'vcvars64.bat 占位' }
)

function Expand-Vsix($vsixPath, $destRoot) {
    $z = [System.IO.Compression.ZipFile]::OpenRead($vsixPath)
    $n = 0
    try {
        foreach ($e in $z.Entries) {
            $fn = $e.FullName
            if ($fn -notlike 'Contents/*') { continue }
            if ($fn.EndsWith('/')) { continue }
            $rel  = $fn.Substring('Contents/'.Length)
            $dest = Join-Path $destRoot ($rel -replace '/', '\')
            $pd   = Split-Path $dest -Parent
            if (-not (Test-Path $pd)) { New-Item -ItemType Directory -Force -Path $pd | Out-Null }
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dest, $true)
            $n++
        }
    } finally { $z.Dispose() }
    return $n
}

function Build-Msvc {
    Say "=== MSVC 工具链 -> $ToolchainRoot ==="
    New-Item -ItemType Directory -Force -Path $ToolchainRoot | Out-Null
    $dirs = Get-ChildItem $LayoutDir -Directory

    foreach ($pkg in $MsvcPackages) {
        $d = $dirs | Where-Object { $_.Name -like $pkg.Pattern } | Select-Object -First 1
        if (-not $d) { Warn "缺包: $($pkg.Pattern)  ($($pkg.What))"; continue }
        $v = Get-ChildItem $d.FullName -Recurse -File -Filter 'payload.vsix' -ErrorAction SilentlyContinue |
             Select-Object -First 1
        if (-not $v) { Warn "包内无 payload.vsix: $($d.Name)"; continue }
        $n = Expand-Vsix $v.FullName $ToolchainRoot
        Say ("  {0,-58} {1,5} 文件   {2}" -f $d.Name.Substring(0, [Math]::Min(56, $d.Name.Length)), $n, $pkg.What)
    }

    foreach ($d in @(Get-ChildItem (Join-Path $ToolchainRoot 'VC\Tools\MSVC') -Directory -ErrorAction SilentlyContinue |
                     Sort-Object Name -Descending | Select-Object -First 1)) {
        Say "  MSVC 版本: $($d.Name)"
    }
}

# ---------------------------------------------------------------------------
# 2) Windows SDK:msiexec /a 逐个 MSI 解包
# ---------------------------------------------------------------------------
function Build-Sdk {
    $sdkPkg = Get-ChildItem $LayoutDir -Directory |
              Where-Object { $_.Name -like 'Win11SDK*' } | Select-Object -First 1
    if (-not $sdkPkg) { Die "布局里找不到 Win11SDK* 目录 —— 下载时是否加了 Windows11SDK 组件?" }

    $inst = Join-Path $sdkPkg.FullName 'Installers'
    if (-not (Test-Path $inst)) { Die "找不到 $inst" }

    $msis = Get-ChildItem $inst -File -Filter *.msi
    $cabs = Get-ChildItem $inst -File -Filter *.cab
    Say "=== Windows SDK -> $SdkRoot ==="
    Say "  包: $($sdkPkg.Name)"
    Say "  $($msis.Count) 个 MSI, $($cabs.Count) 个 cab"
    Say "  逐个 msiexec /a(管理安装 = 纯解包, 不需要管理员)..."

    New-Item -ItemType Directory -Force -Path $SdkRoot | Out-Null
    $changed = 0
    $i = 0
    foreach ($m in ($msis | Sort-Object Name)) {
        $i++
        $before = (Get-ChildItem $SdkRoot -Recurse -File -ErrorAction SilentlyContinue).Count
        # start /wait 是必要的: msiexec 是 GUI 子系统程序, 直接调用会立刻返回
        & cmd.exe /c "start /wait `"`" msiexec.exe /a `"$($m.FullName)`" /qn TARGETDIR=`"$SdkRoot`"" | Out-Null
        $after = (Get-ChildItem $SdkRoot -Recurse -File -ErrorAction SilentlyContinue).Count
        if ($after -gt $before) {
            $changed++
            Write-Host ("        [{0,2}/{1}] {2,-58} +{3}" -f $i, $msis.Count,
                        $m.Name.Substring(0, [Math]::Min(56, $m.Name.Length)), ($after - $before))
        }
    }
    Say "  $changed 个 MSI 贡献了文件"
}

# ---------------------------------------------------------------------------
# 3) 验证
# ---------------------------------------------------------------------------
function Verify-All {
    Say "=== 验证 ==="
    $fail = 0

    $msvcDir = Get-ChildItem (Join-Path $ToolchainRoot 'VC\Tools\MSVC') -Directory -ErrorAction SilentlyContinue |
               Sort-Object Name -Descending | Select-Object -First 1
    if (-not $msvcDir) { Warn "没有 MSVC 目录"; $fail++ }
    else {
        foreach ($rel in @('bin\Hostx64\x64\cl.exe', 'bin\Hostx64\x64\link.exe',
                           'bin\Hostx64\x64\lib.exe', 'include\vector', 'include\vcruntime.h',
                           'lib\x64\libcmt.lib', 'lib\onecore\x64\msvcrt.lib')) {
            $p = Join-Path $msvcDir.FullName $rel
            if (Test-Path $p) { Say ("  OK   {0,-34} {1,9} KB" -f $rel, [math]::Round((Get-Item $p).Length/1KB, 0)) }
            else              { Warn ("  缺   {0}" -f $rel); $fail++ }
        }
    }

    $sdkInc = Get-ChildItem (Join-Path $SdkRoot 'Windows Kits\10\Include') -Directory -ErrorAction SilentlyContinue |
              Sort-Object Name -Descending | Select-Object -First 1
    if (-not $sdkInc) { Warn "没有 SDK Include 目录"; $fail++ }
    else {
        Say "  SDK 版本: $($sdkInc.Name)"
        foreach ($rel in @('um\windows.h', 'um\winnt.h', 'shared\winapifamily.h',
                           'shared\sdkddkver.h', 'ucrt\stdio.h')) {
            $p = Join-Path $sdkInc.FullName $rel
            if (Test-Path $p) { Say ("  OK   {0}" -f $rel) } else { Warn ("  缺   {0}" -f $rel); $fail++ }
        }
        $n = (Get-ChildItem $sdkInc.FullName -Recurse -File -Filter *.h -ErrorAction SilentlyContinue).Count
        Say "  SDK 头文件总数: $n"
    }

    $sdkLib = Get-ChildItem (Join-Path $SdkRoot 'Windows Kits\10\Lib') -Directory -ErrorAction SilentlyContinue |
              Sort-Object Name -Descending | Select-Object -First 1
    if ($sdkLib) {
        foreach ($rel in @('um\x64\kernel32.lib', 'ucrt\x64\ucrt.lib')) {
            $p = Join-Path $sdkLib.FullName $rel
            if (Test-Path $p) { Say ("  OK   {0}" -f $rel) } else { Warn ("  缺   {0}" -f $rel); $fail++ }
        }
    } else { Warn "没有 SDK Lib 目录"; $fail++ }

    if ($fail -eq 0) { Say "工具链完整可用" } else { Warn "$fail 项缺失 —— 检查下载时是否选全了组件" }
    return $fail
}

# ---------------------------------------------------------------------------
Say "步骤: $Part   布局: $LayoutDir"
switch ($Part) {
    'msvc'   { Build-Msvc }
    'sdk'    { Build-Sdk }
    'verify' { Verify-All | Out-Null }
    'all'    { Build-Msvc; Build-Sdk; Verify-All | Out-Null }
}
Say "完成: $Part"
Say "接下来: powershell -ExecutionPolicy Bypass -File build.ps1 -Step env"
