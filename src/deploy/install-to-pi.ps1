# install-to-pi.ps1 —— 把 FKAstro-pxm.dll 部署进一台 PixInsight
#
# 设计原则:**默认什么都不做**。不加 -Sign / -Install 就只是体检并打印计划。
# 要真正动手, 必须显式给开关。
#
# 为什么要这么谨慎:
#   * 会写 D:\PixInsight\bin\ —— PI 的安装目录
#   * 会调用 PixInsight.exe 生成签名
#   * 拷贝前会先备份同名文件, 但 PI 目录不是我们的地盘
#
# 关键事实(来自 PixInsight 官方论坛, 2025-10 帖):
#   * 本地签名身份的 ID **派生自 PI 许可证的 User ID** —— 所以签名是设备相关的,
#     DLL 是设备无关的。换机器时**就地重签**, 不要搬 .xsgn。
#   * 必须把本地签名身份设为**持久化**(Edit → Local Signing Identity… 里的
#     "Make the local signing identity persistent"), 否则加载时报
#     `Unknown code signing identity '<id>'`。论坛上两个人踩的都是这个坑。
#
# 用法(在**目标机器**上跑):
#   # 1) 先体检, 什么都不改
#   powershell -ExecutionPolicy Bypass -File install-to-pi.ps1 -DllPath .\FKAstro-pxm.dll
#
#   # 2) 确认无误后签名 + 安装 + 注册(三件一起做)
#   powershell -ExecutionPolicy Bypass -File install-to-pi.ps1 -DllPath .\FKAstro-pxm.dll `
#       -XsskPath D:\keys\me.xssk -XsskPassword 'xxx' -Sign -Install -Register -Force
#
#   # 3) 重启 PixInsight
#
# 关于 -Register:
#   PI 1.9 不再自动发现 bin\ 里的新 DLL —— 它照 %APPDATA%\Pleiades 里那份
#   模块清单加载。不登记的话, 文件在 bin 里、进程却不会出现在菜单中。
#   -Register 就是往那份清单里加一行(会先备份设置文件)。

param(
    [Parameter(Mandatory=$true)][string]$DllPath,

    [string]$PixInsightRoot = 'D:\PixInsight',
    [string]$XsskPath,
    [string]$XsskPassword,

    [switch]$Sign,
    [switch]$Install,
    [switch]$Register,      # 把模块登记进 PI 的模块表(1.9 必需, 否则不加载)
    [switch]$Force          # 目标已存在同名文件时也覆盖(默认会拒绝)
)

$ErrorActionPreference = 'Stop'

function Say  ([string]$m) { Write-Host "[deploy] $m" -ForegroundColor Cyan }
function Warn ([string]$m) { Write-Host "[warn  ] $m" -ForegroundColor Yellow }
function Die  ([string]$m) { Write-Host "[FAIL  ] $m" -ForegroundColor Red; exit 1 }

$exe = Join-Path $PixInsightRoot 'bin\PixInsight.exe'
$bin = Join-Path $PixInsightRoot 'bin'

# ---------------------------------------------------------------------------
Say "=== 1. 体检 ==="
# ---------------------------------------------------------------------------
if (-not (Test-Path $exe)) { Die "找不到 PixInsight: $exe  (用 -PixInsightRoot 指定)" }
$ver = (Get-Item $exe).VersionInfo.ProductVersion
Say "PixInsight   $ver"
if ($ver -notmatch '^1\.9\.') {
    Warn "本模块针对 1.9.x 构建。版本差异可能导致 ABI 不匹配。"
}

if (-not (Test-Path $DllPath)) { Die "找不到 DLL: $DllPath" }
$dll = Get-Item $DllPath
Say "DLL          $($dll.Name)  $([math]::Round($dll.Length/1KB,1)) KB"

# 依赖是否齐备 —— 这些 PI 自己就带, 但在别的机器上值得确认
$need = @('onnxruntime.dll','MSVCP140.dll','VCRUNTIME140.dll','VCRUNTIME140_1.dll')
foreach ($n in $need) {
    if (Test-Path (Join-Path $bin $n)) { Say "  依赖 OK    $n" }
    else { Warn "  依赖缺失  $n   —— 本模块需要它; 请确认这台机器的 PI 安装完整" }
}

# 已存在的模块数量, 说明 bin 的确是模块目录
$existing = (Get-ChildItem $bin -File -Filter '*-pxm.dll' -ErrorAction SilentlyContinue).Count
Say "bin 里已有 $existing 个原生模块"

# 签名身份是否就绪
if ($Sign) {
    if (-not $XsskPath)      { Die "-Sign 需要 -XsskPath" }
    if (-not (Test-Path $XsskPath)) { Die "找不到密钥文件: $XsskPath" }
    Say "签名密钥     $XsskPath"
    Warn "提醒: 请先在 PI 里 Edit → Local Signing Identity… 选中该文件、输入密码、"
    Warn "      勾选 'Make the local signing identity persistent', 否则加载会被拒。"
} else {
    Say "签名         未请求(加 -Sign 才做)"
}

# ---------------------------------------------------------------------------
Say "=== 2. 计划 ==="
# ---------------------------------------------------------------------------
$targetDll  = Join-Path $bin $dll.Name
$targetXsgn = "$targetDll.xsgn" -replace '\.dll\.xsgn$', '.xsgn'
Say "  目标 DLL    $targetDll"
Say "  目标签名    $targetXsgn"
foreach ($t in @($targetDll, $targetXsgn)) {
    if (Test-Path $t) { Warn "  已存在, 将被备份: $t" }
}
if ($XsskPath) { Say "  签名命令    PixInsight.exe --sign-module-file=`"$DllPath`" --xssk-file=`"$XsskPath`" --xssk-password=***" }

if (-not $Sign -and -not $Install -and -not $Register) {
    Say ""
    Say "以上只是体检。要执行请加 -Sign / -Install / -Register。"
    exit 0
}

# ---------------------------------------------------------------------------
if ($Sign) {
    Say "=== 3. 签名 ==="

    if (Get-Process -Name PixInsight -ErrorAction SilentlyContinue) {
        Die ("PixInsight 正在运行。请先完全退出 PI 再签名 ——`n" +
             "  PI 已在运行时, 再启动它只会把签名请求转交给已运行的实例并立即返回,`n" +
             "  于是签名是异步完成的, 脚本无法判断何时结束。")
    }

    $xsgn = "$($dll.FullName).xsgn" -replace '\.dll\.xsgn$', '.xsgn'
    Remove-Item $xsgn -ErrorAction SilentlyContinue

    # 注意: 签名是设备相关的(身份 ID 派生自 PI 许可证), 所以必须在目标机器上做。
    #
    # PixInsight.exe 是 GUI 程序: PowerShell 不会等它结束, 也不会设置 $LASTEXITCODE。
    # 所以这里既不看退出码, 也不假设它同步完成 —— 而是轮询等签名文件出现。
    & $exe "--sign-module-file=$($dll.FullName)" "--xssk-file=$XsskPath" "--xssk-password=$XsskPassword"

    $deadline = 90
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path $xsgn) -and $sw.Elapsed.TotalSeconds -lt $deadline) {
        Start-Sleep -Milliseconds 300
    }
    # 签名完成后 PI 通常还留着进程, 收掉它
    Get-Process -Name PixInsight -ErrorAction SilentlyContinue |
        ForEach-Object { try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch { } }

    if (-not (Test-Path $xsgn)) {
        Die ("等了 $deadline 秒仍未生成 $xsgn`n" +
             "  最常见原因: Edit → Local Signing Identity… 里没有注册这个 .xssk,`n" +
             "  或密码不对。请先在 PI 里完成注册(勾选 Make the local signing identity persistent)。")
    }
    Say "已生成 $xsgn  ($((Get-Item $xsgn).Length) 字节)"
}

# ---------------------------------------------------------------------------
if ($Install) {
    Say "=== 4. 安装 ==="
    if (-not (Test-Path $bin)) { Die "找不到 $bin" }

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    foreach ($pair in @(@($dll.FullName, $targetDll), @("$($dll.FullName).xsgn", $targetXsgn))) {
        $src, $dst = $pair
        $src = $src -replace '\.dll\.xsgn$', '.xsgn'
        if (-not (Test-Path $src)) { Warn "跳过(无此文件): $src"; continue }
        if ((Test-Path $dst) -and -not $Force) {
            Die "$dst 已存在。若确认要覆盖请加 -Force`n  (它可能是 PI 自带模块, 覆盖会让那个模块失效)"
        }
        if (Test-Path $dst) {
            $bak = "$dst.bak-$stamp"
            Copy-Item $dst $bak -Force
            Say "已备份 -> $(Split-Path $bak -Leaf)"
        }
        Copy-Item $src $dst -Force
        Say "已安装   $(Split-Path $dst -Leaf)"
    }
}

# ---------------------------------------------------------------------------
if ($Register) {
    Say "=== 5. 注册到 PixInsight 模块表 ==="

    # PI 1.9 不再扫描 bin\ 下的新 DLL, 而是照 Settings 里的一份清单加载。
    # 清单在 %APPDATA%\Pleiades\core-001-pxi.settings 的 <i k="Modules"> 节点下。
    # 注意文件里有两个同名节点(一个是文件对话框的历史目录), 要挑含 *-pxm.dll 的那个。
    $settings = Join-Path $env:APPDATA 'Pleiades\core-001-pxi.settings'
    if (-not (Test-Path $settings)) {
        Die "找不到 PI 的设置文件: $settings`n  请先启动一次 PixInsight。"
    }

    $entry = "@pxi_bin_dir/$($dll.Name)"
    $text  = [System.IO.File]::ReadAllText($settings, [System.Text.Encoding]::UTF8)

    if ($text.Contains($entry)) {
        Say "已在模块表中, 无需改动"
    } else {
        $blocks = [regex]::Matches($text, '<i k="Modules">')
        $target = $null
        foreach ($m in $blocks) {
            $end = $text.IndexOf('</i>', $m.Index)
            if ($end -lt 0) { continue }
            $blk = $text.Substring($m.Index, $end - $m.Index)
            if ([regex]::Matches($blk, '-pxm\.dll').Count -gt 10) { $target = @($m.Index, $end, $blk); break }
        }
        if (-not $target) { Die "在设置文件里找不到模块清单节点" }

        $blk = $target[2]
        $nums = [regex]::Matches($blk, '<v k="(0{6,}\d+)"') | ForEach-Object { [int]$_.Groups[1].Value }
        $next = ($nums | Measure-Object -Maximum).Maximum + 1
        $line = '<v k="{0:D8}" t="s">{1}</v>' -f $next, $entry

        $bak = "$settings.bak-fkastro"
        Copy-Item $settings $bak -Force
        Say "已备份设置 -> $(Split-Path $bak -Leaf)"

        $newText = $text.Insert($target[1], $line)
        try { [xml]$null = $newText } catch { Copy-Item $bak $settings -Force; Die "改完 XML 不合法, 已还原: $_" }

        $utf8 = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($settings, $newText, $utf8)
        Say "已登记 $entry  (编号 $('{0:D8}' -f $next))"
    }
}

Say ""
Say "=== 完成 ==="
Say "下一步: 启动 PixInsight。"
Say "  菜单里应出现  Process → Preprocessing → FKAstroInference"
Say "  第一次使用请在最上面的 Model path 里指到 .onnx 模型文件。"
Say ""
if ($Install -and -not $Register) {
    Warn "提醒: 只做了安装、没做注册。PI 1.9 不会自动发现 bin 里的新模块,"
    Warn "      菜单里可能看不到 FKAstroInference。下次请加 -Register。"
}
