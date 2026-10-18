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
#   # 2) 确认无误后签名 + 安装
#   powershell -ExecutionPolicy Bypass -File install-to-pi.ps1 -DllPath .\FKAstro-pxm.dll `
#       -XsskPath D:\keys\me.xssk -XsskPassword 'xxx' -Sign -Install
#
#   # 3) 重启 PixInsight

param(
    [Parameter(Mandatory=$true)][string]$DllPath,

    [string]$PixInsightRoot = 'D:\PixInsight',
    [string]$XsskPath,
    [string]$XsskPassword,

    [switch]$Sign,
    [switch]$Install,
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

if (-not $Sign -and -not $Install) {
    Say ""
    Say "以上只是体检。要执行请加 -Sign 和/或 -Install。"
    exit 0
}

# ---------------------------------------------------------------------------
if ($Sign) {
    Say "=== 3. 签名 ==="
    $xsgn = "$($dll.FullName).xsgn" -replace '\.dll\.xsgn$', '.xsgn'
    Remove-Item $xsgn -ErrorAction SilentlyContinue
    # 注意: 签名是设备相关的(身份 ID 派生自 PI 许可证), 所以必须在目标机器上做。
    & $exe "--sign-module-file=$($dll.FullName)" "--xssk-file=$XsskPath" "--xssk-password=$XsskPassword"
    if ($LASTEXITCODE -ne 0) { Die "签名失败 (exit $LASTEXITCODE)" }
    if (-not (Test-Path $xsgn)) { Die "签名命令返回 0 但没有产出 $xsgn" }
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

Say ""
Say "=== 完成 ==="
Say "下一步: 重启 PixInsight。启动时它会扫描 bin\\ 下的 *-pxm.dll。"
Say "加载后模块名为 FKAstro, 进程参数里用 modelPath 指定 .onnx 模型。"
Say ""
Warn "尚未验证过:PI 1.9.5 是否会弹'发现新模块'确认框, 或是否需要手动注册。"
Warn "若加载失败, 先看 控制台 与 控制台 → 消息; 最常见原因是签名身份没设成持久化。"
