// fetch_ort_headers.mjs —— 取 ONNX Runtime 的 C API 头文件(v1.25.0,与 PI 自带 ORT 1.25 对齐)
//
// 为什么用 Node 而不是 curl / Invoke-WebRequest:
//   本机沙箱下 Schannel 无法获取凭据, curl 与 .NET 对**所有** https 站点均报
//   SEC_E_NO_CREDENTIALS (0x8009030e)。Node 自带 OpenSSL, 不受影响;
//   且环境里已有 NODE_USE_ENV_PROXY=1, fetch 会自动走系统代理。
//
// 用法: node src/third_party/fetch_ort_headers.mjs

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BASE =
  "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.25.0/include/onnxruntime/core/session";
// 输出到本脚本所在目录下的 onnxruntime/
const OUT = join(dirname(fileURLToPath(import.meta.url)), "onnxruntime");

const FILES = [
  "onnxruntime_c_api.h",
  "onnxruntime_ep_c_api.h",
  "onnxruntime_float16.h",
  "onnxruntime_session_options_config_keys.h",
  "onnxruntime_run_options_config_keys.h",
];

mkdirSync(OUT, { recursive: true });

let ok = 0;
for (const f of FILES) {
  const url = `${BASE}/${f}`;
  try {
    const r = await fetch(url, { redirect: "follow" });
    if (!r.ok) {
      console.log(`${f.padEnd(48)} HTTP ${r.status}`);
      continue;
    }
    const t = await r.text();
    writeFileSync(`${OUT}/${f}`, t, "utf8");
    console.log(`${f.padEnd(48)} ${String(t.length).padStart(9)} chars`);
    ok++;
  } catch (e) {
    console.log(`${f.padEnd(48)} FAILED: ${e.message}`);
  }
}
console.log(`\n${ok}/${FILES.length} 已下载 -> ${OUT}`);
