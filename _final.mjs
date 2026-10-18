// final.mjs —— 重建附件并替换 Release 资产。
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";

const OWNER = "HuaiyuanYang";
const REPO = "FK-Astro-BXT-Windows";
const TAG = "v1.0.0";
const ZIP = process.argv[2];

const out = execSync("git credential fill", {
  input: "protocol=https\nhost=github.com\n\n",
  encoding: "utf8",
  stdio: ["pipe", "pipe", "ignore"],
});
const tok = out.split(/\r?\n/).find((l) => l.startsWith("password="))
  .slice("password=".length).trim();
const H = {
  Authorization: `Bearer ${tok}`,
  Accept: "application/vnd.github+json",
  "User-Agent": "fk-astro-publish",
  "X-GitHub-Api-Version": "2022-11-28",
};
const base = `https://api.github.com/repos/${OWNER}/${REPO}`;

const rel0 = await (await fetch(`${base}/releases/tags/${TAG}`, { headers: H })).json();
for (const a of rel0.assets || []) {
  const d = await fetch(`${base}/releases/assets/${a.id}`, { method: "DELETE", headers: H });
  console.log(`  删旧附件 ${a.name}: ${d.status}`);
}

const name = ZIP.split(/[\\/]/).pop();
const buf = readFileSync(ZIP);
const up = await fetch(
  `https://uploads.github.com/repos/${OWNER}/${REPO}/releases/${rel0.id}/assets?name=${encodeURIComponent(name)}`,
  {
    method: "POST",
    headers: { ...H, "Content-Type": "application/zip", "Content-Length": String(buf.length) },
    body: buf,
  });
const a = await up.json();
console.log("  上传:", up.status, a.name, `${(a.size / 1024).toFixed(0)} KB`);
console.log("  ", a.browser_download_url);
