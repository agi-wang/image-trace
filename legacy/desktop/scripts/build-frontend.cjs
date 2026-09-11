const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '..', '..');
const uiDir = path.join(repoRoot, 'ui');
const outDir = path.join(repoRoot, 'desktop', 'renderer');
const distDir = path.join(uiDir, 'dist');
const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';

function run(cmd, args, cwd, env) {
  const r = spawnSync(cmd, args, { cwd, stdio: 'inherit', env });
  if (r.error) throw r.error;
  if (typeof r.status === 'number' && r.status !== 0) {
    process.exit(r.status);
  }
}

// 1) 构建前端
const apiBase = process.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const npmExecPath = process.env.npm_execpath;
if (npmExecPath) {
  // 当脚本由 npm run dist 调用时，npm_execpath 指向 npm-cli.js，跨平台最稳
  run(process.execPath, [npmExecPath, 'run', 'build'], uiDir, {
    ...process.env,
    VITE_API_BASE_URL: apiBase,
  });
} else {
  run(npmCmd, ['run', 'build'], uiDir, {
    ...process.env,
    VITE_API_BASE_URL: apiBase,
  });
}

// 2) 清空旧 renderer 并复制 dist
fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(outDir, { recursive: true });

if (!fs.existsSync(distDir)) {
  throw new Error(`未找到前端构建产物目录：${distDir}（请检查 ui 的构建是否成功）`);
}

// 兼容 Windows：不要依赖 shell 的 cp 命令
for (const entry of fs.readdirSync(distDir)) {
  const src = path.join(distDir, entry);
  const dst = path.join(outDir, entry);
  fs.cpSync(src, dst, { recursive: true, force: true });
}

console.log(`[build-frontend] renderer updated from ui/dist (VITE_API_BASE_URL=${apiBase})`);
