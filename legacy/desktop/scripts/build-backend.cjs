const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

function repoRoot() {
  // 脚本位置：desktop/scripts/build-backend.cjs
  // repo root：desktop/..
  return path.resolve(__dirname, '..', '..');
}

function run(cmd, args, cwd) {
  const r = spawnSync(cmd, args, { cwd, stdio: 'inherit' });
  if (r.error) throw r.error;
  if (typeof r.status === 'number' && r.status !== 0) {
    process.exit(r.status);
  }
}

function detectPython() {
  // 允许显式指定 Python（推荐在 venv 下构建）：
  //   PYTHON=/path/to/python node scripts/build-backend.cjs
  if (process.env.PYTHON) {
    const r = spawnSync(process.env.PYTHON, ['--version'], { stdio: 'inherit' });
    if (r.status === 0) return process.env.PYTHON;
    throw new Error(`PYTHON 指定的解释器不可用：${process.env.PYTHON}`);
  }

  // 优先使用仓库根目录下的 .venv（避免误用 conda/system python）
  const root = repoRoot();
  const venvCandidates = process.platform === 'win32'
    ? [path.join(root, '.venv', 'Scripts', 'python.exe')]
    : [path.join(root, '.venv', 'bin', 'python')];
  for (const p of venvCandidates) {
    if (fs.existsSync(p)) {
      const r = spawnSync(p, ['--version'], { stdio: 'inherit' });
      if (r.status === 0) return p;
    }
  }

  for (const c of ['python', 'python3']) {
    const r = spawnSync(c, ['--version'], { stdio: 'ignore' });
    if (r.status === 0) return c;
  }
  throw new Error('未找到 python/python3。请在构建机安装 Python 3.11+（仅构建时需要，用户不需要）。');
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

const root = repoRoot();
const backendDir = path.join(root, 'backend_simplified');
const entry = path.join(backendDir, 'run_server.py');

if (!fs.existsSync(entry)) {
  throw new Error(`缺少入口文件：${entry}`);
}

const outDir = path.join(root, 'desktop', 'backend_bin');
const workDir = path.join(root, 'desktop', '.pyinstaller-build');
const specDir = path.join(root, 'desktop', '.pyinstaller-spec');
ensureDir(outDir);
ensureDir(workDir);
ensureDir(specDir);

const py = detectPython();
console.log('[build-backend] using', py);

// 说明：默认不自动安装依赖，避免污染全局 Python。
// 请在构建机先创建 venv 并安装：
//   pip install -r backend_simplified/requirements.txt
//   pip install pyinstaller
// 然后再运行本脚本。
run(py, ['-m', 'PyInstaller', '--version'], backendDir);

// 使用 PyInstaller 生成的后端二进制，名称明确标记编译方式
// --onedir 模式（不用 --onefile）：产出目录 backend_bin/image-trace-backend-pyinstaller/
const baseName = 'image-trace-backend-pyinstaller';
const exeName = process.platform === 'win32' ? `${baseName}.exe` : baseName;

run(py, [
  '-m', 'PyInstaller',
  '--noconfirm',
  '--clean',
  // 不使用 --onefile，默认 --onedir（目录模式，构建更快，启动更快）
  '--name', baseName,
  '--distpath', outDir,
  '--workpath', workDir,
  '--specpath', specDir,
  '--paths', backendDir,
  // 增加常见隐式导入，防止 PyInstaller 漏掉 uvicorn/fastapi/cv2 等依赖
  '--hidden-import', 'uvicorn.logging',
  '--hidden-import', 'uvicorn.loops',
  '--hidden-import', 'uvicorn.loops.auto',
  '--hidden-import', 'uvicorn.protocols',
  '--hidden-import', 'uvicorn.protocols.http',
  '--hidden-import', 'uvicorn.protocols.http.auto',
  '--hidden-import', 'uvicorn.lifespan',
  '--hidden-import', 'uvicorn.lifespan.on',
  '--hidden-import', 'engineio.async_drivers.asgi',
  '--collect-all', 'cv2',
  '--collect-all', 'fitz',
  '--collect-all', 'imagehash',
  '--collect-all', 'sqlmodel',
  '--hidden-import', 'pillow_heif',
  '--hidden-import', 'pillow_avif',
  '--hidden-import', 'rawpy',
  '--hidden-import', 'skimage',
  '--hidden-import', 'skimage.metrics',
  entry,
], backendDir);

// --onedir 产出：outDir/baseName/exeName
const builtPath = path.join(outDir, baseName, exeName);
if (!fs.existsSync(builtPath)) {
  throw new Error(`后端可执行文件未生成：${builtPath}`);
}

console.log('[build-backend] OK:', builtPath);
console.log('[build-backend] 目录:', path.join(outDir, baseName));
