const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

function repoRoot() {
  return path.resolve(__dirname, "..", "..");
}

function run(cmd, args, cwd) {
  const r = spawnSync(cmd, args, { cwd, stdio: "inherit" });
  if (r.error) throw r.error;
  if (typeof r.status === "number" && r.status !== 0) {
    process.exit(r.status);
  }
}

function detectPython() {
  if (process.env.PYTHON) {
    const r = spawnSync(process.env.PYTHON, ["--version"], { stdio: "inherit" });
    if (r.status === 0) return process.env.PYTHON;
    throw new Error(`PYTHON 指定的解释器不可用：${process.env.PYTHON}`);
  }

  const root = repoRoot();
  const venvCandidates =
    process.platform === "win32"
      ? [path.join(root, ".venv", "Scripts", "python.exe")]
      : [path.join(root, ".venv", "bin", "python")];
  for (const p of venvCandidates) {
    if (fs.existsSync(p)) {
      const r = spawnSync(p, ["--version"], { stdio: "inherit" });
      if (r.status === 0) return p;
    }
  }

  for (const c of ["python", "python3"]) {
    const r = spawnSync(c, ["--version"], { stdio: "ignore" });
    if (r.status === 0) return c;
  }
  throw new Error("未找到 python/python3。请在构建机安装 Python 3.11+（仅构建时需要，用户不需要）。");
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

const root = repoRoot();
const backendDir = path.join(root, "backend_simplified");
const entry = path.join(backendDir, "run_server.py");

if (!fs.existsSync(entry)) {
  throw new Error(`缺少入口文件：${entry}`);
}

const outDir = path.join(root, "desktop", "backend_bin");
ensureDir(outDir);

const py = detectPython();
console.log("[build-backend-nuitka] using", py);

// 预装 nuitka（如果未安装会尝试装一遍）
const nuitkaCheck = spawnSync(py, ["-m", "nuitka", "--version"], { stdio: "inherit" });
if (typeof nuitkaCheck.status === "number" && nuitkaCheck.status !== 0) {
  console.log("[build-backend-nuitka] installing nuitka...");
  run(py, ["-m", "pip", "install", "--upgrade", "pip"], backendDir);
  run(py, ["-m", "pip", "install", "nuitka"], backendDir);
}

const baseName = "image-trace-backend-nuitka";
const exeName = process.platform === "win32" ? `${baseName}.exe` : baseName;
// Nuitka --standalone 输出目录名为 run_server.dist，
// 构建完后重命名为 image-trace-backend-nuitka 以保持一致
const nuitkaDistDir = path.join(outDir, "run_server.dist");
const finalDir = path.join(outDir, baseName);

const toPosix = (p) => p.split(path.sep).join("/");

function existingTxtGlobs() {
  const appDir = path.join(backendDir, "app");
  if (!fs.existsSync(appDir)) return [];
  const hasTxt = fs.readdirSync(appDir).some((f) => f.endsWith(".txt"));
  return hasTxt ? [`--include-data-files=${toPosix(path.join(appDir, "*.txt"))}=app/`] : [];
}

const args = [
  "-m",
  "nuitka",
  entry,
  // 不使用 --onefile，只用 --standalone（目录模式，构建更快）
  "--standalone",
  "--assume-yes-for-downloads",  // Auto-download dependency walker on Windows CI
  "--follow-imports",
  `--output-dir=${toPosix(outDir)}`,
  `--output-filename=${exeName}`,
  "--include-package=cv2",
  "--include-package=fitz",
  "--include-package=imagehash",
  "--include-package=sqlmodel",
  "--include-package=uvicorn",
  "--include-package=fastapi",
  "--include-package=skimage",
  "--include-package=skimage.metrics",
  "--include-package=pillow_heif",
  "--include-package=pillow_avif",
  `--include-data-files=${toPosix(path.join(backendDir, "app", "*.py"))}=app/`,
  "--nofollow-import-to=tests",
  ...existingTxtGlobs(),
];

run(py, args, backendDir);

// Nuitka --standalone 产出 run_server.dist/ 目录，重命名为 baseName
if (fs.existsSync(nuitkaDistDir)) {
  // 清理旧目录
  if (fs.existsSync(finalDir)) {
    fs.rmSync(finalDir, { recursive: true, force: true });
  }
  fs.renameSync(nuitkaDistDir, finalDir);
  console.log(`[build-backend-nuitka] renamed ${nuitkaDistDir} -> ${finalDir}`);
}

const builtPath = path.join(finalDir, exeName);
if (!fs.existsSync(builtPath)) {
  throw new Error(`后端可执行文件未生成：${builtPath}`);
}

console.log("[build-backend-nuitka] OK:", builtPath);
console.log("[build-backend-nuitka] 目录:", finalDir);
