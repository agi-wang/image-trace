#!/usr/bin/env bash
set -euo pipefail

# 本地一键编译并启动：内置后端 + Electron（你提供的 UI）
# 使用方式：在仓库根目录运行
#   chmod +x local-e2e.sh
#   ./local-e2e.sh
#
# 停止：按 Ctrl+C，会自动清理后台进程。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"
PY="${VENV}/bin/python"

info() { echo -e "\033[1;34m[info]\033[0m $*"; }
err()  { echo -e "\033[1;31m[error]\033[0m $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "缺少命令：$1"; exit 1; }
}

need_cmd python3
need_cmd node
need_cmd npm

info "创建/使用 venv: ${VENV}"
if [[ ! -x "${PY}" ]]; then
  python3 -m venv "${VENV}"
fi

info "安装后端依赖 + PyInstaller"
"${PY}" -m pip install --upgrade pip >/dev/null
"${PY}" -m pip install -r "${ROOT}/backend_simplified/requirements.txt" pyinstaller

info "构建后端可执行文件"
PYTHON="${PY}" node "${ROOT}/desktop/scripts/build-backend.cjs"

info "安装前端依赖并构建（UI -> renderer）"
npm install --prefix "${ROOT}/ui"
node "${ROOT}/desktop/scripts/build-frontend.cjs"

info "安装 Electron 依赖"
npm install --prefix "${ROOT}/desktop"

BACKEND_PID=""
ELECTRON_PID=""
cleanup() {
  info "清理后台进程..."
  [[ -n "${ELECTRON_PID}" ]] && kill "${ELECTRON_PID}" 2>/dev/null || true
  [[ -n "${BACKEND_PID}" ]] && kill "${BACKEND_PID}" 2>/dev/null || true
  wait ${ELECTRON_PID:-} ${BACKEND_PID:-} 2>/dev/null || true
}
trap "cleanup; exit 0" INT TERM
trap cleanup EXIT

info "启动内置后端 (127.0.0.1:8000)"
BACKEND_EXE="${ROOT}/desktop/backend_bin/image-trace-backend-pyinstaller"
if [[ -x "${ROOT}/desktop/backend_bin/image-trace-backend-nuitka" ]]; then
  BACKEND_EXE="${ROOT}/desktop/backend_bin/image-trace-backend-nuitka"
fi
(
  cd "${ROOT}"
  "${BACKEND_EXE}"
) &
BACKEND_PID=$!
sleep 2

info "启动 Electron（加载打包后的 UI）"
(
  cd "${ROOT}/desktop"
  npm run dev
) &
ELECTRON_PID=$!

info "已启动完毕："
info "  后端健康检查: http://127.0.0.1:8000/health"
info "  Electron 窗口请在前台查看；如需 UI dev server，请单独在 ui/ 运行 npm run dev。"
wait
