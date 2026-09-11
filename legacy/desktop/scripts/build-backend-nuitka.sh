#!/usr/bin/env bash
set -euo pipefail

# 基于 Nuitka 的后端构建脚本（并存 PyInstaller）
# 用法：在仓库根目录运行
#   chmod +x desktop/scripts/build-backend-nuitka.sh
#   ./desktop/scripts/build-backend-nuitka.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT}/backend_simplified"
ENTRY="${BACKEND_DIR}/run_server.py"
OUT_DIR="${ROOT}/desktop/backend_bin"
VENV_PY="${ROOT}/.venv/bin/python"

if [[ ! -f "${ENTRY}" ]]; then
  echo "[error] 找不到入口文件: ${ENTRY}" >&2
  exit 1
fi

PY_BIN="${PYTHON:-}"
if [[ -z "${PY_BIN}" ]]; then
  if [[ -x "${VENV_PY}" ]]; then
    PY_BIN="${VENV_PY}"
  else
    PY_BIN="$(command -v python3 || command -v python)"
  fi
fi

if [[ -z "${PY_BIN}" ]]; then
  echo "[error] 未找到可用的 Python 解释器" >&2
  exit 1
fi

echo "[info] 使用 Python: ${PY_BIN}"

# 确保 Nuitka 已安装（可预装以避免联网）
if ! "${PY_BIN}" -m nuitka --version >/dev/null 2>&1; then
  echo "[info] 未检测到 Nuitka，尝试安装..."
  "${PY_BIN}" -m pip install --upgrade pip
  "${PY_BIN}" -m pip install nuitka
fi

mkdir -p "${OUT_DIR}"

BASE_NAME="image-trace-backend-nuitka"
EXE_NAME="${BASE_NAME}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  EXE_NAME="${BASE_NAME}.exe"
fi

echo "[info] 开始 Nuitka 打包..."
"${PY_BIN}" -m nuitka "${ENTRY}" \
  --onefile \
  --standalone \
  --follow-imports \
  --output-dir="${OUT_DIR}" \
  --output-filename="${EXE_NAME}" \
  --include-package=cv2 \
  --include-package=fitz \
  --include-package=imagehash \
  --include-data-files="${BACKEND_DIR}/app/*.py=app/" \
  --include-data-files="${BACKEND_DIR}/app/*.txt=app/" \
  --nofollow-import-to=tests

BIN_PATH="${OUT_DIR}/${EXE_NAME}"
if [[ ! -f "${BIN_PATH}" ]]; then
  echo "[error] 打包失败，未生成可执行文件: ${BIN_PATH}" >&2
  exit 1
fi

echo "[info] 打包完成: ${BIN_PATH}"
