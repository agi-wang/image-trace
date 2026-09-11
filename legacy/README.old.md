# Image Trace

轻量的图像比对与文档图片提取工具，支持本地桌面体验和简单的 API 调用。

## 你可以用它做什么
- 从 PDF / DOCX / PPTX / 图片中提取图片素材
- 去重与相似分组，找出重复或近似图片
- 桌面端直接操作，也可以通过 API 集成到现有流程

## 快速开始

### 下载已编译版本（最快）
1) 前往 Releases：<https://github.com/zots0127/image-trace/releases/latest>  
2) 选择你的平台包并解压/安装：  
   - macOS（Apple/Intel）：`ImageTrace-mac-*.dmg` 或 `.zip`  
   - Windows：`ImageTrace-win-*.exe`  
   - Linux：`Image.Trace-*.AppImage`  
   - 后端二进制同时内置 **PyInstaller** 与 **Nuitka** 版本，应用会优先使用 Nuitka（如存在）。  
3) 首次运行若遇到安全拦截，按下方“常见问题（macOS）”处理 Gatekeeper 或在 Windows 选择“仍要运行”。  
4) 启动后按界面提示创建项目、上传文件并开始比对。

### 桌面应用（推荐，内置后端）
1) 安装 Node.js 18+ 与 npm。  
2) `cd desktop && npm install`  
3) `npm run dev` 启动 Electron。开发态会优先复用本机 `127.0.0.1:8000`，若未启动则自动拉起内置后端。  
4) 在界面中：创建项目 → 上传文件/图片 → 点击“开始比对”。

### 仅跑后端 API
1) `cd backend_simplified`  
2) 准备 Python 3.10+，创建虚拟环境：`python -m venv venv && source venv/bin/activate`（Windows 使用 `venv\Scripts\activate`）  
3) 安装依赖：`pip install -r requirements.txt`  
4) 运行：`uvicorn app.main:app --host 127.0.0.1 --port 8000`  
5) 打开 `http://127.0.0.1:8000/docs` 试调接口。

### Web 前端（可选，指向本地后端）
1) `cd ui && npm install`  
2) 如需自定义后端地址，设置环境变量 `VITE_API_BASE`（默认 `http://127.0.0.1:8000`）。  
3) `npm run dev`，浏览器访问提示的本地端口。

## 使用流程
1) 创建项目（桌面端界面，或 `POST /projects`）。  
2) 上传文档/图片（桌面端拖拽，或 `POST /upload` 携带 `project_id` 与文件）。  
3) 执行比对（桌面端“开始比对”，或 `POST /compare/{project_id}`）。  
4) 查看分组结果（桌面端展示，或 `GET /results/{project_id}`）。  
5) 需要时删除或重新上传文件，重复步骤 2-4。

## 目录速览
- `backend_simplified/`：FastAPI 后端（本地数据存储在 `data/`）。
- `desktop/`：Electron 桌面端，包含内置后端的启动与打包脚本。
- `ui/`：可选的 Web 前端（Vite + React）。

## 常见问题（macOS）
- Gatekeeper 拦截或提示“已损坏”：在 Finder 右键选择“打开”并确认，或在“系统设置 -> 隐私与安全性”允许；仍有隔离时可执行 `xattr -r -d com.apple.quarantine "<App 路径>"`。  
- 无法启动：确认无端口占用；必要时为可执行文件添加权限 `chmod +x "<App 路径>/Contents/MacOS/Image Trace"`。  
- 后端未响应：确认 127.0.0.1:8000 可访问，或在桌面端菜单/设置中重新启动后端。

## 计划
- 多语言界面与文档支持（进行中，欢迎需求反馈）。  
- 更简化的安装包与自动更新。  
- 更多格式支持与批量导出能力。

## 许可证
MIT License