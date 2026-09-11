# Image Trace Need Fix

模型: Codex (GPT-5)

审查时间: 2026-03-24
最后更新: 2026-03-27

本文件作为统一审查记录使用，后续问题与修复状态都合并维护在这里。

## P1

### 1. 前端 lint 未通过，当前仓库不满足基本质量门槛

- 现象:
  - `npm run lint` 失败
  - `ui/src/components/BackendGate.tsx:26`
  - `ui/src/components/BackendGate.tsx:69`
  - `ui/src/components/BackendGate.tsx:88`
  - `ui/src/pages/DuplicateReport.tsx:68`
  - `ui/src/pages/DuplicateReport.tsx:75`
- 影响:
  - 前端质量门禁失效，CI 或发版前检查无法通过
  - `any` 掩盖 Electron 注入对象和异常对象的真实类型
  - `useEffect` 依赖缺失会让报表页逻辑更难维护，也容易引入陈旧闭包问题
- 修复建议:
  - 为 `window.imageTraceDesktop` 补显式类型声明，移除 `any`
  - `catch (e: any)` 改为 `catch (e: unknown)` 并做窄化
  - 将 `DuplicateReport` 的 `loadReport` 包装为 `useCallback`，补齐依赖数组

### 2. 文档上传后的抽取图片必须自动触发特征预计算

- 来源:
  - 合并自 2026-03-27 的 review finding
- 原始问题:
  - 文档上传分支只保存抽取图片，不触发 `_ensure_precompute`
  - 会导致常见流程“上传文档 -> 智能查重”停在 `features_pending`
  - 也与页面“上传时已预计算”的文案不一致
- 当前状态:
  - 已在当前工作区修复
  - 证据见 `backend_simplified/app/main.py:345-348`，抽取图片落库后会逐张调用 `_ensure_precompute(...)`
- 后续建议:
  - 保留并完善对应自动化测试，避免这个回归再次出现

## P2

### 3. 后端测试没有被仓库内机制证明可直接运行

- 现象:
  - 在当前环境执行 `pytest backend_simplified/tests/test_api.py backend_simplified/tests/test_smart_compare.py`，导入 `tests/conftest.py` 时即失败
  - 错误: `ModuleNotFoundError: No module named 'sqlmodel'`
- 影响:
  - 仓库当前缺少“测试可直接跑通”的证据
  - 修复后的回归验证依赖人工补环境，审查成本高
- 修复建议:
  - 提供明确的后端测试启动方式，例如:
    - 一个可执行的 bootstrap 脚本
    - 或 CI 配置/README 中可直接复现的测试步骤
  - 至少保证 `backend_simplified/requirements.txt` 对应的测试环境能被一键安装并验证

### 4. 审查文档需要持续按状态维护

- 现象:
  - 同一轮审查里可能同时存在“已修复项”和“待修复项”
  - 如果只保留结论、不保留状态，很容易让后续验收误判
- 影响:
  - 容易误导后续审查、验收和发版判断
- 修复建议:
  - 将旧文件状态改为按日期维护
  - 把“已修复”和“当前未关闭项”拆开记录

## P3

### 5. 前端生产构建存在过大的单包

- 现象:
  - `npm run build` 通过，但输出警告:
  - `dist/assets/index-BRVtuDwV.js 539.65 kB`
- 影响:
  - 首屏加载、桌面端 renderer 冷启动和后续维护都会受影响
- 修复建议:
  - 对报表页、矩阵视图、可视化页面做动态加载
  - 在 Vite/Rollup 中显式拆分大模块

## 本次审查命令

```bash
cd /Users/kanshan/Documents/GitHub/image-trace/backend_simplified
pytest backend_simplified/tests/test_api.py backend_simplified/tests/test_smart_compare.py

cd /Users/kanshan/Documents/GitHub/image-trace/ui
npm run build
npm run lint
```
