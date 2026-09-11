const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const fsp = fs.promises;
const net = require('net');
const http = require('http');
const { spawn, spawnSync } = require('child_process');

/** Broadcast a log line to all renderer windows */
function sendLog(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  for (const win of BrowserWindow.getAllWindows()) {
    try {
      win.webContents.send('backend:log', line);
    } catch {
      // window may be destroyed
    }
  }
}

function isDev() {
  return !app.isPackaged;
}

function getRuntimeDir() {
  return path.join(app.getPath('userData'), 'runtime');
}

function getRuntimeDataDir() {
  return path.join(getRuntimeDir(), 'data');
}

function getRuntimeLogPath() {
  return path.join(getRuntimeDir(), 'backend.log');
}

function getBackendBinaryDir() {
  // 打包后：extraResources -> process.resourcesPath/backend_bin/<exe>
  return app.isPackaged
    ? path.join(process.resourcesPath, 'backend_bin')
    : path.join(__dirname, 'backend_bin');
}

function resolveBackendBinary() {
  const suffix = process.platform === 'win32' ? '.exe' : '';
  const dir = getBackendBinaryDir();
  const variants = [
    // 目录模式（--onedir / --standalone）：backend_bin/<flavor>/<exe>
    { flavor: 'nuitka', filename: `image-trace-backend-nuitka${suffix}`, subdir: 'image-trace-backend-nuitka' },
    { flavor: 'pyinstaller', filename: `image-trace-backend-pyinstaller${suffix}`, subdir: 'image-trace-backend-pyinstaller' },
    // 兼容旧的单文件模式（--onefile 产出直接在 backend_bin/ 下）
    { flavor: 'nuitka-legacy', filename: `image-trace-backend-nuitka${suffix}`, subdir: null },
    { flavor: 'pyinstaller-legacy', filename: `image-trace-backend-pyinstaller${suffix}`, subdir: null },
    { flavor: 'legacy', filename: `image-trace-backend${suffix}`, subdir: null },
  ];

  const preferred = (process.env.IMAGE_TRACE_BACKEND_FLAVOR || '').toLowerCase();
  let ordered = variants;
  if (preferred) {
    const idx = variants.findIndex((v) => v.flavor.startsWith(preferred));
    if (idx >= 0) {
      ordered = [variants[idx], ...variants.filter((_, i) => i !== idx)];
    }
  }

  const tried = [];
  for (const v of ordered) {
    const full = v.subdir
      ? path.join(dir, v.subdir, v.filename)
      : path.join(dir, v.filename);
    tried.push(full);
    if (fs.existsSync(full)) {
      return { path: full, flavor: v.flavor.replace('-legacy', ''), tried, missing: false };
    }
  }

  // 未找到，返回第一个候选路径用于提示
  return {
    path: path.join(dir, ordered[0].subdir || '', ordered[0].filename),
    flavor: ordered[0].flavor,
    tried,
    missing: true,
  };
}

async function ensureRuntimeDirs() {
  await fsp.mkdir(getRuntimeDir(), { recursive: true });
  await fsp.mkdir(getRuntimeDataDir(), { recursive: true });
  await fsp.mkdir(path.join(getRuntimeDataDir(), 'uploads'), { recursive: true });
  await fsp.mkdir(path.join(getRuntimeDataDir(), 'extracted'), { recursive: true });
}

async function getFreePort(prefer = 8000) {
  const canUse = (port) =>
    new Promise((resolve) => {
      const s = net.createServer();
      s.once('error', () => resolve(false));
      s.once('listening', () => s.close(() => resolve(true)));
      s.listen(port, '127.0.0.1');
    });

  if (await canUse(prefer)) return prefer;

  return await new Promise((resolve, reject) => {
    const s = net.createServer();
    s.once('error', reject);
    s.listen(0, '127.0.0.1', () => {
      const addr = s.address();
      const port = typeof addr === 'object' && addr ? addr.port : prefer;
      s.close(() => resolve(port));
    });
  });
}

function httpGetJson(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let body = '';
      res.on('data', (d) => (body += d.toString('utf-8')));
      res.on('end', () => {
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`HTTP ${res.statusCode}: ${body}`));
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(1500, () => req.destroy(new Error('timeout')));
  });
}

async function waitForHealth(baseUrl, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  let lastErr = null;
  while (Date.now() < deadline) {
    try {
      await httpGetJson(`${baseUrl}/health`);
      return;
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 300));
    }
  }
  throw lastErr || new Error('health check timeout');
}

let backendProc = null;
let backendBaseUrl = null;
let backendPort = null;
let backendManaged = false;

async function startBackendInternal() {
  if (backendProc && backendBaseUrl) {
    sendLog('后端已在运行，检查健康...');
    await waitForHealth(backendBaseUrl, 3000).catch(() => { });
    return { baseUrl: backendBaseUrl, port: backendPort };
  }

  sendLog('初始化运行时目录...');
  await ensureRuntimeDirs();

  // dev 场景：如果本机已经有后端在 8000（例如你用 docker 起了），直接复用
  if (!app.isPackaged) {
    const tryUrl = 'http://127.0.0.1:8000';
    sendLog('开发模式：检查 localhost:8000 是否有现有后端...');
    try {
      await waitForHealth(tryUrl, 800);
      sendLog('✅ 发现已运行的后端服务（端口 8000），直接复用');
      backendProc = null;
      backendManaged = false;
      backendBaseUrl = tryUrl;
      backendPort = 8000;
      return { baseUrl: backendBaseUrl, port: backendPort };
    } catch {
      sendLog('未发现现有后端，将启动内置后端...');
    }
  }

  sendLog('正在获取可用端口...');
  const port = await getFreePort(8000);
  const baseUrl = `http://127.0.0.1:${port}`;
  sendLog(`端口已分配: ${port}`);

  const logStream = fs.createWriteStream(getRuntimeLogPath(), { flags: 'a' });
  sendLog(`日志文件: ${getRuntimeLogPath()}`);

  const env = {
    ...process.env,
    IMAGE_TRACE_HOST: '127.0.0.1',
    IMAGE_TRACE_PORT: String(port),
    IMAGE_TRACE_DATA_DIR: getRuntimeDataDir(),
    IMAGE_TRACE_LOG_LEVEL: 'info',
  };

  sendLog('正在查找后端可执行文件...');
  const resolvedBinary = resolveBackendBinary();
  const exePath = resolvedBinary.path;
  let child;

  if (!resolvedBinary.missing && fs.existsSync(exePath)) {
    sendLog(`✅ 找到后端: ${resolvedBinary.flavor} (${path.basename(exePath)})`);
    // mac/linux: 确保可执行权限（有些打包链路会丢失 chmod）
    if (process.platform !== 'win32') {
      try {
        fs.chmodSync(exePath, 0o755);
        sendLog('已设置可执行权限 (chmod 755)');
      } catch {
        // ignore
      }
    }
    sendLog(`正在启动后端进程... (cwd: ${getRuntimeDir()})`);
    child = spawn(exePath, [], {
      cwd: getRuntimeDir(),
      env: {
        ...env,
        IMAGE_TRACE_BACKEND_FLAVOR: resolvedBinary.flavor,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    backendManaged = true;
    sendLog(`后端进程已启动 (PID: ${child.pid})`);
  } else if (isDev()) {
    sendLog('未找到编译后端，开发模式下尝试使用 python...');
    const pyCandidates = process.platform === 'win32' ? ['python'] : ['python3', 'python'];
    let py = null;
    for (const c of pyCandidates) {
      const r = spawnSync(c, ['--version'], { stdio: 'ignore' });
      if (r.status === 0) {
        py = c;
        break;
      }
    }
    if (!py) {
      const msg = '未找到内置后端可执行文件，且开发模式下也未检测到 python。请先运行 npm run build:backend。';
      sendLog(`❌ ${msg}`);
      throw new Error(msg);
    }

    sendLog(`使用 ${py} 启动开发后端...`);
    const backendDir = path.resolve(__dirname, '..', 'backend_simplified');
    child = spawn(py, ['run_server.py'], {
      cwd: backendDir,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    backendManaged = true;
    sendLog(`开发后端已启动 (PID: ${child.pid})`);
  } else {
    const tried = resolvedBinary.tried && resolvedBinary.tried.length
      ? resolvedBinary.tried.join(', ')
      : exePath;
    const msg = `内置后端可执行文件不存在，已尝试：${tried}`;
    sendLog(`❌ ${msg}`);
    throw new Error(msg);
  }

  // Stream backend stdout/stderr to both log file and renderer
  child.stdout.on('data', (d) => {
    logStream.write(d);
    const lines = d.toString('utf-8').split('\n').filter(Boolean);
    for (const line of lines) sendLog(`[stdout] ${line}`);
  });
  child.stderr.on('data', (d) => {
    logStream.write(d);
    const lines = d.toString('utf-8').split('\n').filter(Boolean);
    for (const line of lines) sendLog(`[stderr] ${line}`);
  });
  child.on('exit', (code, signal) => {
    sendLog(`⚠️ 后端进程退出 (code=${code}, signal=${signal})`);
    backendProc = null;
    backendBaseUrl = null;
    backendPort = null;
  });

  backendProc = child;
  backendBaseUrl = baseUrl;
  backendPort = port;

  sendLog(`正在等待后端 health check (${baseUrl}/health)...`);
  await waitForHealth(baseUrl, 20000);
  sendLog('✅ 后端服务就绪！');
  return { baseUrl, port };
}

async function stopBackendInternal() {
  if (!backendProc) return;
  if (!backendManaged) return;

  const p = backendProc;
  backendProc = null;
  backendBaseUrl = null;
  backendPort = null;
  backendManaged = false;

  try {
    p.kill('SIGTERM');
  } catch {
    // ignore
  }

  await new Promise((r) => setTimeout(r, 800));
  if (!p.killed) {
    try {
      p.kill('SIGKILL');
    } catch {
      // ignore
    }
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 980,
    minHeight: 700,
    backgroundColor: '#0b1020',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // 任何新窗口都交给系统浏览器打开
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

ipcMain.handle('backend:start', async () => {
  try {
    const { baseUrl, port } = await startBackendInternal();
    return { ok: true, baseUrl, port };
  } catch (e) {
    return { ok: false, error: e && e.message ? e.message : String(e) };
  }
});

ipcMain.handle('backend:stop', async () => {
  try {
    if (!backendManaged) {
      return { ok: false, error: '当前后端不是由桌面端启动（无法停止）。' };
    }
    await stopBackendInternal();
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e && e.message ? e.message : String(e) };
  }
});

ipcMain.handle('backend:info', async () => {
  return {
    baseUrl: backendBaseUrl,
    port: backendPort,
    logPath: getRuntimeLogPath(),
    dataDir: getRuntimeDataDir(),
    managed: backendManaged,
  };
});

ipcMain.handle('backend:openDataDir', async () => {
  await ensureRuntimeDirs();
  await shell.openPath(getRuntimeDataDir());
  return { ok: true };
});

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackendInternal().catch(() => { });
});
