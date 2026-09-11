const path = require('node:path');
const { spawn } = require('node:child_process');

// 有些环境会设置 ELECTRON_RUN_AS_NODE=1，导致 electron 退化为 node 执行。
// 桌面端必须显式清掉这个变量。
delete process.env.ELECTRON_RUN_AS_NODE;

const electronBin = require('electron'); // 在非 electron 进程中，这里会返回 electron 可执行文件路径
const cwd = path.resolve(__dirname, '..');

const child = spawn(electronBin, ['.'], {
  cwd,
  env: process.env,
  stdio: 'inherit',
});

child.on('exit', (code) => {
  process.exit(typeof code === 'number' ? code : 1);
});
