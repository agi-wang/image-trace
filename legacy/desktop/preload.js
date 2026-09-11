const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('imageTraceDesktop', {
  startBackend: () => ipcRenderer.invoke('backend:start'),
  stopBackend: () => ipcRenderer.invoke('backend:stop'),
  getBackendInfo: () => ipcRenderer.invoke('backend:info'),
  openDataDir: () => ipcRenderer.invoke('backend:openDataDir'),
  onBackendLog: (callback) => {
    const handler = (_event, line) => callback(line);
    ipcRenderer.on('backend:log', handler);
    // Return cleanup function
    return () => ipcRenderer.removeListener('backend:log', handler);
  },
});
