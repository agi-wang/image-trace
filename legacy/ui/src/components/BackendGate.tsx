import { useEffect, useRef, useState, ReactNode, useCallback } from "react";
import { Loader2, RefreshCw, Terminal } from "lucide-react";
import { checkHealth, setApiBaseUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";

interface BackendGateProps {
  children: ReactNode;
}

type Status = "starting" | "checking" | "ready" | "error";

const MAX_LOG_LINES = 200;

export function BackendGate({ children }: BackendGateProps) {
  const [status, setStatus] = useState<Status>("starting");
  const [showRetry, setShowRetry] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const retryDelayRef = useRef<ReturnType<typeof setTimeout>>();
  const logContainerRef = useRef<HTMLDivElement>(null);
  const startedRef = useRef(false);

  // Subscribe to backend log events from Electron main process
  useEffect(() => {
    const desktop = (window as any).imageTraceDesktop;
    if (desktop?.onBackendLog) {
      const cleanup = desktop.onBackendLog((line: string) => {
        setLogs((prev) => {
          const next = [...prev, line];
          return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next;
        });
      });
      return cleanup;
    }
  }, []);

  // Auto-scroll log container
  useEffect(() => {
    const el = logContainerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = undefined;
    }
  };

  // Health check polling
  const probe = useCallback(async (delayMs = 800) => {
    clearTimer();
    setStatus("checking");
    try {
      await checkHealth();
      setStatus("ready");
    } catch {
      setStatus("error");
      const nextDelay = Math.min(delayMs + 400, 4000);
      timerRef.current = setTimeout(() => probe(nextDelay), nextDelay);
    }
  }, []);

  // Start backend via Electron IPC, then poll health
  const startBackend = useCallback(async () => {
    const desktop = (window as any).imageTraceDesktop;

    if (desktop?.startBackend) {
      // Running in Electron — call IPC to start backend
      setStatus("starting");
      try {
        const result = await desktop.startBackend();
        if (result.ok && result.baseUrl) {
          // Backend started on a specific port — update API base URL
          setApiBaseUrl(result.baseUrl);
          setStatus("ready");
          return;
        } else if (result.error) {
          setErrorMsg(result.error);
          setStatus("error");
          // Still try polling in case the backend starts later
          probe();
          return;
        }
      } catch (e: any) {
        setErrorMsg(e?.message || "Failed to start backend");
      }
    }

    // Not in Electron (browser dev mode) or IPC failed — just poll health
    probe();
  }, [probe]);

  useEffect(() => {
    if (!startedRef.current) {
      startedRef.current = true;
      startBackend();
    }
    retryDelayRef.current = setTimeout(() => setShowRetry(true), 30_000);
    return () => {
      clearTimer();
      if (retryDelayRef.current) {
        clearTimeout(retryDelayRef.current);
        retryDelayRef.current = undefined;
      }
    };
  }, [startBackend]);

  if (status === "ready") {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-6">
      <div className="flex flex-col items-center gap-5 w-full max-w-2xl">
        {/* Spinner and title */}
        <div className="flex items-center gap-3">
          <Loader2 className="h-8 w-8 text-primary animate-spin" />
          <h2 className="text-lg font-semibold text-foreground">
            正在启动 Image Trace 后端服务…
          </h2>
        </div>

        {/* Error message */}
        {errorMsg && (
          <div className="text-sm text-red-400 bg-red-400/10 px-4 py-2 rounded-md border border-red-400/20 max-w-lg">
            {errorMsg}
          </div>
        )}

        {/* Log terminal */}
        <div className="w-full rounded-lg border border-border bg-[#0d1117] overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 bg-[#161b22] border-b border-border">
            <Terminal className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground font-mono">启动日志</span>
            <span className="ml-auto text-xs text-muted-foreground font-mono">
              {logs.length} 行
            </span>
          </div>
          <div
            ref={logContainerRef}
            className="p-3 h-72 overflow-y-auto font-mono text-xs leading-relaxed"
          >
            {logs.length === 0 ? (
              <div className="text-muted-foreground/50 italic">等待日志输出…</div>
            ) : (
              logs.map((line, i) => (
                <div
                  key={i}
                  className={`whitespace-pre-wrap break-all ${line.includes("❌") || line.includes("[stderr]")
                      ? "text-red-400"
                      : line.includes("✅")
                        ? "text-green-400"
                        : line.includes("⚠️")
                          ? "text-yellow-400"
                          : "text-gray-300"
                    }`}
                >
                  {line}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Retry button */}
        {showRetry && (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => startBackend()}
            className="gap-1"
            disabled={status === "starting" || status === "checking"}
          >
            <RefreshCw className="h-4 w-4" />
            重新启动后端
          </Button>
        )}
      </div>
    </div>
  );
}
