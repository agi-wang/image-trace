import React from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message?: string;
  stack?: string;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, message: error.message, stack: error.stack };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // 控制台记录，便于调试/回溯
    console.error("UI 捕获错误:", error, info);
    // 轻量记录到 sessionStorage，便于用户反馈
    try {
      sessionStorage.setItem(
        "last_ui_error",
        JSON.stringify({
          message: error.message,
          stack: error.stack,
          componentStack: info.componentStack,
          time: new Date().toISOString(),
        })
      );
    } catch {
      // ignore storage errors
    }
  }

  handleReload = () => {
    this.setState({ hasError: false });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background px-6">
          <div className="max-w-md w-full rounded-lg border bg-card p-6 shadow-sm space-y-3 text-center">
            <div className="flex justify-center">
              <AlertTriangle className="h-10 w-10 text-destructive" />
            </div>
            <div className="text-lg font-semibold">界面出现了小问题</div>
            <p className="text-sm text-muted-foreground">
              请刷新重试。如果持续出现，可复制错误详情反馈。
            </p>
            {this.state.message && (
              <p className="text-xs text-muted-foreground break-words">
                {this.state.message}
              </p>
            )}
            <div className="flex gap-2 justify-center pt-2">
              <Button size="sm" onClick={this.handleReload} className="gap-1">
                <RefreshCw className="h-4 w-4" />
                刷新
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={async () => {
                  const payload = this.state.stack || this.state.message || "Unknown error";
                  try {
                    await navigator.clipboard.writeText(payload);
                  } catch {
                    // ignore
                  }
                }}
              >
                复制错误
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
