import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Activity, RefreshCw, Server, Clock } from "lucide-react";
import { useEffect, useState } from "react";
import { checkHealth, API_BASE_URL } from "@/lib/api";

interface SystemHealthProps {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function SystemHealth({ autoRefresh = true, refreshInterval = 30000 }: SystemHealthProps) {
  const [health, setHealth] = useState<{ status: string; timestamp?: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const data = await checkHealth();
      setHealth({ ...data, timestamp: new Date().toISOString() });
      setLastUpdate(new Date());
    } catch (error) {
      setHealth({ status: "error" });
      setLastUpdate(new Date());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();

    if (autoRefresh) {
      const interval = setInterval(fetchHealth, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "ok":
      case "healthy":
        return "bg-success text-success-foreground";
      case "error":
        return "bg-destructive text-destructive-foreground";
      default:
        return "bg-secondary text-secondary-foreground";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "ok":
      case "healthy":
        return <Activity className="h-4 w-4 text-success" />;
      case "error":
        return <Activity className="h-4 w-4 text-destructive" />;
      default:
        return <Activity className="h-4 w-4 text-muted-foreground" />;
    }
  };

  return (
    <Card className="border-border/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Server className="h-4 w-4 text-primary" />
            系统状态
          </CardTitle>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={fetchHealth}
            disabled={loading}
          >
            <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {health && getStatusIcon(health.status)}
            <span className="text-sm text-muted-foreground">服务状态</span>
          </div>
          {health && (
            <Badge className={getStatusColor(health.status)} variant="secondary">
              {health.status === "ok" || health.status === "healthy" ? "正常" : "异常"}
            </Badge>
          )}
        </div>

        {lastUpdate && (
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              <span>上次更新</span>
            </div>
            <span>{lastUpdate.toLocaleTimeString("zh-CN")}</span>
          </div>
        )}

        <div className="text-xs text-muted-foreground border-t pt-2">
          <div className="flex items-center justify-between">
            <span>API地址</span>
            <span className="font-mono">
              {API_BASE_URL.replace(/\/$/, "")}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
