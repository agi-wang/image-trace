export interface DetailedError {
  message: string;
  statusCode?: number;
  endpoint?: string;
  timestamp: string;
  stack?: string;
  response?: unknown;
}

export class APIError extends Error {
  statusCode?: number;
  endpoint?: string;
  timestamp: string;
  response?: unknown;

  constructor(message: string, statusCode?: number, endpoint?: string, response?: unknown) {
    super(message);
    this.name = "APIError";
    this.statusCode = statusCode;
    this.endpoint = endpoint;
    this.timestamp = new Date().toISOString();
    this.response = response;
  }

  toDetailedString(): string {
    const parts = [
      `错误信息: ${this.message}`,
      `时间: ${new Date(this.timestamp).toLocaleString("zh-CN")}`,
    ];

    if (this.statusCode) {
      parts.push(`状态码: ${this.statusCode}`);
    }

    if (this.endpoint) {
      parts.push(`接口: ${this.endpoint}`);
    }

    if (this.response) {
      parts.push(`响应详情: ${JSON.stringify(this.response, null, 2)}`);
    }

    if (this.stack) {
      parts.push(`\n堆栈跟踪:\n${this.stack}`);
    }

    return parts.join("\n");
  }
}

export const copyErrorToClipboard = async (error: Error | APIError): Promise<boolean> => {
  try {
    const errorText =
      error instanceof APIError ? error.toDetailedString() : `${error.message}\n\n${error.stack || ""}`;

    await navigator.clipboard.writeText(errorText);
    return true;
  } catch (e) {
    console.error("Failed to copy error to clipboard:", e);
    return false;
  }
};
