import { APIError } from "./errorHandler";

export let API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/** Update the API base URL at runtime (called by BackendGate when backend starts on a dynamic port) */
export function setApiBaseUrl(url: string) {
  API_BASE_URL = url.replace(/\/$/, "");
}

export type HashType =
  | "orb"
  | "brisk"
  | "sift"
  | "akaze"
  | "kaze"
  | "phash"
  | "dhash"
  | "ahash"
  | "whash"
  | "colorhash"
  | "ssim"
  | "histogram"
  | "template"
  | "auto";

function getJsonHeaders(): HeadersInit {
  return { "Content-Type": "application/json" };
}

async function getAuthHeaders(): Promise<HeadersInit> {
  // Local desktop app — no cloud auth needed
  return {};
}

export interface Project {
  id: number;
  name: string;
  description?: string;
  created_at?: string;
  image_count?: number;
}

export interface Image {
  id: number;
  filename: string;
  file_path: string;
  file_size?: number;
  phash?: string;
  dhash?: string | null;
  ahash?: string | null;
  whash?: string | null;
  extracted_from?: string | null;
  project_id?: number;
  created_at?: string;
  public_url?: string;
}

function toStaticUrl(filePath: string): string {
  const sanitized = String(filePath || "").startsWith("data/")
    ? String(filePath).slice("data/".length)
    : String(filePath || "");
  return `${API_BASE_URL.replace(/\/$/, "")}/static/${sanitized}`;
}

function toThumbnailUrl(imageId: number, size = 400): string {
  return `${API_BASE_URL.replace(/\/$/, "")}/thumbnail/${imageId}?size=${size}`;
}

function withPublicUrl(img: Image): Image {
  return { ...img, public_url: toThumbnailUrl(img.id) };
}

function attachPublicUrlToAnalysis(result: AnalysisResult): AnalysisResult {
  return {
    ...result,
    groups: (result.groups || []).map((g) => ({
      ...g,
      images: (g.images || []).map(withPublicUrl),
    })),
    unique_images: (result.unique_images || []).map(withPublicUrl),
  };
}

export interface Document {
  id: string;
  filename: string;
  status: "completed" | "failed";
  project_id: number;
  extracted_images_count?: number;
  images?: Image[];
  error?: string;
}

export interface AnalysisResult {
  project_id: number;
  total_images: number;
  groups: Array<{
    group_id: number;
    similarity_score: number;
    images: Image[];
  }>;
  unique_images: Image[];
  run_id?: number;
}

export interface AnalysisRun {
  id: number;
  project_id: number;
  hash_type: HashType;
  threshold: number;
  total_images: number;
  groups_count: number;
  unique_count: number;
  created_at: string;
  summary?: string;
}

// Projects
export const createProject = async (name: string, description?: string): Promise<Project> => {
  const endpoint = `${API_BASE_URL}/projects`;
  const headers = { ...getJsonHeaders(), ...(await getAuthHeaders()) };
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({ name, description }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "创建项目失败",
      response.status,
      endpoint,
      error
    );
  }
  return response.json();
};

export const getProjects = async (): Promise<Project[]> => {
  const endpoint = `${API_BASE_URL}/projects`;
  try {
    const response = await fetch(endpoint, { headers: await getAuthHeaders() });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new APIError(
        error.detail || "获取项目列表失败",
        response.status,
        endpoint,
        error
      );
    }
    return response.json();
  } catch (error) {
    console.error("API请求失败:", error);
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      "无法连接到后端服务，请检查网络或后端是否运行",
      0,
      endpoint,
      error
    );
  }
};

export const getProject = async (projectId: string): Promise<Project> => {
  const endpoint = `${API_BASE_URL}/projects/${projectId}`;
  const response = await fetch(endpoint, { headers: await getAuthHeaders() });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "获取项目详情失败",
      response.status,
      endpoint,
      error
    );
  }
  return response.json();
};

export const deleteProject = async (projectId: string): Promise<void> => {
  const endpoint = `${API_BASE_URL}/projects/${projectId}`;
  const response = await fetch(endpoint, {
    method: "DELETE",
    headers: await getAuthHeaders(),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "删除项目失败",
      response.status,
      endpoint,
      error
    );
  }
};

// Images - Upload batch
type ProcessedImage = { id: number; filename: string; file_path?: string };

export const uploadImages = async (projectId: string, files: File[]): Promise<Image[]> => {
  const endpoint = `${API_BASE_URL}/upload`;
  const results: Image[] = [];
  const authHeaders = await getAuthHeaders();

  for (const file of files) {
    const fd = new FormData();
    fd.append("project_id", projectId);
    fd.append("file", file);

    const response = await fetch(endpoint, {
      method: "POST",
      headers: authHeaders,
      body: fd,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new APIError(
        error.detail || "上传图片失败",
        response.status,
        endpoint,
        error
      );
    }
    const data = await response.json();
    // data.processed_images 中只有 id/filename/file_path
    if (Array.isArray(data.processed_images)) {
      (data.processed_images as ProcessedImage[]).forEach((img) => {
        results.push({
          id: img.id,
          filename: img.filename,
          file_path: img.file_path || img.filename,
          public_url: toStaticUrl(img.file_path || img.filename),
          project_id: Number(projectId),
        });
      });
    } else {
      results.push({
        id: data.id ?? Date.now(),
        filename: data.filename,
        file_path: data.file_path,
        public_url: toStaticUrl(data.file_path),
        project_id: Number(projectId),
      });
    }
  }

  return results;
};

// Legacy single image upload for backward compatibility
export const uploadImage = async (projectId: string, file: File): Promise<Image> => {
  const results = await uploadImages(projectId, [file]);
  return results[0];
};

// Documents - Upload
export const uploadDocument = async (projectId: string, file: File): Promise<Document> => {
  // 后端 /upload 会同步提取图片，返回 processed_images
  const imgs = await uploadImages(projectId, [file]);
  return {
    id: `${Date.now()}`,
    filename: file.name,
    status: "completed",
    project_id: Number(projectId),
    extracted_images_count: imgs.length,
    images: imgs,
  };
};

// 文档二次提取（对应后端 /extract/{file_path}）
export const extractDocument = async (projectId: string, filePath: string): Promise<Document> => {
  const endpoint = `${API_BASE_URL}/extract/${encodeURIComponent(filePath)}`;
  const fd = new FormData();
  fd.append("project_id", projectId);
  const response = await fetch(endpoint, { method: "POST", headers: await getAuthHeaders(), body: fd });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "文档提取失败",
      response.status,
      endpoint,
      error
    );
  }
  const data = await response.json();
  return {
    id: `${Date.now()}`,
    filename: filePath,
    status: data.status === "success" ? "completed" : "failed",
    project_id: Number(projectId),
    extracted_images_count: Array.isArray(data.images) ? data.images.length : 0,
    images: Array.isArray(data.images)
      ? (data.images as ProcessedImage[]).map((img) => ({
        ...img,
        public_url: toStaticUrl(img.file_path || ""),
      }))
      : [],
    error: data.error,
  };
};

export const getProjectImages = async (projectId: string): Promise<Image[]> => {
  const endpoint = `${API_BASE_URL}/images/${projectId}`;
  const response = await fetch(endpoint, { headers: await getAuthHeaders() });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "获取图片列表失败",
      response.status,
      endpoint,
      error
    );
  }
  const data: Image[] = await response.json();
  return data.map(withPublicUrl);
};

export const deleteImage = async (projectId: string, imageId: string): Promise<void> => {
  const endpoint = `${API_BASE_URL}/images/${imageId}`;
  const response = await fetch(endpoint, { method: "DELETE", headers: await getAuthHeaders() });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "删除图片失败",
      response.status,
      endpoint,
      error
    );
  }
};

// Analysis
export const analyzeImages = async (
  projectId: string,
  hashType: HashType = "orb",
  threshold = 0.85,
  rotationInvariant = false
): Promise<AnalysisResult> => {
  const endpoint = `${API_BASE_URL}/compare/${projectId}`;
  const fd = new FormData();
  fd.append("threshold", String(threshold));
  fd.append("hash_type", hashType);
  fd.append("rotation_invariant", String(rotationInvariant));
  const headers = await getAuthHeaders();
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: fd,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "启动分析失败",
      response.status,
      endpoint,
      error
    );
  }
  const data: AnalysisResult = await response.json();
  return attachPublicUrlToAnalysis(data);
};

// GET 版结果（对应后端 /results/{project_id}）
export const getComparisonResults = async (
  projectId: string,
  hashType: HashType = "orb",
  threshold = 0.85
): Promise<AnalysisResult> => {
  const endpoint = `${API_BASE_URL}/results/${projectId}?threshold=${encodeURIComponent(
    threshold
  )}&hash_type=${encodeURIComponent(hashType)}`;
  const response = await fetch(endpoint, { headers: await getAuthHeaders() });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "获取分析结果失败",
      response.status,
      endpoint,
      error
    );
  }
  const data: AnalysisResult = await response.json();
  return attachPublicUrlToAnalysis(data);
};

export const getAnalysisRuns = async (projectId: string): Promise<AnalysisRun[]> => {
  const endpoint = `${API_BASE_URL}/analysis_runs?project_id=${encodeURIComponent(projectId)}`;
  const res = await fetch(endpoint, { headers: await getAuthHeaders() });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new APIError(error.detail || "获取分析历史失败", res.status, endpoint, error);
  }
  return res.json();
};

export const getAnalysisRunDetail = async (runId: number): Promise<{ run: AnalysisRun; result: AnalysisResult | null }> => {
  const endpoint = `${API_BASE_URL}/analysis_runs/${runId}`;
  const res = await fetch(endpoint, { headers: await getAuthHeaders() });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new APIError(error.detail || "获取分析详情失败", res.status, endpoint, error);
  }
  const data = await res.json();
  const parsed = data.result ? attachPublicUrlToAnalysis(data.result as AnalysisResult) : null;
  return { run: data.run, result: parsed };
};

// Get analysis status/result（同步：直接视为完成）
export const getAnalysisStatus = async (
  analysisId: string,
  hashType: HashType = "orb",
  threshold = 0.85
): Promise<AnalysisResult> => {
  return getComparisonResults(analysisId, hashType, threshold);
};

export const getAnalysisResult = async (
  analysisId: string,
  hashType: HashType = "orb",
  threshold = 0.85
): Promise<AnalysisResult> => {
  return getComparisonResults(analysisId, hashType, threshold);
};

// ─── Smart Compare: 一键智能查重 ───────────────────────────────────

export interface SmartCompareGroup {
  images: { id: number; filename: string; file_path: string }[];
  confidence: number;
  matched_algorithms: string[];
  matched_count: number;
}

export interface SmartCompareResult {
  total_images: number;
  algorithms_used: number;
  found_duplicates: boolean;
  duplicate_groups: SmartCompareGroup[];
  unique_count: number;
  scan_seconds: number;
  summary: string;
  features_pending?: boolean;
}

export const smartCompare = async (
  projectId: string,
  threshold?: number,
): Promise<SmartCompareResult> => {
  const params = new URLSearchParams();
  if (typeof threshold === "number") {
    params.set("threshold", String(threshold));
  }
  const endpoint = `${API_BASE_URL}/smart_compare/${projectId}${params.toString() ? `?${params.toString()}` : ""}`;
  const response = await fetch(
    endpoint,
    { method: "POST" },
  );
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new APIError(
      detail?.detail || `Smart compare failed (${response.status})`,
      response.status,
    );
  }
  return response.json();
};

export const recomputeFeatures = async (
  projectId: string,
): Promise<{ triggered: number; message: string }> => {
  const response = await fetch(
    `${API_BASE_URL}/recompute_features/${projectId}`,
    { method: "POST" },
  );
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new APIError(
      detail?.detail || `Recompute failed (${response.status})`,
      response.status,
    );
  }
  return response.json();
};

export const checkHealth = async (): Promise<{ status: string }> => {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
};

export const visualizeMatch = async (
  imageAId: number,
  imageBId: number,
  hashType: HashType = "orb"
): Promise<{ url: string }> => {
  const endpoint = `${API_BASE_URL}/visualize_match`;
  const fd = new FormData();
  fd.append("image_a_id", String(imageAId));
  fd.append("image_b_id", String(imageBId));
  fd.append("hash_type", hashType);
  const response = await fetch(endpoint, { method: "POST", headers: await getAuthHeaders(), body: fd });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "生成特征匹配图失败",
      response.status,
      endpoint,
      error
    );
  }
  const data = await response.json();
  return { url: toStaticUrl(data.file_path) };
};

// Pairwise Matrix
export interface PairwiseMatrixResult {
  names: string[];
  image_ids: number[];
  matrix: number[][];
  algorithm: string;
  engine?: "matrix" | "legacy";
}

export const getPairwiseMatrix = async (
  projectId: string,
  hashType: HashType = "sift"
): Promise<PairwiseMatrixResult> => {
  const endpoint = `${API_BASE_URL}/pairwise_matrix/${projectId}?hash_type=${encodeURIComponent(hashType)}`;
  const response = await fetch(endpoint, { headers: await getAuthHeaders() });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "获取相似度矩阵失败",
      response.status,
      endpoint,
      error
    );
  }
  return response.json();
};

// Match Data (keypoints + matches as JSON for frontend rendering)
export interface MatchKeypoint {
  x: number;
  y: number;
}

export interface MatchLine {
  a_idx: number;
  b_idx: number;
  distance: number;
}

export interface MatchData {
  image_a: { width: number; height: number; keypoints: MatchKeypoint[] };
  image_b: { width: number; height: number; keypoints: MatchKeypoint[] };
  matches: MatchLine[];
  score: number;
}

export const getMatchData = async (
  imageAId: number,
  imageBId: number,
  hashType: HashType = "sift"
): Promise<MatchData> => {
  const endpoint = `${API_BASE_URL}/match_data`;
  const fd = new FormData();
  fd.append("image_a_id", String(imageAId));
  fd.append("image_b_id", String(imageBId));
  fd.append("hash_type", hashType);
  const response = await fetch(endpoint, { method: "POST", headers: await getAuthHeaders(), body: fd });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(
      error.detail || "Match data request failed",
      response.status,
      endpoint,
      error
    );
  }
  return response.json();
};

export { toThumbnailUrl };

// System Info
export interface SystemInfo {
  engines: string[];
  algorithms: string[];
  matrix_engine: string;
  description: string;
}

export const getSystemInfo = async (): Promise<SystemInfo> => {
  const response = await fetch(`${API_BASE_URL}/system_info`);
  if (!response.ok) throw new Error("Failed to get system info");
  return response.json();
};

// Feature Status
export interface FeatureStatus {
  project_id: number;
  total: number;
  ready: number;
  all_ready: boolean;
  images: Array<{ id: number; filename: string; status: string }>;
}

export const getFeatureStatus = async (projectId: string): Promise<FeatureStatus> => {
  const response = await fetch(`${API_BASE_URL}/feature_status/${projectId}`, {
    headers: await getAuthHeaders(),
  });
  if (!response.ok) throw new Error("Failed to get feature status");
  return response.json();
};
