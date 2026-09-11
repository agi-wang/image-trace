import { useEffect, useState, useMemo, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  getProject,
  getProjectImages,
  analyzeImages,
  smartCompare as smartCompareApi,
  SmartCompareResult as SmartCompareResultType,
  getComparisonResults,
  getPairwiseMatrix,
  getAnalysisRuns,
  getAnalysisRunDetail,
  type Project,
  type Image,
  type AnalysisResult,
  type HashType,
  type AnalysisRun,
  type PairwiseMatrixResult,
} from "@/lib/api";
import { copyErrorToClipboard, APIError } from "@/lib/errorHandler";
import { ImageUploadZone } from "@/components/ImageUploadZone";
import { DocumentUploadZone } from "@/components/DocumentUploadZone";
import { AnalysisPanel } from "@/components/AnalysisPanel";
import { SimilarityMatrix } from "@/components/SimilarityMatrix";
import { SimilarityGraph } from "@/components/SimilarityGraph";
import { FeatureMatchView } from "@/components/FeatureMatchView";
import { SystemHealth } from "@/components/SystemHealth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { ArrowLeft, Loader2, Image as ImageIcon, RefreshCw, Copy, FileText, Search, ChevronDown, ChevronUp } from "lucide-react";
import { SmartCompareResultView } from "@/components/SmartCompareResult";

const ALGORITHM_LABELS: Record<string, string> = {
  phash: "pHash",
  dhash: "dHash",
  ahash: "aHash",
  whash: "wHash",
  colorhash: "ColorHash",
  orb: "ORB",
  brisk: "BRISK",
  sift: "SIFT",
  akaze: "AKAZE",
  kaze: "KAZE",
  ssim: "SSIM",
  histogram: "Histogram",
  template: "Template",
  auto: "Auto (Hybrid)",
};

export default function ProjectDetail() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { t, i18n } = useTranslation();

  const [project, setProject] = useState<Project | null>(null);
  const [images, setImages] = useState<Image[]>([]);
  const [compareResult, setCompareResult] = useState<AnalysisResult | null>(null);
  const [smartResult, setSmartResult] = useState<SmartCompareResultType | null>(null);
  const [matchPair, setMatchPair] = useState<{ aId: number; bId: number; aName: string; bName: string } | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [matchAlgo, setMatchAlgo] = useState<HashType>("orb");
  const [runs, setRuns] = useState<AnalysisRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [smartComparing, setSmartComparing] = useState(false);
  const [pairwiseData, setPairwiseData] = useState<PairwiseMatrixResult | null>(null);
  const [loadingPairwise, setLoadingPairwise] = useState(false);
  const locale = i18n.language?.toLowerCase().startsWith("zh") ? "zh-CN" : "en-US";

  const showErrorToast = useCallback((err: APIError, title: string) => {
    toast({
      title,
      description: err.message,
      variant: "destructive",
      action: (
        <Button
          variant="secondary"
          size="sm"
          className="bg-white/10 hover:bg-white/20 text-white border-white/20"
          onClick={async () => {
            const success = await copyErrorToClipboard(err);
            if (success) {
              toast({ title: t("project.copyDetail") });
            }
          }}
        >
          <Copy className="h-3 w-3 mr-1" />
          {t("common.copy")}
        </Button>
      ),
    });
  }, [t, toast]);

  const loadProject = useCallback(async () => {
    if (!projectId) return;

    setLoading(true);
    try {
      const [projectData, imagesData] = await Promise.all([
        getProject(projectId),
        getProjectImages(projectId),
      ]);
      setProject(projectData);
      setImages(imagesData);
    } catch (error) {
      showErrorToast(error as APIError, t("project.loadFailed"));
    } finally {
      setLoading(false);
    }
  }, [projectId, showErrorToast, t]);

  const prefetchResult = useCallback(async () => {
    if (!projectId) return;
    try {
      const res = await getComparisonResults(projectId);
      setCompareResult(res);
      const matrix = await getPairwiseMatrix(projectId, "orb");
      setPairwiseData(matrix);
      setMatchAlgo("orb");
    } catch {
      // 忽略拉取失败，保持静默
    } finally {
      // no-op
    }
  }, [projectId]);

  const loadRuns = useCallback(async () => {
    if (!projectId) return;
    setLoadingRuns(true);
    try {
      const data = await getAnalysisRuns(projectId);
      setRuns(data);
    } catch {
      // 静默
    } finally {
      setLoadingRuns(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadProject();
    prefetchResult();
    loadRuns();
  }, [loadProject, prefetchResult, loadRuns, projectId]);

  const handleImagesUploaded = (uploadedImages: Image[]) => {
    setImages((prev) => [...prev, ...uploadedImages]);
    setCompareResult(null);
  };

  const handleDocumentUploaded = () => {
    // 文档上传会产出图片，刷新即可
    loadProject();
  };

  const descriptorMatchAlgo = useCallback((algo: HashType): HashType => {
    return ["sift", "orb", "brisk", "akaze", "kaze"].includes(algo) ? algo : "orb";
  }, []);

  const loadPairwiseForAlgo = useCallback(async (algo: HashType) => {
    if (!projectId) return;
    setLoadingPairwise(true);
    try {
      const data = await getPairwiseMatrix(projectId, algo);
      setPairwiseData(data);
    } catch {
      setPairwiseData(null);
    } finally {
      setLoadingPairwise(false);
    }
    setMatchAlgo(descriptorMatchAlgo(algo));
  }, [descriptorMatchAlgo, projectId]);

  const handleAnalyze = useCallback(async (algo: HashType, rotationInvariant: boolean = false) => {
    if (!projectId) return;
    setAnalyzing(true);
    try {
      const res = await analyzeImages(projectId, algo, 0.85, rotationInvariant);
      setCompareResult(res);
      await loadPairwiseForAlgo(algo);

      toast({ title: t("project.analyzeSuccess"), description: t("project.analyzeSuccessDesc", { total: res.total_images, groups: res.groups.length }) });
      loadRuns();
    } catch (error) {
      showErrorToast(error as APIError, t("project.analyzeFailed"));
    } finally {
      setAnalyzing(false);
    }
  }, [loadPairwiseForAlgo, projectId, showErrorToast, t, toast, loadRuns]);

  const handleSmartCompare = useCallback(async () => {
    if (!projectId) return;
    setSmartComparing(true);
    setSmartResult(null);
    try {
      const result = await smartCompareApi(projectId);
      setSmartResult(result);
      if (result.features_pending) {
        toast({ title: "特征计算中", description: result.summary });
      } else {
        toast({ title: "查重完成", description: result.summary });
      }
    } catch (error) {
      const err = error as APIError;
      toast({ title: "查重失败", description: err.message, variant: "destructive" });
    } finally {
      setSmartComparing(false);
    }
  }, [projectId, toast]);

  const handleSmartCompareRetry = useCallback(async () => {
    if (!projectId) return;
    setSmartComparing(true);
    try {
      const r = await smartCompareApi(projectId);
      setSmartResult(r);
    } catch { /* ignore */ }
    setSmartComparing(false);
  }, [projectId]);

  const similarityMatrix = useMemo(() => {
    if (!compareResult) return null;
    // Flatten all images for matrix display
    const allImages: Image[] = [
      ...compareResult.groups.flatMap((g) => g.images),
      ...compareResult.unique_images,
    ];
    const n = allImages.length;
    if (n === 0) return null;
    const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    // For each group, set similarity_score for pairs
    compareResult.groups.forEach((g) => {
      const indices = g.images.map((img) => allImages.findIndex((x) => x.id === img.id)).filter((i) => i >= 0);
      for (let i = 0; i < indices.length; i++) {
        for (let j = i + 1; j < indices.length; j++) {
          matrix[indices[i]][indices[j]] = g.similarity_score;
          matrix[indices[j]][indices[i]] = g.similarity_score;
        }
      }
    });
    return { matrix, images: allImages };
  }, [compareResult]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">{t("project.notFound")}</h2>
          <Button onClick={() => navigate("/")}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            {t("project.backHome")}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/")}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              {t("project.back")}
            </Button>
            <div>
              <h1 className="text-3xl font-bold">{project.name}</h1>
              {project.description && (
                <p className="text-muted-foreground mt-1">{project.description}</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="default"
              size="sm"
              onClick={() => navigate(`/report/${projectId}`)}
            >
              <FileText className="h-4 w-4 mr-2" />
              {t("project.generateReport") || "Generate Report"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={loadProject}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              {t("project.refresh")}
            </Button>
          </div>
        </div>

        {/* System Health */}
        <SystemHealth autoRefresh={true} refreshInterval={30000} />
        {runs.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{t("project.history")}</CardTitle>
                {loadingRuns && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
              </div>
              <CardDescription>{t("project.historyDesc")}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {runs.map((r) => (
                  <div key={r.id} className="rounded border p-3 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{t("project.runId", { id: r.id })}</span>
                      <Badge variant="secondary">{r.hash_type}</Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(r.created_at).toLocaleString(locale)}
                    </div>
                    <div className="text-xs text-muted-foreground flex gap-3">
                      <span>{t("project.groupsCount", { count: r.groups_count })}</span>
                      <span>{t("project.uniqueCount", { count: r.unique_count })}</span>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full mt-1"
                      onClick={async () => {
                        try {
                          const { result } = await getAnalysisRunDetail(r.id);
                          if (result) {
                            setCompareResult(result);
                            await loadPairwiseForAlgo(r.hash_type);
                            toast({ title: t("project.runLoaded"), description: t("project.runId", { id: r.id }) });
                          } else {
                            toast({ title: t("project.runNoResult"), description: t("project.runId", { id: r.id }), variant: "destructive" });
                          }
                        } catch (error) {
                          const err = error as APIError;
                          toast({ title: t("project.loadFailed"), description: err.message, variant: "destructive" });
                        }
                      }}
                    >
                      {t("project.viewResult")}
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle>{t("project.uploadSection")}</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="images" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="images">
                  <ImageIcon className="h-4 w-4 mr-2" />
                  {t("project.uploadImagesTab")}
                </TabsTrigger>
                <TabsTrigger value="documents">
                  <FileText className="h-4 w-4 mr-2" />
                  {t("project.uploadDocsTab")}
                </TabsTrigger>
              </TabsList>
              <TabsContent value="images" className="mt-4">
                <ImageUploadZone
                  projectId={projectId!}
                  onImagesUploaded={handleImagesUploaded}
                />
              </TabsContent>
              <TabsContent value="documents" className="mt-4">
                <DocumentUploadZone
                  projectId={projectId!}
                  onDocumentUploaded={handleDocumentUploaded}
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Uploaded Images */}
        {images.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{t("project.uploadedTitle")}</CardTitle>
                <Badge variant="secondary">{t("project.uploadedCount", { count: images.length })}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {images.map((image) => (
                  <div
                    key={image.id}
                    className="group relative aspect-square rounded-lg border overflow-hidden hover:shadow-lg transition-all"
                  >
                    <img
                      src={image.public_url || "/placeholder.svg"}
                      alt={image.filename}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.currentTarget.src = "/placeholder.svg";
                      }}
                    />
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <div className="text-white text-xs text-center px-2">
                        <p className="font-medium truncate">{image.filename}</p>
                        <p className="text-white/70">
                          {image.file_size ? `${(image.file_size / 1024).toFixed(1)} KB` : "—"}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* ─── Smart Compare: 一键智能查重 ─── */}
        {images.length > 1 && (
          <AnalysisPanel
            hasImages={images.length > 1}
            loading={analyzing}
            onAnalyze={handleAnalyze}
          />
        )}

        {/* ─── Smart Compare: 一键智能查重 ─── */}
        {images.length > 1 && (
          <Card className="border-primary/30">
            <CardContent className="py-6">
              <div className="flex flex-col sm:flex-row items-center gap-4">
                <div className="flex-1">
                  <h3 className="text-base font-semibold mb-1">智能查重</h3>
                  <p className="text-sm text-muted-foreground">
                    使用 10 种算法（含 8 方向旋转检测）自动比对所有图片，特征已在上传时预计算完成
                  </p>
                </div>
                <Button
                  size="lg"
                  className="gap-2 px-8"
                  disabled={smartComparing}
                  onClick={handleSmartCompare}
                >
                  {smartComparing ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Search className="h-5 w-5" />
                  )}
                  {smartComparing ? "正在比对…" : "🔍 一键智能查重"}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Smart Compare Results */}
         {smartResult && (
           <SmartCompareResultView
             result={smartResult}
             projectId={projectId}
             onRetry={handleSmartCompareRetry}
           />
         )}

        {/* ─── Advanced View (collapsed) ─── */}
        {compareResult && similarityMatrix && (
          <Card>
              <CardHeader className="cursor-pointer" onClick={() => setShowAdvanced((v) => !v)}>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    高级分析视图
                    {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </CardTitle>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    {loadingPairwise && <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />}
                    <span>
                      相似度矩阵 · 网络图 · {ALGORITHM_LABELS[(pairwiseData?.algorithm as HashType) || "auto"] || pairwiseData?.algorithm}
                    </span>
                  </div>
                </div>
              </CardHeader>
            {showAdvanced && (
              <CardContent className="space-y-4">
                {pairwiseData && pairwiseData.matrix.length > 0 ? (
                  <SimilarityMatrix matrix={pairwiseData.matrix} imageNames={pairwiseData.names} />
                ) : similarityMatrix ? (
                  <SimilarityMatrix matrix={similarityMatrix.matrix} images={similarityMatrix.images} />
                ) : null}
                <SimilarityGraph
                  groups={compareResult.groups}
                  imageIds={pairwiseData?.image_ids}
                  matchAlgo={matchAlgo}
                  onEdgeClick={(imgAId: number, imgBId: number) => {
                    const allImgs = [...compareResult.groups.flatMap(g => g.images), ...compareResult.unique_images];
                    const aImg = allImgs.find(i => i.id === imgAId);
                    const bImg = allImgs.find(i => i.id === imgBId);
                    setMatchPair({
                      aId: imgAId,
                      bId: imgBId,
                      aName: aImg?.filename || `#${imgAId}`,
                      bName: bImg?.filename || `#${imgBId}`,
                    });
                  }}
                />
              </CardContent>
            )}
          </Card>
        )}

        {matchPair && (
          <FeatureMatchView
            imageAId={matchPair.aId}
            imageBId={matchPair.bId}
            imageAName={matchPair.aName}
            imageBName={matchPair.bName}
            algorithm={matchAlgo}
            onClose={() => setMatchPair(null)}
          />
        )}
      </div>
    </div>
  );
}
