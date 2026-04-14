import { useState } from "react";
import { useTranslation } from "react-i18next";
import { SmartCompareResult as SmartResult, SmartCompareGroup, recomputeFeatures } from "@/lib/api";
import { API_BASE_URL } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
    CheckCircle2,
    XCircle,
    Clock,
    Images,
    Search,
    Shield,
    AlertTriangle,
    Copy,
    RefreshCw,
    Loader2,
} from "lucide-react";

interface SmartCompareResultProps {
    result: SmartResult;
    projectId?: string;
    onRetry?: () => void;
}

function toThumbUrl(imageId: number, size = 300): string {
    return `${API_BASE_URL.replace(/\/$/, "")}/thumbnail/${imageId}?size=${size}`;
}

function confidenceColor(c: number): string {
    if (c >= 0.95) return "text-red-400";
    if (c >= 0.90) return "text-orange-400";
    if (c >= 0.85) return "text-yellow-400";
    return "text-muted-foreground";
}

function confidenceBg(c: number): string {
    if (c >= 0.95) return "bg-red-500/10 border-red-500/30";
    if (c >= 0.90) return "bg-orange-500/10 border-orange-500/30";
    if (c >= 0.85) return "bg-yellow-500/10 border-yellow-500/30";
    return "bg-muted/20 border-border";
}

async function copyToClipboard(text: string): Promise<boolean> {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        return ok;
    }
}

function DuplicateGroup({
    group,
    index,
    totalAlgorithms,
}: {
    group: SmartCompareGroup;
    index: number;
    totalAlgorithms: number;
}) {
    const { t } = useTranslation();

    return (
        <div
            className={`rounded-xl border-2 p-4 space-y-3 ${confidenceBg(
                group.confidence
            )}`}
        >
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <AlertTriangle
                        className={`h-4 w-4 ${confidenceColor(group.confidence)}`}
                    />
                    <span className="text-sm font-semibold">
                        {t("smartCompare.groupLabel", { index: index + 1, count: group.images.length })}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <Badge
                        variant="outline"
                        className={`text-xs ${confidenceColor(group.confidence)}`}
                    >
                        {t("smartCompare.similarity", { percent: (group.confidence * 100).toFixed(1) })}
                    </Badge>
                    <Badge variant="secondary" className="text-xs">
                        <Shield className="h-3 w-3 mr-1" />
                        {t("smartCompare.algoConfirmed", { matched: group.matched_count, total: totalAlgorithms })}
                    </Badge>
                </div>
            </div>

            <div className="flex flex-wrap gap-3">
                {group.images.map((img) => (
                    <div
                        key={img.id}
                        className="relative group rounded-lg overflow-hidden border bg-background"
                    >
                        <img
                            src={toThumbUrl(img.id)}
                            alt={img.filename}
                            className="h-32 w-32 object-cover"
                            loading="lazy"
                        />
                        <div className="absolute bottom-0 inset-x-0 bg-black/70 px-2 py-1">
                            <p className="text-[10px] text-white truncate">{img.filename}</p>
                        </div>
                    </div>
                ))}
            </div>

            <div className="flex flex-wrap gap-1">
                {group.matched_algorithms.map((algo) => (
                    <Badge key={algo} variant="outline" className="text-[10px] px-1.5 py-0">
                        {algo.toUpperCase()}
                    </Badge>
                ))}
            </div>
        </div>
    );
}

export function SmartCompareResultView({ result, projectId, onRetry }: SmartCompareResultProps) {
    const { found_duplicates, duplicate_groups, summary, total_images, algorithms_used, scan_seconds, unique_count } = result;
    const [copied, setCopied] = useState(false);
    const [recomputing, setRecomputing] = useState(false);
    const { t } = useTranslation();

    const handleCopy = async () => {
        const ok = await copyToClipboard(summary);
        if (ok) {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const handleRecompute = async () => {
        if (!projectId) return;
        setRecomputing(true);
        try {
            await recomputeFeatures(projectId);
            setTimeout(() => {
                setRecomputing(false);
                onRetry?.();
            }, 2000);
        } catch {
            setRecomputing(false);
        }
    };

    if (result.features_pending) {
        return (
            <Card>
                <CardContent className="py-8">
                    <div className="flex flex-col items-center gap-3 text-center">
                        <Clock className="h-10 w-10 text-yellow-400 animate-pulse" />
                        <p className="text-sm text-muted-foreground max-w-md">{summary}</p>
                        <div className="flex items-center gap-2 mt-2">
                            <Button
                                variant="outline"
                                size="sm"
                                className="gap-1"
                                onClick={handleCopy}
                            >
                                <Copy className="h-3 w-3" />
                                {copied ? t("smartCompare.copied") : t("smartCompare.copyInfo")}
                            </Button>
                            <Button
                                variant="default"
                                size="sm"
                                className="gap-1"
                                disabled={recomputing}
                                onClick={handleRecompute}
                            >
                                {recomputing ? (
                                    <Loader2 className="h-3 w-3 animate-spin" />
                                ) : (
                                    <RefreshCw className="h-3 w-3" />
                                )}
                                {recomputing ? t("smartCompare.recomputing") : t("smartCompare.recompute")}
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2 text-base">
                            {found_duplicates ? (
                                <AlertTriangle className="h-5 w-5 text-orange-400" />
                            ) : (
                                <CheckCircle2 className="h-5 w-5 text-green-400" />
                            )}
                            {t("smartCompare.title")}
                        </CardTitle>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="gap-1 text-xs"
                            onClick={handleCopy}
                        >
                            <Copy className="h-3 w-3" />
                            {copied ? t("smartCompare.copied") : t("smartCompare.copy")}
                        </Button>
                    </div>
                </CardHeader>
                <CardContent>
                    <p className="text-sm mb-4">{summary}</p>

                    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                            <Images className="h-3.5 w-3.5" />
                            {t("smartCompare.totalImages", { count: total_images })}
                        </span>
                        <span className="flex items-center gap-1">
                            <Search className="h-3.5 w-3.5" />
                            {t("smartCompare.totalAlgorithms", { count: algorithms_used })}
                        </span>
                        <span className="flex items-center gap-1">
                            <Clock className="h-3.5 w-3.5" />
                            {scan_seconds}s
                        </span>
                        {!found_duplicates && (
                            <span className="flex items-center gap-1 text-green-400">
                                <CheckCircle2 className="h-3.5 w-3.5" />
                                {t("smartCompare.allUnique")}
                            </span>
                        )}
                        {found_duplicates && (
                            <span className="flex items-center gap-1 text-orange-400">
                                <XCircle className="h-3.5 w-3.5" />
                                {t("smartCompare.duplicateStats", { groups: duplicate_groups.length, unique: unique_count })}
                            </span>
                        )}
                    </div>
                </CardContent>
            </Card>

            {found_duplicates && (
                <div className="space-y-3">
                    {duplicate_groups.map((group, i) => (
                        <DuplicateGroup key={i} group={group} index={i} totalAlgorithms={algorithms_used} />
                    ))}
                </div>
            )}

            {!found_duplicates && total_images >= 2 && (
                <Card>
                    <CardContent className="py-10">
                        <div className="flex flex-col items-center gap-3 text-center">
                            <CheckCircle2 className="h-12 w-12 text-green-400" />
                            <p className="font-medium">{t("smartCompare.allUniqueTitle")}</p>
                            <p className="text-sm text-muted-foreground">
                                {t("smartCompare.allUniqueDesc", { count: algorithms_used })}
                            </p>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
