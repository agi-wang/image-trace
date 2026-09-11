import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getMatchData, toThumbnailUrl, type MatchData, type HashType } from "@/lib/api";
import { Loader2, X } from "lucide-react";

interface FeatureMatchViewProps {
    imageAId: number;
    imageBId: number;
    imageAName: string;
    imageBName: string;
    algorithm: HashType;
    onClose: () => void;
}

export function FeatureMatchView({
    imageAId,
    imageBId,
    imageAName,
    imageBName,
    algorithm,
    onClose,
}: FeatureMatchViewProps) {
    const [data, setData] = useState<MatchData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showKeypoints, setShowKeypoints] = useState(true);

    useEffect(() => {
        setLoading(true);
        setError(null);
        getMatchData(imageAId, imageBId, algorithm)
            .then(setData)
            .catch((e) => setError(e.message || "Failed to load match data"))
            .finally(() => setLoading(false));
    }, [imageAId, imageBId, algorithm]);

    if (loading) {
        return (
            <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center">
                <Card className="w-96">
                    <CardContent className="flex items-center justify-center py-8 gap-3">
                        <Loader2 className="h-5 w-5 animate-spin text-primary" />
                        <span>Computing feature matches…</span>
                    </CardContent>
                </Card>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center px-4">
                <Card className="max-w-md w-full">
                    <CardHeader>
                        <CardTitle className="text-destructive">Error</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <p className="text-sm">{error || "No data"}</p>
                        <Button onClick={onClose} variant="outline" className="w-full">Close</Button>
                    </CardContent>
                </Card>
            </div>
        );
    }

    // Layout: two images side by side
    const DISPLAY_W = 400;
    const scaleA = DISPLAY_W / data.image_a.width;
    const scaleB = DISPLAY_W / data.image_b.width;
    const hA = data.image_a.height * scaleA;
    const hB = data.image_b.height * scaleB;
    const totalW = DISPLAY_W * 2 + 20; // 20px gap
    const totalH = Math.max(hA, hB);
    const offsetBX = DISPLAY_W + 20;

    // Generate colors for match lines
    const matchColors = data.matches.map((_, i) => {
        const hue = (i * 137.508) % 360; // golden angle for diversity
        return `hsl(${hue}, 80%, 60%)`;
    });

    return (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center px-4">
            <div className="bg-background rounded-lg shadow-xl max-w-5xl w-full overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b">
                    <div className="flex items-center gap-3">
                        <span className="text-sm font-medium">
                            Feature Match: {imageAName} ↔ {imageBName}
                        </span>
                        <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full font-mono">
                            {algorithm.toUpperCase()} · {data.matches.length} matches · {(data.score * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setShowKeypoints((v) => !v)}
                        >
                            {showKeypoints ? "Hide Keypoints" : "Show Keypoints"}
                        </Button>
                        <Button variant="ghost" size="sm" onClick={onClose}>
                            <X className="h-4 w-4" />
                        </Button>
                    </div>
                </div>

                {/* SVG Canvas */}
                <div className="p-4 overflow-auto bg-neutral-950 max-h-[80vh]">
                    <svg
                        width="100%"
                        viewBox={`0 0 ${totalW} ${totalH}`}
                        className="mx-auto"
                        style={{ maxHeight: "70vh" }}
                    >
                        {/* Image A */}
                        <image
                            href={toThumbnailUrl(imageAId, 800)}
                            x={0}
                            y={0}
                            width={DISPLAY_W}
                            height={hA}
                            preserveAspectRatio="xMidYMid slice"
                        />
                        {/* Image B */}
                        <image
                            href={toThumbnailUrl(imageBId, 800)}
                            x={offsetBX}
                            y={0}
                            width={DISPLAY_W}
                            height={hB}
                            preserveAspectRatio="xMidYMid slice"
                        />
                        {/* Border frames */}
                        <rect x={0} y={0} width={DISPLAY_W} height={hA} fill="none" stroke="white" strokeWidth={1} strokeOpacity={0.3} />
                        <rect x={offsetBX} y={0} width={DISPLAY_W} height={hB} fill="none" stroke="white" strokeWidth={1} strokeOpacity={0.3} />

                        {/* Match lines */}
                        {data.matches.map((m, i) => {
                            const kpA = data.image_a.keypoints[m.a_idx];
                            const kpB = data.image_b.keypoints[m.b_idx];
                            if (!kpA || !kpB) return null;
                            return (
                                <line
                                    key={i}
                                    x1={kpA.x * scaleA}
                                    y1={kpA.y * scaleA}
                                    x2={kpB.x * scaleB + offsetBX}
                                    y2={kpB.y * scaleB}
                                    stroke={matchColors[i]}
                                    strokeWidth={1.5}
                                    strokeOpacity={0.7}
                                >
                                    <title>Match #{i + 1} · dist: {m.distance.toFixed(1)}</title>
                                </line>
                            );
                        })}

                        {/* Keypoints */}
                        {showKeypoints && (
                            <>
                                {data.matches.map((m, i) => {
                                    const kpA = data.image_a.keypoints[m.a_idx];
                                    const kpB = data.image_b.keypoints[m.b_idx];
                                    if (!kpA || !kpB) return null;
                                    return (
                                        <g key={`kp-${i}`}>
                                            <circle cx={kpA.x * scaleA} cy={kpA.y * scaleA} r={3} fill={matchColors[i]} stroke="white" strokeWidth={0.5} />
                                            <circle cx={kpB.x * scaleB + offsetBX} cy={kpB.y * scaleB} r={3} fill={matchColors[i]} stroke="white" strokeWidth={0.5} />
                                        </g>
                                    );
                                })}
                            </>
                        )}

                        {/* Labels */}
                        <text x={DISPLAY_W / 2} y={totalH - 8} textAnchor="middle" fill="white" fontSize={12} fontFamily="system-ui" opacity={0.8}>
                            {imageAName}
                        </text>
                        <text x={offsetBX + DISPLAY_W / 2} y={totalH - 8} textAnchor="middle" fill="white" fontSize={12} fontFamily="system-ui" opacity={0.8}>
                            {imageBName}
                        </text>
                    </svg>
                </div>
            </div>
        </div>
    );
}
