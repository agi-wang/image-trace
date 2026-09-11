import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { API_BASE_URL, toThumbnailUrl, type HashType } from "@/lib/api";
import { APIError } from "@/lib/errorHandler";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Loader2, Printer, FileText } from "lucide-react";

interface ReportData {
    project: { id: number; name: string; description?: string };
    generated_at: string;
    algorithm: string;
    threshold: number;
    rotation_invariant: boolean;
    images: Array<{ id: number; filename: string; width?: number; height?: number; file_size?: number }>;
    matrix: { names: string[]; image_ids: number[]; values: number[][] };
    groups: Array<{
        group_id: number;
        similarity_score: number;
        images: Array<{ id: number; filename: string }>;
        pair_matches: Array<{
            image_a_id: number;
            image_b_id: number;
            score: number;
            matches: Array<{ a_idx: number; b_idx: number; distance: number }>;
            keypoints_a: Array<{ x: number; y: number }>;
            keypoints_b: Array<{ x: number; y: number }>;
            image_a_size?: { width: number; height: number };
            image_b_size?: { width: number; height: number };
        }>;
    }>;
    summary: { total_images: number; similar_groups: number; unique_images: number; duplicate_rate: number };
}

const ALGO_LABELS: Record<string, string> = {
    phash: "pHash", dhash: "dHash", ahash: "aHash", whash: "wHash",
    colorhash: "ColorHash", orb: "ORB", brisk: "BRISK", sift: "SIFT",
    akaze: "AKAZE", kaze: "KAZE", ssim: "SSIM", histogram: "Histogram",
    template: "Template", auto: "Auto (Hybrid)",
};

function getHeatColor(v: number): string {
    if (v >= 0.85) return `rgba(74, 222, 128, ${0.3 + v * 0.7})`;
    if (v >= 0.5) return `rgba(250, 204, 21, ${0.3 + v * 0.7})`;
    return `rgba(248, 113, 113, ${0.2 + v * 0.5})`;
}

export default function DuplicateReport() {
    const { projectId } = useParams<{ projectId: string }>();
    const navigate = useNavigate();
    const [data, setData] = useState<ReportData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [algo, setAlgo] = useState<HashType>("sift");
    const [threshold, setThreshold] = useState(0.85);
    const reportRef = useRef<HTMLDivElement>(null);

    const loadReport = async () => {
        setLoading(true);
        setError(null);
        try {
            const url = `${API_BASE_URL}/report/${projectId}?hash_type=${algo}&threshold=${threshold}`;
            const resp = await fetch(url);
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(err.detail || "Failed to generate report");
            }
            setData(await resp.json());
        } catch (e: any) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { loadReport(); }, [projectId]);

    const handlePrint = () => window.print();

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-center space-y-4">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto text-primary" />
                    <p className="text-muted-foreground">Generating report... Computing pairwise similarities and feature matches.</p>
                </div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-center space-y-4">
                    <p className="text-destructive text-lg">Error: {error}</p>
                    <Button onClick={() => navigate(-1)}>Go Back</Button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background">
            {/* Controls (hidden in print) */}
            <div className="print:hidden sticky top-0 z-10 bg-background/95 backdrop-blur border-b px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
                        <ArrowLeft className="h-4 w-4 mr-1" /> Back
                    </Button>
                    <span className="text-sm font-medium">Duplicate Detection Report</span>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={algo}
                        onChange={(e) => setAlgo(e.target.value as HashType)}
                        className="border rounded px-2 py-1 text-sm bg-background"
                    >
                        {Object.entries(ALGO_LABELS).map(([k, v]) => (
                            <option key={k} value={k}>{v}</option>
                        ))}
                    </select>
                    <input
                        type="number" min={0} max={1} step={0.05}
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="border rounded px-2 py-1 text-sm w-20 bg-background"
                    />
                    <Button size="sm" onClick={loadReport}>
                        <FileText className="h-4 w-4 mr-1" /> Regenerate
                    </Button>
                    <Button size="sm" variant="outline" onClick={handlePrint}>
                        <Printer className="h-4 w-4 mr-1" /> Print / PDF
                    </Button>
                </div>
            </div>

            {/* Report Content */}
            <div ref={reportRef} className="max-w-5xl mx-auto p-6 space-y-8 print:p-0 print:max-w-none">
                {/* Header */}
                <div className="border-b pb-6">
                    <h1 className="text-3xl font-bold mb-2">Image Duplicate Detection Report</h1>
                    <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                        <div>
                            <strong>Project:</strong> {data.project.name}
                            {data.project.description && <span className="ml-2">— {data.project.description}</span>}
                        </div>
                        <div><strong>Generated:</strong> {new Date(data.generated_at).toLocaleString()}</div>
                        <div><strong>Algorithm:</strong> {ALGO_LABELS[data.algorithm] || data.algorithm}</div>
                        <div><strong>Threshold:</strong> {data.threshold} · Rotation Invariant: {data.rotation_invariant ? "Yes" : "No"}</div>
                    </div>
                </div>

                {/* Summary Stats */}
                <div className="grid grid-cols-4 gap-4">
                    {[
                        { label: "Total Images", value: data.summary.total_images, color: "text-blue-500" },
                        { label: "Similar Groups", value: data.summary.similar_groups, color: "text-amber-500" },
                        { label: "Unique Images", value: data.summary.unique_images, color: "text-green-500" },
                        { label: "Duplicate Rate", value: `${data.summary.duplicate_rate}%`, color: data.summary.duplicate_rate > 30 ? "text-red-500" : "text-green-500" },
                    ].map((s) => (
                        <div key={s.label} className="border rounded-lg p-4 text-center">
                            <div className={`text-3xl font-bold ${s.color}`}>{s.value}</div>
                            <div className="text-xs text-muted-foreground mt-1">{s.label}</div>
                        </div>
                    ))}
                </div>

                {/* Image Gallery */}
                <section>
                    <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
                        <span className="w-1 h-5 bg-primary rounded-full inline-block" /> Image Overview
                    </h2>
                    <div className="grid grid-cols-6 gap-2 print:grid-cols-8">
                        {data.images.map((img) => (
                            <div key={img.id} className="text-center">
                                <img
                                    src={toThumbnailUrl(img.id, 200)}
                                    alt={img.filename}
                                    className="w-full aspect-square object-cover rounded border"
                                />
                                <div className="text-[10px] text-muted-foreground truncate mt-1">{img.filename}</div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Pairwise Matrix */}
                {data.matrix.values.length > 0 && data.matrix.values.length <= 20 && (
                    <section className="page-break-before">
                        <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
                            <span className="w-1 h-5 bg-primary rounded-full inline-block" /> Pairwise Similarity Matrix
                        </h2>
                        <div className="overflow-auto">
                            <table className="border-collapse text-xs w-full">
                                <thead>
                                    <tr>
                                        <th className="border p-1 bg-muted" />
                                        {data.matrix.names.map((name, i) => (
                                            <th key={i} className="border p-1 bg-muted font-medium" style={{ minWidth: 50 }}>
                                                {name.length > 10 ? name.slice(0, 8) + "…" : name}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.matrix.values.map((row, i) => (
                                        <tr key={i}>
                                            <td className="border p-1 bg-muted font-medium">{data.matrix.names[i]?.slice(0, 10)}</td>
                                            {row.map((v, j) => (
                                                <td
                                                    key={j}
                                                    className="border p-1 text-center font-mono"
                                                    style={{ backgroundColor: getHeatColor(v) }}
                                                >
                                                    {v.toFixed(2)}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </section>
                )}

                {/* Similar Groups with Feature Match Visualization */}
                {data.groups.length > 0 && (
                    <section className="page-break-before">
                        <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
                            <span className="w-1 h-5 bg-amber-500 rounded-full inline-block" /> Similar Image Groups ({data.groups.length})
                        </h2>
                        {data.groups.map((group) => (
                            <div key={group.group_id} className="border rounded-lg mb-6 overflow-hidden page-break-inside-avoid">
                                <div className="bg-muted px-4 py-2 flex items-center justify-between">
                                    <span className="text-sm font-medium">
                                        Group #{group.group_id} — {group.images.length} images
                                    </span>
                                    <span className="text-sm font-mono text-primary">
                                        Avg Similarity: {(group.similarity_score * 100).toFixed(1)}%
                                    </span>
                                </div>

                                {/* Group thumbnails */}
                                <div className="p-4 flex gap-3 flex-wrap">
                                    {group.images.map((img) => (
                                        <div key={img.id} className="text-center">
                                            <img
                                                src={toThumbnailUrl(img.id, 200)}
                                                alt={img.filename}
                                                className="w-24 h-24 object-cover rounded border"
                                            />
                                            <div className="text-[10px] text-muted-foreground truncate w-24 mt-1">{img.filename}</div>
                                        </div>
                                    ))}
                                </div>

                                {/* Per-pair feature match visualizations */}
                                {group.pair_matches.map((pair, pIdx) => (
                                    <div key={pIdx} className="border-t p-4">
                                        <div className="text-xs text-muted-foreground mb-2">
                                            Pair: #{pair.image_a_id} ↔ #{pair.image_b_id} · Score: {(pair.score * 100).toFixed(1)}% · {pair.matches.length} feature matches
                                        </div>
                                        {pair.matches.length > 0 && pair.image_a_size && pair.image_b_size ? (
                                            <MatchPairSVG pair={pair} />
                                        ) : (
                                            <div className="flex gap-3">
                                                <img src={toThumbnailUrl(pair.image_a_id, 300)} alt="" className="h-32 object-cover rounded border" />
                                                <img src={toThumbnailUrl(pair.image_b_id, 300)} alt="" className="h-32 object-cover rounded border" />
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        ))}
                    </section>
                )}

                {/* No duplicates found */}
                {data.groups.length === 0 && (
                    <div className="text-center py-12 text-muted-foreground">
                        <div className="text-4xl mb-3">✅</div>
                        <p className="text-lg font-medium">No duplicates found</p>
                        <p className="text-sm">All {data.summary.total_images} images are unique at threshold {data.threshold}</p>
                    </div>
                )}

                {/* Footer */}
                <div className="border-t pt-4 text-center text-xs text-muted-foreground print:mt-8">
                    <p>Generated by Image Trace v2.0 · {new Date(data.generated_at).toLocaleString()}</p>
                    <p>Algorithm: {ALGO_LABELS[data.algorithm] || data.algorithm} · Threshold: {data.threshold}</p>
                </div>
            </div>
        </div>
    );
}


/* ── Inline SVG for a pair of matched images ── */
function MatchPairSVG({ pair }: { pair: ReportData["groups"][0]["pair_matches"][0] }) {
    const W = 350;
    const sA = pair.image_a_size ? W / pair.image_a_size.width : 1;
    const sB = pair.image_b_size ? W / pair.image_b_size.width : 1;
    const hA = pair.image_a_size ? pair.image_a_size.height * sA : W;
    const hB = pair.image_b_size ? pair.image_b_size.height * sB : W;
    const gap = 16;
    const totalW = W * 2 + gap;
    const totalH = Math.max(hA, hB);

    return (
        <svg width="100%" viewBox={`0 0 ${totalW} ${totalH}`} style={{ maxHeight: 300 }}>
            <image href={toThumbnailUrl(pair.image_a_id, 700)} x={0} y={0} width={W} height={hA} preserveAspectRatio="xMidYMid slice" />
            <image href={toThumbnailUrl(pair.image_b_id, 700)} x={W + gap} y={0} width={W} height={hB} preserveAspectRatio="xMidYMid slice" />
            <rect x={0} y={0} width={W} height={hA} fill="none" stroke="white" strokeWidth={0.5} strokeOpacity={0.4} />
            <rect x={W + gap} y={0} width={W} height={hB} fill="none" stroke="white" strokeWidth={0.5} strokeOpacity={0.4} />

            {pair.matches.map((m, i) => {
                const a = pair.keypoints_a[m.a_idx];
                const b = pair.keypoints_b[m.b_idx];
                if (!a || !b) return null;
                const hue = (i * 137.508) % 360;
                const color = `hsl(${hue}, 80%, 60%)`;
                return (
                    <g key={i}>
                        <line
                            x1={a.x * sA} y1={a.y * sA}
                            x2={b.x * sB + W + gap} y2={b.y * sB}
                            stroke={color} strokeWidth={1} strokeOpacity={0.6}
                        />
                        <circle cx={a.x * sA} cy={a.y * sA} r={2.5} fill={color} stroke="white" strokeWidth={0.3} />
                        <circle cx={b.x * sB + W + gap} cy={b.y * sB} r={2.5} fill={color} stroke="white" strokeWidth={0.3} />
                    </g>
                );
            })}
        </svg>
    );
}
