import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Image as ImageType, HashType, visualizeMatch } from "@/lib/api";
import { Loader2 } from "lucide-react";

interface Edge {
  source: number;
  target: number;
  score: number;
}

interface SimilarityGraphProps {
  groups: Array<{
    group_id: number;
    similarity_score: number;
    images: ImageType[];
  }>;
  imageIds?: number[];
  matchAlgo?: HashType;
  onEdgeClick?: (imgAId: number, imgBId: number) => void;
  maxEdges?: number;
}

export function SimilarityGraph({ groups, imageIds, matchAlgo, onEdgeClick, maxEdges = 80 }: SimilarityGraphProps) {
  const [hoveredEdge, setHoveredEdge] = useState<number | null>(null);
  const [loadingEdge, setLoadingEdge] = useState<number | null>(null);

  const nodesMap = new Map<number, ImageType>();
  const edges: Edge[] = [];

  groups.forEach((g) => {
    g.images.forEach((img) => {
      nodesMap.set(img.id, img);
    });
    for (let i = 0; i < g.images.length; i++) {
      for (let j = i + 1; j < g.images.length; j++) {
        edges.push({
          source: g.images[i].id,
          target: g.images[j].id,
          score: g.similarity_score,
        });
      }
    }
  });

  const nodes = Array.from(nodesMap.values());
  if (nodes.length === 0) return null;

  edges.sort((a, b) => b.score - a.score);
  const clippedEdges = edges.slice(0, maxEdges);

  // Layout
  const width = 900;
  const height = 520;
  const radius = Math.min(width, height) / 2 - 80;
  const centerX = width / 2;
  const centerY = height / 2;

  const positioned = nodes.map((n, idx) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * idx) / nodes.length;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    return { ...n, x, y };
  });

  const posMap = new Map<number, { x: number; y: number }>();
  positioned.forEach((p) => posMap.set(p.id, { x: p.x!, y: p.y! }));

  const getColor = (score: number) => {
    if (score >= 0.85) return "rgb(74, 222, 128)";  // green
    if (score >= 0.60) return "rgb(250, 204, 21)";   // yellow
    return "rgb(248, 113, 113)";                     // red
  };

  const handleEdgeClick = async (edge: Edge, idx: number) => {
    if (!onEdgeClick) return;
    setLoadingEdge(idx);
    try {
      await onEdgeClick(edge.source, edge.target);
    } finally {
      setLoadingEdge(null);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Similarity Network Graph</CardTitle>
        <CardDescription>
          Nodes = images, edges = similarity. Click an edge to view feature matches.
          {loadingEdge !== null && (
            <span className="ml-2 inline-flex items-center gap-1 text-primary">
              <Loader2 className="h-3 w-3 animate-spin" /> Generating match visualization...
            </span>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="w-full overflow-auto">
          <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img">
            {/* Edges */}
            {clippedEdges.map((e, idx) => {
              const a = posMap.get(e.source);
              const b = posMap.get(e.target);
              if (!a || !b) return null;
              const strokeWidth = 1 + e.score * 6;
              const isHovered = hoveredEdge === idx;
              const mx = (a.x + b.x) / 2;
              const my = (a.y + b.y) / 2;

              return (
                <g
                  key={idx}
                  onMouseEnter={() => setHoveredEdge(idx)}
                  onMouseLeave={() => setHoveredEdge(null)}
                  onClick={() => handleEdgeClick(e, idx)}
                  style={{ cursor: onEdgeClick ? "pointer" : "default" }}
                >
                  {/* Hit area */}
                  <line
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke="transparent" strokeWidth={Math.max(12, strokeWidth + 6)}
                  />
                  {/* Visible line */}
                  <line
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={getColor(e.score)}
                    strokeWidth={isHovered ? strokeWidth + 2 : strokeWidth}
                    strokeOpacity={isHovered ? 1 : 0.7}
                  />
                  {/* Score label */}
                  {(isHovered || clippedEdges.length <= 15) && (
                    <g transform={`translate(${mx}, ${my})`}>
                      <rect x={-18} y={-10} width={36} height={18} rx={4} fill="rgba(0,0,0,0.75)" />
                      <text
                        x={0} y={4}
                        textAnchor="middle"
                        fill={getColor(e.score)}
                        fontSize={11}
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        {(e.score * 100).toFixed(0)}%
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            {/* Nodes */}
            {positioned.map((n) => (
              <g key={n.id} transform={`translate(${n.x}, ${n.y})`}>
                <circle r={24} fill="#0ea5e9" fillOpacity={0.1} stroke="#0ea5e9" strokeWidth={1.5} />
                <image
                  href={n.public_url || "/placeholder.svg"}
                  x={-18} y={-18} width={36} height={36}
                  preserveAspectRatio="xMidYMid slice"
                  clipPath="circle(18px at 18px 18px)"
                />
                {/* Filename label below */}
                <text
                  y={32} textAnchor="middle"
                  fill="currentColor" fontSize={9}
                  fontFamily="system-ui"
                  opacity={0.8}
                >
                  {n.filename.length > 16 ? n.filename.slice(0, 14) + "…" : n.filename}
                </text>
                <title>{n.filename}</title>
              </g>
            ))}

            {/* Legend */}
            {[
              { label: "≥85% Match", color: "rgb(74, 222, 128)", y: height - 50 },
              { label: "60-84%", color: "rgb(250, 204, 21)", y: height - 34 },
              { label: "<60%", color: "rgb(248, 113, 113)", y: height - 18 },
            ].map((item) => (
              <g key={item.label}>
                <line x1={12} y1={item.y} x2={40} y2={item.y} stroke={item.color} strokeWidth={3} />
                <text x={46} y={item.y + 4} fill="currentColor" fontSize={10} opacity={0.7}>{item.label}</text>
              </g>
            ))}
          </svg>
        </div>
      </CardContent>
    </Card>
  );
}
