import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Zap, Cpu, Layers, Play, Eye, BarChart3, Blend, type LucideIcon } from "lucide-react";
import { HashType } from "@/lib/api";

interface AnalysisPanelProps {
  hasImages: boolean;
  loading?: boolean;
  onAnalyze: (hashType: HashType, rotationInvariant: boolean) => void;
}

interface AlgoItem {
  value: HashType;
  label: string;
  icon: LucideIcon;
}

const allAlgos: { tier: string; algos: AlgoItem[] }[] = [
  {
    tier: "Hash",
    algos: [
      { value: "phash", label: "pHash", icon: Zap },
      { value: "dhash", label: "dHash", icon: Zap },
      { value: "ahash", label: "aHash", icon: Zap },
      { value: "whash", label: "wHash", icon: Zap },
    ],
  },
  {
    tier: "Pixel",
    algos: [
      { value: "ssim", label: "SSIM", icon: Eye },
      { value: "histogram", label: "Histogram", icon: BarChart3 },
    ],
  },
  {
    tier: "Descriptor",
    algos: [
      { value: "sift", label: "SIFT", icon: Layers },
      { value: "orb", label: "ORB", icon: Cpu },
      { value: "brisk", label: "BRISK", icon: Cpu },
      { value: "akaze", label: "AKAZE", icon: Layers },
      { value: "kaze", label: "KAZE", icon: Layers },
    ],
  },
];

export function AnalysisPanel({ hasImages, onAnalyze, loading }: AnalysisPanelProps) {
  const [rotationInvariant, setRotationInvariant] = useState(false);

  const handleFullAnalysis = () => {
    // Trigger full analysis with "auto" — parent handles running all algos
    onAnalyze("auto" as HashType, rotationInvariant);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Blend className="h-5 w-5 text-primary" />
          Full Image Analysis
        </CardTitle>
        <CardDescription>
          One-click analysis across all 11 algorithms (pre-computed feature matrix).
          Results are instant when features are ready.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Algorithm Overview */}
        <div className="grid grid-cols-3 gap-3">
          {allAlgos.map((tier) => (
            <div key={tier.tier} className="rounded-lg border bg-muted/20 p-3">
              <p className="text-xs font-semibold text-muted-foreground mb-2">{tier.tier}</p>
              <div className="flex flex-wrap gap-1.5">
                {tier.algos.map((algo) => {
                  const Icon = algo.icon;
                  return (
                    <span
                      key={algo.value}
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-background border text-xs font-medium"
                    >
                      <Icon className="h-3 w-3 text-primary/70" />
                      {algo.label}
                    </span>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Rotation Invariance Toggle */}
        <div className="flex items-center justify-between rounded-lg border p-3 bg-muted/30">
          <div>
            <Label htmlFor="rotation-invariant" className="font-medium cursor-pointer">
              Rotation Invariant
            </Label>
            <p className="text-xs text-muted-foreground mt-0.5">
              Test 8 orientations (4 rotations × 2 flips), take highest score
            </p>
          </div>
          <Switch
            id="rotation-invariant"
            checked={rotationInvariant}
            onCheckedChange={setRotationInvariant}
          />
        </div>

        {/* Full Analysis Button */}
        <Button
          onClick={handleFullAnalysis}
          disabled={!hasImages || loading}
          className="w-full gap-2 bg-gradient-to-r from-blue-600 to-emerald-500 hover:from-blue-700 hover:to-emerald-600"
          size="lg"
        >
          <Play className="h-4 w-4" />
          {loading ? "Analyzing..." : "Analysis"}
        </Button>

        {!hasImages && (
          <p className="text-sm text-center text-muted-foreground">
            Upload at least 2 images to start analysis
          </p>
        )}
      </CardContent>
    </Card>
  );
}
