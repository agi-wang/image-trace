import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface SimilarityMatrixProps {
  matrix: number[][];
  imageNames?: string[];
  images?: Array<{ filename?: string; id?: number }>;
}

export function SimilarityMatrix({ matrix, imageNames = [], images }: SimilarityMatrixProps) {
  // Support both imageNames and images props
  const names = imageNames.length > 0
    ? imageNames
    : (images || []).map((img) => img.filename || `#${img.id}`);
  if (!matrix || matrix.length === 0) {
    return null;
  }

  const getColorForValue = (value: number) => {
    // 相似度越高，颜色越深（蓝绿色）
    const intensity = Math.round(value * 255);
    return `rgb(${255 - intensity}, ${255 - Math.round(intensity * 0.3)}, 255)`;
  };

  const size = matrix.length;
  const cellSize = size === 1 ? 200 : Math.min(80, 400 / size);

  return (
    <Card>
      <CardHeader>
        <CardTitle>相似度矩阵</CardTitle>
        <CardDescription>
          图片之间的相似度热力图 (1.0 = 完全相同, 0.0 = 完全不同)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-auto">
          <div className="inline-block min-w-full">
            <table className="border-collapse">
              <thead>
                <tr>
                  <th className="border border-border p-2 bg-muted"></th>
                  {matrix.map((_, i) => (
                    <th
                      key={i}
                      className="border border-border p-2 bg-muted text-xs font-medium"
                      style={{ minWidth: cellSize }}
                    >
                      {names[i] || `Image ${i + 1}`}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map((row, i) => (
                  <tr key={i}>
                    <td className="border border-border p-2 bg-muted text-xs font-medium">
                      {names[i] || `Image ${i + 1}`}
                    </td>
                    {row.map((value, j) => (
                      <td
                        key={j}
                        className="border border-border p-2 text-center text-xs font-mono relative group"
                        style={{
                          backgroundColor: getColorForValue(value),
                          minWidth: cellSize,
                          height: cellSize,
                        }}
                      >
                        <span className="relative z-10 font-semibold drop-shadow-sm">
                          {value.toFixed(2)}
                        </span>
                        <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-10 transition-opacity" />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="mt-4 flex items-center justify-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="w-8 h-4 rounded" style={{ backgroundColor: getColorForValue(0) }} />
            <span>0.0 (不同)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-4 rounded" style={{ backgroundColor: getColorForValue(0.5) }} />
            <span>0.5</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-4 rounded" style={{ backgroundColor: getColorForValue(1) }} />
            <span>1.0 (相同)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
