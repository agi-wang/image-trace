import { useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";

import { Navigation } from "@/components/Navigation";
import {
  Upload,
  X,
  Loader2,
  ArrowRight,
  Lock,
  Image as ImageIcon,
  Sparkles
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface UploadedImage {
  file: File;
  preview: string;
}

const Demo = () => {
  const navigate = useNavigate();

  const { toast } = useToast();
  const { t } = useTranslation();
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<{ similarity: number } | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    const newImages = acceptedFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file)
    }));
    setImages(prev => [...prev, ...newImages]);
    setResult(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    disabled: false
  });

  const removeImage = (index: number) => {
    setImages(prev => {
      const newImages = [...prev];
      URL.revokeObjectURL(newImages[index].preview);
      newImages.splice(index, 1);
      return newImages;
    });
    setResult(null);
  };

  const analyzeImages = async () => {
    if (images.length < 2) {
      toast({
        title: t("demo.needMoreTitle"),
        description: t("demo.needMoreDesc"),
        variant: "destructive"
      });
      return;
    }

    setAnalyzing(true);

    // Simulate analysis (in real app, this would call the backend)
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Generate a random similarity score for demo
    const similarity = Math.random() * 0.4 + 0.3; // 30-70% similarity
    setResult({ similarity });
    setAnalyzing(false);
  };

  const getSimilarityColor = (value: number) => {
    if (value >= 0.7) return "text-destructive";
    if (value >= 0.5) return "text-accent";
    return "text-success";
  };

  const getSimilarityLabel = (value: number) => {
    if (value >= 0.7) return t("demo.similarityHigh");
    if (value >= 0.5) return t("demo.similarityMedium");
    return t("demo.similarityLow");
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-10">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/10 text-secondary text-sm font-medium mb-4">
              <Sparkles className="w-4 h-4" />
              <span>{t("demo.badge")}</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              {t("demo.title")}
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              {t("demo.subtitle")}
            </p>
          </div>

          {/* Upload Area */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                {t("demo.uploadTitle")}
              </CardTitle>
              <CardDescription>{t("demo.uploadDesc")}</CardDescription>
            </CardHeader>
            <CardContent>
              {/* Image Previews */}
              {images.length > 0 && (
                <div className="grid grid-cols-2 gap-4 mb-6">
                  {images.map((img, index) => (
                    <div key={index} className="relative group">
                      <div className="aspect-square rounded-lg overflow-hidden bg-muted border border-border">
                        <img
                          src={img.preview}
                          alt={`Image ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <button
                        onClick={() => removeImage(index)}
                        className="absolute top-2 right-2 p-1.5 rounded-full bg-background/80 hover:bg-background shadow-sm opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <X className="w-4 h-4 text-foreground" />
                      </button>
                      <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-background/80 text-xs font-medium">
                        {t("demo.imageLabel", { index: index + 1 })}
                      </div>
                    </div>
                  ))}

                  {/* Add more */}
                  <div
                    {...getRootProps()}
                    className="aspect-square rounded-lg border-2 border-dashed border-border flex items-center justify-center cursor-pointer hover:border-primary/50 hover:bg-muted/50 transition-colors"
                  >
                    <input {...getInputProps()} />
                    <div className="text-center">
                      <ImageIcon className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                      <span className="text-sm text-muted-foreground">{t("demo.addMore")}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Dropzone */}
              {images.length === 0 && (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragActive
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50 hover:bg-muted/50"
                    }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium text-foreground mb-2">
                    {isDragActive ? t("demo.dropHintActive") : t("demo.dropHintIdle")}
                  </p>
                  <p className="text-muted-foreground">
                    {t("demo.dropHintSub")}
                  </p>
                </div>
              )}

              {/* Analyze Button */}
              {images.length >= 2 && (
                <Button
                  onClick={analyzeImages}
                  disabled={analyzing}
                  className="w-full mt-4"
                  size="lg"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      {t("demo.analyzing")}
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5 mr-2" />
                      {t("demo.analyze")}
                    </>
                  )}
                </Button>
              )}
            </CardContent>
          </Card>

          {/* Results */}
          {result && (
            <Card className="mb-8 border-primary/20">
              <CardHeader>
                <CardTitle>{t("demo.resultTitle")}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-6">
                  <div className={`text-6xl font-bold mb-2 ${getSimilarityColor(result.similarity)}`}>
                    {(result.similarity * 100).toFixed(1)}%
                  </div>
                  <div className={`text-lg font-medium ${getSimilarityColor(result.similarity)}`}>
                    {getSimilarityLabel(result.similarity)}
                  </div>
                  <p className="text-muted-foreground mt-4">
                    {t("demo.resultDesc", { percent: (result.similarity * 100).toFixed(1) })}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}



          {/* Go to Dashboard after results */}
          {result && (
            <Card className="bg-gradient-to-br from-success/5 to-primary/5 border-success/20">
              <CardContent className="p-8">
                <div className="flex flex-col md:flex-row items-center gap-6">
                  <div className="flex-1 text-center md:text-left">
                    <h3 className="text-xl font-semibold text-foreground mb-2">
                      {t("demo.loggedTitle")}
                    </h3>
                    <p className="text-muted-foreground">
                      {t("demo.loggedDesc")}
                    </p>
                  </div>
                  <Button
                    size="lg"
                    onClick={() => navigate("/")}
                  >
                    {t("demo.goDashboard")}
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default Demo;
