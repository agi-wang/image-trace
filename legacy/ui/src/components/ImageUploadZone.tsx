import { useCallback } from "react";
import * as React from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Image as ImageIcon, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { uploadImages, type Image } from "@/lib/api";
import { copyErrorToClipboard, APIError } from "@/lib/errorHandler";
import { useToast } from "@/hooks/use-toast";

interface ImageUploadZoneProps {
  projectId: string;
  onImagesUploaded: (images: Image[]) => void;
}

export function ImageUploadZone({ projectId, onImagesUploaded }: ImageUploadZoneProps) {
  const [uploading, setUploading] = React.useState(false);
  const { toast } = useToast();

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      setUploading(true);
      try {
        const uploadedImages = await uploadImages(projectId, acceptedFiles);
        toast({
          title: "上传成功",
          description: `成功上传 ${acceptedFiles.length} 张图片`,
        });
        onImagesUploaded(uploadedImages);
      } catch (error) {
        const err = error as APIError;
        toast({
          title: "上传失败",
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
                  toast({ title: "已复制错误详情" });
                }
              }}
            >
              <Copy className="h-3 w-3 mr-1" />
              复制
            </Button>
          ),
        });
      } finally {
        setUploading(false);
      }
    },
    [projectId, onImagesUploaded, toast]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".gif", ".webp"],
    },
    multiple: true,
    disabled: uploading,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all",
        isDragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/50",
        uploading && "opacity-50 cursor-not-allowed"
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-3">
        {uploading ? (
          <>
            <Upload className="h-12 w-12 text-primary animate-pulse" />
            <p className="text-sm text-muted-foreground">上传中...</p>
          </>
        ) : (
          <>
            <ImageIcon className="h-12 w-12 text-muted-foreground" />
            <div>
              <p className="text-sm font-medium">
                {isDragActive ? "放开以上传图片" : "拖拽图片到此处或点击上传"}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                支持批量上传，格式：PNG、JPG、JPEG、GIF、WEBP
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
