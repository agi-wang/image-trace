import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Plus, Copy } from "lucide-react";
import { createProject } from "@/lib/api";
import { copyErrorToClipboard, APIError } from "@/lib/errorHandler";
import { useToast } from "@/hooks/use-toast";

interface CreateProjectDialogProps {
  onProjectCreated: () => void;
}

export function CreateProjectDialog({ onProjectCreated }: CreateProjectDialogProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    setLoading(true);
    try {
      await createProject(name, description || undefined);
      toast({
        title: "项目已创建",
        description: "您的新项目已成功创建",
      });
      setOpen(false);
      setName("");
      setDescription("");
      onProjectCreated();
    } catch (error) {
      const err = error as APIError;
      toast({
        title: "创建失败",
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
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="gap-2">
          <Plus className="h-4 w-4" />
          新建项目
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>创建新项目</DialogTitle>
            <DialogDescription>
              为您的图片分析创建一个新项目
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">项目名称 *</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="输入项目名称"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">描述（可选）</Label>
              <Textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="描述项目用途或备注"
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="submit" disabled={loading || !name.trim()}>
              {loading ? "创建中..." : "创建项目"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
