import { useEffect, useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { getProjects, deleteProject, type Project } from "@/lib/api";
import { copyErrorToClipboard, APIError } from "@/lib/errorHandler";
import { Button } from "@/components/ui/button";
import { CreateProjectDialog } from "@/components/CreateProjectDialog";
import { ProjectCard } from "@/components/ProjectCard";
import { SystemHealth } from "@/components/SystemHealth";
import { OnboardingTour } from "@/components/OnboardingTour";
import { useToast } from "@/hooks/use-toast";
import { Loader2, FolderOpen, Copy, RefreshCw } from "lucide-react";

export default function Dashboard() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const { toast } = useToast();
  const { t } = useTranslation();

  const loadProjects = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getProjects();
      setProjects(data);
    } catch (error) {
      const err = error as APIError;
      toast({
        title: t("common.loadFailed"),
        description: err.message || t("common.loadFailedDesc"),
        variant: "destructive",
        action: (
          <Button
            variant="secondary"
            size="sm"
            className="bg-white/10 hover:bg-white/20 text-white border-white/20"
            onClick={async () => {
              const success = await copyErrorToClipboard(err);
              if (success) {
                toast({ title: t("common.copy") });
              }
            }}
          >
            <Copy className="h-3 w-3 mr-1" />
            {t("common.copy")}
          </Button>
        ),
      });
    } finally {
      setLoading(false);
    }
  }, [toast, t]);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    const hasCompletedOnboarding = localStorage.getItem("onboarding-completed");
    if (!hasCompletedOnboarding) {
      setShowOnboarding(true);
    }
  }, []);

  const handleDelete = async (projectId: string) => {
    try {
      await deleteProject(projectId);
      toast({
        title: t("common.deleted"),
        description: t("common.deleteDesc"),
      });
      loadProjects();
    } catch (error) {
      const err = error as APIError;
      toast({
        title: t("common.deleteFailed"),
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
                toast({ title: t("common.copy") });
              }
            }}
          >
            <Copy className="h-3 w-3 mr-1" />
            {t("common.copy")}
          </Button>
        ),
      });
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {showOnboarding && (
        <OnboardingTour onComplete={() => setShowOnboarding(false)} />
      )}
      <header className="border-b bg-card">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground">{t("common.appName")}</h1>
              <p className="text-muted-foreground mt-1">
                {t("common.landingSub")}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={loadProjects}
                disabled={loading}
                className="gap-2"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                {t("common.refresh")}
              </Button>
              <div id="create-project-btn">
                <CreateProjectDialog onProjectCreated={loadProjects} />
              </div>

            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="mb-6">
          <SystemHealth autoRefresh refreshInterval={30000} />
        </div>
        {loading ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : projects.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-[400px] text-center">
            <FolderOpen className="h-16 w-16 text-muted-foreground/50 mb-4" />
            <h2 className="text-2xl font-semibold mb-2">{t("common.noProjects")}</h2>
            <p className="text-muted-foreground mb-6">{t("common.createFirst")}</p>
            <CreateProjectDialog onProjectCreated={loadProjects} />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onDelete={() => handleDelete(project.id)}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
