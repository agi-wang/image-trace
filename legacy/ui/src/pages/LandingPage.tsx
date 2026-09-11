import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { useTranslation } from "react-i18next";

const LandingPage = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();

  const handleGetStarted = () => {
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen flex flex-col bg-background relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-primary/5 via-transparent to-transparent rounded-full blur-3xl animate-pulse" />
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-primary/5 via-transparent to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      </div>

      {/* Main content */}
      <main className="flex-1 flex items-center justify-center px-6">
        <div className="text-center max-w-2xl mx-auto relative z-10">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-foreground mb-6">
            {t("common.landingHero")}
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground mb-10">
            {t("common.landingSub")}
          </p>
          <Button
            size="lg"
            onClick={handleGetStarted}
            className="h-12 px-8 text-base font-medium gap-2 group"
          >
            {t("common.landingCta")}
            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </Button>
        </div>
      </main>

      {/* Minimal footer */}
      <footer className="py-6 text-center text-sm text-muted-foreground relative z-10">
        <p>© 2024 ImageTrace. {t("common.footerRights")}</p>
      </footer>
    </div>
  );
};

export default LandingPage;
