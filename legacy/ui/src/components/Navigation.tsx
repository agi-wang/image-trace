import { Button } from "@/components/ui/button";
import { useNavigate, useLocation } from "react-router-dom";
import { Menu, X, Image as ImageIcon, LayoutDashboard } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

export function Navigation() {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { t, i18n } = useTranslation();

  const isDashboard = location.pathname === "/dashboard" || location.pathname.startsWith("/project/");

  const toggleLanguage = () => {
    const next = i18n.language === "zh" ? "en" : "zh";
    i18n.changeLanguage(next);
  };

  return (
    <nav className="sticky top-0 z-50 backdrop-blur-lg bg-background/80 border-b border-border">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
              <ImageIcon className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-bold text-lg text-foreground">{t("common.appName")}</span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-6">
            <button
              onClick={() => navigate("/dashboard")}
              className={`text-sm font-medium transition-colors hover:text-primary ${isDashboard ? "text-primary" : "text-muted-foreground"
                }`}
            >
              {t("common.dashboard")}
            </button>
            <button
              onClick={() => navigate("/demo")}
              className={`text-sm font-medium transition-colors hover:text-primary ${location.pathname === "/demo" ? "text-primary" : "text-muted-foreground"
                }`}
            >
              {t("common.demo")}
            </button>
          </div>

          {/* Desktop Buttons */}
          <div className="hidden md:flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/dashboard")}
            >
              <LayoutDashboard className="w-4 h-4 mr-2" />
              {t("common.dashboard")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleLanguage}
            >
              {i18n.language === "zh" ? "EN" : "中文"}
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <X className="w-6 h-6 text-foreground" />
            ) : (
              <Menu className="w-6 h-6 text-foreground" />
            )}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-border">
            <div className="flex flex-col gap-2">
              <button
                onClick={() => { navigate("/demo"); setMobileMenuOpen(false); }}
                className="px-4 py-2 text-left text-foreground hover:bg-muted rounded-md"
              >
                {t("common.demo")}
              </button>
              <button
                onClick={() => { navigate("/dashboard"); setMobileMenuOpen(false); }}
                className="px-4 py-2 text-left text-foreground hover:bg-muted rounded-md"
              >
                {t("common.dashboard")}
              </button>
              <div className="border-t border-border my-2" />
              <Button
                variant="ghost"
                size="sm"
                className="mx-4 mt-2"
                onClick={() => { toggleLanguage(); setMobileMenuOpen(false); }}
              >
                {i18n.language === "zh" ? "EN" : "中文"}
              </Button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
