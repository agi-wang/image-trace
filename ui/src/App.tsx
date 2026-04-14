import { Suspense, lazy } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { BackendGate } from "@/components/BackendGate";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import LandingPage from "./pages/LandingPage";
import Dashboard from "./pages/Dashboard";
import ProjectDetail from "./pages/ProjectDetail";
import NotFound from "./pages/NotFound";

const Demo = lazy(() => import("./pages/Demo"));
const DuplicateReport = lazy(() => import("./pages/DuplicateReport"));

const App = () => {
  useTranslation();

  return (
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <ErrorBoundary>
        <BackendGate>
          <HashRouter>
            <Routes>
              <Route path="/landing" element={<LandingPage />} />
              <Route path="/demo" element={<Suspense><Demo /></Suspense>} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/project/:projectId" element={<ProjectDetail />} />
              <Route path="/report/:projectId" element={<Suspense><DuplicateReport /></Suspense>} />
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </HashRouter>
        </BackendGate>
      </ErrorBoundary>
    </TooltipProvider>
  );
};

export default App;
