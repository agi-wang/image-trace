import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { X, ArrowRight, ArrowLeft, FolderPlus, Upload, BarChart3, Sparkles } from "lucide-react";

interface Step {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  target?: string;
}

const steps: Step[] = [
  {
    id: "welcome",
    title: "欢迎使用图片溯源分析系统",
    description: "让我们快速了解如何使用这个平台来分析和追踪图片来源。只需几个简单步骤，您就能开始使用了！",
    icon: <Sparkles className="h-8 w-8 text-primary" />,
  },
  {
    id: "create-project",
    title: "创建您的第一个项目",
    description: "点击「新建项目」按钮来创建一个新的分析项目。每个项目可以包含多张需要分析的图片。",
    icon: <FolderPlus className="h-8 w-8 text-primary" />,
    target: "create-project-btn",
  },
  {
    id: "upload-images",
    title: "上传图片",
    description: "在项目中上传您想要分析的图片。系统支持批量上传，支持 JPG、PNG 等常见格式。",
    icon: <Upload className="h-8 w-8 text-primary" />,
  },
  {
    id: "view-results",
    title: "查看分析结果",
    description: "系统会自动分析图片之间的相似度，生成可视化的相似度矩阵，帮助您快速发现相似或重复的图片。",
    icon: <BarChart3 className="h-8 w-8 text-primary" />,
  },
];

interface OnboardingTourProps {
  onComplete: () => void;
}

export function OnboardingTour({ onComplete }: OnboardingTourProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleComplete = () => {
    setIsVisible(false);
    localStorage.setItem("onboarding-completed", "true");
    onComplete();
  };

  const handleSkip = () => {
    handleComplete();
  };

  useEffect(() => {
    const target = steps[currentStep].target;
    if (target) {
      const element = document.getElementById(target);
      if (element) {
        element.classList.add("ring-2", "ring-primary", "ring-offset-2", "z-50", "relative");
        return () => {
          element.classList.remove("ring-2", "ring-primary", "ring-offset-2", "z-50", "relative");
        };
      }
    }
  }, [currentStep]);

  if (!isVisible) return null;

  const step = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isFirstStep = currentStep === 0;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40" />

      {/* Tour Card */}
      <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
        <Card className="w-full max-w-md shadow-2xl border-primary/20 animate-scale-in">
          <CardContent className="p-6">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  {step.icon}
                </div>
                <div className="text-sm text-muted-foreground">
                  步骤 {currentStep + 1} / {steps.length}
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 -mr-2 -mt-2"
                onClick={handleSkip}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Content */}
            <div className="space-y-3 mb-6">
              <h3 className="text-xl font-semibold text-foreground">
                {step.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {step.description}
              </p>
            </div>

            {/* Progress Dots */}
            <div className="flex justify-center gap-2 mb-6">
              {steps.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentStep(index)}
                  className={`h-2 rounded-full transition-all ${
                    index === currentStep
                      ? "w-6 bg-primary"
                      : index < currentStep
                      ? "w-2 bg-primary/50"
                      : "w-2 bg-muted"
                  }`}
                />
              ))}
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between">
              <Button
                variant="ghost"
                onClick={handleSkip}
                className="text-muted-foreground"
              >
                跳过引导
              </Button>

              <div className="flex gap-2">
                {!isFirstStep && (
                  <Button variant="outline" onClick={handlePrev}>
                    <ArrowLeft className="h-4 w-4 mr-1" />
                    上一步
                  </Button>
                )}
                <Button onClick={handleNext}>
                  {isLastStep ? (
                    "开始使用"
                  ) : (
                    <>
                      下一步
                      <ArrowRight className="h-4 w-4 ml-1" />
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
