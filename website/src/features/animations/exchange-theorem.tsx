"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { exchangeNodes, exchangeEdges, exchangeSteps, exchangePaths, initialMatching, initialDrivers } from "./data/exchangeData";
import { APP_CONFIG } from "@/lib/config";
import { useVisualizationSettings } from "@/providers/visualization-settings-provider";
import { Card, FormatMathText, KeyEdge, KeyNode, Stepper, SurfaceTitle } from "@/components/ui/core";
import { DirectedNetwork } from "./network/directedNetwork";
import { BipartiteNetwork } from "./network/bipartiteNetwork";
import { NetworkNode, ExchangePresentation, ExchangePhase } from "./network/types";
import { cn } from "@/lib/utils";
import { useI18n } from "@/providers/i18n-provider";

export default function ExchangeTheorem() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<{ id: number; side: "source" | "target" | "both" } | null>(null);

  // New hierarchical state
  const [activeS, setActiveS] = useState(2);
  const [activeTemplateIdx, setActiveTemplateIdx] = useState(0);
  const [activeTargetIdx, setActiveTargetIdx] = useState(0);

  const { t } = useI18n();
  const { styleConfig } = useVisualizationSettings();
  const directedStyleConfig = useMemo(
    () => ({
      ...styleConfig,
      nodes: {
        ...styleConfig.nodes,
        all: {
          ...styleConfig.nodes.all,
          textFontSize: APP_CONFIG.visual.network.text_size
        }
      }
    }),
    [styleConfig]
  );

  const matchingEdgeColor = styleConfig.edges.matching.color;
  const nonMatchingEdgeColor = styleConfig.edges.nonMatching.color;
  const altNonMatchedEdgeColor = styleConfig.edges.alternativeNonMatching.color;
  const altMatchedEdgeColor = styleConfig.edges.alternativeMatching.color;
  const hoverEdgeColor =
    styleConfig.edges.hover.color === "no-change" ? styleConfig.edges.matching.color : styleConfig.edges.hover.color;
  const driverNodeColor = styleConfig.nodes.driver.strokeColor;
  const matchedNodeColor = styleConfig.nodes.matching.strokeColor;

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => {
      setStep((current) => {
        if (current === 3) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 2000);

    return () => window.clearInterval(timer);
  }, [playing]);

  // Utility to format internal template format (<-/->) to Display LaTeX (\leftarrow/\rightarrow)
  const formatArrowsForDisplay = (str: string) => {
    return str
      .replace(/<-/g, "\\leftarrow ")
      .replace(/->/g, "\\rightarrow ");
  };

  // Utility to parse description from template by truncating at target t
  const parseDescriptionFromTemplate = (template: string, targetT: number) => {
    // Identify truncation point on raw ASCII template
    // Correctly escape ^ for the superscript match
    const targetPattern = new RegExp(`${targetT}(\\^[-+])?`);
    const match = template.match(targetPattern);

    let part = (match && match.index !== undefined)
      ? template.substring(0, match.index + match[0].length)
      : template;

    // Ensure the truncated LaTeX string is closed
    if (template.startsWith("$") && part !== template && !part.endsWith("$")) {
      part += "$";
    }

    return formatArrowsForDisplay(part);
  };

  // Derive the currently selected path details
  const availableTemplates = useMemo(() => {
    return exchangePaths.map((p, idx) => ({ ...p, originalIdx: idx })).filter(p => p.s === activeS);
  }, [activeS]);

  const safeTemplateIdx = Math.min(activeTemplateIdx, availableTemplates.length - 1);
  const currentTemplate = availableTemplates[safeTemplateIdx] || availableTemplates[0];

  const safeTargetIdx = Math.min(activeTargetIdx, (currentTemplate?.targets.length || 1) - 1);
  const currentTarget = currentTemplate?.targets[safeTargetIdx] || currentTemplate?.targets[0];

  const selectedS = activeS;
  const selectedT = currentTarget?.t || 0;
  const selectedEdges = currentTarget?.edges || [];
  const pathKey = `${selectedS}-${safeTemplateIdx}-${safeTargetIdx}`;
  const pathSelectionUnlocked = step >= 1;
  const targetSelectionUnlocked = step >= 1;
  const stripedNodeIds = useMemo(() => {
    const ids = new Set<number>();
    if (selectedS) ids.add(selectedS);
    if (selectedT && step >= 1) ids.add(selectedT);
    return Array.from(ids);
  }, [selectedS, selectedT, step]);

  useEffect(() => {
    const prevStep = prevStepRef.current;

    if (step !== 2) {
      setExchangePhase(null);
      prevStepRef.current = step;
      return;
    }

    // 只有从 1 -> 2 才播放平移动画
    if (prevStep === 1) {
      setExchangePhase("animating");

      const timer = window.setTimeout(() => {
        setExchangePhase("preview");
      }, 850);

      prevStepRef.current = step;
      return () => window.clearTimeout(timer);
    }

    // 例如 3 -> 2，直接进入 preview，不播放平移
    setExchangePhase("preview");
    prevStepRef.current = step;
  }, [step, pathKey]);

  // Dynamic description generation
  const selectedDescription = useMemo(() => {
    if (!currentTemplate || !selectedT) return "";
    return parseDescriptionFromTemplate(currentTemplate.template, selectedT);
  }, [currentTemplate, selectedT]);

  // Helper to hydrate templates from exchangeData.ts
  const hydrateBody = (template: string, vars: { s: number; t: number; drivers: number[]; path: string; initialDrivers: number[] }) => {
    return template
      .replace(/{s}/g, vars.s.toString())
      .replace(/{t}/g, vars.t.toString())
      .replace(/{path}/g, vars.path)
      .replace(/{drivers}/g, vars.drivers.join(", "))
      .replace(/{initialDrivers}/g, vars.initialDrivers.join(", "));
  };

  // Dynamically generate steps based on selected path
  const dynamicSteps = useMemo(() => {
    const toggleMatching = (m: [number, number][], p: [number, number][]) => {
      let result = [...m.map(e => [...e] as [number, number])];
      for (const [pf, pt] of p) {
        const realIdx = result.findIndex(e => (e[0] === pf && e[1] === pt) || (e[1] === pf && e[0] === pt));
        if (realIdx > -1) result.splice(realIdx, 1);
        else result.push([pf, pt]);
      }
      return result;
    };

    const finalMatching = toggleMatching(initialMatching, selectedEdges);
    const finalDrivers = [...initialDrivers.filter(id => id !== selectedS)];
    if (!finalDrivers.includes(selectedT) && selectedT !== 0) {
      finalDrivers.push(selectedT);
    }
    finalDrivers.sort((a, b) => a - b);

    const commonVars = {
      s: selectedS,
      t: selectedT,
      path: selectedDescription,
      initialDrivers: initialDrivers
    };

    return exchangeSteps.map((stepInfo, idx) => {
      // Steps 0 and 1 use initial state.
      // Steps 2 and 3 use final state (post-symmetric difference).
      const isPostSymmetric = idx >= 3;
      const displayDrivers = isPostSymmetric ? finalDrivers : initialDrivers;
      const displayMatching = isPostSymmetric ? finalMatching : initialMatching;
      const title = t(stepInfo.titleKey as Parameters<typeof t>[0]);
      const bodyTemplate = t(stepInfo.bodyKey as Parameters<typeof t>[0]);

      return {
        title,
        body: hydrateBody(bodyTemplate, { ...commonVars, drivers: displayDrivers }),
        matching: displayMatching,
        highlightEdges: (idx === 1 || idx === 2) ? selectedEdges : [],
        drivers: displayDrivers
      };
    });
  }, [selectedS, selectedT, selectedEdges, selectedDescription, t]);

  const currentStep = dynamicSteps[step] || dynamicSteps[0];

  const positions: NetworkNode[] = useMemo(() => {
    const sorted = [...exchangeNodes].sort((a, b) => a.id - b.id);
    const layout = styleConfig.layout.orientation;
    const margin = APP_CONFIG.visual.network.layout.margin;
    const span = APP_CONFIG.visual.network.layout.span;
    const y_source = APP_CONFIG.visual.network.layout.y_source;
    const y_target = APP_CONFIG.visual.network.layout.y_target;

    return exchangeNodes.map(node => {
      const index = sorted.findIndex(n => n.id === node.id);
      const x = margin + (index * span) / (sorted.length - 1);
      const isHorizontalFlow = layout === "horizontal";

      // Vertical layout refinements: tighter y-spread (smaller gap), wider x-spread (distance between sets)
      const verticalY = APP_CONFIG.visual.network.layout.y_margin + (index * APP_CONFIG.visual.network.layout.y_span) / (sorted.length - 1);
      const verticalXSource = APP_CONFIG.visual.network.layout.x_source;
      const verticalXTarget = APP_CONFIG.visual.network.layout.x_target;
      
      return {
        id: node.id,
        origX: node.x,
        origY: node.y,
        left: isHorizontalFlow ? { x, y: y_source } : { x: verticalXSource, y: verticalY },
        right: isHorizontalFlow ? { x, y: y_target } : { x: verticalXTarget, y: verticalY }
      };
    });
  }, [styleConfig.layout.orientation]);

  const [exchangePhase, setExchangePhase] = useState<ExchangePhase | null>(null);
  const prevStepRef = useRef(0);

  const exchangePresentation = useMemo<ExchangePresentation | undefined>(() => {
    if (step !== 2) return undefined;

    let phase: ExchangePhase;
    if (exchangePhase) {
      phase = exchangePhase;
    } else {
      phase = prevStepRef.current === 1 ? "animating" : "preview";
    }

    return {
      active: true,
      phase,
      pathKey,
      pathEdges: selectedEdges,
      beforeMatching: dynamicSteps[1]?.matching || initialMatching,
      afterMatching: dynamicSteps[3]?.matching || initialMatching,
      beforeDrivers: dynamicSteps[1]?.drivers || initialDrivers,
      afterDrivers: dynamicSteps[3]?.drivers || initialDrivers,
    };
  }, [
    step,
    exchangePhase,
    pathKey,
    selectedEdges,
    dynamicSteps,
    initialMatching,
    initialDrivers,
  ]);

  return (
    <Card className="overflow-hidden">
      <div className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <SurfaceTitle title={t("exchange.animation.title")} body={t("exchange.animation.body")} />
          <div className="flex flex-wrap items-center gap-3 justify-start lg:justify-end">
            <button
              type="button"
              onClick={() => {
                if (!playing && step === dynamicSteps.length - 1) {
                  setStep(0);
                }
                setPlaying((current) => !current);
              }}
              className="rounded-full border border-ink/10 px-4 py-2 text-sm text-ink transition hover:bg-ink/5 outline-none"
            >
              {playing ? t("controls.pause") : t("controls.play")}
            </button>
            <Stepper
              current={step}
              total={dynamicSteps.length}
              onNext={() => setStep((s) => Math.min(s + 1, dynamicSteps.length - 1))}
              onPrevious={() => setStep((s) => Math.max(s - 1, 0))}
              onReset={() => {
                setPlaying(false);
                setStep(0);
              }}
            />
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2 items-stretch">
          <div className="rounded-[26px] border border-ink/8 bg-surface p-1 h-[360px] lg:h-[400px] flex items-center justify-center">
            <svg viewBox="0 0 100 85" className="w-full h-full">
              <BipartiteNetwork
                stage={step}
                nodes={positions}
                hoveredNode={hoveredNode}
                setHoveredNode={setHoveredNode}
                customEdges={exchangeEdges}
                customMatching={currentStep.matching}
                customDrivers={currentStep.drivers}
                customHighlightEdges={currentStep.highlightEdges}
                exchangePresentation={exchangePresentation}
                forceShowMatching={true}
                styleConfig={styleConfig}
                layout={styleConfig.layout.orientation}
                title={t("network.title.bipartite")}
                titleFontSize={4.5}
                stripedNodeIds={stripedNodeIds}
              />
            </svg>
          </div>

          <div className="rounded-[26px] border border-ink/8 bg-surface p-1 h-[360px] lg:h-[400px] flex items-center justify-center overflow-hidden">
            <svg viewBox="5 -5 94 85" className="w-full h-full">
              <defs>
                <marker id="arrowhead" markerWidth={APP_CONFIG.visual.network.arrow_size.width} markerHeight={APP_CONFIG.visual.network.arrow_size.height} refX="0.1" refY="1.5" orient="auto">
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={nonMatchingEdgeColor} />
                </marker>
                <marker id="arrowhead-matched" markerWidth={APP_CONFIG.visual.network.arrow_size.width} markerHeight={APP_CONFIG.visual.network.arrow_size.height} refX="0.1" refY="1.5" orient="auto">
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={matchingEdgeColor} />
                </marker>
                <marker id="arrowhead-alt-matched" markerWidth={APP_CONFIG.visual.network.arrow_size.width} markerHeight={APP_CONFIG.visual.network.arrow_size.height} refX="0.1" refY="1.5" orient="auto">
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={altMatchedEdgeColor} />
                </marker>
                <marker id="arrowhead-regular" markerWidth={APP_CONFIG.visual.network.arrow_size.width} markerHeight={APP_CONFIG.visual.network.arrow_size.height} refX="0.1" refY="1.5" orient="auto">
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={nonMatchingEdgeColor} />
                </marker>
                <marker id="arrowhead-hover" markerWidth={APP_CONFIG.visual.network.arrow_size.width} markerHeight={APP_CONFIG.visual.network.arrow_size.height} refX="0.1" refY="1.5" orient="auto">
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={hoverEdgeColor} />
                </marker>
                <marker id="arrowhead-alternating" markerWidth={APP_CONFIG.visual.network.arrow_size.width} markerHeight={APP_CONFIG.visual.network.arrow_size.height} refX="0.1" refY="1.5" orient="auto">
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={altNonMatchedEdgeColor} />
                </marker>
              </defs>
              <DirectedNetwork
                stage={step}
                nodes={positions}
                hoveredNode={hoveredNode}
                setHoveredNode={setHoveredNode}
                styleConfig={directedStyleConfig}
                customEdges={exchangeEdges}
                customMatching={currentStep.matching}
                customDrivers={currentStep.drivers}
                customHighlightEdges={currentStep.highlightEdges}
                exchangePresentation={exchangePresentation}
                forceShowMatching={true}
                title={t("network.title.directed")}
                titleFontSize={4.5}
                stripedNodeIds={stripedNodeIds}
              />
            </svg>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div
            key={"exchange-theorem-optional-exchanges"}
            className="rounded-[26px] border border-ink/8 bg-surface/90 p-5"
          >
            <div className="mb-6 text-base font-semibold text-ink">
              <FormatMathText text={t("exchange.optionalSummary")} />
            </div>

            <div className="flex flex-col space-y-5">
              <div className="flex flex-col space-y-5">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-widest text-ink/40 mb-3 ml-1 flex items-center gap-1">
                    <span>{t("exchange.optional.startLabel")}</span>
                    <FormatMathText text="$s$" />
                  </div>
                  <div className="flex flex-wrap gap-2.5">
                    {[2, 3, 4].map((sId) => (
                      <button
                        key={`s-node-${sId}`}
                        onClick={() => {
                          setActiveS(sId);
                          setActiveTemplateIdx(0);
                          setActiveTargetIdx(0);
                        }}
                        className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all border outline-none",
                          activeS === sId
                            ? "bg-ink text-white dark:text-slate-900 border-ink shadow-md scale-105"
                            : "bg-surface text-ink/70 border-ink/10 hover:border-ink/30"
                        )}
                      >
                        {sId}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex flex-col">
                  <div className="text-[10px] font-bold uppercase tracking-widest text-ink/40 mb-3 ml-1">
                    {t("exchange.optional.pathLabel")}
                  </div>
                  {pathSelectionUnlocked ? (
                    <div className="flex flex-col gap-2">
                      {availableTemplates.map((template, idx) => (
                        <button
                          key={`temp-${idx}`}
                          onClick={() => {
                            setActiveTemplateIdx(idx);
                            setActiveTargetIdx(0);
                          }}
                          className={cn(
                            "w-full text-left p-5 rounded-2xl border transition-all flex items-center gap-3 group outline-none",
                            activeTemplateIdx === idx
                              ? "bg-ink/5 border-ink/30 shadow-sm"
                              : "bg-surface border-ink/10 text-ink/60 hover:border-ink/25"
                          )}
                        >
                          <div
                            className={cn(
                              "w-1.5 h-1.5 rounded-full",
                              activeTemplateIdx === idx ? "bg-amber-400" : "bg-ink/10 group-hover:bg-ink/30"
                            )}
                          />
                          <div className={cn("text-xs font-semibold", activeTemplateIdx === idx ? "text-ink" : "text-ink/60")}>
                            <FormatMathText text={formatArrowsForDisplay(template.template)} />
                          </div>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-2xl border border-dashed border-ink/20 bg-surface/60 px-4 py-3 text-xs text-ink/60">
                      {t("exchange.optional.pathLocked")}
                    </div>
                  )}
                </div>

                <div>
                  <div className="text-[10px] font-bold uppercase tracking-widest text-ink/40 mb-3 ml-1 flex items-center gap-1">
                    <span>{t("exchange.optional.targetLabel")}</span>
                    <FormatMathText text="$t$" />
                  </div>
                  {targetSelectionUnlocked ? (
                    <div className="flex flex-wrap gap-2.5">
                      {currentTemplate?.targets.map((target, idx) => (
                        <button
                          key={`target-${idx}`}
                          onClick={() => setActiveTargetIdx(idx)}
                          className={cn(
                            "w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all border outline-none",
                            activeTargetIdx === idx
                              ? "bg-ink text-white dark:text-slate-900 border-ink shadow-md scale-105"
                              : "bg-surface text-ink/70 border-ink/10 hover:border-ink/30"
                          )}
                        >
                          {target.t}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-2xl border border-dashed border-ink/20 bg-surface/60 px-4 py-3 text-xs text-ink/60">
                      {t("exchange.optional.targetLocked")}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-[26px] border border-ink/8 bg-surface/90 p-5 flex flex-col justify-between">
            <div className="flex-shrink-0">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-ink/55">{t("exchange.currentStep")}</div>
              <div className="mt-2 text-2xl font-semibold text-ink">{currentStep.title}</div>
              <div className="mt-3 text-sm leading-relaxed text-ink/72">
                <FormatMathText text={currentStep.body} />
              </div>
              <div className="mt-6 pt-5 mx-3 border-t border-ink/5">
                <div className="flex flex-wrap mb-4 gap-x-8 gap-y-4">
                  <KeyEdge color={matchingEdgeColor} label={t("exchange.legend.matched")} />
                  <KeyEdge color={nonMatchingEdgeColor} label={t("exchange.legend.unmatched")} />
                  <KeyEdge color={altNonMatchedEdgeColor} label={t("exchange.legend.alternating")} />
                </div>
                <div className="flex flex-wrap gap-x-8 gap-y-4">
                  <KeyNode color={driverNodeColor} label={t("exchange.legend.driver")} />
                  <KeyNode color={matchedNodeColor} label={t("exchange.legend.matchedNode")} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
