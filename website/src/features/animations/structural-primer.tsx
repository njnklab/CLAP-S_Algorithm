"use client";

import { useEffect, useMemo, useState } from "react";
import { motion, LayoutGroup } from "framer-motion";

import { primerEdges, primerMatchingOptions, primerNetworkNodes } from "@/features/animations/data/demoData";
import { cn } from "@/lib/utils";

import { APP_CONFIG } from "@/lib/config";
import { useVisualizationSettings } from "@/providers/visualization-settings-provider";
import { useI18n } from "@/providers/i18n-provider";
import { Card, FormatMathText, KeyNode, KeyEdge, Stepper, SurfaceTitle } from "@/components/ui/core";
import { DirectedNetwork } from "./network/directedNetwork";
import { BipartiteNetwork } from "./network/bipartiteNetwork";

export function StructuralPrimer() {
  const [stage, setStage] = useState(0);
  const [matchingCase, setMatchingCase] = useState<keyof typeof primerMatchingOptions>("A");
  const [hoveredNode, setHoveredNode] = useState<{ id: number; side: "source" | "target" | "both" } | null>(null);
  const [playing, setPlaying] = useState(false);
  const { styleConfig } = useVisualizationSettings();
  const layout = styleConfig.layout.orientation;
  const { t } = useI18n();
  const stages = useMemo(
    () => [
      {
        title: t("primer.stage.directed.title"),
        body: t("primer.stage.directed.body")
      },
      {
        title: t("primer.stage.bipartite.title"),
        body: t("primer.stage.bipartite.body")
      },
      {
        title: t("primer.stage.matching.title"),
        body: t("primer.stage.matching.body")
      },
      {
        title: t("primer.stage.drivers.title"),
        body: t("primer.stage.drivers.body")
      },
      {
        title: t("primer.stage.final.title"),
        body: t("primer.stage.final.body")
      }
    ],
    [t]
  );

  const matchingEdgeColor = styleConfig.edges.matching.color;
  const nonMatchingEdgeColor = styleConfig.edges.nonMatching.color;
  const altMatchedEdgeColor = styleConfig.edges.alternativeMatching.color;
  const alternatingEdgeColor = styleConfig.edges.alternativeNonMatching.color;
  const hoverEdgeColor =
    styleConfig.edges.hover.color === "no-change" ? styleConfig.edges.matching.color : styleConfig.edges.hover.color;
  const driverNodeColor = styleConfig.nodes.driver.strokeColor;
  const matchedNodeColor = styleConfig.nodes.matching.strokeColor;

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => {
      setStage((current) => {
        if (current === stages.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 3000);

    return () => window.clearInterval(timer);
  }, [playing, stages.length]);

  const neighbors = useMemo(() => {
    if (!hoveredNode) return [];
    if (hoveredNode.side === "source") {
      // Show targets for this source
      return primerEdges.filter(([f]) => f === hoveredNode.id).map(([_, t]) => `Node ${t}-`);
    }
    if (hoveredNode.side === "target") {
      // Show sources for this target
      return primerEdges.filter(([, t]) => t === hoveredNode.id).map(([f]) => `Node ${f}+`);
    }
    // "both" (directed stage)
    const outs = primerEdges.filter(([f]) => f === hoveredNode.id).map(([_, t]) => `Node ${t} (out)`);
    const ins = primerEdges.filter(([, t]) => t === hoveredNode.id).map(([f]) => `Node ${f} (in)`);
    return [...outs, ...ins];
  }, [hoveredNode]);

  const bipartitePositions = useMemo(() => {
    const sortedNodes = [...primerNetworkNodes].sort((a, b) => a.id - b.id);
    const centerX = APP_CONFIG.visual.network.layout.center;
    const centerY = APP_CONFIG.visual.network.layout.center;
    const radius = APP_CONFIG.visual.network.layout.radius;
    const margin = APP_CONFIG.visual.network.layout.margin;
    const span = APP_CONFIG.visual.network.layout.span;
    const ySource = APP_CONFIG.visual.network.layout.y_source;
    const yTarget = APP_CONFIG.visual.network.layout.y_target;

    return sortedNodes.map((node, index) => {
      const angle = (index * 2 * Math.PI) / sortedNodes.length - Math.PI / 2;
      const origX = centerX + radius * Math.cos(angle);
      const origY = centerY + radius * Math.sin(angle);

      const x = margin + (index * span) / (sortedNodes.length - 1);

      const isHorizontalFlow = layout === "horizontal";

      // Vertical layout tweaks: smaller node spacing (vertical spread), larger source/target distance (horizontal spread)
      const verticalY = APP_CONFIG.visual.network.layout.y_margin + (index * APP_CONFIG.visual.network.layout.y_span) / (sortedNodes.length - 1);
      const verticalXSource = APP_CONFIG.visual.network.layout.x_source;
      const verticalXTarget = APP_CONFIG.visual.network.layout.x_target;

      return {
        id: node.id,
        origX,
        origY,
        left: isHorizontalFlow ? { x, y: ySource } : { x: verticalXSource, y: verticalY },
        right: isHorizontalFlow ? { x, y: yTarget } : { x: verticalXTarget, y: verticalY }
      };
    });
  }, [layout]);

  return (
    <Card className="overflow-hidden">
      <div className="space-y-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <SurfaceTitle title={t("primer.animation.title")} body={t("primer.animation.body")} />
          <div className="flex flex-wrap items-center gap-3 justify-start lg:justify-end">
            <button
              type="button"
              onClick={() => {
                if (!playing && stage === stages.length - 1) {
                  setStage(0);
                }
                setPlaying((current) => !current);
              }}
              className="rounded-full border border-ink/10 px-4 py-2 text-sm text-ink transition hover:bg-ink/5 outline-none"
            >
              {playing ? t("controls.pause") : t("controls.play")}
            </button>
            <Stepper
              current={stage}
              total={stages.length}
              onNext={() => setStage((current) => Math.min(current + 1, stages.length - 1))}
              onPrevious={() => setStage((current) => Math.max(current - 1, 0))}
              onReset={() => {
                setPlaying(false);
                setStage(0);
              }}
            />
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr] items-start">
          <div className="rounded-[26px] border border-ink/8 bg-surface p-4 relative">
            <svg viewBox="0 0 100 100" className="w-full">
              <defs>
                <marker id="arrowhead" markerWidth={APP_CONFIG.visual.network.arrow_size.width} markerHeight={APP_CONFIG.visual.network.arrow_size.height} refX="0.1" refY="1.5" orient="auto">
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={nonMatchingEdgeColor} />
                </marker>
                <marker
                  id="arrowhead-matched"
                  markerWidth={APP_CONFIG.visual.network.arrow_size.width}
                  markerHeight={APP_CONFIG.visual.network.arrow_size.height}
                  refX="0.1"
                  refY="1.5"
                  orient="auto"
                >
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={matchingEdgeColor} />
                </marker>
                <marker
                  id="arrowhead-alt-matched"
                  markerWidth="4"
                  markerHeight="3"
                  refX="0.1"
                  refY="1.5"
                  orient="auto"
                >
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={altMatchedEdgeColor} />
                </marker>
                <marker
                  id="arrowhead-regular"
                  markerWidth="4"
                  markerHeight="3"
                  refX="0.1"
                  refY="1.5"
                  orient="auto"
                >
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={nonMatchingEdgeColor} />
                </marker>
                <marker
                  id="arrowhead-hover"
                  markerWidth="4"
                  markerHeight="3"
                  refX="0.1"
                  refY="1.5"
                  orient="auto"
                >
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={hoverEdgeColor} />
                </marker>
                <marker
                  id="arrowhead-alternating"
                  markerWidth="4"
                  markerHeight="3"
                  refX="0.1"
                  refY="1.5"
                  orient="auto"
                >
                  <path d="M 0 0 L 4 1.5 L 0 3 z" fill={alternatingEdgeColor} />
                </marker>
              </defs>

              <LayoutGroup id="primer-animation">
                {/* Unified Edges Layer - Always mounted at parent level for smooth transition across component swap */}
                {primerEdges.map(([from, to]) => {
                  const nodeFrom = bipartitePositions.find((n) => n.id === from);
                  const nodeTo = bipartitePositions.find((n) => n.id === to);
                  if (!nodeFrom || !nodeTo) return null;

                  const isDirectedView = stage === 0 || stage === 4;

                  // Calculate coordinates based on concept view
                  const x1 = isDirectedView ? nodeFrom.origX : nodeFrom.left.x;
                  const y1 = isDirectedView ? nodeFrom.origY : nodeFrom.left.y;
                  const x2 = isDirectedView ? nodeTo.origX : nodeTo.right.x;
                  const y2 = isDirectedView ? nodeTo.origY : nodeTo.right.y;

                  const angle = Math.atan2(y2 - y1, x2 - x1);
                  const nodeRadius = styleConfig.nodes.all.radius;
                  const edgeStrokeWidth = styleConfig.edges.all.strokeWidth;

                  const startPadding = nodeRadius + APP_CONFIG.visual.network.edge_padding;
                  const endPadding = isDirectedView
                    ? nodeRadius + APP_CONFIG.visual.network.edge_padding_directed
                    : nodeRadius + APP_CONFIG.visual.network.edge_padding;

                  const targetX1 = x1 + startPadding * Math.cos(angle);
                  const targetY1 = y1 + startPadding * Math.sin(angle);
                  const targetX2 = x2 - endPadding * Math.cos(angle);
                  const targetY2 = y2 - endPadding * Math.sin(angle);

                  // Shared logic for styles in Animation B
                  const matchingToUse = primerMatchingOptions[matchingCase].edges;
                  const isMatched = (stage >= 2) && matchingToUse.some(([mFrom, mTo]) => mFrom === from && mTo === to);
                  const isHovered = hoveredNode && (
                    hoveredNode.side === "both"
                      ? (from === hoveredNode.id || to === hoveredNode.id)
                      : (hoveredNode.side === "source" ? from === hoveredNode.id : to === hoveredNode.id)
                  );

                  let strokeColor = nonMatchingEdgeColor;
                  let visualStrokeWidth = edgeStrokeWidth;
                  let markerEnd = isDirectedView ? "url(#arrowhead)" : "none";
                  let opacity = styleConfig.edges.nonMatching.colorOpacity;

                  if (isHovered) {
                    strokeColor = hoverEdgeColor;
                    visualStrokeWidth = edgeStrokeWidth * APP_CONFIG.visual.network.hover_multiplier;
                    opacity = styleConfig.edges.hover.colorOpacity === "no-change"
                      ? styleConfig.edges.matching.colorOpacity
                      : styleConfig.edges.hover.colorOpacity;
                    if (isDirectedView) markerEnd = "url(#arrowhead-hover)";
                  } else if (isMatched) {
                    strokeColor = matchingEdgeColor;
                    visualStrokeWidth = edgeStrokeWidth * APP_CONFIG.visual.network.match_multiplier;
                    opacity = styleConfig.edges.matching.colorOpacity;
                    if (isDirectedView) markerEnd = "url(#arrowhead-matched)";
                  } else if (stage === 4) {
                    if (isDirectedView) markerEnd = "url(#arrowhead-regular)";
                  }

                  return (
                    <motion.line
                      key={`shared-edge-${from}-${to}`}
                      layoutId={`edge-${from}-${to}`}
                      initial={false}
                      animate={{
                        x1: targetX1,
                        y1: targetY1,
                        x2: targetX2,
                        y2: targetY2,
                        stroke: strokeColor,
                        strokeWidth: visualStrokeWidth,
                        opacity: opacity
                      }}
                      transition={{
                        stroke: { duration: isHovered ? 0 : 0.3 },
                        default: { type: "spring", stiffness: 60, damping: 15 }
                      }}
                      markerEnd={markerEnd}
                    />
                  );
                })}

                {stage === 0 || stage === 4 ? (
                  <DirectedNetwork
                    key="directed-view"
                    stage={stage}
                    nodes={bipartitePositions}
                    matchingCase={matchingCase}
                    hoveredNode={hoveredNode}
                    setHoveredNode={setHoveredNode}
                    renderEdges={false}
                    styleConfig={styleConfig}
                    title={t("network.title.directed")}
                    titleFontSize={4.5}
                  />
                ) : (
                  <BipartiteNetwork
                    key="bipartite-view"
                    stage={stage}
                    nodes={bipartitePositions}
                    matchingCase={matchingCase}
                    hoveredNode={hoveredNode}
                    setHoveredNode={setHoveredNode}
                    layout={layout}
                    renderEdges={false}
                    styleConfig={styleConfig}
                    title={t("network.title.bipartite")}
                    titleFontSize={4.5}
                  />
                )}
              </LayoutGroup>
            </svg>
          </div>
          <div className="space-y-3">
            <div className="rounded-[26px] border border-ink/8 bg-surface/90 p-5">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-ink/55">
                {t("primer.currentFrame")}
              </div>
              <div className="mt-3 text-2xl font-semibold text-ink">{stages[stage].title}</div>
              <div className="mt-3 text-sm leading-7 text-ink/72">
                <FormatMathText text={stages[stage].body} />
              </div>
              <div className="mt-6 pt-5 mx-3 border-t border-ink/5">
                <div className="flex flex-wrap mb-4 gap-x-8 gap-y-4">
                  <KeyEdge color={matchingEdgeColor} label={t("primer.legend.matchedEdge")} />
                  <KeyEdge color={nonMatchingEdgeColor} label={t("primer.legend.unmatchedEdge")} />
                </div>
                <div className="flex flex-wrap gap-x-8 gap-y-4">
                  <KeyNode color={driverNodeColor} label={t("primer.legend.driver")} />
                  <KeyNode color={matchedNodeColor} label={t("primer.legend.matchedNode")} />
                </div>
              </div>
            </div>
            <div className="rounded-[26px] border border-ink/8 bg-surface/90 p-5">
              {stage >= 2 && (
                <>
                  <p className="mt-2 text-[13px] text-ink/75 leading-relaxed bg-surface/40 p-3 rounded-xl border border-ink/5">
                    <FormatMathText text={t("primer.note")} />
                  </p>

                  <div className="mt-3 flex items-center gap-2">
                    <span className="text-[12px] font-semibold text-ink/60 uppercase tracking-wider">
                      {t("primer.altMatching")}
                    </span>
                    <div className="flex bg-mist/40 p-1 rounded-lg border border-ink/5">
                      {(["A", "B"] as const).map((opt) => (
                        <button
                          key={opt}
                          onClick={() => setMatchingCase(opt)}
                          className={cn(
                            "px-3 py-1 text-[12px] rounded-md transition-all font-medium",
                            matchingCase === opt
                              ? "bg-surface text-ink shadow-sm"
                              : "text-ink/60 hover:text-ink hover:bg-surface/30"
                          )}
                        >
                          {primerMatchingOptions[opt].label}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}

              <p className="mt-2">
                <FormatMathText text={t("primer.driverNote")} />
              </p>

              <p className="mt-2 mb-4 pb-4 border-b border-ink/5 text-[12px] text-ink/65 italic leading-relaxed">
                <strong className="mr-1">{t("primer.constraints.title")}</strong>
                <FormatMathText text={t("primer.constraints.body")} />
              </p>

              {hoveredNode ? (
                <div>
                  <p className="font-semibold text-ink">
                    {t("primer.focusLabel", {
                      id: hoveredNode.id,
                      suffix: hoveredNode.side === "source" ? "+" : hoveredNode.side === "target" ? "-" : ""
                    })}
                  </p>
                  <div className="mt-2 text-[13px] text-ink/80 flex flex-wrap gap-2">
                    <span>{t("primer.neighbors.label")}</span>
                    {neighbors.length > 0 ? (
                      neighbors.map((n, i) => (
                        <span key={i} className="px-1.5 bg-surface/50 rounded border border-ink/5 text-ink text-[12px]">
                          {n}
                        </span>
                      ))
                    ) : (
                      <span className="italic">{t("primer.neighbors.none")}</span>
                    )}
                  </div>
                </div>
              ) : (
                <p>{t("primer.hoverHint")}</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
