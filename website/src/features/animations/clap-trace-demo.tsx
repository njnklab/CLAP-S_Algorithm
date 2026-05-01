"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { useVisualizationSettings } from "@/providers/visualization-settings-provider";

import { toyBaseEdges, toyNodePositions, traceSteps, toyInitialMatchings, type TraceStep, type ClapAlternatingPath } from "@/features/animations/data/demoData";
import { APP_CONFIG } from "@/lib/config";
import { Card, FormatMathText, KeyNode, StatCard, Stepper, SurfaceTitle } from "@/components/ui/core";
import { BipartiteNetwork } from "./network/bipartiteNetwork";
import { DirectedNetwork } from "./network/directedNetwork";
import { cn } from "@/lib/utils";
import type { NetworkNode } from "./network/types";
import { useI18n } from "@/providers/i18n-provider";

type ClapPhase =
  | { type: "initial" }
  | { type: "search"; layer: 1 | 2; segmentIdx: number; nodes: number[]; isTerminal: boolean }
  | { type: "exchange_anim"; segments: ClapAlternatingPath[] }
  | { type: "final" };

const DUPLEX_BG_OPACITY_HEX = Math.round(APP_CONFIG.opacity.duplex_node_bg * 255)
  .toString(16)
  .padStart(2, "0");

function partitionColor(step: TraceStep, node: number) {
  if (step.dd1.includes(node)) return APP_CONFIG.colors.light.layer1;
  if (step.dd2.includes(node)) return APP_CONFIG.colors.light.layer2;
  if (step.cds.includes(node)) return APP_CONFIG.colors.light.cds;
  return APP_CONFIG.colors.light.cms;
}

function activeNodeType(step: TraceStep, node: number) {
  if (!step.clap) return null;
  const firstSegment = step.clap.segments[0];
  const lastSegment = step.clap.segments[step.clap.segments.length - 1];
  if (firstSegment.s === node) return "source";
  if (lastSegment.t === node) return "target";
  if (step.clap.segments.some((segment) => segment.nodes.includes(node))) return "relay";
  return null;
}

function buildPhaseSequence(step: TraceStep): ClapPhase[] {
  if (!step.clap) return [{ type: "final" }];

  const segments = step.clap.segments;
  const phases: ClapPhase[] = [{ type: "initial" }];

  // Search phases: one per segment
  segments.forEach((seg, idx) => {
    phases.push({
      type: "search",
      layer: seg.layer,
      segmentIdx: idx,
      nodes: seg.nodes,
      isTerminal: idx === segments.length - 1
    });
  });

  // Exchange animation phase (C1-C2)
  phases.push({
    type: "exchange_anim",
    segments
  });

  // Final phase (C2-C3)
  phases.push({ type: "final" });

  return phases;
}


function describePhase(phase: ClapPhase | undefined, t: any) {
  if (!phase) return "Idle";
  if (phase.type === "initial") return t("clap.phase.initial");
  if (phase.type === "exchange_anim") return t("clap.phase.exchange");
  if (phase.type === "final") return t("clap.phase.final");

  const prefix = phase.isTerminal ? t("clap.phase.reachTarget") : t("clap.phase.followRelay");
  return `${prefix} · ${t("clap.layer")} ${phase.layer} (${phase.nodes[0]} → ${phase.nodes.at(-1)})`;
}

function toggleEdges(matching: [number, number][], path: [number, number][]) {
  const result = [...matching.map((e) => [...e] as [number, number])];
  for (const [pf, pt] of path) {
    const idx = result.findIndex(
      (e) => e[0] === pf && e[1] === pt
    );
    if (idx > -1) result.splice(idx, 1);
    else result.push([pf, pt]);
  }
  return result;
}

function findCanonical(layer: 1 | 2, u: number, v: number): [number, number] {
  const edges = toyBaseEdges[layer];
  if (edges.some((e) => e[0] === u && e[1] === v)) return [u, v];
  if (edges.some((e) => e[0] === v && e[1] === u)) return [v, u];
  return [u, v];
}

export function ClapTraceDemo() {
  const [stepIndex, setStepIndex] = useState(0);
  const [phaseCursor, setPhaseCursor] = useState(0);
  const [detailedPause, setDetailedPause] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<{ id: number; side: "source" | "target" | "both" } | null>(null);
  const [exchangePhase, setExchangePhase] = useState<"animating" | "preview" | null>(null);
  const prevPhaseCursorRef = useRef(0);
  const { t } = useI18n();

  const step = traceSteps[stepIndex];
  const phaseSequence = useMemo(() => buildPhaseSequence(step), [step]);
  const phaseCount = phaseSequence.length;
  const currentPhase = phaseSequence[phaseCursor];

  // Pre-calculate matchings for all steps
  const matchingsByStep = useMemo(() => {
    let m1 = [...toyInitialMatchings[1]];
    let m2 = [...toyInitialMatchings[2]];
    const results = [{ 1: m1, 2: m2 }];

    for (let i = 1; i < traceSteps.length; i++) {
      const clap = traceSteps[i].clap;
      if (!clap) {
        results.push({ 1: m1, 2: m2 });
        continue;
      }
      const p1Edges = clap.segments.filter(s => s.layer === 1).flatMap(s => s.edges);
      const p2Edges = clap.segments.filter(s => s.layer === 2).flatMap(s => s.edges);
      m1 = toggleEdges(m1, p1Edges);
      m2 = toggleEdges(m2, p2Edges);
      results.push({ 1: m1, 2: m2 });
    }
    return results;
  }, []);

  const currentMatchings = matchingsByStep[stepIndex];
  const prevStepData = stepIndex === 0 ? traceSteps[0] : traceSteps[stepIndex - 1];
  const prevMatchings = stepIndex === 0 ? { 1: toyInitialMatchings[1], 2: toyInitialMatchings[2] } : matchingsByStep[stepIndex - 1];

  const currentDisplayData = useMemo(() => {
    // If we are in 'final' phase, or in 'exchange_anim' and already previewing, show the new data
    if (currentPhase?.type === "final" || (currentPhase?.type === "exchange_anim" && exchangePhase === "preview")) {
      return step;
    }
    // Otherwise show previous step's numbers
    return prevStepData;
  }, [step, prevStepData, currentPhase, exchangePhase]);

  const visibleSegments = useMemo(() => {
    if (!step.clap || !currentPhase || currentPhase.type === "initial") return [];
    if (currentPhase.type === "exchange_anim" || currentPhase.type === "final") {
      return step.clap.segments;
    }
    // Show all segments up to current index
    const count = currentPhase.segmentIdx + 1;
    return step.clap.segments.slice(0, count);
  }, [step, currentPhase]);

  const phaseLabel = describePhase(currentPhase, t);

  const displayNote = useMemo(() => {
    if (!currentPhase) return "Idle";
    if (currentPhase.type === "initial") return t("clap.note.initial");
    if (currentPhase.type === "search") {
      return t("clap.note.search", {
        layer: currentPhase.layer,
        s: currentPhase.nodes[0],
        t: currentPhase.nodes.at(-1)
      });
    }
    if (currentPhase.type === "exchange_anim") {
      return t("clap.note.exchange");
    }
    return t(`clap.step.note.${stepIndex}`);
  }, [step, currentPhase, t]);

  const stripedNodesByLayer = useMemo(() => {
    const results: Record<number, number[]> = { 1: [], 2: [] };
    if (!step.clap || currentPhase?.type === "final") return results;

    const s1 = new Set<number>();
    const s2 = new Set<number>();
    step.clap.segments.forEach((seg) => {
      if (seg.layer === 1) {
        s1.add(seg.s);
        s1.add(seg.t);
      } else {
        s2.add(seg.s);
        s2.add(seg.t);
      }
    });
    results[1] = Array.from(s1);
    results[2] = Array.from(s2);
    return results;
  }, [step, currentPhase]);

  const unionSize = new Set([...currentDisplayData.d1, ...currentDisplayData.d2]).size;
  const deltaChange = stepIndex === 0 ? null : traceSteps[stepIndex - 1].delta - step.delta;
  const { styleConfig } = useVisualizationSettings();

  const matchingEdgeColor = styleConfig.edges.matching.color;
  const nonMatchingEdgeColor = styleConfig.edges.nonMatching.color;
  const altNonMatchedEdgeColor = styleConfig.edges.alternativeNonMatching.color;
  const altMatchedEdgeColor = styleConfig.edges.alternativeMatching.color;
  const hoverEdgeColor =
    styleConfig.edges.hover.color === "no-change"
      ? styleConfig.edges.matching.color
      : styleConfig.edges.hover.color;

  const networkNodes = useMemo<NetworkNode[]>(() => {
    const sorted = [...toyNodePositions].sort((a, b) => {
      const order = [1, 5, 7, 6, 3, 4, 2, 8, 9];
      return order.indexOf(a.id) - order.indexOf(b.id);
    });
    const span = sorted.length > 1 ? sorted.length - 1 : 1;
    const layout = styleConfig.layout.orientation;
    const margin = APP_CONFIG.visual.network.layout.margin;
    const lspan = APP_CONFIG.visual.network.layout.span;
    const y_source = APP_CONFIG.visual.network.layout.y_source;
    const y_target = APP_CONFIG.visual.network.layout.y_target;

    return sorted.map<NetworkNode>((node, idx) => {
      const x = margin + (idx / span) * lspan;
      const isHorizontalFlow = layout === "horizontal";

      const verticalY = APP_CONFIG.visual.network.layout.y_margin + (idx * APP_CONFIG.visual.network.layout.y_span) / (sorted.length - 1);
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

  const segmentsByLayer = useMemo(
    () => {
      const result: { 1: [number, number][], 2: [number, number][] } = { 1: [], 2: [] };
      visibleSegments.forEach(seg => {
        seg.edges.forEach(edge => {
          result[seg.layer].push(edge);
        });
      });
      return result;
    },
    [visibleSegments]
  );


  useEffect(() => {
    if (!currentPhase || currentPhase.type !== "exchange_anim") {
      setExchangePhase(null);
      prevPhaseCursorRef.current = phaseCursor;
      return;
    }

    // Only play animating when entering from a search phase
    if (prevPhaseCursorRef.current < phaseCursor) {
      setExchangePhase("animating");
      const timer = window.setTimeout(() => {
        setExchangePhase("preview");
      }, 850);
      prevPhaseCursorRef.current = phaseCursor;
      return () => window.clearTimeout(timer);
    }

    setExchangePhase("preview");
    prevPhaseCursorRef.current = phaseCursor;
  }, [phaseCursor, currentPhase]);

  const exchangePresentationByLayer = useMemo(() => {
    if (!currentPhase || currentPhase.type !== "exchange_anim" || !step.clap) return { 1: undefined, 2: undefined };

    const segments = step.clap.segments;
    const getLayerPresentation = (layer: 1 | 2) => {
      const layerSegments = segments.filter(s => s.layer === layer);
      if (layerSegments.length === 0) return undefined;

      // Extract edges in chronological order of segments
      const pathEdges = layerSegments.flatMap(s => s.edges);

      return {
        active: true,
        phase: exchangePhase || ("animating" as const),
        pathKey: `clap-${stepIndex}-${layer}`,
        pathEdges,
        beforeMatching: prevMatchings[layer as 1 | 2],
        afterMatching: currentMatchings[layer as 1 | 2],
        beforeDrivers: layer === 1 ? prevStepData.d1 : prevStepData.d2,
        afterDrivers: layer === 1 ? step.d1 : step.d2,
      };
    };

    return {
      1: getLayerPresentation(1),
      2: getLayerPresentation(2)
    };
  }, [currentPhase, step, prevStepData, stepIndex, exchangePhase]);

  const layerDriversForPhase = (layer: 1 | 2) => {
    if (!currentPhase || currentPhase.type === "final") {
      return layer === 1 ? step.d1 : step.d2;
    }
    return layer === 1 ? prevStepData.d1 : prevStepData.d2;
  };

  useEffect(() => {
    setPlaying(false);
    setPhaseCursor(0);
  }, [stepIndex]);

  // Fast auto-play when detailed mode is off
  useEffect(() => {
    if (!detailedPause && phaseCursor >= 0 && phaseCursor < phaseCount - 1 && !playing) {
      const timer = window.setTimeout(() => {
        setPhaseCursor((prev) => Math.min(prev + 1, phaseCount - 1));
      }, 700);
      return () => window.clearTimeout(timer);
    }
  }, [detailedPause, phaseCursor, phaseCount, playing]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setTimeout(() => {
      const advanced = advanceTimeline();
      if (!advanced) {
        setPlaying(false);
      }
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [playing, stepIndex, phaseCursor, phaseCount]);

  const advanceTimeline = () => {
    if (phaseCursor < phaseCount - 1) {
      setPhaseCursor((prev) => prev + 1);
      return true;
    }
    if (stepIndex < traceSteps.length - 1) {
      setStepIndex(stepIndex + 1);
      return true;
    }
    return false;
  };

  const handleNext = () => {
    if (detailedPause) {
      if (!advanceTimeline()) {
        setPlaying(false);
      }
    } else {
      // Step mode logic
      if (phaseCursor < phaseCount - 1) {
        // If we are currently manual-playing phases within a step
        setPhaseCursor(prev => prev + 1);
      } else if (stepIndex < traceSteps.length - 1) {
        // Look ahead to find phaseCount for next step to avoid out-of-bounds
        const nextStep = traceSteps[stepIndex + 1];
        const nextPhasesCount = buildPhaseSequence(nextStep).length;
        setStepIndex(stepIndex + 1);
        setPhaseCursor(Math.min(1, nextPhasesCount - 1));
      } else {
        setPlaying(false);
      }
    }
  };

  const handlePrevious = () => {
    setPlaying(false);
    if (detailedPause) {
      if (phaseCursor > 0) {
        setPhaseCursor((prev) => prev - 1);
        return;
      }
      if (stepIndex > 0) {
        setStepIndex(stepIndex - 1);
      }
    } else {
      // Step mode logic
      if (stepIndex > 0) {
        setStepIndex(stepIndex - 1);
        setPhaseCursor(0);
      }
    }
  };

  const handleReset = () => {
    setPlaying(false);
    setStepIndex(0);
    setPhaseCursor(0);
  };

  const toggleDetailedPause = () => {
    setPlaying(false);
    setDetailedPause((prev) => !prev);
  };

  const stepperCurrent = stepIndex;
  const stepperTotal = traceSteps.length;
  const hasMoreTimeline =
    (detailedPause && (phaseCursor < phaseCount - 1 || stepIndex < traceSteps.length - 1)) ||
    (!detailedPause && stepIndex < traceSteps.length - 1);

  return (
    <Card className="overflow-hidden">
      <div className="grid gap-4 lg:grid-cols-[1.3fr_0.8fr]">
        <div className="space-y-2">
          <SurfaceTitle
            title={t("clap.title")}
            body={t("clap.subtitle.body")}
          />
          <div className="flex flex-wrap items-center gap-3 border border-ink/10 rounded-[24px] p-4 bg-surface/95">
            <button
              type="button"
              onClick={toggleDetailedPause}
              aria-pressed={detailedPause}
              className={cn(
                "rounded-full border px-4 py-2 text-sm transition",
                detailedPause 
                  ? "bg-ink text-surface border-ink shadow-sm"
                  : "border-ink/15 text-ink/70 hover:border-ink/30 bg-transparent"
              )}
            >
              {detailedPause ? t("clap.detailedOn") : t("clap.detailedOff")}
            </button>
            <button
              type="button"
              onClick={() => {
                if (playing) {
                  setPlaying(false);
                } else {
                  if (!hasMoreTimeline) {
                    handleReset();
                  }
                  setPlaying(true);
                }
              }}
              className="rounded-full border border-ink/15 px-4 py-2 text-sm text-ink transition hover:border-ink/35"
            >
              {playing ? t("controls.pause") : t("controls.play")}
            </button>
            <div className="flex-1 min-w-[220px]">
              <Stepper
                current={stepperCurrent}
                total={stepperTotal}
                onNext={handleNext}
                onPrevious={handlePrevious}
                onReset={handleReset}
              />
            </div>
          </div>

          <div className="rounded-[22px] border border-ink/8 bg-surface/95 p-4 text-xs font-medium uppercase tracking-[0.2em] text-ink/45">
            {t("clap.phase", { current: phaseCursor + 1, total: phaseCount, label: phaseLabel })}
          </div>

          {[1, 2].map((layer) => (
            <div key={layer}>
              <div className="grid gap-2 md:grid-cols-2">
                <div className="rounded-[22px] border border-ink/10 bg-gradient-to-b from-surface to-mist/80 p-1 h-[240px] lg:h-[280px]">
                  <svg viewBox="0 0 100 85" className="w-full h-full">
                    <BipartiteNetwork
                      stage={currentPhase?.type === "final" ? 3 : (currentPhase?.type === "initial" ? 0 : (currentPhase?.type === "exchange_anim" ? 2 : 1))}
                      nodes={networkNodes}
                      hoveredNode={hoveredNode}
                      setHoveredNode={setHoveredNode}
                      layout={styleConfig.layout.orientation}
                      styleConfig={styleConfig}
                      customEdges={toyBaseEdges[layer as 1 | 2]}
                      customMatching={currentPhase?.type === "final" ? currentMatchings[layer as 1 | 2] : prevMatchings[layer as 1 | 2]}
                      customDrivers={layerDriversForPhase(layer as 1 | 2)}
                      customHighlightEdges={currentPhase?.type === "final" ? [] : segmentsByLayer[layer as 1 | 2]}
                      exchangePresentation={exchangePresentationByLayer[layer as 1 | 2]}
                      forceShowMatching
                      title={t("clap.view.bipartite", { layer })}
                      titleFontSize={4}
                      stripedNodeIds={stripedNodesByLayer[layer]}
                    />
                  </svg>
                </div>
                <div className="rounded-[22px] border border-ink/10 bg-gradient-to-b from-surface to-mist/80 p-1 h-[240px] lg:h-[280px]">
                  <svg viewBox="5 0 94 85" className="w-full h-full">
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
                      stage={currentPhase?.type === "final" ? 3 : (currentPhase?.type === "initial" ? 0 : (currentPhase?.type === "exchange_anim" ? 2 : 1))}
                      nodes={networkNodes}
                      hoveredNode={hoveredNode}
                      setHoveredNode={setHoveredNode}
                      styleConfig={styleConfig}
                      customEdges={toyBaseEdges[layer as 1 | 2]}
                      customMatching={currentPhase?.type === "final" ? currentMatchings[layer as 1 | 2] : prevMatchings[layer as 1 | 2]}
                      customDrivers={layerDriversForPhase(layer as 1 | 2)}
                      customHighlightEdges={currentPhase?.type === "final" ? [] : segmentsByLayer[layer as 1 | 2]}
                      exchangePresentation={exchangePresentationByLayer[layer as 1 | 2]}
                      forceShowMatching
                      title={t("clap.view.directed", { layer })}
                      titleFontSize={4}
                      stripedNodeIds={stripedNodesByLayer[layer]}
                    />
                  </svg>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-2">
            <StatCard
              label={t("clap.delta")}
              value={currentDisplayData.delta}
              hint={t("union.stats.deltaHint")}
            />
            <StatCard
              label={t("clap.unionSize")}
              value={unionSize}
              hint={t("union.stats.unionHint")}
            />
            <StatCard
              label="$DD_1$"
              value={currentDisplayData.dd1.length}
              hint={t("union.membership.layer1")}
            />
            <StatCard
              label="$DD_2$"
              value={currentDisplayData.dd2.length}
              hint={t("union.membership.layer2")}
            />
          </div>
          <div className="rounded-[24px] border border-ink/8 bg-surface/90 p-5">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
              {t("clap.thisFrameExplains")}
            </div>
            <div className="mt-3 text-sm leading-7 text-ink/74">
              <FormatMathText text={displayNote} />
            </div>
            {deltaChange && (currentPhase?.type === "final" || (currentPhase?.type === "exchange_anim" && exchangePhase === "preview")) ? (
              <div className="mt-4 inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium">
                <FormatMathText text={t("clap.deltaDecrease", { delta: deltaChange })} />
              </div>
            ) : null}
          </div>
          <div className="rounded-[24px] border border-ink/8 bg-gradient-to-br from-surface to-mist/70 p-5">
            <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
              {t("clap.currentSets")}
            </div>
            <div className="space-y-2 text-sm text-ink/76">
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$D_1$:" /></span> {currentDisplayData.d1.join(", ")}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$D_2$:" /></span> {currentDisplayData.d2.join(", ")}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$DD_1$:" /></span>{" "}
                {currentDisplayData.dd1.length ? currentDisplayData.dd1.join(", ") : t("clap.empty")}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$DD_2$:" /></span>{" "}
                {currentDisplayData.dd2.length ? currentDisplayData.dd2.join(", ") : t("clap.empty")}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$CDS$:" /></span>{" "}
                {currentDisplayData.cds.length ? currentDisplayData.cds.join(", ") : t("clap.empty")}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$CMS$:" /></span>{" "}
                {currentDisplayData.cms.length ? currentDisplayData.cms.join(", ") : t("clap.empty")}
              </div>
            </div>
          </div>
          <div className="rounded-[24px] border border-ink/8 bg-mist/65 p-5 text-sm leading-7 text-ink/72">
            <FormatMathText text={t("clap.relayRule")} />
          </div>
        </div>
      </div>
    </Card>
  );

}
