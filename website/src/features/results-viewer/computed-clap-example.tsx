"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";

import example from "@/features/results-viewer/data/computed-clap-example.json";
import { formatNumber } from "@/lib/utils";

import { APP_CONFIG } from "@/lib/config";
import { Card, FormatMathText, KeyNode, StatCard, SurfaceTitle } from "@/components/ui/core";

type ExampleData = typeof example;
type ExampleFrame = ExampleData["frames"][number];
type ExampleSegment = ExampleFrame["activeSegments"][number];

const partitionPalette = {
  dd1: APP_CONFIG.colors.light.layer1,
  dd2: APP_CONFIG.colors.light.layer2,
  cds: APP_CONFIG.colors.light.cds,
  cms: APP_CONFIG.colors.light.cms
} as const;

const layerStroke = {
  1: APP_CONFIG.colors.light.layer1,
  2: APP_CONFIG.colors.light.layer2
} as const;

function asNumberList(values: readonly unknown[]) {
  return values as number[];
}

function partitionOf(frame: ExampleFrame, node: number) {
  if (asNumberList(frame.state.dd1).includes(node)) return "dd1";
  if (asNumberList(frame.state.dd2).includes(node)) return "dd2";
  if (asNumberList(frame.state.cds).includes(node)) return "cds";
  return "cms";
}

function driverSetForLayer(frame: ExampleFrame, layer: 1 | 2) {
  return asNumberList(layer === 1 ? frame.state.d1 : frame.state.d2);
}

function matchingForLayer(frame: ExampleFrame, layer: 1 | 2) {
  return frame.state.matching[String(layer) as "1" | "2"];
}

function segmentPath(segment: ExampleSegment) {
  const points = segment.nodes
    .map((id) => example.nodes.find((node) => node.id === id))
    .filter((node): node is ExampleData["nodes"][number] => Boolean(node))
    .map((node) => `${node.x},${node.y}`);

  return points.join(" ");
}

function segmentDescription(segment: ExampleSegment) {
  const via = segment.nodes.slice(1, -1);
  if (!via.length) {
    return `Layer ${segment.layer}: ${segment.from} -> ${segment.to}`;
  }

  return `Layer ${segment.layer}: ${segment.from} -> ${segment.to} via ${via.join(", ")}`;
}

function overlapSize(frame: ExampleFrame) {
  return frame.state.cds.length;
}

function unionProgress(frame: ExampleFrame) {
  const initial = example.meta.initialUnion;
  const current = frame.state.unionSize;
  return ((initial - current) / Math.max(initial, 1)) * 100;
}

function Controls({
  autoplay,
  current,
  total,
  onToggleAutoplay,
  onPrevious,
  onNext,
  onReset
}: {
  autoplay: boolean;
  current: number;
  total: number;
  onToggleAutoplay: () => void;
  onPrevious: () => void;
  onNext: () => void;
  onReset: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        type="button"
        onClick={onPrevious}
        data-testid="computed-prev"
        className="rounded-full border border-ink/10 px-4 py-2 text-sm text-ink transition hover:border-ink/20 hover:bg-ink/5"
      >
        Previous
      </button>
      <button
        type="button"
        onClick={onToggleAutoplay}
        data-testid="computed-play"
        className="rounded-full bg-ink px-4 py-2 text-sm text-white transition hover:bg-ink/90 dark:text-slate-900"
      >
        {autoplay ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        onClick={onNext}
        data-testid="computed-next"
        className="rounded-full border border-ink/10 px-4 py-2 text-sm text-ink transition hover:border-ink/20 hover:bg-ink/5"
      >
        Next
      </button>
      <button
        type="button"
        onClick={onReset}
        data-testid="computed-reset"
        className="rounded-full border border-ink/10 px-4 py-2 text-sm text-ink transition hover:border-ink/20 hover:bg-ink/5"
      >
        Reset
      </button>
      <div className="ml-auto text-sm text-ink/60" data-testid="computed-step">
        Frame {current + 1} / {total}
      </div>
    </div>
  );
}

function LayerPanel({
  frame,
  layer
}: {
  frame: ExampleFrame;
  layer: 1 | 2;
}) {
  const currentDrivers = driverSetForLayer(frame, layer);
  const activeSegments = frame.activeSegments.filter((segment) => segment.layer === layer);
  const duplexNodeBgOpacity = Math.round(APP_CONFIG.opacity.duplex_node_bg * 255).toString(16).padStart(2, '0');

  return (
    <div className="rounded-[24px] border border-ink/8 bg-gradient-to-br from-surface to-mist/80 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="text-sm font-medium text-ink">Layer {layer}</div>
        <div className="text-xs uppercase tracking-[0.18em] text-ink/50">
          {frame.kind === "search" ? "search state" : frame.kind === "update" ? "post-update" : "initial"}
        </div>
      </div>
      <svg viewBox="0 0 100 100" className="w-full overflow-visible">
        <defs>
          <marker
            id={`computed-arrow-${layer}`}
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill={layerStroke[layer]} />
          </marker>
        </defs>

        {example.edges[String(layer) as "1" | "2"].map(([from, to]) => {
          const a = example.nodes.find((node) => node.id === from)!;
          const b = example.nodes.find((node) => node.id === to)!;

          return (
            <line
              key={`edge-${layer}-${from}-${to}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={APP_CONFIG.colors.light.non_matching_edge}
              strokeWidth={1.1}
              strokeLinecap="round"
              opacity={0.8}
            />
          );
        })}

        {matchingForLayer(frame, layer).map(([from, to]) => {
          const a = example.nodes.find((node) => node.id === from)!;
          const b = example.nodes.find((node) => node.id === to)!;

          return (
            <line
              key={`matching-${layer}-${from}-${to}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={layerStroke[layer]}
              strokeWidth={2}
              strokeLinecap="round"
              opacity={0.6}
            />
          );
        })}

        {activeSegments.map((segment, index) => {
          const points = segmentPath(segment);

          return (
            <polyline
              key={`segment-${layer}-${segment.from}-${segment.to}-${index}`}
              points={points}
              fill="none"
              stroke={layerStroke[layer]}
              strokeWidth={4.3}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray="8 6"
              markerEnd={`url(#computed-arrow-${layer})`}
            />
          );
        })}

        {example.nodes.map((node) => {
          const partition = partitionOf(frame, node.id);
          const color = partitionPalette[partition];
          const isDriver = currentDrivers.includes(node.id);
          const isActive = asNumberList(frame.activeNodes).includes(node.id);

          return (
            <g key={`node-${layer}-${node.id}`}>
              {isActive ? (
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={6.1}
                  fill="none"
                  stroke={APP_CONFIG.colors.light.hover}
                  strokeWidth={1.8}
                  opacity={0.85}
                />
              ) : null}
              <circle
                cx={node.x}
                cy={node.y}
                r={isDriver ? 4.8 : 4.1}
                fill={`${color}${duplexNodeBgOpacity}`}
                stroke={isActive ? APP_CONFIG.colors.light.hover : color}
                strokeWidth={isActive ? 1.9 : 1.4}
              />
              {isDriver ? (
                <circle cx={node.x} cy={node.y} r={1.8} fill={layerStroke[layer]} opacity={0.92} />
              ) : null}
              <text
                x={node.x}
                y={node.y + 0.5}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-ink text-[3.7px] font-semibold"
              >
                {node.id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function ComputedClapExample() {
  const [index, setIndex] = useState(0);
  const [autoplay, setAutoplay] = useState(false);

  const frame = example.frames[index];
  const progress = unionProgress(frame);

  useEffect(() => {
    if (!autoplay) return undefined;

    const timeout = window.setTimeout(() => {
      setIndex((current) => (current + 1) % example.frames.length);
    }, 2300);

    return () => window.clearTimeout(timeout);
  }, [autoplay, index]);

  const segmentSummary = useMemo(
    () => frame.activeSegments.map((segment) => segmentDescription(segment)),
    [frame]
  );

  return (
    <Card id="computed-example" className="overflow-hidden" data-testid="computed-example">
      <div
        className="grid items-start gap-8 xl:grid-cols-[1.1fr_0.9fr]"
        data-testid="computed-stage"
      >
        <div>
          <SurfaceTitle
            title="Algorithm-generated example · a denser synthetic duplex"
            body="This walkthrough is not hand-authored. It is a traced CLAP-S run on a generated ER-ER duplex with n = 16, mean degree 4.0, similarity 0.3, and seed 6."
          />

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard
              label="$|U|$"
              value={example.meta.initialUnion}
              hint="Before any CLAP is applied"
            />
            <StatCard
              label="Final |U|"
              value={example.meta.finalUnion}
              hint="After 3 successful CLAPs"
            />
            <StatCard
              label="Fixed-budget ILP"
              value={example.meta.ilpFixedOptimum}
              hint="Exact optimum for this instance"
            />
            <StatCard
              label="Replay progress"
              value={`${formatNumber(progress, 0)}%`}
              hint="Union reduction relative to the start"
            />
          </div>

          <div className="mt-5 rounded-[24px] border border-ink/8 bg-surface/80 p-4 text-sm leading-7 text-ink/72">
            The two layer-wise budgets stay fixed at <FormatMathText text="$k_1 = 3$" /> and{" "}
            <FormatMathText text="$k_2 = 3$" />. The animation shows only budget-preserving reconfiguration.
            The matching changes, the driver composition changes, and the union contracts from 6 to
            3.
          </div>

          <div className="mt-6 grid gap-5 xl:grid-cols-2">
            <LayerPanel frame={frame} layer={1} />
            <LayerPanel frame={frame} layer={2} />
          </div>

          <div className="mt-5 flex flex-wrap gap-4 text-sm">
            <KeyNode color={partitionPalette.dd1} label="$DD_1$" />
            <KeyNode color={partitionPalette.dd2} label="$DD_2$" />
            <KeyNode color={partitionPalette.cds} label="$CDS$" />
            <KeyNode color={partitionPalette.cms} label="$CMS$" />
            <KeyNode color={APP_CONFIG.colors.light.hover} label="Active CLAP nodes" />
          </div>

          <div className="mt-3 flex flex-wrap gap-4 text-sm">
            <KeyNode color={layerStroke[1]} label="Layer 1 Matching" />
            <KeyNode color={layerStroke[2]} label="Layer 2 Matching" />
          </div>

          <div className="mt-6">
            <Controls
              autoplay={autoplay}
              current={index}
              total={example.frames.length}
              onToggleAutoplay={() => setAutoplay((current) => !current)}
              onPrevious={() => {
                setAutoplay(false);
                setIndex((current) => (current - 1 + example.frames.length) % example.frames.length);
              }}
              onNext={() => {
                setAutoplay(false);
                setIndex((current) => (current + 1) % example.frames.length);
              }}
              onReset={() => {
                setAutoplay(false);
                setIndex(0);
              }}
            />
          </div>
        </div>

        <div className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-2">
            <StatCard label="$\Delta$" value={frame.state.delta} hint="Difference mass" />
            <StatCard label="$|U|$" value={frame.state.unionSize} hint="Union driver set size" />
            <StatCard label="$|CDS|$" value={overlapSize(frame)} hint="Shared drivers" />
            <StatCard label="Frame type" value={frame.kind} hint={frame.title} />
          </div>

          <div className="rounded-[24px] border border-ink/8 bg-surface/90 p-5">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
              Current frame
            </div>
            <div className="mt-3 text-lg font-semibold text-ink">{frame.title}</div>
            <p className="mt-3 text-sm leading-7 text-ink/74"><FormatMathText text={frame.note} /></p>
            {frame.kind === "update" ? (
              <div className="mt-4 inline-flex rounded-full px-4 py-2 text-sm font-medium">
                <FormatMathText text="$\Delta$ decreases, so the union contracts by one." />
              </div>
            ) : null}
          </div>

          <div className="rounded-[24px] border border-ink/8 bg-gradient-to-br from-surface to-mist/70 p-5">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
              Active CLAP segments
            </div>
            <div className="mt-3 space-y-2 text-sm leading-7 text-ink/74">
              {segmentSummary.length ? (
                segmentSummary.map((segment) => <p key={segment}>• <FormatMathText text={segment.replace("->", String.raw`$\to$`)} /></p>)
              ) : (
                <p>• No CLAP segment is active yet. This frame shows the initial disagreement.</p>
              )}
            </div>
          </div>

          <div className="rounded-[24px] border border-ink/8 bg-gradient-to-br from-surface to-mist/70 p-5">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
              Current sets
            </div>
            <div className="mt-3 space-y-2 text-sm text-ink/76">
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$D_1$:" /></span> {frame.state.d1.join(", ")}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$D_2$:" /></span> {frame.state.d2.join(", ")}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$DD_1$:" /></span>{" "}
                {frame.state.dd1.length ? frame.state.dd1.join(", ") : "empty"}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$DD_2$:" /></span>{" "}
                {frame.state.dd2.length ? frame.state.dd2.join(", ") : "empty"}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$CDS$:" /></span>{" "}
                {frame.state.cds.length ? frame.state.cds.join(", ") : "empty"}
              </div>
              <div>
                <span className="font-medium text-ink"><FormatMathText text="$CMS$:" /></span>{" "}
                {frame.state.cms.length ? frame.state.cms.join(", ") : "empty"}
              </div>
            </div>
          </div>

        </div>
      </div>
    </Card>
  );
}
