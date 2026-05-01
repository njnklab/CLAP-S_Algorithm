"use client";

import React, { useMemo } from "react";
import { motion } from "framer-motion";
import { primerEdges, primerMatchingOptions } from "../data/demoData";
import { NetworkProps, DEFAULT_STYLE_CONFIG, EdgeTuple, LineGeometry } from "./types";
import { ExchangeOverlay, buildPreviewStyleMap, buildStep2ExchangeTokens } from "./exchangeOverlay";
import { useI18n } from "@/providers/i18n-provider";
import { APP_CONFIG } from "@/lib/config";

function edgeEq(a: EdgeTuple, b: EdgeTuple) {
  return a[0] === b[0] && a[1] === b[1];
}

function getBipartiteEdgeGeometry(
  edge: EdgeTuple,
  nodes: NetworkProps["nodes"],
  nodeRadius: number
): LineGeometry | null {
  const [from, to] = edge;
  const nodeFrom = nodes.find((n) => n.id === from);
  const nodeTo = nodes.find((n) => n.id === to);
  if (!nodeFrom || !nodeTo) return null;

  const x1 = nodeFrom.left.x;
  const y1 = nodeFrom.left.y;
  const x2 = nodeTo.right.x;
  const y2 = nodeTo.right.y;

  const angle = Math.atan2(y2 - y1, x2 - x1);
  const padding = nodeRadius + 1.2;

  return {
    x1: x1 + padding * Math.cos(angle),
    y1: y1 + padding * Math.sin(angle),
    x2: x2 - padding * Math.cos(angle),
    y2: y2 - padding * Math.sin(angle),
  };
}

export function BipartiteNetwork({
  stage,
  nodes,
  hoveredNode,
  setHoveredNode,
  matchingCase,
  layout = "horizontal",
  styleConfig = DEFAULT_STYLE_CONFIG,
  customEdges,
  customMatching,
  customDrivers,
  customHighlightEdges,
  exchangePresentation,
  forceShowMatching = false,
  title,
  titleFontSize,
  renderEdges = true,
  stripedNodeIds
}: NetworkProps) {
  const isBipartiteFull = (stage >= 1 && stage <= 3) || forceShowMatching;
  const { t } = useI18n();
  
  const { 
    radius: nodeRadius, 
    strokeWidth: nodeStrokeWidth, 
    textFontSize, 
    textOffset 
  } = styleConfig.nodes.all;
  
  const { strokeWidth: edgeStrokeWidth } = styleConfig.edges.all;

  const edgesToRender = customEdges || primerEdges;
  const isHorizontal = layout === "horizontal"; // We want this to mean Side-by-Side (Left-Right)

  const currentMatching = customMatching || (matchingCase ? primerMatchingOptions[matchingCase].edges : []);
  const drivers = customDrivers || (matchingCase ? primerMatchingOptions[matchingCase].drivers : []);
  const highlightEdgesToUse = customHighlightEdges || [];

  // 边的语义：step 2 仍以 beforeMatching 为基础
  const edgeMatching = exchangePresentation?.active
    ? exchangePresentation.beforeMatching
    : currentMatching;

  // 节点的语义：preview 才切到 after
  const nodeMatching =
    exchangePresentation?.active
      ? exchangePresentation.phase === "preview"
        ? exchangePresentation.afterMatching
        : exchangePresentation.beforeMatching
      : currentMatching;

  const nodeDrivers =
    exchangePresentation?.active
      ? exchangePresentation.phase === "preview"
        ? exchangePresentation.afterDrivers
        : exchangePresentation.beforeDrivers
      : drivers;

  const exchangeTokens = useMemo(() => {
    if (!exchangePresentation?.active) return [];
    return buildStep2ExchangeTokens(
      exchangePresentation.pathEdges,
      exchangePresentation.beforeMatching,
      exchangePresentation.afterMatching,
      exchangePresentation.pathKey
    );
  }, [exchangePresentation]);

  const previewStyleMap = useMemo(() => {
    return buildPreviewStyleMap(exchangeTokens);
  }, [exchangeTokens]);
  const stripedNodeSet = useMemo(() => new Set(stripedNodeIds || []), [stripedNodeIds]);
  const stripeDefs: React.ReactNode[] = [];
  const registeredPatterns = new Set<string>();
  const registerStripePattern = (patternId: string, stripeColor: string, stripeOpacity: number) => {
    if (registeredPatterns.has(patternId)) return;
    stripeDefs.push(
      <pattern
        id={patternId}
        key={patternId}
        patternUnits="userSpaceOnUse"
        width="0.8"
        height="0.8"
        patternTransform="rotate(45)"
      >
        <rect width="0.8" height="0.8" fill="transparent" />
        <rect width="0.4" height="0.8" fill={stripeColor} fillOpacity={stripeOpacity} />
      </pattern>
    );
    registeredPatterns.add(patternId);
  };

  const sourceNodes = nodes.map((node) => {
    const isMatched = (stage >= 3 || forceShowMatching) && nodeMatching.some(([from, _]) => from === node.id);
    const isDriver = (stage >= 3 || forceShowMatching) && (nodeDrivers as readonly number[]).includes(node.id);
    const isSourceHovered = hoveredNode?.id === node.id && (hoveredNode.side === "source" || hoveredNode.side === "both");

    let fillColor: string;
    let strokeColor: string;
    let textColor: string;
    let fillColorOpacity: number;
    let currentRadius = nodeRadius;
    if (isDriver) {
      fillColor = styleConfig.nodes.driver.fillColor;
      strokeColor = styleConfig.nodes.driver.strokeColor;
      textColor = styleConfig.nodes.driver.textColor;
      fillColorOpacity = styleConfig.nodes.driver.fillColorOpacity;
    } else if (isMatched) {
      fillColor = styleConfig.nodes.matching.fillColor;
      strokeColor = styleConfig.nodes.matching.strokeColor;
      textColor = styleConfig.nodes.matching.textColor;
      fillColorOpacity = styleConfig.nodes.matching.fillColorOpacity;
    } else {
      fillColor = styleConfig.nodes.regular.fillColor;
      strokeColor = styleConfig.nodes.regular.strokeColor;
      textColor = styleConfig.nodes.regular.textColor;
      fillColorOpacity = styleConfig.nodes.regular.fillColorOpacity;
    }

    if (isSourceHovered) {
      if (styleConfig.nodes.hover.fillColor !== "no-change") fillColor = styleConfig.nodes.hover.fillColor;
      if (styleConfig.nodes.hover.strokeColor !== "no-change") strokeColor = styleConfig.nodes.hover.strokeColor;
      if (styleConfig.nodes.hover.textColor !== "no-change") textColor = styleConfig.nodes.hover.textColor;
      if (styleConfig.nodes.hover.fillColorOpacity !== "no-change") fillColorOpacity = styleConfig.nodes.hover.fillColorOpacity;
      if (styleConfig.nodes.hover.scale !== "no-change") currentRadius *= styleConfig.nodes.hover.scale;
    }

    return (
      <motion.g
        key={`node-source-${node.id}`}
        layoutId={`node-group-left-${node.id}`}
        layout
        initial={{ x: node.origX, y: node.origY, opacity: 0 }}
        animate={{
          x: node.left.x,
          y: node.left.y,
          opacity: 1
        }}
        exit={{ x: node.origX, y: node.origY, opacity: 0 }}
        transition={{ type: "spring", stiffness: 60, damping: 15 }}
      >
        <motion.circle
          cx={0}
          cy={0}
          r={currentRadius}
          initial={false}
          animate={{
            fill: fillColor,
            stroke: strokeColor,
            fillOpacity: fillColorOpacity,
          }}
          transition={{ duration: isSourceHovered ? 0 : 0.5 }}
          strokeWidth={nodeStrokeWidth}
          onMouseEnter={() => setHoveredNode({ id: node.id, side: "source" })}
          onMouseLeave={() => setHoveredNode(null)}
          className="cursor-pointer transition-all duration-0"
            />
            {/* no striped overlay on source copies */}
            <motion.text
              x={0}
              y={textOffset}
          textAnchor="middle"
          dominantBaseline="middle"
          className="font-semibold select-none pointer-events-none"
          initial={false}
          animate={{
            fill: textColor,
          }}
          transition={{ duration: isSourceHovered ? 0 : 0.5 }}
          style={{
            fontSize: `${textFontSize}px`
          }}
        >
          {`${node.id}+`}
        </motion.text>
      </motion.g>
    );
  });

  const targetNodes = nodes.map((node) => {
    const isMatched = (stage >= 3 || forceShowMatching) && nodeMatching.some(([_, to]) => to === node.id);
    const isDriver = (stage >= 3 || forceShowMatching) && (nodeDrivers as readonly number[]).includes(node.id);
    const isTargetHovered = hoveredNode?.id === node.id && (hoveredNode.side === "target" || hoveredNode.side === "both");

    let fillColor: string;
    let strokeColor: string;
    let textColor: string;
    let fillColorOpacity: number;
    let currentRadius = nodeRadius;

    if (isDriver) {
      fillColor = styleConfig.nodes.driver.fillColor;
      strokeColor = styleConfig.nodes.driver.strokeColor;
      textColor = styleConfig.nodes.driver.textColor;
      fillColorOpacity = styleConfig.nodes.driver.fillColorOpacity;
    } else if (isMatched) {
      fillColor = styleConfig.nodes.matching.fillColor;
      strokeColor = styleConfig.nodes.matching.strokeColor;
      textColor = styleConfig.nodes.matching.textColor;
      fillColorOpacity = styleConfig.nodes.matching.fillColorOpacity;
    } else {
      fillColor = styleConfig.nodes.regular.fillColor;
      strokeColor = styleConfig.nodes.regular.strokeColor;
      textColor = styleConfig.nodes.regular.textColor;
      fillColorOpacity = styleConfig.nodes.regular.fillColorOpacity;
    }

    if (isTargetHovered) {
      if (styleConfig.nodes.hover.fillColor !== "no-change") fillColor = styleConfig.nodes.hover.fillColor;
      if (styleConfig.nodes.hover.strokeColor !== "no-change") strokeColor = styleConfig.nodes.hover.strokeColor;
      if (styleConfig.nodes.hover.textColor !== "no-change") textColor = styleConfig.nodes.hover.textColor;
      if (styleConfig.nodes.hover.fillColorOpacity !== "no-change") fillColorOpacity = styleConfig.nodes.hover.fillColorOpacity;
      if (styleConfig.nodes.hover.scale !== "no-change") currentRadius *= styleConfig.nodes.hover.scale;
    }

    const stripePatternId = stripedNodeSet.has(node.id) ? `stripe-node-${title?.replace(/\s+/g, '-')}-${node.id}-target` : null;
    if (stripePatternId) {
      registerStripePattern(stripePatternId, strokeColor, 0.45);
    }

    return (
      <motion.g
        key={`node-target-${node.id}`}
        layoutId={`node-group-right-${node.id}`}
        layout
        initial={{ x: node.origX, y: node.origY, opacity: 0 }}
        animate={{
          x: node.right.x,
          y: node.right.y,
          opacity: 1
        }}
        exit={{ x: node.origX, y: node.origY, opacity: 0 }}
        transition={{ type: "spring", stiffness: 60, damping: 15 }}
      >
        <motion.circle
          cx={0}
          cy={0}
          r={currentRadius}
          initial={false}
          animate={{
            fill: fillColor,
            stroke: strokeColor,
            fillOpacity: fillColorOpacity,
          }}
          transition={{ duration: isTargetHovered ? 0 : 0.5 }}
          strokeWidth={nodeStrokeWidth}
          onMouseEnter={() => setHoveredNode({ id: node.id, side: "target" })}
          onMouseLeave={() => setHoveredNode(null)}
          className="cursor-pointer transition-all duration-0"
            />
            {stripePatternId ? (
              <motion.circle
                cx={0}
                cy={0}
                r={currentRadius}
                initial={false}
                animate={{ opacity: 1 }}
                fill={`url(#${stripePatternId})`}
                stroke="none"
                pointerEvents="none"
              />
            ) : null}
            <motion.text
              x={0}
              y={textOffset}
          textAnchor="middle"
          dominantBaseline="middle"
          className="font-semibold select-none pointer-events-none"
          initial={false}
          animate={{
            fill: textColor,
          }}
          transition={{ duration: isTargetHovered ? 0 : 0.5 }}
          style={{
            fontSize: `${textFontSize}px`
          }}
        >
          {`${node.id}-`}
        </motion.text>
      </motion.g>
    );
  });

  return (
    <g className="bipartite-network">
      {stripeDefs.length > 0 && <defs>{stripeDefs}</defs>}
      {title && (
        <text
          x={50}
          y={6}
          fill="currentColor"
          textAnchor="middle"
          dominantBaseline="hanging"
          className="font-bold text-ink/75 uppercase tracking-[0.2em] select-none pointer-events-none"
          style={{ fontSize: `${titleFontSize || textFontSize * 1.1}px` }}
        >
          {title}
        </text>
      )}

      {isBipartiteFull && (
        <g className="bipartite-guides">
          {nodes.map((node) => {
            const x1 = node.left.x;
            const y1 = node.left.y;
            const x2 = node.right.x;
            const y2 = node.right.y;

            const angle = Math.atan2(y2 - y1, x2 - x1);
            const padding = nodeRadius + 0.7;

            return (
              <line
                key={`guide-${node.id}`}
                x1={x1 + padding * Math.cos(angle)}
                y1={y1 + padding * Math.sin(angle)}
                x2={x2 - padding * Math.cos(angle)}
                y2={y2 - padding * Math.sin(angle)}
                stroke={styleConfig.edges.virtual.color}
                strokeDasharray={styleConfig.edges.virtual.dashed ? styleConfig.edges.virtual.dashArray : "none"}
                strokeOpacity={styleConfig.edges.virtual.colorOpacity}
                strokeWidth={edgeStrokeWidth}
              />
            );
          })}
        </g>
      )}

      {isBipartiteFull && (
        <g className="bipartite-labels">
          {isHorizontal ? (
            <>
              <foreignObject x="0" y={APP_CONFIG.visual.network.layout.y_source - nodeRadius - APP_CONFIG.visual.network.layout.label_offset_source_horizontal} width="100" height="6">
                <div
                  className="text-ink/75 font-bold uppercase tracking-[0.2em] flex items-center justify-center gap-1"
                  style={{ fontSize: `${textFontSize * 0.7}px` }}
                >
                  {t("network.labels.sourceCopies")}
                </div>
              </foreignObject>
              <foreignObject x="0" y={APP_CONFIG.visual.network.layout.y_target + nodeRadius + APP_CONFIG.visual.network.layout.label_offset_target_horizontal} width="100" height="6">
                <div
                  className="text-ink/75 font-bold uppercase tracking-[0.2em] flex items-center justify-center gap-1"
                  style={{ fontSize: `${textFontSize * 0.7}px` }}
                >
                  {t("network.labels.targetCopies")}
                </div>
              </foreignObject>
            </>
          ) : (
            <>
              <foreignObject x={APP_CONFIG.visual.network.layout.x_source - nodeRadius - APP_CONFIG.visual.network.layout.label_offset_source_vertical} y="0" width="8" height="100">
                <div
                  className="h-full text-ink/75 font-bold uppercase tracking-[0.2em] flex items-center justify-center [writing-mode:vertical-rl]"
                  style={{ fontSize: `${textFontSize * 0.7}px` }}
                >
                  {t("network.labels.sourceCopies")}
                </div>
              </foreignObject>
              <foreignObject x={APP_CONFIG.visual.network.layout.x_target + nodeRadius - APP_CONFIG.visual.network.layout.label_offset_target_vertical} y="0" width="8" height="100">
                <div
                  className="h-full text-ink/75 font-bold uppercase tracking-[0.2em] flex items-center justify-center [writing-mode:vertical-rl]"
                  style={{ fontSize: `${textFontSize * 0.7}px` }}
                >
                  {t("network.labels.targetCopies")}
                </div>
              </foreignObject>
            </>
          )}
        </g>
      )}

      {renderEdges && edgesToRender.map(([from, to]) => {
        const nodeFrom = nodes.find((n) => n.id === from);
        const nodeTo = nodes.find((n) => n.id === to);
        if (!nodeFrom || !nodeTo) return null;

        const x1 = nodeFrom.left.x;
        const y1 = nodeFrom.left.y;
        const x2 = nodeTo.right.x;
        const y2 = nodeTo.right.y;

        const angle = Math.atan2(y2 - y1, x2 - x1);
        const padding = nodeRadius + 1.2;

        const targetX1 = x1 + padding * Math.cos(angle);
        const targetY1 = y1 + padding * Math.sin(angle);
        const targetX2 = x2 - padding * Math.cos(angle);
        const targetY2 = y2 - padding * Math.sin(angle);

        const edgeKey = `${from}-${to}`;

        const isEdgeOnExchangePath =
          !!exchangePresentation?.active &&
          exchangePresentation.pathEdges.some((e) => edgeEq(e, [from, to]));

        // step 2 / animating 时，底层路径完全隐藏
        if (exchangePresentation?.active && exchangePresentation.phase === "animating" && isEdgeOnExchangePath) {
          return null;
        }

        const previewKind =
          exchangePresentation?.active && exchangePresentation.phase === "preview"
            ? previewStyleMap.get(edgeKey)
            : undefined;

        const isHighlighted = highlightEdgesToUse.some(([hFrom, hTo]) => hFrom === from && hTo === to);
        const isMatchedOnCurrentSemantic = edgeMatching.some(([mFrom, mTo]) => mFrom === from && mTo === to);

        const isHovered = hoveredNode && (
          hoveredNode.side === "both"
            ? (from === hoveredNode.id || to === hoveredNode.id)
            : (hoveredNode.side === "source" ? from === hoveredNode.id : to === hoveredNode.id)
        );

        const pathKind = previewKind || (isHighlighted ? (isMatchedOnCurrentSemantic ? "matched" : "alternating") : undefined);

        let strokeColor: string;
        let strokeWidth = edgeStrokeWidth;
        let strokeDasharray: string;
        let opacity: number;

        // Base styles
        if (pathKind === "matched") {
          const style = styleConfig.edges.alternativeMatching;
          strokeColor = style.color;
          strokeDasharray = style.dashed ? style.dashArray : "none";
          opacity = style.colorOpacity;
        } else if (pathKind === "alternating") {
          const style = styleConfig.edges.alternativeNonMatching;
          strokeColor = style.color;
          strokeDasharray = style.dashed ? style.dashArray : "none";
          opacity = style.colorOpacity;
        } else if (isMatchedOnCurrentSemantic) {
          strokeColor = styleConfig.edges.matching.color;
          strokeDasharray = styleConfig.edges.matching.dashed ? styleConfig.edges.matching.dashArray : "none";
          opacity = styleConfig.edges.matching.colorOpacity;
        } else {
          strokeColor = styleConfig.edges.nonMatching.color;
          strokeDasharray = styleConfig.edges.nonMatching.dashed ? styleConfig.edges.nonMatching.dashArray : "none";
          opacity = styleConfig.edges.nonMatching.colorOpacity;
        }

        // Hover overrides
        if (isHovered) {
          if (styleConfig.edges.hover.color !== "no-change") strokeColor = styleConfig.edges.hover.color;
          if (styleConfig.edges.hover.dashed !== "no-change") {
            strokeDasharray = styleConfig.edges.hover.dashed ? (styleConfig.edges.hover.dashArray as string) : "none";
          }
          if (styleConfig.edges.hover.colorOpacity !== "no-change") opacity = styleConfig.edges.hover.colorOpacity;
          if (styleConfig.edges.hover.scale !== "no-change") strokeWidth *= styleConfig.edges.hover.scale;
        }

        return (
          <motion.line
            key={`edge-bipartite-${from}-${to}`}
            layoutId={`edge-${from}-${to}`}
            initial={false}
            animate={{
              x1: targetX1,
              y1: targetY1,
              x2: targetX2,
              y2: targetY2,
              stroke: strokeColor,
              strokeWidth,
              strokeDasharray,
              opacity
            }}
            transition={{
              stroke: { duration: isHovered ? 0 : 0.12 },
              default: { type: "spring", stiffness: 60, damping: 15 }
            }}
          />
        );
      })}

      {exchangePresentation?.active && exchangePresentation.phase === "animating" && (
        <ExchangeOverlay
          tokens={exchangeTokens}
          getGeometry={(edge) => getBipartiteEdgeGeometry(edge, nodes, nodeRadius)}
          edgeStrokeWidth={edgeStrokeWidth}
          styleConfig={styleConfig}
        />
      )}

      {sourceNodes}
      {targetNodes}
    </g>
  );
}
