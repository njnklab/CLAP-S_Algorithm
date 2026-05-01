"use client";

import React, { useMemo } from "react";
import { motion } from "framer-motion";
import { primerEdges, primerMatchingOptions } from "../data/demoData";
import { NetworkProps, DEFAULT_STYLE_CONFIG, EdgeTuple, LineGeometry } from "./types";
import { ExchangeOverlay, buildPreviewStyleMap, buildStep2ExchangeTokens } from "./exchangeOverlay";

function edgeEq(a: EdgeTuple, b: EdgeTuple) {
  return a[0] === b[0] && a[1] === b[1];
}

function getDirectedEdgeGeometry(
  edge: EdgeTuple,
  nodes: NetworkProps["nodes"],
  nodeRadius: number,
  allEdges?: readonly EdgeTuple[]
): LineGeometry | null {
  const [from, to] = edge;
  const nodeFrom = nodes.find((n) => n.id === from);
  const nodeTo = nodes.find((n) => n.id === to);
  if (!nodeFrom || !nodeTo) return null;

  const fromX = nodeFrom.origX;
  const fromY = nodeFrom.origY;
  const toX = nodeTo.origX;
  const toY = nodeTo.origY;

  const angle = Math.atan2(toY - fromY, toX - fromX);
  const startPadding = nodeRadius + 1.2;
  const endPadding = nodeRadius + 4.5;

  const hasReverse = allEdges?.some((e) => e[0] === edge[1] && e[1] === edge[0]);
  const offset = hasReverse ? 1.6 : 0;
  const normalX = Math.cos(angle + Math.PI / 2) * offset;
  const normalY = Math.sin(angle + Math.PI / 2) * offset;

  return {
    x1: fromX + startPadding * Math.cos(angle) + normalX,
    y1: fromY + startPadding * Math.sin(angle) + normalY,
    x2: toX - endPadding * Math.cos(angle) + normalX,
    y2: toY - endPadding * Math.sin(angle) + normalY,
  };
}

export function DirectedNetwork({
  stage,
  nodes,
  hoveredNode,
  setHoveredNode,
  matchingCase = "A",
  styleConfig = DEFAULT_STYLE_CONFIG,
  customEdges,
  customMatching,
  customDrivers,
  customHighlightEdges,
  exchangePresentation,
  forceShowMatching,
  title,
  titleFontSize,
  renderEdges = true,
  stripedNodeIds
}: NetworkProps) {
  const isFinalResult = stage >= 3 || forceShowMatching;
  
  const { 
    radius: nodeRadius, 
    strokeWidth: nodeStrokeWidth, 
    textFontSize, 
    textOffset 
  } = styleConfig.nodes.all;
  
  const { strokeWidth: edgeStrokeWidth } = styleConfig.edges.all;

  const edgesToRender = customEdges || primerEdges;
  const defaultMatching = (primerMatchingOptions[matchingCase] || primerMatchingOptions["A"]).edges;
  const defaultDrivers = (primerMatchingOptions[matchingCase] || primerMatchingOptions["A"]).drivers as readonly number[];

  const matchingToUse = customMatching || defaultMatching;
  const driversToUse = customDrivers || defaultDrivers;
  const highlightEdgesToUse = customHighlightEdges || [];

  // 边的语义：step 2 仍以 beforeMatching 为基础；preview 由 previewStyleMap 覆盖路径边样式
  const edgeMatching = exchangePresentation?.active
    ? exchangePresentation.beforeMatching
    : matchingToUse;

  // 节点的语义：step2 animating 仍显示 before；step2 preview 切到 after；step3 保持 after
  const nodeMatching =
    exchangePresentation?.active
      ? exchangePresentation.phase === "preview"
        ? exchangePresentation.afterMatching
        : exchangePresentation.beforeMatching
      : matchingToUse;

  const nodeDrivers =
    exchangePresentation?.active
      ? exchangePresentation.phase === "preview"
        ? exchangePresentation.afterDrivers
        : exchangePresentation.beforeDrivers
      : driversToUse;

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

  const nodeElements = nodes.map((node) => {
    const isDriver = isFinalResult && nodeDrivers.includes(node.id);
    const isMatchedAsTarget = isFinalResult && nodeMatching.some(([_, to]) => to === node.id);
    const isHovered = hoveredNode?.id === node.id;

    let fillColor: string;
    let strokeColor: string;
    let textColor: string;
    let fillColorOpacity: number;
    let currentRadius = nodeRadius;

    if (isFinalResult && isMatchedAsTarget) {
      fillColor = styleConfig.nodes.matching.fillColor;
      strokeColor = styleConfig.nodes.matching.strokeColor;
      textColor = styleConfig.nodes.matching.textColor;
      fillColorOpacity = styleConfig.nodes.matching.fillColorOpacity;
    } else if (isFinalResult && isDriver) {
      fillColor = styleConfig.nodes.driver.fillColor;
      strokeColor = styleConfig.nodes.driver.strokeColor;
      textColor = styleConfig.nodes.driver.textColor;
      fillColorOpacity = styleConfig.nodes.driver.fillColorOpacity;
    } else {
      fillColor = styleConfig.nodes.regular.fillColor;
      strokeColor = styleConfig.nodes.regular.strokeColor;
      textColor = styleConfig.nodes.regular.textColor;
      fillColorOpacity = styleConfig.nodes.regular.fillColorOpacity;
    }

    if (isHovered) {
      if (styleConfig.nodes.hover.fillColor !== "no-change") fillColor = styleConfig.nodes.hover.fillColor;
      if (styleConfig.nodes.hover.strokeColor !== "no-change") strokeColor = styleConfig.nodes.hover.strokeColor;
      if (styleConfig.nodes.hover.textColor !== "no-change") textColor = styleConfig.nodes.hover.textColor;
      if (styleConfig.nodes.hover.fillColorOpacity !== "no-change") fillColorOpacity = styleConfig.nodes.hover.fillColorOpacity as number;
      if (styleConfig.nodes.hover.scale !== "no-change") currentRadius *= styleConfig.nodes.hover.scale;
    }

    const stripePatternId = stripedNodeSet.has(node.id) ? `stripe-node-${title?.replace(/\s+/g, '-')}-${node.id}` : null;
    if (stripePatternId) {
      registerStripePattern(stripePatternId, strokeColor, 0.45);
    }

    return (
      <motion.g
        key={`node-directed-${node.id}`}
        layoutId={stage === 0 ? `node-group-left-${node.id}` : `node-group-right-${node.id}`}
        layout
        initial={{ x: node.origX, y: node.origY, opacity: 0 }}
        animate={{
          x: node.origX,
          y: node.origY,
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
          transition={{ duration: isHovered ? 0 : 0.5 }}
          strokeWidth={nodeStrokeWidth}
          className="transition-all duration-0 cursor-pointer"
          onMouseEnter={() => setHoveredNode({ id: node.id, side: "both" })}
          onMouseLeave={() => setHoveredNode(null)}
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
          transition={{ duration: isHovered ? 0 : 0.5 }}
          style={{
            fontSize: `${textFontSize}px`
          }}
        >
          {node.id}
        </motion.text>
      </motion.g>
    );
  });

  return (
    <g className="directed-network">
      {stripeDefs.length > 0 && <defs>{stripeDefs}</defs>}
      {title && (
        <text
          x={50}
          y={2}
          fill="currentColor"
          textAnchor="middle"
          dominantBaseline="hanging"
          className="font-bold text-ink/75 uppercase tracking-[0.2em] select-none pointer-events-none"
          style={{ fontSize: `${titleFontSize || textFontSize * 1.1}px` }}
        >
          {title}
        </text>
      )}

      {renderEdges && edgesToRender.map(([from, to]) => {
        const nodeFrom = nodes.find((n) => n.id === from);
        const nodeTo = nodes.find((n) => n.id === to);
        if (!nodeFrom || !nodeTo) return null;

        const geometry = getDirectedEdgeGeometry([from, to], nodes, nodeRadius, edgesToRender);
        if (!geometry) return null;

        const { x1: targetX1, y1: targetY1, x2: targetX2, y2: targetY2 } = geometry;

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

        const pathKind =
          previewKind || (isHighlighted ? (isMatchedOnCurrentSemantic ? "matched" : "alternating") : undefined);

        let strokeColor: string;
        let strokeWidth = edgeStrokeWidth;
        let markerEnd = "url(#arrowhead)";
        let strokeDasharray: string;
        let opacity: number;

        // Base styles
        if (pathKind === "matched") {
          const style = styleConfig.edges.alternativeMatching;
          strokeColor = style.color;
          strokeDasharray = style.dashed ? style.dashArray : "none";
          opacity = style.colorOpacity;
          markerEnd = "url(#arrowhead-alt-matched)";
        } else if (pathKind === "alternating") {
          const style = styleConfig.edges.alternativeNonMatching;
          strokeColor = style.color;
          strokeDasharray = style.dashed ? style.dashArray : "none";
          opacity = style.colorOpacity;
          markerEnd = "url(#arrowhead-alternating)";
        } else if (isMatchedOnCurrentSemantic) {
          strokeColor = styleConfig.edges.matching.color;
          strokeDasharray = styleConfig.edges.matching.dashed ? styleConfig.edges.matching.dashArray : "none";
          opacity = styleConfig.edges.matching.colorOpacity;
          markerEnd = "url(#arrowhead-matched)";
        } else {
          strokeColor = styleConfig.edges.nonMatching.color;
          strokeDasharray = styleConfig.edges.nonMatching.dashed ? styleConfig.edges.nonMatching.dashArray : "none";
          opacity = styleConfig.edges.nonMatching.colorOpacity;
          markerEnd = "url(#arrowhead-regular)";
        }

        // Hover overrides
        if (isHovered) {
          if (styleConfig.edges.hover.color !== "no-change") strokeColor = styleConfig.edges.hover.color;
          if (styleConfig.edges.hover.dashed !== "no-change") {
            strokeDasharray = styleConfig.edges.hover.dashed ? (styleConfig.edges.hover.dashArray as string) : "none";
          }
          if (styleConfig.edges.hover.colorOpacity !== "no-change") opacity = styleConfig.edges.hover.colorOpacity as number;
          if (styleConfig.edges.hover.scale !== "no-change") strokeWidth *= styleConfig.edges.hover.scale;
          markerEnd = "url(#arrowhead-hover)";
        }

        return (
          <motion.line
            key={`edge-directed-${from}-${to}`}
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
            markerEnd={markerEnd}
          />
        );
      })}

      {exchangePresentation?.active && exchangePresentation.phase === "animating" && (
        <ExchangeOverlay
          tokens={exchangeTokens}
          getGeometry={(edge) => getDirectedEdgeGeometry(edge, nodes, nodeRadius, edgesToRender)}
          edgeStrokeWidth={edgeStrokeWidth}
          directed
          styleConfig={styleConfig}
        />
      )}

      {nodeElements}
    </g>
  );
}
