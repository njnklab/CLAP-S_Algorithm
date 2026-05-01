import { primerMatchingOptions } from "../data/demoData";
import { APP_CONFIG } from "@/lib/config";

export interface NetworkNode {
  id: number;
  origX: number;
  origY: number;
  left: { x: number; y: number };
  right: { x: number; y: number };
}

export type MatchingCase = keyof typeof primerMatchingOptions;

export interface NetworkStyleConfig {
  nodes: {
    all: {
      radius: number;
      strokeWidth: number;
      textFontSize: number;
      textOffset: number;
    };
    matching: {
      strokeColor: string;
      textColor: string;
      fillColor: string;
      fillColorOpacity: number;
    };
    driver: {
      strokeColor: string;
      textColor: string;
      fillColor: string;
      fillColorOpacity: number;
    };
    regular: {
      strokeColor: string;
      textColor: string;
      fillColor: string;
      fillColorOpacity: number;
    };
    hover: {
      strokeColor: string | "no-change";
      textColor: string | "no-change";
      fillColor: string | "no-change";
      fillColorOpacity: number | "no-change";
      scale: number | "no-change";
    };
  };
  edges: {
    all: {
      strokeWidth: number;
    };
    matching: {
      color: string;
      colorOpacity: number;
      dashed: boolean;
      dashArray: string;
    };
    nonMatching: {
      color: string;
      colorOpacity: number;
      dashed: boolean;
      dashArray: string;
    };
    alternativeNonMatching: {
      color: string;
      colorOpacity: number;
      dashed: boolean;
      dashArray: string;
    };
    alternativeMatching: {
      color: string;
      colorOpacity: number;
      dashed: boolean;
      dashArray: string;
    };
    virtual: {
      color: string;
      colorOpacity: number;
      dashed: boolean;
      dashArray: string;
    };
    hover: {
      color: string | "no-change";
      colorOpacity: number | "no-change";
      dashed: boolean | "no-change";
      dashArray: string | "no-change";
      scale: number | "no-change";
    };
  };
  layout: {
    orientation: "horizontal" | "vertical";
  };
}

export interface NetworkProps {
  stage: number;
  nodes: NetworkNode[];
  matchingCase?: MatchingCase;
  hoveredNode: { id: number; side: "source" | "target" | "both" } | null;
  setHoveredNode: (node: { id: number; side: "source" | "target" | "both" } | null) => void;
  layout?: "vertical" | "horizontal";
  styleConfig?: NetworkStyleConfig;
  // Custom data overrides for different animations
  customEdges?: readonly (readonly [number, number])[];
  customMatching?: readonly (readonly [number, number])[];
  customDrivers?: readonly number[];
  customHighlightEdges?: readonly (readonly [number, number])[];
  forceShowMatching?: boolean;
  title?: string;
  titleFontSize?: number;
  renderEdges?: boolean;
  exchangePresentation?: ExchangePresentation;
  stripedNodeIds?: number[];
}

export type EdgeTuple = readonly [number, number];

export type LineGeometry = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type ExchangePhase = "animating" | "preview";

export type ExchangePresentation = {
  active: boolean;
  phase: ExchangePhase;
  pathKey: string;
  pathEdges: readonly EdgeTuple[];
  beforeMatching: readonly EdgeTuple[];
  afterMatching: readonly EdgeTuple[];
  beforeDrivers: number[];
  afterDrivers: number[];
};

export const DEFAULT_STYLE_CONFIG: NetworkStyleConfig = {
  nodes: {
    all: {
      radius: APP_CONFIG.visual.network.node_radius,
      strokeWidth: APP_CONFIG.visual.network.node_stroke_width,
      textFontSize: APP_CONFIG.visual.network.text_size,
      textOffset: APP_CONFIG.visual.network.text_offset
    },
    matching: {
      strokeColor: APP_CONFIG.colors.light.matched,
      textColor: APP_CONFIG.colors.light.matched,
      fillColor: APP_CONFIG.colors.light.matched,
      fillColorOpacity: APP_CONFIG.visual.network.node_fill_opacity,
    },
    driver: {
      strokeColor: APP_CONFIG.colors.light.driver,
      textColor: APP_CONFIG.colors.light.driver,
      fillColor: APP_CONFIG.colors.light.driver,
      fillColorOpacity: APP_CONFIG.visual.network.node_fill_opacity,
    },
    regular: {
      strokeColor: APP_CONFIG.colors.light.ink,
      textColor: APP_CONFIG.colors.light.ink,
      fillColor: APP_CONFIG.colors.light.surface,
      fillColorOpacity: APP_CONFIG.visual.network.node_fill_opacity,
    },
    hover: {
      strokeColor: APP_CONFIG.colors.light.hover,
      textColor: APP_CONFIG.colors.light.hover,
      fillColor: "no-change",
      fillColorOpacity: "no-change",
      scale: APP_CONFIG.visual.node_hover_scale,
    }
  },
  edges: {
    all: {
      strokeWidth: APP_CONFIG.visual.network.edge_width
    },
    matching: {
      color: APP_CONFIG.colors.light.matching_edge,
      colorOpacity: 1,
      dashed: false,
      dashArray: "none",
    },
    nonMatching: {
      color: APP_CONFIG.colors.light.non_matching_edge,
      colorOpacity: 0.5,
      dashed: false,
      dashArray: "none",
    },
    alternativeNonMatching: {
      color: APP_CONFIG.colors.light.alternating_edge,
      colorOpacity: 1,
      dashed: true,
      dashArray: APP_CONFIG.visual.network.dash_alt,
    },
    alternativeMatching: {
      color: APP_CONFIG.colors.light.matching_edge,
      colorOpacity: 1,
      dashed: true,
      dashArray: APP_CONFIG.visual.network.dash_alt,
    },
    virtual: {
      color: APP_CONFIG.colors.light.virtual,
      colorOpacity: 1,
      dashed: true,
      dashArray: APP_CONFIG.visual.network.dash_virtual,
    },
    hover: {
      color: APP_CONFIG.colors.light.hover,
      colorOpacity: 1,
      dashed: false,
      dashArray: "none",
      scale: APP_CONFIG.visual.edge_hover_scale,
    }
  },
  layout: {
    orientation: "horizontal"
  }
};
