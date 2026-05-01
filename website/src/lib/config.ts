
const PALETTE = {
  // Base primitives
  red: { light: "#E63946", dark: "#fb7185", soft: "#fca5a5", deep: "#be123c" },
  blue: { light: "#0077B6", dark: "#7dd3fc", soft: "#93c5fd", deep: "#1e3a8a" },
  orange: { light: "#f97316", dark: "#fb923c" },
  pink: { light: "#ec4899", dark: "#f472b6" },
  purple: { light: "#8b5cf6", dark: "#a78bfa" },
  yellow: { light: "#facc15", dark: "#fde68a" },
  green: { light: "#22c55e", dark: "#4ade80" },
  gray: { light: "#64748b", dark: "#94a3b8", soft: "#e2e8f0", deep: "#1e293b" },
  slate: { light: "#16212b", dark: "#020617" },
};

/**
 * Converts a hex color to an RGB triplet string (e.g., "255 255 255")
 * for use in CSS variables with Tailwind's opacity modifier.
 */
export function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r} ${g} ${b}`;
}

export const APP_CONFIG = {
  colors: {
    light: {
      // Single layer (A-C)
      driver: PALETTE.red.light,
      matched: "#1f2937",
      matching_edge: PALETTE.red.light,
      non_matching_edge: PALETTE.gray.light,
      alternating_edge: PALETTE.blue.light,

      // Duplex / Triple Layer (D+)
      layer1: PALETTE.orange.light,
      layer2: PALETTE.pink.light,
      layer3: PALETTE.purple.light,

      // Specific sets for overlap animations
      cds: "#f87171",
      cms: "#0ea5e9",
      common_driver: PALETTE.red.deep,

      // Interactions
      hover: PALETTE.yellow.light,
      virtual: "rgba(22, 33, 43, 0.05)",
      ghost: "rgba(22, 33, 43, 0.02)",

      // UI
      ink: PALETTE.slate.light,
      surface: "#ffffff",
      paper: "#fbf9f3",
      mist: "#eef3f5",
      accent: PALETTE.orange.light,
      accent_secondary: PALETTE.pink.light,

      // Aux
      source: "#3b82f6",
      relay: "#facd15",
      target: "#60a5fa",
    },
    dark: {
      driver: PALETTE.red.dark,
      matched: "#cbd5f1",
      matching_edge: PALETTE.red.dark,
      non_matching_edge: PALETTE.gray.dark,
      alternating_edge: PALETTE.blue.dark,

      layer1: PALETTE.orange.dark,
      layer2: PALETTE.pink.dark,
      layer3: PALETTE.purple.dark,

      cds: "#fca5a5",
      cms: "#38bdf8",
      common_driver: PALETTE.red.soft,

      hover: PALETTE.yellow.dark,
      virtual: "rgba(255, 255, 255, 0.12)",
      ghost: "rgba(255, 255, 255, 0.05)",

      ink: "#e5ecf5",
      surface: "#0f172a",
      paper: "#020617",
      mist: "#1e293b",
      accent: PALETTE.orange.dark,
      accent_secondary: PALETTE.pink.dark,

      source: "#93c5fd",
      relay: "#fde68a",
      target: "#a5b4fc",
    },
    charts: {
      claps: "#16212b",
      ilp: "#365486",
      clapg: "#0b7285",
      rsu: "#d97706"
    },
  },
  visual: {
    shadows: {
      card: "0 30px 80px -40px rgba(22, 33, 43, 0.25)",
    },
    gradients: {
      // Mesh backgrounds
      mesh: {
        light: "radial-gradient(circle at 10% 20%, rgba(249,115,22,0.12), transparent 28%), radial-gradient(circle at 90% 0%, rgba(236,72,153,0.1), transparent 26%), linear-gradient(180deg, rgba(255,255,255,0.92), rgba(238,243,245,0.72))",
        dark: "radial-gradient(circle at 10% 20%, rgba(251,146,60,0.15), transparent 28%), radial-gradient(circle at 90% 0%, rgba(244,114,182,0.12), transparent 26%), linear-gradient(180deg, rgba(2,6,23,0.95), rgba(15,23,42,0.85))"
      },
      // Body backgrounds
      body: {
        light: "radial-gradient(circle at top left, rgba(11, 114, 133, 0.16), transparent 26%), radial-gradient(circle at top right, rgba(199, 119, 26, 0.14), transparent 22%), linear-gradient(180deg, #fbf9f3 0%, #eef3f5 100%)",
        dark: "radial-gradient(circle at top left, rgba(14, 116, 144, 0.34), transparent 32%), radial-gradient(circle at top right, rgba(202, 138, 4, 0.22), transparent 26%), linear-gradient(180deg, #020617 0%, #0f172a 100%)"
      }
    },
    node_hover_scale: 1.25,
    edge_hover_scale: 1.5,
    network: {
      node_radius: 3.8,
      node_stroke_width: 0.6,
      node_fill_opacity: 0.06,
      text_size: 3.8,
      text_offset: 0.6,
      edge_width: 0.6,
      dash_alt: "2 1.5",
      dash_virtual: "2 3",
      arrow_size: { width: 4, height: 3 },
      edge_padding: 1.2,
      edge_padding_directed: 4.5,
      hover_multiplier: 1.5,
      match_multiplier: 1.2,
      layout: {
        // Horizontal orientation (Top-Bottom rows)
        margin: 15,
        span: 70,
        y_source: 25,
        y_target: 70,
        label_offset_source_horizontal: 6.5,
        label_offset_target_horizontal: 3.5,

        // Vertical orientation (Left-Right columns)
        x_source: 20,
        x_target: 80,
        y_margin: 20,
        y_span: 55,
        label_offset_source_vertical: 6.5,
        label_offset_target_vertical: -1.0,

        // 节点的环向排列
        center: 50,
        radius: 35
      }
    }
  },
  opacity: {
    duplex_node_bg: 0.15,
  }
} as const;
