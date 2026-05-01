import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "rgb(var(--color-ink) / <alpha-value>)",
        mist: "rgb(var(--color-mist) / <alpha-value>)",
        paper: "rgb(var(--color-paper) / <alpha-value>)",
        surface: "rgb(var(--color-surface) / <alpha-value>)",
        accent: "rgb(var(--color-accent) / <alpha-value>)",
        "accent-secondary": "rgb(var(--color-accent_secondary) / <alpha-value>)"
      },
      boxShadow: {
        card: "var(--card-shadow)"
      },
      backgroundImage: {
        mesh: "var(--mesh-bg)"
      }
    }
  },
  plugins: []
};

export default config;
