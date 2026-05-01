"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { APP_CONFIG, hexToRgb } from "@/lib/config";

export type ThemeMode = "light" | "dark";
export type ThemePreference = ThemeMode | "system";

type ThemeContextType = {
  theme: ThemeMode;
  preference: ThemePreference;
  setPreference: (preference: ThemePreference) => void;
  toggleTheme: () => void;
};

const THEME_STORAGE_KEY = "claps-theme";
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

function getSystemTheme(): ThemeMode {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [preference, setPreferenceState] = useState<ThemePreference>("system");
  const [theme, setTheme] = useState<ThemeMode>("light");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY) as ThemePreference | null;
    if (stored === "light" || stored === "dark" || stored === "system") {
      setPreferenceState(stored);
      if (stored === "light" || stored === "dark") {
        setTheme(stored);
      } else {
        setTheme(getSystemTheme());
      }
    } else {
      setTheme(getSystemTheme());
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (preference === "system") {
      setTheme(getSystemTheme());
    } else {
      setTheme(preference);
    }
  }, [preference]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (event: MediaQueryListEvent) => {
      if (preference === "system") {
        setTheme(event.matches ? "dark" : "light");
      }
    };

    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, [preference]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    
    // Toggle dark class
    root.classList.toggle("dark", theme === "dark");
    root.dataset.theme = theme;
    root.style.setProperty("color-scheme", theme);

    // Inject semantic colors as RGB triplets for Tailwind
    const colors = APP_CONFIG.colors[theme];
    Object.entries(colors).forEach(([key, value]) => {
      if (typeof value === "string" && value.startsWith("#")) {
        root.style.setProperty(`--color-${key}`, hexToRgb(value));
      }
    });

    // Inject other visual properties
    const visual = APP_CONFIG.visual;
    root.style.setProperty("--mesh-bg", visual.gradients.mesh[theme]);
    root.style.setProperty("--body-bg", visual.gradients.body[theme]);
    root.style.setProperty("--card-shadow", visual.shadows.card);

  }, [theme]);

  const setPreference = useCallback((value: ThemePreference) => {
    setPreferenceState(value);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, value);
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setPreference(theme === "dark" ? "light" : "dark");
  }, [setPreference, theme]);

  const value = useMemo(
    () => ({
      theme,
      preference,
      setPreference,
      toggleTheme
    }),
    [theme, preference, setPreference, toggleTheme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useThemeMode() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useThemeMode must be used within a ThemeProvider");
  }
  return context;
}
