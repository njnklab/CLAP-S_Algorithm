"use client";

import { createContext, useCallback, useContext, useMemo, useState } from "react";

import { DEFAULT_STYLE_CONFIG, type NetworkStyleConfig } from "@/features/animations/network/types";
import { deepMerge, setByPath, type DeepPartial } from "@/lib/object";
import { APP_CONFIG } from "@/lib/config";
import { useThemeMode } from "./theme-provider";

type VisualizationSettingsContextType = {
  styleConfig: NetworkStyleConfig;
  applyPatch: (patch: DeepPartial<NetworkStyleConfig>) => void;
  setValue: (path: string, value: number | string | boolean) => void;
  reset: () => void;
};

const VisualizationSettingsContext = createContext<VisualizationSettingsContextType | undefined>(undefined);

const DARK_STYLE_OVERRIDES: DeepPartial<NetworkStyleConfig> = {
  nodes: {
    matching: {
      strokeColor: APP_CONFIG.colors.dark.matched,
      textColor: APP_CONFIG.colors.dark.matched,
      fillColor: APP_CONFIG.colors.dark.surface,
      fillColorOpacity: 0.3
    },
    driver: {
      strokeColor: APP_CONFIG.colors.dark.driver,
      textColor: APP_CONFIG.colors.dark.driver,
      fillColor: APP_CONFIG.colors.dark.mist,
      fillColorOpacity: 0.35
    },
    regular: {
      strokeColor: APP_CONFIG.colors.dark.non_matching_edge,
      textColor: APP_CONFIG.colors.dark.ink,
      fillColor: APP_CONFIG.colors.dark.paper,
      fillColorOpacity: 1
    },
    hover: {
      strokeColor: APP_CONFIG.colors.dark.hover,
      textColor: APP_CONFIG.colors.dark.hover
    }
  },
  edges: {
    matching: {
      color: APP_CONFIG.colors.dark.matching_edge
    },
    nonMatching: {
      color: APP_CONFIG.colors.dark.non_matching_edge
    },
    alternativeNonMatching: {
      color: APP_CONFIG.colors.dark.alternating_edge
    },
    alternativeMatching: {
      color: APP_CONFIG.colors.dark.matching_edge
    },
    virtual: {
      color: APP_CONFIG.colors.dark.virtual
    },
    hover: {
      color: APP_CONFIG.colors.dark.hover
    }
  }
};

const mergePartialOverrides = (...patches: DeepPartial<NetworkStyleConfig>[]): DeepPartial<NetworkStyleConfig> => {
  return deepMerge({} as NetworkStyleConfig, ...patches) as DeepPartial<NetworkStyleConfig>;
};

export function VisualizationSettingsProvider({ children }: { children: React.ReactNode }) {
  const { theme } = useThemeMode();
  const [overrides, setOverrides] = useState<DeepPartial<NetworkStyleConfig>>({});

  const baseConfig = useMemo(
    () => (theme === "dark" ? deepMerge(DEFAULT_STYLE_CONFIG, DARK_STYLE_OVERRIDES) : DEFAULT_STYLE_CONFIG),
    [theme]
  );

  const styleConfig = useMemo(
    () => deepMerge(baseConfig, overrides),
    [baseConfig, overrides]
  );

  const applyPatch = useCallback((patch: DeepPartial<NetworkStyleConfig>) => {
    setOverrides((prev) => mergePartialOverrides(prev ?? {}, patch));
  }, []);

  const setValue = useCallback((path: string, value: number | string | boolean) => {
    setOverrides((prev) => setByPath(prev ?? {}, path, value));
  }, []);

  const reset = useCallback(() => setOverrides({}), []);

  const contextValue = useMemo(
    () => ({
      styleConfig,
      applyPatch,
      setValue,
      reset
    }),
    [styleConfig, applyPatch, setValue, reset]
  );

  return <VisualizationSettingsContext.Provider value={contextValue}>{children}</VisualizationSettingsContext.Provider>;
}

export function useVisualizationSettings() {
  const context = useContext(VisualizationSettingsContext);
  if (!context) {
    throw new Error("useVisualizationSettings must be used within a VisualizationSettingsProvider");
  }
  return context;
}
