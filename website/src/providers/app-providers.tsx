"use client";

import type React from "react";

import { I18nProvider } from "./i18n-provider";
import { ThemeProvider } from "./theme-provider";
import { VisualizationSettingsProvider } from "./visualization-settings-provider";

export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <I18nProvider>
      <ThemeProvider>
        <VisualizationSettingsProvider>{children}</VisualizationSettingsProvider>
      </ThemeProvider>
    </I18nProvider>
  );
}
