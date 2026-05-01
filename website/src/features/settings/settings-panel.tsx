"use client";

import { useEffect } from "react";

import { useVisualizationSettings } from "@/providers/visualization-settings-provider";
import { useThemeMode } from "@/providers/theme-provider";
import { useI18n } from "@/providers/i18n-provider";
import { cn } from "@/lib/utils";

type SettingsPanelProps = {
  open: boolean;
  onClose: () => void;
};

export function SettingsPanel({ open, onClose }: SettingsPanelProps) {
  const { t } = useI18n();
  const { styleConfig, setValue, applyPatch, reset } = useVisualizationSettings();
  const { preference, setPreference } = useThemeMode();

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-end bg-black/40 px-4 py-6 sm:px-8"
      role="dialog"
      aria-modal="true"
      aria-label={t("settings.title")}
    >
      <button type="button" className="absolute inset-0 h-full w-full cursor-default" onClick={onClose} aria-hidden />

      <div className="relative z-[61] flex w-full max-w-md max-h-[calc(100vh-3rem)] flex-col rounded-3xl border border-white/20 bg-paper/95 p-6 text-sm text-ink shadow-card dark:border-white/10 dark:bg-slate-900/95 transition-all">
        {/* Header - Fixed */}
        <div className="mb-6 flex items-center justify-between shrink-0">
          <div>
            <div className="text-xs uppercase tracking-[0.28em] text-accent">{t("settings.eyebrow")}</div>
            <h3 className="mt-2 text-xl font-semibold text-ink">{t("settings.title")}</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-ink/10 text-ink transition hover:bg-ink/5 dark:border-white/15 dark:text-white dark:hover:bg-surface/40"
            aria-label={t("settings.close")}
          >
            <span aria-hidden>&times;</span>
          </button>
        </div>

        {/* Scrollable Content Area */}
        <div className="space-y-6 overflow-y-auto pr-2 custom-scrollbar">
          <section>
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-accent-secondary">{t("settings.appearance")}</p>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {(["system", "light", "dark"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setPreference(mode)}
                  className={cn(
                    "rounded-2xl border px-3 py-2 text-sm font-medium transition",
                    preference === mode
                      ? "border-ink bg-ink text-white dark:border-white dark:bg-surface dark:text-white"
                      : "border-ink/15 text-ink/80 hover:border-ink/40 dark:border-white/10 dark:text-white/70 dark:hover:text-white"
                  )}
                >
                  {t(`settings.theme.${mode}`)}
                </button>
              ))}
            </div>
          </section>

          <section>
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-accent-secondary">{t("settings.layout")}</p>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {(["horizontal", "vertical"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setValue("layout.orientation", mode)}
                  className={cn(
                    "rounded-2xl border px-3 py-2 text-xs font-medium transition",
                    styleConfig.layout.orientation === mode
                      ? "border-ink bg-ink text-white dark:border-white dark:bg-surface dark:text-white"
                      : "border-ink/15 text-ink/80 hover:border-ink/40 dark:border-white/10 dark:text-white/70 dark:hover:text-white"
                  )}
                >
                  {t(`settings.orientation.${mode}`)}
                </button>
              ))}
            </div>
          </section>

          <section>
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-accent-secondary">{t("settings.nodes")}</p>
            <div className="mt-4 space-y-4">
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/60">{t("settings.nodeRadius")}</span>
                <input
                  type="range"
                  min={2.5}
                  max={7.5}
                  step={0.1}
                  value={styleConfig.nodes.all.radius}
                  onChange={(event) => setValue("nodes.all.radius", Number(event.target.value))}
                  className="mt-2 w-full"
                />
                <span className="mt-1 block text-xs text-ink/60">{styleConfig.nodes.all.radius.toFixed(1)} px</span>
              </label>
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/60">{t("settings.nodeStrokeWidth")}</span>
                <input
                  type="range"
                  min={0.4}
                  max={1.4}
                  step={0.05}
                  value={styleConfig.nodes.all.strokeWidth}
                  onChange={(event) => setValue("nodes.all.strokeWidth", Number(event.target.value))}
                  className="mt-2 w-full"
                />
                <span className="mt-1 block text-xs text-ink/60">{styleConfig.nodes.all.strokeWidth.toFixed(2)} px</span>
              </label>
              <ColorPicker
                label={t("settings.driverColor")}
                value={styleConfig.nodes.driver.strokeColor}
                onChange={(color) =>
                  applyPatch({
                    nodes: {
                      driver: {
                        strokeColor: color,
                        textColor: color,
                        fillColor: color
                      }
                    }
                  })
                }
              />
              <ColorPicker
                label={t("settings.matchingColor")}
                value={styleConfig.nodes.matching.strokeColor}
                onChange={(color) =>
                  applyPatch({
                    nodes: {
                      matching: {
                        strokeColor: color,
                        textColor: color,
                        fillColor: color
                      }
                    }
                  })
                }
              />
              <ColorPicker
                label={t("settings.regularStrokeColor")}
                value={styleConfig.nodes.regular.strokeColor}
                onChange={(color) =>
                  applyPatch({
                    nodes: {
                      regular: {
                        strokeColor: color,
                        textColor: color
                      }
                    }
                  })
                }
              />
              <ColorPicker
                label={t("settings.regularFillColor")}
                value={styleConfig.nodes.regular.fillColor}
                onChange={(color) =>
                  applyPatch({
                    nodes: {
                      regular: {
                        fillColor: color
                      }
                    }
                  })
                }
              />
            </div>
          </section>

          <section>
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-accent-secondary">{t("settings.edges")}</p>
            <div className="mt-4 space-y-4">
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/60">{t("settings.edgeWidth")}</span>
                <input
                  type="range"
                  min={0.3}
                  max={1.5}
                  step={0.05}
                  value={styleConfig.edges.all.strokeWidth}
                  onChange={(event) => setValue("edges.all.strokeWidth", Number(event.target.value))}
                  className="mt-2 w-full"
                />
                <span className="mt-1 block text-xs text-ink/60">{styleConfig.edges.all.strokeWidth.toFixed(2)} px</span>
              </label>
              <ColorPicker
                label={t("settings.edgeMatchingColor")}
                value={styleConfig.edges.matching.color}
                onChange={(color) =>
                  applyPatch({
                    edges: {
                      matching: { color }
                    }
                  })
                }
              />
              <ColorPicker
                label={t("settings.edgeNonMatchingColor")}
                value={styleConfig.edges.nonMatching.color}
                onChange={(color) =>
                  applyPatch({
                    edges: {
                      nonMatching: { color }
                    }
                  })
                }
              />
              <ColorPicker
                label={t("settings.edgeAltMatchingColor")}
                value={styleConfig.edges.alternativeMatching.color}
                onChange={(color) =>
                  applyPatch({
                    edges: {
                      alternativeMatching: { color }
                    }
                  })
                }
              />
              <ColorPicker
                label={t("settings.edgeAltNonMatchingColor")}
                value={styleConfig.edges.alternativeNonMatching.color}
                onChange={(color) =>
                  applyPatch({
                    edges: {
                      alternativeNonMatching: { color }
                    }
                  })
                }
              />
              <ColorPicker
                label={t("settings.edgeHoverColor")}
                value={typeof styleConfig.edges.hover.color === "string" ? styleConfig.edges.hover.color : "#fde68a"}
                onChange={(color) =>
                  applyPatch({
                    edges: {
                      hover: { color }
                    }
                  })
                }
              />
            </div>
          </section>

          <section>
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-accent-secondary">{t("settings.hover")}</p>
            <div className="mt-4 space-y-4">
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/60">{t("settings.nodeHoverScale")}</span>
                <input
                  type="range"
                  min={1}
                  max={1.6}
                  step={0.05}
                  value={typeof styleConfig.nodes.hover.scale === "number" ? styleConfig.nodes.hover.scale : 1}
                  onChange={(event) => setValue("nodes.hover.scale", Number(event.target.value))}
                  className="mt-2 w-full"
                />
              </label>
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/60">{t("settings.edgeHoverScale")}</span>
                <input
                  type="range"
                  min={1}
                  max={2}
                  step={0.1}
                  value={typeof styleConfig.edges.hover.scale === "number" ? styleConfig.edges.hover.scale : 1}
                  onChange={(event) => setValue("edges.hover.scale", Number(event.target.value))}
                  className="mt-2 w-full"
                />
              </label>
            </div>
          </section>
        </div>

        {/* Footer - Fixed */}
        <div className="mt-6 flex items-center justify-between border-t border-ink/10 pt-4 shrink-0 dark:border-white/10">
          <button
            type="button"
            onClick={reset}
            className="rounded-full border border-ink/15 px-4 py-2 text-sm font-medium text-ink transition hover:bg-ink/5 dark:border-white/15 dark:text-white"
          >
            {t("settings.reset")}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full bg-ink px-5 py-2 text-sm font-semibold text-white transition hover:bg-ink/90 dark:bg-surface dark:text-white"
          >
            {t("settings.close")}
          </button>
        </div>
      </div>
    </div>
  );
}

function ColorPicker({ label, value, onChange }: { label: string; value: string; onChange: (color: string) => void }) {
  return (
    <label className="flex items-center justify-between gap-3 rounded-2xl border border-ink/10 px-3 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink/70 dark:border-white/15 dark:text-white/80">
      <span>{label}</span>
      <input
        type="color"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-8 w-16 cursor-pointer rounded-xl border-none bg-transparent p-0"
      />
    </label>
  );
}
