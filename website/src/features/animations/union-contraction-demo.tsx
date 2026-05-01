"use client";

import { useMemo, useState } from "react";

import { unionContractionStates } from "@/features/animations/data/demoData";
import { cn } from "@/lib/utils";

import { APP_CONFIG } from "@/lib/config";
import { Card, FormatMathText, KeyNode, StatCard, SurfaceTitle } from "@/components/ui/core";
import { useI18n } from "@/providers/i18n-provider";

const universe = [1, 2, 3, 4, 5];

function difference(d1: number[], d2: number[]) {
  return d1.filter((node) => !d2.includes(node)).length + d2.filter((node) => !d1.includes(node)).length;
}

export function UnionContractionDemo() {
  const [index, setIndex] = useState(0);
  const state = unionContractionStates[index];
  const { t } = useI18n();

  const partition = useMemo(() => {
    return universe.map((node) => {
      const inL1 = state.d1.includes(node);
      const inL2 = state.d2.includes(node);
      if (inL1 && inL2) return { node, label: "$CDS$", color: APP_CONFIG.colors.light.cds };
      if (inL1) return { node, label: "$DD_1$", color: APP_CONFIG.colors.light.layer1 };
      if (inL2) return { node, label: "$DD_2$", color: APP_CONFIG.colors.light.layer2 };
      return { node, label: "$CMS$", color: APP_CONFIG.colors.light.cms };
    });
  }, [state]);

  const setGroups = useMemo(() => {
    const groups: Record<string, number[]> = {
      "$DD_1$": [],
      "$DD_2$": [],
      "$CDS$": [],
      "$CMS$": []
    };
    partition.forEach((p) => {
      groups[p.label].push(p.node);
    });
    return groups;
  }, [partition]);

  const labelColors: Record<string, string> = {
    "$DD_1$": APP_CONFIG.colors.light.layer1,
    "$DD_2$": APP_CONFIG.colors.light.layer2,
    "$CDS$": APP_CONFIG.colors.light.cds,
    "$CMS$": APP_CONFIG.colors.light.cms
  };

  const renderSet = (label: string, nodes: number[]) => {
    const color = labelColors[label];
    const hasMembers = nodes.length > 0;
    const formattedSet = hasMembers ? String.raw`$\{${nodes.sort().join(", ")}\}$` : "";
    return (
      <div
        className="group relative overflow-hidden rounded-2xl border border-ink/5 p-3 transition-all hover:border-ink/10"
        style={{ backgroundColor: `${color}${Math.round(APP_CONFIG.opacity.duplex_node_bg * 255).toString(16).padStart(2, '0')}` }}
      >
        <div className="flex items-center justify-between mb-2.5">
          <div
            className="flex h-6 items-center rounded-lg px-2 text-[10px] font-bold text-white shadow-sm"
            style={{ backgroundColor: color }}
          >
            <FormatMathText text={label} />
          </div>
          <div className="text-[10px] font-semibold text-ink/30 uppercase tracking-wider">
            {t("union.membership.items", { count: nodes.length })}
          </div>
        </div>
        <div className="min-h-[1.5rem] break-all font-mono text-xs leading-relaxed" style={{ color: `${color}ee` }}>
          {hasMembers ? <FormatMathText text={formattedSet} /> : <span>{t("union.sets.empty")}</span>}
        </div>
      </div>
    );
  };

  const unionSize = new Set([...state.d1, ...state.d2]).size;
  const delta = difference(state.d1, state.d2);

  return (
    <Card>
      <div className="grid gap-8 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="space-y-5">
          <SurfaceTitle
            title={t("union.animation.title")}
            body={t("union.animation.body")}
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <StatCard label="$|D_1|$" value={state.d1.length} color={APP_CONFIG.colors.light.layer1} hint={t("union.stats.d1Hint")} />
            <StatCard label="$|D_2|$" value={state.d2.length} color={APP_CONFIG.colors.light.layer2} hint={t("union.stats.d2Hint")} />
            <StatCard label="$|U|$" value={unionSize} color={APP_CONFIG.colors.light.non_matching_edge} hint={t("union.stats.unionHint")} />
            <StatCard label="$\Delta$" value={delta} color={APP_CONFIG.colors.light.non_matching_edge} hint={t("union.stats.deltaHint")} />
          </div>
          <div className="text-sm leading-7 text-ink/72">
            <FormatMathText text={t(state.explanationKey)} />
          </div>

        </div>
        <div className="rounded-[26px] border border-ink/8 bg-gradient-to-br from-surface to-mist/80 p-5">
          <div className="mb-5 text-xs font-semibold uppercase tracking-[0.2em] text-ink/55">
            {t("union.membership.title")}
          </div>
          <div className="space-y-4">
            <div>
              <div className="mb-2 text-sm font-medium" style={{ color: APP_CONFIG.colors.light.layer1 }}>{t("union.membership.layer1")}</div>
              <div className="flex flex-wrap gap-3">
                {universe.map((node) => (
                  <div
                    key={`l1-${node}`}
                    className={`flex h-12 w-12 items-center justify-center rounded-2xl border text-sm font-semibold transition-colors`}
                    style={state.d1.includes(node)
                      ? { borderColor: `${APP_CONFIG.colors.light.layer1}33`, backgroundColor: `${APP_CONFIG.colors.light.layer1}1a`, color: APP_CONFIG.colors.light.layer1 }
                      : {
                        borderColor: "rgba(var(--color-ink), 0.1)",
                        backgroundColor: "rgba(var(--color-surface), 1)",
                        color: "rgba(var(--color-ink), 0.45)"
                      }}
                  >
                    {node}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2 text-sm font-medium" style={{ color: APP_CONFIG.colors.light.layer2 }}>{t("union.membership.layer2")}</div>
              <div className="flex flex-wrap gap-3">
                {universe.map((node) => (
                  <div
                    key={`l2-${node}`}
                    className={`flex h-12 w-12 items-center justify-center rounded-2xl border text-sm font-semibold transition-colors`}
                    style={state.d2.includes(node)
                      ? { borderColor: `${APP_CONFIG.colors.light.layer2}33`, backgroundColor: `${APP_CONFIG.colors.light.layer2}1a`, color: APP_CONFIG.colors.light.layer2 }
                      : {
                        borderColor: "rgba(var(--color-ink), 0.1)",
                        backgroundColor: "rgba(var(--color-surface), 1)",
                        color: "rgba(var(--color-ink), 0.45)"
                      }}
                  >
                    {node}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="mt-6 space-y-4 rounded-[28px] border border-ink/8 bg-surface/90 p-5 shadow-sm">
            <div className="flex items-center justify-between px-1">
              <div className="text-sm font-bold text-ink">{t("union.partition.title")}</div>
              <div className="text-[10px] font-medium text-ink/40 uppercase tracking-widest">{t("union.partition.status")}</div>
            </div>

            <div className="grid grid-cols-[1fr_auto_1fr] gap-4">
              <div className="space-y-3">
                <div className="px-1 text-[10px] font-bold text-ink/50 uppercase tracking-widest flex items-center gap-2">
                  <span className="w-1 h-1 rounded-full bg-ink/30" />
                  {t("union.sets.difference")}
                </div>
                {renderSet("$DD_1$", setGroups["$DD_1$"])}
                {renderSet("$DD_2$", setGroups["$DD_2$"])}
              </div>

              <div className="w-px bg-ink/5 self-stretch" />

              <div className="space-y-3">
                <div className="px-1 text-[10px] font-bold text-ink/50 uppercase tracking-widest flex items-center gap-2">
                  <span className="w-1 h-1 rounded-full bg-ink/30" />
                  {t("union.sets.consistent")}
                </div>
                {renderSet("$CDS$", setGroups["$CDS$"])}
                {renderSet("$CMS$", setGroups["$CMS$"])}
              </div>
            </div>
          </div>
          <div className="mt-5 overflow-x-auto no-scrollbar">
            <div className="text-[10px] font-bold text-ink/30 uppercase tracking-[0.2em] mb-4 px-1">
              {t("union.transitions.title")}
            </div>
            <div className="flex mx-2 my-2 gap-3">
              {unionContractionStates.map((item, itemIndex) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setIndex(itemIndex)}
                  className={cn(
                    "px-4 py-2 text-xs font-semibold rounded-2xl transition-all whitespace-nowrap outline-none",
                    index === itemIndex
                      ? "bg-ink text-white shadow-sm ring-1 ring-ink dark:text-slate-900"
                      : "bg-surface text-ink/60 border border-ink/10 hover:border-ink/20 hover:text-ink"
                  )}
                >
                  {t(item.labelKey)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
