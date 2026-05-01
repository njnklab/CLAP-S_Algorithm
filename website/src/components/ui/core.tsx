"use client";

import { motion } from "framer-motion";
import { BlockMath, InlineMath } from "react-katex";
import React, { useState, useEffect } from "react";

import { cn } from "@/lib/utils";
import { useI18n } from "@/providers/i18n-provider";

export function SafeInlineMath({ math, as = "span" }: { math: string; as?: any }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const Component = as;
  if (!mounted) return <Component suppressHydrationWarning className="opacity-0">{`$${math}$`}</Component>;
  return <InlineMath math={math} />;
}

export function SafeBlockMath({ math }: { math: string }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div suppressHydrationWarning className="py-4 text-center opacity-0">{`\\[${math}\\]`}</div>;
  return <BlockMath math={math} />;
}


type FormatMathTextProps<T extends keyof React.JSX.IntrinsicElements = "span"> = {
  text?: string;
  children?: React.ReactNode;
  as?: T;
} & Omit<React.JSX.IntrinsicElements[T], "children">;

export function FormatMathText<T extends keyof React.JSX.IntrinsicElements = "span">({
  text,
  children,
  as,
  ...rest
}: FormatMathTextProps<T>) {
  const content = text || (typeof children === "string" ? children : "");
  if (!content) return children || null;

  // Supports multi-line math blocks and treats \ as literal if passed via JSX children
  const parts = content.split(/(\$[\s\S]*?\$)/g);
  const Component = (as || "span") as any;
  const InlineWrapper =
    as === "tspan" || as === "text" ? "tspan" : "span";

  return (
    <Component suppressHydrationWarning {...(rest as any)}>
      {parts.map((part, index) => {
        if (!part) return null;
        if (part.startsWith("$") && part.endsWith("$")) {
          return <SafeInlineMath key={index} math={part.slice(1, -1)} as={InlineWrapper} />;
        }
        const Wrapper = InlineWrapper as any;
        return <Wrapper key={index}>{part}</Wrapper>;
      })}
    </Component>
  );
}



export function Card({
  className,
  children,
  ...props
}: {
  className?: string;
  children: React.ReactNode;
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...props}
      className={cn(
        "rounded-[28px] border border-white/80 bg-surface/88 p-6 shadow-card backdrop-blur-sm dark:border-white/5 dark:bg-slate-900/80",
        className
      )}
    >
      {children}
    </div>
  );
}

export function StatCard({
  label,
  value,
  tone = "default",
  color,
  hint
}: {
  label: string;
  value: React.ReactNode;
  tone?: "default" | "layer1" | "layer2";
  color?: string;
  hint?: string;
}) {
  const toneStyle =
    tone === "layer1"
      ? { background: "linear-gradient(to bottom right, rgba(var(--color-layer1), 0.15), rgba(var(--color-layer1), 0.05))" }
      : tone === "layer2"
        ? { background: "linear-gradient(to bottom right, rgba(var(--color-layer2), 0.15), rgba(var(--color-layer2), 0.05))" }
        : { background: "linear-gradient(to bottom right, rgba(var(--color-ink), 0.1), rgba(var(--color-surface), 0.95))" };

  const customStyle = color
    ? {
      background: `linear-gradient(to bottom right, ${color}1a, rgba(var(--color-surface), 0.95))`
    }
    : toneStyle;

  return (
    <div
      className={cn(
        "rounded-3xl border border-ink/8 p-4 dark:border-white/10 dark:bg-slate-900/60"
      )}
      style={customStyle}
    >
      <div className="text-xs uppercase tracking-[0.2em] text-ink/55">
        <FormatMathText text={label} />
      </div>
      <div className="mt-2 text-2xl font-semibold text-ink">
        {typeof value === "string" ? <FormatMathText text={value} /> : value}
      </div>
      {hint ? (
        <div className="mt-1 text-sm text-ink/65">
          <FormatMathText text={hint} />
        </div>
      ) : null}
    </div>
  );
}

export function EquationCard({
  title,
  formula,
  body
}: {
  title: string;
  formula: string;
  body: string;
}) {
  return (
    <Card className="h-full">
      <div className="text-sm font-semibold uppercase tracking-[0.18em] text-ink/55">
        <FormatMathText text={title} />
      </div>
      <div className="mt-5 overflow-x-auto rounded-2xl bg-mist/80 px-3 py-2 text-ink">
        <SafeBlockMath math={formula} />
      </div>
      <p className="mt-4 text-sm leading-7 text-ink/75">
        <FormatMathText text={body} />
      </p>
    </Card>
  );
}

export function FormulaPill({ math }: { math: string }) {
  return (
    <span className="inline-flex items-center rounded-full border border-ink/10 bg-surface px-4 py-2 text-sm text-ink shadow-sm">
      <SafeInlineMath math={math} />
    </span>
  );
}

export function KeyNode({
  color,
  label,
  fillOpacity = 0.15
}: {
  color: string;
  label: string;
  fillOpacity?: number;
}) {
  return (
    <div className="inline-flex items-center gap-2 text-sm text-ink/70">
      <svg width="12" height="12" viewBox="0 0 12 12" className="flex-shrink-0">
        <circle
          cx="6"
          cy="6"
          r="4.5"
          fill={`${color}${Math.round(fillOpacity * 255).toString(16).padStart(2, '0')}`}
          stroke={color}
          strokeWidth="1.2"
        />
      </svg>
      <span><FormatMathText text={label} /></span>
    </div>
  );
}

export function KeyEdge({
  color,
  label,
  strokeWidth = 1.4
}: {
  color: string;
  label: string;
  strokeWidth?: number;
}) {
  return (
    <div className="inline-flex items-center gap-2 text-sm text-ink/70">
      <svg width="24" height="12" viewBox="0 0 24 12" className="flex-shrink-0 overflow-visible">
        <defs>
          <marker
            id={`legend-arrow-${color.replace('#', '')}`}
            markerWidth="6"
            markerHeight="4.5"
            refX="0.1"
            refY="2.25"
            orient="auto"
          >
            <path d="M0,0 L0,4.5 L6,2.25 Z" fill={color} />
          </marker>
        </defs>
        <line
          x1="2"
          y1="6"
          x2="18"
          y2="6"
          stroke={color}
          strokeWidth={strokeWidth}
          markerEnd={`url(#legend-arrow-${color.replace('#', '')})`}
        />
      </svg>
      <span><FormatMathText text={label} /></span>
    </div>
  );
}

export function Stepper({
  current,
  total,
  onNext,
  onPrevious,
  onReset
}: {
  current: number;
  total: number;
  onNext: () => void;
  onPrevious: () => void;
  onReset: () => void;
}) {
  const { t } = useI18n();

  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        type="button"
        onClick={onPrevious}
        className="rounded-full border border-ink/10 px-4 py-2 text-sm text-ink transition hover:border-ink/20 hover:bg-ink/5"
      >
        {t("ui.stepper.previous")}
      </button>
      <button
        type="button"
        onClick={onNext}
        className="rounded-full bg-ink px-4 py-2 text-sm text-white transition hover:bg-ink/90 dark:text-slate-900"
      >
        {t("ui.stepper.next")}
      </button>
      <button
        type="button"
        onClick={onReset}
        className="rounded-full border border-ink/10 px-4 py-2 text-sm text-ink transition hover:border-ink/20 hover:bg-ink/5"
      >
        {t("ui.stepper.reset")}
      </button>
      <div className="ml-auto text-sm text-ink/60">
        {t("ui.stepper.progress", { current: current + 1, total })}
      </div>
    </div>
  );
}

export function SurfaceTitle({
  title,
  body
}: {
  title: string;
  body?: string;
}) {
  return (
    <div>
      <div className="text-lg font-semibold text-ink">
        <FormatMathText text={title} />
      </div>
      {body ? (
        <p className="mt-2 text-sm leading-7 text-ink/70">
          <FormatMathText text={body} />
        </p>
      ) : null}
    </div>
  );
}

export function FadeIn({
  children,
  delay = 0
}: {
  children: React.ReactNode;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.35, ease: "easeOut", delay }}
    >
      {children}
    </motion.div>
  );
}
