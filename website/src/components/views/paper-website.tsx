"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { motion } from "framer-motion";

import { SiteNavbar } from "@/components/layout/site-navbar";
import { ClapTraceDemo } from "@/features/animations/clap-trace-demo";
import { ResultsExplorer } from "@/features/results-viewer/results-explorer";
import { SectionShell } from "@/components/ui/section-shell";
import { StructuralPrimer } from "@/features/animations/structural-primer";
import ExchangeTheorem from "@/features/animations/exchange-theorem";
import { UnionContractionDemo } from "@/features/animations/union-contraction-demo";
import { SettingsPanel } from "@/features/settings/settings-panel";
import { useSiteContent } from "@/hooks/use-site-content";
import type { RealSummary, SyntheticSummary } from "@/lib/results";
import { formatNumber } from "@/lib/utils";
import { useI18n } from "@/providers/i18n-provider";
import type { SectionCopy } from "@/data/content/site-content";

import { Card, EquationCard, FadeIn, FormulaPill, StatCard, FormatMathText, SafeInlineMath } from "@/components/ui/core";

type PaperWebsiteProps = {
  synthetic: SyntheticSummary[];
  real: RealSummary[];
  highlights: {
    lowOverlapMeanGain: number;
    avgRealPathLength: number;
  };
};

function byId(sectionCopy: SectionCopy[], id: string) {
  return sectionCopy.find((item) => item.id === id)!;
}

export function PaperWebsite({ synthetic, real, highlights }: PaperWebsiteProps) {
  const { t } = useI18n();
  const {
    heroOptions,
    navItems,
    faqItems,
    glossaryItems,
    sectionCopy,
    applications,
    bibtex,
    paperAbstract,
    theoryModules,
    whyNew
  } = useSiteContent();
  const [heroIndex, setHeroIndex] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const hero = heroOptions[heroIndex];
  const why = byId(sectionCopy, "why");
  const primer = byId(sectionCopy, "primer");
  const problem = byId(sectionCopy, "problem");
  const algorithm = byId(sectionCopy, "algorithm");
  const theory = byId(sectionCopy, "theory");
  const results = byId(sectionCopy, "results");
  const applicationsCopy = byId(sectionCopy, "applications");
  const limits = byId(sectionCopy, "limits");

  const summaryStats = useMemo(() => {
    const avgRelativeGain =
      synthetic.reduce((sum, item) => sum + item.relativeGain, 0) / Math.max(synthetic.length, 1);
    return {
      avgRelativeGain,
      avgPath: highlights.avgRealPathLength,
      lowOverlap: highlights.lowOverlapMeanGain
    };
  }, [highlights, synthetic]);

  return (
    <div className="relative min-h-screen bg-inherit text-ink">
      <div className="absolute inset-x-0 top-0 -z-10 h-[28rem] bg-[radial-gradient(circle_at_top,rgba(11,114,133,0.2),transparent_42%)]" />
      <SiteNavbar navItems={navItems} onOpenSettings={() => setSettingsOpen(true)} />
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      <main>
        <section id="hero" className="px-4 pb-12 pt-10 sm:px-6 lg:px-10 lg:pt-16">
          <div className="mx-auto grid max-w-7xl gap-8 lg:grid-cols-[1.1fr_0.9fr]">
            <div>
              <motion.div
                key={hero.id}
                initial={{ opacity: 0, y: 22 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, ease: "easeOut" }}
                className="max-w-3xl"
              >
                <div className="inline-flex items-center gap-3 rounded-full border border-white/85 bg-surface/85 px-4 py-2 text-sm text-ink/72 shadow-sm">
                  <span className="font-semibold text-accent">{hero.tone}</span>
                  <span>{t("hero.badge")}</span>
                </div>
                <h1 className="mt-6 max-w-4xl text-balance font-serif text-5xl leading-[1.02] text-ink sm:text-6xl">
                  {hero.title}
                </h1>
                <p className="mt-6 max-w-3xl text-pretty text-xl leading-9 text-ink/78">
                  <FormatMathText text={hero.subtitle} />
                </p>
                <p className="mt-5 max-w-2xl text-pretty text-base leading-8 text-ink/72">
                  <FormatMathText text={hero.value} />
                </p>
                <div className="mt-8 flex flex-wrap gap-4">
                  <Link
                    href="/paper/revision.pdf"
                    className="rounded-full bg-ink px-5 py-3 text-sm font-medium text-white transition hover:bg-ink/92 dark:text-slate-900"
                  >
                    {hero.primaryCta}
                  </Link>
                  <Link
                    href="#algorithm"
                    className="rounded-full border border-ink/12 bg-surface px-5 py-3 text-sm font-medium text-ink transition hover:bg-ink/5"
                  >
                    {hero.secondaryCta}
                  </Link>
                </div>
                <div className="mt-10 flex flex-wrap gap-3">
                  {heroOptions.map((option, index) => (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => setHeroIndex(index)}
                      className={`rounded-full px-4 py-2 text-sm transition ${heroIndex === index
                        ? "bg-ink text-white dark:text-slate-900"
                        : "border border-ink/10 bg-surface text-ink hover:bg-ink/5"
                        }`}
                    >
                      {option.tone}
                    </button>
                  ))}
                </div>
              </motion.div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
              <Card className="overflow-hidden bg-[radial-gradient(circle_at_top_right,rgba(11,114,133,0.14),rgba(var(--color-surface),0.92))]">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
                  {t("stats.banner.title")}
                </div>
                <p className="mt-4 text-lg leading-8 text-ink/82">
                  <FormatMathText text={t("stats.banner.body")} />
                </p>
              </Card>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-2">
                <StatCard
                  label={t("stats.card.matching.label")}
                  value={t("stats.card.matching.value")}
                  hint={t("stats.card.matching.hint")}
                />
                <StatCard
                  label={t("stats.card.avgLength.label")}
                  value={formatNumber(summaryStats.avgPath, 2)}
                  hint={t("stats.card.avgLength.hint")}
                />
                <StatCard
                  label={t("stats.card.relativeGain.label")}
                  value={`${formatNumber(summaryStats.avgRelativeGain, 2)}%`}
                  hint={t("stats.card.relativeGain.hint")}
                />
                <StatCard
                  label={t("stats.card.lowOverlap.label")}
                  value={formatNumber(summaryStats.lowOverlap, 0)}
                  hint={t("stats.card.lowOverlap.hint")}
                />
              </div>
            </div>
          </div>
        </section>

        <SectionShell
          id="faq"
          eyebrow={t("faq.eyebrow")}
          title={t("faq.title")}
          lead={t("faq.lead")}
        >
          <div className="grid gap-6 xl:grid-cols-[0.98fr_1.02fr]">
            <div className="space-y-4">
              {faqItems.map((item) => (
                <details
                  key={item.question}
                  open
                  className="group rounded-[26px] border border-ink/8 bg-surface/90 p-5"
                >
                  <summary className="flex cursor-pointer list-none items-baseline justify-between text-base font-semibold text-ink [&::-webkit-details-marker]:hidden">
                    <span><FormatMathText text={item.question} /></span>
                    <span className="ml-4 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ink/5 text-ink/40 transition-transform duration-300 group-open:rotate-180 group-open:bg-ink group-open:text-white group-open:dark:text-slate-900">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
                    </span>
                  </summary>
                  <p className="mt-4 text-sm leading-7 text-ink/72"><FormatMathText text={item.answer} /></p>
                </details>
              ))}
            </div>
            <div className="space-y-4">
              {glossaryItems.map((item, index) => (
                <details
                  key={item.term}
                  open={index === 0}
                  className="group rounded-[22px] border border-ink/8 bg-surface/90 p-5 px-6"
                >
                  <summary className="flex cursor-pointer list-none items-baseline justify-between text-base font-semibold text-ink [&::-webkit-details-marker]:hidden">
                    <span><FormatMathText text={item.term} /></span>
                    <span className="ml-4 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-ink/5 text-ink/40 transition-transform duration-300 group-open:rotate-180 group-open:bg-ink group-open:text-white group-open:dark:text-slate-900">
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
                    </span>
                  </summary>
                  <p className="mt-4 text-sm leading-7 text-ink/72"><FormatMathText text={item.meaning} /></p>
                </details>
              ))}
            </div>
          </div>
        </SectionShell>

        <SectionShell
          id="why"
          eyebrow={why.eyebrow}
          title={why.title}
          lead={why.lead}
        >
          <div className="grid gap-8 lg:grid-cols-[0.8fr_1.2fr]">
            <div className="space-y-4 text-base leading-8 text-ink/75">
              {why.paragraphs.map((paragraph) => (
                <p key={paragraph}><FormatMathText text={paragraph} /></p>
              ))}
              <ul className="space-y-3 rounded-[26px] border border-ink/8 bg-surface/90 p-5 text-sm leading-7 text-ink/72">
                {why.bullets?.map((bullet) => <li key={bullet}>• <FormatMathText text={bullet} /></li>)}
              </ul>
            </div>
            <UnionContractionDemo />
          </div>
        </SectionShell>

        <SectionShell
          id="primer"
          eyebrow={primer.eyebrow}
          title={primer.title}
          lead={primer.lead}
        >
          <div className="space-y-8">
            <div className="grid gap-6 lg:grid-cols-2">
              <Card className="space-y-4 text-base leading-8 text-ink/75">
                {primer.paragraphs.slice(0, 2).map((paragraph) => (
                  <p key={paragraph}><FormatMathText text={paragraph} /></p>
                ))}
              </Card>
              <Card className="space-y-4 text-base leading-8 text-ink/75">
                {primer.paragraphs.slice(2).map((paragraph) => (
                  <p key={paragraph}><FormatMathText text={paragraph} /></p>
                ))}
              </Card>
            </div>
            <StructuralPrimer />
            <ExchangeTheorem />
          </div>
        </SectionShell>

        <SectionShell
          id="problem"
          eyebrow={problem.eyebrow}
          title={problem.title}
          lead={problem.lead}
        >
          <div className="grid gap-8 xl:grid-cols-[1.05fr_0.95fr]">
            <div className="space-y-4 text-base leading-8 text-ink/75">
              {problem.paragraphs.map((paragraph) => (
                <p key={paragraph}><FormatMathText text={paragraph} /></p>
              ))}
              <ul className="space-y-3 rounded-[26px] border border-ink/8 bg-surface/90 p-5 text-sm leading-7 text-ink/72">
                {problem.bullets?.map((bullet) => <li key={bullet}>• <FormatMathText text={bullet} /></li>)}
              </ul>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <EquationCard
                title={t("problem.cards.fixedBudgets.title")}
                formula={String.raw`U = D_1 \cup D_2`}
                body={t("problem.cards.fixedBudgets.body")}
              />
              <EquationCard
                title={t("problem.cards.differenceMass.title")}
                formula={String.raw`\Delta = |DD_1| + |DD_2|`}
                body={t("problem.cards.differenceMass.body")}
              />
              <EquationCard
                title={t("problem.cards.unionIdentity.title")}
                formula={String.raw`|U| = \frac{k_1 + k_2 + \Delta}{2}`}
                body={t("problem.cards.unionIdentity.body")}
              />
              <EquationCard
                title={t("problem.cards.improvement.title")}
                formula={String.raw`\Delta \downarrow 2 \Rightarrow |U| \downarrow 1`}
                body={t("problem.cards.improvement.body")}
              />
            </div>
          </div>

          <FadeIn delay={0.08}>
            <div className="mt-8 grid gap-8 lg:grid-cols-[0.92fr_1.08fr]">
              <Card className="bg-[radial-gradient(circle_at_top_right,rgba(199,119,26,0.14),rgba(var(--color-surface),0.92))]">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
                  {t("problem.new.title")}
                </div>
                <div className="mt-4 space-y-4 text-sm leading-7 text-ink/74">
                  {whyNew.map((item) => (
                    <p key={item}>• <FormatMathText text={item} /></p>
                  ))}
                </div>
              </Card>
              <Card>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
                  {t("problem.proof.title")}
                </div>
                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <div className="rounded-[22px] border border-ink/8 bg-mist/70 p-5">
                    <div className="text-lg font-semibold text-ink">
                      <FormatMathText text={t("problem.proof.card1.title")} />
                    </div>
                    <p className="mt-3 text-sm leading-7 text-ink/72">
                      <FormatMathText text={t("problem.proof.card1.body")} />
                    </p>
                  </div>
                  <div className="rounded-[22px] border border-ink/8 bg-mist/70 p-5">
                    <div className="text-lg font-semibold text-ink">{t("problem.proof.card2.title")}</div>
                    <p className="mt-3 text-sm leading-7 text-ink/72">
                      <FormatMathText text={t("problem.proof.card2.body")} />
                    </p>
                  </div>
                </div>
              </Card>
            </div>
          </FadeIn>
        </SectionShell>

        <SectionShell
          id="algorithm"
          eyebrow={algorithm.eyebrow}
          title={algorithm.title}
          lead={algorithm.lead}
        >
          <div className="grid gap-6 lg:grid-cols-2">
            {algorithm.paragraphs.map((paragraph) => (
              <Card key={paragraph} className="text-base leading-8 text-ink/75">
                <FormatMathText text={paragraph} />
              </Card>
            ))}
          </div>
          <div className="mt-8 space-y-8">
            <ClapTraceDemo />
          </div>
        </SectionShell>

        <SectionShell
          id="theory"
          eyebrow={theory.eyebrow}
          title={theory.title}
          lead={theory.lead}
        >
          <div className="space-y-6">
            <div className="grid gap-6 xl:grid-cols-2">
              {theory.paragraphs.map((paragraph) => (
                <Card key={paragraph} className="text-base leading-8 text-ink/75">
                  <FormatMathText text={paragraph} />
                </Card>
              ))}
            </div>
            <div className="grid gap-6 xl:grid-cols-2">
              {theoryModules.map((module, index) => (
                <FadeIn key={module.id} delay={0.04 * index}>
                  <Card className="h-full">
                    <div className="mb-3 text-xs font-semibold uppercase tracking-[0.28em] text-accent">
                      <FormatMathText text={module.title} />
                    </div>
                    <div className="mt-4 grid gap-5">
                      <div>
                        <div className="text-sm font-semibold text-ink">{t("theory.module.intuition")}</div>
                        <div className="mt-2 space-y-2 text-sm leading-7 text-ink/72">
                          {module.intuition.map((item) => (
                            <p key={item}>• <FormatMathText text={item} /></p>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-ink">{t("theory.module.formal")}</div>
                        <div className="mt-2 space-y-2 text-sm leading-7 text-ink/72">
                          {module.formal.map((item) => (
                            <p key={item}>• <FormatMathText text={item} /></p>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-ink">{t("theory.module.why")}</div>
                        <p className="mt-2 text-sm leading-7 text-ink/72"><FormatMathText text={module.why} /></p>
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-ink">{t("theory.module.visual")}</div>
                        <p className="mt-2 text-sm leading-7 text-ink/72"><FormatMathText text={module.visualAid} /></p>
                      </div>
                    </div>
                  </Card>
                </FadeIn>
              ))}
            </div>
          </div>
        </SectionShell>

        <SectionShell
          id="results"
          eyebrow={results.eyebrow}
          title={results.title}
          lead={results.lead}
        >
          <div className="grid gap-6 lg:grid-cols-2">
            {results.paragraphs.map((paragraph) => (
              <Card key={paragraph} className="text-base leading-8 text-ink/75">
                <FormatMathText text={paragraph} />
              </Card>
            ))}
          </div>
          <div className="mt-8">
            <ResultsExplorer synthetic={synthetic} real={real} />
          </div>
        </SectionShell>

        <SectionShell
          id="applications"
          eyebrow={applicationsCopy.eyebrow}
          title={applicationsCopy.title}
          lead={applicationsCopy.lead}
        >
          <div className="space-y-8">
            <div className="grid gap-8 lg:grid-cols-[1fr_0.8fr]">
              <div className="space-y-4 text-base leading-8 text-ink/75">
                {applicationsCopy.paragraphs.map((paragraph) => (
                  <p key={paragraph}><FormatMathText text={paragraph} /></p>
                ))}
              </div>
              <Card className="bg-[radial-gradient(circle_at_top_right,rgba(11,114,133,0.14),rgba(var(--color-surface),0.92))]">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
                  {t("applications.distinguish.title")}
                </div>
                <div className="mt-4 space-y-5 text-sm leading-7 text-ink/73">
                  <p>
                    <span className="font-semibold text-ink">{t("applications.distinguish.paper.title")}</span>{" "}
                    {t("applications.distinguish.paper.body")}
                  </p>
                  <p>
                    <span className="font-semibold text-ink">{t("applications.distinguish.experiments.title")}</span>{" "}
                    {t("applications.distinguish.experiments.body")}
                  </p>
                  <p>
                    <span className="font-semibold text-ink">{t("applications.distinguish.future.title")}</span>{" "}
                    {t("applications.distinguish.future.body")}
                  </p>
                </div>
              </Card>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              {applications.map((item) => (
                <Card key={item.title} className="bg-surface/90 border-ink/5 p-5 transition-all hover:bg-mist/40 cursor-default">
                  <div className="text-base font-semibold text-ink leading-tight"><FormatMathText text={item.title} /></div>
                  <p className="mt-2.5 text-sm leading-6 text-ink/70 text-pretty"><FormatMathText text={item.body} /></p>
                </Card>
              ))}
            </div>
          </div>
        </SectionShell>

        <SectionShell
          id="limits"
          eyebrow={limits.eyebrow}
          title={limits.title}
          lead={limits.lead}
        >
          <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="space-y-4 text-base leading-8 text-ink/75">
              {limits.paragraphs.map((paragraph) => (
                <p key={paragraph}><FormatMathText text={paragraph} /></p>
              ))}
              <ul className="space-y-3 rounded-[26px] border border-ink/8 bg-surface/90 p-5 text-sm leading-7 text-ink/72">
                {limits.bullets?.map((bullet) => <li key={bullet}>• <FormatMathText text={bullet} /></li>)}
              </ul>
            </div>
            <Card className="bg-[radial-gradient(circle_at_top_right,rgba(199,119,26,0.12),rgba(var(--color-surface),0.92))]">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
                {t("limits.scope.title")}
              </div>
              <div className="mt-4 space-y-4 text-sm leading-7 text-ink/73">
                <p><FormatMathText text={t("limits.scope.body1")} /></p>
                <p><FormatMathText text={t("limits.scope.body2")} /></p>
              </div>
            </Card>
          </div>
        </SectionShell>

        <SectionShell
          id="resources"
          eyebrow={t("resources.eyebrow")}
          title={t("resources.title")}
          lead={t("resources.lead")}
        >
          <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
            <Card className="space-y-5">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
                {t("resources.abstract")}
              </div>
              <p className="text-sm leading-8 text-ink/74"><FormatMathText text={paperAbstract} /></p>
              <div className="grid gap-4 md:grid-cols-2">
                <Link
                  href="/paper/revision.pdf"
                  className="rounded-[22px] border border-ink/10 bg-mist/55 p-5 transition hover:bg-mist"
                >
                  <div className="text-sm font-semibold text-ink">{t("resources.links.paper.label")}</div>
                  <div className="mt-2 text-sm text-ink/65">{t("resources.links.paper.caption")}</div>
                </Link>
                <Link
                  href="https://github.com/njnklab/CLAPs-algorithm"
                  className="rounded-[22px] border border-ink/10 bg-mist/55 p-5 transition hover:bg-mist"
                  target="_blank"
                  rel="noreferrer"
                >
                  <div className="text-sm font-semibold text-ink">{t("resources.links.code.label")}</div>
                  <div className="mt-2 text-sm text-ink/65">{t("resources.links.code.caption")}</div>
                </Link>
                <Link
                  href="/paper/supplementary-materials.pdf"
                  className="rounded-[22px] border border-ink/10 bg-mist/55 p-5 transition hover:bg-mist"
                >
                  <div className="text-sm font-semibold text-ink">{t("resources.links.supplement.label")}</div>
                  <div className="mt-2 text-sm text-ink/65">{t("resources.links.supplement.caption")}</div>
                </Link>
                <Link
                  href="mailto:zhangxizhe@njmu.edu.cn"
                  className="rounded-[22px] border border-ink/10 bg-mist/55 p-5 transition hover:bg-mist"
                >
                  <div className="text-sm font-semibold text-ink">{t("resources.links.contact.label")}</div>
                  <div className="mt-2 text-sm text-ink/65">{t("resources.links.contact.caption")}</div>
                </Link>
              </div>
            </Card>

            <div className="space-y-6">
              <Card className="border-[#0f172a] bg-[#0f172a] text-white">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-white/70">
                  {t("resources.bibtex")}
                </div>
                <pre className="mt-4 overflow-x-auto rounded-[22px] bg-[#050b16] px-5 py-4 text-xs leading-6 text-[#f8fafc]">
                  {bibtex}
                </pre>
              </Card>
              <Card>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/55">
                  {t("resources.authors.title")}
                </div>
                <div className="mt-4 space-y-4 text-sm leading-7 text-ink/72">
                  <div>
                    <div className="font-semibold text-ink">{t("resources.authors.haoyu.name")}</div>
                    <div>{t("resources.authors.haoyu.affiliation")}</div>
                  </div>
                  <div>
                    <div className="font-semibold text-ink">{t("resources.authors.xizhe.name")}</div>
                    <div>{t("resources.authors.xizhe.affiliation1")}</div>
                    <div>{t("resources.authors.xizhe.affiliation2")}</div>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </SectionShell>


        <section className="px-4 pb-20 pt-4 sm:px-6 lg:px-10">
          <div className="mx-auto max-w-7xl rounded-[32px] border border-white/75 bg-surface/85 px-6 py-8 text-sm text-ink/62 shadow-card">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="max-w-xl">
                <FormatMathText text={t("footer.builtBy")} />
              </div>
              <div className="flex items-center gap-3">
                <Link href="#hero" className="rounded-full border border-ink/10 px-4 py-2 text-ink transition hover:bg-ink/5">
                  {t("nav.backToTop")}
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
