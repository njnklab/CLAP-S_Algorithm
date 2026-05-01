"use client";

import { motion } from "framer-motion";
import { FormatMathText } from "@/components/ui/core";

type SectionShellProps = {
  id: string;
  eyebrow: string;
  title: string;
  lead?: string;
  children: React.ReactNode;
};

export function SectionShell({
  id,
  eyebrow,
  title,
  lead,
  children
}: SectionShellProps) {
  return (
    <section id={id} className="scroll-mt-24 px-4 py-16 sm:px-6 lg:px-10">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.45, ease: "easeOut" }}
        className="mx-auto max-w-7xl"
      >
        <div className="mb-10 max-w-3xl">
          <div className="mb-3 text-xs font-semibold uppercase tracking-[0.28em] text-accent">
            {eyebrow}
          </div>
          <h2 className="text-balance font-serif text-3xl text-ink sm:text-4xl">{title}</h2>
          {lead ? <p className="mt-5 text-pretty text-lg leading-8 text-ink/78"><FormatMathText text={lead} /></p> : null}
        </div>
        {children}
      </motion.div>
    </section>
  );
}
