"use client";

import { useMemo } from "react";

import { siteContent, type SiteContent } from "@/data/content/site-content";
import { useI18n } from "@/providers/i18n-provider";

export function useSiteContent(): SiteContent {
  const { language } = useI18n();

  return useMemo(() => siteContent[language] as SiteContent, [language]);
}
