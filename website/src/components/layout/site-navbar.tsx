"use client";

import Link from "next/link";
import Image from "next/image";

import { useMemo, useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

import { type Language, useI18n } from "@/providers/i18n-provider";
import { useThemeMode } from "@/providers/theme-provider";
import { cn } from "@/lib/utils";
import { US, CN } from "country-flag-icons/react/3x2";
import { Sun, Moon, Github } from "lucide-react";

type NavItem = {
  id: string;
  label: string;
};

type SiteNavbarProps = {
  navItems: NavItem[];
  onOpenSettings: () => void;
};

export function SiteNavbar({ navItems, onOpenSettings }: SiteNavbarProps) {
  const { t } = useI18n();

  return (
    <header className="sticky top-0 z-50 border-b border-white/70 bg-paper/85 backdrop-blur-md dark:border-white/5 dark:bg-slate-900/85">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-10">
        <Link href="#hero" className="flex items-center gap-3">
          <Image
            src="/icon.svg"
            alt="CLAP-S icon"
            width={40}
            height={40}
            priority
            className="h-10 w-10"
          />
          <div>
            <div className="text-sm font-semibold tracking-[0.16em] text-ink">CLAP-S</div>
            <div className="text-xs text-ink/55">{t("navbar.subtitle")}</div>
          </div>
        </Link>

        <nav className="hidden items-center gap-5 text-sm text-ink/72 xl:flex">
          {navItems.map((item) => (
            <Link key={item.id} href={`#${item.id}`} className="transition hover:text-ink">
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Link
              href="https://github.com/njnklab/CLAPs-algorithm"
              target="_blank"
              rel="noreferrer"
              className="mr-1 inline-flex h-9 w-9 items-center justify-center rounded-full border border-ink/10 bg-surface/90 text-ink shadow-sm transition-all hover:border-ink/25 hover:bg-surface dark:border-white/10 dark:bg-slate-900/50 dark:text-white dark:hover:border-white/20"
              aria-label="GitHub Repository"
            >
              <Github className="h-4 w-4" />
            </Link>
            <LanguageSwitcher />
            <ThemeToggle />
          </div>
          <button
            type="button"
            onClick={onOpenSettings}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-full border border-ink/12 bg-surface px-3 text-sm font-medium text-ink shadow-sm transition hover:bg-ink/5 dark:border-white/10 dark:bg-slate-900/60 dark:text-white dark:hover:bg-slate-800"
            aria-label={t("settings.open")}
          >
            <SettingsIcon />
            <span className="hidden md:inline">{t("settings.open")}</span>
          </button>
        </div>
      </div>
    </header>
  );
}

function ThemeToggle() {
  const { theme, toggleTheme } = useThemeMode();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className={cn(
        "inline-flex h-9 w-9 items-center justify-center rounded-full border border-ink/10 bg-surface/90 text-ink shadow-sm transition-all hover:border-ink/25 hover:bg-surface dark:border-white/10 dark:bg-slate-900/50 dark:text-white dark:hover:border-white/20"
      )}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={theme}
          initial={{ y: 10, opacity: 0, rotate: -90 }}
          animate={{ y: 0, opacity: 1, rotate: 0 }}
          exit={{ y: -10, opacity: 0, rotate: 90 }}
          transition={{ duration: 0.2 }}
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </motion.div>
      </AnimatePresence>
    </button>
  );
}

function LanguageSwitcher() {
  const { language, setLanguage, t } = useI18n();
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const options: { code: Language; label: string; secondary: string }[] = useMemo(() => [
    { code: "en", label: t("language.option.en"), secondary: "EN" },
    { code: "zh", label: t("language.option.zh"), secondary: "ZH" }
  ], [t]);

  const currentOption = options.find((opt) => opt.code === language) || options[0];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "inline-flex h-9 items-center gap-2 rounded-full border px-3 text-xs font-semibold transition-all duration-200 shadow-sm",
          isOpen
            ? "border-ink bg-ink text-white dark:border-white dark:bg-slate-800"
            : "border-ink/10 bg-surface/90 text-ink hover:border-ink/25 hover:bg-surface dark:border-white/10 dark:bg-slate-900/50 dark:text-white dark:hover:border-white/20"
        )}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <GlobeIcon className={cn("h-3.5 w-3.5", isOpen ? "text-white" : "text-ink/60 dark:text-white/70")} />
        <span>{currentOption.secondary}</span>
        <ChevronIcon className={cn("h-2.5 w-2.5 transition-transform duration-200", isOpen ? "rotate-180" : "rotate-0")} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.98 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="absolute right-0 mt-2 min-w-[140px] overflow-hidden rounded-2xl border border-ink/10 bg-white/95 p-1.5 shadow-xl backdrop-blur-xl dark:border-white/10 dark:bg-slate-900/95"
          >
            <div className="flex flex-col gap-1" role="listbox">
              {options.map((option) => (
                <button
                  key={option.code}
                  type="button"
                  onClick={() => {
                    setLanguage(option.code);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "flex items-center justify-between rounded-xl px-3 py-2 text-left text-xs font-medium transition-colors",
                    language === option.code
                      ? "bg-ink/5 text-ink dark:bg-white/10 dark:text-white"
                      : "text-ink/65 hover:bg-ink/[0.03] hover:text-ink dark:text-white/60 dark:hover:bg-white/5 dark:hover:text-white"
                  )}
                  role="option"
                  aria-selected={language === option.code}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-5 w-5 shrink-0 items-center justify-center overflow-hidden rounded-sm border border-black/5 bg-white shadow-sm dark:border-white/10 dark:bg-white/5">
                      {option.code === "en" ? <US className="h-full w-full object-cover" /> : <CN className="h-full w-full object-cover" />}
                    </div>
                    <span className="font-semibold">{option.label}</span>
                  </div>
                  {language === option.code && (
                    <div className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-ink text-white dark:bg-white dark:text-slate-900">
                      <CheckIcon className="h-2.5 w-2.5" />
                    </div>
                  )}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function GlobeIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="4"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06A1.65 1.65 0 0 0 15 19.4a1.65 1.65 0 0 0-1.61 1.2 1.65 1.65 0 0 1-3.18 0A1.65 1.65 0 0 0 8.6 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.2-1.61 1.65 1.65 0 0 1 0-3.18A1.65 1.65 0 0 0 4.6 8.6a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1.61-1.2 1.65 1.65 0 0 1 3.18 0A1.65 1.65 0 0 0 15 4.6a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.83 0 1.54.58 1.7 1.39a1.65 1.65 0 0 1 0 3.18A1.65 1.65 0 0 0 19.4 15z" />
    </svg>
  );
}
