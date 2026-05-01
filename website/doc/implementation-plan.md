# Website Refactor & Feature Plan (updated 2026-03-19)

> Status legend: ☐ pending · ◐ in progress · ☑ done

## Goals
- Establish a clearer `src` structure (providers/layout/features) so i18n, theming, and visualization controls have dedicated homes.
- Ship navbar-based bilingual support (en/zh) with automatic browser-language detection and persisted choice.
- Introduce a floating visualization settings panel connected to `NetworkStyleConfig` knobs, triggered from the navbar.
- Add dark mode with automatic system preference detection and share the toggle inside the settings panel; ensure visualization styling responds to theme changes.

## Work Breakdown
1. ☑ **Structure realignment**
   - Move context/provider logic into `src/providers/` (i18n, new theme + visualization settings providers).
   - Carve out `src/components/layout/` for shared chrome (navbar, floating panels) and relocate the navbar from `views`.
   - Keep animation + results modules under `src/features/` but ensure they rely on shared providers/hooks.
   - Update import aliases + Tailwind config paths after the move; adjust README + docs to describe the new layout.
2. ☑ **i18n enablement**
   - Expand the existing provider to hydrate its initial state from `navigator.language`, expose translation hooks, and wrap `app/layout`.
   - Replace hard-coded English navigation/text (hero CTA copy, FAQ headings, footer) with `siteContent` lookups keyed by the active language.
   - Add a navbar language switcher (two-option segmented control) that stores selection in `localStorage`.
3. ☑ **Visualization settings panel**
   - Create a settings button on the navbar’s rightmost slot.
   - Build a floating panel using portal + headless UI primitives that surfaces grouped controls for `NetworkStyleConfig` values (colors, sizes, hover behavior).
   - Wire controls to a context (`VisualizationSettingsProvider`) that feeds the relevant animation components.
4. ☑ **Dark mode**
   - Introduce a theme system (auto-detect via `matchMedia`, persist preference) and update Tailwind + global styles to support `class`-based dark mode.
   - Mirror the theme toggle inside the settings panel alongside the visualization controls; default to system preference on first load.
   - Ensure default style tokens (colors, gradients, panel backgrounds) respect CSS variables for both themes and propagate into `NetworkStyleConfig`.

## Tracking
- Latest tailwind/theme decisions live here.
- Update each section’s checkbox/status as milestones complete.
- 2026-03-19: Audited dark-mode surfaces and replaced the lingering `bg-white*` backgrounds with the theme-driven `bg-surface` token + new CSS var so elevated cards stay legible at night.
- 2026-03-19: Softened gradient cards/buttons, localized the primer + union-demo text (incl. new language dropdown), synced visualization legends/arrows with the live settings palette, and pointed site metadata at `src/app/icon.svg` for consistent favicons.
- 2026-03-19: Locked the BibTeX block styling, swapped the navbar emblem for `src/app/icon.svg`, expanded the visualization settings (node stroke + regular colors, edge palettes, hover color), tightened dark-mode button contrast, and wired directed-arrowheads to the configurable palette.
- 2026-03-19: **Animation C refinement** plan (done)
  - ☑ Audit `src/features/animations/exchange-theorem` to map where Animation C derives its step data, selectable options, and two-pane layout (bipartite vs directed graph) so we know the right hooks to extend.
  - ☑ Update the step logic so optional source selection (`s`) appears from step 1, while the thicker alternate path + terminal node `t` only unlock starting at step 2; verify the stepper copy + CTA hints remain localized.
  - ☑ Re-balance the layout so the bipartite and directed canvases render at equal widths/heights on wide screens, and ensure optional actions + step tips stay visible alongside the canvases (no overflow/cropping in light or dark themes).
  - ☑ Exercise the new flow on desktop + tablet breakpoints and document any additional copy/visual tweaks needed for clarity inside this plan.
- 2026-03-19: Added localized step copy + striped node accents so the currently selected $s$/$t$ pairs stand out in both the bipartite and directed canvases without breaking the dark-mode palette.
- 2026-03-19: Rebuilt Animation D with the dual-layer visualization grid (bipartite + directed views per layer), synchronized hover states, phase-aware playback (C1→C2 search pauses + C2→C3→C4 exchanges), and the detailed-pause toggle so reviewers can inspect each alternating segment before the exchange fires.
