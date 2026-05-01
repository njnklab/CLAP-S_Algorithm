# CLAP-S Paper Website

Interactive paper website for **Optimized Control of Duplex Networks**.

[Live site](https://zxzok.github.io/CLAPs-website/) · [Paper PDF](https://zxzok.github.io/CLAPs-website/paper/revision.pdf) · [Supplementary materials](https://zxzok.github.io/CLAPs-website/paper/supplementary-materials.pdf)

## Overview

This repository contains a single-page explanatory website for the paper *Optimized Control of Duplex Networks*. The site is designed as a scrollytelling companion to the manuscript: it introduces matching-based structural controllability in directed networks, explains why driver redundancy appears in aligned-node duplex systems, and visualizes how CLAP-S reduces the union of driver nodes while keeping each layer's minimum driver budget fixed.

The website stays within the scope studied by the paper:

- aligned-node duplex networks
- uncoupled layers with no explicit inter-layer state-transition links
- fixed minimum driver budgets obtained from maximum matchings
- CLAP-S and CLAP-G as matching reconfiguration methods under that setting

It does **not** reframe the work as minimum dominating set, feedback vertex set, deep learning, reinforcement learning, or a generic AI optimizer.

## What The Website Covers

- a structural controllability primer based on bipartite representation and maximum matching
- the fixed-budget union driver set objective
- interactive explanations of difference mass, DD1, DD2, CDS, CMS, and CLAP updates
- a results explorer built from the paper's synthetic and real-world benchmark tables
- resources for reading, citing, and downloading the paper

## Tech Stack

- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Framer Motion
- D3
- KaTeX

## Repository Structure

```text
src/
  app/                  Next.js app entry
  components/           shared UI plus layout shells
    layout/             navbar, floating panels, and shells
    ui/                 primitive cards, legends, steppers
    views/              legacy page-level composites
  features/             domain-specific modules (animations, results)
  data/                 website copy and toy example data
  providers/            i18n, theme, and visualization contexts
  lib/                  CSV loading and result summarization
data/results/           benchmark CSV files used by the explorer
public/paper/           paper PDF and supplementary materials
doc/                    implementation plan & change log
docs/                   legacy design/content brief
.github/workflows/      GitHub Pages deployment workflow
```

## Local Development

Install dependencies and start the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) or the next available port printed by Next.js.

## Production Build

Generate the static export used by GitHub Pages:

```bash
npm run build
```

Serve the exported `out/` directory locally after a build:

```bash
npm run start
```

This uses `python3 -m http.server` and serves the static export on port `3000`.

## Data Provenance

The interactive results section reads from the CSV files committed under [`data/results/`](./data/results):

- `optimization_amount.csv`
- `optimization_proportion.csv`
- `real_networks.csv`

These are the benchmark tables used by the site to summarize:

- synthetic ER-ER, BA-BA, and hybrid ER-BA duplex settings
- runtime, gain, and average CLAP path length statistics
- real-world biological, neuronal, social, and human relationship duplex networks

## GitHub Pages Deployment

This repository is configured for GitHub Pages via GitHub Actions.

- every push to `main` triggers the deployment workflow
- Next.js builds a static export into `out/`
- the workflow publishes that artifact to GitHub Pages

If you fork this repository, make sure Pages is enabled in the repository settings and set to **GitHub Actions** as the source.

## Included Assets

- website source code under `src/`
- result data used by the charts under `data/results/`
- manuscript PDF under `public/paper/revision.pdf`
- supplementary materials under `public/paper/supplementary-materials.pdf`
- roadmap and change log under `doc/implementation-plan.md`
- Chinese design/content planning document under `docs/paper-website-spec.md`

## Citation

If you use the website or adapt its content, please cite the underlying paper rather than the website itself.
