import type { NextConfig } from "next";

const isGitHubActions = process.env.GITHUB_ACTIONS === "true";
const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1] ?? "CLAPs-algorithm";
const basePath = isGitHubActions ? `/${repoName}` : "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  eslint: {
    ignoreDuringBuilds: true
  },
  images: {
    unoptimized: true
  },
  basePath,
  assetPrefix: basePath,
  experimental: {
    optimizePackageImports: ["react-katex"]
  }
};

export default nextConfig;
