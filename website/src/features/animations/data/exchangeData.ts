export type ExchangeNode = {
  id: number;
  x: number;
  y: number;
};

// Based on the paper figure C
export const exchangeNodes: ExchangeNode[] = [
  { id: 1, x: 20, y: 45 },
  { id: 2, x: 35, y: 20 },
  { id: 3, x: 65, y: 20 },
  { id: 4, x: 50, y: 45 },
  { id: 5, x: 35, y: 70 },
  { id: 6, x: 65, y: 70 },
  { id: 7, x: 80, y: 45 }
];

export const exchangeEdges = [
  [2, 1], [2, 3], [3, 6], [4, 2], [4, 3], [4, 5], [6, 4], [6, 5], [6, 7]
] as const;

// Base configuration for dynamic generation
export const initialMatching: [number, number][] = [[2, 1], [3, 6], [4, 5], [6, 7]];
export const initialDrivers = [2, 3, 4];

export type ExchangeStepInfo = {
  titleKey: string;
  bodyKey: string;
};

export const exchangeSteps: ExchangeStepInfo[] = [
  {
    titleKey: "exchange.steps.initial.title",
    bodyKey: "exchange.steps.initial.body",
  },
  {
    titleKey: "exchange.steps.alternating.title",
    bodyKey: "exchange.steps.alternating.body",
  },
  {
    titleKey: "exchange.steps.symmetric.title",
    bodyKey: "exchange.steps.symmetric.body",
  },
  {
    titleKey: "exchange.steps.reconfigured.title",
    bodyKey: "exchange.steps.reconfigured.body",
  }
];

export type ExchangePath = {
  s: number;
  template: string; // Internal arrow format: <- and ->
  targets: {
    t: number;
    edges: [number, number][];
    // description will be intelligently parsed from template in the UI
  }[];
};

export const exchangePaths: ExchangePath[] = [
  {
    s: 2,
    template: "$2^- <- 4^+ -> 5^- <- 6^+ -> 7^-$",
    targets: [
      {
        t: 5,
        edges: [[4, 2], [4, 5]]
      },
      {
        t: 7,
        edges: [[4, 2], [4, 5], [6, 5], [6, 7]]
      }
    ]
  },
  {
    s: 3,
    template: "$3^- <- 4^+ -> 5^- <- 6^+ -> 7^-$",
    targets: [
      {
        t: 5,
        edges: [[4, 3], [4, 5]]
      },
      {
        t: 7,
        edges: [[4, 3], [4, 5], [6, 5], [6, 7]]
      }
    ]
  },
  {
    s: 3,
    template: "$3^- <- 2^+ -> 1^-$",
    targets: [
      {
        t: 1,
        edges: [[2, 3], [2, 1]]
      }
    ]
  },
  {
    s: 4,
    template: "$4^- <- 6^+ -> 7^-$",
    targets: [
      {
        t: 7,
        edges: [[6, 4], [6, 7]]
      }
    ]
  }
];
