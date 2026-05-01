export type PrimerNode = {
  id: number;
  x: number;
  y: number;
};

export const primerNetworkNodes: PrimerNode[] = [
  { id: 1, x: 50, y: 20 },
  { id: 2, x: 18, y: 44 },
  { id: 3, x: 42, y: 48 },
  { id: 4, x: 66, y: 46 },
  { id: 5, x: 86, y: 40 },
  { id: 6, x: 58, y: 74 }
];

export const primerEdges = [
  [1, 3],
  [1, 4],
  [2, 3],
  [3, 5],
  [4, 5],
  [5, 6],
  [2, 6]
] as const;

export const primerMatchingOptions = {
  A: {
    edges: [
      [1, 4],
      [2, 3],
      [3, 5],
      [5, 6]
    ] as [number, number][],
    drivers: [1, 2],
    label: "Case A"
  },
  B: {
    edges: [
      [1, 4],
      [2, 3],
      [4, 5],
      [5, 6]
    ] as [number, number][],
    drivers: [1, 2],
    label: "Case B"
  }
} as const;

export type DuplexState = {
  id: string;
  labelKey: string;
  explanationKey: string;
  d1: number[];
  d2: number[];
  clap?: string;
};

export const unionContractionStates: DuplexState[] = [
  {
    id: "initial",
    labelKey: "union.steps.initial.label",
    explanationKey: "union.steps.initial.explanation",
    d1: [1, 2],
    d2: [4, 5],
  },
  {
    id: "reconfigured",
    labelKey: "union.steps.reconfigured.label",
    explanationKey: "union.steps.reconfigured.explanation",
    d1: [3, 4],
    d2: [3, 4],
  }
];

export interface ClapAlternatingPath {
  layer: 1 | 2;
  s: number;
  t: number;
  isSRelay: boolean;
  isTRelay: boolean;
  edges: [number, number][];
  nodes: number[];
}

export type TraceStep = {
  id: string;
  clap?: {
    s: number;
    t: number;
    segments: ClapAlternatingPath[];
  };
  d1: number[];
  d2: number[];
  dd1: number[];
  dd2: number[];
  cds: number[];
  cms: number[];
  note: string;
  delta: number;
  unionSize: number;
};

export const toyNodePositions = [
  { id: 1, x: 80, y: 20 },
  { id: 2, x: 50, y: 50 },
  { id: 3, x: 20, y: 20 },
  { id: 4, x: 20, y: 80 },
  { id: 5, x: 80, y: 80 },
  { id: 6, x: 80, y: 50 },
  { id: 7, x: 50, y: 80 },
  { id: 8, x: 20, y: 50 },
  { id: 9, x: 50, y: 20 }
];

export const toyBaseEdges = {
  1: [
    [2, 7], [2, 9], [2, 6], [4, 7], [4, 8], [6, 1], [6, 2], [7, 4], [7, 5], [8, 2], [8, 3], [8, 4]
  ] as const,
  2: [
    [1, 9], [2, 5], [2, 6], [2, 7], [6, 1], [6, 2], [8, 2], [8, 4], [9, 3]
  ] as const
};

export const toyInitialMatchings = {
  1: [[2, 6], [4, 8], [6, 2], [7, 5], [8, 4]] as [number, number][],
  2: [[1, 9], [2, 5], [6, 1], [8, 2], [9, 3]] as [number, number][]
};

export const traceSteps: TraceStep[] = [
  {
    id: "start",
    d1: [1, 3, 7, 9],
    d2: [4, 6, 7, 8],
    dd1: [1, 3, 9],
    dd2: [4, 6, 8],
    cds: [7],
    cms: [2, 5],
    note:
      String.raw`Initial state. The drivers in $L_1$ are $\{1,3,5,9\}$ and in $L_2$ are $\{4,5,7,8\}$. Shared drivers $\{5\}$ are in $CDS$.`,
    delta: 6,
    unionSize: 7
  },
  {
    id: "p1",
    clap: {
      s: 1,
      t: 4,
      segments: [
        {
          layer: 1,
          s: 1,
          t: 4,
          isSRelay: false,
          isTRelay: false,
          edges: [[6, 1], [6, 2], [8, 2], [8, 4]],
          nodes: [1, 6, 2, 8, 4]
        }
      ]
    },
    d1: [3, 4, 7, 9],
    d2: [4, 6, 7, 8],
    dd1: [3, 9],
    dd2: [6, 8],
    cds: [4, 7],
    cms: [1, 2, 5],
    note:
      String.raw`CLAP 1: $1 \gets 6 \to 2 \gets 8 \to 4$ in $L_1$. $s=1$ and $t=4$ are removed from their respective driver sets.`,
    delta: 4,
    unionSize: 6
  },
  {
    id: "p2",
    clap: {
      s: 3,
      t: 6,
      segments: [
        {
          layer: 1,
          s: 3,
          t: 2,
          isSRelay: false,
          isTRelay: true,
          edges: [[8, 3], [8, 2]],
          nodes: [3, 8, 2]
        },
        {
          layer: 2,
          s: 2,
          t: 4,
          isSRelay: true,
          isTRelay: true,
          edges: [[8, 2], [8, 4]],
          nodes: [2, 8, 4]
        },
        {
          layer: 1,
          s: 4,
          t: 5,
          isSRelay: true,
          isTRelay: true,
          edges: [[7, 4], [7, 5]],
          nodes: [4, 7, 5]
        },
        {
          layer: 2,
          s: 5,
          t: 6,
          isSRelay: true,
          isTRelay: false,
          edges: [[2, 5], [2, 6]],
          nodes: [5, 2, 6]
        }
      ]
    },
    d1: [2, 5, 7, 9],
    d2: [2, 5, 7, 8],
    dd1: [9],
    dd2: [8],
    cds: [2, 5, 7],
    cms: [1, 3, 4, 6],
    note:
      String.raw`CLAP 2: Cross-layer path $3 \to 2 \to 4 \to 5 \to 6$. Node 3 is removed from $D_1$.`,
    delta: 2,
    unionSize: 5
  },
  {
    id: "p3",
    clap: {
      s: 9,
      t: 8,
      segments: [
        {
          layer: 1,
          s: 9,
          t: 6,
          isSRelay: false,
          isTRelay: true,
          edges: [[2, 9], [2, 6]],
          nodes: [9, 2, 6]
        },
        {
          layer: 2,
          s: 6,
          t: 7,
          isSRelay: true,
          isTRelay: true,
          edges: [[2, 6], [2, 7]],
          nodes: [6, 2, 7]
        },
        {
          layer: 1,
          s: 7,
          t: 8,
          isSRelay: true,
          isTRelay: false,
          edges: [[4, 7], [4, 8]],
          nodes: [7, 4, 8]
        }
      ]
    },
    d1: [2, 5, 6, 8],
    d2: [2, 5, 6, 8],
    dd1: [],
    dd2: [],
    cds: [2, 5, 6, 8],
    cms: [1, 3, 4, 7, 9],
    note:
      String.raw`CLAP 3: $9 \to 6 \to 7 \to 8$. Node 9 is removed from $D_1$ and 8 is removed from $D_2$. State approaches stability.`,
    delta: 0,
    unionSize: 4
  }
];