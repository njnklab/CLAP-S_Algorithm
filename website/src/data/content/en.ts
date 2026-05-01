export type HeroOption = {
  id: string;
  tone: string;
  title: string;
  subtitle: string;
  value: string;
  primaryCta: string;
  secondaryCta: string;
};

export type SectionCopy = {
  id: string;
  eyebrow: string;
  title: string;
  lead: string;
  paragraphs: string[];
  bullets?: string[];
};

export type TheoryModule = {
  id: string;
  title: string;
  intuition: string[];
  formal: string[];
  why: string;
  visualAid: string;
};

export type FaqItem = {
  question: string;
  answer: string;
};

export type GlossaryItem = {
  term: string;
  meaning: string;
};

export const heroOptions: HeroOption[] = [
  {
    id: "academic",
    tone: "Academic",
    title: "Optimized Control of Duplex Networks",
    subtitle:
      "A matching-based framework for minimizing the union of driver nodes while preserving each layer's minimum control budget.",
    value:
      String.raw`CLAP-S reconfigures layer-wise maximum matchings to reuse actuators across two directed layers without changing $k_1$ or $k_2$.`,
    primaryCta: "Read the paper",
    secondaryCta: "Explore the algorithm"
  },
  {
    id: "explainer",
    tone: "Explainer",
    title: "Two layers can each be controllable and still waste actuators together",
    subtitle:
      "This paper shows how to keep both layers minimally controllable while shrinking the number of unique nodes that must be actuated.",
    value:
      "The key is not adding more control, but rearranging maximum matchings so that the two minimum driver sets overlap more.",
    primaryCta: "See why redundancy happens",
    secondaryCta: "Watch CLAP-S step by step"
  },
  {
    id: "visual",
    tone: "Visual",
    title: "Align the unmatched. Contract the union.",
    subtitle:
      "A duplex network can keep both minimal driver budgets fixed and still become cheaper to actuate through cross-layer reconfiguration.",
    value:
      String.raw`Each successful CLAP update removes two cross-layer conflicts and contracts the union driver set by exactly one node.`,
    primaryCta: "Download PDF",
    secondaryCta: "View the demo"
  }
];

export const navItems = [
  { id: "hero", label: "Overview" },
  { id: "faq", label: "Glossary" },
  { id: "why", label: "Why It Matters" },
  { id: "primer", label: "Primer" },
  { id: "problem", label: "Problem" },
  { id: "algorithm", label: "Algorithm" },
  { id: "theory", label: "Theory" },
  { id: "results", label: "Results" },
  { id: "resources", label: "Resources" }
];

export const sectionCopy: SectionCopy[] = [
  {
    id: "why",
    eyebrow: "Why this problem matters",
    title: "Independent layer control creates avoidable actuator redundancy",
    lead:
      "In an aligned-node duplex, the same physical entities appear in both layers. If each layer independently chooses one minimum driver set, the two choices can be structurally valid and still poorly aligned.",
    paragraphs: [
      "That mismatch matters whenever actuation is attached to shared entities: instrumenting a gene, stimulating a brain region, intervening on an account, or monitoring a financial institution. The engineering cost is paid in unique actuated nodes, not in two separate layer-wise lists.",
      "The paper therefore does not ask whether a layer can be controlled with fewer drivers. Each layer already sits at its minimum budget implied by maximum matching. The question is subtler: can those layer-wise minimum driver sets be reconfigured so that more of them land on the same nodes?",
      "This is exactly where driver redundancy appears. A node that is a driver in only one layer contributes to deployment cost without contributing to reuse. The optimization target is to remove as many of those cross-layer disagreements as the network structure allows."
    ],
    bullets: [
      "Same node set, two directed layers, no explicit inter-layer state-transition links.",
      String.raw`Per-layer budgets $k_1$ and $k_2$ stay fixed throughout.`,
      String.raw`Cost is measured by the union driver set $U = D_1 \cup D_2$, not by either layer alone.`
    ]
  },
  {
    id: "primer",
    eyebrow: "Primer",
    title: "From single-layer structural controllability to duplex coordination",
    lead:
      "The site starts from the standard matching-based view of structural controllability and then lifts it into a duplex setting.",
    paragraphs: [
      "For a directed network, the first move is to build its bipartite representation: every node is copied into an out-side and an in-side, and each directed edge becomes a bipartite edge from $u^+$ to $v^-$. A maximum matching then selects as many target-side copies as possible without conflict.",
      String.raw`The unmatched target-side copies correspond to driver nodes. Their count is $N_D = \max(1, N - |M^*|)$, and any driver set produced this way is a minimum driver set in the structural controllability sense.`,
      "The crucial freedom is that maximum matchings are often not unique. Alternating paths let one valid maximum matching transform into another without changing its size. In a single layer, that means a minimum driver set can change composition while keeping the same minimum budget.",
      "The duplex problem begins when both layers have this freedom at once. Each layer can remain minimally actuated, but the two chosen driver sets may overlap poorly. The optimization space is therefore the product of two maximum-matching spaces, one per layer."
    ]
  },
  {
    id: "problem",
    eyebrow: "Problem formulation",
    title: "Fixed budgets, minimum union",
    lead:
      "The paper studies the aligned-node, uncoupled duplex setting and formulates a budget-preserving union contraction problem.",
    paragraphs: [
      "Let $M_1$ and $M_2$ be maximum matchings for layer 1 and layer 2. Their driver sets $D_1$ and $D_2$ each have fixed sizes $k_1$ and $k_2$, because those sizes come from the corresponding maximum matching cardinalities. The task is to search only within these layer-wise maximum matching spaces.",
      String.raw`The objective is the Union Driver Set $U = D_1 \cup D_2$: the set of unique nodes that must be actuated so both layers are structurally controllable. The optimum is the smallest possible $|U|$ over all pairs of layer-wise maximum matchings.`,
      "This is not the same as taking one arbitrary minimum driver set from each layer and merging them. It is also not a dominating-set problem, and it does not require driver locations to be identical across layers. The paper optimizes reuse inside the matching-based structural controllability framework."
    ],
    bullets: [
      String.raw`Feasible set: $\Omega = M_1(k_1) \times M_2(k_2)$.`,
      String.raw`Objective: minimize $|U(M_1, M_2)| = |D_1 \cup D_2|$.`,
      "Constraint: every move must preserve each layer's maximum matching size and therefore its minimum driver budget."
    ]
  },
  {
    id: "algorithm",
    eyebrow: "Key idea of CLAP-S",
    title: "Use cross-layer alternating operations to align driver sets",
    lead:
      "Because a minimum driver set is not unique, budget-preserving driver exchange is possible within a layer. CLAP-S turns that local flexibility into a cross-layer optimization mechanism.",
    paragraphs: [
      String.raw`Inside one layer, an alternating path can swap a driver node $s$ with a non-driver node $t$ while preserving the maximum matching size. This Driver Exchange Principle is the atomic move underlying the whole framework.`,
      String.raw`Across two layers, the paper chains such exchanges into a Cross-Layer Augmenting Path, or CLAP. A CLAP starts at a node in $DD_1$, ends at a node in $DD_2$, alternates layers along the way, and uses relay nodes that must satisfy strict consistency conditions.`,
      String.raw`Difference mass $\Delta = |DD_1| + |DD_2|$ is the key bookkeeping variable. Under fixed budgets, $|U| = (k_1 + k_2 + \Delta) / 2$, so shrinking the union is exactly the same as shrinking $\Delta$. Every successful CLAP update reduces $\Delta$ by 2 and therefore reduces $|U|$ by 1.`,
      "CLAP-S searches for the shortest such path with a layer-alternating BFS. Shortest paths are not just faster to find; in the paper they also support an atomic update by ensuring the constituent witness paths can be applied without internal conflict."
    ]
  },
  {
    id: "theory",
    eyebrow: "Theory made intuitive",
    title: "A local move with a global certificate",
    lead:
      "The theoretical point is not only that CLAP updates help, but that they are complete: if no CLAP exists, no further improvement is possible inside the fixed-budget feasible space.",
    paragraphs: [
      String.raw`The paper partitions nodes into four sets: $CDS$, $CMS$, $DD_1$, and $DD_2$. Difference nodes are where cross-layer redundancy lives. Relays must alternate between $CMS$ and $CDS$ so that the path transports disagreement without creating new disagreement in the middle.`,
      String.raw`The CLAP Gain Theorem shows that applying a valid CLAP preserves $k_1$ and $k_2$, decreases $\Delta$ by exactly 2, and contracts $|U|$ by exactly 1. This makes every accepted path an exact one-step descent in the union objective.`,
      "The CLAP-or-Optimal theorem plays the role of an optimality certificate. A configuration is globally optimal over the feasible space if and only if it is CLAP-stable, meaning that no feasible CLAP remains."
    ]
  },
  {
    id: "results",
    eyebrow: "Experimental evidence",
    title: "CLAP-S matches the exact optimum and outperforms random sampling",
    lead:
      "The paper evaluates CLAP-S on synthetic ER-ER, BA-BA, and ER-BA duplex networks and on diverse real-world duplex systems.",
    paragraphs: [
      "On the main synthetic benchmark, CLAP-S matches ILP-Exact on every tested configuration. In the figures, the CLAP-S and ILP curves overlap, showing that the path-based method reaches the same optimum union size while remaining far more scalable than exact search.",
      "Against RSU, a baseline that samples many maximum matchings and keeps the best observed union, CLAP-S achieves consistently smaller unions and does so more reliably. CLAP-G, a greedy one-segment variant, gives a speed-quality compromise: faster than CLAP-S, typically better than RSU, but not always optimal.",
      "Optimization space depends on density, overlap, and initial disagreement. Sparse to moderately sparse regimes offer the most headroom, and the initial difference between layer-wise driver sets strongly predicts how much union contraction is possible.",
      "In real-world networks spanning biological, neuronal, social, and human relationship domains, average CLAP lengths stay close to 1. This matters because it suggests most cross-layer conflicts can be resolved by very local reconfiguration rather than deep global surgery."
    ]
  },
  {
    id: "applications",
    eyebrow: "Applications and implications",
    title: "Why actuator reuse matters in practice",
    lead:
      "The paper does not claim a universal controller for all multilayer systems. Its contribution is narrower and more useful: a principled way to reuse intervention points across two minimally actuated directed layers.",
    paragraphs: [
      "In gene-regulatory and interaction duplexes, the same molecule or gene may appear in both layers, so a smaller union means fewer unique intervention targets. In brain-network settings, it means fewer distinct loci that need stimulation or monitoring when two directed connectivity descriptions are both relevant.",
      "For social or information networks, different directed interaction layers can share accounts or actors. A smaller union means fewer unique intervention points to influence or observe across the combined system. In finance, it translates into fewer institutions that must be instrumented across multiple directed relation types.",
      "The key value is not abstract sparsity. It is coordinated reuse under layer-wise minimality: keep each layer at the smallest structural budget it needs, but spend that budget on overlapping physical nodes whenever the topology allows."
    ]
  },
  {
    id: "limits",
    eyebrow: "Limitations and future work",
    title: "Exact within scope, not beyond scope",
    lead:
      "The optimality claim is tied to a specific regime: aligned-node duplex networks, uncoupled layers, and fixed minimum budgets derived from maximum matchings.",
    paragraphs: [
      String.raw`If budgets are relaxed beyond $k_1$ and $k_2$, the current solution remains feasible but need not be globally optimal for the larger search space. If more than two layers are involved, pairwise coordination is a reasonable baseline but not the same theorem. If explicit inter-layer coupling edges are introduced, the block-diagonal decomposition no longer applies and a different control problem emerges.`,
      String.raw`The paper also notes a structural limitation: some networks simply do not admit enough usable relay nodes in $CDS$ or $CMS$. In those cases, CLAP-S still returns the true optimum within the fixed-budget feasible space, but the optimum may remain well above the trivial lower bound $\max(k_1, k_2)$.`
    ],
    bullets: [
      "Future directions named in the paper: relaxed budgets, more than two layers, explicit inter-layer coupling.",
      "None of those extensions are claimed as solved here."
    ]
  }
];

export const theoryModules: TheoryModule[] = [
  {
    id: "minimum-input",
    title: "Minimum input theorem",
    intuition: [
      "In matching-based structural controllability, drivers are not chosen heuristically. They are determined by which target-side vertices remain unmatched after a maximum matching is chosen.",
      "The matching size tells you how many targets can be internally covered. The rest must receive direct external input."
    ],
    formal: [
      String.raw`For a directed graph with N nodes and maximum matching $M^*$, the minimum driver count is $N_D = \max(1, N - |M^*|)$.`,
      String.raw`For duplex layer $l$, the fixed minimal budget is $k_l = N - |M^*_l|$ whenever at least one driver is required.`
    ],
    why:
      String.raw`This is the reason the paper treats $k_1$ and $k_2$ as structural invariants rather than optimization variables.`,
    visualAid:
      "Animate a directed graph, its bipartite representation, and the unmatched target-side copies becoming drivers."
  },
  {
    id: "objective-equivalence",
    title: String.raw`Why minimizing $|U|$ is the same as minimizing $\Delta$`,
    intuition: [
      "Under fixed budgets, the total amount of driver mass across both layers is fixed. The only thing that can change is how much of that mass is shared versus split across different nodes.",
      String.raw`Difference mass $\Delta$ counts exactly the driver nodes that are unique to one layer. Reduce those disagreements, and the union shrinks automatically.`
    ],
    formal: [
      String.raw`Let $DD_1 = D_1 \setminus D_2$ and $DD_2 = D_2 \setminus D_1$, and define $\Delta = |DD_1| + |DD_2|$.`,
      String.raw`Then $|U| = (k_1 + k_2 + \Delta) / 2$.`
    ],
    why:
      "This converts the engineering objective of actuator reuse into a graph-reconfiguration objective over disagreement mass.",
    visualAid:
      "Show the four-set partition $CDS$, $CMS$, $DD_1$, $DD_2$ and update the equation live as nodes move between sets."
  },
  {
    id: "driver-exchange",
    title: "Driver Exchange Principle",
    intuition: [
      "A maximum matching is often not unique. That non-uniqueness creates room to swap one driver out and another node in without increasing the required number of drivers.",
      "Alternating paths are the safe routes along which that swap can happen."
    ],
    formal: [
      String.raw`If $s$ is a driver and $t$ is a non-driver in one layer, there exists an M-alternating path from $s^-$ to $t^-$ if and only if toggling along that path preserves the matching size and replaces s with t in the driver set.`
    ],
    why:
      "Every CLAP is built from these budget-preserving single-layer exchanges.",
    visualAid:
      "Highlight an alternating path, toggle matched and unmatched edges, then show the driver label move from s to t."
  },
  {
    id: "clap-gain",
    title: "CLAP Gain",
    intuition: [
      String.raw`A CLAP starts where layer 1 has a surplus driver relative to layer 2 and ends where layer 2 has the opposite mismatch. The path transports that disagreement across consistent relay nodes.`,
      "Only the endpoints reduce disagreement; the relays stay difference-neutral."
    ],
    formal: [
      String.raw`Applying a valid CLAP preserves $|D_1| = k_1$ and $|D_2| = k_2$, decreases $\Delta$ by 2, and therefore decreases $|U|$ by 1.`
    ],
    why:
      "This makes each successful CLAP an exact, auditable improvement step rather than a heuristic nudge.",
    visualAid:
      String.raw`Use a step trace that updates $DD_1$, $DD_2$, $CDS$, $CMS$ after each segment and flashes $\Delta$ minus 2.`
  },
  {
    id: "clap-optimal",
    title: "CLAP-or-Optimal",
    intuition: [
      String.raw`If a better fixed-budget configuration existed, the difference between the current and better matchings would contain a path-like structure that carries surplus from $DD_1$ to $DD_2$.`,
      "That structure is exactly what the paper calls a CLAP."
    ],
    formal: [
      String.raw`A feasible configuration minimizes $|U|$ over $\Omega$ if and only if it is CLAP-stable, meaning no feasible CLAP exists.`
    ],
    why:
      "This is the global optimality certificate that distinguishes CLAP-S from sampling and local heuristics.",
    visualAid:
      "Show two states side by side, outline a hypothetical improving path, then collapse to the statement: no CLAP, no improvement."
  },
  {
    id: "shortest-search",
    title: "Why shortest-path BFS is enough",
    intuition: [
      "Searching the whole combinatorial space of matching pairs would be hopeless. But the improvement primitive is local enough that a layer-alternating BFS can discover the next valid CLAP directly from the current state.",
      "Choosing the shortest CLAP is useful both computationally and operationally: it tends to keep updates local and supports an atomic application."
    ],
    formal: [
      "The paper's FindShortestCLAP routine performs a BFS over nodes paired with the next layer label, tracks predecessors, and reconstructs the first reached target in $DD_2$.",
      "For shortest CLAPs, the witness paths used in each layer are edge-disjoint, which supports a conflict-free atomic update."
    ],
    why:
      "This is the algorithmic reason CLAP-S can be both exact and scalable enough to compare favorably with ILP and RSU.",
    visualAid:
      "Animate queue, frontier, predecessor map, and target hit in technical mode; show only expanding rings and layer switches in intuition mode."
  }
];

export const whyNew = [
  "It does not simply compute one minimum driver set per layer and take their union. It searches within the space of layer-wise maximum matchings to improve overlap.",
  "It does not force the same driver locations in both layers. Identical placement is a stricter constraint than the paper's minimum-union objective.",
  "It is not a multilayer minimum dominating set or FVS problem. The framework stays inside matching-based structural controllability for directed networks.",
  "Its distinctive contribution is a complete cross-layer alternating-path primitive with a verifiable global optimality certificate."
];

export const applications = [
  {
    title: "Gene regulation and molecular interaction duplexes",
    body:
      "Different directed biological layers may require distinct driver sets if solved independently. A smaller union means fewer unique intervention targets while keeping each layer minimally controllable."
  },
  {
    title: "Brain networks",
    body:
      "If two directed descriptions of the same brain system must both remain controllable, actuator reuse translates into fewer distinct loci that need stimulation or monitoring."
  },
  {
    title: "Social and information systems",
    body:
      "Retweet, mention, follow, or other directed layers share the same actors. Union minimization means fewer unique accounts or agents must be targeted across the combined intervention plan."
  },
  {
    title: "Financial multi-relation networks",
    body:
      "Different directed exposure or ownership layers can share institutions. Reusing driver locations reduces the number of unique institutions that must be instrumented or controlled."
  }
];

export const faqItems: FaqItem[] = [
  {
    question: "Why not just find an MDS for each layer and take the union?",
    answer:
      "Because minimum driver sets are generally not unique. An arbitrary pair of per-layer MDS choices can contain avoidable disagreement. The paper optimizes within the space of maximum matchings to find a smaller union without changing either layer's minimum budget."
  },
  {
    question: "Why is the fixed-budget assumption important?",
    answer:
      String.raw`It keeps the problem inside the matching-based minimum-budget regime where each layer already uses the fewest possible drivers. That is what makes $\Delta$ a sufficient objective variable and what enables the CLAP-or-Optimal theorem.`
  },
  {
    question: "Why does 'no CLAP' imply optimality?",
    answer:
      String.raw`The paper proves that any strictly better fixed-budget state would induce a cross-layer improving path between $DD_1$ and $DD_2$. Therefore, if no feasible CLAP exists, no further budget-preserving reduction of $|U|$ exists either.`
  },
  {
    question: "How is this different from requiring identical drivers across layers?",
    answer:
      "Identical driver placement is a stronger constraint. The paper only minimizes the union. One layer can keep drivers the other layer does not use, as long as the total number of unique actuated nodes is as small as possible."
  },
  {
    question: "How is this different from dominating-set or FVS approaches?",
    answer:
      "Those optimize different control surrogates. This work remains in matching-based structural controllability, where unmatched target-side nodes of a maximum matching define the minimum driver set."
  },
  {
    question: String.raw`Can the optimum always reach $\max(k_1, k_2)$?`,
    answer:
      String.raw`No. That lower bound is achievable only if the topology permits one driver set to nest completely inside the other. CLAP-S finds the true fixed-budget optimum whether or not that lower bound is attainable.`
  }
];

export const glossaryItems: GlossaryItem[] = [
  {
    term: "Bipartite representation",
    meaning:
      "A graph construction that duplicates each node into source-side and target-side copies so matching theory can be applied to structural controllability."
  },
  {
    term: "Maximum matching",
    meaning:
      "A largest possible set of non-conflicting bipartite edges. Its size determines the minimum number of required driver nodes."
  },
  {
    term: "Driver nodes",
    meaning:
      "Nodes whose target-side copies are unmatched by the chosen maximum matching and therefore need external control input, recorded as $D_\ell$ for layer $\ell$."
  },
  {
    term: "Union Driver Set $(UDS)$",
    meaning:
      String.raw`The set of unique nodes appearing in at least one layer's driver set: $U = D_1 \cup D_2$.`
  },
  {
    term: String.raw`Difference mass $\Delta$`,
    meaning:
      String.raw`The total number of nodes that are drivers in exactly one layer: $\Delta = |DD_1| + |DD_2|$.`
  },
  {
    term: String.raw`$DD_1$ / $DD_2$`,
    meaning:
      "Nodes that are drivers only in layer 1 or only in layer 2. They quantify cross-layer redundancy."
  },
  {
    term: "Consistent Driver Set $(CDS)$",
    meaning:
      "Nodes that are drivers in both layers."
  },
  {
    term: "Consistent Matched Set $(CMS)$",
    meaning:
      "Nodes that are matched (non-drivers) in both layers."
  },
  {
    term: "Admissible segment",
    meaning:
      "A single-layer directional driver displacement supported by an alternating path. Layer 1 pushes a driver forward; layer 2 pulls a driver backward."
  },
  {
    term: "Relay feasibility",
    meaning:
      "The rule that relays reached via layer 1 must lie in $CMS$, while relays reached via layer 2 must lie in $CDS$."
  },
  {
    term: "Cross-Layer Augmenting Path (CLAP)",
    meaning:
      "A layer-alternating chain of admissible segments that starts in $DD_1$, ends in $DD_2$, and yields a valid cross-layer improvement."
  },
  {
    term: "CLAP-stability",
    meaning:
      "A state with no feasible CLAP. Within the paper's feasible space, that means no further union contraction is possible."
  },
  {
    term: "CLAP-or-Optimal",
    meaning:
      "The theorem stating that a feasible configuration is globally optimal if and only if it is CLAP-stable."
  }
];

export const paperAbstract = `Many real-world complex systems can be modeled as multiplex networks, where each layer represents a distinct set of interactions among the same entities. This paper formulates the minimum-union problem for matching-based structural controllability in duplex networks: under fixed layer-wise minimum driver budgets, find the smallest union of driver nodes that makes both layers structurally controllable. The proposed Shortest Cross-Layer Augmenting Path Search (CLAP-S) algorithm reconfigures layer-wise maximum matchings to increase driver overlap while preserving each layer's budget. The paper proves that each successful CLAP update reduces the difference mass by exactly 2, that the union size drops by 1 accordingly, and that the absence of any CLAP certifies global optimality within the fixed-budget feasible space. Experiments on synthetic and real-world duplex networks show that CLAP-S matches exact ILP solutions and significantly outperforms random sampling baselines, while CLAP-G offers a faster greedy tradeoff.`;

export const bibtex = `@misc{zheng_optimized_duplex_control,
  title        = {Optimized Control of Duplex Networks},
  author       = {Haoyu Zheng and Xizhe Zhang},
  note         = {Manuscript under revision},
  howpublished = {Matching-based structural controllability for aligned-node duplex networks},
  url          = {https://github.com/njnklab/CLAPs-algorithm}
}`;
