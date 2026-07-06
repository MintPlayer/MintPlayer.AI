# FruitCake AI — Breaking the Plateau (PRD & Plan)

> The shipped FruitCake DQN has **plateaued**: best play reaches a **pineapple** (tier ~9–10),
> almost never a **watermelon** (top tier 11). This PRD diagnoses *why* and lays out the levers to
> break it, prioritized by a four-angle investigation that **all converged on the same conclusion**.

- **Status:** Draft v1.0 · 2026-06-27 (investigation complete; not started)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:** the FruitCake AI (`FRUITCAKE_AI_PRD.md`), the DQN stack (`DqnTrainer`/`DuelingQNet`),
  the FruitCake A/B harness (`tools/…Lab/FruitCakeAb.cs`), and the 2048 expectimax precedent
  (`Game2048/Expectimax2048.cs`). Companion to [PLAN.md](PLAN.md) (M28 NoisyNets verdict, M29 here).

---

## 1. Diagnosis — why it plateaus (triangulated, high-confidence)

A four-agent investigation (observation, reward/algorithm, serving-side search, external SOTA) **independently
reached the same diagnosis**: the plateau is **reward-design + perception bound, NOT capacity or exploration
bound**, and the game fundamentally **rewards planning with a forward model far more than a reactive net**.

- **Reward is myopic.** Reward = `mergePoints/10 − 1 on death`. A watermelon's final merge is worth ~5.5
  (shaped), while the steady drip of cheap low-tier merges dominates the return — so *maximizing cumulative
  score is maximized by myopic play*. The agent correctly converged to the pineapple optimum. Nothing rewards
  tier-progression or building toward big merges. (`FruitCakeEnv.Step`; `FruitCatalog` MergePoints are ~linear.)
- **Perception is impoverished.** The 41-dim observation is a flattened **1-D skyline** (per-column surface
  height + the tier of *only the topmost* fruit) + current/next + 3 globals. It has **no mergeable-adjacency
  signal** (the entire game is "put equal tiers next to each other"), **no buried structure**, **no per-column
  danger margin** (the danger line never even enters the obs), lossy column-binning, and only 1-ahead.
- **NOT capacity-bound** (41→[256,256]→14 is ample) and **NOT exploration-bound** — continued ε-greedy *and*
  NoisyNets both failed (PLAN M28). NoisyNets failing is the tell: it's not a *discovery* problem (the agent
  finds merges), it's a *preference* problem (the objective doesn't value watermelon-building) compounded by a
  *credit-assignment* problem (γ=0.99 over a ~1000-drop horizon makes the rare watermelon payoff near-invisible).
- **The literature confirms it exactly.** A 2025 Leiden MSc thesis benchmarked DQN on a Suika clone: **DQN
  barely beat random and mode-collapsed** (dropped everything in one spot), while a **shallow Monte-Carlo
  forward-model planner beat the human** (≈6,600–7,300 vs human 5,076 vs DQN ≈2,900). Its best *learned* model
  used **hand-crafted relational features**; the reward that worked was **dense afterstate shaping**
  (+merge, +no-height-increase, +next-tier-neighbor). (Sources in §8.)

**Two independent angles fingered the same missing concept — adjacency of equal tiers:** perception can't *see*
it, and reward doesn't *value* it. That is the core of the fix.

---

## 2. Goals & Non-Goals

### Goals
- **Break the pineapple plateau** — reach watermelon (tier 11) at a materially higher rate, judged by a robust
  multi-seed A/B on the **max-tier distribution + mean score** (never a single 10-episode eval — those are
  seed-luck, PLAN M28).
- Treat this as an **SDK opportunity**, not a one-off: the forward-model planner, n-step returns, and
  relational-feature pattern are reusable beyond FruitCake.
- Keep **serving deterministic** and every **shipped checkpoint loadable** (the project's invariants).

### Non-Goals (v1)
- A CNN / image observation (no conv layer in the library; the literature shows hand-crafted relational
  features beat pixel DQN here anyway).
- AlphaZero-scale self-play (the planner-distillation endgame is a *stretch*, F6).
- Guaranteeing "always watermelon" — even the best published Suika agents reach strong-human, not perfection
  (§7 honest ceiling).

---

## 3. The three levers (priority by impact ÷ effort)

| # | Lever | Retrain? | Why it's ranked here |
|---|---|---|---|
| **A** | **Serving-side forward-model search** (depth-2) | **No** | The literature's #1 lever (the only thing that beat humans), the 2048-expectimax pattern we already shipped, and *cleaner* here — both pieces known + zero physics randomness ⇒ deterministic maximization, no chance node. Amplifies the **current** net immediately. |
| **B** | **Richer relational inputs** | Yes | The structural fix for the *learned* policy — give the net mergeable-adjacency, per-column danger margin, buried structure, next-next. (Your steer; the Snake M27 precedent: better inputs > more training.) |
| **C** | **Dense reward shaping + n-step** | Yes | Make the objective value tier-progression & chain-setup (afterstate shaping), and propagate the sparse high-tier reward (n-step returns, γ↑). Fixes the *preference* + *credit-assignment* halves of the diagnosis. |

A is independent and ships first (no training). B+C are the "training session" — one retrain combining richer
inputs, shaped reward, and n-step. The endgame (F6) unifies them: a net good enough to be the planner's leaf
evaluator, optionally distilling the planner back into the policy (the 2048/AlphaZero recipe).

---

## 4. Design

### 4.A Serving-side forward-model search (`FruitCakeSearch`) — no retrain
The physics is **fully deterministic** (no RNG in `FruitCakeWorld`) and the current+next fruit are both known,
so this is deterministic maximization (no expectimax chance node — *simpler* than 2048):
- For each of 14 columns: `world.Clone(enableRotation:false)` → `SpawnFruit(current, …)` → `SettleAfterDrop(…)`
  (the exact loop `FruitCakeHeuristic` already runs live) → if it loses (`AnyEjected`/`AnyRestingAboveDangerLine`)
  prune; else recurse one ply on the **next** fruit (14 columns).
- **Leaf value** = accumulated real merge points along the path **+ the trained net's max-Q** of the leaf state
  (`max(GreedyQAgent.QValues(BuildObservation(leaf, …)))`) — mirroring 2048's `reward + V(afterstate)`, reusing
  the net unchanged. Heuristic-score fallback when no checkpoint is loaded.
- **Cost:** depth-1 = 14 settles (the heuristic already does this live); depth-2 = ≤210 settles, pruned to
  ~30–50 via lose-pruning + top-K first-ply expansion — fits the controller's 250 ms between-drops window.
- **Integration:** replace the 1-ply `agent.Act` in `FruitCakeController` (the single call site) with
  `FruitCakeSearch.ChooseColumn(world, current, next, agent?)`. Clone rotation-off (proven to transfer); the
  live world stays rotation-on. **Serving-only first** (zero training-pipeline risk), exactly like `Expectimax2048`.

### 4.B Richer relational observation (retrain)
Augment `FruitCakeEnv.BuildObservation` (the serving path uses the same static method, so it stays in sync).
Prioritized:
- **B1 — Mergeable-adjacency (highest impact, ~low dims):** per column, flags for "top tier == current/next
  droppable" and "top tier == neighbor's top tier" (a one-drop merge is available here). Injects the single
  most important relationship, currently entirely absent.
- **B2 — Per-column danger margin (~14 dims, trivial):** `(topY[c] − DangerLineY)/H` per column; the danger line
  is a constant never currently exposed. Localized survival awareness for the long horizon.
- **B3 — Next-next fruit (+5 dims):** maintain a length-2 preview queue (+ keep the live frame DTO in sync) for
  the 2-ahead planning Suika rewards.
- **B4 — Tier-occupancy grid (~112–140 dims, the structural fix):** a 14-col × ~8–10-row grid encoding
  occupancy/dominant-tier — exposes *buried* structure and 2-D adjacency the skyline destroys (the direct Snake
  9×9-patch analog). MLP consumes it flattened (no CNN). Do this if B1–B3 underdeliver.

  > **Implemented + measured 2026-07-06 (branch `fruitcake-tier-grid`) — NULL result, NOT shipped.** Appended a
  > **14×10 tier-occupancy grid** (140 cells, each = max tier of any fruit whose bounding box overlaps it ÷11;
  > 0 = empty) to `buildObservation` in the single-source `fruitcake_solver.pg`, inserted *before* the big-fruit
  > block so big-fruit stays the last 6 dims → **obs 89 → 229**. Trained a FRESH net at width 229 (same recipe as
  > F5/G3: `--shape --nstep 3 --gamma 0.997`) to **321k drops** (greedy plateaued ~800–944, *at or slightly below*
  > the 963/1019 F5/G3 baselines). Judged on the deployed metric — depth-3 net+search (`--search-eval --depth 3
  > --topk 5 --topk2 2`): a 50-game read = **2519 score / 52% watermelon / meanTier 10.52**, i.e. **dead on the
  > ~2505 / ~50% bar** — a tie within seed noise (200-game confirmation launched). The grid gives the reactive net
  > no edge the deployed search system can exploit. **Consistent with the saturated-net prior** (F2 relational
  > inputs, big-fruit positions M30/G4, NoisyNets M28, planner distillation F6, and reverse-curriculum all
  > null/negative): on Suika the strength is the **search**, not the net — richer perception doesn't move the
  > ceiling. Branch kept as a validated-capability artifact (like NoisyNets/F6); **must not merge to master** (the
  > 229-dim obs would desync the live 89-dim browser net). Write-up in `docs/OPTIMIZATIONS.md`.
- **B5 — Per-tier board counts (+11 dims):** cheap "how close to a high-tier pair" signal.
- *Sequence:* ship **B1+B2(+B3)** first (a handful of dims, fills the glaring blind spots, one retrain); escalate
  to **B4** if needed. Caveat: column-binning is lossy for large fruit — B4's grid is the true representational fix.

> **Implemented (2026-06-27, F2 first cut): 41 → 83 dims.** `BuildObservation` now appends three per-column
> (×14) relational blocks to the skyline: **danger margin** `clamp((topY−DangerLineY)/(H−DangerLineY),0,1)`
> (1=floor/safe, 0=at/above the line) = B2; **merge-with-current** `topTier[c]==current` = B1 immediate-merge
> map; **adjacent-equal-pair** `topTier[c]==topTier[c±1] (nonzero)` = B1 mergeable-adjacency. Layout:
> 14 surface-h + 14 top-tier + 14 danger + 14 merge-cur + 14 adj-pair + 5 current + 5 next + 3 globals = **83**.
> **B3 (next-next) deferred** — it needs new env preview state + a serving-DTO/`BuildObservation`-signature change,
> whereas B1+B2 are a self-contained pure-function edit. Add B3/B4 only if the B1+B2 retrain underdelivers.
> The old 41-dim checkpoint no longer loads (width guard refuses it) — F5 trains a fresh net at width 83.

### 4.C Dense reward shaping + n-step (retrain)
All shaping is an edit to `FruitCakeEnv.Step` (precedent: `RushHourEnv` already does potential-based shaping):
- **C1 — Max-tier-reached bonus:** a one-time, geometrically-scaled bonus the first time each new highest tier
  (6→11) appears in an episode. One-time ⇒ unfarmable; directly rewards the goal the score objective hides.
- **C2 — Potential-based shaping toward stackable adjacency:** Φ(board) rewarding same-tier fruit adjacent/
  stackable; add `γ·Φ(s′) − Φ(s)`. **Policy-invariant (Ng et al. 1999) ⇒ zero reward-hacking risk** — the
  principled fix, and the perception-side mirror of B1. (The thesis's working reward = +merge, +no-height-
  increase, +next-tier-neighbor — the same idea.)
- **C3 — Small danger/height penalty** as the pile nears the line (keep small; prefer the potential form).
- **C4 — n-step returns:** add an `NStep` knob to `DqnOptions` + n-step accumulation in `DqnTrainer`. Propagates
  the sparse watermelon reward backward fast — the best *algorithmic* lever; **new library capability**, reusable.
  > *Implemented as `NStepAccumulator`* (folds transitions before they hit the buffer; the **buffer format is
  > unchanged** — every non-terminal transition bootstraps with a single global `γ^n`). Truncation drops the
  > ≤n−1 partial-window tails (a per-transition discount would otherwise be needed; negligible). The accumulator's
  > window is persisted (state v4) so resume is bitwise-identical.
- **C5 — γ → ~0.997** (trivial, `--gamma`); lengthens the effective horizon. PER is a later option if needed.

---

## 5. Milestone plan

- **F0 — Robust measurement.** Use `FruitCakeAb` to quantify the baseline **max-tier distribution** over ≥200
  paired games (confirm "rarely watermelon" with a number, not a vibe); ensure it reports the tier histogram.
  *Gate:* a reproducible baseline metric the rest is judged against. (Eval is seed-noisy — this is the yardstick.)
- **F1 — Serving-side search (no retrain).** ✅ *Shipped 2026-06-27 (branch `fruitcake-forward-search`).*
  `FruitCakeSearch` (depth-2, lose-prune + top-K=5, leaf = realized merge points + injected board value =
  the net's max-Q marginalized over the unknown upcoming fruit; heuristic fallback) wired into
  `FruitCakeController` (the single policy call site). **Robust 200-game paired eval (`--search-eval`, same net
  & seeds, search vs greedy):** greedy 963.5 / meanTier 8.75 / **watermelon 0/200** → search **2278 / meanTier
  10.25 / watermelon 60/200 (30%)** at the tuned **topK-10**, **wins 200/200**, **96% of games reach tier 10**.
  ~11 ms/drop (well inside the 250 ms between-drops budget). **The watermelon breakthrough — pure amplification
  of the shipped net, no retrain.** Reproduces the literature (forward-model search beats the reactive net at Suika).
  - *Tuning (200-game sweep):* search **width** is the lever — topK 5→10 lifts watermelon **17% → 30%**; topK-14
    (exhaustive depth-2) ties topK-10 (29.5%) at higher cost, so **topK-10 is the depth-2 sweet spot**. Hand-crafted
    "tier-seeking" leaves (tierpot/blend) **lost** to the net leaf (they hoard big fruit and lose) — the net's
    learned board sense is the better leaf despite being pineapple-capped. Leaf is the net's max-Q marginalized
    over the unknown upcoming fruit; heuristic fallback (−pile height) when no net.
- **F2 — Richer inputs (retrain).** ✅ *Observation built (2026-06-27):* B1+B2 in `BuildObservation`,
  `ObservationSize` 41→83, builds + all 8 FruitCake tests green. B3 deferred (see §4.B note). *Remaining gate:*
  trains end-to-end; A/B vs baseline (multi-seed) ≥ baseline on max-tier — folded into F5.
- **F3 — Reward shaping (retrain, with F2).** ✅ *Built (2026-06-27):* C1 one-time geometric tier-reached bonus
  + C2/C3 potential-based shaping (`γ·Φ(s′)−Φ(s)`, Φ = tier-weighted same-tier near-pairs − normalized pile
  height) in `FruitCakeEnv`, **opt-in (`ShapeRewards`, default off)** with configurable weights; the eval env
  stays unshaped so keep-best/A/B judge real merge points. Φ(terminal)≡0. *Remaining gate:* A/B improves (F5).
- **F4 — n-step + γ.** ✅ *Built (2026-06-27):* `DqnOptions.NStep` + `NStepAccumulator` (folds transitions
  before the buffer; TD target bootstraps with `γ^n`); buffer format unchanged. C5 = `--gamma 0.997`. *Gate met:*
  5 contract tests green (n=1 single-step parity; discounted-sum target; terminal flush; truncation-drop;
  Save/Load round-trip). **Bitwise-resume preserved** — the in-flight window is persisted (`DqnTrainingState` v4;
  n=1 window is always empty at step boundaries ⇒ identical to v3).
- **F5 — Train & ship (conditional).** ✅ *Shipped 2026-06-27.* Trained from scratch at width 83 with
  `--shape --nstep 3 --gamma 0.997` (stopped ~90 min in; run is resumable from `data/fruitcake-bundle`).
  Robust 200-game greedy measurement: **mean 963.5 ±260 SD, median 961, meanTier 8.75, maxTier histogram
  t10:15/t9:127/t8:53/t7:4/t5:1** vs recorded baseline ~707 / meanTier ~8.5 / watermelon ~never → **+36%
  score, more frequent tier-10 play** ⇒ shipped `models/fruitcake.dqn.ckpt` (LFS) on PR #17. **Watermelon (11)
  still 0/200** — the score win landed, the tier-ceiling breakthrough did not; that's the F1/B4/F6 work.
  *Measurement note:* old (41-dim) vs new (83-dim) can't be paired in one build, so the gate was the new net's
  absolute distribution vs the recorded baseline (same game/scoring, only the net's input changed).
- **F6 — Stretch: planner-guided training / afterstate value.** Use the F1 planner as an expert (generate
  demonstrations / stronger behavior policy) or learn `V(simulated-afterstate)` and act by search over it — the
  2048/AlphaZero recipe; gated on F1–F5.
  - **ATTEMPTED 2026-06-28 — NEGATIVE RESULT, not shipped.** Distilled the depth-3 search into a two-headed
    policy/value net (CE on the planner's column + Huber on return-to-go). 30-game paired A/B: dqn+search 43%
    watermelon vs distilled-policy-alone 0% vs distilled-policy+search 3%. Column-imitation accuracy plateaued
    at ~23% — a **reactive net cannot reproduce a planner's forward-simulation lookahead** (the same ceiling
    that capped the DQN). Confirms the §7 thesis: the lever is the SEARCH, not the net. Full write-up +
    principle in [`docs/OPTIMIZATIONS.md`](../OPTIMIZATIONS.md) ("Investigated, not pursued — FruitCake planner
    distillation"). Code recoverable from branch `fruitcake-planner-distillation` (commit `7201dd5`).

**Recommended first session (what to do after compacting):** F0 → F1 (banks the no-retrain win + validates the
forward model) → then the training session F2+F3+F4 together → F5 A/B.

---

## 6. Measurement
Judge on the **max-tier distribution** (fraction of games reaching tier 9/10/**11**) **and** mean score, over
**≥200 paired games across multiple seeds** via `FruitCakeAb` — never a single 10-episode eval (PLAN M28: those
bounced 750–971 on the *same* net; the robust estimate was ~700–714). Serving stays deterministic; keep-best
gating protects the shipped model.

## 7. Honest ceiling
Even the best published Suika agents (the thesis's Monte-Carlo planner; TaiYo) reach **strong-human, not
"always watermelon."** The game is long-horizon with a sparse top-tier reward and chaotic physics. Realistic
target: **watermelon goes from ~never to a meaningful fraction**, and mean score rises clearly — not perfection.
The forward-model planner (F1/F6) is the most likely path to the high end.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Reward shaping → reward-hacking | Prefer **potential-based** shaping (policy-invariant) + one-time tier bonus (unfarmable); keep penalties small. |
| Depth-2 search too slow for serving | Lose-pruning + top-K first ply (~30–50 settles); depth-1 fallback; it's off the render path in the 250 ms window. |
| Obs change breaks back-compat | Input-width guard already exists (`FruitCakeModelService`); a retrained net ships its own width; old net still loads. |
| n-step bug corrupts targets | Unit/contract test: n=1 parity with current; explicit n-step target check; bitwise-resume test. |
| Eval seed-noise misleads (M28 trap) | Gate on the ≥200-game paired A/B max-tier distribution, never single evals. |
| No real gain | F1 is no-retrain (low cost); F2–F4 gated by A/B; the capabilities (search, n-step, relational features) are reusable regardless. |

## 9. Sources
- **Poelsma, J. — *Creating an AI that plays Suika game*** (MSc thesis, LIACS, Leiden Univ., 2025) — the
  decisive benchmark: DQN ≈ random + mode-collapse; **Monte-Carlo forward-model planner beat the human**; best
  learned model = hand-crafted relational features; working reward = +merge/+no-height-increase/+next-tier-neighbor.
- Guei, H. — *On RL for the Game of 2048* (arXiv:2212.11087) & Jaśkowski (arXiv:1604.05085) — n-tuple afterstate
  learning + **expectimax** (search ≈ doubles a good value function).
- Ng et al. 1999 — potential-based reward shaping is **policy-invariant**.
- *Tiered Reward* (arXiv:2212.03733) — tier-progression rewards learn faster for reach-the-tier goals.
- Internal: PLAN M28 (NoisyNets matched, not beaten; eval seed-noise), M27 (Snake: inputs > training),
  `Game2048/Expectimax2048.cs` (serving-side search precedent), `FruitCakeAb.cs` (the A/B harness).
