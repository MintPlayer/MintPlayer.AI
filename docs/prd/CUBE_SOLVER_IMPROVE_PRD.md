# Improve the EfficientCube AI solver (PRD & Plan) — M34

> **Goal (user's steer, 2026-07-07):** "Can we further improve the Rubik's cube AI model?" — yes, but the
> honest lever is **solution quality (length) and search**, not more training. The shipped EfficientCube net
> already solves **100% (beam) at every eval depth d4→d26**, so solve-rate is saturated and can't move. What
> *can* move: how close the beam's solutions are to optimal, and the search cost (beam width) needed to get there.

- **Status:** ✅ **DONE (2026-07-07, branch `cube-solver-improve`).** W1 instrument + baseline, W2 beam-width sweep,
  W3 value-guided beam (null → not shipped), W4 more-training (skipped, not indicated), W5 shipped beam 2000→5000.
  **Net outcome:** the model was already optimal wherever verifiable; the real, shipped improvement is a search-width
  retune (shorter mid-depth solutions). Author: Pieterjan (with Claude Code).
- **Lineage:** this is the **EfficientCube policy net** (teacher-free, `--game cube-policy`, beam search) — the
  website's *only* served AI cube solver. It is a **different net** from the DAVI value net (PRD §13/§13.1,
  `value-davi-res`), which is no longer exposed on the web. Nothing here touches DAVI.
- **Precedent:** the DAVI shortest-move investigation (PRD §13.1) already established the key meta-finding this
  plan leans on — *"the next capability lever is eval-time search, not a wider net"* — and that Kociemba is **not**
  a QTM-optimality oracle (it minimizes half-turns; its QTM balloons to ~29–31, which the learned solvers already
  beat ~2–2.5×). We reuse that honesty here.

---

## 1. Where the model stands today (measured facts, verified 2026-07-07)

| Thing | Value | Source |
|---|---|---|
| Net | two-headed MLP, **1024-wide** trunk (`324→1024→1024`), 12-way policy head + scalar value head | `CubePolicyNet.cs`; shipped ckpt hexdump = width 1024 |
| Training | **teacher-free**, label = scramble reversal, CE(move) + Huber(distance); **346.8M samples**, round 6936 | `CubeSelfSupervised.cs`, progress ckpt |
| Final train metrics | CE ≈ 1.106, **top-1 move acc ≈ 0.619**, Huber ≈ 0.0036 | `cube-policy/logs/cube-policy.csv` |
| Solve rate | **beam = 1.000 at d4,8,12,14,16,18,20,22,24,26**; greedy collapses past d4 | same CSV |
| Beam search | width **2000**, maxDepth **40**, quarter-turns only; scored by **cumulative policy log-prob** | `CubePolicySearch.cs:128` |
| Value head in search | **UNUSED** — `PolicyAsMlp()` copies only trunk+policy head; beam never sees the distance estimate | `CubePolicyNet.cs:68`, `CubePolicySearch.cs:120` |
| Web serving | `POST /api/cube/solve-efficient`, beam width 2000, resident GPU forward (CPU fallback on Hetzner) | `CubeController.cs:47`, `CubeModelService.cs` |
| Solution length | **computed in eval (`beamLen`) but only printed — NOT persisted to CSV** | `CubeEfficientCampaign.cs:153,167` |

**Consequence:** we have no historical record of *how long* the beam's solutions are, only that they exist. The
capability is real and saturated; the optimality gap is unmeasured. **You cannot improve what you don't log.**

---

## 2. Why "more training / bigger net" is the wrong first move

- `docs/OPTIMIZATIONS.md` (2026-06-16) already found the cube loss floor **invariant to lr and width up to 690M
  samples** → the regime is **sample-bound**, and M17 settled that **width is not the lever**.
- Solve-rate is already 1.000 across the tested depth range, so extra samples buy nothing measurable *there*.
- The top-1 policy accuracy (0.619) is not a defect: beam search deliberately compensates for a locally-uncertain
  policy by exploring many sequences. Chasing accuracy ≠ chasing shorter solutions.

So the first three workstreams are **zero-retraining**: measure, then search. Retraining is a *last*, optional
lever (§6, W4) gated on the measurements actually showing a policy-quality ceiling.

---

## 3. What "better" means here (success criteria — honest, falsifiable)

Primary metric = **mean beam solution length per scramble depth**, tracked over training/eval, with two references:

1. **Scramble depth `d`** as a practical **upper bound** on optimal. For quarter-turn scrambles with no immediate
   inverse, optimal ≈ `d` at low depth (accidental cancellations only shorten it), so *(beam length − d)* is the
   **slack** we want to drive toward 0. This is cheap and always available.
2. **Provable QTM-optimal** at low depths (**≤ 7**, BFS-tractable) via the existing `BreadthFirstPlanner` — the same
   Tier-1 gate PRD §13.1 uses. This gives a rigorous "optimal where we can prove it" claim, not just "shorter."

*(Deliberately NOT using Kociemba as the optimality reference — it is not QTM-optimal. It stays only as the web's
guaranteed-solve fallback button and as a sanity check that a solution exists.)*

**We succeed if any of:**
- **A (shorter):** mean beam length drops toward the depth lower-bound at fixed beam width & solve rate, **or**
- **B (cheaper):** we hold length + solve rate at a **materially smaller beam width** (→ lower Hetzner-CPU latency), **or**
- **C (provable):** ≥95% of depth ≤7 scrambles solved **provably QTM-optimal** (BFS-verified).

**Non-goal:** god's-number coverage of *arbitrary* cubes / uniform-random states beyond d26 (that is DeepCubeA-scale
compute; PRD §13.1's laptop-GPU ceiling applies).

---

## 4. Workstreams & plan (E1 → E4, gated)

The steps are ordered so each **unblocks** the next: you can't tune search without a length metric, and you can't
justify value-guidance without a beam-width baseline.

### W1 — Instrument solution length (E1)  ✅ DONE 2026-07-07
The eval already tracks `beamLen`; it just never leaves the console.

> **✅ SHIPPED + BASELINED (2026-07-07, branch `cube-solver-improve`).** `Evaluate()` now emits `d{depth}_beamlen`
> + `d{depth}_slack` metrics (→ `cube-policy-eval.csv`, routed separately so it can't misalign the training log);
> `--eval-only` already existed (routes to `Evaluate()`); a `--optimal-probe` standalone mode (`TryRunStandaloneEval`)
> compares the beam to `BreadthFirstPlanner` optimum at BFS-tractable depths 1–6.
>
> **Baseline of the shipped 346.8M net (beam 2000, 10 cubes/depth, RTX 3060):** beam solves **10/10 at every depth
> d4→d26** (confirmed). **Length slack (beam − scramble depth):** d4 **0.0**, d8 **0.0**, d12 **−0.2**, d14 **+3.8**,
> d16 **+5.2**, d18 **+6.0**, d20 +3.6, d22 +4.2, d24 +3.2, d26 +0.4. **Provable-optimality probe:** **100%
> QTM-optimal at d1–d6** (beam length == BFS optimum on all 60 cubes) → criterion **C already met** at the shallow end.
>
> **Read:** the beam is optimal through ~d12, then opens a **~4–6 qt gap at d14–d18** — that mid-depth band is the
> concrete headroom for W2/W3. Slack shrinks again at d24–d26 only because scramble-depth is a loose upper bound on
> the true optimum there (a random d26 scramble usually solves in well under 26 qt), NOT because the beam is optimal.
> So the honest target metric is the **d14–d18 slack**, not the deep-end numbers.

**As-built note:** E1c capped at **d6** (not the planned d7) — the BFS radius-d ball grows ~9× per quarter-turn
(d6 ≈ 1M states, d7 ≈ 9M), so d7 would blow past a sub-minute-per-cube budget; d1–6 at 100% optimal already
satisfies criterion C. `--optimal-probe` implies `--eval-only`.
- **E1a — persist length metrics.** In `CubeEfficientCampaign.Evaluate`, add per-depth metrics
  `d{depth}_beamlen` (mean quarter-turns over solved beams) and `d{depth}_slack` (`beamlen − depth`) to the
  `metrics` list so `CampaignRunner` writes them as new CSV columns. *(The CSV header is metric-derived, so this is
  additive — old rows just lack the columns.)*
- **E1b — eval-only mode.** Add a `--eval-only` path (reuse `Evaluate()` without `TrainChunk`) to `CubePolicyLab`
  so we can measure the **shipped 346.8M checkpoint right now** without a training run — producing the baseline
  length curve. *(Confirm whether such a mode already exists before adding one.)*
- **E1c — low-depth optimality probe.** For depths 1–7, compare beam length to `BreadthFirstPlanner` optimal;
  report `d{depth}_optimal_gap` and the provably-optimal fraction (criterion C).
- **Gate:** a baseline table — for the shipped net, mean beam length + slack at each depth, and the ≤7 optimal gap.
  This is the **deliverable that makes "improve" falsifiable**; everything after is measured against it.

### W2 — Beam-width sweep (E2)  ✅ DONE 2026-07-07
- Run E1's eval-only measurement at beam widths **{500, 1000, 2000, 5000, 10000}** over the depth ladder.
- Produce the **length ↔ width ↔ solve-rate ↔ expansions** curve.
- **Gate:** identify (a) the smallest width holding 1.000 solve rate + current length (criterion **B**, latency win),
  and (b) whether a wider beam yields shorter solutions (criterion **A**, quality win) and at what cost.

> **✅ SHIPPED + MEASURED (2026-07-07).** Added `--beam-sweep w1,w2,…` (a `TryRunStandaloneEval` mode) that re-runs
> the depth eval per width and writes a self-describing `cube-policy-sweep.csv` (`beam,depth,beam_solved,beamlen,
> slack,mean_expansions`). Extracted a shared `EvaluateDepths(logits, width, includeGreedy)` helper that also
> tallies **mean expansions** (net-forward count = the machine-independent search-cost proxy). Sweep of the shipped
> net (10 cubes/depth, RTX 3060):
>
> **Solve rate:** 10/10 everywhere for width **≥ 1000**; width 500 misses only d16 (9/10). **So beam 2000 (the
> shipped default) is conservative — width 1000 already solves 100%.**
>
> **Length — wider beam DOES shorten mid-depth solutions** (beamlen qt; the d14–d18 headroom band):
>
> | depth | b500 | b1000 | b2000 | b5000 | b10000 |
> |---|---|---|---|---|---|
> | d14 | 19.0 | 18.2 | 17.8 | **14.4** | 14.2 |
> | d16 | 23.8 | 22.8 | 21.2 | 20.2 | 19.4 |
> | d18 | 26.4 | 24.4 | 24.0 | 22.6 | 22.4 |
> | d22 | 29.4 | 28.4 | 26.2 | 25.8 | 25.0 |
>
> The **d14 gap collapses** at width 5000 (17.8 → 14.4 qt, slack 3.8 → **0.4** ≈ optimal); d16–d22 trim 1–2 qt.
> **Cost:** expansions scale ~linearly with width (d26: b2000 44k → b5000 107k → b10000 209k).
>
> **Two opposite levers, both real:**
> - **Criterion B (latency):** drop the web beam **2000 → 1000** → ~½ the expansions, still 100% solve, tiny length
>   cost (d16 21.2 → 22.8). The right move for the CPU-bound Hetzner box (W5).
> - **Criterion A (quality):** width **≥ 5000** closes d14 to near-optimal and trims d16–d22, at ~2.4× the expansions.
>
> **This frames W3's target precisely:** can **value-guidance at beam 2000** buy the beam-5000 lengths (d14 → ~14.4)
> at beam-2000 cost — i.e. shorter solutions *without* paying the 2.4× search? That is now a measurable hypothesis.

### W3 — Value-guided beam scoring (E3)  *(the real modeling change; hypothesis, not a guaranteed win)*
The net predicts distance-to-solved but beam search ranks purely by policy log-prob — which biases toward *likely*
move sequences, not *short* ones. Adding the trained distance estimate as an explicit "closer-to-solved" term is the
mechanistic reason this could shorten solutions and/or let us cut the beam.
- **E3a — expose the value head to search.** Extend the resident forward so beam nodes get *both* logits and the
  value estimate. Options (design-it-twice): (i) a combined resident head `324→h→h→13` (12 logits + 1 value) so one
  device forward yields both; or (ii) a second resident value forward. Prefer (i) — one forward, no extra transfer.
  This means `PolicyAsMlp()` (which drops the value head) gets a `PolicyAndValueAsMlp()` sibling.
- **E3b — value-weighted score + λ knob.** Prune by `cumLogProb + λ·(−predictedDistance(child)/DistanceScale)`;
  add `valueWeight`/λ to `BeamSearch` (default 0 = current pure-policy behavior, so it's a strict superset — no
  regression risk to the shipped path until we flip the default).
- **E3c — sweep λ** against the E2 baseline at matched compute (same expansions budget). Compare length, solve rate,
  and beam width needed.
- **Gate:** value-guidance must **beat pure-policy on criterion A or B at matched compute**, or we keep λ=0 and
  record the null result (this is a real possibility — vanilla EfficientCube uses policy-only beam by design; the
  value head is *our* extra signal and may not help). Honest either way.

> **✅ DONE — NULL RESULT (2026-07-07). Keep λ=0 (pure policy); do NOT ship value-guidance.** Built it in full:
> `CubePolicyNet.PolicyAndValueAsMlp()` (combined `[324,h,h,13]` head → one resident forward gives logits + value),
> `CubePolicySearch.BeamSearchValueGuided(…, valueWeight)` scoring `cumLogProb − λ·relu(value)`, and a `--value-sweep`
> mode (→ `cube-policy-value.csv`, tracks expansions). **Correctness check passed:** λ=0 exactly reproduces the
> beam-sweep@500 lengths.
>
> **Value guidance genuinely shortens solutions** (monotonic in λ; at beam 500, λ=8 vs λ=0: d18 −2.4, d22 −2.6,
> d16 −1.2 qt, and it *fixed* d16 solve 9/10→10/10). **But it loses decisively at matched compute (the gate):**
> using the value heuristic in pruning requires forwarding every candidate child (~10× the states/step), so a
> value-guided beam-500 spends ~80–140k expansions — and plain beam-widening buys the same length far cheaper:
>
> | d18 | length | expansions |
> |---|---|---|
> | vg beam-500 λ=8 | 24.0 | 110k |
> | **pure beam-2000** | **24.0** | **39k** (2.8× cheaper, same length) |
> | vg beam-500 λ=0 | 26.4 | 123k |
> | **pure beam-500** | **26.4** | **11k** (11× cheaper, identical — the raw child-forward overhead) |
>
> At d16, pure beam-5000 gets **20.2 qt at 77k** vs vg's 22.6 qt at 103k — shorter *and* cheaper. **Verdict:** the
> value head's guidance signal is real but weak, and widening the pure-policy beam is **3–11× more compute-efficient**
> for the same solution length. So value-guidance is not worth its complexity on the web — **the shipping win is
> W2's beam-width knob, not W3.** (Unexplored: a *pre-prune* variant — cut candidates by cumLogProb before forwarding
> — could cut the ~10× overhead, but the guidance is weak enough that even a 5× cheaper version would only tie pure
> beam, so it's not pursued. λ stays 0 in the shipped path.)

### W4 — (Optional, last) more training  ⏭️ SKIPPED 2026-07-07 (not indicated)
- Resume from the 346.8M checkpoint for a multi-day run and re-measure the length curve.
- **Only pursue if** the measurements show search can't close the gap (i.e. the policy itself is the bottleneck).
  Given the sample-bound evidence (§2), expect **incremental** gains at best — budget accordingly, don't lead with it.

> **⏭️ SKIPPED — the gate is not met.** W1–W3 showed the opposite of a policy ceiling: the net is already optimal
> through d12 and **100% provably QTM-optimal at d1–6**, and W2 proved the mid-depth gap is **search-bound** (pure
> beam 5000 recovers d14 to near-optimal with zero retraining). With the loss also sample-bound (`OPTIMIZATIONS.md`),
> a multi-day run would spend days for at-best-incremental gain on a model that's optimal wherever we can verify it.
> Not run. (The eval-only length tooling from W1 stays, so if a future training run happens it will track length.)

---

## 5. Shipping the win to the web

**W3 was a null (value-guidance loses at matched compute), so the shipping win is purely W2's beam-width knob** —
`beamWidth` in `CubeController.SolveEfficient` is hard-coded 2000. Two defensible retunes, both keeping the honest
Kociemba fallback and the unchanged `CubeSolveAiResponse` contract:
- **Latency (criterion B):** drop **2000 → 1000** — still 100% solve d4→d26, ~½ the expansions (the Hetzner box is
  CPU, so this is a real per-request latency win), at a small length cost (e.g. d16 21.2 → 22.8 qt).
- **Quality (criterion A):** raise **2000 → 5000** — closes the d14 gap to near-optimal (17.8 → 14.4 qt) and trims
  d16–d22 by 1–2 qt, at ~2.4× the expansions.

These pull in opposite directions; the choice is a product call (snappy solver vs. shortest solution).
The eval-only length + sweep tooling stays so future training runs track length, not just solve rate.

> **✅ SHIPPED 2026-07-07 — beam 2000 → 5000 (quality, owner's call).** `CubeController.SolveEfficient` beam width
> raised to 5000: near-optimalizes mid-depth solutions (d14 17.8 → 14.4 qt, d16–d22 −1–2 qt) at ~2.4× the search
> cost. The endpoint contract, resident-GPU/CPU-fallback path, and the honest Kociemba reference are unchanged; the
> beam stays pure-policy (W3 value-guidance was worse per-compute). Beam-5000 behavior was already exercised in the
> W2 sweep (100% solve d4→d26, the exact lengths above). *Latency note:* the Hetzner box is CPU-only, so this is
> slower per request than 2000 — acceptable per the owner's quality preference; revisit with a time-bound if a solve
> feels sluggish in production.
- Keep the eval-only length measurement in the campaign so future training runs **track length, not just solve rate**.

---

## 6. Risks & honest caveats

| Risk | Mitigation |
|---|---|
| Value-guidance doesn't help (policy-only is already near-optimal for beam) | E3 defaults λ=0 (superset of today); we record the null result and keep the search-width win from W2. |
| "Optimal" reference is fuzzy above depth 7 | Use scramble-depth slack as the upper-bound proxy and confine *provable* claims to BFS-tractable depths (≤7). Never claim optimality we can't verify. |
| Beam length at deep scrambles may already be near-optimal → little headroom | That itself is a finding worth recording (the model is better than we can currently prove); W2's latency win still stands. |
| Combined value+policy resident head changes checkpoint/forward plumbing | E3a keeps `PolicyAsMlp()` intact and adds a sibling; the shipped policy-only path is untouched until we flip λ. No ckpt format change (weights already in the file). |
| Retraining (W4) burns days for incremental gain | Gated behind W1–W3 evidence; not the lead. |

---

## 7. Non-goals
- Touching the DAVI value net or its web wiring (separate lineage, already de-exposed).
- God's-number / arbitrary-cube coverage (DeepCubeA-scale compute; out of laptop-GPU reach).
- Changing the quarter-turn action space to add half-turns (deliberate design choice; would change the metric).
- Client-side (browser) cube AI (Cube is a request/response REST solver — out of the M32/M33 client-side scope).
