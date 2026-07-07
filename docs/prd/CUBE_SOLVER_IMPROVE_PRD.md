# Improve the EfficientCube AI solver (PRD & Plan) — M34

> **Goal (user's steer, 2026-07-07):** "Can we further improve the Rubik's cube AI model?" — yes, but the
> honest lever is **solution quality (length) and search**, not more training. The shipped EfficientCube net
> already solves **100% (beam) at every eval depth d4→d26**, so solve-rate is saturated and can't move. What
> *can* move: how close the beam's solutions are to optimal, and the search cost (beam width) needed to get there.

- **Status:** 🔜 **PLANNED (2026-07-07).** Not started. Author: Pieterjan (with Claude Code).
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

### W1 — Instrument solution length (E1)  *(tiny, unblocks everything)*
The eval already tracks `beamLen`; it just never leaves the console.
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

### W2 — Beam-width sweep (E2)  *(free; pure eval)*
- Run E1's eval-only measurement at beam widths **{500, 1000, 2000, 5000, 10000}** over the depth ladder.
- Produce the **length ↔ width ↔ solve-rate ↔ expansions** curve.
- **Gate:** identify (a) the smallest width holding 1.000 solve rate + current length (criterion **B**, latency win),
  and (b) whether a wider beam yields shorter solutions (criterion **A**, quality win) and at what cost.

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

### W4 — (Optional, last) more training  *(only if W1–W3 show a policy-quality ceiling)*
- Resume from the 346.8M checkpoint for a multi-day run and re-measure the length curve.
- **Only pursue if** the measurements show search can't close the gap (i.e. the policy itself is the bottleneck).
  Given the sample-bound evidence (§2), expect **incremental** gains at best — budget accordingly, don't lead with it.

---

## 5. Shipping the win to the web

Whatever W2/W3 conclude:
- If a smaller beam width holds quality (criterion B) → lower `beamWidth` in `CubeController.SolveEfficient`
  (currently hard-coded 2000) → faster Hetzner-CPU solves at no quality cost.
- If value-guidance wins (criterion A/B) → wire the combined resident forward + chosen λ into `CubeModelService`
  (the resident `DeviceMlp` build) and `CubeController`. The endpoint contract (`CubeSolveAiResponse`) is unchanged;
  the honest Kociemba fallback stays.
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
