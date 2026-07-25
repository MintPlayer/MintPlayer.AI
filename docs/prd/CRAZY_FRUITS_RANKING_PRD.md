# Crazy Fruits — missed-opportunity ranking (net prefers a 3-match over an available 4-match) — PRD

**Status:** 🔜 planned 2026-07-25 (4-agent investigation: reward structure, inference path, training loss, technique survey)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M51 · extends [CRAZY_FRUITS_SPECIALS_PRD.md](CRAZY_FRUITS_SPECIALS_PRD.md) (M50, shipped) · branch: continue the arc branch `m49-crazy-fruits` if PR #38 is still open, else `m51-crazy-fruits-ranking`

## 1. Problem

Owner observation on the shipped `cf5train` net: on boards where the same fruit offers **both** a 3-in-a-row
and a 4-in-a-row swap, the AI takes the 3-match — forfeiting the striped special the 4-match would create.
The net was trained short ("just enough to publish") and will train longer regardless; the owner's question
is whether we can additionally **punish the model harder when it misses important opportunities**.

Follow-up question (owner, same day): should combining two specials earn a higher reward, or does that
already happen through the score? **Resolved by investigation — already correct, no change (§2.7).**

## 2. Investigation findings (4 agents, 2026-07-25)

1. **The loss cannot punish a ranking error — root cause.** `DqnTrainer.TrainStep` regresses ONLY the chosen
   action's Q (`Gather(batch.Actions)`, `DqnTrainer.cs:291`; Huber δ=1, `:292`). At γ=0 the target is the raw
   scaled reward (`:283-286`). Ranking a 3-match above a co-available 4-match costs **zero loss** — the
   4-match's Q is simply never updated on that transition. "Punish harder for missed opportunities" is
   therefore not a reward tweak; it's a **loss-structure** change.
2. **The reward already distinguishes 3 vs 4 — not a reward-definition gap.** Clean single match at step 0
   (`crazyfruits_solver.pg:658,68-70`; RewardScale=100, shaping +40/+60/+100 on the train env only,
   `CrazyFruitsEnv.cs:97-99,148-152`):

   | Match | game pts | train reward | eval reward |
   |---|---|---|---|
   | 3-run | 30 | 0.30 | 0.30 |
   | 4-run | 60 (+20 line bonus) | **1.00** (+40 shaping) | 0.60 |
   | 5-run | 100 (+50 bonus) | **2.00** (+100 shaping) | 1.00 |

   No clipping anywhere; ÷100 keeps both targets in Huber's quadratic zone — normalization is ruled out.
3. **But the signal is small against refill-cascade variance.** Mean reward ≈0.86/move with high variance;
   a 3-match cascading one extra step (~6 cells at k=1 → 1.50) legitimately out-rewards a flat 4-match
   (1.00 train / 0.60 eval). γ=0 must average this noise per (s,a) from sparse visits — an SNR problem.
4. **The creation bonus is invisible in the observation.** The +40/+60/+100 exists only in the training
   TARGET. Both per-action input planes are **fire-only** (`immediateScore/300`, `deterministicValue/300`,
   `pg:775-777`): the input ranks the 4-match only ~1.3× above the 3-match while the target ranks it
   2.6–3.3×. The net must infer "this geometry creates a striped worth +0.4" from raw fruit planes —
   fighting its own most informative features. A per-action *shaped* plane was deliberately deferred in the
   M50 v1 ("don't feed the answer"); it is now the identified missing feature.
5. **The web `net` tier does NOT search.** `netAction` (`pg:985` region) is a pure masked argmax over one
   forward pass; `expectimax`/`expectimax2` are separate scripted tiers that never touch the net. (Corrects
   the memory note "expectimax-1 uses the net".) There is no inference-side mechanism that could rescue the
   ranking today.
6. **Data rarity compounds it.** Special-creating swaps are ~12% of random moves; post-warmup the collector
   is ~95% greedy, so a net settled into the 3-match habit rarely revisits 4-matches — the chosen-action-only
   loss then never corrects them.
7. **Special+special combos — already priced correctly, RESOLVED, no change.** Combos fire ON the swap
   (staged combo executes in `clearStep(0)`), so their full score lands in the immediate reward the γ=0 net
   regresses: striped+striped 1.50 · wrapped+wrapped ~2.50+ · striped+wrapped ~3.90+ · bomb+bomb ~6.40 —
   versus ~0.80–1.10 for firing one special alone; combining always wins, and the `immediateScore/300` input
   plane (a real `clearStep(0)` incl. `executeCombo`) makes that value visible to the net. Combo shaping
   would double-count and invite reward-hacking: the deliberate asymmetry is **shape what pays later
   (creation), never what pays now (firing/combining)**.
8. **Owner-confirmed constraint (2026-07-25): planning only sees fruits already on the board.** Refill is
   unknowable noise — all search is and stays refill-free (`resolveCascades(false)`), so each extra search
   ply plans on an increasingly depleted fictional board. Deeper search (depth 3+) is compute-feasible with
   beams but decays in value; the *expected* refill continuation is the net's job, not the search's.
9. **γ=0 ceiling (context for expectations).** At γ=0 the net regresses a quantity the deterministic
   simulator computes exactly, plus refill noise; its only genuine add over the scripted `specialsGreedy`
   tier is the *expected refill continuation*. Every immediate-term lever below therefore converges the net
   toward (a denoised) `specialsGreedy` — which fixes the owner's complaint but cannot claim the
   hold-for-combo gap (M50.3 gate 2). That gap needs a teacher with cross-move vision (§3, lever C).
   Exhausted levers (do NOT retry): γ>0 reward schedules (γ=0.99 M49, γ=0.5+3-step+PBRS `cf8train` — n=2
   losses), bigger shaping magnitudes (net chases shaping, gates measure bare score).

## 3. Design — three levers, cheapest first

**Lever A — inference re-rank: DROPPED at design time (2026-07-25), superseded by Lever B.**
The λ re-rank was scoped for the SHIPPED 928-input ckpt, but Lever B changes the observation width in the
same single-source `.pg` — the moment the obs grows, the old ckpt can't forward at all, so "quick fix on the
old net" and "retrain" cannot coexist in one working tree. Since the retrain runs immediately (this
session), the re-rank buys nothing. What replaces it: a **stale-ckpt guard** in the web loader (net input
width ≠ engine observation width → treat the net as missing, fall back to the expectimax tier) — needed for
the 928→1040 transition anyway and permanent robustness for every future obs change.

**Lever B — retrain with a ranking-aware loss + the missing feature (the owner's "punish harder", made
precise).** Three changes, one from-scratch run:
1. **Shaped per-action plane:** add a NEW `deterministicValueShaped(a)/300` third per-action block — obs
   928 → **1040**. *(Design correction over the first draft's `immediateScoreShaped`: the shaped
   DETERMINISTIC value — full refill-free cascade + creation weights, incl. cascade-made creations — matches
   the realized training reward up to refill noise. An immediate-only shaped target would re-create the exact
   bias we're fixing: a cascading 3-match's target would include its cascade while a 4-match's would not.)*
   Keep the two fire-only planes (the eval scoreboard is fire-only and the net should see both). Closes the
   §2.4 perception gap; this feature class is the exact lever that gated M49.
2. **Dense all-action regression:** per sampled state, supervise EVERY legal action — the taken action
   toward its realized reward (keeps refill-cascade expectation learnable), every other legal action toward
   `deterministicValueShaped(a)/100`, masked Huber. Weight semantics (locked): the dense term is normalized
   per supervised entry so the WHOLE dense term carries `DenseTargetWeight` (default 1.0) × the realized-
   reward term's total gradient mass — the ranking signal and the refill-expectation signal get equal say;
   per-entry weighting would let ~30 legal actions/state drown the realized term 15:1. The targets are
   already computed each step for the obs planes — this lever is nearly free, and it is the only one that
   makes a wrong ORDERING cost loss. Trainer seam: `DqnOptions.DenseTargets` (per-obs delegate, NaN =
   unsupervised action), guarded to γ=0 (only there is a dense target computable from s alone).
3. **Regret emphasis (pre-registered, only if the probe gate still fails):** add a margin hinge
   `max(0, Q(a_best) + m − Q(a_taken))` on transitions where the behaviour policy missed a
   strictly-better deterministic action (`a_best` from the shaped oracle). Registered up front to avoid
   schedule-tinkering drift; margin distorts Q as a reward estimate, so it stays OFF unless needed.
   Training length: 400k steps for comparability with `cf5train`; one optional 800k continuation if gates
   are close (the owner intends longer training anyway).

**Lever C — pre-registered escalation: expectimax-2 distillation (DAgger-style).** Trigger: gate 2
(gap-share ≥64%) still missed after Lever B. Teacher = `expectimax2Action` labels on states the *student*
visits (fixes distribution shift); auxiliary cross-entropy on the teacher's argmax alongside the dense
regression. This is the only lever that teaches value beyond the deterministic oracle (create→fire,
hold-for-combo) — the M50.3 close-out already pointed here ("search-guided play, not another reward
schedule"). Ceiling: the teacher is 2-ply and refill-free.

**Rejected:** bigger creation-shaping bonuses (known trap, §2.8) · combo shaping (§2.7, double-count) ·
γ>0 schedules (n=2) · regret-prioritized replay (subsumed by dense regression — every action gets a target
every update, so per-state prioritization by the missed action loses its point).

## 4. Probe — make "misses important opportunities" measurable (M51.0, before any fix)

A deterministic seeded probe over N≥500 states sampled by random walks, two metrics:
- **Strict 3-vs-4 probe:** states whose legal mask contains both a special-creating swap and a plain 3-match
  swap; metric = P(policy picks a creating swap).
- **Opportunity take-rate:** states where any special-creating swap exists; fraction taken. Baselines:
  random, greedy, specialsGreedy, expectimax-1/2, shipped net.
Plus a **combo take-rate** stat (states with a legal special+special swap) — expected already-high (§2.7),
recorded to confirm, not gated.

**M51.0 results (2026-07-25, 300 eps × 30 random-walk moves, seeds 9000+e; Lab `--probe 300`):** 9000 states,
3888 with a creating swap (all strict), 256 with a legal special+special combo.

| policy | strict take-rate | combo take-rate |
|---|---|---|
| random | 14.2% | 8.6% |
| greedy | 71.7% | 95.7% |
| specials-greedy | 91.4% | 94.9% |
| expectimax-1 | 33.1% | 71.5% |
| expectimax-2 | 38.4% | 24.2% |
| **shipped net (`cf5train`)** | **17.6%** | **48.0%** |

Two lessons locked in: (a) the shipped net is barely above random on the owner's exact scenario — complaint
confirmed and quantified; (b) **the gate must be RELATIVE to specials-greedy, not an absolute 95%** — the
draft assumed the shaped oracle takes a creating swap 100% of the time, but it measures 91.4%: sometimes a
non-creating swap (typically firing an existing special) legitimately out-scores creation even under shaped
scoring, and a net pinned to 100% would be WRONG in those states. Gate re-locked: **net strict take-rate ≥
specials-greedy's − 5 pts on the same probe** (≥ 86.4% against today's baseline). The fire-only expectimax
tiers' 33–38% independently confirm §2 finding 4 (creation invisible to fire-only value).

**Probe-bar post-mortem (2026-07-25, after the cf9train results — recorded in full because the bar moved
twice):** the ≥86.4% re-lock was ALSO mis-calibrated, and the cf9train run proved it: every policy that
actually scores well takes creating swaps far less often than specials-greedy (raw take: e1 33.1% at score
5951, e2 38.4% at 8098, cf9train net 35.5% at 5666 — vs specials-greedy's 91.4% at 3903). Creation-chasing
is simply not what optimal play looks like; a bar derived from specials-greedy selects for a 3903-scoring
behavior pattern. The probe was upgraded with the honest metric — **opportune take-rate = P(picks a creating
swap | a creating swap is the argmax of `deterministicValueShaped`)**, i.e. an opportunity only counts when
creating was actually the best move — plus an oracle-match column. On that metric the strong policies
cluster at 50–56% (e1 55.7%, e2 49.8%) because even the shaped 1-ply argmax is not score-optimal (e2 matches
it only 48% of the time while out-scoring everything). **Final gate (locked): net opportune take-rate within
5 pts of expectimax-1's — the strongest same-horizon planner — i.e. ≥ 50.7% against today's 55.7%.** The
absolute-90% idea died for the same reason the specials-greedy bar did: it demands behavior no strong policy
exhibits. What the probe is FOR is the before/after: cf5train sat at random-class behavior (raw 17.6% vs
random's 14.2%); the dense-loss net sits at search-class behavior on every column.

## 5. Milestones

- **M51.0 — Probe + baselines.** ✅ (2026-07-25) The §4 probe in the Lab (eval-only, seeded), run BEFORE the
  obs change so the shipped 928-ckpt still forwards. Creating-swap detection =
  `immediateScoreShaped(a) > immediateScore(a)` (works on the pre-change engine; step-0 creations are exactly
  the owner's scenario). **Gate:** probe deterministic across runs; shipped-net strict-probe and take-rate
  numbers recorded in this PRD. *Green: results table in §4 — net 17.6% vs specials-greedy 91.4%.*
- **M51.1 — Web stale-ckpt guard (replaces the dropped λ re-rank, §3).** ✅ (2026-07-25) Net input width ≠
  engine observation width ⇒ net treated as missing (expectimax fallback + console note) in the director's
  load callback. **Gate:** with the 1040 engine and the old 928 ckpt the net tier plays expectimax with zero
  console errors.
- **M51.2 — Retrain (Lever B).** ✅ (2026-07-25) Obs 1040 + dense loss (+ registered margin if probe-gated).
  *Built: `deterministicValueShaped` + third obs plane in the `.pg` (parity pin UNCHANGED 481681208 — no rule
  change, C#=TS re-verified); `DqnOptions.DenseTargets`/`DenseTargetWeight` in the shared trainer (γ=0
  guard; unsupervised entries via target:=prediction ⇒ zero grad; dense mass normalized per supervised
  entry); campaign extractor reads the shaped plane ×3; Lab `--dense`/`--dense-weight`; tests 63/63 green
  incl. 3 new dense-trainer tests (never-sampled-arm ranking, NaN unsupervised, γ-guard) + 3 engine tests.
  Smoke (6k steps): eval mean score 4607 — already above cf5train's 400k-step 4040 (20-ep eval, noisy, but
  the dense signal clearly bites). Full 400k run `cf9train` launched (--gamma 0 --dense --seed 1).* **Gates
  (M50.3 bars + the §4 final probe gate):** ≥ +30% over random CI-separated · created ≥7.3 / fired ≥5.6 per
  ep · gap-share ≥64% · net opportune take-rate ≥ e1 − 5 pts.
  ***RESULTS (cf9train, 500-ep gate protocol, final-rules baselines): ALL GATES PASS — the first Crazy
  Fruits net to do so.** Net **5666.4 ± 155.2** = **+117.2% over random** (bar +30%, CI-separated; cf5train
  was +54.9%) · **gap-share 91%** (bar 64%; cf5train missed at 43%) · created **9.57** / fired **10.39**
  (bars 7.3/5.6; cf5train missed at 5.81) · +61.4% over greedy, +45% over specials-greedy, 95% of e1's mean.
  Probe: opportune take **54.9%** vs e1's 55.7% (bar ≥50.7% PASS — search-class; the margin hinge stays
  unused) · raw take 17.6% → 35.5% (e1/e2-class) · combo take 48% → 66% · oracle match 54.2%. The margin
  hinge and any escalation are NOT needed.*
- **M51.3 — Escalation (Lever C, trigger-gated).** ❌ Not triggered — gate 2 passed at 91% (bar 64%). The
  remaining ceiling is the e1→e2 gap (hold-for-combo), out of scope per the M50.3 close-out.
- **M51.4 — Ship.** ✅ (2026-07-25) `cf9train` → `wwwroot/models/crazyfruits.dqn.ckpt` (the 1040-input net;
  the director's new dims guard retires the stale-ckpt window); round-over bar "~4 000" → "~5 650";
  PLAN/this PRD synced (ARCHITECTURE.md needed no change — no hard-coded obs width). Live watch-tier check
  on the running host + parity pin 481681208 unchanged.

## 6. Key code references

`DqnTrainer.cs` — `DqnOptions.DenseTargets`/`DenseTargetWeight` + the dense term in `TrainStep` ·
`CrazyFruitsEnv.cs:89-99,134-156` (reward + shaping) · `CrazyFruitsBoard.cs` (per-action oracles:
`ImmediateScore`/`DeterministicValue`/`ImmediateScoreShaped`/**`DeterministicValueShaped`**) ·
`crazyfruits_solver.pg` — `immediateScoreShaped`, `deterministicValue`, **`deterministicValueShaped`**,
`buildObservation` (3 per-action planes), `netAction` · `CrazyFruitsDqnCampaign.cs` —
`DenseTargetsFromObservation` (shaped plane ×3 → reward units) · `CrazyFruitsLab.cs` —
`--dense`/`--dense-weight`/`--probe N` (`cf9train` = γ=0 + creation shaping + dense, 400k steps) ·
`crazy-fruits-director.ts` — stale-ckpt guard · `DqnDenseTargetsTests.cs` (three-arm ranking proof).
