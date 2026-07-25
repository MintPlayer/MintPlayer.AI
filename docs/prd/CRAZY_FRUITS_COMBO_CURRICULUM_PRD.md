# Crazy Fruits — combo curriculum (seeded-specials training starts) — PRD

**Status:** 🔜 planned 2026-07-25 (2-agent design verification: code-safety + RL sanity)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M52 · extends [CRAZY_FRUITS_RANKING_PRD.md](CRAZY_FRUITS_RANKING_PRD.md)
(M51, shipped; PR #39 merged 2026-07-25) · branch `m52-crazyfruits-combo-curriculum` → master

## 1. Problem

Owner observation after M51 shipped: the net still occasionally skips a 5-in-a-row, and — the sharper
question — it has plausibly *never experienced* what a wrapped+wrapped or bomb+bomb combo pays. The M51
per-kind probe quantified the residuals on `cf9train`: combo take-rate **63%** (greedy 92%, specials-greedy
90% — combos pay 1.5–6.4 reward *immediately*, so unlike creations a high take-rate is genuinely correct
here), bomb-bucket opportune take **80%**.

What the net already has without any engineering (M51's dense loss + per-action planes): the *deterministic*
combo payoff is in its INPUT (the planes run `executeCombo` in simulation) and in its dense TARGETS (every
legal combo swap's Q is supervised toward that payoff even when never played). What it lacks:
1. **Frequency** — combo-legal states are ~2.8% of natural random-walk states (845/30 000; adjacent
   bomb+bomb far rarer), so those dense targets get few gradient hits and the dueling trunk barely shapes
   itself around them.
2. **The aftermath** — the realized post-combo continuation (a huge random refill after a board wipe) is
   only learnable by actually PLAYING combos.
3. **Creating the setup** — deliberately steering two specials together is a multi-move plan a γ=0 reactive
   net cannot represent; that is the e1→e2 hold-for-combo gap, explicitly OUT OF SCOPE here (reserved for
   the search-guided lever, RANKING PRD lever C).

## 2. Design — seed the situations, keep the yardstick honest

Two levers, per the M52.0 review (which also killed one idea — see Rejected):

**Lever 1 (primary) — combo-biased ε-exploration, `CrazyFruitsEnv.ComboExploreBias` (q, default 0).** When an
ε-exploration step fires and a legal special+special swap exists, pick uniformly among the combo swaps with
probability q instead of uniformly over all legal swaps. This buys realized post-combo experience (the piece
dense targets can't supply) with ZERO distribution shift — the boards stay natural. Trainer seam:
`DqnOptions.ExploreBias` (`Func<rng,int>`, −1 = no suggestion), consulted only inside the ε-branch; null ⇒
zero extra RNG draws, so every other game stays bitwise-identical.

**Lever 2 — seeded-specials training starts, `CrazyFruitsEnv.SeedSpecialsProb` (p, default 0 — train env
only; the eval env and every gate stay on the natural distribution).** On `Reset` (seeded and autoreset paths
alike), with probability p the fresh board is dealt *combo-ready*:
- one **adjacent special pair** (a special+special swap is legal immediately) — ADJACENT ONLY: a near pair
  one swap apart was considered and rejected, because at γ=0 the bring-together swap pays ~0 immediately and
  the oracle prices only immediate firing, so its training label is flat — such states teach nothing (the
  cross-move setup skill belongs to the search-distillation lever, RANKING PRD lever C);
- plus up to 2 extra singles on random plain cells;
- kinds drawn uniformly over {stripedH, stripedV, wrapped, bomb}.
This lever exists specifically to manufacture the states ε-bias can't reach naturally (adjacent bomb+bomb).

**Injection is invariant-safe by construction:** a cell's packed value is overwritten keeping its fruit type
(striped/wrapped) — typewise match structure unchanged, so the dealt board's no-instant-match and
has-legal-swap guarantees survive; a bomb (type 0) never joins runs, and bomb swaps are always legal, so
legality only grows. All randomness comes from the env's own RNG stream (`_rng`), never the engine's refill
stream — seeded runs stay deterministic, and the unseeded autoreset path continues the same stream.

**Why this is sound off-policy:** DQN learns from the replay distribution, not the on-policy one, and at γ=0
every update is a pure per-(s,a) regression — there is no bootstrap through which rare-state value error can
propagate. Seeding shifts *which states* get gradient, nothing else. The natural-board 500-ep score gate
guards against forgetting the common case.

**Checkpoint reuse (owner question — YES, upgraded to a full RESUME):** the observation is unchanged (1040),
and the M52.0 review flagged warm-start's two weaknesses (cold Adam moments, empty replay buffer) — both of
which a full training-state resume solves for free: copy BOTH `crazyfruits.dqn.ckpt` + `crazyfruits.dqn-state.ckpt`
into the new data dir and raise `--steps` to the absolute 800k. The run continues with cf9train's optimizer
state, its 100k natural-board replay buffer (new seeded transitions blend in gradually — the reviewer's
pre-fill concern mooted), ε already at its 0.05 floor (the reviewer's ≤0.1 recommendation), and keep-best
seeds its baseline from the resumed net's own eval — the deployable net is only overwritten by something
that BEATS it. The curriculum options live on the env instance, not in the state file, so the resumed run
picks them up on the next reset.

**Rejected for v1:** ε-exploration biased toward combo swaps (adds an on-policy knob the seeding may make
unnecessary; registered as the escalation if gates miss) · prioritized replay for seeded episodes (same
reason) · mid-episode injection (Reset-only keeps the invariant argument trivial).

## 3. Milestones

- **M52.0 — Design verification.** ✅ (2026-07-25) 2-agent check. Code-safety verdicts: injection
  invariant-safe (kind overwrite keeps the fruit type ⇒ typewise match structure byte-identical; a bomb
  never joins runs and its swaps are always legal — note the precise invariant is "≥1 legal swap
  guaranteed", not "legality only grows", since a bomb can remove a match-based swap elsewhere); reshuffle
  with seeded specials is bounded (Fisher-Yates keeps the multiset; 1000-cap → plain re-deal); env-stream
  RNG keeps per-seed determinism; save/restore round-trips packed specials untouched; hard requirement =
  option defaults OFF (all existing env tests build default envs). RL verdicts: at γ=0 there is NO bootstrap
  propagation path — curriculum risk reduces to benign covariate shift with identical labels; **p = 0.25**
  (fallback 0.1); **adjacent pairs only** (rejected near-pairs: the bring-together swap's γ=0 label is flat);
  **combo-biased ε is the PRIMARY lever** (q, realized combo experience at zero distribution shift); no PER;
  prefer full resume over warm-start (Adam moments + natural replay carry over).
- **M52.1 — Env + trainer seam + Lab.** ✅ (2026-07-25) `SeedSpecialsProb` (facade-level `GridSnapshot` →
  modify → `LoadGrid` on Reset) + `ComboExploreBias`/`SuggestComboExploration` on the env; generic
  `DqnOptions.ExploreBias` hook consulted only inside the ε-branch (null ⇒ zero extra RNG draws ⇒ every
  other game bitwise-identical); campaign wires the hook from the injected train env; Lab
  `--seed-specials`/`--combo-explore`; per-kind created counts in `--baselines` (gate 5's instrument).
  **Gate:** 67/67 green — 4 new tests: seeded invariants + adjacent pair + legal combo swap, per-seed
  determinism + default-env-stays-plain, suggestion hook legality, agent bias redirect.
- **M52.2 — Spike.** ✅ (2026-07-25) Scratch-dir resume of cf9train +10k curriculum steps: resumed at 400k,
  keep-best baseline 6014 armed, loss 0.082 (normal band), 20-ep eval 5012 (inside cf9train's own eval noise
  band — no collapse).
- **M52.3 — Train (`cf10train`).** 🔄 (launched 2026-07-25) Full resume of cf9train (both ckpt files) →
  `--steps 800000` absolute = **400k new steps** (owner choice: reuse ckpt + full budget), `--seed-specials
  0.25 --combo-explore 0.5 --gamma 0 --dense --seed 1`.
  **Gates (pre-registered):** (1) 500-ep natural-board score NOT CI-separated below `cf9train`'s 5666.4;
  (2) probe (1000 eps, natural boards): combo take-rate ≥ 75% (from 63%); (3) bomb-bucket opportune ≥ 85%
  (from 80%); (4) all M50.3 bars stay green (≥+30%/random · gap-share ≥64% · created ≥7.3 / fired ≥5.6);
  (5) per-kind created non-regression vs cf9train, ESPECIALLY bomb-created (free seeded specials are exactly
  the pressure to under-create the rarest kind). Keep-best ships only a net that beats the resumed net's
  eval. Escalation if (2)/(3) miss: raise q, one run, then stop-loss.
- **M52.4 — Ship + PR.** Best net → `wwwroot/models/crazyfruits.dqn.ckpt` (drop-in, no web change needed);
  docs/memory synced; PR → master.

## 4. Results

*(to be filled at each gate)*
