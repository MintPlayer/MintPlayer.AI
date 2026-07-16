# Draughts (dammen) self-play showcase — PRD

**Status:** 🔜 planned 2026-07-15 (4-agent investigation: repo-fit, game domain, field evidence, chess post-mortem)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M47 · **Replaces** chess as the self-play *strength* showcase (chess strength thread closed 2026-07-15, see [RESIDUAL_CONV_NET_PRD.md](RESIDUAL_CONV_NET_PRD.md) status; the chess demo itself stays on the site).

## 1. Problem — why chess failed, why draughts should work

The chess self-play effort produced excellent *infrastructure* (conv net, GPU-resident forward/train, multi-GPU
generation, ladder, adjudication) but no net that beats even a depth-1 material minimax: at laptop scale, ~35
branching × expensive movegen starved the 64-sim search, and weak-net games ended as 120-ply shuffle-draws
carrying no outcome signal (draw-collapse — diagnosed and patched with material adjudication, but the signal
stayed thin). Final scoreboard: 9 configs, every net 33–40% vs minimax-d1 with 0–1 wins while bleeding 20–30
pawns (journal: `data/chess-conv-autorun-log.md`).

Draughts inverts every pressure point, *by rule*:

| Failure driver (chess) | Draughts |
|---|---|
| Branching ~35 → <2 visits/child at 64 sims | Avg branching ~4 (10×10; forced captures often reduce to 1–3) → 8–20 visits/child, a real search |
| Expensive movegen (pins, castling, EP) | Trivial movegen (~44M pos/s on 2008 hardware for 8×8) → the research top lever (generations) becomes affordable |
| Weak-net games = non-decisive shuffle draws (z≈0) | **Forced captures + majority rule ⇒ weak-level games are decided by material blunders within ~10 moves** — dense, natural win/loss signal, no synthetic adjudication needed early |
| Policy head 4672 (every piece moves differently) | All men move identically (owner's observation): policy 2500 (10×10 from-to) / 1024 (8×8) — each output gets ~5× the training signal |

**Field evidence (GO):** AlphaCheckers-Zero reached 8×8 checkers' game-theoretic ceiling (50/50 draws vs
minimax **depth-8**) with ~12,500 self-play games at 80 sims in **~10 h on a Colab T4** (converged by ~5,000
games); AlexMGitHub/Checkers-MCTS beat casual online bots with **800 games on a 2015 laptop**; galvanise_zero
trained 10×10 international draughts AlphaZero-style at hobby scale. Throughput reference: ~1,250 games/hour at
80 sims on T4-class hardware (an RTX 3060 is ≥ T4). The chess run's ~300 games ≈ 15 minutes of draughts
self-play. Honest ceiling: laptop self-play climbs the weak-to-amateur ladder, **not** classical-engine strength
(a Russian-checkers AZ hobby net lost 0–100 vs a classical engine; Scan/Kingsrow-class is out of scope).

## 2. Variant decision: International 10×10 ("dammen") first, parameterized so 8×8 falls out

Primary showcase = **International draughts, 10×10, 20 men/side** — the variant actually played in NL/BE (KNDB,
Sijbrands/Wiersma/Boomstra; 8×8 English checkers reads as "toy checkers" to that audience), with the most
dramatic tactics (majority-rule *coups*, flying kings) and the most decisive weak-level play.

Rules that matter (and their flags): captures forced; **majority rule** (must play a sequence capturing the
maximum number of pieces); men capture forward AND backward; **flying kings** (slide any distance, land anywhere
beyond the captured piece); captured pieces removed only at sequence end and cannot be re-jumped (**Turkish
strike**); promotion only if the move *ends* on the back row. The engine is parameterized (board size +
rule flags: majority, backward men-capture, flying kings) so **English 8×8 checkers is a config**, kept for
A/B and for the "draws depth-8 minimax = solved-game ceiling reached" story (Chinook 2007: 8×8 is a draw).

Drawishness (>90% GM draws in 10×10) is a *strong-play* phenomenon — irrelevant at showcase strength, where
forced captures make blunders immediately fatal. The real draw risk is king-shuffle endgames, handled in §3.5.

## 3. Design

### 3.1 Engine — `draughts_solver.pg`, single-source from day one
Author the rules in MintPlayer.Polyglot (`draughts_solver.pg`, C# emitted at build, TS committed) with a thin C#
facade, exactly like chess (`chess_solver.pg`/`ChessBoard.cs`) — but .pg-FIRST this time, so the future browser
page (§7) is a pure-frontend milestone. Movegen enumerates **complete capture sequences** (recursive DFS with
majority filter + Turkish-strike constraint + promotion-at-end); Polyglot 0.6.0's multi-`.pg` build fix removes
the old codegen risk. `CheckersState`/`DraughtsState` is small (three bitboards fit 8×8; 10×10 needs 50-square
boards — two ulongs per piece kind).

### 3.2 Move encoding — one capture sequence = ONE move (the critical decision)
`IZeroSumGame.Apply` FLIPS the side to move (`IZeroSumGame.cs:24`) and MCTS/SelfPlayCampaign/negamax all
sign-flip per ply. A multi-jump is one turn, so: **a full capture sequence is a single move index**. Policy =
`from × to` over playable squares: **50×50 = 2500** (10×10), **32×32 = 1024** (8×8). Distinct maximum-capture
sequences sharing (from,to) but capturing different pieces (possible under the Turkish strike) collide — resolve
by a **deterministic canonical pick** (e.g. lexicographically smallest jumped-square list). This mildly restricts
the net's move menu on rare forks (always still a legal maximum capture; PDN notation itself is ambiguous here)
and is the price of NOT touching the MCTS sign-flip contract.
**Rejected:** micro-move decomposition (each jump segment = one action, mover doesn't flip mid-sequence). It's
rules-perfect and gives a smaller head (~1800), but requires an `IMoverInfo` capability seam with conditional
sign-flips through `Mcts`, `SelfPlayCampaign`, and `MaterialMinimaxPlayer` — the highest-risk kind of change
(silent training corruption; chess post-mortem risk (b)) — and the majority rule needs the full-sequence
enumerator for legality masking *anyway*, so micro-moves save nothing.
Internal movegen still enumerates full sequences distinctly: **perft runs on sequences** (matches published
tables); the policy map dedupes on top.

### 3.3 Observation — 5 planes
`my-men, my-kings, their-men, their-kings` (side-to-move relative, chess convention) + a normalized
**no-progress counter** plane (evidence pitfall: nets need to see the draw clock). 5×100 = 500 floats (10×10);
`ConvNetBuilder(planes: 5, boardH: 10, boardW: 10, filters, blocks)` — confirmed fully generic, policy head =
`Linear(2·H·W → 2500)` ≈ 500k params, fine.

### 3.4 Material — `IMaterialScore` (existing seam, no campaign changes)
man = 1, king = 3, side-to-move relative. Powers value-target blending (`MaterialWeight`), capped-game
adjudication, ladder `PromoteMaterial`, and `MaterialMinimaxPlayer` — all existing.

### 3.5 Draws defined out of existence
In-engine **no-progress rule**: N reversible moves (no capture, no man advance; N≈40 own-moves, FMJD-style) ⇒
draw — king-shuffle non-games terminate *by rule* instead of by ply cap. Keep ply cap (~150) + material
adjudication (±1.5 man-units, kings weighted) as the backstop.

### 3.6 Reused as-is (M46 infrastructure)
`Mcts` (incl. batched leaves), `SelfPlayCampaign` (adjudication, ladder, arenas, parallel generation, GPU
forward/trainer wiring), `AddSelfPlayCampaign<TState>` + `[Register]` game registration, `LabHost`. New code:
the game, its Lab entry (~130 lines cloned from ChessLab), and one generalization — `ChessStrengthEval` →
`StrengthEval<TState>` (it's chess-specific only in ctor/builder/ckpt defaults; `MaterialMinimaxPlayer<TState>`
is already generic).

## 4. Locked training constants (chess post-mortem, do not re-derive)
`lr 3e-4` (1e-3 peak-then-regressed) · `material-weight 0.5` (0.3 broke the gate) · `arena-games ≥ 40` (12 was
±1-pawn noise) · sims 64 start, **floored by decisiveness, never cut for wall-clock** · parallel eval/arena ·
ladder on · `--vs-minimax` wired from day one and run automatically at every promotion. Dropped as signals:
winRate-vs-random (saturates → 100% here) and material-vs-champion (self-referential) — cosmetic only.

### 4.1 GPU recipe (owner Q 2026-07-15: "too few matmuls?" — no)
Run M47.4 with `--gpu --leaf-batch`. The **training step is the proven win**: the M44 resident trainer measured
**~24×** per 128-batch on the chess conv tower, is generic over `ConvResidualPolicyValueNet`, and the draughts
tower does ~1.5× the per-sample GEMM work of chess (100 spatial positions vs 64; the thinner 5-plane input only
affects the first layer) — plus M44.1 showed the train step dominating chunk wall-clock (→~98%) at the default
40k window, so the 24× applies to the dominant cost. **Generation is the caveat**: batch-1 MCTS barely uses a
GPU; `--leaf-batch` (M42.5 virtual-loss batching) helps, but with 64 sims × branching ~4 the effective leaf
batches are modest — expect a real but unspectacular generation speedup, which matters little since draughts'
cheap movegen makes CPU generation fast anyway.

### 4.2 Net growth (owner Q 2026-07-15: "does the net auto-widen when saturated?" — no)
The SDK's Net2Net toolkit (`WidenTo`/`Deepen` on MLP-trunk nets, driven by `DqnGrowth`/`PolicyGrowth`/the
cube-DAVI width ladder) is wired into the DQN and imitation campaign families only — and it grows on a fixed
sample cadence (`GrowEvery`, opt-in), not on a saturation signal. The self-play path draughts uses has no
growth seam at all: the net is built once by its `IPolicyValueNetBuilder`, and `ConvResidualPolicyValueNet`
doesn't implement `IGrowableTrunkNet` (no conv-tower `WidenTo` exists). If the draughts net saturates, the
lever is a relaunch with bigger `--filters`/`--blocks` (fresh weights). Auto-growing the conv tower would be
real feature work: Net2Net across residual blocks + LayerNorm, a growth seam in the builder/checkpoint-kind
scheme, and rebuilding the M43/M44 GPU-resident forward/trainer whose kernels compile against a fixed shape.

## 5. Milestones & gates (falsifiable, in order)

- **M47.1 — Engine.** ✅ SHIPPED 2026-07-15. `draughts_solver.pg` parameterized 10×10/8×8, full-sequence movegen, `IZeroSumGame` +
  `IMaterialScore`, no-progress rule; C# facade + `[Register]`. **Gate: perft matches published tables for BOTH
  variants** (10×10 start: 9/81/658/4265/27117…; 8×8: 7/49/302/1469/7361…) **+ capture-dense positions
  exercising majority, Turkish strike, promotion-mid-sequence.** No training before this gate.
  *Gate result: perft green 10×10 d1–d8 (incl. 6,483,961) + 8×8 d1–d9 (incl. 3,963,680); 8 hand-verified rule
  tests. Design note: english "man crowning mid-jump stops" is EMERGENT in sequence movegen (english men capture
  forward-only; no forward jump exists from the last row) — three rule flags suffice, not four.*
- **M47.2 — Encoding + observation.** ✅ SHIPPED 2026-07-15. (from,to) policy map + canonical tie-break; 5-plane observation.
  **Gate: encode→decode→apply round-trip over thousands of random playouts, with a collision audit** (count and
  log canonical-pick events; zero unmapped legal moves).
  *Gate result: 27,345 intl + 21,486 english random-playout positions round-trip clean; collisions 4 intl / 0
  english (~0.015% — the predicted rare Turkish forks) + a directed english fork pins the canonical pick.
  Mover-relative frame (obs AND indices 180°-rotated for Black — one perspective to learn, no side-to-move
  plane needed) proven by start-position index symmetry + plane rotation tests.*
- **M47.3 — Lab + eval + tests.** `--game draughts` (`--variant checkers8` flag), `StrengthEval<TState>`
  generalization, DI smoke tests, self-play contract + determinism-SHA tests instantiated with the new state.
  **Gate: one training chunk end-to-end + bitwise DOP-invariance SHA + `--vs-minimax` produces a number + a
  `--bench-forward`-style micro-bench of the 5×10×10 tower** (real resident-forward/trainer numbers for this
  net before the showcase run, not extrapolations from chess).
- **M47.4 — The showcase run** (decided with owner 2026-07-15): **start with a cheap 8×8 validation run** —
  strongest field precedent, smallest policy head, fastest possible proof that the whole pipeline learns — and
  once it demonstrably climbs the minimax ladder, **flip the variant flag to 10×10 for the real dammen
  showcase**. Constants of §4, GPU recipe of §4.1. **Gates:** natural-decisive fraction ≥ 50% by game 200 (else
  stop-and-diagnose); **beat minimax-d1 with ≥ 60% score INCLUDING ≥ 10 real wins per 40 games, within 500
  self-play games**; d2 ≥ 55% within ~2,000 games; capped-equal-material games ≤ 30%. **No-thrash stop-loss:**
  judge each config at g160–200, one lever per intervention, two same-gate failures ⇒ it's a code/design
  problem — stop and write up. Ladder tiers promoted on the way = the site's future difficulty roster.
  **✅ 8×8 validation leg PASSED 2026-07-15** — ONE 1-hour run (`--variant checkers8 --gpu --leaf-batch 32
  --parallel --ladder`, RTX 3060 laptop), 312 games, zero interventions: **vs minimax-d1 26W 13D 1L = 81.2%,
  +6.55 men** (gate: ≥60% incl. ≥10 wins, within 500 games — passed at 312 with 26 wins); **vs minimax-d2
  10W 25D 5L = 56.2%, +2.80 men** (gate: ≥55% within ~2,000 — already passed at 312); two ladder tiers
  promoted (L2 at g200, +3.08 material over champion); winRate-vs-random saturated at 100% by g40 (as
  predicted — cosmetic); mid-run probe at the g200 judge point was already 87.5% vs d1. Reference: chess's
  best-ever config was 40% with 0 wins vs d1 after days of runs. The natural-decisive fraction isn't logged
  as a metric (worth instrumenting for 10×10), but the strength results subsume the concern. **Remaining leg:
  flip to 10×10 (drop `--variant`) for the 3–8 h dammen showcase.**
  **Extension run 2026-07-16** (4 h resumed, 824 games, ladder → scratch dir): promoted L2–L5 (+1.28/+1.63/
  +1.05/+0.78 men) then plateaued (3 failed arenas vs L5) — the 8×8 config is near its ceiling. External eval
  at BROWSER conditions (8 sims, 40 games — n.b. training-time probes run at 64 sims, so the 56.2%-vs-d2 gate
  number is NOT comparable; always re-measure the control under the same protocol): deployed 87.5%/16.3% vs
  d1/d2, **L3 90.0%/26.3% — dominant on both, shipped as the 4th tier "Master"** (`checkers8.az.master.ckpt`,
  8 sims); L5 was stronger in-family but weaker externally (80.0% vs d1) — self-play drift, not shipped.
- **M47.5 — Browser play.** TS side of the `.pg` + `.ckpt` parser + Angular page, the M40
  chess pattern; pure frontend by construction. Not part of the M47 gate.
  **✅ 8×8 leg SHIPPED 2026-07-15** (owner pulled it forward to ship the validated checkers8 net): `/draughts`
  page with **play-vs-AI + AI-vs-AI watch** (the chess page pattern), fully client-side. New over chess: the
  browser runs the **conv tower** (chess ships an MLP tier) — net + MCTS added to `draughts_solver.pg`
  (types `PgDraughts*`; the generated C# shares chess's global namespace), TS twin emitted via the bundled
  polyglot CLI, and `draughts-net.ts` parses kind `selfplay-pv-conv` (dims from the file; byte-level
  reference = `DraughtsNetParityTests`, which also pin conv-forward parity to f32 tolerance, observation
  parity, and MCTS legality). Tiers (`wwwroot/models/draughts-difficulties.json`, ckpts Git-LFS): Beginner
  (d1 ckpt, 1 sim, T=1), Casual (d2, 2 sims, T=0.5), Strong (final, 8 sims, T=0) — sims chosen by
  measurement: 1 sim = 32.5% vs minimax-d1, **8 sims = 82.5% ≈ the full 64-sim strength**, and the emitted
  JS conv costs ~161 ms/forward, so Strong "thinks" ~1.2 s/move (node simulation of the exact browser path:
  real ckpt parsed, full legal game played). The 10×10 dammen net drops in via the manifest + a
  `PgDraughtsState.international()` start-state switch once its campaign runs.

Effort (repo-fit estimate): 10×10 trainable end-to-end ≈ 2–3 weeks of milestones; M47.4 itself is
evenings-scale compute (evidence: ceiling in ~10 T4-hours for 8×8).

## 6. Risks
1. **Multi-jump encoding bug silently corrupting training** — the checkers analogue of a movegen bug, mitigated
   by the hard M47.1/M47.2 gates *before* any training.
2. **Draw plateau at the top** — as the net strengthens, draws-with-material-parity vs strong minimax are the
   *correct* outcome (8×8 is a solved draw). The ladder must count parity-draws as progress at the top; success
   messaging is "climbs the ladder to its ceiling", not "wins forever".
3. **10×10 evidence is thinner than 8×8** (galvanise_zero vs two quantified 8×8 repos) — hedged by the
   parameterized engine: if 10×10 underdelivers, 8×8 is a config switch with the strongest precedent.

## 7. Out of scope
Browser page (M47.5, deferred) · classical-engine strength (Scan/Kingsrow use ML *pattern* evals + alpha-beta —
different technology) · killer-draughts / anti-draw variants · 8×8 3-move-ballot opening rules.

## 8. References
AlphaCheckers-Zero (github.com/MadrasLe/AlphaCheckers-Zero) · Checkers-MCTS (github.com/AlexMGitHub/Checkers-MCTS)
· galvanise_zero (github.com/alreadydone/galvanise_zero) · alpha-nagibator (github.com/evg-tyurin/alpha-nagibator)
· Schaeffer et al., *Checkers Is Solved*, Science 2007 · Samuel 1959 (IBM J. R&D — self-play checkers on 1950s
hardware) · A. Jones, *Scaling Scaling Laws with Board Games* (arXiv:2104.03113) · en.wikipedia.org/wiki/International_draughts
· en.wikipedia.org/wiki/Game_complexity · chess post-mortem: `data/chess-conv-autorun-log.md`.
