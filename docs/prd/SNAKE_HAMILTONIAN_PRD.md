# Snake — safety-cycle mode ("never lock yourself in") — PRD

**Status:** shipped (M48.1–.3) · 2026-07-16 · branch `m48-snake-hamilton` (off `master`)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M48 · **Depends on:** M34 (net-guided search, `SNAKE_SEARCH_PRD.md`), M35 (renderer)

## 1. Problem

The M34 search snake averages ~81 food@12 (single games 106–108) but **still occasionally locks itself in** —
self-traps that form beyond the 12-ply search horizon. `SNAKE_SEARCH_PRD.md` §8 already names the fix: a
**tail-reachability survival invariant / Hamiltonian-style endgame** as the next milestone. This PRD is that
milestone: a **new mode, side-by-side with the existing "Watch AI"** (which stays untouched as the pure
learned-search showcase), where the snake provably cannot die, while the trained net/search still provides the
speed toward food.

## 2. Goal & success criteria

A second watch mode in which the snake **cannot self-lock, ever**, and plays out to a full board.

- **Gate (C#, Lab eval, 12×12, ≥50 episodes): 0 deaths; ≥95% of episodes end board-full (win).** The remaining
  ≤5% may end by livelock cutoff (see §6 risks), never by death.
- **Speed gate (M48.2):** with model-guided shortcuts/rebuild enabled, mean steps-to-win ≥20% lower than pure
  cycle-following (measured in the Lab harness).
- **No retraining, no observation change** — `snake-net.ckpt` reused verbatim; the net/search keeps choosing the
  path toward food, exactly as today, but only among *certified-safe* moves.
- **Single-source:** all cycle logic lives in `snake_solver.pg`, transpiled to C# (Lab eval/tests) and the
  browser TS twin, byte-identically.
- **Existing "Watch AI" mode is byte-for-byte unchanged.**

## 3. Key decision — settled by a 2-agent investigation (2026-07-16)

Two agents ran: a repo map (snake is 100% client-side since M33; the seam for a new mode is the director +
`.pg`) and an algorithm survey (prior art, complexity, benchmarks).

**The owner's original scheme** — per food spawn: shortest path to food, then complete a **full Hamiltonian
cycle** through {body + food path} covering every remaining cell — was evaluated and **rejected in its literal
form**: Hamiltonicity with forced subpaths is NP-complete even on planar/grid graphs (Itai–Papadimitriou–
Szwarcfiter 1982; forced-edge variant), and — worse in practice — the *shortest* food path very often leaves the
free region with a **parity violation** (checkerboard argument: a Hamiltonian cycle alternates colors, so black
/white counts and endpoint colors are forced), meaning **no completion exists at all** and the scheme has no
answer. The Umans–Lenhart P-time result for solid grid graphs doesn't rescue it: the snake body carves holes,
so the free region is generally not solid.

**The owner's relaxation (2026-07-16) — "the cycle doesn't need to cover every cell, just as many as
possible" — is what makes it tractable**, and it converges with the best-known practical designs:

- **Tapsell's perturbed Hamiltonian cycle (PHC):** fixed precomputed cycle; each tick, shortcut along the cycle
  order when safe (1D ordering invariant tail < head, no shortcuts past 50% board fill). O(1)/tick, 100% win
  rate on benchmarks — but ~2× slower to win than repair methods.
- **Dynamic Hamiltonian Cycle Repair (DHCR, Haidet):** maintain a mutable cycle, locally splice it toward the
  food path, fall back to following the current cycle when repair fails. ~40% fewer steps than PHC, 97% win on
  30×30 (its rare repair failures are exactly what a fallback-to-cycle + tail-guard neutralizes).

**Decision:** ship a **maintained safety-cycle invariant** — the snake always holds a stored cycle `C`
containing its body as a contiguous segment with `|C| ≥ length + pending growth`; following `C` is always legal,
so death is impossible by construction. On top of that invariant, the **net/search picks moves toward food among
cycle-safe options** (M48.1, PHC-style — the guarantee), then the **per-food max-coverage cycle rebuild** — the
owner's scheme, relaxed — replaces the fixed cycle (M48.2, DHCR-style — the speed). Every rebuild failure falls
back to "keep following the current cycle", which is always available.

**Rejected alternatives:** literal full-Hamiltonian completion per food (NP-hard + frequently infeasible, see
above); TS-only implementation in the director (loses the C# Lab benchmark harness and breaks the repo's
single-source convention); cell-tree/2×2-cell methods (best benchmark numbers but an entirely different, more
intricate construction — not needed to hit the gate, and it sidelines the trained net the mode is meant to
showcase).

## 4. Design (`snake_solver.pg` → C# + TS)

### 4.1 Cycle representation & invariant

- `cycle: List<i32>` (cell indices, closed loop) + `cycleIndex: List<i32>` (cell → position on cycle, −1 if
  off-cycle), stored on `PgSnakeEnv` (or a small `record PgSnakeCycle` — mind the M34 §9 transpiler rule: typed
  `record` elements, never bare `List` locals that infer `any`).
- **Invariant (checked in debug/tests, never violated by construction):** the body is a contiguous segment of
  `cycle`; `|cycle| ≥ length + growthPerFood + margin`; every cycle cell is on-board and distinct.
- 12×12 is even × even, so a full-board Hamiltonian cycle always exists (boustrophedon columns + a return seam,
  or the half-grid maze walk). Odd×odd boards get a near-full cycle (one cell short) — fine under the relaxed
  scheme; `size ≥ 5` stays supported.

### 4.2 M48.1 — fixed full-board cycle + model-scored safe shortcuts (the guarantee)

`chooseActionCycle(net, …)` per tick:

1. Candidate moves = legal neighbors that **preserve the 1D cycle ordering**: the target's cycle index lies
   strictly between tail and head in cycle order (distance arithmetic mod `|C|`), with a slack buffer
   (`dist_to_tail − pending_growth − margin`) and **no shortcuts once free cells < 50% of the board** (late game
   reverts to pure cycle-following, which finishes the board cleanly).
2. Score the surviving candidates with the **existing net/search** (reuse `chooseActionSearch`'s scoring or a
   thin rootQ ranking — the "quickest path to food" the model is already good at), pick the best.
3. No safe shortcut ⇒ take `cycle-next(head)` — always legal. **This is the no-death proof: step 3 always
   exists.**

### 4.3 M48.2 — per-food max-coverage cycle rebuild (the owner's scheme, relaxed)

On each food spawn (and retried on later ticks if it fails — the head's position on the cycle changes each
tick, giving fresh geometry):

1. **Path to food** `P1: head → food` — the reused model/search (or plain BFS shortest path scored by the net;
   measure both in the Lab).
2. **Return path** `P2: food → tail-follow cell` through remaining free cells (BFS; prefer space-filling).
3. `C0 = body ⧺ P1 ⧺ P2` is a cycle containing the body.
4. **Extension:** repeatedly absorb adjacent free-cell *dominoes* — two adjacent free cells lying next to a
   cycle edge splice in as a detour (`a→b` becomes `a→x→y→b`) — until no domino fits (O(1) splices on a
   successor map). Full absorption is always parity-possible: `C0` is a cycle on a bipartite board, so its
   length is even and so is the leftover free region.
5. **Commit criterion (hardened during implementation): the new cycle must cover the whole board.** A partial
   cycle can strand future food off-cycle forever (the livelock this PRD originally only mitigated); requiring
   full coverage keeps the invariant *the food is always on the cycle*, so the M48.1 no-death argument extends
   to a no-livelock argument. **Any failure at any step ⇒ keep the previous full-board `C` unchanged** (still
   valid — the body never left it) and keep following it; that fallback is a plain M48.1 tick. The "as many
   cells as possible" relaxation lives on in the extension mechanism; "as many as possible" that is *less than
   all* is exactly the case that must not commit.

All simulation stays RNG-free (M34 rule: RNG lives with the caller), so C#/TS remain byte-identical.

### 4.4 Facade & Lab

- `SnakeEnv` facade: `ChooseActionCycle(SnakeCycleConfig)` mirroring `ChooseActionSearch`; config record in
  `Snake/SnakeSearch.cs` style (margin, shortcut cutoff, rebuild on/off — the on/off flag is how the Lab
  measures M48.1 vs M48.2).
- `SnakeLab`: extend the existing `--search` eval path (`RunSearchEval`) with a `--cycle` variant reporting
  win rate, deaths, mean steps-to-win, mean food, ms/move.
- Tests (`SnakeEnvTests.cs`): cycle-invariant checks after every step of scripted + random games; parity/
  validity unit tests for the rebuild; determinism (same seed ⇒ same trajectory, C# vs committed expectations).

### 4.5 Frontend (all client-side, no server change)

- `snake.ts`: widen `mode` to include `'watch-cycle'`; `watchCycle()` as a near-clone of `watchAi()`; third
  button in `snake.html` (label, owner-decided 2026-07-16: **"Watch AI (Hamiltonian cycle)"**).
- `snake-director.ts`: strategy parameter selecting `chooseActionCycle` vs `chooseActionSearch` (one-line seam).
- `snake_solver.ts` regenerated from the `.pg` (never hand-edited). Renderer untouched (mode-agnostic).
- A full-board win at `WATCH_TICK_MS=120` takes ~10–20 min — consider a faster tick for this mode (owner call).
- *(Stretch, optional)* faint overlay drawing the current cycle — great for demoing *why* it never dies.

## 5. Results (measured — C#, shipped 177-dim net, 12×12, 50 eps, seed 1)

| config | wins | deaths | truncations | food (min) | steps-to-win | ms/move |
|---|---|---|---|---|---|---|
| M48.1 fixed cycle + shortcuts | **50/50** | 0 | 0 | 141.0 (141) | 2,902 | 1.47 |
| **M48.2 + per-food rebuild (SHIPPED)** | **50/50** | 0 | 0 | 141.0 (141) | **2,841** | 1.29 |

Every game reaches the maximum 141 food (board full). The safety gate passed outright; the **speed gate
(≥20% fewer steps-to-win) missed honestly at −2.1%**: with a full-board cycle and unrestricted early-game
shortcuts, the fixed cycle already approaches the food near-directly, and in the late game — where most steps
are spent — rebuilds rarely succeed under fragmentation, so both levers bind on the same (early) phase.

Shortcut-cutoff sweep (`ShortcutMinFree`, 20 eps; safety is cutoff-independent — all configs 100% wins as the
ordering-invariant proof predicts): **late-game shortcuts hurt both modes** (half-board 2,805/2,917 rebuild/no;
24 free 3,257/3,789; 8 free 3,495/4,308) — greedy skipping late in the game wastes laps, so Tapsell's classic
half-board cutoff is confirmed and kept as the default.

## 6. Non-goals

- Retraining, new observation, new net — out of scope (unchanged from M34).
- Beating the cell-tree benchmark numbers — the gate is *never dying*, not optimal steps-to-win.
- Touching the existing "Watch AI" mode, human play (`snake-logic.ts`), or the renderer.
- Full-Hamiltonian completion per food (rejected, §3).

## 7. Risks

- **Off-cycle food livelock: designed out** (was this PRD's main risk). The rebuild commits only full-board
  cycles (§4.3 step 5), so the food always sits on the cycle and plain cycle-following always reaches it; a
  failed rebuild merely costs speed, never progress. (Implementation note: the first partial-coverage draft
  livelocked exactly as predicted — an 8×8 full-game test hit the step ceiling — which is what forced the
  hardened criterion.)
- **Transpiler pitfalls:** same as M34 §9 — typed records for list elements, no branch-typed locals; the 0.6.0+
  MSBuild package has fixed the incremental-prelude flake.
- **Rebuild cost:** everything is O(cells)–O(cells²) on 144 cells (µs–ms); the net stays a per-root ranking as
  in M34, so the 120 ms tick budget is comfortable.

## 8. Untouched

`snake-net.ckpt`, `snake-net.ts`, `snake-logic.ts` (human play), `snake-renderer.ts`, the existing
`chooseActionSearch` path and "Watch AI" button, all training/campaign code, all server code (there is none for
snake).
