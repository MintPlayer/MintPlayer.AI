# Snake — net-guided look-ahead search (client-side) — PRD

**Status:** in progress · 2026-07-10 · branch `m34-snake-search` (off `master`)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M34 · **Depends on:** M33 (Snake single-source `.pg` + fully client-side AI)

## 1. Problem

The shipped Snake demo (M33) plays with a **masked-greedy one-step** policy over the trained 177-dim dueling-Q net.
That policy is stuck at **~50 food on 12×12** — and PLAN M27 already established this is a **structural ceiling of a
reactive learned policy, not a training shortfall**: capacity (128→256), features, reward shaping and horizon were all
swept and all plateaued at ~50. A reactive net cannot avoid walking into a region it can no longer fit its body into,
because that trap forms several moves ahead of the one-step decision.

The observation *already* carries the anti-trap signal M27 added — per-move `reachableFreeSpace / cells` (4 inputs) and
a 1-ply flood-fill **shield** in the action mask — so "add a reachability input" is largely already done. The missing
piece is **planning more than one ply ahead** with that signal.

## 2. Goal & success criteria

Make the demo Snake **strong** while keeping it the trained model playing (the net stays in the decision) and fully
client-side (zero server inference, single-source `.pg`).

- **Gate:** food@12 (12×12, ≥ 12-ep mean) **markedly past the ~50 reactive plateau** — measured **≈ 70** (per-episode
  variance is high, ~50–88, since one bad trap ends a game). Stretch: a clean 100.
- **No retraining, no observation change** — the existing shipped `snake-net.ckpt` is reused verbatim.
- **Single-source:** the search lives once in `snake_solver.pg` and transpiles to C# (training/eval) **and** the browser
  TS director, byte-identically.
- **Browser cadence stays watchable** — one AI move per visible tick; the planner must fit that budget.

## 3. Key decision — port the idea, don't merge the branch

PR **#11** (`snake-ray-obs`) already proved the lever: **net-guided multi-ply look-ahead → food@12 ~50 → ~78.6** (3.7×
the original 21). But PR #11 **predates the M32/M33 Polyglot + client-side rewrite** — it is a hand-written C#
`SnakeSearchAgent` driving a **server-side** `SnakeController.Live`, plus a 39-dim ray-cast observation. It shows
`CONFLICTING` against `master` and cannot be merged or cleanly cherry-picked (it resurrects deleted server files and a
different observation).

**Decision:** take PR #11 as *inspiration* and **re-implement the search inside the single-source `snake_solver.pg`**,
on top of master's current 177-dim net and the `reachableFreeSpace` flood-fill it already ships. The net is kept as the
**leaf evaluator** (PR #11's finding: the survival term carries the search, the net is a marginal tiebreak — but keeping
it in the loop keeps the demo a genuine "watch the trained model" showcase).

## 4. Design (`snake_solver.pg` → C# + TS)

`chooseActionSearch(net, maxDepth, beamWidth, foodWeight, trapPenalty, netWeight, spaceWeight, foodDistWeight)`:

- **Receding-horizon beam search.** From the live state, simulate every legal (non-reversal) line to `maxDepth` plies
  on **cloned** envs, keeping the best `beamWidth` survivors per ply; play the first move of the best-scoring line and
  re-plan next tick. Snake is deterministic between food, so the look-ahead is exact.
- **Leaf score** (mirrors PR #11's `LeafScore`): a board-full win dominates; a death is dominated but ranked to prefer a
  *later* death; a survived leaf = `foodGained·FoodWeight − TrapPenalty·[reachable < length] + freeSpaceAhead·SpaceWeight
  + maxQ·NetWeight − headФoodDist·FoodDistWeight`. `freeSpaceAhead` is the max `reachableFreeSpace` over the 4 next-head
  cells — the exact survivability term. The net forward is **skipped entirely when `NetWeight == 0`**.
- **Reuses what M33 already shipped:** `reachableFreeSpace` (= PR #11's `FreeSpaceAhead`), deterministic dynamics, the
  177-dim observation, `PgSnakeNet`.
- **RNG-free simulation:** when a simulated line eats, food is respawned at the **first free cell** (`simSpawnFood`), not
  a random one — the single source keeps RNG with the caller, so the search must be deterministic to stay byte-identical
  across C#/TS. The fictional food cell barely matters: survival scoring drives the search and every tick re-plans.
- **Live env runs with `safeMask: false`** — the planner's multi-ply survival scoring supersedes the reactive 1-ply
  shield, and the returned move is always a non-reversal (hence always legal).

**Public surface.** The internal transpiled `PgSnakeNet` is exposed only through the `SnakeEnv` facade:
`LoadSearchNet(Stream)` (a C# twin of the browser's `snake-net.ts` parser — same RLNC/`dueling-q` bytes) +
`ChooseActionSearch(SnakeSearchConfig)`.

## 5. Results (measured — C#, shipped 177-dim net, **no retraining**)

food@12 on 12×12 (mean, high variance — min/max in parens); greedy baseline ≈ 50 (M27):

| config | food@12 | latency | verdict |
|---|---|---|---|
| pure-survival — d12/b16, net 0 | **70.7** (51–88, 12 eps) | 10.7 ms/move | strongest; net unused |
| **net-tiebreak — d12/b16, net 50 (SHIPPED)** | **≈ 70** | ~11 ms/move | ~= pure survival; keeps the net in the loop as a true near-tie breaker |
| net-tiebreak — d12/b16, net 500 | 67.8 (52–81, 12 eps) | 10.6 ms/move | a heavy net weight overrides survival moves and slightly hurts |
| net-guided *per node* — d10/b16, net 500 | 74.0 (2 eps) | **89 ms/move** | rejected: ~9× cost, no strength gain (the survival term carries it) |
| pure-survival — d20/b32, net 0 | 66.2 (62–70, 4 eps) | 38 ms/move | rejected: deeper/wider MISRANKS under beam pruning → weaker + slower |
| _(reference)_ PR #11 net-guided d20/b32 | ~78.6 | server-side | the original pre-Polyglot result (39-dim ray obs) |

All configs run the **identical** `.pg` search, so the browser reproduces them byte-for-byte. **Findings:** (1) depth
has a sweet spot (~12) — deeper misranks; (2) evaluating the net at every node buys no strength for ~9× the latency, so
the net is demoted to a cheap root-move tiebreak with a small weight. The search — not the net — is the strength lever.

## 6. Client integration

`snake-director.ts`: swap the greedy `chooseAction(net)` for `chooseActionSearch(net, …)`, drive the live core with
`safeMask: false`, and tune `SEARCH_DEPTH`/`SEARCH_BEAM` for a watchable move cadence. The shipped `snake-net.ckpt` and
`snake-net.ts` parser are unchanged; `snake_solver.ts` is regenerated from the `.pg` (never hand-edited).

## 7. Non-goals

- **Retraining / a new net / a new observation** — out of scope; the reactive net cannot beat ~50 regardless (M27), and
  the search delivers the strength on top of the existing net.
- **A guaranteed clean 100** — see §8.

## 8. Honest ceiling & follow-up

The residual cap (~80) is self-traps that form *beyond* the search horizon. A clean **100** (filling ~70% of the board)
needs a **tail-reachability survival invariant** (guarantee the head can always reach its own tail → follow it
indefinitely) and/or explicit **Hamiltonian-style endgame** play. Proposed as the next milestone, not built here.

## 9. Transpiler findings (MintPlayer.Polyglot 0.3.1)

Two issues surfaced; both were worked around from the `.pg` side.

- **TS `any`-inference (shaped the search).** The transpiler drops type annotations on local `List` variables (emits
  `let x = []`). A first attempt used **parallel `List` frontiers** (`List<PgSnakeEnv>` + `List<i32>` + `List<f64>`);
  because `childFirst` derived from `beamFirst[n]` and `beamFirst` was reassigned from a list holding `childFirst`,
  TS strict-mode saw a **circular `any`** and failed to compile (`TS7022`/`TS7034`). Fix: use a **`record
  PgSnakeBeamNode`** as the frontier element (like FruitCake's `PgPlyResult`) — the typed list element breaks the
  cycle. Also compute the net's `rootQ` unconditionally (typed by `net.forward`) rather than in a branch.
- **Incremental-rebuild prelude collision (build flake, not shipping).** In a multi-`.pg` project, an *incremental*
  rebuild after editing a `.pg` intermittently emits one unit in "standalone" mode (duplicate global
  `Option/Some/None` prelude + a **non-`partial`** `PolyglotProgram`) → `CS0260`/`CS0101`/`CS8863`. Root cause: the CLI
  doesn't clean its `--out` dir, so a stale `__polyglot_prelude.cs` collides on re-transpile; the target's incremental
  gate then caches the bad `.cs`. **Clean/CI builds are unaffected** (verified). Dev workaround:
  `rm obj/*/net10.0/polyglot/*.cs` then rebuild. (This is independent of the record above — records build fine cleanly,
  as FruitCake proves.)

Full repro + fix ideas: `docs/prd/polyglot-pilot/POLYGLOT_TOPLEVEL_RECORD_BUG.md`.
