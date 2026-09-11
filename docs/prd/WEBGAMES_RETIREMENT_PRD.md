# WebGames retirement + Block Dude / Lunar Lockout migration — PRD

**Status:** planned 2026-09-11 · **Milestone:** M58 · **Branch:** `m58-webgames-retirement`
**Goal:** retire `C:\Repos\WebGames` (and the two salvageable corners of `C:\Repos\Spelletjes`) by
landing everything worth keeping in this repo, so both source repos can be deleted without loss.

---

## 1. Why

The owner wants `C:\Repos\WebGames` gone. This repo already hosts 11 playable games, most backed by a
trained model, so the question is only: *what would deleting WebGames actually destroy?*

Answer, from the audit in §2: **three games, not six** — and one of them (Tic Tac Toe) was deliberately
dropped (§9), so **two** are actually migrated. The other three WebGames projects are strictly weaker
duplicates of what is already here.

Two premises going in turned out to be wrong, and they change the plan materially:

| Premise | Reality |
|---|---|
| "Block Dude is still fully blank" | **False.** `C:\Repos\WebGames\BlockDude\Scripts\game.ts` is 504 lines of working TypeScript (movement, climbing, pick up / place, gravity, canvas render) with 10 ASCII levels in `levels.ts`. |
| "Migrate Block Dude and Lunar Lockout from `C:\Repos\Spelletjes`" | **Half false.** Spelletjes Block Dude is real (619-line `SpeelVeld.cs`) but ships **zero level files**. Spelletjes **Lunar Lockout has 0 lines of game logic** — a 64-line WinForms scaffold whose only live code draws one rocket through a GDI+ colour-remap table and pops a `MessageBox` inside `Paint`. There is nothing there to migrate. |

**Consequence:** the two games draw from *different* sources than assumed — see §4.1 and §5.1.

### 1.1 Risk to act on before anything else

In `C:\Repos\WebGames`, `git status` reports `BlockDude/`, `LunarLockout/`, `RushHour/` and `TicTacToe/`
as **untracked** — only Game2048 and Rubiksolver are committed. Four of the six games exist *only in the
working tree*. Deleting that folder today is irrecoverable. **M58.0 (§7) is a backup commit, and it lands
before any other work.**

---

## 2. Audit results

### 2.1 WebGames — all six are real, none are blank

Uniform architecture, no shared code: an ASP.NET Core `Microsoft.NET.Sdk.Web` project on net10.0 used as
a **static host only** (8–10 line `Program.cs`, `UseStaticFiles` + `MapFallbackToFile`), with the game in
hand-written TypeScript (`Scripts/*.ts` → `wwwroot/js` via `Microsoft.TypeScript.MSBuild` 5.7.3) and SCSS
(`DartSassBuilder`). No Angular, no Blazor. Rubiksolver is the only one with real server code.

| Project | ~LOC (src) | Verdict | Core files |
|---|---|---|---|
| BlockDude | 504 TS + 103 levels + 216 SCSS | real | `Scripts/game.ts`, `Scripts/levels.ts` |
| Game2048 | 773 TS (7 modules) + 724 SCSS | real (Cirulli port) | `Scripts/game_manager.ts`, `grid.ts`, `tile.ts`, … |
| LunarLockout | 485 TS + 180 levels + 312 SCSS | real | `Scripts/game.ts`, `Scripts/levels.ts` |
| Rubiksolver | ~3000 C# + ~960 TS + 436 SCSS | real (biggest) | `Kociemba/*.cs`, `Services/CubeSolver.cs`, `Scripts/rubiksCube.ts` |
| RushHour | 337 TS + **726 TS levels** + 257 SCSS | real | `Scripts/app.ts`, `Scripts/levels.ts` |
| TicTacToe | 158 TS + 183 SCSS | real but minimal, **no AI** (hotseat only) | `Scripts/game.ts` |

### 2.2 This repo — what already exists

11 Angular routes (`src/RLDemo.Web/ClientApp/src/app/app.routes.ts`): `rushhour`, `2048`, `cube`, `snake`,
`mountaincar`, `fruitcake`, `crazyfruits`, `tetris`, `chess`, `draughts`, plus `gallery`. 21 committed LFS
checkpoints. Seven games run **100% in the browser** via a Polyglot `.pg` single source; only Rush Hour,
2048 and Cube are server-authoritative.

Confirmed absent, repo-wide case-insensitive grep: **Block Dude** (zero hits), **Lunar Lockout** (`lunar`
appears once, in `docs/prd/PRD.md:72`, as an explicit *non-goal* — and that is about LunarLander, a
different game), **Tic Tac Toe** (two aspirational doc mentions only).

### 2.3 The retirement matrix

| WebGames project | Covered here? | Action |
|---|---|---|
| Game2048 | ✅ `/2048` — classic Cirulli engine + n-tuple agent + expectimax, `models/2048.ntuple.ckpt` | **delete**, nothing to salvage |
| Rubiksolver | ✅ `/cube` — Kociemba *plus* DAVI, policy-beam and DQN solvers, 5 checkpoints | **delete**, this repo's is strictly stronger |
| RushHour | ✅ `/rushhour` — env, solver, oracle, imitation policy, 79-level deck | **delete** after the level harvest in §6.1 |
| TicTacToe | ❌ absent | **dropped** (D3) — see §9 |
| BlockDude | ❌ absent | **migrate** — §4 |
| LunarLockout | ❌ absent | **migrate** — §5 |

---

## 3. Decisions — resolved

All settled with the owner on 2026-09-11. The governing instruction for judgement calls was: **"pick the
most resilient approach, not the cheapest."** Where a choice below looks more expensive than necessary,
that is why.

| # | Decision | Outcome |
|---|---|---|
| **D1** | Block Dude rules: bug-compatible or corrected? | **Corrected — fix all four defects** (§4.3). Cells may hold multiple entities; a door is entered if *any* entity in the target cell is a door. |
| **D2** | AI scope | **Two-phase: imitation from an exact BFS oracle, then expert iteration on boards the oracle cannot label** (§8.1, §8.1b). Not a scripted solver, not DQN, not scramble-reversal. |
| **D3** | Tic Tac Toe | **Dropped** — recorded in §9 as deliberately not migrated. |
| **D4** | Level content | **12 authored Block Dude levels, 16 Lunar Lockout**, all with BFS-verified optimal counts, all **held out of training** as the gate set. |
| **D5** | Training-level source | **Heightmap generator + oracle rejection filter**, with bounded shelves (§4.4a). |
| **D6** | Ship gate | Policy-guided A* solves all authored levels at ≤1.25× optimal, **and** ≥90% of 200 generated hold-out boards solved greedily (§8.4). |
| **D7** | Lunar Lockout rules | **Verified against the published ThinkFun rules** (§5.2) — the WebGames implementation is faithful. BFS then re-verifies all 16 authored `minMoves`. |
| **D8** | Implementation order | **Lunar Lockout end-to-end first**, then Block Dude (§7). |

### 3.1 Why D2 changed from the first draft

This PRD originally recommended a scripted BFS solver and put RL explicitly out of scope, arguing that
search is provably optimal where training could only approximate it. The owner overrode it, and the
override is right for a reason the draft under-weighted: **in this repo the SDK is the deliverable and the
games are showcases.** A game that exercises `.pg` → C# campaign → TS browser inference end-to-end is worth
more than a game that merely plays well. The `.pg` single source is the point: one engine definition, C# for
the training campaign, TypeScript for the Angular app.

The BFS solver did not disappear — **imitation learning requires it** as the label source, which is exactly
the Rush Hour pattern (`RushHourOracle` + `RushHourImitationCampaign`). So D2 = RL *contains* the original
recommendation rather than replacing it.

---

## 4. Block Dude

### 4.1 Sources — take rules from one repo, levels from the other

| Need | Source | Why |
|---|---|---|
| Rules | `C:\Repos\Spelletjes\Block Dude\SpeelVeld.cs` (619 lines) + `Element.cs` | The complete, authentic rule set incl. carry semantics, climb preconditions and gravity. Fully documented in §4.2 — **the port does not need to re-read the C#**. |
| Level content | `C:\Repos\WebGames\BlockDude\Scripts\levels.ts` (10 ASCII levels) | Spelletjes ships **none** (levels came from an `OpenFileDialog`, and its loader's multi-level guard is inverted so it can only ever read a single-level file). |
| Graphics | **neither** | Owner rejected the WebGames visuals. Fresh art direction in §4.6. |

### 4.2 Rules specification (derived from `SpeelVeld.cs`; authoritative for the port)

**Coordinates.** 1-based, **Y-up** — cell `(1,1)` is bottom-left. Board size is `width × height`. Nothing
exists implicitly: no floor, no boundary walls unless placed.

**Entities** (`Element.cs:75-81`): `Petie` (player, 0) · `Deur` (door/exit, 1) · `Steen` (immovable wall,
2) · `Blok` (carryable block, 3). Player carries a facing, `Links` | `Rechts`.

| entity | solid? | climbable? | carryable? |
|---|---|---|---|
| Steen | yes | yes | **no** |
| Blok | yes | yes | yes |
| Deur | no (passable; win trigger) | no | no |
| Petie | yes (supports a fall) | no | no |

**Gravity** (`Element.cs:24-38`) — instantaneous and iterative: step down while the cell below is empty.
- "Supported" = *any* element directly below, **including another Petie**.
- A **Petie falls through a Deur**; a **Blok resting on a Deur is supported**.
- Applied **only** after a left/right walk and after a block is dropped. **Never after a climb**, and
  there is no global settle pass.
- With no floor an entity falls off-board and stays there; there is no death or reset.

**Moves** — one keypress = one discrete step (`SpeelVeld.cs:127-384`).

*Left / Right:*
1. Set facing to that direction — **turning costs a move even if the move is then blocked**.
2. Blocked by the board edge → stop (facing has already changed).
3. Target cell empty → move, then apply gravity.
4. Target cell holds a Deur → move there, apply gravity, **fire level-complete**.
5. Otherwise (Steen / Blok / Petie) → no move.
6. Carried-block follow-up, which runs **even when the move was blocked**: if the cell diagonally
   forward-and-up from the *old* position is occupied, the carried block falls from where it is and is
   **knocked out of your hands**; otherwise it is placed directly above the player's new position.

*Up = climb:* with `d = -1` for Links, `+1` for Rechts —
1. `(x+d, y)` must be occupied (something to climb).
2. `(x, y+1)` must be empty **or** be exactly the carried block (headroom).
3. That neighbour must be a **Blok or Steen** (you cannot climb a Deur or a Petie).
4. Not carrying: destination `(x+d, y+1)` must be empty, or hold a Deur or Petie.
5. Carrying: additionally `(x+d, y+2)` must be empty; player → `(x+d, y+1)`, block → directly above.
6. **No gravity after a climb**, even if the destination is unsupported.
7. If the new position holds a Deur → **level complete**.

*Down = pick up / put down:*
- Not carrying → **pick up**: `(x, y+1)` must be empty; `(x+d, y)` must be **strictly a Blok** (never a
  Steen); `(x+d, y+1)` must be empty or a Deur. The block moves onto the player's head. No gravity.
- Carrying → **put down**: if `(x+d, y+1)` is empty the block moves there and then falls; otherwise
  nothing happens and you keep it.
- Note the asymmetry: you pick up from **beside you at your own level**, but drop **diagonally up-ahead**
  and let it fall.

**Win.** Fired from exactly three places: walking left into a door, walking right into a door, and
climbing onto a door. **No requirement to be empty-handed.** No move limit.

### 4.3 The four defects (input to decision D1)

1. **Right-wall off-by-one** (`SpeelVeld.cs:208`) — bound is `x+1 >= width`, so the rightmost reachable
   column is `width-1` while left reaches column 1. The last column is unreachable.
2. **Asymmetric door detection** — left uses an any-match scan (`:187`), right uses first-element-only
   (`:215`). Since multiple entities may share a cell and lookup returns the first in *insertion order*, a
   door can be enterable from one side and not the other.
3. **`return` vs `continue` in the climb branch** (`:245` vs `:276`) — on the right, a failed climb aborts
   the whole entity loop instead of skipping one entity.
4. **Inverted multi-level guard** (`:548`) — `if (lines.Count() > level) return;` bails out when the file
   has *more* lines than the requested index, so a multi-level file can never load.
Under D1 = corrected, fix all four.

**Not a defect — a vacuous non-issue (recorded so it isn't "fixed" later).** Gravity is never applied after
a climb, which *looks* like an omission next to the walk and drop paths. It has no observable effect. The
climb requires `(x±d, y)` to be a Blok or Steen and lands the player at `(x±d, y+1)` — **directly on top of
the cell whose solidity the climb just required** — so the player is supported by construction and a gravity
pass there can never find anything to do. The carried block likewise lands on the player's head, supported
by the player. Applying gravity after a climb would be behaviourally identical; leave the call out (it
matches the original and costs nothing), but do not record it as preserved-bug-compatibility.

### 4.4 Shipped level content (the gate set)

Port the 10 WebGames ASCII levels (`W`=wall, `B`=block, `P`=player, `D`=door, `.`=empty — a good authoring
format, keep it), re-verify each is solvable under the *corrected* rules with the oracle (§4.5), and author
2 more for **12 total**. Store as a committed JSON/TS asset alongside the component, each level carrying
`{ name, grid, optimalMoves }` with `optimalMoves` from BFS, mirroring how the Rush Hour deck carries a
BFS-computed `OptimalMoves` (`src/RLDemo.Web/Services/RushHourDeckStore.cs`).

**These 12 are held out of training entirely** — the generator must be structurally incapable of emitting
them, exactly as `RushHourImitationCampaign` walls off ThinkFun cards 1/38/39/40.

### 4.4a Training-level generator (D5)

Measured fact that makes this cheap: **Block Dude terrain is a 1D height profile.** Excluding the ceiling
row, 8 of the 10 WebGames levels have **zero** overhangs; L9 and L10 have exactly two cells each (one small
shelf). So a generator is a per-column `height[x]` array, a door column, and block columns — and every board
it emits is gravity-consistent by construction, which is what made naive random 2D grids collapse.

Pipeline, mirroring `RushHourGenerator`: sample a height profile → place the door on a ledge → scatter N
blocks → run the oracle → **accept only** boards whose optimal length lands in a target band *and* that
require at least one carry.

Two requirements that are not optional:
- **Bounded shelves (0–3 cells).** A pure heightmap generator cannot produce overhangs, but L9/L10 have
  them — so without this the shipped levels are partly out-of-distribution relative to training, and the
  failure shows up only at gate time as a net that solves generated boards and stumbles on level 9.
- **Log the rejection rate, not just the accepted count.** The oracle cap (below) means deep boards yield
  no labels; unlogged, that silently biases training toward shallow puzzles.

### 4.5 The oracle (exact supervision)

State = `(playerX, playerY, facing, carrying, block positions)`; Steen and Deur are static. Branch factor 4.

**The Rush Hour oracle's shortcut does not transfer.** `RushHourOracle` gets its reverse graph for free —
its own comment says *"sliding moves are reversible, so the graph is undirected"* — so it can run a
multi-source backward BFS from all solved states. Block Dude gravity is **irreversible** and a Lunar
Lockout slide-to-contact has no general inverse. So both oracles must materialise the forward reachable
graph **with explicit edges**, reverse them in memory, then multi-source BFS from the solved states. Same
labels (optimal-action mask + distance-to-goal for every reachable state), more memory.

**Cap and fail loudly.** `RushHourOracle` returns null past `MaxStatesPerConfig = 150_000`; do the same, but
a board that exceeds the cap must be **logged as a rejection**, never silently dropped.

Supervision shape follows `RushHourImitationCampaign`: soft cross-entropy over **all** equally-optimal
actions (never a single arbitrary representative — that flattens the policy) plus Huber on
distance-to-goal, on a DAgger mix of on-policy and stratified samples.

### 4.5a State encoding (resilient, not cheapest)

Block Dude has **no i32-sized key**. Player position needs ~7 bits, facing 1, carrying 1, and five block
positions over ~108 interior cells need ~27 bits as a combinatorial rank — ~36 bits total.

A per-column *block count* encoding would compress this to ~24 bits and fit one `i32`, because given a fixed
terrain profile a block's height is determined by what is stacked beneath it. **Rejected deliberately:** it
is correct only while no level has an overhang, and L9/L10 already violate that. Cheap but brittle.

**Decision: two `i32` words + a hand-rolled open-addressing `IntSet`** storing both words for exact
comparison. See §8.3 for why this is the only shape available in `.pg`, and for the `i64`/`bigint` trap that
rules out a single 64-bit key.

### 4.6 Visual design

Full spec in §10. Owner constraint: **do not copy the WebGames graphics** — target **primitive and sober**.

---

## 5. Lunar Lockout

### 5.1 Source — WebGames only

`C:\Repos\Spelletjes\Lunar Lockout` contains **no game** (0 lines of logic; see §1). The only working
implementation is `C:\Repos\WebGames\LunarLockout\Scripts\game.ts` (485 lines) + `levels.ts` (180 lines).
Port the rules and levels from there; draw the graphics fresh (§10).

The one reusable *idea* from Spelletjes is its recolour-from-one-master approach (a single rocket sprite
remapped per robot). On the web that becomes an SVG/Canvas shape with a variable fill — **do not** reuse
the PNGs with a hue-rotate filter: `ImageAttributes.SetRemapTable` does exact-colour replacement, so the
sprite is flat-colour by necessity and a hue filter will not reproduce it.

### 5.2 Rules

5×5 grid. Each move slides a robot orthogonally until it **collides with another robot**, stopping
adjacent to it. A robot with no blocker in that direction **cannot move at all** — so pieces can never
leave the grid. Win when the **target robot rests on the centre cell**. The existing implementation has
`calculateLandingPosition` for the slide, an **undo stack**, a move counter, and click + keyboard
selection — all worth keeping.

**✅ Verified against the published rules (D7, 2026-09-11).** The rules above are not merely asserted by
the implementation — they match ThinkFun's own instructions and the secondary literature:

| published rule | implementation |
|---|---|
| 5×5 grid | `gridSize = 5` |
| a robot may slide only if **another robot lies in that direction** | `calculateLandingPosition` returns **`null`** unless `foundBlocker` — so a legal move always ends against a robot |
| it slides until it touches the first robot it meets | `while` walk, stopping one cell before the blocker |
| borders do **not** stop a robot, and crossing one is not allowed | hitting a bound breaks the loop with `foundBlocker = false` → move refused, so an edge can never be used as a backstop |
| win by moving the **red/target** robot to the centre | `selectedRobot.isMain && newPos.x === 2 && newPos.y === 2` (0-indexed centre = the sources' 1-indexed (3,3)) |

The open sub-question — *must the target be alone on centre?* — **dissolves**: robots are strictly one per
cell, so "alone" is automatic, and a non-target robot parked on centre simply acts as a blocker, which is
canonical rather than an accident of the code.

Sources: [ThinkFun Lunar Landing instructions (PDF)](https://legacy.thinkfun.com/wp-content/uploads/2017/01/ALL-Lunar-6802-Instructions-Updated.pdf)
· [SIAM Review](https://epubs.siam.org/doi/pdf/10.1137/S003614450139517)
· [CodinGame exercise](https://www.codingame.com/training/medium/lunar-lockout)

### 5.3 Level content and oracle

16 levels exist, typed `{ name, grid: string[], minMoves }`, with `minMoves` **authored by hand and never
verified**. The oracle is exhaustive here — ≤5 robots on 25 cells enumerates completely, and the state key
fits one `i32` without cleverness — so BFS **re-verifies every authored `minMoves`** as a 16-case regression
suite.

**A mismatch must not be silently "fixed".** Both the rules and the `minMoves` came from the same unverified
source, so a disagreement cannot say which side is wrong. Any mismatch **blocks that level from the training
set and from shipping** until the owner adjudicates — otherwise a rules bug gets laundered into the level
data and then trained into the net. Informative shapes: a level **unsolvable** under our rules is a
near-certain rule error; a *systematic* skew (every BFS result below the authored number) points at rules;
scattered one-off mismatches point at sloppy authoring.

These 16 are the held-out gate set (D4); the generator must not be able to emit them.

---

## 6. Salvage from the redundant three

### 6.1 Rush Hour levels — harvest, then dedupe

`C:\Repos\WebGames\RushHour\Scripts\levels.ts` holds **40 official ThinkFun levels** (ids 1–40, typed
`{ id, difficulty: 'beginner'|'intermediate'|'advanced'|'expert', vehicles[] }`).

This repo's committed deck (`src/RLDemo.Web/wwwroot/rushhour-deck.json`) **already holds 79 levels**, so
the marginal gain may be small or zero. Therefore: convert the 40 boards to `VehicleDto[]`, run them
through `RushHourDeckStore.Upsert` (which validates the board and computes BFS-optimal moves), **compare
against the existing 79 by canonical board signature, and add only genuinely new boards**. If the overlap
is total, record that and add nothing — the harvest is then just proof that deleting WebGames loses no
content. The ThinkFun `difficulty` tier is metadata the current deck lacks; carry it over if boards match.

### 6.2 Everything else

Nothing. 2048 is a Cirulli port this repo already has in `game-2048-classic.ts`. Rubiksolver's Kociemba is
already present under `Environments/RubiksCube/Kociemba/`, and it builds its pruning tables at runtime with
no data files to salvage. Tic Tac Toe is subject to D3.

**Deployment: nothing is live.** `C:\Repos\WebGames\.github\workflows\` ships `publish-2048.yml`,
`publish-rubiksolver.yml` and `deploy.yml` (a VPS deploy hook), but the owner confirmed on 2026-09-11 that
WebGames is **not deployed anywhere**. Nothing breaks when the repo goes; the workflows die with it and
need no migration.

---

## 7. Milestones

Single branch, single PR (`m58-webgames-retirement`), per repo convention.

**M58.0 — Preserve before anything else.** In `C:\Repos\WebGames`: `git add` the four untracked projects,
commit, push. Non-negotiable prerequisite; §1.1.

**Lunar Lockout goes first (D8), end-to-end.** It is the de-risked game — rules externally verified, state
space exhaustively enumerable, no dead ends, no gravity, no generator subtlety, single-`i32` key. Running it
all the way through proves every *unproven piece of infrastructure* (the first hand-rolled `IntSet` in a
`.pg`, the oracle→campaign label pipeline for irreversible moves, DI registration, the `.ckpt` TypeScript
parser, the gate harness) on the easy case. Block Dude's hard parts then land on plumbing that already
works, so a failure there is unambiguously the *game*, not the pipeline.

*Accepted cost:* Block Dude is the game the owner actually asked for, and it lands second — if M58 is
interrupted, the finished game is the less-wanted one.

**M58.1 — Lunar Lockout engine + oracle.** `Environments/LunarLockout/polyglot/lunarlockout_solver.pg`:
rules per §5.2, BFS oracle, `IntSet`. TS twin routed in `pgconfig.json` to
`app/lunar-lockout/lunarlockout_solver`. C# facade + `LunarLockoutEnv.cs`. **Gate: all 16 authored
`minMoves` re-verified** (§5.3); any mismatch surfaces for adjudication and blocks that level.

**M58.2 — Lunar Lockout campaign + net.** `Campaigns/LunarLockout/LunarLockoutImitationCampaign.cs`, DI
registration in `CampaignServiceCollectionExtensions.cs` + a `CampaignRegistrationTests.cs` case, Lab
`--game lunarlockout`, level generator, training run, checkpoint to `wwwroot/models/`.

**M58.3 — Lunar Lockout UI.** `app/lunar-lockout/` + route + nav + home card, renderer per §10.4,
`lunarlockout-net.ts` `.ckpt` parser + director with the stale-checkpoint guard.

**M58.4 — Block Dude engine + oracle.** `Environments/BlockDude/polyglot/blockdude_solver.pg` — rules per
§4.2 (corrected per D1), oracle per §4.5, two-word key + `IntSet` per §4.5a. Tests:
`BlockDudeEngineTests.cs` (every rule in §4.2 as a case, including fall-through-door and
block-supported-by-door), `BlockDudeParityTests.cs`, and an oracle test asserting `optimalMoves` per level.

**M58.5 — Block Dude generator + campaign + net (phase 1).** Heightmap generator with bounded shelves per
§4.4a, imitation campaign, Lab `--game blockdude`, training run to the §8.4 gate legs 1–3, checkpoint.

**M58.5a — Block Dude expert iteration (phase 2).** Per §8.1b: oversized-board generator tier, A*/beam
self-solving loop, distinct checkpoint ids (`blockdude.policy-xit*`) leaving the phase-1 net untouched, and
§8.4 gate leg 4. **This is the milestone that answers the imitation-ceiling objection** — if leg 4 does not
beat the phase-1 baseline, phase 2 has failed and the phase-1 net ships as the browser tier.

**M58.6 — Block Dude UI.** `app/block-dude/` + route + nav + home card, renderer per §10.3, keyboard +
on-screen mobile controls, **undo (mandatory — the game is irreversible)**, hint, level picker, Watch-AI.

**M58.7 — Rush Hour level harvest** per §6.1 (dedupe first; may be a no-op).

**M58.8 — Retire the repos.** Update `docs/ARCHITECTURE.md`, `docs/prd/PLAN.md` (M58 entry) and
`docs/ADDING_A_GAME.md`; then the owner deletes `C:\Repos\WebGames`. Leave `C:\Repos\Spelletjes` alone — it
still holds Rush Hour, `RushHour.Core`, its tests and the designer PRDs, none of which are in scope here.

**M58.9 — Fix the stale home card** noted in the audit: `app/home/home.ts:65` still says FruitCake has
"no AI (yet)", untrue since M32. One-line fix, lands in the same PR.

Test suites run **once**, at the end of M58.6 (after all engine, campaign and UI milestones), per the repo's
batch-tests-at-the-end rule. Intermediate milestones are verified by reading code and type-checking. The two
training runs are long jobs — background them and wait for the notification; never poll.

---

## 8. Design notes

### 8.1 Why imitation learning, and not DQN or teacher-free policy (supporting D2)

Both games are **sparse-reward planning** problems: perfect information, deterministic, small, with a single
terminal event and no per-step score. That is Rush Hour's shape, not Tetris's — and the repo already solved
it once, with imitation from an exact oracle.

- **DQN (the Tetris / Crazy Fruits machinery) — rejected.** Those games have dense per-step score. M54's
  finding was that γ-bootstrap went **0-for-2** on puzzle games and the fix was γ=0 **plus dense targets** —
  which a single terminal reward cannot supply. Confirmed by the owner: *"we're not chasing a high-score but
  a single end-goal/terminal-event."*
- **Scramble-reversal labelling (the `cube-policy` / EfficientCube label trick) — blocked.** It generates
  labels by *reversing* a scramble. Block Dude's gravity is lossy, so a solved state cannot be run
  backwards; likewise a Lunar Lockout slide has no general inverse. **Note the narrow scope of this:** only
  the *labelling* is blocked, not EfficientCube's actual winning mechanism (search distilled back into the
  net), which transfers fine — see §8.1b.
- **Imitation from a BFS oracle — chosen.** Exact optimal labels at every reachable depth, no reward
  shaping, no exploration problem.

### 8.1b The imitation ceiling, and why phase 2 exists

**The owner's objection (2026-09-11):** *"if we pick imitation-learning, the net will never be stronger than
the teacher"* — with a concrete in-repo precedent. The cube was first trained to imitate Kociemba
(`CubeImitationCampaign`, `--game cube`), which capped it; the pivot to `CubeEfficientCampaign`
(`--game cube-policy`, *"Teacher-FREE: the label is the cube's own scramble reversal (no Kociemba)"*) now
finds **shorter solutions than Kociemba**.

**Why the cap is vacuous for phase 1.** Kociemba is *suboptimal* — a two-phase solver, called in this repo
with `maxDepth: 22`, above the 20-move optimum. There was real headroom above that teacher. **BFS is exact
optimal**, so on boards the oracle can solve there is no shorter solution to find: the teacher is the
ceiling, not a ceiling below it. Matching it is the best achievable outcome, and §8.4's gate measures how
close inference gets (≤1.25× optimal) rather than asserting it.

**Why a phase 2 is still needed — and a correction.** An earlier draft of this PRD said teacher-free
learning was *"structurally blocked"* here because gravity is irreversible. That is true only of
EfficientCube's **specific label trick** (scramble reversal needs invertible moves). It is *not* the
mechanism that beat Kociemba: EfficientCube's labels are scramble reversals, which are **worse** than
Kociemba's solutions. It wins because **beam search at inference explores past the labels, and search is
then distilled back into the net.** That mechanism needs only *forward* search, so it transfers to Block
Dude intact.

**Phase 2 = expert iteration, on the axis where headroom actually exists: scale.**
1. Generate **oversized** boards (bigger grids, more blocks) where the oracle exceeds `MaxStates` and can
   label **nothing**.
2. Solve what the current net + A*/beam can solve on them.
3. Train on those solutions; repeat.

The result is a net that demonstrably does what its teacher cannot — measured in **board size**, not move
count. Keep it under **distinct checkpoint ids**, exactly as `CubeEfficientCampaign` does (*"so the imitation
net is never touched"*), and hold phase-1 weights as the fallback tier.

Where headroom genuinely does **not** exist: **Lunar Lockout**. ≤5 robots on 25 cells enumerates
exhaustively, so the oracle is optimal everywhere and there is no "beyond the oracle" regime to escape into.
Phase 2 is Block Dude only.

### 8.1a Irreversibility — the structural difference from Rush Hour

**Rush Hour has no dead ends**: every slide is reversible, so a myopic policy that mispredicts can always
back out. That is why reactive greedy play is a meaningful metric there.

**Block Dude is irreversible**: drop a block into a pit you then cannot climb out of and the level is
unsolvable *from that state* — nothing in the rules recovers it. So a greedy net that errs once can
permanently lose a level it otherwise "knows". Lunar Lockout sits in between: slides are not invertible, but
no move destroys reachability the way a misplaced block does.

Three consequences, all load-bearing:
1. The shipped artefact is **net-as-heuristic + small search**, like the chess and Tetris tiers — not pure
   argmax (§8.4).
2. The distance-to-goal head is not decoration; it is the A* heuristic.
3. **Undo is mandatory** in the Block Dude UI, for humans for the same reason.

### 8.2 Follow the client-side Polyglot pattern, not `ADDING_A_GAME.md`

`docs/ADDING_A_GAME.md` documents the **older server-side** path (env → `*ModelService` → controller) and
predates the client-side pattern; it is contradicted by M26/M32/M33. Every game since M32 is browser-only.
Both new games are small, deterministic and need no server — so: rules + solver in a `.pg` single source,
TS twin routed via `pgconfig.json`, zero controllers, zero per-viewer server cost. Reference implementation
to copy: Tetris (`Environments/Tetris/polyglot/tetris_solver.pg` + `app/tetris/`). Use
`constructor(...)`, never `init(...)` (Polyglot 0.9.9). Never edit a generated `*_solver.ts`.

M58 adds **two** `.ckpt` files to `src/RLDemo.Web/wwwroot/models/` (LFS-tracked automatically by
`.gitattributes`), one per game, each with a `<game>-net.ts` parser mirroring the C# reader and a
stale-checkpoint guard (input width ≠ observation width → scripted fallback).

### 8.3 Polyglot constraints — what the `.pg` oracle can and cannot use

Verified against Polyglot HEAD (post-P37). **No compiler change is required.**

The entire collection surface is `List<T>` — `count`, `add`, `clear`, `removeAll(pred)`, `removeAt(i)`, plus
indexing get/set and `for x in xs` — and `Array<T>`. **There is no dictionary, map, set or queue**, and
`Array<T>()` cannot be sized (to get N slots, `add` N times, as `snake_solver.pg` does).

The BFS recipe, all of it already-proven `.pg` surface:
- **Queue** = `List<i32>` + an integer **head cursor**. There is no `removeFirst`, so never dequeue — advance
  the cursor. Memory is monotonic: 10⁶ `i32` ≈ 4 MB in C#, ≈ 8 MB as a JS array.
- **Visited set** = a dense direct-indexed `List<bool>` where the key space packs (Snake's technique), or a
  hand-rolled **open-addressing `IntSet`** over parallel `List<i32>`s for sparse spaces. Never a
  `List<state>` + linear `contains` (Draughts' pattern) at this scale: 10⁶ × O(n) ≈ 10¹² ops.
- **Key = a packed integer.** Guaranteed bit-identical across C# and TS by the spec's reproducibility tier.

Three traps to avoid:
1. **Never key on a string.** `std.strings` has eight methods, `+` works only on `string + string`, and
   `<`/`>` on strings is a hard compile error — so there is no ordering and no map, and a string-keyed
   visited set degenerates to a linear scan.
2. **Avoid `i64` keys.** `i64`/`u64` lower to TypeScript **`bigint`** — correct but allocating and
   materially slower at 10⁶ probes. Stay ≤31 bits, or use two `i32` words (which is why §4.5a does).
3. **Never key on a record.** Record `==` is structural, but `List<T>`/array/class fields compare **by
   reference** — so `record State(board: List<i32>, …)` silently fails to compare structurally.

Two pieces of repo folklore are **stale and should not be propagated**: the "no `while` / no BFS queue"
constraint list (that came from a self-imposed style note in `chess_solver.pg`, never a language limit —
`while`, `do…while`, `break`/`continue`, early `return` and recursion are all available and exercised), and
the TS7022 workaround that makes `snake_solver.pg` use O(cells²) relaxation instead of a real queue — that
was Polyglot issue **#27, since fixed**, so a proper frontier queue is safe today.

### 8.4 The ship gate (D6)

Mirrors `RushHourImitationCampaign`, which evaluates held-out ThinkFun cards under **both** reactive play and
policy-guided A*, plus a random hold-out set. Per game:

1. **All authored levels** (12 Block Dude / 16 Lunar Lockout) solved by **policy-guided A*** — the net's
   distance head as the heuristic — within **1.25× optimal**.
2. **≥90% of 200 generated hold-out boards** solved by **greedy argmax**.
3. Report a **stuck rate** (episodes reaching an unsolvable state) as a first-class metric, not just
   solve-rate — it is the number that exposes §8.1a.
4. **Block Dude phase 2 only — the beyond-the-teacher leg:** solve-rate on a held-out set of **oversized
   boards the oracle cannot label at all** (it exceeds `MaxStates`). Phase 1's number here is the baseline;
   phase 2 must beat it. This is the metric that answers "is the net stronger than its teacher?" — the
   teacher scores **zero** on this set by construction.

A pure-argmax gate on (1) was considered and rejected: for an irreversible game it would likely never go
green, and forcing it would mean either far more training or shipping occasional unsolvable states.

---

## 9. Out of scope (genuinely not being done)

- **Tic Tac Toe — dropped (D3).** A solved, AI-less 158-line game. Perfect play is a draw from every
  position, so there is no skill ceiling to showcase and no puzzle to solve; a route backed by nothing would
  undercut the playground's pitch. Not deferred — deliberately discarded, which means the retirement is
  *not* literally lossless and that was the accepted trade.
- **A light theme for the app.** The app is dark-only today (§10.1). The new renderers carry their own light
  palettes, but introducing app-wide theming is not M58 work.
- **The per-column block-count state encoding.** Rejected on brittleness grounds, not cost — see §4.5a.
- **The Block Dude level designer.** Spelletjes has a full WinForms designer (`Form2` + context-menu
  placement). Not migrating it: the ASCII grid format is editable in any text editor, which is the whole
  point of keeping it.
- **Retiring `C:\Repos\Spelletjes`.** Only its Block Dude rules are consumed here. Rush Hour,
  `RushHour.Core` (the one clean, WinForms-free, unit-tested model in that solution), `RushHour.Rendering`
  and the designer PRDs stay put.
- **Spelletjes' other scaffolds** — GoGetter and Schuifpuzzel are empty shells (81 and 89 LOC, ctor only).
  Nothing to migrate; not new games for this playground either.
- **The `.rsh` legacy format** (BinaryFormatter + Rijndael with a hard-coded key). Deliberately not ported.

---

## 10. Art direction

Design register: **primitive and sober** — flat, geometric, minimal. Solid fills only. No gradients, gloss,
bevels, drop shadows, glow or bloom. Colour carries meaning and nothing else.

### 10.1 The house style, as measured

Canvas 2D is the house default for grid games (Snake, FruitCake, Tetris, Crazy Fruits, Rush Hour); DOM
appears only in 2048 (a deliberate port of Cirulli's CSS) and chess/draughts. Notable: **there is no SCSS
variable file and no CSS custom properties** beyond the Bootstrap re-point in `ClientApp/src/styles.scss:5-27`
— every game hard-codes hex. The shared set: page `#14171f`, text `#e6e8ee`, accent `#6ea8fe`, panels
`#1a1f2b`/`#232938`, borders `#2b3245`/`#3a4154`, muted `#aab2c5`→`#8b93a7`→`#5b6378`, semantic
`#4ade80`/`#f87171`/`#fbbf24`, snake green `#4caf82`. Radii 6/8/12px. Canvas text is always
`system-ui, sans-serif`.

**There are zero shadows and zero gradients in any app SCSS**, and depth is done with a single flat white
top-highlight strip in Tetris — which this design does **not** adopt, being the one piece of house gloss.

**⚠️ Scope note: the app is dark-only today.** No `prefers-color-scheme`, no `data-bs-theme`, no theme
toggle anywhere in `src/app`. In-game skins are renderer-level palette objects (`fruit-cake-fruits.ts:61-65`).
So the light palettes below live **inside the new renderers**, selected via
`matchMedia('(prefers-color-scheme: light)')` — they are not an app-wide theming feature, and introducing
one is not in M58's scope.

### 10.2 Why the WebGames visuals were rejected

Recorded so the mistakes aren't repeated: seven unrelated CSS-named hues from three eras (`#8B4513`
saddle-brown blocks, `#228B22` forest-green door, `#FFD700` gold) none of which exist in the app's token
set; decorative brick and "wood grain" texture that carries no information while terrain and blocks sit at
nearly the same *value*, so the board reads as mush; hard-coded pixel offsets (`py + 10`, `arc(…,8,…)`)
that only look right at one cell size; a stick figure whose sole facing cue is a ±2px eye dot — sub-pixel
on a phone, where facing is the single most important state in the game; no animation at all (every action
snaps, gravity teleports through a `while` loop); and a level-complete modal that yanks you onward on a
blind 2s `setTimeout`.

### 10.3 Block Dude

**Render tech.** Canvas 2D in a pure view module `app/block-dude/block-dude-render.ts` taking an immutable
snapshot. Logical space + `ctx.scale(cssW / LOGICAL_W)` per `tetris-render.ts`; DPR-scaled backing store per
`snake-renderer.ts:40-48`. `LOGICAL_W = cols * 48`, so `C = 48` in logical units. SVG rejected: no precedent
in the repo, and Canvas+DPR already covers the scaling need.

**Palette — 6 tokens, one meaning each.** A record mirroring `THEMES` in `fruit-cake-fruits.ts`:

| token | dark | light | meaning |
|---|---|---|---|
| `void_` | `#14171f` | `#f4f6fa` | nothing |
| `terrain` | `#2b3245` | `#c3cad8` | immovable |
| `block` | `#aab2c5` | `#5b6378` | carryable |
| `dude` | `#6ea8fe` | `#2563eb` | you |
| `exit` | `#4caf82` | `#2f8f66` | goal |
| `hair` | `#3a4154` | `#d3d9e4` | hairline |

Every value is an existing app token except `exit`, which is Snake's body green. No seventh colour, no
tints. The stage's CSS `background` must be set from `void_` so frame and canvas never disagree.

**Entities, in cell units (`C`).** Hairline `max(1, 0.03C)`, snapped to `round(px)+0.5`. **Radius 0
everywhere except the carryable block.**
- *Void:* `fillRect` the canvas, then a single `hair` lattice over the playable rect only — the
  `snake-renderer.ts:196-205` construction but at full opacity, because here the grid is load-bearing (you
  count cells to plan a climb). No parallax, nothing behind it.
- *Terrain:* solid full-bleed `fillRect`, **no inset, no per-tile outline**, so contiguous terrain fuses
  into one silhouetted mass; then one `hair` stroke along region *boundaries* only (where the 4-neighbour
  isn't terrain). A plain landscape, not 300 outlined boxes.
- *Carryable block:* `roundRect(+0.10C, +0.10C, 0.80C, 0.80C, 0.08C)` filled `block`, no outline. The only
  rounded, only inset element on the board — three independent cues (value, gap, radius) all say "loose".
- *The dude:* solid `dude`, flat, **5 rectangles, no face, no outline**; height `0.84C`, footprint `0.44C`,
  baseline `y*C + 0.96C`, mirrored about cell centre by facing `f = ±1`. Torso
  `rect(cx-0.22C, base-0.66C, 0.44C, 0.42C)`; head `rect(cx-0.15C+f*0.05C, base-0.84C, 0.30C, 0.20C)`
  shifted toward the facing side; **brow notch** — the facing cue, a plain rectangular prow
  `rect(cx+f*0.20C, base-0.80C, f*0.10C, 0.07C)`, readable at a 22px cell; two legs
  `rect(_, base-0.24C, 0.13C, 0.24C)`. *Idle* is fully static — no breathing, no bob. *Walking* is a
  two-frame leg swap, not a cycle (`p ∈ [0.25,0.75)` → leading leg `+f*0.07C`, trailing `-f*0.07C`).
  *Carrying* adds two straight arms `rect(_, base-0.80C, 0.09C, 0.16C)`, no elbow, and draws the held block
  as a normal carryable glyph at `(x, y-1)` — **in the same `block` colour**, so "what I hold is the same
  substance as what's on the floor" is literal.
- *Exit:* a **hollow aperture**, shape-coded so it never relies on hue. Clear to `void_`, stroke
  `rect(+0.18C, +0.14C, 0.64C, 0.86C)` in `exit` at `0.06C` **open at the bottom — three sides, a doorway
  not a box**, with one `exit` chevron pointing inward. Nothing filled, no handle, no glow.

**Readability.** Carryable ⟂ terrain is triple-coded (value + inset + radius); the exit is the only
*unfilled* element; the dude is the only chromatic mass and the only ~0.84C figure. An optional
`showLabels` toggle (precedent: `fruit-cake-render.ts:224-237`) stamps `B`/`E` glyphs — off by default.

**Animation — rAF glide, linear, no easing theatrics.** Structure copied from `SnakeTubeRenderer`:
`{prev, next, t0, durMs}`, `push()` re-kicks, and the loop **parks itself at `p ≥ 1`** so a resting board
costs zero frames (`snake-renderer.ts:86-98`). Level load / reset snap instead of gliding.

| action | duration | curve |
|---|---|---|
| walk one cell | 110 ms | linear in x |
| climb | 150 ms | **up-then-over, never diagonal** — 0–60 ms in y, 60–150 ms in x |
| step down | 110 ms | linear x, then gravity takes over |
| pick up / place | 90 ms | block lerps; arms appear at `p ≥ 0.5` |
| gravity fall, `n` cells | `70·√n` ms, cap 260 | `offset = n·p²` — real acceleration, no bounce, no squash |
| level complete | 260 ms | `void_` veil to 0.55 alpha + static text. **No auto-advance — a Next button.** |

Moves queued mid-glide buffer one deep rather than interrupting; the engine stays turn-based.

**Juice: deliberately nearly none.** Explicitly dropped — parallax, dust puffs, landing particles, exit
glow/pulse, bloom, screen shake, score popups, overshoot, idle animation, canvas hover effects. The only
inessential motion permitted is a 160 ms board fade on level load, so a level change doesn't flicker.

**Responsive / mobile.** Stage per `fruit-cake.scss:22-75` / `tetris.scss:32-93`: `width: min(760px, 100%)`,
`aspect-ratio` from level dims, 8px radius, `overflow: hidden`, canvas `touch-action: none`, fullscreen
rule. At 360px a 16-wide level is ~22 CSS px/cell — **so no control may be a single cell.** Gestures follow
`tetris.ts:227-273` (Pointer Events + `setPointerCapture`): tap left/right of the dude to step, hold-repeat
after 300 ms at 110 ms, swipe up ≥`1.5C` to grab/place. Plus a persistent 3-button row in the stage at
**56×56px** (≥44px WCAG), `#232938`/`#3a4154`/`#aab2c5`, and reset + fullscreen at 2.2rem top-right.
`screen-wake-lock.ts` engaged while a level is open.

**Accessibility.** All gameplay-critical pairs ≥ 4.5:1 (block-on-terrain ≈ 5.9:1, dude-on-void ≈ 8.0:1,
exit-on-void ≈ 6.5:1; light theme mirrors at 4.3:1 / 5.5:1); HUD text ≥ 7:1. Colour-blind: hue is
**redundant everywhere** — under deuteranopia dude and exit could converge, so the exit is the only hollow
three-sided chevron glyph and the dude the only tall figure. `prefers-reduced-motion` sets every duration
to **0 ms**, routing through the renderer's existing snap path — nothing becomes unplayable and no state is
conveyed by motion alone. Keyboard `←`/`→`/`↑`/`Space`/`R`/`Esc` in visible hint text per `tetris.ts:15-16`;
canvas gets `tabindex="0"` + `role="application"` + an `aria-label` board summary updated per move.

### 10.4 Lunar Lockout

Same module shape; `LOGICAL = 5 * 96 = 480` square, `C = 96`, stage `min(520px, 100%)`, `aspect-ratio: 1`,
8px radius.

**Palette — 5 tokens, reusing the Block Dude set:** `void_`, `hair`, `robot` = `#aab2c5`/`#5b6378` (helpers),
`target` = `#6ea8fe`/`#2563eb`, `goal` = `#4caf82`/`#2f8f66`. **No per-robot rainbow** — Rush Hour's
16-colour ramp is explicitly not adopted; identity comes from position, not hue.

- *Board:* `void_` fill, full-opacity `hair` 5×5 lattice, no cell fills, no border ornament.
- *Robots:* flat `arc(cx, cy, 0.34C)`, solid fill, no outline or shadow. The **target** is `target` **plus a
  `0.05C` `void_` inner ring at `r = 0.20C`** so it is shape-distinct, not merely hue-distinct. Selection is
  a `0.04C` body-colour ring at `r = 0.44C` — the same move as `rush-hour.ts:436-439`. Optional letter
  labels per `rush-hour.ts:446-449`.
- *Centre goal:* hollow inset `rect` stroked `goal` plus a `goal` cross — **plainly marked, never filled**,
  so a robot sitting on it stays fully visible inside the marking.
- *Slide-path hint (unobtrusive):* on selection only, a `setLineDash([5,7])` 1px `hair` line to each legal
  landing cell plus a hollow `hair` circle there. No arrowheads, no fill, no animation; cleared the instant
  the move commits.
- *Slide:* linear, **55 ms/cell clamped to 220 ms total**, same park-at-`p≥1` loop. No easing, no overshoot,
  no impact flash. Complete: 200 ms veil + static `Solved · N moves`. Reduced motion → 0 ms.
- *Mobile:* `touch-action: manipulation`, tap-to-select (a cell is ~70 CSS px at 360px, already above
  target), plus a 56×56 four-button d-pad per `rush-hour.html:121-132`; arrow keys on desktop.

### 10.5 Old assets: all four dropped — both games ship 100% code-drawn

| asset | verdict |
|---|---|
| `Spelletjes\Block Dude\Resources\Steen.png` (60×60, **268 bytes**) | **drop** — 268 bytes is a flat near-featureless fill; blurs at 96px, aliases at 22px, and the sober terrain spec is a `fillRect` |
| `Spelletjes\Lunar Lockout\Resources\rocket.png` (2400×2400, 226 KB) | **drop** — 25× over-render for a 96px cell, and a pictorial rocket is the decorated register this direction rejects |
| `…\Resources\rocketklein.png` (120×120, 8-bit colormap) | **drop** — a 256-colour indexed bitmap cannot be re-tinted per theme |
| `…\rocket.png` (500×500, **no alpha**) | **drop** — stray duplicate, and RGB without transparency is unusable over either background |

No other images exist in either project (both `.resx` files hold only `ResXFileRef` pointers to these four;
the `Form1.resx` files are stock designer resources). Zero assets migrate — which matches the FruitCake
precedent ("Asset-free: pure Canvas 2D", `fruit-cake-art.ts:7`), keeps the ClientApp asset-free apart from
`public/favicon.ico`, and needs no external downloads.

### 10.6 Files to create / copy from

New: `app/block-dude/{block-dude-render.ts, block-dude.ts, .html, .scss}` and
`app/lunar-lockout/{lunar-lockout-render.ts, …}`. Copy the rAF/snap/park structure from
`app/snake/snake-renderer.ts`; logical-space + `ctx.scale` + overlay text from `app/tetris/tetris-render.ts`;
stage/fullscreen SCSS from `app/fruit-cake/fruit-cake.scss:22-75` and `app/tetris/tetris.scss:32-93`;
pointer gestures from `app/tetris/tetris.ts:227-273`; DOM move buttons from
`app/rush-hour/rush-hour.html:121-132`.

---

## 11. Open questions

**None blocking.** All eight decisions are resolved (§3) and every question raised during planning closed:

| question | resolution |
|---|---|
| D1–D4, plus D5–D8 raised while planning | §3 decision table |
| Is anything deployed from WebGames? | **No** — owner, 2026-09-11 (§6.2) |
| Lunar Lockout rules unverified | **Verified** against ThinkFun's published instructions (§5.2) |
| Can a BFS with a visited set be written in `.pg`? | **Yes, no compiler change needed** (§8.3) |
| Is the Block Dude state space BFS-tractable? | Yes for the shipped 20×7 / ≤5-block levels; the oracle caps and logs rejections regardless (§4.5) |
| Is Block Dude terrain arbitrary 2D or a height profile? | **Height profile** — 0 overhangs in 8 of 10 levels, 2 cells each in L9/L10 (§4.4a) |

Two items need the owner's eye *during* implementation rather than before it:

1. **Any `minMoves` mismatch in Lunar Lockout** blocks that level pending adjudication (§5.3) — by
   construction it cannot be auto-resolved.
2. **Shared-occupancy cells** (D1) were chosen on my recommendation, not an explicit owner ruling. Cheap to
   revisit until `blockdude_solver.pg` exists; expensive after the net trains on it.
