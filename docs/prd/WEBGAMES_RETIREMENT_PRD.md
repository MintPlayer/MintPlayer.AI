# WebGames retirement + Block Dude / Lunar Lockout migration — PRD

**Status:** in progress, 2026-09-11 · **Milestone:** M58 · **Branch:** `m58-webgames-retirement`
**Goal:** retire `C:\Repos\WebGames` (and the two salvageable corners of `C:\Repos\Spelletjes`) by
landing everything worth keeping in this repo, so both source repos can be deleted without loss.

| Milestone | State |
|---|---|
| M58.0 preserve WebGames | ✅ committed and pushed |
| M58.1 Lunar Lockout engine + oracle | ✅ 12-level ladder, BFS-verified |
| M58.2 Lunar Lockout campaign | ⛔ dropped — the game is exhaustively solvable, so it ships the exact oracle rather than a trained net |
| M58.3 Lunar Lockout page | ✅ |
| M58.4 Block Dude engine + oracle | ✅ 11 original levels + 4 CSE bonus levels (§4.4b) |
| M58.5 Block Dude generator + campaign | ✅ trainable; no run completed yet |
| M58.5a Block Dude expert iteration | ⬜ not started |
| M58.6 Block Dude page | ✅ |
| M58.7 Rush Hour level harvest | ⛔ abandoned — source data does not decode (§6.1) |
| M58.8 retire the repos | 🟡 docs updated; deletion is the owner's to do |
| M58.9 stale home card | ✅ |
| M58.10 direct manipulation | ✅ both games |
| M58.11 responsive boards + canvas sizing | ✅ §12.4a |

### Where to pick this up

**Not done, in the order I would take them:**

1. **Run the full suite.** It has not run since the campaign work landed. Last known green was 597 tests
   (`dotnet test --filter "Category!=Slow"`); since then the Block Dude campaign, generator, curriculum, both
   pages and the canvas fixes have all landed untested as a whole.
2. **Tests for the two new pages.** Neither has any. Everything claimed about them rests on browser checks:
   Lunar Lockout aim/launch/solve, Block Dude walk/blocked/carry, Rush Hour drag counting per cell, and the
   390 px responsive pass. A renderer is awkward to unit-test, but the components' pure logic — aim resolution
   with hysteresis, the drag clamp and re-anchor, per-cell move counting — is not.
3. **Start a Block Dude training run.** The pipeline is complete and verified end to end on a short run (8192
   samples, 73% policy accuracy against a 25% random baseline, 15% board accept rate). Nothing has been trained
   to a gate yet, and no checkpoint is committed.
   `dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- --game blockdude --fresh
   --data data/bd-s1 --seed 1 --hours 9`
4. **Phase 2, expert iteration** (M58.5a, §8.1b) — the part that would let the net reach the big levels, and the
   answer to "the net will never beat its teacher".
5. **Delete `C:\Repos\WebGames`** — the owner's to do. Its four untracked projects were committed and pushed
   first (M58.0), and §6.1 / §5.3 establish that no level content is lost.

**Environment note:** the ASP.NET host was started by the assistant during this work and is running on port 5210.

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
| **D1** | Block Dude rules: bug-compatible or corrected? | **Corrected — fix all four defects** (§4.3). **Single-occupancy cells with one exception: the player may stand in the door cell**, which is how a level is won (owner ruling, superseding the earlier shared-occupancy recommendation). |
| **D2** | AI scope | **Two-phase: imitation from an exact BFS oracle, then expert iteration on boards the oracle cannot label** (§8.1, §8.1b). Not a scripted solver, not DQN, not scramble-reversal. |
| **D3** | Tic Tac Toe | **Dropped** — recorded in §9 as deliberately not migrated. |
| **D4** | Level content | **All 11 original Block Dude levels shipped as-is** (owner: *"Ship them all… I understand that they won't be AI solvable, that's fine"*), plus a **12-level generated Lunar Lockout ladder**. See §4.4 / §5.3. |
| **D5** | Training-level source | **Generated boards under a size curriculum** (§7.1). The original heightmap-only design is **dead** — see §4.4a. |
| **D6** | Ship gate | §8.4. Legs 1–3 as specified; **level 11 is a stretch benchmark, not an acceptance criterion** (§8.5). |
| **D7** | Lunar Lockout rules | **Verified against the published ThinkFun rules** (§5.2) — the WebGames implementation is faithful. Its *level pack* was not: 13 of 15 grids are unplayable (§5.3). |
| **D9** | Curriculum + reproducibility | Train small→large, and **any run must be re-startable from a blank slate** (owner). Design in §7.1. |
| **D10** | Level packs | **Canonical JSON, single source of truth** (owner): embedded in the Environments assembly for training, linked into `wwwroot/levels` for the browser. Same bytes both sides. |
| **D11** | Input model | **Direct manipulation, replacing select-then-press-a-button in BOTH games** (owner). Lunar Lockout = point-and-launch with a rotating rocket; Rush Hour = grab-and-slide with sub-cell motion. §12. |
| **D12** | Mouse vs touch | Branch the **behaviour** by input type, via ONE Pointer Events stream discriminated on `pointerType`, not separate `mousemove`/`touchstart` handlers (§12.2). |
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

### 4.3 The defects (four as input to decision D1; a fifth found later in our own port)

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

**A fifth, found later — in OUR port, not the original (owner report, level 9).** The carried-block follow-up
of §4.2 rule 6 is judged against the cell diagonally forward-and-up **from the old position**. The `.pg`
judged it from the position *after* gravity instead. The two agree whenever the step is level, which is why it
survived every test: it only diverges when the step ALSO drops the player. Walking left off a ledge at
`(16,10)` on level 9 with wall at `(15,9)`, the block's own destination is solid rock — but the player falls
to `(15,11)`, the check then looks at `(15,10)`, finds it empty, and the block arrives there **still in his
hands, having passed diagonally through the wall.** Fixed: the follow-up is now decided from the old position
before the step and before gravity, so the block is knocked out of his hands and falls in its own column,
behind him. Pinned by `ACarriedBlockIsKnockedOutOfHisHands_WhenItsOwnWayForwardIsBlocked`.

**And its mirror, which must NOT be "fixed" (owner ruling, same report).** A stone directly above the carried
block does **not** knock it out on a climb: the climb moves the block diagonally up-and-forward, so the cell
directly above it is never on its path — it is a low ceiling he slides out from under. The engine already
behaved this way; it is now pinned by `ClimbingKeepsTheBlock_EvenWithAStoneDirectlyAboveIt` so the knock-out
rule above cannot later be widened into "anything solid near the block".

Both rules are pinned a third time on **real shipped content**, by
`OnLevelTen_TheSameBlockSurvivesAClimbAndIsThenKnockedOffByAWalk`: on level 10, reachable in 14 moves (found
by breadth-first search over the engine), the player holds a block under a ceiling at `(19,13)`, climbs
up-left and **keeps** it — then walks back right and **loses** it to that very same wall. One block, two
consecutive moves, opposite outcomes, because a climb carries it diagonally and a walk drags it sideways.
That is the whole distinction in one test, on a board a player actually meets.

**Still open — §4.2 rule 6 says the follow-up runs "even when the move was blocked", and the `.pg` does not.**
Walking into a wall while carrying returns early, skipping the follow-up entirely. The TI-84+CE port takes a
third position: `game.c` reverts the whole move when the carried block's destination is blocked, rather than
dropping the block. Not changed, because it would alter every level where a carrier bumps a wall — possibly
solvability — and no one has reported it. Needs an owner ruling against the real game.

**Not a defect — a vacuous non-issue (recorded so it isn't "fixed" later).** Gravity is never applied after
a climb, which *looks* like an omission next to the walk and drop paths. It has no observable effect. The
climb requires `(x±d, y)` to be a Blok or Steen and lands the player at `(x±d, y+1)` — **directly on top of
the cell whose solidity the climb just required** — so the player is supported by construction and a gravity
pass there can never find anything to do. The carried block likewise lands on the player's head, supported
by the player. Applying gravity after a climb would be behaviourally identical; leave the call out (it
matches the original and costs nothing), but do not record it as preserved-bug-compatibility.

### 4.4 Shipped level content — the 11 originals, plus 4 bonus levels (BUILT)

**All 11 original levels ship as-is** (Brandon Sterner, TI-83+ PuzzPack 2001), imported by
`tools/blockdude_levels.py` from the TI-84+CE port at `github.com/merthsoft/blockdudece` (`src/level.c`,
Unlicense) into `…/Environments/BlockDude/levels/blockdude-levels.json`. Tile codes came from that port's
`game.c` (`EMPTY 0 / WALL 1 / BLOCK 2 / DOOR 3`), which also independently corroborated two rules in §4.2.

Measured shape — and the reason the training plan cannot use this content:

| level | size | blocks | overhangs | floating blocks |
|---|---|---|---|---|
| 1 | 20×8 | 2 | 23 | 0 |
| 3 | 19×11 | 6 | 27 | 0 |
| 5 | 22×14 | 10 | 25 | 0 |
| 8 | 27×17 | 18 | 54 | 0 |
| 10 | 27×19 | 24 | 66 | 0 |
| 11 | 29×19 | **42** | 68 | **14** |

They carry **no `optimalMoves`**: only levels 1–3 are exactly solvable at all (§4.5). The owner accepted this
explicitly — *"Ship them all, I understand that they won't be AI solvable, that's fine."*

**Not gravity-settled on load.** Level 11 ships 14 blocks floating in mid-air, which is authored content; a
global settle pass would silently rewrite the puzzle. Pinned by `BlockDudeEngineTests`.

#### 4.4b Bonus 1–4 — the CSE extra level set (BUILT)

Four further levels ship after the originals, named `Bonus 1`–`Bonus 4`. They come from `BLOCKLV2`, the
optional extra level set of the same author's earlier **TI-84+CSE** release, which — unlike the CE port — was
never published as source. The appvar is therefore **vendored** at `tools/blockdude/BLOCKLV2.8xv` and decoded
by `tools/blockdude_ti_levels.py`, a reader for the TI level-pack format (one TI-BASIC line per level:
`W`/`H` size, `I`/`J` player start, and a map string of two decimal digits per cell, row-major TOP-first,
`00` empty / `01` wall / `02` block / `03` door). It reads both the `.8xv` appvar and its Token IDE `.txt`
export; the two decode identically, which is the cross-check that the reader is right.

All four are 20×12 — the CSE screen — with 4, 11, 14 and 8 blocks.

The CSE release's **main** pack, `BLOCKLVL`, was examined and **deliberately not imported**: it holds the same
11 originals, 7 byte-identical to what we already ship and 4 reframed for the narrower CSE screen (levels 4
and 5 also move the player start). Each pack's first entry is the game's home screen — a title drawn in
blocks, not a puzzle — and is skipped on import.

**These levels broke the one-door assumption.** Bonus 2 seals its exit behind a row of **seven** door cells
and Bonus 4 offers **two separate exits**, so `BlockDudeBoard.FromGrid` now requires *at least* one door
rather than exactly one. No engine rule changed: the win check was already `tileAt(px, py) == door`, so any
door cell wins. The only thing that needs a single cell is the observation's door bearing, which takes the
last door in scan order. Pinned by `BlockDudeEngineTests.ALevelMayHoldSeveralDoors_AndAnyOfThemWins`.

The renderer had the same latent assumption and it was a **visible** bug: `drawDoor` took
`tiles.indexOf(2)` and drew exactly one aperture, so Bonus 2's other six door cells and — worse — one of
Bonus 4's two exits rendered as plain floor, hiding a genuine way out. Now `drawDoors` loops over every door
cell. Caught by playing the levels in the browser, not by any test; there is no renderer test.

### 4.4a Training-level generator — the heightmap design is DEAD

**Superseded. Recorded because it was wrong in an instructive way.** An earlier revision asserted, as
measured fact, that *"Block Dude terrain is a 1D height profile"* — 8 of 10 levels with zero overhangs — and
built the generator, the state encoding and a family of search heuristics on top of it.

That measurement was taken on the **WebGames synthetic levels**, not the originals. **All 11 originals have
overhangs, 18 to 68 each.** The claim was a generalisation from the wrong sample, stated with unearned
confidence, and it cost real work: it produced a generator design that cannot emit shipped-level terrain, a
state encoding sized for a board shape that does not exist, and it misled a later investigation into
proposing height-profile heuristics.

Consequences that stand:
- A generator **must** produce overhangs and shelves, not a `height[x]` array with bounded shelf patches.
- **Log the rejection rate, not just the accepted count.** The oracle cap means deep boards yield no labels;
  unlogged, that silently biases training toward shallow puzzles.
- Generated boards must be structurally incapable of reproducing a shipped level, so the 11 stay a clean
  hold-out — the same wall `RushHourImitationCampaign` puts around ThinkFun cards 1/38/39/40.

The replacement is a **curriculum** over generated boards: see §7.1.

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

#### 4.5.1 MEASURED reach of the exact oracle

| level | size | free cells | blocks | states | optimal |
|---|---|---|---|---|---|
| 1 | 20×8 | 116 | 2 | 256 | **19** |
| 2 | 22×10 | 174 | 5 | 55,314 | **73** |
| 3 | 19×11 | 143 | 6 | 14,199 | **94** |
| 4 | 24×16 | 312 | 8 | truncated at a **3M** cap (9 GB, 35 s) | — |
| 9 | 20×16 | 242 | 9 | truncated at 3M | — |
| 11 | 29×19 | 348 | 42 | truncated at 250k in 3.9 s | — |

**Only levels 1, 2 and 3 are exactly solvable.** Cost is **memory**-bound, not CPU: ≈1.3–3.0 KB and ≈7–10 µs
per state, so the hard wall on a 40 GB machine is ≈10–12M states. Growth is ≈×4.5–7 per extra block.

Frontier: roughly **≤6–7 blocks over ≤150 free cells**. Note it is *not* simply board size — level 3 has
more blocks than level 2 yet yields a quarter of the states, because tight corridors constrain the player
while open sky explodes. Level 11 is not "hard", it is off-scale.

### 4.5a State encoding (as built)

Block Dude states are far too wide for an integer key: level 11 is 42 blocks over 551 cells. Two rejected
alternatives, both recorded because each looked attractive:

- **Per-column block counts** (~24 bits, one `i32`): correct only while no level has an overhang. Every
  original violates that (§4.4a). Cheap but brittle.
- **Two `i32` words + exact comparison**: what an earlier revision specified. Sized for ~5 blocks over ~108
  cells — a board shape that does not exist in the shipped content.

**As built: a 32-bit FNV-1a hash of the mobile state, chained.** `PgBdVisited` open-addresses on the hash
and keeps a per-bucket chain, comparing candidates with `sameState` — so a collision costs a comparison
rather than silently merging two positions. A hash alone is never a state identity. Terrain is excluded from
the hash since it is immutable.

Two `.pg` traps hit while implementing this (see also §8.3): the FNV offset basis `0x811c9dc5` does **not**
fit `i32`, and spelling it as an out-of-range literal makes the transpiler widen the whole expression to
64-bit — which lowers to TypeScript `bigint`, correct but allocating. It is written as the signed equivalent
`-2128831035`. And `PgIntMap` could not be reused from `lunarlockout_solver.pg`: both files emit into the
same assembly's global namespace, so shared helper names collide.

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

### 5.3 Level content — the inherited pack was unplayable; the ladder is generated (BUILT)

**Correction to earlier revisions of this document:** the WebGames pack has **15** levels, not 16. The
sixteenth `name:` belongs to the `interface LevelData` declaration — a miscount inherited from the audit and
propagated here unchecked.

Then the real finding. Running the oracle over all 15 revealed that **13 are unsolvable and 1 is
mis-numbered**; only *Getting Started* is correct as authored:

| symptom | levels |
|---|---|
| **no legal first move at all** (1 reachable state — no two robots share a row or column) | 4 |
| unsolvable after some play | 9 |
| solvable but mis-numbered (authored 7, optimal 6) | 1 |
| correct | 1 |

So `minMoves` was not merely unverified: the **grids** are synthetic filler with invented numbers.

**How this was caught, and why it needed catching twice.** §5.3's own diagnostic said a *systematic* skew
points at the rules rather than the data — and 13 of 15 failing is about as systematic as it gets. That
diagnostic pointed the wrong way. Only re-implementing the rules from scratch in Python, independently of
the `.pg`, and getting **exactly** the same 13 unsolvable levels and the same 6-vs-7 mismatch, distinguished
"my engine is broken" from "the content is broken". Two independent implementations agreeing is the evidence
that mattered; a single implementation's verdict on its own data source could not settle it.

**Replacement:** `tools/lunarlockout_levels.py` generates the shipped ladder deterministically (seed 58),
exhaustively verifying each entry and canonicalising under the 8 symmetries of the square so no two rungs
are rotations of the same puzzle. **12 levels, optimal 1→12**, robot count ramping 3→5, difficulty ordered.
`LunarLockoutOracleTests` re-derives every count through the `.pg` oracle, so the generator and the engine
cross-check each other permanently.

These 12 are the held-out gate set (D4); a training generator must not be able to emit them.

---

## 6. Salvage from the redundant three

### 6.1 Rush Hour levels — harvest ABANDONED: the source data is not decodable

`C:\Repos\WebGames\RushHour\Scripts\levels.ts` advertises **40 official ThinkFun levels**. The plan was to
convert them to `VehicleDto[]`, run them through `RushHourDeckStore.Upsert` (which validates the board and
computes BFS-optimal moves) and add whichever were not already among the deck's 79.

**They do not decode into legal Rush Hour boards, under any plausible reading of their encoding.**

The file's own helper says `width === 1` means *vertical* — yet the red car is declared with `width: 1`, and
a vertical red car can never reach the exit. That contradiction prompted a systematic check rather than a
guess: all 16 combinations of (swap the first two arguments) × (which width value means horizontal) ×
(anchor cell is the near or far end) × (row axis flipped) were scored against hard invariants — every vehicle
inside the 6×6 grid, no two overlapping, exactly one main car, horizontal, on the exit row.

**All 16 conventions produced ZERO legal boards out of 40.** Level 1 under the plain reading has a vehicle
running off the board edge, two overlapping pairs, and a vertical main car off the exit row. For comparison,
the genuine card 1 is hard-coded in `RushHourImitationCampaign` as a held-out evaluation puzzle, and the
decoded data matches it under no convention.

**Consequences:**
- Nothing is harvested; the deck keeps its 79 levels unchanged.
- This is the **third** WebGames level pack to fail verification, after Lunar Lockout (13 of 15 grids with no
  legal first move) and Block Dude's synthetic flat terrain. The pattern is consistent: WebGames' *code* is
  real, but its *level data* is unreliable throughout.
- Retiring WebGames therefore loses **no level content at all** — a stronger result than the "probably
  redundant" this section originally anticipated.
- The official cards this repo actually relies on (1, 38, 39, 40) already live in the campaign source and are
  unaffected.

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

**✅ M58.0 — Preserve before anything else.** In `C:\Repos\WebGames`: `git add` the four untracked projects,
commit, push. Non-negotiable prerequisite; §1.1.

**Ordering, as revised.** D8 originally put Lunar Lockout first end-to-end, on the reasoning that it is the
de-risked game — rules externally verified, state space exhaustively enumerable, no dead ends, no gravity,
single-`i32` key — so it would prove every unproven piece of infrastructure on the easy case before Block
Dude's hard parts landed on it.

That rationale held for the *engines*, and both are now built in that order (M58.1 then M58.4), which is why
the first hand-rolled `IntSet` in a `.pg` and the two-pass irreversible-graph oracle were both debugged on
the simpler game. **The rest was reordered**: the owner wants an overnight training run, so Block Dude's
generator, campaign and Lab entry come before Lunar Lockout's UI. A trainable Block Dude is worth more at
11pm than a finished Lunar Lockout page.

*Accepted cost of the original order, now largely spent:* Block Dude is the game the owner actually asked
for, and its engine landed second.

**✅ M58.1 — Lunar Lockout engine + oracle.** `Environments/LunarLockout/polyglot/lunarlockout_solver.pg`:
rules per §5.2, BFS oracle, `IntSet`. TS twin routed in `pgconfig.json` to
`app/lunar-lockout/lunarlockout_solver`. C# facade + `LunarLockoutEnv.cs`. **Gate: every shipped level's
optimal count re-derived by the `.pg` oracle** (§5.3). This gate is what caught the inherited pack as
unplayable; the ladder is now generated and all 12 counts are BFS-proven.

**M58.2 — Lunar Lockout campaign + net.** `Campaigns/LunarLockout/LunarLockoutImitationCampaign.cs`, DI
registration in `CampaignServiceCollectionExtensions.cs` + a `CampaignRegistrationTests.cs` case, Lab
`--game lunarlockout`, level generator, training run, checkpoint to `wwwroot/models/`.

**✅ M58.3 — Lunar Lockout UI.** `app/lunar-lockout/` + route + nav + home card, renderer per §10.4,
`lunarlockout-net.ts` `.ckpt` parser + director with the stale-checkpoint guard.

**✅ M58.4 — Block Dude engine + oracle.** `Environments/BlockDude/polyglot/blockdude_solver.pg` — rules per
§4.2 (corrected per D1), oracle per §4.5, two-word key + `IntSet` per §4.5a. Tests:
`BlockDudeEngineTests.cs` (every rule in §4.2 as a case, including fall-through-door and
block-supported-by-door), `BlockDudeParityTests.cs`, and an oracle test asserting `optimalMoves` per level.

**✅ M58.5 — Block Dude generator + campaign + net (phase 1).** Heightmap generator with bounded shelves per
§4.4a, imitation campaign, Lab `--game blockdude`, training run to the §8.4 gate legs 1–3, checkpoint.

**M58.5a — Block Dude expert iteration (phase 2).** Per §8.1b: oversized-board generator tier, A*/beam
self-solving loop, distinct checkpoint ids (`blockdude.policy-xit*`) leaving the phase-1 net untouched, and
§8.4 gate leg 4. **This is the milestone that answers the imitation-ceiling objection** — if leg 4 does not
beat the phase-1 baseline, phase 2 has failed and the phase-1 net ships as the browser tier.

**✅ M58.6 — Block Dude UI.** `app/block-dude/` + route + nav + home card, renderer per §10.3, keyboard +
on-screen mobile controls, **undo (mandatory — the game is irreversible)**, hint, level picker, Watch-AI.

**M58.7 — Rush Hour level harvest: ABANDONED.** The source data does not decode into legal boards under any
of 16 candidate conventions (§6.1). The deck keeps its 79 levels; nothing is lost by retiring WebGames.

**✅ M58.10 — Direct manipulation in both games** (§12). Lunar Lockout: rocket glyph, hover-to-aim, press-and-drag on
touch, retargeted hints, renderer `hover()` entry point. Rush Hour: axis-locked sub-cell drag with anchor
re-coupling, settle animation, per-cell move counting, `touch-action: none`, mode-guarded editor coexistence.
Includes the two pre-existing defects in §12.6.

**M58.8 — Retire the repos.** Update `docs/ARCHITECTURE.md`, `docs/prd/PLAN.md` (M58 entry) and
`docs/ADDING_A_GAME.md`; then the owner deletes `C:\Repos\WebGames`. Leave `C:\Repos\Spelletjes` alone — it
still holds Rush Hour, `RushHour.Core`, its tests and the designer PRDs, none of which are in scope here.

**✅ M58.9 — Fix the stale home card** noted in the audit: `app/home/home.ts:65` still says FruitCake has
"no AI (yet)", untrue since M32. One-line fix, lands in the same PR.

Test suites run **once**, at the end of M58.6 (after all engine, campaign and UI milestones), per the repo's
batch-tests-at-the-end rule. Intermediate milestones are verified by reading code and type-checking. The two
training runs are long jobs — background them and wait for the notification; never poll.

---

## 7.1 Curriculum + blank-slate reproducibility (D9)

Owner requirement: train on very small boards first and grow them as the model improves, **and** *"be able to
re-run the training at any time from a blank slate."* The second half is the hard one — it means a stage must
never depend on wall-clock or on whatever checkpoint happens to be lying around.

**Stage advance is a pure function of persisted state:**

```
Advance(stage, stageSamples, lastGateRate) =
  stage + 1  iff  stage < LastStage
               && stageSamples >= Stages[stage].MinStageSamples
               && (lastGateRate >= Stages[stage].PromoteSolveRate
                   || stageSamples >= Stages[stage].MaxStageSamples)   // logged as a FORCED advance
```

- The gate is refreshed **inside `TrainChunk`** on a sample cadence — **never in `Evaluate()`**, because
  `CampaignRunner` fires that on the wall clock, which would make the trajectory machine-speed-dependent.
- The per-stage gate set is fixed from `new Xoshiro256StarStar(4242 + stage)`, deliberately **independent of
  `--seed`** (the same trick `RushHourImitationCampaign` uses with seed 777), so runs at different seeds stay
  comparable and the gate never depends on training history.
- Advance is monotone; the top rungs mix ~20% of earlier-stage boards to prevent forgetting, as a constant in
  the stage table rather than a runtime decision.

**Stage boundaries are set by §4.5.1's measured frontier** (≤6–7 blocks over ≤150 free cells). The top rungs
will reject most boards — that rejection rate is a first-class logged metric, and it is exactly the signal
that phase 2 (§8.1b) exists to consume.

**Persisted state** — a new sidecar `blockdude.policy-state`: observation shape + action count, a
`RunFingerprint` over (seed, obs shape, curriculum version, stage table, batch size, LR), `Stage`,
`StageSamples`, `TotalSamples`, per-stage gate rates, `BoardAttempts` (**attempts, not accepted boards**, so
rejections don't shift the generator stream), accept/truncate/reject counts, and the owner-thread RNG states.

**Blank slate:** `--fresh` deletes the three checkpoint ids *after* taking `TrainingDirectoryLock` (so it can
never race another run) and rotates the CSV — `CampaignCli` appends, so without rotation a fresh run silently
continues the old log. Without `--fresh`, a `RunFingerprint` mismatch refuses loudly and starts fresh rather
than diverging silently. A net present with no state sidecar is treated as an unlabelled warm start: stage 0,
logged — reproducibility is a property of the state file, not the weights.

**Determinism rests on an existing primitive.** `DeterministicParallel` derives item *i*'s RNG from its
index, so output is invariant to worker count — replaying generation needs only the persisted counter. Two
hard rules: all net-dependent work stays on the owner thread (DAgger rollouts read the net while Adam mutates
it), and no dictionary/group enumeration order may feed an RNG.

### 7.1a Observation encoding — the decision a size curriculum forces

The net must accept boards from ~8×5 up to 29×19. **Egocentric 21×13 window centred on the player × 4 planes
(wall, block, door, off-board) = 1092, plus 24 global features = 1116 floats, constant at every stage.**

So advancing a stage is purely a data-distribution change: no net surgery, no `GrowInput`, no checkpoint
invalidation, and level 11 produces the same tensor shape as an 8×6 board.

Rejected: **fixed max-size padded planes** (a stage-1 board would train ~97% dead inputs, and it forces the
max board size to be committed on day one) and **fully convolutional** (needs a new pooled-head net type in
Core; `ConvResidualPolicyValueNet` bakes H×W into its parameter shapes and its `LayerNorm` runs over the
whole map, so padding shifts normalisation statistics between stages).

`buildObservation` goes in the `.pg` so the browser computes byte-identical inputs and its stale-checkpoint
guard is meaningful. Actions stay absolute — `Left`/`Right` are absolute in `BlockDudeAction`, and mirroring
by facing would desynchronise action semantics from the observation.

---

### 7.1b Capacity growth on a SATURATION signal (BUILT 2026-09-13)

**Owner Q, 2026-09-13: "will the net automatically grow when necessary?" — it does now.** The same question
was asked and answered *no* on 2026-07-15 (`DRAUGHTS_SELFPLAY_PRD.md` §4.2: "it grows on a fixed sample
cadence (`GrowEvery`, opt-in), **not on a saturation signal**"). The mechanism existed — function-preserving
Net2Net `WidenTo`/`Deepen` — but the *policy* was a clock. `--grow` stepped the architecture every
`--grow-every` samples regardless of whether the net needed capacity, which is close to the worst of both
worlds: grow early and pay for capacity that cannot yet be used, grow late and waste samples on a saturated
net.

**Two defects found while building it, both of which invalidated the obvious experiment.**

1. **The shared ladder made Block Dude's net SMALLER.** `DqnGrowth.Stages` is
   `[16] → [32] → [32,32] → [64,64] → [64,64,64] → [128,128,128]`, sized for the Snake/FruitCake DQN nets.
   Block Dude's default trunk is `[512,512]` against a 1181-wide observation, so the shared ladder's **top**
   rung holds less capacity than the net this game already trains with. A `--grow` run would have started 32×
   narrower and finished smaller than it started — and any conclusion drawn from it about the stage-3 plateau
   would have been backwards. Block Dude now has its own ladder in `BlockDudeGrowth`, rung 0 of which **is**
   the default trunk, so enabling growth never starts a run smaller than not enabling it.
2. **The rung was guessed, not recorded.** `PolicyGrowth`/`DqnGrowth` recover the current rung by matching the
   live trunk against the schedule and **fall back to rung 0** when nothing matches — which is exactly what a
   net built outside the ladder (any non-growing run) looks like. Resuming such a net with growth enabled would
   read it as rung 0 and, at a high sample count, jump it to the top of the ladder in a single step. The rung
   is now persisted in `BlockDudeTrainingState` (sidecar **v3**; v2 refused, not upgraded), and
   `BlockDudeGrowth.GrowOne` **refuses to grow** when the recorded rung disagrees with the live trunk rather
   than guessing.

**The trigger, and why it is shaped this way.** Growth fires when the **gate** — the held-out solve rate —
stops producing new highs: `GrowthPlateau` keeps a running maximum plus a patience counter, exactly as early
stopping does, and reports saturation after `--grow-patience` (default 6) gate evaluations without beating the
window's best by `--grow-min-improvement` (default 0.04).

- **Why the gate and not the loss.** Falling loss with a flat gate *is* the saturation signature: the net is
  still learning to reproduce the oracle's chosen action and still failing to solve boards. A loss-driven
  trigger cannot see that — it reads the falling loss as healthy progress and never fires. This is precisely
  `bd2`'s stage-3 shape (accuracy 0.927, gate ~0.50–0.59).
- **Why a running maximum and not a comparison of consecutive values.** The gate is noisy: measured swinging
  **~10 points between consecutive evaluations** on the 64-board hold-out (61% → 50% → 52%) with the loss
  falling throughout. A consecutive-value detector fires on that noise almost immediately. Against a running
  maximum a downward swing can never trigger growth; upward noise can only *delay* it, which is the safe
  direction to be wrong in — a late grow costs samples, an early grow costs samples **and** resets Adam on a
  net that was still improving. `--grow-min-improvement` must stay above the jitter floor or upward noise
  resets patience forever and the trigger silently never fires; the default sits a little under half the
  measured swing.
- **Why the window resets on promotion.** A harder rung gates on harder boards, so the solve rate legitimately
  **drops**. To a detector watching for "no new highs" that is indistinguishable from saturation, and it would
  grow the net for the one reason that is not a capacity problem. The window also resets after growing: the
  grown net inherits the old one's function exactly, so it would otherwise inherit its best gate and have to
  beat the plateau it was added to break — and one saturation event could cascade up the whole ladder.

**Spikes that shaped it** (cheap, and each killed an option):

| Spike | Question | Result |
|---|---|---|
| Replay the measured gate series `0.61, 0.50, 0.52, 0.66, 0.55, 0.58, 0.72` through the detector | does real noise on a rising trend trigger growth? | **No** — three apparent "declines", zero patience spent. A consecutive-value rule fires three times. Pinned by `TheMeasuredGateSwingDoesNotTriggerGrowth_ForANetThatIsStillImproving`. |
| Feed a collapsing series `0.80, 0.10, 0.05, 0.02` | can a crash be mistaken for saturation? | **No** — patience advances at exactly the flat-metric rate, never faster. Pinned by `DownwardNoiseAloneCanNeverTriggerGrowth`. |
| Feed a 1-point-per-eval creep against a 4-point threshold | does upward jitter starve the trigger? | It **saturates** as intended; with the threshold set below the step size the same series reads as progress. Pinned by `AnImprovementSmallerThanTheNoiseFloorDoesNotResetPatience`. |
| Forward shipped level 1 through a net before and after one rung | is the grow really function-preserving? | Logits and value identical to 3 dp. Pinned by `GrowthIsFunctionPreserving_SoCapacityArrivesWithoutALossSpike`. |
| Compare rung 0 against `DqnGrowth.Stages[^1]` | is the shared ladder usable here? | **No** — defect 1 above. Pinned by `RungZeroIsTheDefaultTrunk_SoGrowingNeverStartsSmallerThanNotGrowing`. |

**Not generalised to the other campaigns.** `PolicyGrowth`/`DqnGrowth` keep their sample cadence, so Cube,
Rush Hour and the DQN games are untouched and their trajectories stay comparable to their own history. The
`GrowthPlateau` detector is game-agnostic and lives in `Campaigns/Shared`, so adopting it elsewhere is a
ladder plus a call site — but each game needs its own ladder sized to its own net, which is the lesson of
defect 1 and should not be skipped.

**Still open.** The ladder tops out at `[1024,1024,1024]`; a run that saturates there has no capacity lever
left and says so in the log rather than silently no-op'ing. Whether the stage-3 plateau is *actually* capacity
is the hypothesis this feature exists to test, and it is untested until a run has been read.

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
   materially slower at 10⁶ probes. Stay ≤31 bits (Lunar Lockout packs into 30), or — where the state is far
   too wide for that, as Block Dude's is — hash into an `i32` and **chain**, comparing candidates exactly
   (§4.5a). Note the related literal trap: an offset basis like `0x811c9dc5` does not fit `i32`, and writing it
   as an out-of-range literal silently widens the whole expression to 64-bit.
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

1. **All shipped levels solved** by **policy-guided A*** — the net's distance head as the heuristic. The
   **≤1.25× optimal** bound applies only where an optimal count exists: all 12 Lunar Lockout levels and Block
   Dude levels 1–3. For Block Dude 4–11 the criterion is *solved at all*, with move count tracked as a
   regression baseline rather than compared to an unknown optimum (§4.5.1), and **level 11 excluded from the
   gate entirely** (§8.5).
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

### 8.5 Level 11 is a stretch benchmark, not an acceptance criterion

Level 11 will be **implemented and playable** — it already is, floating blocks and all — but it will not be
*exactly* solved, and it should not gate the milestone.

- **Exact optimality is impossible**, not merely expensive: proving no shorter solution exists means
  exhausting the reachable space, and 42 blocks over 348 free cells is off-scale by dozens of orders of
  magnitude (§4.5.1). *Finding* a solution is a different and much easier problem than *proving it minimal*.
- **Depth is the real obstacle for search.** Level 3 already needs 94 optimal moves; level 11 plausibly needs
  600–1500. The cube's beam search succeeds partly *because* its horizon is ≤40 moves — failure probability
  compounds per step, so a beam that is right 99.9% of the time per step still fails often over 1000 steps.
- **And the cube has no dead ends.** Every cube state is ≤20 moves from solved, so a wandering beam is merely
  slow. Block Dude states can be **terminally lost**, so all 2000 nodes of a beam can be dead at once — a
  failure mode the cube literally cannot have. **Dead-end detection is therefore a precondition for search to
  work at all, not an optimisation.**
- Realistic shape of success: **subgoal decomposition** — one terrain obstacle at a time, each a short-horizon
  search with an irrecoverability check gating every commit — not one monolithic 1000-move beam. Levels 4, 5
  and 9 (8–10 blocks) look reachable; 6, 7 and 8 (14–18) are a genuine research push.

### 8.6 Reuse `Core\Planning`, don't write another search

Contrary to an earlier assumption in this document that every game's search is bespoke, `Core\Planning`
already holds a **game-agnostic single-agent planning layer**: `IDeterministicModel` (`ActionCount`,
`IsGoal`, `Apply`, `StateKey`), `BreadthFirstPlanner`, `GreedyValuePlanner`, and `ValueGuidedSearch` —
weighted A* plus a batched DeepCubeA-style variant that scores a whole expansion front in one forward call.
Core carries no NN dependency: the net always enters as a delegate.

`BlockDudeBoard` already satisfies the contract (immutable `Apply`, `Won`, `ActionCount = 4`) apart from
`StateKey`, since `StateHash` is explicitly not an identity. `CubeValueSearch.cs` is a ~90-line adapter to
mirror. Beam search is the one piece *not* yet generic — it lives typed to `FaceletCube` in
`CubePolicySearch.BeamSearch`, whose dedupe-by-key-keeping-best-route and log-prob pruning are the parts to
lift.

Two measured warnings from elsewhere in the repo: Snake found that **per-node net scoring made its search
weaker** (the net is a root-move tiebreak there, not a per-node score), and turn-as-a-move inflates Block
Dude's branching with near-duplicate states, so move-redundancy pruning (never turn twice, never
grab-then-ungrab in place) is needed the way the cube masks no-undo moves.

### 8.7 Two latent bugs fixed en route (FIXED)

Found while designing §7.1, both pre-existing and unrelated to the new games:

1. **`PolicyValueNet.Load` silently corrupted a stale checkpoint.** `inputSize` is not stored in the file, and
   a shorter stored weight array still satisfied `Span.CopyTo`, so loading a net trained at a different
   observation width left the tail of layer 0 at fresh random init — no exception, no warning, a quietly
   broken policy. Now every layer's float count must match exactly.

   **Deliberately fixed without bumping the checkpoint format.** Storing `(inputSize, actions)` would be more
   self-describing, but the binary layout is a **cross-language contract** — the transpiled `.pg` solvers parse
   it in the browser, and `ChessNetParityTests.LoadPg` pins the version at 2. Bumping to v3 broke exactly those
   two tests, and would have broken the shipped chess and draughts net readers. The exact-length check needs no
   format change and, unlike stored metadata, also validates the v1/v2 files written before it existed.

2. **Imitation and self-play campaigns persisted no progress.** `RushHourImitationCampaign`,
   `CubeImitationCampaign` and `SelfPlayCampaign` saved net and Adam state but re-derived every counter and
   owner-thread RNG from the seed on each `Resume`. A restarted run replayed the same generated data *and*
   reset `PolicyGrowth`'s stage target, which keys off the sample counter — so a repeatedly-interrupted run
   kept re-growing its trunk from the first stage. Self-play was hit twice: `MaybePromoteDifficulty` reads
   `_lastWinRate`, unpersisted, so ladder promotion depended on whether the run had evaluated since the last
   restart. Fixed with an additive `CampaignProgressState` sidecar; a store without it starts at zero, i.e. the
   old behaviour, so existing checkpoints keep resuming.

---

## 9. Out of scope (genuinely not being done)

- **Tic Tac Toe — dropped (D3).** A solved, AI-less 158-line game. Perfect play is a draw from every
  position, so there is no skill ceiling to showcase and no puzzle to solve; a route backed by nothing would
  undercut the playground's pitch. Not deferred — deliberately discarded, which means the retirement is
  *not* literally lossless and that was the accepted trade.
- **A light theme for the app.** The app is dark-only today (§10.1). The new renderers carry their own light
  palettes, but introducing app-wide theming is not M58 work.
- **The per-column block-count state encoding.** Rejected on brittleness grounds, not cost — see §4.5a.
- **Exact optimal solutions for Block Dude levels 4–11.** Not deferred — *impossible*, and measured as such
  (§4.5.1). Levels 1–3 get proven optimal counts; the rest get solutions whose length is observed, never
  claimed minimal.
- **A 1D height-profile terrain model**, and every heuristic built on it. The premise was false for the
  shipped levels (§4.4a).
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

## 12. Direct manipulation — Lunar Lockout and Rush Hour (D11)

Owner request, 2026-09-11: both games currently use *select a piece, then press a direction button*, which is a poor
fit for direct-manipulation puzzles. Replace it. Lunar Lockout should mirror the Windows desktop original — the
rocket rests at 45°, points toward the hovered side, and clicking launches it. Rush Hour should let the player
"tap and drag the car forward/backward until it's road-blocked", with **smooth sub-cell motion, not snap-to-grid**.
Both must be playable on a phone.

### 12.1 One gesture vocabulary, two verbs

The two games deliberately differ in what a move *is*, so the vocabulary is shared but the verb is not:

- **Lunar Lockout = point-and-launch.** The player chooses a *direction*; the rules decide the distance.
- **Rush Hour = grab-and-slide.** The player chooses the *distance*; the axis is a property of the vehicle.

Shared rules, binding on both:

| Concept | Rule |
|---|---|
| Engage | `pointerdown` on a piece, always with `setPointerCapture(pointerId)` |
| Preview | Continuous and non-committing — rocket rotation / the car under the finger |
| **Commit** | **`pointerup`, never `pointerdown`** — so a started gesture can always be abandoned |
| Abort | `pointercancel`, or `pointerup` in a neutral state: no move, no counter change |
| Tap vs drag | Under the threshold it is a *tap* (selects); above it is a *drag* and never also selects |
| Threshold unit | A fraction of a cell scaled to CSS px, never raw px — the existing convention |
| `touch-action` | **`none`** on both canvases |
| Illegal feedback | Never silent: the piece visibly refuses **and** the status line explains why |

### 12.2 Input branching — one Pointer Events stream (D12)

Owner: *"when on desktop: process mouse-events; when on mobile: process touch-events."* Implemented as a
requirement on **behaviour**, through a single Pointer Events stream rather than separate `mousemove`/`touchstart`
handlers:

- `event.pointerType === 'touch'` selects the flick path; `'mouse'`/`'pen'` select the hover path.
- `matchMedia('(hover: hover) and (pointer: fine)')` decides whether hover tracking is bound at all.

Three reasons, recorded so this is not re-litigated:
1. **Touch devices synthesise mouse events.** Binding both sets double-fires every tap, and the usual remedy is a
   tangle of `preventDefault` and timers.
2. **Desktop-versus-mobile is not binary.** Touchscreen laptops and 2-in-1s are both, and switch mid-session.
   Per-event capability handles that; a device guess does not.
3. **The repo already standardises on Pointer Events** — Tetris and Crazy Fruits both use them with
   `setPointerCapture`, and no game reads `mousemove` or `touchstart`.

On a hybrid device both paths coexist: hover tracking is bound because `(hover: hover)` matches, but an individual
`pointerType === 'touch'` event still takes the flick path, so picking up the pen or touching the screen gets the
right model for *that* interaction with no mode switch.

**Three non-negotiable implementation details**, which are the real source of "pointer events are flaky on mobile":
`touch-action: none` (or the browser claims the gesture and fires `pointercancel` mid-drag), handling
`pointercancel` (the browser can revoke a gesture at any time), and `setPointerCapture` (or a drag dies the moment
the pointer leaves the element).

### 12.3 Lunar Lockout — aim and launch

**Hover model: edge-zones with a dead core, not quadrants.** Quadrants have no rest state — the rocket would snap
to a direction the instant the pointer entered the cell, and the diagonal boundaries run through the glyph itself,
so jitter over the body flickers. The 45° rest pose must be reachable while the pointer is still on the piece.

Two-axis hysteresis, in cell units from the cell centre:

| Constant | Value | Meaning |
|---|---|---|
| `HYST_IN` / `HYST_OUT` | 0.16 / 0.11 | radius to acquire a direction / to fall back to rest |
| `ANGLE_IN` / `ANGLE_OUT` | 40° / 50° | wedge half-width when acquiring / when holding |
| touch `ARM_R` / `RELEASE_R` | 0.22 / 0.15 | fat-finger equivalents (~16 px / ~11 px at a 72 px cell) |
| touch `LATCH_R` | 0.75 | past this the direction latches and the wedge test stops |
| `TAP_MAX_R` / `TAP_MAX_MS` | 0.20 / 250 ms | below both on release it was a tap, not a drag |

Both boundaries are sticky toward "keep what you have", so a pointer parked on a boundary never oscillates.

**Illegal directions teach by absence.** If the aimed direction has no blocker, the rocket **does not rotate** — it
stays at 45°, so it never *looks* launchable — the cursor drops to `default`, and a dashed refusal stub is drawn
(a hairline 0.30C in the aimed direction, capped by a perpendicular bar, in `hair`; no red, no new token). Since
the edge is not a backstop, most directions are illegal most of the time, so this carries much of the rules
teaching. Hover never writes the status line: it is `role="status"`, so every write is announced.

**The rocket glyph** is a flat dart with a notched tail, drawn in code, filled with the existing `robot`/`target`
tokens — no gradients, no second colour. Bounding radius 0.40C, so the board's visual density is unchanged from the
circles. The target robot keeps two *shape* cues (a porthole and a tail bar in `void_`), because with a rotating
glyph a hue-only distinction gets worse, not better, under deuteranopia.

Rest angle **−45°** for every robot. Rotation is **shortest-arc, 140 ms, `easeOutQuad`** — deliberately not the
linear curve used for slides, because a slide is a *result* while a rotation is a *response to the hand* and must
feel front-loaded. On launch the rocket holds its aimed angle for the whole flight and returns to rest afterwards.
Reduced motion snaps the angle, reusing the renderer's existing `animated` predicate.

**Landing hints are retargeted, not kept as-is.** Today all four legal landings are drawn for the selected robot.
With a rotating rocket that is two competing signals, so the pointer path draws **only the aimed direction's** hint.
The keyboard path keeps all four, because without a pointer there is no aimed direction.

**Mobile: press-and-drag from the rocket, release to launch** — the same code path as the mouse, no branches. The
finger is on the piece, and the aiming feedback lives in the rocket plus the dashed hint *ahead* of it, outside the
thumb. Dragging back into the dead core before releasing cancels. Rejected alternatives: tap-then-tap-a-zone (two
identical-looking taps invite double-fires, and the thumb covers the zone being chosen), and tap-then-swipe-anywhere
(breaks the direct-manipulation premise and relies on invisible selection state).

### 12.4 Rush Hour — grab and slide

**Axis-locked continuous drag.** A horizontal vehicle tracks pointer X only, vertical tracks Y only; cross-axis
drift is ignored entirely, because the finger wanders and the car should not stop. Clamp bounds are computed **once**
on `pointerdown` from the occupancy grid with the dragged vehicle removed — nothing else moves during a drag, so
per-frame recomputation would be waste.

**The drift fix, which is what makes a hand-rolled drag feel broken if omitted.** With a naive `pos0 + delta`, a
pointer that overshoots a blocker by three cells must travel three cells back before the car responds — the control
feels dead. Instead, whenever the raw position is clamped, the **anchor is rewritten** so the pointer is
re-referenced to the clamp, and the car starts moving on the first pixel of return travel. The finger's absolute
position no longer maps to the car after an overshoot, which is exactly what a physically blocked object should do.

**Settle on release**: round to the **nearest** legal cell over a fixed **90 ms, linear** — no easing, no bounce,
no overshoot, matching Lunar Lockout's slide policy exactly. (An earlier draft of this section said 60–90 ms with
`easeOutQuad`; that was wrong and is corrected here — the whole family is linear, and the settle distance is at
most half a cell, so a per-distance formula buys nothing.) "Nearest" and "the cell it has travelled most of the way
into" are the same rule at a 0.5 threshold, and *nearest* is the one a player can predict without being told.
Instant under reduced motion — but note the **drag itself is never suppressed** by reduced-motion, because it is
direct manipulation rather than decoration; only the release settle collapses.

**`pointercancel` reverts to the grab position**, it does not settle to the nearest cell: a cancelled gesture is a
cancelled move, and the browser can revoke a gesture for reasons that have nothing to do with intent.

**A settle in flight must not be interruptible** by a fresh grab on the same vehicle, or `startPos` is read
mid-animation and the clamp is computed against a fractional position. Snap the settle to its target first, then
grab.

**Show the whole legal interval while dragging**: a hairline rectangle spanning `[lo, hi + length]` along the axis,
in the lattice hue. It is the Rush Hour analogue of Lunar's landing hints, it is strictly more information than the
two arrow buttons ever conveyed, and it costs one `strokeRect`.

**Move counting — verified, and load-bearing.** `RushHourOracle` enumerates **one-cell edges**
(`RushHourOracle.cs:41-52`), and the whole backward-BFS distance labelling is in those units — which is what both
`analysis().optimalMoves` and `DeckLevel.optimalMoves` report. Therefore a three-cell drag **must count as three
moves**. Counting it as one would let a player "beat the optimum" and make the page's "Solved in N (optimal M)"
line read as a bug. Caption copy becomes "Moves: N (one per square)" so the counter is not mistaken for a drag
count.

**The two games count moves differently, by design.** Lunar Lockout's `applyAction` teleports a robot to its
landing cell in a single action, so one slide is one move regardless of distance. This mirrors each game's own
solver and must not be "harmonised".

**`touch-action: manipulation` is the single most important line to change.** It disables double-tap zoom but still
lets the browser claim pan-x and pan-y — so a vertical drag on a vertical truck scrolls the page, the car never
moves, and the browser fires `pointercancel` mid-stroke. `pan-y` is not sufficient; vertical trucks are half the
board. It must become `none`, matching Tetris and Crazy Fruits.

**Editor coexistence is by mode**, using the `mode` signal that already exists. Edit mode keeps committing on
`pointerdown` (placing is a discrete act with nothing to preview) and never enters the drag path; `setPointerCapture`
is taken only in play mode. `touch-action` is bound to the mode so edit and playback keep page panning. Deliberately
**not** separated by gesture — a "drag" on an empty cell in edit mode has no meaningful reading, and the mode is
already visible in the cursor, side panel and caption.

### 12.4a Responsive board (DONE) — and the two defects it exposed

`draw()` used to hard-set `canvas.style.width/height` from logical constants (`CELL = 72`, `PAD = 14`,
`EXIT_W = 42`), giving a fixed **502 × 460** board that simply overflowed a phone. The canvas now scales, `draw()`
sizes its backing store from `canvas.clientWidth`, and pointer mapping goes through `getBoundingClientRect()`
rather than reading `clientX` against the logical constants — which breaks the moment the canvas is CSS-scaled.
At 390 px that yields ~47 px cells: below the 44 px WCAG figure for a discrete *tap*, but a drag grabs a two- or
three-cell body (~94–140 px), so the grab target is comfortable.

Doing it surfaced two bugs that presented as one symptom — the board rendering at **300 × 275**, then staying
300 × 275 while being *drawn larger* after entering play mode:

1. **Circular sizing.** `.board-panel` is a bare flex item, so it shrink-to-fits its content — while the canvas
   inside asks for `width: 100%` of that same panel. The browser breaks the cycle with the canvas's **intrinsic
   300 px default**, and `aspect-ratio` derives the 275. Fixed with a definite width on the panel.
2. **A stale backing store.** `draw()` sizes the buffer from the element but only runs when a signal changes.
   Switching to play mode reflows the row and grows the canvas's CSS box **without writing any signal**, so the
   old buffer was scaled up — same size, rendered larger, blurry.

**The general rule, which applies to every canvas here:** a canvas sized from its element needs *both* a definite
parent width *and* a `ResizeObserver`. The CSS alone gives a correct layout with a stale buffer; the observer
alone cannot escape the circular sizing. All three renderers (Rush Hour, Lunar Lockout, Block Dude) now observe
their canvas, which also covers window resizes and phone rotation — neither of which writes any game state.

**Site nav overflow (fixed here too).** Verifying at 390 px showed the board fitting while the *page* still
scrolled sideways: thirteen game links in a non-wrapping nav forced the document to ~1130 px. M58 added two of
those links. The topbar and nav now wrap.

### 12.4b Rush Hour has no keyboard path to selection at all

Selecting a vehicle currently *requires* a `pointerdown`, so a keyboard-only user cannot select one, and therefore
cannot move anything — the arrow keys work, but only on a selection they have no way to make. That is a
pre-existing accessibility hole, and this work should close it rather than widen it: add vehicle cycling
(`[`/`]` or Tab), mirroring Lunar's number keys, plus `Home`/`End` to slide to the clamp bounds in one press
(counting `|Δ|` moves, consistent with §12.4's per-cell rule).

### 12.4c Where the clamp logic belongs

`clampRange(vehicles, positions, index)` goes in `rush-hour-logic.ts`, not the component: that file is the browser
mirror of `RushHourBoard` and is where legality already lives, and the multi-cell scan is the natural
generalisation of its existing single-step `canMove`. A unit test can then assert the two agree — `clampRange`
must return exactly the interval reachable by iterating `canMove`.

It must also be computed **once per grab**, not per frame: `canMove` rebuilds the whole 36-cell occupancy grid on
every call, so calling it per `pointermove` would mean ~120 grid rebuilds a second for a value that cannot change
while a drag is in flight.

### 12.5 Accessibility — the button paths stay, demoted

Both games keep their direction buttons, shrunk and **disabled per illegal direction** rather than accepting a
click and then refusing it. They are the only pointer affordance for someone who can click but cannot drag —
head-pointer and switch users — and the visible discoverability surface for the keyboard path. Removing them would
strand exactly the users the hover model already excludes.

Keyboard is unchanged and gains focusability: `tabindex="0"` plus `role="application"` on both canvases, so the
keyboard path no longer depends on a window-level listener.

### 12.6 Two pre-existing defects this work must fix

1. **Lunar Lockout hijacks page scrolling for every visitor.** The component binds `keydown` on `window` and calls
   `preventDefault()` on the arrow keys, so arrow-key scrolling is dead on that route whether or not anyone is
   playing. It moves onto the focusable canvas. *(Introduced by M58.3 — mine.)*
2. **Rush Hour reallocates its canvas backing store on every `draw()`.** `canvas.width`/`height` are assigned
   unconditionally, which clears and reallocates the buffer each call. Harmless while repaints were rare; with a
   drag repainting per pointer move it will visibly stutter on low-end devices. Guard the assignment on change, as
   the Lunar renderer already does.

### 12.7 Evidence behind the two contested choices

**Release-in-the-dead-zone cancels, and commit is on `pointerup`.** The best-documented analogue for
select-then-commit on a grid is chess apps, where there are years of first-person reports of drag-and-drop causing
**accidental commits on the wrong square** under time pressure. A pure swipe-to-fire has no cancel path: the finger
is over the piece at the exact moment of commit.

**Quadrant tap targets were rejected on measurement.** A 5×5 board at a 360 px viewport gives ~64–72 px cells;
quartering a piece yields ~32 px zones, which clears WCAG 2.5.8 AA (24 px) but fails Apple's 44 pt and Material's
48 dp — and a fingertip covers roughly a whole cell, with no hover to disambiguate.

**Threshold reconciliation.** The interaction design proposed latching a direction at 0.35 cell (≈24 px, reused from
Crazy Fruits); the mobile research recommended 12 px, between Android's `TOUCH_SLOP` (8 dp) and `PAGING_TOUCH_SLOP`
(16 dp). They measure different things and both are right: **preview early, commit on release**. The adopted
`ARM_R` of 0.22 cell (~16 px) sits between them, with a larger radius required to *switch* direction. Requiring
24 px before any feedback appears would mean a third of a cell of dead travel. Note also that the 100–200 px swipe
thresholds found in some gesture libraries target full-screen page swipes and are wrong by an order of magnitude
for a gesture inside a 64 px cell.

---

## 11. Open questions

**None blocking.** All ten decisions are resolved (§3). Every question raised during planning closed — two of
them by being answered *wrongly first*, which is recorded here rather than quietly overwritten:

| question | resolution |
|---|---|
| D1–D10 | §3 decision table |
| Is anything deployed from WebGames? | **No** — owner, 2026-09-11 (§6.2) |
| Lunar Lockout *rules* unverified | **Verified** against ThinkFun's published instructions (§5.2) |
| Lunar Lockout *levels* | **13 of 15 unplayable**; ladder regenerated (§5.3) |
| Can a BFS with a visited set be written in `.pg`? | **Yes, no compiler change needed** (§8.3) |
| Is the Block Dude state space BFS-tractable? | **Only levels 1–3.** Frontier ≈ ≤6–7 blocks over ≤150 free cells; memory-bound (§4.5.1) |
| Is Block Dude terrain a height profile? | **No — every original level has overhangs (18–68).** The earlier "yes" was measured on the WebGames synthetic levels and was wrong (§4.4a) |
| Can level 11 be solved exactly? | **No, impossible** — implemented and playable, but a stretch benchmark only (§8.5) |
| Is a new search engine needed? | **No** — `Core\Planning` is already game-agnostic (§8.6) |
| Cell occupancy model | **Single-occupancy, player-in-door the sole exception** — owner ruling, superseding my shared-occupancy recommendation (D1) |

### 11.1 Two claims this document got wrong, and what they cost

Kept deliberately, because both were stated with more confidence than the evidence supported:

1. **"Block Dude terrain is a 1D height profile."** Measured on the wrong sample (WebGames synthetic, not the
   originals). It produced a generator design that cannot emit shipped-level terrain, a state encoding sized
   for a nonexistent board shape, and it misled a later investigation into proposing height-profile
   heuristics. §4.4a.
2. **"Defect 5: gravity is never applied after a climb."** Listed as a defect, then justified as
   *load-bearing* — "it is how you climb a stack you just built." Both wrong: the climb lands the player
   directly on the cell whose solidity it just required, so he is supported by construction and a gravity pass
   there could never do anything. Reclassified as a vacuous non-issue. §4.3.

The general lesson, worth keeping: a measurement is only as good as the sample it was taken on, and a *reason*
invented after the fact for an observed behaviour is not evidence that the behaviour matters.
