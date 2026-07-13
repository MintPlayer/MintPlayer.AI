# Play the chess AI in the browser — single-source via MintPlayer.Polyglot — PRD

**Status:** Planned · 2026-07-12 · branch TBD (off the M39 branch / `master`)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M40 · **Depends on:** M39 (the chess engine, the self-play-trained `PolicyValueNet`, the `.ckpt` format) and the MintPlayer.Polyglot single-source toolchain already used by FruitCake ([../ARCHITECTURE.md](../ARCHITECTURE.md) §10; `Environments/FruitCake/polyglot/`).

## 1. Problem

You can *watch* a game the chess AI played (a replay), but you can't *play against it in the browser* — there's no web chess page, and the AI is C# (net + MCTS), so a static page can't run it. The wrong fix is a server `ChessController` that runs MCTS per move (server inference, per-viewer CPU — the exact thing M32/M33 removed for FruitCake/Snake). The **right** fix is the pattern MintPlayer.Polyglot exists for and FruitCake already proves: **write the inference path once in a `.pg`, transpile to C# (training/serving) and TypeScript (browser), and run the whole AI client-side, with the browser downloading and parsing the `.ckpt`.** Zero server inference.

## 2. Feasibility — verified

The open question was whether chess's inference math and move-gen fit Polyglot's subset. Probed by transpiling (2026-07-12):

- **`Math.exp`, `Math.tanh`, `Math.log`, `Math.sqrt` all transpile** (with `import { Math } from "std.math"`) — so MCTS's masked-softmax priors, the `tanh` value head, and the PUCT `sqrt` term are all expressible. (FruitCake avoided transcendentals because its physics must be *bit-exact* across C#/JS; only `+ - * / sqrt` are. **Chess inference does not need bit-exactness** — the browser AI just needs to play well, not match C# to the ULP — so non-bit-exact `exp`/`tanh` are fine here.)
- **Bitwise operators `& | << >>` transpile** — castling-rights flags and any bit tricks work (booleans are also an option).
- **Polyglot itself is extendable if a gap appears.** The language/toolchain lives at `C:\Repos\MintPlayer.Polyglot`; a missing feature (e.g. a `match`/`switch`, an enum, a Math function) can be added there and PR'd. The known limits to design around (from the FruitCake solver): **no nested-generic params** (`List<List<f64>>` is mishandled — use flat `List<f64>` + offsets, as `PgDuelingNet` does), and prefer `i32` constants + `if/else` over enums/`switch` unless we add them upstream.

**Conclusion:** the chess engine + observation + net forward + MCTS can be single-sourced to `chess_solver.pg` and run client-side. This is a build, not a research risk.

## 3. Goal & success criteria

- **Play a full game against the self-taught AI in the browser** — an interactive board (you're White; click/drag to move, legal moves highlighted), the AI (Black) replies from a `.pg`-single-sourced net + MCTS running **in your browser**, over the shipped `.ckpt` it downloads. No server move computation.
- **Single source.** The chess **engine** (legal move-gen, apply, terminal, move encoding), the **observation**, the **net forward**, and the **MCTS** live once in `chess_solver.pg`, transpiled to C# (compiled into the Environments assembly, like FruitCake) and to committed TypeScript for the web client. A parity test pins the C# facade to the generated core; **perft still passes 25/25 against the generated engine** (the M39.2 gate carries over onto the single source).
- **Zero server inference / load-only web app.** The web app only serves the `.ckpt` (static file, `application/octet-stream` mapping like the FruitCake net) and the SPA. Per-viewer server cost is zero — consistent with M32/M33.
- **Honest scope.** It's the M39 network — a small MLP, briefly CPU-trained — so it plays *legal, still-learning* chess, beatable by a decent human. Browser MCTS is tuned (modest sims) for interactive latency (~1–2 s/move).
- **Training unchanged.** Training stays on the SDK (autograd/GEMM); the `.pg` is **inference + rules** only. The engine (generated C#) is shared by training's self-play; the net *forward* in the `.pg` is inference-only (training learns the weights via autograd, exactly as FruitCake).

**Non-goals.** Engine strength; a server chess endpoint; bit-exact C#/JS transcendentals (not needed — this isn't a shared fairness sim); underpromotion-free or reduced rules (the engine is already full + perft-verified); training in the browser.

## 4. Design — follow the FruitCake pattern

| Piece | Where | Note |
|---|---|---|
| `chess_solver.pg` | `Environments/Chess/polyglot/` | **single source:** the engine (a `Pg`-prefixed board + `legalMoves`/`apply`/`result`/`isAttacked`), `writeObservation`, `PgPolicyValueNet.forward` (flat-array trunk + policy/value heads, `tanh` value), the 4672 `encode`/`decode`, and a chess `mcts` (PUCT, masked-softmax priors, `sqrt` exploration, **no Dirichlet** — inference only). Transpiled to C# (`obj/`, build-time) + committed `chess_solver.ts`. |
| C# facade | `Environments/Chess/ChessGame.cs` (rewired) | Wraps the generated `Pg` engine as `IZeroSumGame<TState>` (float/host view), so M39's `SelfPlayCampaign`/`ChessGame` keep working on the single source. Replaces the hand-written `ChessBoard.cs` engine internals; **perft tests re-point at the generated core**. |
| TS `.ckpt` parser | `ClientApp/src/app/chess/chess-net.ts` | Mirrors the C# `PolicyValueNet` reader (magic `RLNC`, kind `selfplay-pv`, v2 trunk-widths, params in `Parameters()` order) → builds the generated `PgPolicyValueNet`. The one non-Polyglot piece (binary I/O), exactly like `fruitcake-net.ts`. |
| Chess director | `ClientApp/src/app/chess/chess-director.ts` | Runs the transpiled `mcts` + net over the loaded `.ckpt` to pick the AI's reply — the analog of `fruit-cake-director.ts`. |
| Angular page | `ClientApp/src/app/chess/` + route/nav | Interactive board (unicode glyphs or simple SVG), human move input validated by the transpiled `legalMoves`, AI reply from the director, status (check/mate/draw), last-move highlight. |
| Shipped weights | `wwwroot/models/chess.az.ckpt` (LFS) + the `.ckpt` MIME mapping | Fetched by the browser; the checkpoint the director loads. |

## 5. Phased plan (each ends on a green build + its gate)

- **M40.1 — single-source the engine.** Port `ChessBoard`/`ChessRules` + `ChessMoveEncoding` into `chess_solver.pg`; wire the Polyglot MSBuild transpile; rewire `ChessGame`/facade onto the generated core; **re-run perft (25/25) + the encoding round-trip on the generated engine.** De-risks the biggest port (rules) first; training self-play still passes its contract test.
- **M40.2 — single-source the inference math + TS parser.** Add `PgPolicyValueNet.forward` (flat arrays, `tanh`), `writeObservation`, and a chess `mcts` to the `.pg`; a `chess-net.ts` `.ckpt` parser; a parity test (C# `PolicyValueNet.Forward` vs the generated `PgPolicyValueNet` on a fixed position, within f32 tolerance). Emit the committed `chess_solver.ts`.
- **M40.3 — the browser page.** `chess-director.ts` + the Angular chess component (board, input, AI reply, status) + route/nav; ship `wwwroot/models/chess.az.ckpt`; the `.ckpt` static-file mapping. **Gate:** play a full legal game vs the AI in the browser with no server move calls (verified via the network panel / Playwright); tune sims for latency. A longer training run first, so it's a worthier opponent.

## 6. Risks

1. **Browser MCTS performance.** Chess move-gen per node is heavier than FruitCake's depth-3 search; hundreds of sims/move in transpiled TS may be slow. Mitigation: modest inference sims (100–300), a ply/time budget, and profile — it's a latency knob, not a correctness one. (Batched-leaf / WASM are later levers.)
2. **Polyglot language gaps.** The port may want a `switch`/`match`, enums, or a Math function not yet in `std.math`. Mitigation: design around them (i32 consts + `if/else`, flat arrays — proven by FruitCake) **or add them to `C:\Repos\MintPlayer.Polyglot` and PR** (now an available option). No nested-generic params.
3. **Re-validating the engine after the port.** Rewriting a perft-verified engine in another language risks a subtle regression. Mitigation: perft (25/25) + the encoding round-trip run against the *generated* engine as the M40.1 gate; a facade-vs-core parity test.
4. **Honest strength.** Still a small, briefly-trained net → beatable. Framed as a from-scratch, self-taught, learning AI (the charm), not an engine.

## 7. Verification

- **M40.1:** solution builds (the `.pg` transpiles into the assembly); **perft 25/25** + encoding round-trip on the generated engine; `SelfPlayCampaign` chess contract test still green.
- **M40.2:** C#-vs-generated `PgPolicyValueNet` parity within f32 tolerance on fixed positions; `chess_solver.ts` emitted + committed; the TS parser round-trips a shipped `.ckpt` (a small Node/Jest or a Playwright check).
- **M40.3:** in-browser, a full legal game vs the AI, **no `/api/chess/*` move calls** (Network panel/Playwright), check/mate/draw detected, last-move highlight. Honest latency (~1–2 s/move at the chosen sims).

See [PLAN.md](PLAN.md) M40.

## 8. Reference appendix (execution details — read this if picking up cold)

### 8a. What M39 already shipped (branch `m39-chess-selfplay-plan`, PR #32, stacked on the M38 branch #31)
The chess self-play stack is **done and committed**; M40 builds on it. Key artifacts:
- **Reusable seam/search/training (Core + Lab):** `Core/Planning/IZeroSumGame.cs` (+ `GameResult`), `Core/Planning/Mcts.cs`
  (PUCT: select `Q+U` → expand-with-priors → net-leaf → **value negated every ply** → `Search` returns the root
  visit-count π; Dirichlet root noise + `Config(Simulations, Cpuct=1.25, DirichletAlpha=0.3, RootNoiseFrac=0.25)`),
  `Lab/PolicyValueTraining.TrainStep` (soft-CE(π) + MSE(`tanh` value, z)), `Lab/SelfPlayCampaign<TState>`
  (`ITrainingCampaign`; plays MCTS self-play → `(obs,π,z)` window → trains `PolicyValueNet`; `--opponent-random` frac
  mixes learner-vs-random games; store ids below).
- **Chess (Environments/Chess/):** `ChessBoard.cs` (`ChessState` + `ChessRules`: `LegalMoves`, `MakeMove`,
  `IsSquareAttacked`, `InCheck`, `Result`, `Perft`; mailbox `sbyte[64]`, sq = rank*8+file, White moves +rank; piece
  codes ±1..6 = P,N,B,R,Q,K; castling byte bits WK=1,WQ=2,BK=4,BQ=8; **threefold repetition NOT modelled** — 50-move +
  ply cap bound loops), `ChessFen.cs`, `ChessMoveEncoding.cs` (AlphaZero **4672 = 64×73**: 56 queen planes [8 dir ×7
  dist] + 8 knight + 9 underpromotion; queen-promo rides the queen planes, decoded promo inferred on apply),
  `ChessGame.cs` (`IZeroSumGame<ChessState>`; `PolicySize=4672`, `ObservationSize=1152` = 18 planes×64: [0–5] White
  P,N,B,R,Q,K, [6–11] Black, [12] side-to-move, [13–16] castling WK/WQ/BK/BQ, [17] en-passant square).
- **Lab:** `--game chess` (`ChessLab.cs`, flags `--sims --games --eval-games --hidden --opponent-random`),
  `--game chess --demo` (`ChessDemo.cs`: net as White + MCTS vs random Black, prints one FEN per ply to stdout),
  `--game connect4` (`Connect4Lab.cs` + `Environments/Connect4/`).
- **Net:** the shared `Core/Nn/PolicyValueNet.cs` (variable-depth ReLU trunk → policy logits + scalar value), used
  directly (no chess wrapper). **Value head is linear** — self-play wraps it in `tanh` at the call site.
- **Tests:** `ChessPerftTests` (25/25, incl. startpos d5=4,865,609, Kiwipete d4=4,085,603), `ChessEncodingTests`
  (encode→decode→apply round-trip + mate/stalemate), `Connect4Tests` (MCTS-vs-negamax), `SelfPlayCampaignTests`
  (resume roundtrip + vs-random). 355 fast + deep-perft (Slow).

### 8b. Model-store ids / checkpoint (what the browser must parse)
`SelfPlayCampaign` saves to store `(env, algo)`: net = `("chess","az")` written `PolicyValueNet.Save(stream, kind:"selfplay-pv")`;
Adam = `("chess","az-adam")`. File on disk: `<data>/chess.az.ckpt`. **The TS `.ckpt` parser (chess-net.ts) reads
(little-endian, mirroring `CheckpointFormat` + `PolicyValueNet.Save`; see `fruitcake-net.ts` for the exact idiom):**
```
uint32 magic  = 0x434E4C52 ("RLNC")
string kind   = "selfplay-pv"   (BinaryWriter 7-bit-encoded-int length prefix + UTF-8 bytes)
int32  version (= 2)
int32  trunkCount, then trunkCount × int32   (the hidden widths, e.g. 256,256)
per layer, in Parameters() order = [each trunk layer, then policyHead, then valueHead]:
    int32 wCount + wCount × float32   (weight, row-major [inDim, outDim] → w[i*outDim + o])
    int32 bCount + bCount × float32   (bias)
```
`inputSize` (1152) and `actions` (4672) are **not** stored — supply them from `ChessGame` (they’re Load params).
Forward: `x = obs; for each trunk layer: x = relu(x·W + b); logits = x·W_pol + b_pol; value = tanh(x·W_val + b_val)`.
MCTS priors = masked-softmax of `logits` over legal moves; leaf value = `value`.

### 8c. Commands
```
# train chess (writes <data>/chess.az.ckpt; time-bounded; --opponent-random for robustness):
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game chess --hours 2 --sims 48 --games 10 --opponent-random 0.25 --data <dir> --seed 7
# watch a game (prints FENs; loads the checkpoint from <dir>):
dotnet run -c Release --project tools/…Lab -- --game chess --demo --sims 400 --demo-plies 120 --data <dir> --seed 11
# perft/encoding gate:  dotnet test --filter "FullyQualifiedName~ChessPerftTests"  (add Category!=Slow for shallow)
# Polyglot transpile (bundled CLI; win-x64):
#   ~/.nuget/packages/mintplayer.polyglot.msbuild/0.3.1/tools/win-x64/polyglot.exe \
#     build <file>.pg --target typescript --out <dir>   (and --target csharp)
```

### 8d. Polyglot facts (from the FruitCake precedent — `Environments/FruitCake/polyglot/`)
- Source: `fruitcake_solver.pg` (single source) → C# in `obj/` (build-time, MSBuild PackageReference
  `MintPlayer.Polyglot.MSBuild` v0.3.1, `PrivateAssets=all`) + committed TS `fruitcake_solver.ts`. macOS dev must
  point `$(PolyglotTool)` at a local `polyglot` binary. `dotnet watch` re-transpiles on save.
- Syntax: `import { List } from "std.collections"`, `import { Math } from "std.math"`; `record X(f: t)`; `class X { var f: t; init(...) {...}; fn m(...): t {...} }`; `fn f(a: i32): t => expr`; `for i in 0..n`, `for v in list`;
  types `f64 i32 bool List<T> T?`; `Math.min/max/sqrt/exp/tanh/log/PI`. Generated C# types are `Pg`-prefixed + internal,
  wrapped by a hand-written **facade** (`FruitCakeWorld.cs`); a **parity test** pins facade↔core.
- **Constraints:** no nested-generic params (`List<List<f64>>` mishandled → flat `List<f64>` + offsets, see
  `PgDuelingNet`); prefer `i32` consts + `if/else` over enums/`switch`. If blocked, extend Polyglot at
  `C:\Repos\MintPlayer.Polyglot` and PR (owner-authorized 2026-07-12).
- Browser wiring (analogs to build): `chess-net.ts` (`.ckpt` parser → `PgPolicyValueNet`), `chess-director.ts`
  (runs transpiled mcts+net), the Angular `chess/` component + route; ship `wwwroot/models/chess.az.ckpt` (LFS) + the
  `.ckpt`→`application/octet-stream` static-file mapping in `Program.cs` (see how FruitCake's net is served).

### 8e. Status of the "watch" demo (already delivered)
A browser **replay** artifact of one AI game exists (published via the Artifact tool from `--game chess --demo`
FENs). That is watch-only; M40 is the interactive *play-against-it* page. The demo net was trained ~21 min on CPU
(reached ~50% vs random, policy loss 4.4→2.2) — legal but weak; longer training recommended before M40.3.

## 9. Difficulty (M40.4) — design from the investigation team (2026-07-13)

**Goal:** let the visitor pick a difficulty in **both** modes (Play the AI, Watch AI-vs-AI). Three read-only agents
investigated the training/checkpoint machinery, the web surfaces, and the difficulty-composition design; this section
is the synthesis.

### 9.1 What difficulty is made of — decision

Compose difficulty from three possible axes, in priority order:

1. **Search budget (`sims`) on ONE net — the spine.** Already a live knob (`ChessDirector.sims` → `PgChessMcts.chooseMove(net,state,sims,cpuct)`). More sims = stronger, slower; it's the classic engine difficulty lever (search depth/time), monotonic and reproducible, bounded by the in-browser latency budget (~1–2 s/move). Zero download cost, zero `.pg` change.
2. **Move-selection temperature — for human-like *variety* at the low end.** `PgChessMcts.search` already returns the full **visit-count distribution π** over 4672; `chooseMove` is just its argmax. So temperature lives **caller-side in `chess-director.ts`**: sample `∝ πᵢ^(1/T)` over the **visited** moves (every candidate is one MCTS actually explored → varied but never "unexplored garbage"). `T=0` = today's argmax. **No `.pg` change and no RNG in the Polyglot core** (`Math.random()` is browser-only; the single source stays pure/deterministic). Optional stronger variety lever: top-k-visited sampling.
3. **Network ladder (different-strength checkpoints) — optional NOVELTY only.** The owner's idea. Verdict: *not* the difficulty spine. This is a small, briefly-trained MLP — intermediate checkpoints are noisy and **not reliably rankable** (an under-trained net has a near-uniform policy → plays *confused*, i.e. the "randomly dumb" feel we want to avoid at Easy), and each net is **~5–6 MB** LFS. Its real value is narrative: *"play the AI at an early stage of its self-taught learning."* Ship **at most two** nets (a "Rookie (early training)" novelty + the final), framed as a novelty, not a balanced tier.

**Why Easy should weaken via fewer sims (+ mild temperature), NOT a weaker net or high temperature:** fewer sims still plays the net's *considered* move (decent priors) — it loses *gracefully* (doesn't calculate deep tactics) rather than hanging pieces at random. A weaker net or high-T sampling both feel "randomly dumb." Floor: keep Easy at **low-but-nonzero** sims (~24) — `sims=0` returns raw priors (the `total==0` fallback) and loses the one-ply tactical safety net. At very low sims, visits concentrate, so temperature has little to sample among → Easy needs the *combination* (low-nonzero sims + slightly higher `cpuct` + `T≈1`).

**Honest scope / labels:** the strongest shippable config (best net, ~256–300 sims, `T=0`) is still club-beatable. Label the top tier **"Full strength"** (or "Hard") — **never "Grandmaster."** A longer training run lifts the whole ladder uniformly but doesn't change its composition.

### 9.2 Proposed ladder

| Tier | Checkpoint | sims | temperature | cpuct | Feel |
|---|---|---|---|---|---|
| Easy | final net | ~24 | ~1.0 | ~2.0 | fast (<0.5 s), varied, casual — misses tactics |
| Medium | final net | ~96 (current) | ~0.5 | 1.5 | ~1 s, mostly sensible, occasional miss |
| Full strength | final net | ~256–300 | 0 (argmax = today) | 1.25–1.5 | ~1–2 s, the net's genuine best |
| *Rookie (opt.)* | *early checkpoint* | ~64 | ~0.6 | 1.5 | *novelty: "watch it before it learned"* |

### 9.3 Delivery — a committed manifest (shippable now on ONE net)

Ship `wwwroot/models/chess-difficulties.json` (static; `.json` is already served — mirrors the `rushhour-deck.json` precedent), fetched by the director at load, with a hardcoded fallback:

```json
[
  { "label": "Easy",          "ckpt": "/models/chess.az.ckpt", "sims": 24,  "temperature": 1.0, "cpuct": 2.0 },
  { "label": "Medium",        "ckpt": "/models/chess.az.ckpt", "sims": 96,  "temperature": 0.5, "cpuct": 1.5 },
  { "label": "Full strength", "ckpt": "/models/chess.az.ckpt", "sims": 256, "temperature": 0.0, "cpuct": 1.5 }
]
```

**First cut needs only the one net we already ship** — the three tiers differ by `sims`/`temperature`/`cpuct`. Adding real multi-checkpoint tiers later is a manifest edit + dropping `.ckpt` files into `wwwroot/models/` — **zero code change** (`chess-net.ts` `loadChessNet(url)` is already parameterized and reads trunk widths from the file). Missing-ckpt caveat: the SPA fallback returns `index.html` with `resp.ok===true`, but the parser throws on the magic check → `catch` → `null` → the director's random-move fallback (silent degradation — surface it in `statusText`).

### 9.4 Code changes

- **`chess-director.ts`:** add `difficulties[]` + `current` (loaded from the manifest, hardcoded fallback); `setDifficulty(d)` sets `sims`/`cpuct`/`temperature` and re-fetches the net **only if the ckpt URL changed** (cache nets by URL, so sims-only switches don't refetch); in `aiStep()`, use `PgChessMcts.search(...)` + temperature sampling (`T=0` → keep `chooseMove` argmax).
- **`chess.ts`:** a difficulty selector (segmented control / dropdown) in **both** modes. Play = the opponent's strength. Watch = a **single shared level first**; per-side White/Black levels (the "strong vs weak" demo) is a clean follow-up (director holds two nets + picks by `whiteToMove`). Changing difficulty in watch mode should `loopGen++` + `reset()` (mirror `setMode`) and re-await readiness.
- **No change** to `chess-net.ts`, `Program.cs`, `CheckpointFormat`, `PolicyValueNet`, `chess_solver.pg`, or `.gitattributes`.

### 9.5 Capturing a net ladder (only if we do the Rookie/novelty or per-side tiers)

Training overwrites `chess.az.ckpt` in place ~every 10 min (`CampaignRunner` eval cadence); no history is kept, and the only strength signal is **winRate-vs-random** (saturating, noisy at 10 eval games) + games-trained — **there is no net-vs-net Elo/arena**. To capture a ladder:
- **Option B (zero code, recommended for a 1–2-net novelty):** manually copy `<data>/chess.az.ckpt` to a tier name (`chess.az.l1.ckpt`) at a chosen milestone.
- **Option A (automated):** a `--ladder-winrates`/`--ladder-games` flag in `ChessLab` → `SelfPlayCampaign.Checkpoint` dumps named snapshots (`store.Save(env, "az-tier{n}", …)`; the store already namespaces by algo id, so no store change).
- A **net-vs-net arena** (an earlier tier as opponent, paralleling `ArenaVsRandom`) is the one genuinely-new capability that would make tiers *reliably* ordered — deferred; not needed for sims+temperature difficulty.
