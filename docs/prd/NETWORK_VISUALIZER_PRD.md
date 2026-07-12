# Network Visualizer — see the net, and watch it learn — PRD

**Status:** in progress · 2026-07-12 · branch `m36-network-visualizer` (off `master`)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M36 · **Depends on:** the Core NN + checkpoint layer (§2/§5 of [../ARCHITECTURE.md](../ARCHITECTURE.md)) and the Lab training harness (§8) — additive, no change to existing training behaviour.

## 1. Problem

The project trains a family of neural networks (`Mlp`, `DuelingQNet`, `ResidualMlp`, the two-headed policy nets) and
ships them as versioned binary `.ckpt` files, but a network is only ever visible as **numbers**: a checkpoint on
disk, a CSV of eval metrics, a line of console output per evaluation. There is no way to *see* a network — its
shape, the structure in its weights — and, more importantly, **no way to watch a network change as it is being
trained.** You start a campaign, wait minutes per eval, and read a scalar. The learning itself is invisible.

We want two things:

1. **Inspect a `.ckpt`** — load any shipped checkpoint and see its architecture (layers, widths) and its weights.
2. **Watch a network evolve while it trains** *(the priority)* — a live view, updating continuously as the training
   loop runs, so you can literally see the weights move from random init toward a learned policy.

## 2. Goal & success criteria

- **Live gate (the priority, human judgement, in-browser):** start any game's training run with `--viz`; open the
  served page; **the network's weights visibly change as training proceeds** — edges and per-layer heatmaps repaint
  on a steady cadence, and the metric read-outs (step, loss, eval, ε) advance. Opening the page *mid-run* immediately
  shows the current graph, then continues live. Closing/refreshing the browser never disturbs training.
- **All games:** `--viz` works for **every trainable game** (`snake`, `fruitcake`, `rushhour`, `cube`, `cube-policy`,
  `cube-davi`) — DQN, imitation-policy, EfficientCube-policy, and DAVI value nets alike — through one generic seam,
  with no per-trainer code.
- **Beginner-friendly & environment-aware:** hovering any neuron, connection, or heatmap shows a plain-language
  tooltip. Where the environment supplies semantics (FruitCake does), each **input** names the exact observation
  feature it is (e.g. "Column 7: surface height") with its **current value**, and each **output** names the action
  it controls ("Drop the current fruit in column 7 of 14") with its **live Q-value** — so a newcomer can read not
  just the shape but what the net is actually seeing and deciding right now. Environments without labels fall back to
  generic per-role tooltips.
- **Dev-only, gated:** the live socket only comes up in a **Development** host environment; a Production-configured
  process ignores `--viz` (with a note). It is never part of the deployed web app.
- **Zero training impact:** attaching the visualizer leaves training **bitwise-identical** (verified: viz vs no-viz
  checkpoints are SHA256-equal) — sampling is a read-only observation of the parameter tensors on a background
  thread; it never touches the RNG, optimizer, or net, and costs nothing while no browser is connected.
- **Fully async & dependency-light:** no blocking network I/O on any path (per-viewer async send queue + async
  sample loop), and no new NuGet/npm packages (uses `HttpListener`'s built-in WebSocket support + the browser's
  native `WebSocket`); the viewer page is a single self-contained HTML file (inline CSS/JS, no external requests).

**Non-goals (v1).** No editing/altering weights from the UI. No viewer→trainer control messages *yet* (the WebSocket
is chosen so they can be added without changing transport). No gradient-flow animation or **hidden-neuron**
activation tracing (input/output values are shown; interior activations aren't exposed — see §7). No 3-D layout. No serving the live stream from the public
web app (training is dev-side; see §3). No diffing two checkpoints. Continuous-control PPO/SAC (Pendulum, MountainCar)
aren't trained through the Lab's `--game` dispatch, so they're out of scope until they are.

## 3. Key decision — stream from the *training process*, not the web app

An investigation (4 parallel agents, 2026-07-12) established the load-bearing constraints:

- **The web app never trains and has no server WebSocket infra.** `RLDemo.Web` is load-only; the
  server-authoritative "Watch AI" WebSocket stack was deliberately **removed in M32/M33** — every game now computes
  client-side over REST. Reintroducing server sockets into the web app to watch *training* would cut against that
  grain and, worse, the web app isn't where training happens.
- **Training lives only in the Lab CLI** (`tools/…Lab`, `CampaignRunner` on `AIHost`). That process owns the net,
  the optimizer, and the loop — it is the only place the weights actually move.
- **Reading weights mid-training is free on the CPU path.** Every game net's parameters are host-resident
  `float[]` (`Tensor.Data`); reading them perturbs nothing. (The GPU-resident DAVI cube path keeps a CPU master net
  that's synced periodically — the viewer reads that, so it shows the last-synced weights with no device lock.)

So the live view is **served by the training process itself**: the Lab hosts a tiny `HttpListener` on localhost
that serves one HTML page (`GET /`) and streams telemetry to it over a **WebSocket** (`GET /ws`). The static-inspection
mode has a different, equally natural home: the browser (**Angular `/network` route**) already ships `.ckpt` parsers
(`snake-net.ts` etc.), so it can inspect a checkpoint entirely client-side with no server at all.

**Transport — WebSocket (owner's call, 2026-07-12).** Telemetry today is one-way (server→client), which Server-Sent
Events models with the least code — no handshake, browser auto-reconnect for free. We chose a **WebSocket** anyway,
deliberately, because the channel is meant to become **bidirectional**: the viewer should be able to send control
messages back to the training run (pause/step, change the frame cadence, select a layer, request an on-demand
snapshot) without swapping transports later. `HttpListener` supports the server side natively
(`AcceptWebSocketAsync`) and the browser has `WebSocket` built in, so there's no new dependency; the only cost over
SSE is the framing/lifecycle we manage in `VizServer` (send serialized per socket + bounded by a timeout, a receive
loop that also drains future control messages, and a small client-side reconnect). The removed-in-M32/M33 server WS
stack was *web-app* infra; this is a dev-only server in the training process and doesn't reintroduce it there.

**Design-it-twice note.** The other alternative — persist per-step snapshots to a JSONL/`.ckpt` sidecar and have a
separate viewer tail the file — was rejected for the live path: it doubles the I/O, adds a polling viewer, and still
needs a transport for "push me the latest." Streaming from the training process is strictly simpler and genuinely
live. The sidecar idea survives only as the trivial static case (read one `.ckpt`).

## 4. Design — a read-only *pull* seam + a viewer served by the trainer

### 4.A Core: the telemetry source (`Core/Telemetry/NetworkTelemetry.cs`)

A **pull** model, not a push: the campaign publishes what it has; a viewer samples it on its own cadence. This is
what makes "all games" free — no trainer calls anything.

```csharp
public interface INetworkTelemetrySource
{
    string NetKind { get; }                        // e.g. "dueling-q", "cube-policy", "value-davi-res"
    IReadOnlyList<Tensor>? SnapshotParameters();   // current weights+biases, or null until the net exists
    NetworkMetrics Sample();                        // step / maxSteps / loss / eval / epsilon (NaN where N/A)
    // Optional semantics (default null → generic tooltips):
    IReadOnlyList<string>? InputLabels => null;     // name of each input feature
    IReadOnlyList<string>? OutputLabels => null;    // name of each action/output
    (float[] Input, float[] Output)? SampleIo() => null;  // current observation + the net's output for it
}
```

The label/IO members are **default-implemented** (null), so a game opts in without every other campaign changing.
`FruitCakeEnv` publishes `ObservationLabels` (all 89 features, mirroring `fruitcake_solver.pg`) + `ActionLabels`
(the 14 drop columns); `FruitCakeDqnCampaign.SampleIo()` returns the most-recent observation and the net's forward
pass on it (per-column Q-values). The forward is read-only (no `Backward`) so it can't perturb training — verified
SHA256-identical **with a viewer connected** during a run.

- `NetworkInspector.Describe(parameters, kind)` / `.CaptureFrame(parameters, metrics)` turn **any** net's parameter
  tensors into telemetry by pairing each rank-2 weight with the rank-1 bias that follows it — the layout every net in
  the library emits — so no per-architecture code is needed. Keying off `Parameters()` (not a concrete type) is why
  the non-`IModule` policy nets (`CubePolicyNet`, `RushHourPolicyNet`) work too.
- A frame carries, per layer: weight min/max/mean-|w|/L2, bias mean-|w|, and a **downsampled magnitude heatmap**
  (block-averaged |w| capped at 24×24). The cap makes a frame's size independent of the true matrix size — a
  1024-wide layer streams the same few KB as a 16-wide one. The source, not the viewer, bounds the payload.

### 4.B Campaign wiring — one interface, six games

Each `ITrainingCampaign` also implements `INetworkTelemetrySource` in ~4 lines: return the live net's parameters
(`_state.Online` for DQN, `_net` for the policy/DAVI campaigns) and a `NetworkMetrics` from fields it already tracks
(step, last loss, eval). No trainer changes at all. Sampling reads the parameter arrays on the viewer's background
thread — a benign race with the training thread's writes (a torn float only flickers one heatmap pixel; never a
fault), and provably harmless to training (no writes, no RNG, no ordering) — hence bitwise-identical checkpoints.

### 4.C Live viewer — the Lab's `--viz` server (`tools/…Lab/VizServer.cs` + `VizLauncher.cs`)

`VizServer` starts an `HttpListener` on `http://localhost:<port>/` and runs a background **sample loop** that pulls
the source every ~150 ms and broadcasts:

- `GET /` → one self-contained HTML page (Canvas 2D): a node-link graph (layers as columns of neurons, capped for
  display; edge brightness = weight magnitude) + a per-layer weight-heatmap strip + a loss sparkline, and — for
  newcomers to neural nets — a **hover tooltip** on every neuron / connection / heatmap. Labeled input/output
  columns are drawn in full (not capped) so each neuron is individually hoverable and shows its name + live value;
  the canvas grows in height to fit. Unlabeled/hidden columns stay capped with generic wording.
- `GET /ws` → the WebSocket. Messages self-describe (`{"type":"topology"|"frame","data":…}`). Topology is broadcast
  only when the shape changes (e.g. a DAVI widen/grow); the latest is retained so a mid-run joiner gets the graph at
  once. The client auto-reconnects on drop.
- **Fully async, fault-tolerant:** each viewer has a bounded outbound `Channel` drained by an async send pump (no
  blocking sends; a slow viewer just drops stale frames), and the sample loop does nothing — costs nothing — while
  no browser is connected.

`VizLauncher.TryStart(args, campaign, env)` is the shared `--viz [port]` handler used by every Lab: it parses the
flag, **requires a Development environment** (else prints a note and skips), checks the campaign is a telemetry
source, and starts the server. Wired into all six games — `--game <snake|fruitcake|rushhour|cube|cube-policy|cube-davi> --viz`.

### 4.D Static viewer — Angular `/network` (follow-on phase)

A standalone `network-visualizer` component + lazy route + home card, reusing the existing browser `.ckpt` parsers
to render the same graph/heatmap for a chosen shipped checkpoint — client-side, no server. Shares the visual
language of the live page.

## 5. Milestone plan

- **M36.1 — live during training, all games, beginner tooltips (this PRD's gate; SHIPPED).** Pull-based Core seam +
  every campaign as a source + the async Lab `--viz` viewer with hover tooltips, gated to Development. **Gate (met):**
  `--game snake --viz` → the net visibly evolves in-browser; tooltips explain each part; a Production process skips
  the socket; viz vs no-viz checkpoints are SHA256-identical.
- **M36.2 — static `.ckpt` inspection in the web app (planned).** Angular `/network` route reusing the browser
  `.ckpt` parsers to render the same graph/heatmap for a chosen shipped checkpoint, client-side.
- **M36.3 — richer live view (optional, planned).** Signed (diverging) heatmaps; viewer→trainer controls over the
  existing WebSocket (pause/step, cadence, layer-select); activation tracing (needs a `Forward` hook, see §7);
  continuous-control PPO/SAC once they train through the Lab.

## 6. Risks / watch-items

- **`HttpListener` prefix permissions.** `http://localhost:<port>/` is allowed for non-admin users on Windows
  (unlike `http://+:port/`); keep it localhost-only.
- **Concurrent read race.** Sampling reads weights on a background thread while training writes them — chosen
  deliberately (it keeps the seam out of the trainers) and benign for a magnitude heatmap; it is provably harmless to
  training output (SHA256-verified). Not suitable if we ever needed a *consistent* snapshot (we don't).
- **Frame rate vs. throughput.** ~150 ms cadence → a few Hz; capture is cheap (stats + ≤24² downsample) and skipped
  entirely when no browser is connected. A very large net (DAVI 34 MB) allocates a params snapshot per sampled frame
  — fine for a dev tool, and only while someone is watching.
- **Sign is dropped in the heatmap.** v1 shows |w| magnitude (bounded, always positive); signed diverging colour is
  a possible M36.3 refinement.
- **GPU-resident nets.** The cube DAVI net's device weights are mirrored by a periodically-synced CPU master; the
  viewer reads that master (last-synced weights) — correct and lock-free, just slightly behind the device.

## 7. Honest ceiling

This shows **weights** evolving, which is what "watch it learn" most directly means and what the tensors make cheap
to observe. It does **not** show per-neuron **activations** or gradient flow — those intermediate tensors aren't
currently surfaced by the forward pass (they'd need an interception hook in each net's `Forward`). That's a
deliberate v1 boundary, not an oversight; activation tracing is a clean follow-up once the weight view proves its
worth.

## 8. Sources

Investigation reports (2026-07-12): checkpoint/model layer, web/WebSocket surface, training-loop hooks, and docs
conventions — summarized inline above. Code references: `Core/Nn/Modules.cs`, `Core/Nn/DuelingQNet.cs`,
`Core/Training/DqnTrainer.cs`, `Core/Training/CampaignRunner.cs`, `tools/…Lab/SnakeDqnCampaign.cs`,
`RLDemo.Web/ClientApp/src/app/*/*-net.ts`, and `../ARCHITECTURE.md` §2/§5/§6/§8.
