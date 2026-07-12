# Checkpoints

Versioned, resumable persistence for training runs and deployable models. Everything here sits
between the in-memory runtime objects (networks, optimizer, agents) and raw `Stream`s; the
`FileModelStore` then puts those streams on disk atomically.

## When do I use which?

One rule covers most cases:

- **Resuming training** → save the *composite state* (`DqnTrainingState` for DQN, `TabularCheckpoint`
  for tabular). It carries everything needed to continue *bitwise-identically* — nets, optimizer,
  replay buffer, RNG streams.
- **Shipping a trained brain for inference** (the web app, an eval-only run) → save the *individual
  net* checkpoint (`DuelingQNetCheckpoint` / `MlpCheckpoint` / `ResidualMlpCheckpoint`). Just shape +
  weights; no optimizer or buffer.

The campaigns do both: they persist the full resume state under `<env>/<algo>-state` and, keep-best
gated, the deployable net under `<env>/dqn` — the id the web `ModelService` loads.

## The types

| Checkpoint | Serializes | Standalone or embedded | Format |
|---|---|---|---|
| `MlpCheckpoint` / `ResidualMlpCheckpoint` / `DuelingQNetCheckpoint` | one network's architecture + weights | standalone deployable file **or** embedded via `QNetCheckpoint` | binary |
| `QNetCheckpoint` | any `IValueNet` (type-tagged: `Mlp` \| `DuelingQNet`) | embedded (inside a training state) | binary |
| `AdamCheckpoint` | optimizer hyperparameters + moment estimates — needs the net's parameters to reload | embedded only | binary |
| `ReplayBufferCheckpoint` | DQN replay transitions | embedded only | binary |
| `DqnTrainingState` | **the whole DQN run** — online+target nets + Adam + replay buffer + RNGs + env snapshot | top-level resume file | binary |
| `TabularCheckpoint` | a Q-table | standalone resume **and** deploy (small enough to be both) | JSON (human-inspectable, lossless) |

`CheckpointFormat` holds the shared binary primitives (4-byte magic, kind + version header, float/int/
bool arrays, RNG state). Every binary checkpoint above is built from it, so they share one on-disk
convention and one versioning scheme.

## Persistence seam

`IModelStore` / `FileModelStore` (`ModelStore.cs`) is the file layer: one *current* checkpoint per
`(environmentId, algorithmId)` pair, written to `<root>/<env>.<algo>.ckpt` via a temp-file + rename so
a crash mid-save never corrupts the previous checkpoint. It deals only in raw `Stream`s and knows
nothing about net types — the `*Checkpoint` classes above own that knowledge.

## Versioning

Each checkpoint kind carries its own format version and stays backward-compatible: a newer reader
loads older files (e.g. `DqnTrainingState` v2 files load with a default noise RNG; `DuelingQNet` v1
files load as plain non-noisy nets). Bump the version and branch on it in `Read`/`Load` when you add a
field — never reorder existing fields.
