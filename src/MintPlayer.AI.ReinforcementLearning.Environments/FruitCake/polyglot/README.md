# FruitCake solver — single source (MintPlayer.Polyglot)

`fruitcake_solver.pg` is the **single source of truth** for the FruitCake circle-physics solver, transpiled to
C# (this project) and TypeScript (the web client, PG2). See `docs/prd/POLYGLOT_FRUITCAKE_PRD.md`.

- **`fruitcake_solver.pg`** — the source. Edit this; there is no committed generated `.cs`.
- **Build-time transpilation.** `MintPlayer.Polyglot.MSBuild` (PackageReference in this project, `PrivateAssets=all`)
  transpiles every `**/*.pg` to `obj/…/polyglot/*.cs` **before CoreCompile**, so the generated types are compiled into
  this assembly (and appear in IntelliSense). Nothing generated is committed. The 0.1.3+ CLI is bundled for
  `win-x64` / `linux-x64` / `linux-arm64` (Windows dev + Linux CI/deploy); **macOS** dev must point `$(PolyglotTool)`
  at a local `polyglot` binary (from the GitHub release). `dotnet watch` re-transpiles on save.
- The generated C# is `#nullable enable` with faithful annotations (0.1.3), so it compiles clean under this project's
  `<Nullable>enable</Nullable>` — no `#nullable disable` shim.
- Generated types are `Pg`-prefixed and **internal**; the public `FruitCakeWorld`/`FruitBody` **facade**
  (`../FruitCakeWorld.cs`) wraps them (float view over the f64 core) and adds host-only helpers
  (danger/eject/PileHeight/Clone/LoadBody, and body-state Save/Restore at double precision). A parity test
  (`tests/…/PolyglotSolverParityTests.cs`) pins the facade to the generated core.

## Working on the solver
Just edit `fruitcake_solver.pg` and build (`dotnet build`) — the transpile runs automatically. To transpile by hand
(e.g. to inspect the output or emit the TypeScript twin):

```
polyglot build fruitcake_solver.pg --target csharp --out <dir>     # C# (what the build does)
polyglot build fruitcake_solver.pg --target typescript --out <dir> # TS twin (PG2)
```

Run the parity test after changing the solver: `dotnet test --filter FullyQualifiedName~PolyglotSolverParityTests`.
