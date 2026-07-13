# CLAUDE.md — operational rules for this repo

Project-level instructions loaded every session. Read `docs/ARCHITECTURE.md` for the code map and
`docs/prd/PLAN.md` for milestone history.

## Running / verifying the web app (RLDemo.Web) — READ THIS FIRST

**To build, serve, run, or visually verify the frontend, do exactly one thing:**

```bash
dotnet run --project src/RLDemo.Web        # Development profile; http://localhost:5210
```

The ASP.NET Core host **builds and serves the Angular frontend itself** — in Development it spawns and
proxies the Angular dev server (`UseAngularCliServer` via `UseSpaImproved` in `Program.cs`). There is
**nothing else to do for the frontend.**

- **Never** run `ng serve` / `npm start` / `ng build` / `ng test` yourself. The host already runs one; a
  second instance just fights for ports and can wedge the dev-server file watcher.
- **To see a code change:** just save the file under `ClientApp/src` — the running host live-reloads the
  browser. No manual build, usually no manual reload.
- **To verify what's actually served** (suspected staleness): `curl -sk http://localhost:5210/main.js | grep <identifier>`
  — do not reach for `ng build` to "check".
- If output looks stale, suspect a wedged dev-server watcher: **restart the ASP.NET host**, never `ng build`.

## Build failures are usually NOT the frontend

If `dotnet run --project src/RLDemo.Web` fails, read the error before touching anything Angular-related.

**Fixed (2026-07-13, Polyglot 0.6.0): the multi-`.pg` incremental-rebuild codegen bug** (`CS0260` "missing
partial modifier on PolyglotProgram" / `CS0101` duplicate `Option`/`Some`/`None` prelude). It was an MSBuild
`.targets` bug — a single-`.pg` edit made MSBuild's partial-incremental build hand the transpiler a subset,
which then emitted a standalone/duplicate prelude. `MintPlayer.Polyglot.MSBuild` **0.6.0** (PR #26, stamp
`Outputs` + `RemoveDir`) re-transpiles the full `.pg` set on any edit, so this no longer occurs. If you somehow
hit it on an **older** package, the manual recovery was
`rm -f src/MintPlayer.AI.ReinforcementLearning.Environments/obj/*/net10.0/polyglot/*.cs` then rebuild. History +
root cause: `docs/prd/polyglot-pilot/POLYGLOT_TOPLEVEL_RECORD_BUG.md`. Never loop build retries against stale output.

## Stopping a `dotnet run` host

`dotnet run` is a runner that spawns the app (`<App>.exe`) as a child. To stop it, kill the runner's whole
process tree (the background shell/task that launched it), not just the child exe — otherwise the next build
fails with a DLL lock (MSB3027/MSB3021). Verify nothing is left with
`Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'dotnet.exe','RLDemo.Web.exe' }`.
