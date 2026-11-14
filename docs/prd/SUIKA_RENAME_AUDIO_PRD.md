# Suika: rename FruitCake → Suika, and the never-ending buzz

- **Milestone:** M71 (see `PLAN.md`)
- **Date:** 2026-11-14
- **Branch:** `suika-rename-and-music-fix`

## 1. Goals

1. Call the game by its original name. Every mention of *FruitCake* (types, namespaces, folders,
   files, route, campaign id, docs) becomes *Suika*.
2. Fix the bug where a buzzing sound starts when the first fruit lands and never stops.
3. Add a regression test for (2).

## 2. Rename

Done mechanically with a script, then hand-checked. The casing map:

| Before | After |
|---|---|
| `fruit-cake` (Angular folder and file kebab names) | `suika` |
| `FruitCake` / `FRUITCAKE` | `Suika` / `SUIKA` |
| `fruitcake` / `fruitCake` / `Fruitcake` | `suika` / `suika` / `Suika` |

- 40 tracked paths were moved with `git mv`, so history follows them. This covers the Angular folder,
  the `Suika/` folders in Environments, Campaigns and Lab, the `.pg` solver, the shipped
  `suika-net.ckpt`, the two test classes and the `SUIKA_*_PRD.md` docs.
- `pgconfig.json` routes `Suika/polyglot/suika_solver.pg` to `ClientApp/src/app/suika/suika_solver`.
- Prose that read *"Suika (Suika-style …)"* after the substitution was reworded.
- The historical PRDs are renamed too, as asked ("all mentions"). The pre-rename names can still be
  recovered with `git log --follow`.

### Compatibility shims (the only places the old name survives)

| What | Why |
|---|---|
| Route `/fruitcake` → redirects to `/suika` | Keeps existing links and bookmarks working. |
| `fruitcake:*` localStorage keys are migrated once to `suika:*` | Keeps players' high score, theme, music preference and saved game. The migration never overwrites a newer `suika:*` value. |
| Lab `--game fruitcake` is accepted as an alias of `--game suika` | Keeps existing training scripts working. |

Local, gitignored training output dirs `data/fruitcake-*` were renamed to `data/suika-*` on the dev
machine, to match the code paths.

## 3. The buzz

### Root cause

`SuikaGame`'s constructor called `audio.setMusic(this.musicOn)`. Music is off by default, but the
choice is persisted (`suika:music`), so once a player has opted in it is on for every later visit.

1. When the game is constructed (on page load, before any user gesture), `setMusic(true)` creates the
   `AudioContext`. Because there was no gesture yet, the browser autoplay policy keeps the context
   **suspended**. It then starts two endless sine oscillators (C3 130.8 Hz and G3 196 Hz, low-passed)
   plus an LFO, none of which has a stop time.
2. Dropping a fruit makes no audio call.
3. The first landing calls `playLand` → `ac()` → `ctx.resume()`. That un-suspends the context, so the
   drone becomes audible **at the moment the first fruit lands**. On laptop speakers a pure low tone
   like this sounds like a buzz.
4. The 🔊 button only set `muted`, which gated the sound effects but not the music. Only 🎵 could stop
   the drone, so it seemed to "never stop".

Ruled out: the landing thud and merge pop. Each of their oscillators ramps down and calls
`stop(t + 0.18)`, and `onLanded` fires at most once per fruit (speed threshold plus a `hasLanded`
latch).

### Fix

- The constructor no longer touches music.
- New `SuikaGame.unlockAudio()`, which the component calls on every human pointer-down. The first call
  starts music if the player opted in. A pointer-down is a real gesture, so the context starts running
  instead of being parked suspended.
- A single `syncMusic()` sets `setMusic(unlocked && musicOn && !muted)`. 🔊 mute now silences music
  too, and 🎵 and 🔊 both count as gestures.
- Music fades in over 1.5 s (no click or sudden onset).

## 4. Regression test

`ClientApp/src/app/suika/suika-audio.spec.ts` (vitest, jsdom):

- With `suika:music=1`: constructing the game and letting a dropped fruit land (no gesture) never
  turns music on. This test fails on the old code.
- `unlockAudio()` starts opted-in music. `toggleMute()` stops it and starts it again.
- Music never starts when the player did not opt in.
- Legacy `fruitcake:*` keys are migrated and removed.
- With a fake `AudioContext`: every landing and merge oscillator has a stop scheduled, and
  `setMusic(false)` stops every music oscillator.

## 5. Out of scope

- Changing the music itself (timbre or loudness).
- Memory notes outside the repo. They are updated separately, not shipped in this PR.
