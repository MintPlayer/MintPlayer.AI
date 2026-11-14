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

## 6. M72: the buzz survived M71 (2026-11-14, branch `suika-audio-suspend`)

After M71 was deployed (`7b7ae15`), the owner reported the buzz still starts when a fruit lands, and
**🔊 mute does not help**. That second fact refutes §3 as the whole story. In M71, mute silences both the
music and the sound effects, so a buzz that survives mute is not any sound the game plays.

### What the code rules out

- `suika-audio.ts` is the only audio source in the client. There is no `<audio>`, `new Audio` or
  `vibrate` anywhere.
- Every one-shot oscillator (landing thud, merge pop) schedules `stop(t + 0.18)`.
- `onLanded` fires at most once per fruit (`hasLanded` latch, impact-speed threshold).
- Mute stops the music (M71).

### Root cause

Nothing ever suspended or closed the `AudioContext` while the game was open.

- After M71 the context is created on the first sound, which is the first landing.
- A running context keeps the OS audio output stream open, playing silence, until the page is left.
- On some hardware (USB/HDMI audio, Bluetooth headsets, some laptop codecs) that open stream is itself
  audible as a constant hum or buzz.
- Mute cannot reach it, because mute only gates new sounds.

This matches all three symptoms (starts at the first landing, never stops, immune to mute). It was not
reproduced: hardware hum can't be observed from a headless browser. It is the only mechanism left that
fits them.

M71's music fix stays correct: it is a separate path to an audible drone.

### Fix

`SuikaAudio` suspends the context when idle, which releases the output stream:

- Each one-shot schedules a suspend `0.18 s + IdleSuspendMs` (400 ms) later.
- Stopping music schedules a suspend `IdleSuspendMs` later.
- The next sound cancels a pending suspend and resumes the context. It also resumes when a suspend was
  requested but hasn't taken effect yet.
- Nothing suspends while music plays. `dispose()` clears the timer.

### Regression tests (`suika-audio.spec.ts`, fake timers)

- The context suspends after a thud, but not a millisecond before the sound could have ended.
- A rapid cascade keeps pushing the suspend back.
- The context resumes for the next sound.
- Music blocks suspending, and stopping music allows it.
- A sound during an in-flight suspend re-resumes the context.
- No timer is left behind after `dispose()`.

### Owner verification needed

Play, let a fruit land, and wait about half a second. The hum should stop by itself. If it does not, the
cause is outside the page (see the browser's audio indicator on the tab), and we look there next.

**Outcome:** it did not stop. §7 has the real cause. The idle suspend is still worth keeping: an idle page
now releases the audio output.

## 7. M73: the buzz IS the music (2026-11-14, branch `suika-phone-music`)

### How it was finally pinned down

M71 and M72 were each a plausible theory shipped without a reproduction. Both were wrong about the buzz
itself. Four checks by the owner on the Android phone settled it:

1. With media volume at 0 the buzz stops, and it comes back when volume is raised. So it is browser audio
   output, not vibration.
2. It is only a sound; the phone does not vibrate.
3. Since M71, 🔊 mute stops it and unmuting brings it back. That rules out the one-shots: each fruit lands
   once and the thud lasts 0.18 s, so unmuting cannot restart a buzz. Unmuting can only restart the music.
4. Tapping 🎵 stops it. Confirmed: the buzz is the background music.

The owner had **never heard any music** in this game. That is the core of the bug. The music was two
endless sine tones, C3 and G3 (130.8 Hz and 196 Hz), low-passed at 420 Hz with a 0.05 Hz swell. Phone
speakers roll off steeply below roughly 300–400 Hz, so all that came out was a faint distorted hum: a buzz
that never stopped, because the tones never stopped. The setting was on because the `music` preference is
persisted (and was migrated from `fruitcake:music` in M71). The buzz started at the first landing because
the music fades in after the first tap (since M71), and before M71 it was woken by the first landing's
`resume()`. Why only Android: desktop speakers reproduce 131 Hz as a soft drone.

### Fix: phone-friendly music (owner's choice)

- The music is now a 32-step C-major-pentatonic melody at 100 BPM, in C5–C6 (523–1047 Hz). It uses
  triangle-wave notes with a 15 ms attack and a pluck-like decay of about 0.42 s.
- Every note is a self-stopping one-shot. A standard lookahead scheduler (60 ms tick, 250 ms window)
  queues them. There are no endless oscillators any more.
- Turning music off clears the scheduler and stops the notes already queued. The master gain fades in
  over 1 s.
- `SuikaAudio.MinMusicHz = 500` is the speaker-safe floor, and a test enforces it.

### Regression tests (`suika-audio.spec.ts`)

- No music tone, scheduled or in the pattern, is below `MinMusicHz`. This would have failed on the old
  131 Hz drone.
- Every music note is a triangle one-shot with its stop scheduled at creation.
- The melody keeps scheduling over time, and no new notes appear after music is turned off.

### Lesson

Ask for the cheapest discriminating observation before shipping a theory. "Does 🎵 stop it?" was one tap
away from the start.
