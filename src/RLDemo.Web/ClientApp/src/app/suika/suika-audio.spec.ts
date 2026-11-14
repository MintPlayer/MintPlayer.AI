import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SuikaAudio } from './suika-audio';
import { SuikaGame } from './suika-game';

// Regression: "when the first fruit lands, a buzzing sound starts that never stops".
//
// Root cause: with music opted in (`suika:music=1`, persisted), the game constructor called
// `setMusic(true)` before any user gesture. The browser keeps that AudioContext suspended, so the two
// endless sine drones were silent — until the first landing thud called `ctx.resume()` and woke them.
// The 🔊 mute button only gated the sound effects, so nothing but 🎵 could stop the drone.

/** Records every `setMusic` call; the last one is the music state the game asked for. */
class RecordingAudio {
  readonly music: boolean[] = [];
  lands = 0;
  setMusic(on: boolean): void {
    this.music.push(on);
  }
  playLand(): void {
    this.lands++;
  }
  playMerge(): void {}
  dispose(): void {}
  get musicOn(): boolean {
    return this.music.at(-1) ?? false;
  }
}

function newGame(audio: RecordingAudio): SuikaGame {
  return new SuikaGame(audio as unknown as SuikaAudio);
}

describe('SuikaGame music lifecycle', () => {
  beforeEach(() => localStorage.clear());

  it('does not start opted-in music before a user gesture, even when the first fruit lands', () => {
    localStorage.setItem('suika:music', '1');
    const audio = new RecordingAudio();
    const game = newGame(audio);
    expect(audio.musicOn).toBe(false);

    // A programmatic drop (no gesture) all the way to the first landing thud.
    game.drop();
    for (let i = 0; i < 600; i++) game.step(1 / 60);
    expect(audio.lands).toBeGreaterThan(0);
    expect(audio.musicOn).toBe(false);
  });

  it('starts opted-in music on the first gesture, and mute stops it', () => {
    localStorage.setItem('suika:music', '1');
    const audio = new RecordingAudio();
    const game = newGame(audio);

    game.unlockAudio();
    expect(audio.musicOn).toBe(true);

    game.toggleMute();
    expect(audio.musicOn).toBe(false);
    game.toggleMute();
    expect(audio.musicOn).toBe(true);
  });

  it('never starts music when it was not opted in', () => {
    const audio = new RecordingAudio();
    const game = newGame(audio);
    game.unlockAudio();
    game.toggleMute();
    game.toggleMute();
    expect(audio.musicOn).toBe(false);
  });

  it('carries pre-rename fruitcake:* keys over to suika:* once', () => {
    localStorage.setItem('fruitcake:score', '4321');
    localStorage.setItem('fruitcake:music', '1');
    const game = newGame(new RecordingAudio());
    expect(game.highScore).toBe(4321);
    expect(game.musicOn).toBe(true);
    expect(localStorage.getItem('suika:score')).toBe('4321');
    expect(localStorage.getItem('fruitcake:score')).toBeNull();
  });
});

// ── SuikaAudio against a fake Web Audio graph ─────────────────────────────────────────────

interface FakeOsc {
  started: boolean;
  stopAt: number | undefined;
  stopped: boolean;
}

/** `calls` logs every suspend/resume, in order; `state()` reads the (single) context's state. */
interface FakeAudio {
  oscs: FakeOsc[];
  calls: string[];
  state(): string | undefined;
}

function installFakeAudioContext(): FakeAudio {
  const oscs: FakeOsc[] = [];
  const calls: string[] = [];
  let ctx: FakeContext | undefined;
  const param = () => ({ value: 0, setValueAtTime() {}, linearRampToValueAtTime() {}, exponentialRampToValueAtTime() {} });
  class FakeContext {
    state = 'running';
    currentTime = 0;
    destination = {};
    constructor() {
      ctx = this; // eslint-disable-line @typescript-eslint/no-this-alias
    }
    resume() {
      calls.push('resume');
      this.state = 'running';
      return Promise.resolve();
    }
    suspend() {
      calls.push('suspend');
      this.state = 'suspended';
      return Promise.resolve();
    }
    close() {
      return Promise.resolve();
    }
    createGain() {
      return { gain: param(), connect() {}, disconnect() {} };
    }
    createBiquadFilter() {
      return { type: '', frequency: param(), Q: param(), connect() {} };
    }
    createOscillator() {
      const rec: FakeOsc = { started: false, stopAt: undefined, stopped: false };
      oscs.push(rec);
      return {
        type: '',
        frequency: param(),
        connect() {},
        start() {
          rec.started = true;
        },
        stop(t?: number) {
          rec.stopped = true;
          rec.stopAt = t;
        },
      };
    }
  }
  vi.stubGlobal('AudioContext', FakeContext);
  (window as unknown as { AudioContext: unknown }).AudioContext = FakeContext;
  return { oscs, calls, state: () => ctx?.state };
}

describe('SuikaAudio oscillator lifetimes', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('schedules a stop for every landing and merge oscillator', () => {
    const { oscs } = installFakeAudioContext();
    const audio = new SuikaAudio();
    audio.playLand(0.8);
    audio.playMerge(3);
    expect(oscs.length).toBe(2);
    for (const o of oscs) {
      expect(o.started).toBe(true);
      expect(o.stopped).toBe(true);
      expect(o.stopAt).toBeGreaterThan(0);
    }
  });

  it('stops every music oscillator when music is turned off', () => {
    const { oscs } = installFakeAudioContext();
    const audio = new SuikaAudio();
    audio.setMusic(true);
    expect(oscs.length).toBeGreaterThan(0);
    expect(oscs.every(o => o.started && !o.stopped)).toBe(true);
    audio.setMusic(false);
    expect(oscs.every(o => o.stopped)).toBe(true);
  });
});

// Regression (M72): the buzz survived M71 and mute, because it was never a sound we played. A RUNNING
// AudioContext keeps the OS output stream open (playing silence), and on some hardware that open stream
// hums. Nothing ever suspended the context, so from the first landing on, the hum never stopped.
describe('SuikaAudio releases the output stream when idle', () => {
  const afterOneShot = 180 + SuikaAudio.IdleSuspendMs;

  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('suspends the context shortly after a landing thud has finished', () => {
    const fake = installFakeAudioContext();
    const audio = new SuikaAudio();
    audio.playLand(0.8);
    expect(fake.state()).toBe('running');

    vi.advanceTimersByTime(afterOneShot - 1);
    expect(fake.state()).toBe('running'); // never cut a sound short

    vi.advanceTimersByTime(1);
    expect(fake.state()).toBe('suspended');
  });

  it('resumes for the next sound and keeps a rapid burst running until it ends', () => {
    const fake = installFakeAudioContext();
    const audio = new SuikaAudio();
    audio.playLand(0.5);
    vi.advanceTimersByTime(afterOneShot);
    expect(fake.state()).toBe('suspended');

    audio.playMerge(2);
    expect(fake.state()).toBe('running');
    vi.advanceTimersByTime(afterOneShot - 50);
    audio.playMerge(3); // a cascade: each sound pushes the suspend back
    vi.advanceTimersByTime(afterOneShot - 50);
    expect(fake.state()).toBe('running');
    vi.advanceTimersByTime(50);
    expect(fake.state()).toBe('suspended');
  });

  it('never suspends while music plays, and suspends once music stops', () => {
    const fake = installFakeAudioContext();
    const audio = new SuikaAudio();
    audio.setMusic(true);
    audio.playLand(0.5);
    vi.advanceTimersByTime(10_000);
    expect(fake.state()).toBe('running');

    audio.setMusic(false);
    vi.advanceTimersByTime(SuikaAudio.IdleSuspendMs);
    expect(fake.state()).toBe('suspended');
  });

  it('re-resumes when a sound arrives while a suspend is still in flight', () => {
    const fake = installFakeAudioContext();
    const audio = new SuikaAudio();
    audio.playLand(0.5);
    vi.advanceTimersByTime(afterOneShot);
    expect(fake.calls).toEqual(['suspend']);

    audio.playLand(0.5);
    expect(fake.calls).toEqual(['suspend', 'resume']);
  });

  it('does not leave a suspend timer behind after dispose', () => {
    const fake = installFakeAudioContext();
    const audio = new SuikaAudio();
    audio.playLand(0.5);
    audio.dispose();
    vi.advanceTimersByTime(afterOneShot);
    expect(fake.calls).not.toContain('suspend');
  });
});
