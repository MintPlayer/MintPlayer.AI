/**
 * Asset-free synthesized sound via Web Audio: merge pop, landing thud, and an ambient music loop.
 * A faithful port of the original game's `suikaAudio`. The context is created lazily on first
 * use (and resumed) so it satisfies the browser's user-gesture autoplay policy — the first sound
 * always follows a tap/drop. Every call is best-effort: failures are swallowed so audio never
 * breaks gameplay.
 *
 * The context is SUSPENDED whenever nothing is audible (shortly after the last one-shot ends, with no
 * music playing) and resumed by the next sound. A running context keeps the OS audio output stream
 * open, playing silence, and on some hardware (USB/HDMI audio, Bluetooth headsets, some laptop codecs)
 * that open stream is itself an audible, never-ending hum or buzz — one that mute cannot stop, because
 * mute only gates new sounds. That was the "buzz from the first landing onward" bug (PLAN M72).
 *
 * Music (PLAN M73) is a short melodic loop of soft triangle notes in C5–C6, each a self-stopping
 * one-shot scheduled a little ahead of time. It used to be two endless low sines (C3 + G3, 131/196 Hz)
 * under a slow swell. Phone speakers cannot reproduce tones that low, so on Android the "music" was only
 * ever heard as a never-ending buzz, which was the bug report all along. Keep every music tone at or
 * above {@link SuikaAudio.MinMusicHz}.
 */
export class SuikaAudio {
  /** How long after the last sound ends before the context is suspended (output stream released). */
  static readonly IdleSuspendMs = 400;
  /** Lowest music frequency. Phone speakers roll off steeply below ~300–400 Hz and turn lower tones into buzz. */
  static readonly MinMusicHz = 500;
  /** Longest one-shot (landing thud / merge pop), in seconds. */
  private static readonly OneShotSeconds = 0.18;

  // Music loop: C major pentatonic, one entry per eighth note (null = rest). C5 D5 E5 G5 A5 C6.
  private static readonly C5 = 523.25;
  private static readonly D5 = 587.33;
  private static readonly E5 = 659.25;
  private static readonly G5 = 783.99;
  private static readonly A5 = 880.0;
  private static readonly C6 = 1046.5;
  static readonly MusicPattern: readonly (number | null)[] = [
    SuikaAudio.C5, SuikaAudio.E5, SuikaAudio.G5, null, SuikaAudio.E5, SuikaAudio.G5, SuikaAudio.A5, null,
    SuikaAudio.G5, SuikaAudio.E5, SuikaAudio.D5, null, SuikaAudio.E5, SuikaAudio.C5, null, null,
    SuikaAudio.C5, SuikaAudio.E5, SuikaAudio.G5, null, SuikaAudio.A5, SuikaAudio.C6, SuikaAudio.A5, null,
    SuikaAudio.G5, SuikaAudio.E5, SuikaAudio.D5, SuikaAudio.E5, SuikaAudio.C5, null, null, null,
  ];
  private static readonly EighthSeconds = 0.3; // 100 BPM
  private static readonly NoteSeconds = 0.42; // decay tail; slightly longer than a step, so notes overlap softly
  private static readonly MusicVolume = 0.045;
  private static readonly LookaheadSeconds = 0.25;
  private static readonly SchedulerMs = 60;

  private ctx: AudioContext | null = null;
  private music: { gain: GainNode; timer: ReturnType<typeof setInterval>; notes: Set<OscillatorNode>; nextTime: number; step: number } | null = null;
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private suspendRequested = false;

  private ac(): AudioContext {
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx ??= new Ctor();
    this.cancelIdleSuspend();
    // Also resume when a suspend was requested but the state hasn't flipped yet: the calls queue in order.
    if (this.ctx.state === 'suspended' || this.suspendRequested) {
      this.suspendRequested = false;
      void this.ctx.resume();
    }
    return this.ctx;
  }

  /** Suspend the context `delayMs` from now unless another sound (or music) arrives first. */
  private suspendWhenIdle(delayMs: number): void {
    this.cancelIdleSuspend();
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      if (this.music || !this.ctx || this.ctx.state !== 'running') return;
      this.suspendRequested = true;
      void this.ctx.suspend();
    }, delayMs);
  }

  private cancelIdleSuspend(): void {
    if (this.idleTimer !== null) clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  /** Merge pop. `tier` is the resulting fruit's tier — higher pitch for larger fruit. */
  playMerge(tier: number): void {
    this.pop(1 + tier * 0.08);
  }

  /** Landing thud when a falling fruit first hits hard. `impact` is 0..1 (for volume). */
  playLand(impact: number): void {
    try {
      const c = this.ac();
      const o = c.createOscillator();
      const g = c.createGain();
      o.type = 'sine';
      o.frequency.value = 110;
      const t = c.currentTime;
      g.gain.setValueAtTime(Math.max(0.05, impact * 0.45), t);
      g.gain.exponentialRampToValueAtTime(0.0008, t + 0.16);
      o.connect(g);
      g.connect(c.destination);
      o.start(t);
      o.stop(t + SuikaAudio.OneShotSeconds);
      this.suspendWhenIdle(SuikaAudio.OneShotSeconds * 1000 + SuikaAudio.IdleSuspendMs);
    } catch {
      /* best-effort */
    }
  }

  /** Start or stop the looping background music (a soft pentatonic melody, phone-speaker friendly). */
  setMusic(on: boolean): void {
    if (!on && !this.music) return; // nothing playing — don't spin up an audio context just to "stop"
    try {
      const c = this.ac();
      if (on) {
        if (this.music) return;
        const gain = c.createGain();
        gain.gain.setValueAtTime(0, c.currentTime);
        gain.gain.linearRampToValueAtTime(SuikaAudio.MusicVolume, c.currentTime + 1.0); // fade in, no click
        gain.connect(c.destination);
        const music = { gain, timer: setInterval(() => this.scheduleMusic(), SuikaAudio.SchedulerMs), notes: new Set<OscillatorNode>(), nextTime: c.currentTime + 0.05, step: 0 };
        this.music = music;
        this.scheduleMusic(); // first notes now, not one scheduler tick later
      } else if (this.music) {
        clearInterval(this.music.timer);
        for (const o of this.music.notes) {
          try {
            o.stop(); // notes already queued in the lookahead window
          } catch {
            /* already ended */
          }
        }
        this.music.gain.disconnect();
        this.music = null;
        this.suspendWhenIdle(SuikaAudio.IdleSuspendMs); // music was the last audible thing: release the stream
      }
    } catch {
      /* best-effort */
    }
  }

  /** Queue every music note that starts within the lookahead window (a standard Web Audio scheduler). */
  private scheduleMusic(): void {
    const m = this.music;
    const c = this.ctx;
    if (!m || !c) return;
    try {
      while (m.nextTime < c.currentTime + SuikaAudio.LookaheadSeconds) {
        const freq = SuikaAudio.MusicPattern[m.step % SuikaAudio.MusicPattern.length];
        if (freq !== null) this.musicNote(c, m, freq, m.nextTime);
        m.nextTime += SuikaAudio.EighthSeconds;
        m.step++;
      }
    } catch {
      /* best-effort */
    }
  }

  private musicNote(c: AudioContext, m: NonNullable<SuikaAudio['music']>, freq: number, t: number): void {
    const o = c.createOscillator();
    const g = c.createGain();
    o.type = 'triangle'; // soft, few harmonics, carries on small speakers without sounding harsh
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(1, t + 0.015); // quick, click-free attack
    g.gain.exponentialRampToValueAtTime(0.0001, t + SuikaAudio.NoteSeconds); // pluck-like decay
    o.connect(g);
    g.connect(m.gain);
    o.onended = () => m.notes.delete(o);
    m.notes.add(o);
    o.start(t);
    o.stop(t + SuikaAudio.NoteSeconds + 0.02);
  }

  /** Stop music and release the audio context (call on teardown). */
  dispose(): void {
    this.setMusic(false);
    this.cancelIdleSuspend();
    void this.ctx?.close();
    this.ctx = null;
  }

  private pop(rate: number): void {
    try {
      const c = this.ac();
      const o = c.createOscillator();
      const g = c.createGain();
      o.type = 'sine';
      o.frequency.value = 220 * rate;
      const t = c.currentTime;
      g.gain.setValueAtTime(0.18, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.18);
      o.connect(g);
      g.connect(c.destination);
      o.start(t);
      o.stop(t + SuikaAudio.OneShotSeconds);
      this.suspendWhenIdle(SuikaAudio.OneShotSeconds * 1000 + SuikaAudio.IdleSuspendMs);
    } catch {
      /* best-effort */
    }
  }
}
