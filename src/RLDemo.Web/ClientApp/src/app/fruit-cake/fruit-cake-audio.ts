/**
 * Asset-free synthesized sound via Web Audio: merge pop, landing thud, and an ambient music loop.
 * A faithful port of the original game's `fruitcakeAudio`. The context is created lazily on first
 * use (and resumed) so it satisfies the browser's user-gesture autoplay policy — the first sound
 * always follows a tap/drop. Every call is best-effort: failures are swallowed so audio never
 * breaks gameplay.
 */
export class FruitCakeAudio {
  private ctx: AudioContext | null = null;
  private music: { gain: GainNode; oscs: OscillatorNode[]; lfo: OscillatorNode } | null = null;

  private ac(): AudioContext {
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx ??= new Ctor();
    if (this.ctx.state === 'suspended') void this.ctx.resume();
    return this.ctx;
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
      o.stop(t + 0.18);
    } catch {
      /* best-effort */
    }
  }

  /** Start or stop the looping background music (soft C + G with a slow swell). */
  setMusic(on: boolean): void {
    if (!on && !this.music) return; // nothing playing — don't spin up an audio context just to "stop"
    try {
      const c = this.ac();
      if (on) {
        if (this.music) return;
        const gain = c.createGain();
        gain.gain.value = 0.018; // quiet
        const lp = c.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency.value = 420;
        lp.Q.value = 0.6; // mellow, removes harsh harmonics
        lp.connect(gain);
        gain.connect(c.destination);
        const oscs = [130.81, 196.0].map(f => {
          const o = c.createOscillator();
          o.type = 'sine';
          o.frequency.value = f;
          o.connect(lp);
          o.start();
          return o;
        });
        const lfo = c.createOscillator();
        const lg = c.createGain();
        lfo.frequency.value = 0.05;
        lg.gain.value = 0.008;
        lfo.connect(lg);
        lg.connect(gain.gain);
        lfo.start();
        this.music = { gain, oscs, lfo };
      } else if (this.music) {
        this.music.oscs.forEach(o => o.stop());
        this.music.lfo.stop();
        this.music.gain.disconnect();
        this.music = null;
      }
    } catch {
      /* best-effort */
    }
  }

  /** Stop music and release the audio context (call on teardown). */
  dispose(): void {
    this.setMusic(false);
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
      o.stop(t + 0.18);
    } catch {
      /* best-effort */
    }
  }
}
