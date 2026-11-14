/** A floating "+N" that rises and fades from a merge point. */
export interface ScorePopup {
  xPx: number;
  yPx: number;
  age: number;
  points: number;
}
export const POPUP_LIFE_SECONDS = 0.6;

/** A small colored dot flung out of a merge — physics-lite, short-lived. */
export interface Particle {
  xPx: number;
  yPx: number;
  vxPx: number;
  vyPx: number;
  age: number;
  radiusPx: number;
  color: number;
}
export const PARTICLE_LIFE_SECONDS = 0.32;

/**
 * Transient merge feedback: a particle burst + a floating score popup per merge, plus screen
 * shake and a full-screen flash for the big moments. Advanced each step, drawn by the renderer.
 * Restraint by design — one burst, one popup, no trails.
 */
export class Effects {
  private static readonly ParticleGravityPx = 600;
  static readonly FlashLifeSeconds = 0.3;

  readonly popups: ScorePopup[] = [];
  readonly particles: Particle[] = [];
  private shakeMag = 0;
  private flashRemain = 0;
  private flashPeak = 0;

  shakeDx = 0;
  shakeDy = 0;

  get flashAlpha(): number {
    return this.flashPeak * (this.flashRemain / Effects.FlashLifeSeconds);
  }

  /** Kick the screen shake to at least `magnitudePx`. */
  shake(magnitudePx: number): void {
    this.shakeMag = Math.max(this.shakeMag, magnitudePx);
  }

  /** Trigger a full-screen flash at `intensity` (0..1). */
  flash(intensity: number): void {
    this.flashPeak = Math.min(1, Math.max(0, intensity));
    this.flashRemain = Effects.FlashLifeSeconds;
  }

  /** Emit feedback for a merge at the given point, tinted with the fruit's color. */
  burst(xPx: number, yPx: number, points: number, color: number, extraParticles = 0): void {
    this.popups.push({ xPx, yPx, age: 0, points });

    const count = extraParticles > 0 ? extraParticles : 5 + Math.floor(Math.random() * 4);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 140 + Math.random() * 160;
      this.particles.push({
        xPx,
        yPx,
        vxPx: Math.cos(angle) * speed,
        vyPx: Math.sin(angle) * speed - 60, // bias slightly upward
        age: 0,
        radiusPx: 3 + Math.random() * 3,
        color,
      });
    }
  }

  update(dt: number): void {
    if (this.shakeMag > 0.3) {
      this.shakeMag *= Math.exp(-14 * dt); // quick decay, ~0.2s
      this.shakeDx = (Math.random() * 2 - 1) * this.shakeMag;
      this.shakeDy = (Math.random() * 2 - 1) * this.shakeMag;
    } else {
      this.shakeMag = 0;
      this.shakeDx = this.shakeDy = 0;
    }

    if (this.flashRemain > 0) this.flashRemain = Math.max(0, this.flashRemain - dt);

    for (let i = this.popups.length - 1; i >= 0; i--) {
      const p = this.popups[i];
      p.age += dt;
      p.yPx -= 40 * dt; // drift upward
      if (p.age >= POPUP_LIFE_SECONDS) this.popups.splice(i, 1);
    }

    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.age += dt;
      p.vyPx += Effects.ParticleGravityPx * dt;
      p.xPx += p.vxPx * dt;
      p.yPx += p.vyPx * dt;
      if (p.age >= PARTICLE_LIFE_SECONDS) this.particles.splice(i, 1);
    }
  }
}
