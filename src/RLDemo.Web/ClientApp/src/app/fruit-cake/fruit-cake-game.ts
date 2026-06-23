// Behaviour mirror: the C# training env is src/MintPlayer.AI.ReinforcementLearning.Environments/FruitCake/FruitCakeEnv.cs
// (training subset of these rules) — keep in sync (PRD docs/prd/FRUITCAKE_AI_PRD.md §4.8).
import { FruitCakeAudio } from './fruit-cake-audio';
import { Effects } from './fruit-cake-effects';
import { byTier, DROPPABLE, FruitTheme, THEMES } from './fruit-cake-fruits';
import { FruitWorld, MergeEvent } from './fruit-cake-physics';

export enum GamePhase {
  Playing,
  GameOver,
}

const SCORE_KEY = 'fruitcake:score';
const THEME_KEY = 'fruitcake:theme';
const MUSIC_KEY = 'fruitcake:music';
const SNAPSHOT_KEY = 'fruitcake:snapshot';

interface Snapshot {
  score: number;
  currentTier: number;
  nextTier: number;
  aimXPx: number;
  cooldownRemaining: number;
  fruits: Array<{ tier: number; x: number; y: number }>;
}

/**
 * The playable game: a drop queue (current + next, tiers 1–5), pointer aiming, a drop cooldown,
 * scoring, danger-line game-over, and restart. The component forwards pointer position to
 * {@link aimTo}, taps to {@link drop}/{@link restart}, time to {@link step}, and renders from the
 * exposed state. All game rules live here. High score, theme, music choice, and an in-progress
 * game are persisted to `localStorage`.
 */
export class FruitCakeGame {
  static readonly DropCooldownSeconds = 0.5;
  static readonly GraceSeconds = 1.5; // rest-above-line grace
  private static readonly RestSpeedPx = 40; // "settled" threshold
  private static readonly WallInset = 6; // keep the held fruit clear of the walls

  world!: FruitWorld;
  readonly effects = new Effects();

  phase = GamePhase.Playing;
  score = 0;
  highScore = 0;
  currentTier = 1;
  nextTier = 1;
  /** Horizontal aim position of the held fruit (px), clamped inside the walls. */
  aimXPx = FruitWorld.ContainerWidthPx / 2;
  /** Seconds until the next drop is allowed (0 = ready). */
  cooldownRemaining = 0;
  /** True while a fruit is settled above the danger line — drives the red pulse. */
  dangerActive = false;

  muted = false;
  colorblindLabels = false;
  musicOn = false;
  themeIndex = 0;
  /** When set, big-merge screen shake and the watermelon flash are suppressed (accessibility). */
  reduceMotion = false;

  private aboveLineSeconds = 0;

  constructor(private readonly audio: FruitCakeAudio) {
    this.highScore = this.getInt(SCORE_KEY, 0);
    this.themeIndex = Math.min(THEMES.length - 1, Math.max(0, this.getInt(THEME_KEY, 0)));
    this.musicOn = this.getInt(MUSIC_KEY, 0) !== 0; // off by default; opt in via the 'mus' button
    this.reduceMotion = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
    this.restart();
    this.tryResume();
    this.audio.setMusic(this.musicOn);
  }

  get theme(): FruitTheme {
    return THEMES[this.themeIndex];
  }

  get canDrop(): boolean {
    return this.phase === GamePhase.Playing && this.cooldownRemaining <= 0;
  }

  /** Y at which the held/next fruit sits, just above the danger line. */
  static heldYPx(tier: number): number {
    return FruitWorld.DangerLineYPx - byTier(tier).radiusPx - 4;
  }

  /** Start a fresh game. */
  restart(): void {
    this.world = new FruitWorld();
    this.world.onMerged = e => this.onMerged(e);
    this.world.onLanded = impact => this.onLanded(impact);
    this.phase = GamePhase.Playing;
    this.score = 0;
    this.cooldownRemaining = 0;
    this.aboveLineSeconds = 0;
    this.dangerActive = false;
    this.currentTier = this.randomDroppable();
    this.nextTier = this.randomDroppable();
    this.aimXPx = FruitWorld.ContainerWidthPx / 2;
  }

  /** Aim the held fruit to a horizontal pixel position (clamped to stay inside the walls). */
  aimTo(xPx: number): void {
    const margin = byTier(this.currentTier).radiusPx + FruitCakeGame.WallInset;
    this.aimXPx = Math.min(FruitWorld.ContainerWidthPx - margin, Math.max(margin, xPx));
  }

  /** Drop the held fruit. No-op (returns false) during cooldown or game-over. */
  drop(): boolean {
    if (!this.canDrop) return false;
    this.world.spawnFruit(this.currentTier, this.aimXPx, FruitCakeGame.heldYPx(this.currentTier));
    this.currentTier = this.nextTier;
    this.nextTier = this.randomDroppable();
    this.cooldownRemaining = FruitCakeGame.DropCooldownSeconds;
    this.aimTo(this.aimXPx); // re-clamp for the new (possibly larger) current fruit
    this.saveSnapshot();
    return true;
  }

  step(dt: number): void {
    this.effects.update(dt); // let popups/particles finish even into game-over
    if (this.phase !== GamePhase.Playing) return;
    if (this.cooldownRemaining > 0) this.cooldownRemaining = Math.max(0, this.cooldownRemaining - dt);
    this.world.step(dt);
    this.checkGameOver(dt);
  }

  toggleMute(): void {
    this.muted = !this.muted;
  }

  toggleColorblindLabels(): void {
    this.colorblindLabels = !this.colorblindLabels;
  }

  toggleMusic(): void {
    this.musicOn = !this.musicOn;
    this.setInt(MUSIC_KEY, this.musicOn ? 1 : 0);
    this.audio.setMusic(this.musicOn);
  }

  cycleTheme(): void {
    this.themeIndex = (this.themeIndex + 1) % THEMES.length;
    this.setInt(THEME_KEY, this.themeIndex);
  }

  private checkGameOver(dt: number): void {
    let restingAbove = false;
    for (const f of this.world.fruits) {
      if (f.yPx < 0) {
        this.endGame();
        return;
      } // ejected over the rim — instant
      if (f.yPx < FruitWorld.DangerLineYPx && f.speedPx < FruitCakeGame.RestSpeedPx) restingAbove = true;
    }

    this.dangerActive = restingAbove;
    this.aboveLineSeconds = restingAbove ? this.aboveLineSeconds + dt : 0;
    if (this.aboveLineSeconds >= FruitCakeGame.GraceSeconds) this.endGame();
  }

  private endGame(): void {
    this.phase = GamePhase.GameOver;
    this.dangerActive = false;
    this.remove(SNAPSHOT_KEY); // a finished game isn't resumable
  }

  private onMerged(e: MergeEvent): void {
    this.score += e.points;
    if (this.score > this.highScore) {
      this.highScore = this.score;
      this.setInt(SCORE_KEY, this.highScore);
    }

    const watermelonPair = e.resultTier === null; // two watermelons → the climax
    this.effects.burst(e.xPx, e.yPx, e.points, byTier(e.sourceTier).color, watermelonPair ? 18 : 0);
    if (!this.muted) this.audio.playMerge(e.resultTier ?? e.sourceTier);

    if (!this.reduceMotion) {
      if (watermelonPair) {
        this.effects.shake(16);
        this.effects.flash(0.7);
      } else if (e.resultTier !== null && e.resultTier >= 9) {
        this.effects.shake(4 + (e.resultTier - 9) * 3); // pineapple and above
      }
    }
  }

  private onLanded(impact: number): void {
    if (!this.muted) this.audio.playLand(impact);
  }

  private randomDroppable(): number {
    return DROPPABLE[Math.floor(Math.random() * DROPPABLE.length)].tier;
  }

  // ── persistence ────────────────────────────────────────────────────────────────────────

  /** Persist the in-progress game so it can be resumed after navigating away or a reload. */
  saveSnapshot(): void {
    if (this.phase !== GamePhase.Playing) return;
    const snap: Snapshot = {
      score: this.score,
      currentTier: this.currentTier,
      nextTier: this.nextTier,
      aimXPx: this.aimXPx,
      cooldownRemaining: this.cooldownRemaining,
      fruits: this.world.fruits.map(f => ({ tier: f.tier, x: f.xPx, y: f.yPx })),
    };
    this.set(SNAPSHOT_KEY, JSON.stringify(snap));
  }

  private tryResume(): void {
    const json = this.get(SNAPSHOT_KEY);
    if (!json) return;
    let snap: Snapshot;
    try {
      snap = JSON.parse(json) as Snapshot;
    } catch {
      this.remove(SNAPSHOT_KEY); // corrupt snapshot — discard
      return;
    }
    this.score = snap.score;
    this.currentTier = snap.currentTier;
    this.nextTier = snap.nextTier;
    this.aimXPx = snap.aimXPx;
    this.cooldownRemaining = snap.cooldownRemaining;
    for (const f of snap.fruits) this.world.spawnFruit(f.tier, f.x, f.y);
  }

  private getInt(key: string, fallback: number): number {
    const raw = this.get(key);
    const n = raw === null ? NaN : Number.parseInt(raw, 10);
    return Number.isFinite(n) ? n : fallback;
  }

  private setInt(key: string, value: number): void {
    this.set(key, String(value));
  }

  private get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  private set(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* storage unavailable (private mode) — non-fatal */
    }
  }

  private remove(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch {
      /* non-fatal */
    }
  }
}
