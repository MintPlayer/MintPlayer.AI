import { AfterViewInit, Component, DestroyRef, ElementRef, NgZone, inject, signal, viewChild } from '@angular/core';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { TetrisDirector, Tier } from './tetris-director';
import { TetrisGame } from './tetris-game';
import { techniqueHz, type Technique } from './tetris-das';
import { LOGICAL_H, LOGICAL_W, cellWidthCss, render } from './tetris-render';
import { ScreenWakeLock } from '../screen-wake-lock';

/**
 * Tetris (PLAN M54) — afterstate AI over the single-source engine (tetris_solver.pg): the same rules code
 * the AI trains on runs the human game and every watch tier, entirely in the browser (Pattern C — no
 * server inference). The rising-garbage mode (a gapped bottom row every 10 placements — TETRIS_PRD.md §1)
 * is both a playable challenge and the AI's primary evaluation protocol.
 *
 * Input: keyboard (←/→ move, ↑/X rotate CW, Z rotate CCW, ↓ soft drop, Space hard drop) + unified Pointer Events for touch
 * (horizontal drag moves cell-by-cell, tap rotates, downward swipe hard-drops).
 */
@Component({
  selector: 'app-tetris',
  templateUrl: './tetris.html',
  styleUrl: './tetris.scss',
  imports: [BsButtonTypeDirective],
  host: {
    '(window:keydown)': 'onKeyDown($event)',
    '(window:keyup)': 'onKeyUp($event)',
    '(window:blur)': 'onFocusLost()',
    '(document:visibilitychange)': 'onVisibilityChange()',
    '(document:fullscreenchange)': 'onFullscreenChange()',
  },
})
export class Tetris implements AfterViewInit {
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('board');
  private readonly zone = inject(NgZone);
  private readonly wakeLock = inject(ScreenWakeLock);

  protected readonly colors = Color;
  protected readonly game = new TetrisGame();

  /** 'human' = play locally; 'watch' = a selectable tier plays — everything runs in the browser. */
  protected readonly mode = signal<'human' | 'watch'>('human');
  // M62.4a (owner decision, supersedes D12's della-search): the hand-tuned Dellacherie evaluator is the
  // default player. It builds tetrises — the trained net measurably does not (18.1% vs 2.7% of lines
  // cleared as tetrises) — and it decides in ~0.06 ms, so the board never stalls. Every tier stays one
  // click away.
  protected readonly tier = signal<Tier>('dellacherie');
  /** Rising-garbage mode: a full bottom row with one random gap every 10 placements. */
  protected readonly garbage = signal(false);

  // M62.2 — NES authenticity controls. The engine has always supported a start level (setStartLevel) but
  // nothing ever called it, so every game began at 0. The CTWC-relevant starts are 0/9/15/18/19/29.
  protected readonly startLevel = signal(0);
  protected readonly startLevels = [0, 9, 15, 18, 19, 29];
  // Post-level-29 variant. AUTHENTIC NES has no speed change above 29 (flat 1 frame/row to 255); the
  // faster-than-29 behaviour seen in CTWC Masters is a ROM hack, so it is opt-in and never the default.
  protected readonly killscreen = signal<0 | 1 | 2>(0);

  // M62.3 — the input technique dial (owner ask 4). Governs the AI's tap budget, which the engine's
  // evaluator already consults, so this is a strength control and not just an animation speed.
  // M62.4a: rolling by default — the modern technique, and the one that keeps the AI able to feed a well
  // once gravity gets fast. Switch to DAS at a high start level to watch it stop being able to.
  protected readonly technique = signal<Technique>('roll');
  protected readonly techniqueHz = techniqueHz;
  /** Esc pause: freezes the game AND hides the field (the render covers the canvas). */
  protected readonly paused = signal(false);

  // View controls (owner request: pro players may find the view too large). Zoom scales the stage (and,
  // in fullscreen, the canvas within the viewport) via the --zoom CSS variable; the buttons live INSIDE
  // the stage element so they carry into the fullscreen layout. Zoom is a per-browser convenience.
  protected readonly zoom = signal(this.loadZoom());
  protected readonly isFullscreen = signal(false);
  private static readonly ZOOM_MIN = 0.5;
  private static readonly ZOOM_MAX = 1.5;

  private readonly stageRef = viewChild.required<ElementRef<HTMLElement>>('stage');

  private loadZoom(): number {
    try {
      const v = parseFloat(localStorage.getItem('tetris.zoom') ?? '');
      return Number.isFinite(v) ? Math.min(Tetris.ZOOM_MAX, Math.max(Tetris.ZOOM_MIN, v)) : 1;
    } catch {
      return 1;
    }
  }

  protected zoomBy(delta: number): void {
    this.zoom.update(z => Math.min(Tetris.ZOOM_MAX, Math.max(Tetris.ZOOM_MIN, Math.round((z + delta) * 8) / 8)));
    try { localStorage.setItem('tetris.zoom', String(this.zoom())); } catch { /* private mode etc. */ }
  }

  protected toggleFullscreen(): void {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void this.stageRef().nativeElement.requestFullscreen?.();
  }

  protected onFullscreenChange(): void {
    this.isFullscreen.set(!!document.fullscreenElement);
  }
  private director: TetrisDirector | null = null;

  private ctx: CanvasRenderingContext2D | null = null;
  private rafId = 0;
  private lastMs = 0;

  /** Active touch gesture: start position + how many cells the drag has already moved. */
  private drag: { sx: number; sy: number; movedCells: number; startMs: number; consumed: boolean } | null = null;

  constructor() {
    inject(DestroyRef).onDestroy(() => {
      cancelAnimationFrame(this.rafId);
      void this.wakeLock.release();
    });
  }

  protected setMode(mode: 'human' | 'watch'): void {
    if (this.mode() === mode) return;
    this.mode.set(mode);
    if (mode === 'watch') {
      this.director ??= new TetrisDirector(this.game);
      this.director.tier = this.tier();
      this.director.reset();
      void this.wakeLock.acquire(); // keep the phone screen on while the AI plays
    } else {
      this.game.newGame();
      void this.wakeLock.release();
    }
  }

  protected setTier(tier: Tier): void {
    this.tier.set(tier);
    if (this.director) this.director.tier = tier;
  }

  protected toggleGarbage(): void {
    this.garbage.update(v => !v);
    this.game.garbageEvery = this.garbage() ? 10 : 0;
    this.game.newGame();
  }

  protected newGame(): void {
    this.game.newGame();
  }

  /** Both settings only take effect on a fresh board, so changing either starts a new game. */
  protected setStartLevel(level: number): void {
    this.startLevel.set(level);
    this.game.startLevel = level;
    this.game.newGame();
  }

  /** Takes effect immediately — the tap budget is read per placement, so no restart is needed. */
  protected setTechnique(t: Technique): void {
    this.technique.set(t);
    this.game.technique = t;
    this.game.applyTechnique();
  }

  protected setKillscreen(mode: 0 | 1 | 2): void {
    this.killscreen.set(mode);
    this.game.killscreenMode = mode;
    this.game.newGame();
  }

  ngAfterViewInit(): void {
    this.ctx = this.canvasRef().nativeElement.getContext('2d');
    // The game draws itself to the canvas every frame — run outside Angular so per-frame change detection
    // never happens (all HUD text is drawn on the canvas for the same reason).
    this.zone.runOutsideAngular(() => {
      this.rafId = requestAnimationFrame(this.frame);
    });
  }

  private readonly frame = (nowMs: number): void => {
    const dt = this.lastMs ? Math.min(250, nowMs - this.lastMs) : 0;
    this.lastMs = nowMs;
    if (!this.paused()) {
      if (this.mode() === 'watch') this.director?.update(dt);
      this.game.update(dt, this.mode() === 'human');
    }
    this.draw();
    this.rafId = requestAnimationFrame(this.frame);
  };

  private draw(): void {
    const canvas = this.canvasRef().nativeElement;
    if (!this.ctx) return;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render(this.ctx, this.game, cssW, cssH, this.statusLine(), this.mode(), this.paused());
  }

  private statusLine(): string {
    if (this.mode() === 'human')
      return this.garbage() ? 'rising garbage: a gapped row every 10 pieces' : '←/→ move · ↑/X rotate · Z rotate back · space drop';
    const d = this.director;
    if (!d) return '';
    const tier = (d.tier === 'net' || d.tier === 'net-search') && d.netStatus !== 'ready'
      ? (d.netStatus === 'loading' ? `${d.tier} (loading…)` : 'net missing → dellacherie')
      : d.effectiveTier;
    const last = d.episodes > 0 ? ` · last: ${d.lastLines} lines` : '';
    const t = this.technique();
    const tapName = t === 'das' ? 'DAS' : t === 'hypertap' ? 'hypertapping' : 'rolling';
    const tap = ` · ${tapName} (${this.techniqueHz(t)} Hz)`;
    return `AI: ${tier}${tap}${this.garbage() ? ' · garbage/10' : ''}${last}`;
  }

  // Auto-pause when the window/tab loses focus (owner request): a running play-yourself game must not
  // keep falling unseen. Watch mode is exempt (watching in a second window is legitimate), and there is
  // nothing to protect after game over. Focus regain does NOT auto-resume — Esc or a tap does.
  protected onFocusLost(): void {
    if (this.mode() === 'human' && !this.game.gameOver && !this.paused()) {
      this.paused.set(true);
      this.game.input.clear(); // keyup events are lost once focus is gone — drop all held keys
    }
  }

  protected onVisibilityChange(): void {
    if (document.hidden) this.onFocusLost();
  }

  // ── Keyboard (desktop) ───────────────────────────────────────────────────────────────────────────────────

  protected onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      this.paused.update(v => !v);
      this.game.input.clear(); // never resume into held keys (the DAS charge itself survives)
      event.preventDefault();
      return;
    }
    if (this.paused()) return; // the game is frozen — ignore play keys
    if (this.mode() !== 'human') return;
    if (this.game.gameOver) {
      if (event.key === 'Enter' || event.key === ' ') { this.game.newGame(); event.preventDefault(); }
      return;
    }
    // NES-authentic input (PLAN M55): the OS/browser key auto-repeat is IGNORED entirely
    // (event.repeat filtered) — keydown/keyup are pure press/release edges, and all repeat timing
    // (DAS 16/10/6, soft-drop 3-then-2) belongs to the frame-locked machine in tetris-das.ts.
    if (event.repeat) { event.preventDefault(); return; }
    switch (event.key) {
      case 'ArrowLeft': case 'a': this.game.input.press(-1); break;
      case 'ArrowRight': case 'd': this.game.input.press(1); break;
      // One rotation per press (NES). X = the A button = clockwise, Z = the B button = counter-clockwise.
      case 'ArrowUp': case 'x': case 'w': this.game.rotate(); break;
      case 'z': case 'Control': this.game.rotateCcw(); break;
      case 'ArrowDown': case 's': this.game.input.pressDown(); break;
      case ' ': this.game.hardDrop(); break;
      default: return;
    }
    event.preventDefault();
  }

  protected onKeyUp(event: KeyboardEvent): void {
    switch (event.key) {
      case 'ArrowLeft': case 'a': this.game.input.release(-1); break;
      case 'ArrowRight': case 'd': this.game.input.release(1); break;
      case 'ArrowDown': case 's': this.game.input.releaseDown(); break;
    }
  }

  // ── Pointer input: ONE path for mouse + touch + pen ─────────────────────────────────────────────────────

  protected onPointerDown(event: PointerEvent): void {
    event.preventDefault();
    if (this.paused()) {
      this.paused.set(false); // tap anywhere to resume
      return;
    }
    if (this.mode() !== 'human') return;
    if (this.game.gameOver) {
      this.game.newGame(); // tap to play again
      return;
    }
    const { sx, sy } = this.toSurface(event);
    this.game.input.clear(); // pointer takes over — a stale keyboard hold must not keep auto-shifting
    this.canvasRef().nativeElement.setPointerCapture(event.pointerId);
    this.drag = { sx, sy, movedCells: 0, startMs: performance.now(), consumed: false };
  }

  protected onPointerMove(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || this.game.gameOver) return;
    const { sx, sy, w } = this.toSurface(event);
    const cellCss = cellWidthCss(w);

    // Horizontal drag: one shift per cell width crossed (keeps the piece under the finger).
    const cells = Math.trunc((sx - drag.sx) / cellCss);
    while (drag.movedCells < cells) { this.game.moveRight(); drag.movedCells++; drag.consumed = true; }
    while (drag.movedCells > cells) { this.game.moveLeft(); drag.movedCells--; drag.consumed = true; }

    // Downward swipe (fast + far, mostly vertical): hard drop.
    const dy = sy - drag.sy;
    if (!drag.consumed && dy > 2.2 * cellCss && Math.abs(sx - drag.sx) < cellCss
        && performance.now() - drag.startMs < 320) {
      drag.consumed = true;
      this.drag = null;
      this.game.hardDrop();
    }
  }

  protected onPointerUp(event: PointerEvent): void {
    const drag = this.drag;
    this.drag = null;
    if (!drag || drag.consumed || this.game.gameOver) return;
    // A tap (no drag, no swipe) rotates.
    const { sx, sy, w } = this.toSurface(event);
    if (Math.abs(sx - drag.sx) < 0.4 * cellWidthCss(w) && Math.abs(sy - drag.sy) < 0.4 * cellWidthCss(w))
      this.game.rotate();
  }

  private toSurface(event: PointerEvent): { sx: number; sy: number; w: number; h: number } {
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    return { sx: event.clientX - rect.left, sy: event.clientY - rect.top, w: rect.width, h: rect.height };
  }

  protected readonly stageAspect = `${LOGICAL_W} / ${LOGICAL_H}`;
}
