import { AfterViewInit, Component, DestroyRef, ElementRef, NgZone, inject, signal, viewChild } from '@angular/core';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { TetrisDirector, Tier } from './tetris-director';
import { TetrisGame } from './tetris-game';
import { LOGICAL_H, LOGICAL_W, cellWidthCss, render } from './tetris-render';
import { ScreenWakeLock } from '../screen-wake-lock';

/**
 * Tetris (PLAN M54) — afterstate AI over the single-source engine (tetris_solver.pg): the same rules code
 * the AI trains on runs the human game and every watch tier, entirely in the browser (Pattern C — no
 * server inference). The rising-garbage mode (a gapped bottom row every 10 placements — TETRIS_PRD.md §1)
 * is both a playable challenge and the AI's primary evaluation protocol.
 *
 * Input: keyboard (←/→ move, ↑/X rotate, ↓ soft drop, Space hard drop) + unified Pointer Events for touch
 * (horizontal drag moves cell-by-cell, tap rotates, downward swipe hard-drops).
 */
@Component({
  selector: 'app-tetris',
  templateUrl: './tetris.html',
  styleUrl: './tetris.scss',
  imports: [BsButtonTypeDirective],
  host: { '(window:keydown)': 'onKeyDown($event)', '(window:keyup)': 'onKeyUp($event)' },
})
export class Tetris implements AfterViewInit {
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('board');
  private readonly zone = inject(NgZone);
  private readonly wakeLock = inject(ScreenWakeLock);

  protected readonly colors = Color;
  protected readonly game = new TetrisGame();

  /** 'human' = play locally; 'watch' = a selectable tier plays — everything runs in the browser. */
  protected readonly mode = signal<'human' | 'watch'>('human');
  protected readonly tier = signal<Tier>('net');
  /** Rising-garbage mode: a full bottom row with one random gap every 10 placements. */
  protected readonly garbage = signal(false);
  /** Esc pause: freezes the game AND hides the field (the render covers the canvas). */
  protected readonly paused = signal(false);
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
      return this.garbage() ? 'rising garbage: a gapped row every 10 pieces' : '←/→ move · ↑ rotate · space drop';
    const d = this.director;
    if (!d) return '';
    const tier = (d.tier === 'net' || d.tier === 'net-search') && d.netStatus !== 'ready'
      ? (d.netStatus === 'loading' ? `${d.tier} (loading…)` : 'net missing → dellacherie')
      : d.effectiveTier;
    const last = d.episodes > 0 ? ` · last: ${d.lastLines} lines` : '';
    return `AI: ${tier}${this.garbage() ? ' · garbage/10' : ''}${last}`;
  }

  // ── Keyboard (desktop) ───────────────────────────────────────────────────────────────────────────────────

  protected onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      this.paused.update(v => !v);
      this.game.setSoftDrop(false); // never resume into a held drop
      event.preventDefault();
      return;
    }
    if (this.paused()) return; // the game is frozen — ignore play keys
    if (this.mode() !== 'human') return;
    if (this.game.gameOver) {
      if (event.key === 'Enter' || event.key === ' ') { this.game.newGame(); event.preventDefault(); }
      return;
    }
    switch (event.key) {
      case 'ArrowLeft': case 'a': this.game.moveLeft(); break;
      case 'ArrowRight': case 'd': this.game.moveRight(); break;
      case 'ArrowUp': case 'x': case 'w': this.game.rotate(); break;
      // Ignore key auto-repeat: after a lock cancels the soft drop, only a FRESH press re-arms it.
      case 'ArrowDown': case 's': if (!event.repeat) this.game.setSoftDrop(true); break;
      case ' ': this.game.hardDrop(); break;
      default: return;
    }
    event.preventDefault();
  }

  protected onKeyUp(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown' || event.key === 's') this.game.setSoftDrop(false);
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
