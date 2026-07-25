import { AfterViewInit, Component, DestroyRef, ElementRef, NgZone, inject, signal, viewChild } from '@angular/core';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { CrazyFruitsDirector, MOVES_PER_EPISODE, Tier } from './crazy-fruits-director';
import { CrazyFruitsGame, SIZE } from './crazy-fruits-game';
import { dragThreshold, pointToCell, render } from './crazy-fruits-render';
import { ScreenWakeLock } from '../screen-wake-lock';

/**
 * Crazy Fruits — a match-3 in the spirit of the KidCity (kidcity.be) Flash original (PLAN M49). Swap two
 * adjacent fruits to line up 3+ of the same kind; matches clear, fruit falls, cascades chain. Runs entirely
 * in the browser — the rules engine is the SAME single-source code (crazyfruits_solver.pg) the AI trains on.
 *
 * Input (PRD §3.10) is unified Pointer Events — one code path for touch (touchstart/move/end) and mouse
 * (mousedown/move/up): drag a fruit toward a neighbour to swap, or tap-select then tap an adjacent fruit.
 */
@Component({
  selector: 'app-crazy-fruits',
  templateUrl: './crazy-fruits.html',
  styleUrl: './crazy-fruits.scss',
  imports: [BsButtonTypeDirective],
})
export class CrazyFruits implements AfterViewInit {
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('board');
  private readonly zone = inject(NgZone);

  protected readonly colors = Color;
  protected readonly game = new CrazyFruitsGame();
  private readonly wakeLock = inject(ScreenWakeLock);

  /** 'human' = play locally; 'watch' = a selectable tier plays — everything runs in the browser. */
  protected readonly mode = signal<'human' | 'watch'>('human');
  protected readonly tier = signal<Tier>('net');
  /** Endless mode: keep playing past 30 moves; the score stops counting toward "best". */
  protected readonly freePlay = signal(false);
  // Created lazily on first watch; the trained net loads itself (falls back to expectimax while absent).
  private director: CrazyFruitsDirector | null = null;

  private ctx: CanvasRenderingContext2D | null = null;
  private rafId = 0;
  private lastMs = 0;

  /** Active drag gesture: the cell it started on + the pointer-down position (canvas CSS px). */
  private drag: { cell: number; sx: number; sy: number; dragged: boolean } | null = null;

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
      this.director ??= new CrazyFruitsDirector(this.game);
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

  protected toggleFreePlay(): void {
    this.freePlay.update(v => !v);
    this.game.setFreePlay(this.freePlay());
  }

  ngAfterViewInit(): void {
    this.ctx = this.canvasRef().nativeElement.getContext('2d');
    // The game draws itself to the canvas every frame — run outside Angular so per-frame change detection
    // never happens (all HUD text is drawn on the canvas for the same reason).
    this.zone.runOutsideAngular(() => {
      this.rafId = requestAnimationFrame(this.frame);
    });
  }

  /** Measured 500-episode tier means (RANKING PRD M51.2 `cf9train`) — the round-over screen's challenge lines. */
  private static readonly ROUND_BARS = [
    'random plays ~2 600 a round',
    'the trained net ~5 650',
    'expectimax-2 plays ~8 000',
  ];

  private readonly frame = (nowMs: number): void => {
    const dt = this.lastMs ? Math.min(250, nowMs - this.lastMs) : 0;
    this.lastMs = nowMs;
    if (this.mode() === 'watch') this.director?.update(dt);
    else this.game.checkRoundOver(30);
    this.game.update(dt);
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
    render(this.ctx, this.game, cssW, cssH, this.statusLine(), CrazyFruits.ROUND_BARS);
  }

  private statusLine(): string {
    if (this.mode() === 'human')
      return this.freePlay() ? 'endless — score won’t count for best' : 'drag or tap-tap to swap';
    const d = this.director;
    if (!d) return '';
    const tier = d.tier === 'net' && d.netStatus !== 'ready'
      ? (d.netStatus === 'loading' ? 'net (loading…)' : 'net missing → expectimax')
      : d.effectiveTier;
    const episode = d.lastScore > 0 ? ` · last ${d.lastScore}` : '';
    return `AI: ${tier} · move ${this.game.board.movesMade}/${MOVES_PER_EPISODE}${episode}`;
  }

  protected newGame(): void {
    this.game.newGame();
  }

  // ── Pointer input: ONE path for mouse + touch + pen (PRD §3.10) ──────────────────────────────────────────

  protected onPointerDown(event: PointerEvent): void {
    event.preventDefault();
    if (this.mode() !== 'human') return;
    if (this.game.roundOver) {
      this.game.newGame();          // tap to play again
      return;
    }
    const { sx, sy, w } = this.toSurface(event);
    const cell = pointToCell(sx, sy, w);
    if (cell < 0 || this.game.animating) {
      this.game.selected = -1;
      return;
    }
    // The gesture survives leaving the canvas mid-drag.
    this.canvasRef().nativeElement.setPointerCapture(event.pointerId);
    this.drag = { cell, sx, sy, dragged: false };
  }

  protected onPointerMove(event: PointerEvent): void {
    if (!this.drag || this.drag.dragged || this.game.animating) return;
    const { sx, sy, w } = this.toSurface(event);
    const dx = sx - this.drag.sx;
    const dy = sy - this.drag.sy;
    const threshold = dragThreshold(w);
    if (Math.abs(dx) < threshold && Math.abs(dy) < threshold) return;

    // Drag-swap: past half a cell toward a neighbour ⇒ attempt that swap.
    const cell = this.drag.cell;
    const r = Math.floor(cell / SIZE);
    const c = cell % SIZE;
    let target = -1;
    if (Math.abs(dx) > Math.abs(dy)) target = dx > 0 ? (c < SIZE - 1 ? cell + 1 : -1) : (c > 0 ? cell - 1 : -1);
    else target = dy > 0 ? (r < SIZE - 1 ? cell + SIZE : -1) : (r > 0 ? cell - SIZE : -1);

    this.drag.dragged = true;
    if (target >= 0) {
      this.game.selected = -1;
      this.game.trySwap(cell, target);
    }
  }

  protected onPointerUp(event: PointerEvent): void {
    const drag = this.drag;
    this.drag = null;
    if (!drag || drag.dragged || this.game.animating) return;

    // Tap-tap: first tap selects, a tap on an orthogonal neighbour swaps, any other tap moves the selection.
    const { sx, sy, w } = this.toSurface(event);
    const cell = pointToCell(sx, sy, w);
    if (cell < 0) {
      this.game.selected = -1;
      return;
    }
    const selected = this.game.selected;
    if (selected >= 0 && CrazyFruitsGame.actionFor(selected, cell) >= 0) {
      this.game.trySwap(selected, cell);
    } else {
      this.game.selected = cell === selected ? -1 : cell;
    }
  }

  private toSurface(event: PointerEvent): { sx: number; sy: number; w: number; h: number } {
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    return { sx: event.clientX - rect.left, sy: event.clientY - rect.top, w: rect.width, h: rect.height };
  }
}
