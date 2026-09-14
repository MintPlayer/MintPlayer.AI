import { Component, DestroyRef, ElementRef, afterNextRender, inject, signal, viewChild } from '@angular/core';
import { Dir, SIZE, SnakeGame } from './snake-logic';
import { SnakeAiFrame, SnakeDirector } from './snake-director';
import { SnakeTubeRenderer } from './snake-renderer';
import { ScreenWakeLock } from '../screen-wake-lock';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';

const BOARD_PX = 480;
const HUMAN_TICK_MS = 150;

// Visitor settings (M61). The board edge steps by 2 because the Hamiltonian mode needs an even edge — an
// odd×odd grid graph has no Hamiltonian cycle at all — and the net's 177-dim observation is size-independent,
// so the shipped 12×12 checkpoint drives every size on the dial.
export const MIN_SIZE = 6;
export const MAX_SIZE = 50;
export const SIZE_STEP = 2;
/** AI speed in moves per second. 8/s ≈ the 120 ms tick the AI modes shipped with. */
export const MIN_SPEED = 1;
export const MAX_SPEED = 30;
export const DEFAULT_SPEED = 8;

/**
 * Snake page. Both modes now run **entirely in the browser** (M33):
 * - **Watch AI** = the single-source physics + net + masked-greedy policy run client-side (SnakeDirector) —
 *   no server, no WebSocket.
 * - **Play yourself** = a JS timer ticks the local engine.
 */
@Component({
  selector: 'app-snake',
  templateUrl: './snake.html',
  styleUrl: './snake.scss',
  imports: [BsButtonTypeDirective],
  host: { '(window:keydown)': 'onKey($event)' },
})
export class Snake {
  private readonly wakeLock = inject(ScreenWakeLock);

  protected readonly colors = Color;
  protected readonly mode = signal<'idle' | 'watch' | 'watch-cycle' | 'human'>('idle');
  protected readonly foodEaten = signal(0);
  protected readonly status = signal('Watch the self-taught AI play, or play it yourself.');
  /** M60: the cycle overlay is the explanation of the Hamiltonian mode, so it starts visible. Not persisted. */
  protected readonly showRoute = signal(true);
  /** M61: visitor-chosen board edge and AI tick rate. Not persisted — a reload is back to the shipped defaults. */
  protected readonly boardSize = signal(SIZE);
  protected readonly aiSpeed = signal(DEFAULT_SPEED);

  protected readonly minSize = MIN_SIZE;
  protected readonly maxSize = MAX_SIZE;
  protected readonly sizeStep = SIZE_STEP;
  protected readonly minSpeed = MIN_SPEED;
  protected readonly maxSpeed = MAX_SPEED;

  private readonly boardRef = viewChild.required<ElementRef<HTMLCanvasElement>>('board');
  private renderer: SnakeTubeRenderer | null = null;

  private timer: ReturnType<typeof setInterval> | null = null;
  private game: SnakeGame | null = null;
  private director: SnakeDirector | null = null;
  private lastFrame: SnakeAiFrame | null = null;

  constructor() {
    afterNextRender(() => {
      this.renderer = new SnakeTubeRenderer(this.boardRef().nativeElement, this.boardSize(), BOARD_PX);
    });
    inject(DestroyRef).onDestroy(() => { this.stop(); this.renderer?.destroy(); });
  }

  // --- Watch AI: the whole AI (physics + net + search/cycle policy) runs in the browser (M33/M34/M48) ---
  protected watchAi(): void {
    this.startWatch('watch');
  }

  protected watchCycle(): void {
    this.startWatch('watch-cycle');
  }

  private startWatch(mode: 'watch' | 'watch-cycle'): void {
    this.stop();
    this.mode.set(mode);
    void this.wakeLock.acquire(); // keep the phone screen on so an auto-lock doesn't freeze the game
    this.status.set('Loading the AI…');
    this.renderer?.begin(this.aiTickMs());
    this.director = new SnakeDirector(mode === 'watch-cycle' ? 'cycle' : 'search', this.boardSize());
    this.armWatchTimer(mode);
  }

  /** (Re)start the AI tick at the current speed. Split out so the speed picker can re-time a game in flight. */
  private armWatchTimer(mode: 'watch' | 'watch-cycle'): void {
    this.clearTimer();
    const cells = this.boardSize() * this.boardSize();
    this.timer = setInterval(() => {
      const f = this.director?.step();
      if (!f) return; // checkpoint still loading
      this.lastFrame = f;
      this.render(f.body, f.food, f.foodEaten, f.cycle, f.cycleEpoch);
      if (f.done) {
        this.status.set(f.length === cells
          ? `AI filled the whole board — a perfect game (${f.foodEaten} food). Restarting…`
          : `AI died after eating ${f.foodEaten} (length ${f.length}). Restarting…`);
      } else {
        this.status.set(mode === 'watch-cycle'
          ? 'Watching the AI — it holds a Hamiltonian cycle it can never die on, shortcutting toward the food.'
          : 'Watching the AI play — it all runs in your browser.');
      }
    }, this.aiTickMs());
  }

  // --- Human play (client-side, JS timer) ---
  protected playHuman(): void {
    this.stop();
    this.mode.set('human');
    this.status.set('Your game — arrow keys or WASD to steer.');
    this.renderer?.begin(HUMAN_TICK_MS);
    const g = this.game = new SnakeGame(this.boardSize());
    this.render(g.body, g.food, g.foodEaten);
    this.timer = setInterval(() => {
      g.tick();
      this.render(g.body, g.food, g.foodEaten);
      if (g.dead) {
        this.status.set(`Game over — you ate ${g.foodEaten} food.`);
        this.clearTimer();
      }
    }, HUMAN_TICK_MS);
  }

  protected onKey(event: KeyboardEvent): void {
    if (this.mode() !== 'human' || !this.game) return;
    const map: Record<string, Dir> = {
      ArrowUp: 0, w: 0, ArrowDown: 1, s: 1, ArrowLeft: 2, a: 2, ArrowRight: 3, d: 3,
    };
    const dir = map[event.key];
    if (dir !== undefined) { event.preventDefault(); this.game.setDirection(dir); }
  }

  protected stop(): void {
    this.clearTimer();
    this.renderer?.stop();
    this.game = null;
    this.director = null;
    this.lastFrame = null;
    void this.wakeLock.release();
    if (this.mode() !== 'idle') this.mode.set('idle');
  }

  /**
   * Board size picker. The grid is baked into the engine, the director's cycle and every cell index on screen,
   * so a size change cannot be applied to a game in flight — the running mode is restarted on the new board.
   */
  protected setBoardSize(value: number | string): void {
    const size = this.clampSize(value);
    if (size === this.boardSize()) return;
    const mode = this.mode();
    this.boardSize.set(size);
    this.renderer?.setSize(size);
    if (mode === 'human') this.playHuman();
    else if (mode !== 'idle') this.startWatch(mode);
  }

  /** Speed picker — re-times the AI tick in place, so a game in flight just speeds up or slows down. */
  protected setSpeed(value: number | string): void {
    const speed = this.clampSpeed(value);
    if (speed === this.aiSpeed()) return;
    this.aiSpeed.set(speed);
    const mode = this.mode();
    if (mode === 'watch' || mode === 'watch-cycle') {
      this.renderer?.setTickMs(this.aiTickMs());
      this.armWatchTimer(mode);
    }
  }

  /** Toggle the cycle overlay and repaint at once, rather than waiting up to a full tick for the next frame. */
  protected toggleRoute(on: boolean): void {
    this.showRoute.set(on);
    const f = this.lastFrame;
    if (f) this.render(f.body, f.food, f.foodEaten, f.cycle, f.cycleEpoch);
  }

  private aiTickMs(): number {
    return Math.round(1000 / this.aiSpeed());
  }

  /** Snap to the even grid the Hamiltonian cycle needs, and keep a blank/garbage input at the current size. */
  private clampSize(value: number | string): number {
    const n = Math.round(Number(value));
    if (value === '' || !Number.isFinite(n)) return this.boardSize();
    const even = n - (n % SIZE_STEP);
    return Math.min(MAX_SIZE, Math.max(MIN_SIZE, even));
  }

  private clampSpeed(value: number | string): number {
    const n = Math.round(Number(value));
    if (value === '' || !Number.isFinite(n)) return this.aiSpeed();
    return Math.min(MAX_SPEED, Math.max(MIN_SPEED, n));
  }

  private render(body: number[], food: number, eaten: number, cycle: number[] | null = null, epoch = 0): void {
    this.foodEaten.set(eaten);
    // Hiding is a data decision, so the renderer stays a pure function of the snapshot it is handed.
    this.renderer?.push(body, food, eaten, this.showRoute() ? cycle : null, epoch);
  }

  private clearTimer(): void {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }
}
