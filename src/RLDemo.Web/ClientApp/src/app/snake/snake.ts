import { Component, DestroyRef, inject, signal } from '@angular/core';
import { CELLS, Dir, SIZE, SnakeGame } from './snake-logic';
import { SnakeDirector } from './snake-director';
import { ScreenWakeLock } from '../screen-wake-lock';

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
  host: { '(window:keydown)': 'onKey($event)', '(document:visibilitychange)': 'onVisibilityChange()' },
})
export class Snake {
  private readonly wakeLock = inject(ScreenWakeLock);

  protected readonly size = SIZE;
  protected readonly cells = Array.from({ length: CELLS }, (_, i) => i);

  protected readonly mode = signal<'idle' | 'watch' | 'human'>('idle');
  protected readonly head = signal(-1);
  protected readonly bodySet = signal<Set<number>>(new Set());
  protected readonly food = signal(-1);
  protected readonly foodEaten = signal(0);
  protected readonly status = signal('Watch the self-taught AI play, or play it yourself.');

  private timer: ReturnType<typeof setInterval> | null = null;
  private game: SnakeGame | null = null;
  private director: SnakeDirector | null = null;

  constructor() {
    inject(DestroyRef).onDestroy(() => this.stop());
  }

  protected cellClass(i: number): string {
    if (i === this.head()) return 'cell head';
    if (this.bodySet().has(i)) return 'cell body';
    if (i === this.food()) return 'cell food';
    return 'cell';
  }

  // --- Watch AI: the whole AI (physics + net + masked-greedy policy) runs in the browser (M33) ---
  protected watchAi(): void {
    this.stop();
    this.mode.set('watch');
    void this.wakeLock.acquire(); // keep the phone screen on so an auto-lock doesn't freeze the game
    this.status.set('Loading the AI…');
    this.director = new SnakeDirector();
    this.timer = setInterval(() => {
      const f = this.director?.step();
      if (!f) return; // checkpoint still loading
      this.render(f.body, f.food, f.foodEaten);
      this.status.set(f.done
        ? `AI died after eating ${f.foodEaten} (length ${f.length}). Restarting…`
        : 'Watching the AI play — it all runs in your browser.');
    }, 120);
  }

  // --- Human play (client-side, JS timer) ---
  protected playHuman(): void {
    this.stop();
    this.mode.set('human');
    this.status.set('Your game — arrow keys or WASD to steer.');
    const g = this.game = new SnakeGame();
    this.render(g.body, g.food, g.foodEaten);
    this.timer = setInterval(() => {
      g.tick();
      this.render(g.body, g.food, g.foodEaten);
      if (g.dead) {
        this.status.set(`Game over — you ate ${g.foodEaten} food.`);
        this.clearTimer();
      }
    }, 150);
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
    this.game = null;
    this.director = null;
    void this.wakeLock.release();
    if (this.mode() !== 'idle') this.mode.set('idle');
  }

  // A backgrounded tab is throttled by the OS and drops the wake-lock; on returning to the foreground still in
  // watch mode, re-acquire it (the director just keeps ticking — no socket to reopen).
  protected onVisibilityChange(): void {
    if (document.visibilityState === 'visible' && this.mode() === 'watch') void this.wakeLock.acquire();
  }

  private render(body: number[], food: number, eaten: number): void {
    this.head.set(body[0] ?? -1);
    this.bodySet.set(new Set(body));
    this.food.set(food);
    this.foodEaten.set(eaten);
  }

  private clearTimer(): void {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }
}
