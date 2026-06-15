import { Component, DestroyRef, inject, signal } from '@angular/core';
import { SnakeApi, SnakeStatus } from './snake-api';
import { CELLS, Dir, SIZE, SnakeGame } from './snake-logic';

/**
 * Snake page with the two interaction principles side by side (PRD §7.1):
 * - **Watch AI** = principle B: the backend drives a live game over a WebSocket and pushes frames;
 *   this component is a pure renderer (no game timer of its own).
 * - **Play yourself** = client-side: a JS timer ticks a local engine; the backend is not involved.
 */
@Component({
  selector: 'app-snake',
  templateUrl: './snake.html',
  styleUrl: './snake.scss',
  host: { '(window:keydown)': 'onKey($event)' },
})
export class Snake {
  private readonly api = inject(SnakeApi);

  protected readonly size = SIZE;
  protected readonly cells = Array.from({ length: CELLS }, (_, i) => i);

  protected readonly mode = signal<'idle' | 'watch' | 'human'>('idle');
  protected readonly head = signal(-1);
  protected readonly bodySet = signal<Set<number>>(new Set());
  protected readonly food = signal(-1);
  protected readonly foodEaten = signal(0);
  protected readonly status = signal('Watch the self-taught AI play, or play it yourself.');
  protected readonly modelStatus = signal<SnakeStatus | null>(null);

  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private game: SnakeGame | null = null;

  constructor() {
    this.pollStatus();
    inject(DestroyRef).onDestroy(() => this.stop());
  }

  protected cellClass(i: number): string {
    if (i === this.head()) return 'cell head';
    if (this.bodySet().has(i)) return 'cell body';
    if (i === this.food()) return 'cell food';
    return 'cell';
  }

  // --- Watch AI (principle B: backend drives, we render) ---
  protected async watchAi(): Promise<void> {
    this.stop();
    const st = await this.api.status().catch(() => null);
    this.modelStatus.set(st);
    if (!st || st.status !== 'ready') {
      this.status.set(st?.status === 'training' ? 'The AI is still training — try again shortly.' : 'AI unavailable.');
      return;
    }
    this.mode.set('watch');
    this.status.set('Watching the AI play (the server drives the game)…');
    this.socket = this.api.connectLive(
      f => {
        this.render(f.body, f.food, f.foodEaten);
        if (f.done) this.status.set(`AI died after eating ${f.foodEaten} (length ${f.length}). Restarting…`);
      },
      () => { if (this.mode() === 'watch') this.status.set('Stream closed.'); });
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
    this.socket?.close();
    this.socket = null;
    this.clearTimer();
    this.game = null;
    if (this.mode() !== 'idle') this.mode.set('idle');
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

  private pollStatus(): void {
    void (async () => {
      try {
        const s = await this.api.status();
        this.modelStatus.set(s);
        if (s.status === 'loading' || s.status === 'training') setTimeout(() => this.pollStatus(), 3000);
      } catch {
        // backend unreachable — leave status unknown; buttons stay usable
      }
    })();
  }
}
