import { Component, DestroyRef, ElementRef, afterNextRender, inject, signal, viewChild } from '@angular/core';
import { Dir, SIZE, SnakeGame } from './snake-logic';
import { SnakeDirector } from './snake-director';
import { SnakeTubeRenderer } from './snake-renderer';
import { ScreenWakeLock } from '../screen-wake-lock';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';

const BOARD_PX = 480;
const WATCH_TICK_MS = 120;
const HUMAN_TICK_MS = 150;

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
  protected readonly mode = signal<'idle' | 'watch' | 'human'>('idle');
  protected readonly foodEaten = signal(0);
  protected readonly status = signal('Watch the self-taught AI play, or play it yourself.');

  private readonly boardRef = viewChild.required<ElementRef<HTMLCanvasElement>>('board');
  private renderer: SnakeTubeRenderer | null = null;

  private timer: ReturnType<typeof setInterval> | null = null;
  private game: SnakeGame | null = null;
  private director: SnakeDirector | null = null;

  constructor() {
    afterNextRender(() => {
      this.renderer = new SnakeTubeRenderer(this.boardRef().nativeElement, SIZE, BOARD_PX);
    });
    inject(DestroyRef).onDestroy(() => this.stop());
  }

  // --- Watch AI: the whole AI (physics + net + masked-greedy policy) runs in the browser (M33) ---
  protected watchAi(): void {
    this.stop();
    this.mode.set('watch');
    void this.wakeLock.acquire(); // keep the phone screen on so an auto-lock doesn't freeze the game
    this.status.set('Loading the AI…');
    this.renderer?.begin(WATCH_TICK_MS);
    this.director = new SnakeDirector();
    this.timer = setInterval(() => {
      const f = this.director?.step();
      if (!f) return; // checkpoint still loading
      this.render(f.body, f.food, f.foodEaten);
      this.status.set(f.done
        ? `AI died after eating ${f.foodEaten} (length ${f.length}). Restarting…`
        : 'Watching the AI play — it all runs in your browser.');
    }, WATCH_TICK_MS);
  }

  // --- Human play (client-side, JS timer) ---
  protected playHuman(): void {
    this.stop();
    this.mode.set('human');
    this.status.set('Your game — arrow keys or WASD to steer.');
    this.renderer?.begin(HUMAN_TICK_MS);
    const g = this.game = new SnakeGame();
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
    void this.wakeLock.release();
    if (this.mode() !== 'idle') this.mode.set('idle');
  }

  private render(body: number[], food: number, eaten: number): void {
    this.foodEaten.set(eaten);
    this.renderer?.push(body, food, eaten);
  }

  private clearTimer(): void {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }
}
