import { Component, DestroyRef, ElementRef, afterNextRender, computed, inject, signal, viewChild } from '@angular/core';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { LUNAR_SIZE, LunarLockoutRenderer, LunarSnapshot } from './lunar-lockout-render';
import { PgLunarBoard, PgLunarOracle } from './lunarlockout_solver';

interface LunarLevel {
  name: string;
  grid: string[];
  optimalMoves: number;
}

/**
 * Lunar Lockout page. Entirely client-side: the rules AND the exact BFS oracle come from the single-source
 * `.pg`, so the hint here is the same search that verified every shipped level's move count during the build.
 *
 * Robots slide until they hit another robot; the board edge is NOT a backstop, so a slide with nothing ahead
 * is illegal. Win by parking the target robot on the centre.
 */
@Component({
  selector: 'app-lunar-lockout',
  templateUrl: './lunar-lockout.html',
  styleUrl: './lunar-lockout.scss',
  imports: [BsButtonTypeDirective],
  host: { '(window:keydown)': 'onKeyDown($event)' },
})
export class LunarLockout {
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('lunarCanvas');

  protected readonly colors = Color;
  protected readonly levels = signal<LunarLevel[]>([]);
  protected readonly levelIndex = signal(0);
  protected readonly moves = signal(0);
  protected readonly solved = signal(false);
  protected readonly status = signal('Loading levels…');
  protected readonly hint = signal<string | null>(null);

  protected readonly level = computed(() => this.levels()[this.levelIndex()]);
  protected readonly canUndo = computed(() => this.history().length > 0);
  private readonly history = signal<PgLunarBoard[]>([]);

  private renderer: LunarLockoutRenderer | null = null;
  private board: PgLunarBoard | null = null;
  private selected = -1;

  constructor() {
    afterNextRender(() => {
      this.renderer = new LunarLockoutRenderer(this.canvasRef().nativeElement);
      void this.load();
    });
    inject(DestroyRef).onDestroy(() => this.renderer?.dispose());
  }

  /** Levels come from the SAME JSON the training campaign embeds — one source of truth, no drift. */
  private async load(): Promise<void> {
    try {
      const response = await fetch('levels/lunarlockout-levels.json');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const document = await response.json() as { levels: LunarLevel[] };
      this.levels.set(document.levels ?? []);
      if (this.levels().length === 0) throw new Error('the level pack is empty');
      this.reset();
    } catch (error) {
      this.status.set(`Could not load the levels (${error instanceof Error ? error.message : error}).`);
    }
  }

  protected reset(): void {
    const level = this.level();
    if (!level) return;
    this.board = PgLunarBoard.fromGrid(level.grid);
    this.history.set([]);
    this.moves.set(0);
    this.solved.set(false);
    this.selected = -1;
    this.hint.set(null);
    this.status.set(`${level.name} — solvable in ${level.optimalMoves} move${level.optimalMoves === 1 ? '' : 's'}.`);
    this.paint();
  }

  protected changeLevel(delta: number): void {
    const next = this.levelIndex() + delta;
    if (next < 0 || next >= this.levels().length) return;
    this.levelIndex.set(next);
    this.reset();
  }

  protected undo(): void {
    const stack = this.history();
    if (stack.length === 0) return;
    this.board = stack[stack.length - 1];
    this.history.set(stack.slice(0, -1));
    this.moves.update(n => Math.max(0, n - 1));
    this.solved.set(false);
    this.paint();
  }

  /** The exact oracle, running in the browser — the same code that verified the shipped move counts. */
  protected showHint(): void {
    if (!this.board || this.solved()) return;
    const oracle = new PgLunarOracle(this.board, 200000);
    const distance = oracle.optimalFromStart();
    this.hint.set(distance < 0
      ? 'No solution from here — undo a move.'
      : `${distance} move${distance === 1 ? '' : 's'} from here.`);
  }

  protected onCanvasPointer(event: PointerEvent): void {
    if (!this.board || this.solved()) return;
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * LUNAR_SIZE);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * LUNAR_SIZE);
    if (col < 0 || col >= LUNAR_SIZE || row < 0 || row >= LUNAR_SIZE) return;

    const cell = row * LUNAR_SIZE + col;
    const robot = this.board.robots.indexOf(cell);
    if (robot >= 0) {
      this.selected = this.selected === robot ? -1 : robot;
      this.paint();
    }
  }

  protected onKeyDown(event: KeyboardEvent): void {
    const directions: Record<string, number> = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3 };
    if (event.key in directions) {
      event.preventDefault();
      this.slide(directions[event.key]);
      return;
    }
    if (event.key === 'r' || event.key === 'R') this.reset();
    if (event.key === 'z' || event.key === 'Z') this.undo();
    // Number keys pick a robot, so the board is playable without a pointer.
    const digit = Number.parseInt(event.key, 10);
    if (!Number.isNaN(digit) && this.board && digit >= 1 && digit <= this.board.robots.length) {
      this.selected = digit - 1;
      this.paint();
    }
  }

  protected slide(direction: number): void {
    if (!this.board || this.solved()) return;
    if (this.selected < 0) {
      this.status.set('Pick a robot first — tap it, or press its number.');
      return;
    }

    const from = this.board.robots[this.selected];
    const action = this.selected * 4 + direction;
    if (!this.board.isLegal(action)) {
      // Not a UI failure: a robot with nothing ahead genuinely cannot move, because the edge is not a backstop.
      this.status.set('That robot has nothing to stop it that way.');
      return;
    }

    const previous = this.board;
    this.board = this.board.applyAction(action);
    this.history.update(stack => [...stack, previous]);
    this.moves.update(n => n + 1);
    this.hint.set(null);

    if (this.board.isSolved()) {
      this.solved.set(true);
      const optimal = this.level()?.optimalMoves ?? 0;
      this.status.set(this.moves() === optimal
        ? `Solved in ${this.moves()} — that is optimal.`
        : `Solved in ${this.moves()}; the optimum is ${optimal}.`);
    }
    this.paint({ robot: this.selected, from });
  }

  private paint(slide?: { robot: number; from: number }): void {
    if (!this.renderer || !this.board) return;
    this.renderer.push(this.snapshot(), slide);
  }

  private snapshot(): LunarSnapshot {
    const board = this.board!;
    const hints: number[] = [];
    if (this.selected >= 0 && !this.solved()) {
      for (let direction = 0; direction < 4; direction++) {
        const landing = board.landing(this.selected, direction);
        if (landing >= 0) hints.push(landing);
      }
    }
    return {
      robots: [...board.robots],
      selected: this.selected,
      hints,
      solved: this.solved(),
      moves: this.moves(),
    };
  }
}
