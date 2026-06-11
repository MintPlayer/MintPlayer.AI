import { DecimalPipe } from '@angular/common';
import { Component, DestroyRef, ElementRef, computed, effect, inject, signal, viewChild } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Game2048Api, SolveResponse2048, Status2048 } from './game-2048-api';
import { Board, anyMoveAvailable, applyMove, exponentOf, maxTile, spawn } from './game-2048-logic';

type Mode = 'edit' | 'play' | 'playback';

const CELL = 96;
const PAD = 12;
const GAP = 8;

const TILE_COLORS: Record<number, string> = {
  1: '#3a4154', 2: '#4a5168', 3: '#b8722d', 4: '#c4621a', 5: '#d35028', 6: '#d93a23',
  7: '#c9a227', 8: '#c79a1d', 9: '#c59213', 10: '#c38a0a', 11: '#b07d1c', 12: '#7c5cd6',
  13: '#6a4fc4', 14: '#5843b2', 15: '#4637a0',
};

@Component({
  selector: 'app-game-2048',
  imports: [DecimalPipe],
  templateUrl: './game-2048.html',
  styleUrl: './game-2048.scss',
  host: { '(window:keydown)': 'onKey($event)' },
})
export class Game2048 {
  private readonly api = inject(Game2048Api);
  private readonly route = inject(ActivatedRoute);
  private readonly canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('board');

  // Drawn board (exponents). Default: two starter tiles, like a fresh game.
  protected readonly drawn = signal<Board>(startingBoard());
  protected readonly mode = signal<Mode>('edit');
  protected readonly message = signal<string | null>(null);

  // Manual play.
  protected readonly playBoard = signal<Board>([]);
  protected readonly playScore = signal(0);
  protected readonly playOver = signal(false);

  // AI playback: board states reconstructed once, then scrubbed instantly.
  protected readonly solution = signal<SolveResponse2048 | null>(null);
  private playbackStates: Board[] = [];
  private playbackScores: number[] = [];
  protected readonly playbackIndex = signal(0);
  protected readonly playing = signal(false);
  protected readonly busy = signal(false);

  protected readonly modelStatus = signal<Status2048 | null>(null);

  private playbackTimer: ReturnType<typeof setInterval> | null = null;

  protected readonly displayBoard = computed<Board>(() => {
    switch (this.mode()) {
      case 'edit': return this.drawn();
      case 'play': return this.playBoard();
      case 'playback': return this.playbackStates[this.playbackIndex()] ?? this.drawn();
    }
  });

  protected readonly playbackScore = computed(() => this.playbackScores[this.playbackIndex()] ?? 0);
  protected readonly stepCount = computed(() => this.solution()?.steps.length ?? 0);

  constructor() {
    effect(() => this.draw());
    this.pollStatus();
    inject(DestroyRef).onDestroy(() => this.stopPlayback());

    const replayId = this.route.snapshot.queryParamMap.get('replay');
    if (replayId) void this.loadGalleryEntry(replayId);
  }

  private async loadGalleryEntry(id: string): Promise<void> {
    const response = await fetch(`/api/gallery/${id}`);
    if (!response.ok) return;
    const entry = await response.json();
    if (entry.game !== '2048') return;
    const solution: SolveResponse2048 = entry.response;
    this.drawn.set(solution.initialCells.map(exponentOf));
    this.startPlayback(solution);
  }

  // ------------------------------------------------------------------ editing

  protected onCanvasClick(event: PointerEvent): void {
    const index = this.cellAt(event);
    if (index < 0) return;

    if (this.mode() === 'edit') {
      const cells = [...this.drawn()];
      cells[index] = (cells[index] + 1) % 16; // cycle empty → 2 → 4 → … → 32768 → empty
      this.drawn.set(cells);
      this.solution.set(null);
      this.message.set(null);
    }
  }

  protected onCanvasRightClick(event: MouseEvent): void {
    event.preventDefault();
    if (this.mode() !== 'edit') return;
    const index = this.cellAt(event);
    if (index < 0) return;
    const cells = [...this.drawn()];
    cells[index] = (cells[index] + 15) % 16; // cycle down
    this.drawn.set(cells);
    this.solution.set(null);
  }

  private cellAt(event: { clientX: number; clientY: number }): number {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return -1;
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left - PAD) / (CELL + GAP));
    const row = Math.floor((event.clientY - rect.top - PAD) / (CELL + GAP));
    return row >= 0 && row < 4 && col >= 0 && col < 4 ? row * 4 + col : -1;
  }

  protected clearBoard(): void {
    this.drawn.set(startingBoard());
    this.solution.set(null);
    this.message.set(null);
  }

  // ------------------------------------------------------------------ manual play

  protected enterPlay(): void {
    if (this.drawn().every(t => t === 0)) {
      this.message.set('Place at least one tile first.');
      return;
    }
    this.stopPlayback();
    this.mode.set('play');
    this.resetPlay();
  }

  protected resetPlay(): void {
    this.playBoard.set([...this.drawn()]);
    this.playScore.set(0);
    this.playOver.set(false);
  }

  protected backToEdit(): void {
    this.stopPlayback();
    this.mode.set('edit');
  }

  protected onKey(event: KeyboardEvent): void {
    if (this.mode() !== 'play' || this.playOver()) return;
    const map: Record<string, number | undefined> =
      { ArrowLeft: 0, ArrowDown: 1, ArrowRight: 2, ArrowUp: 3 };
    const action = map[event.key];
    if (action === undefined) return;
    event.preventDefault();

    const board = [...this.playBoard()];
    const { moved, gained } = applyMove(board, action);
    if (!moved) return;
    spawn(board);
    this.playBoard.set(board);
    this.playScore.update(s => s + gained);
    if (!anyMoveAvailable(board)) this.playOver.set(true);
  }

  protected playMaxTile = () => maxTile(this.playBoard());

  // ------------------------------------------------------------------ AI playout

  protected async solve(): Promise<void> {
    this.busy.set(true);
    this.message.set(null);
    try {
      const values = this.drawn().map(e => (e === 0 ? 0 : 1 << e));
      const result = await this.api.solve(values);
      switch (result.kind) {
        case 'solved':
          this.startPlayback(result.value);
          break;
        case 'training':
          this.modelStatus.set(result.status);
          this.message.set('The model is still training — try again in a moment.');
          this.pollStatus();
          break;
        case 'invalid':
          this.message.set(result.error);
          break;
      }
    } finally {
      this.busy.set(false);
    }
  }

  private startPlayback(solution: SolveResponse2048): void {
    // Reconstruct every board state once (actions + spawn events are deterministic).
    const states: Board[] = [];
    const scores: number[] = [];
    const board = solution.initialCells.map(exponentOf);
    let score = 0;
    states.push([...board]);
    scores.push(0);
    for (const step of solution.steps) {
      applyMove(board, step.action);
      board[step.spawnIndex] = exponentOf(step.spawnValue);
      score += step.scoreGained;
      states.push([...board]);
      scores.push(score);
    }
    this.playbackStates = states;
    this.playbackScores = scores;
    this.solution.set(solution);
    this.playbackIndex.set(0);
    this.mode.set('playback');
  }

  protected seek(index: number): void {
    this.playbackIndex.set(Math.max(0, Math.min(this.stepCount(), index)));
  }

  protected step(delta: number): void {
    this.stopPlayback();
    this.seek(this.playbackIndex() + delta);
  }

  protected togglePlay(): void {
    if (this.playing()) {
      this.stopPlayback();
      return;
    }
    if (this.playbackIndex() >= this.stepCount()) this.playbackIndex.set(0);
    this.playing.set(true);
    this.playbackTimer = setInterval(() => {
      const next = this.playbackIndex() + 1;
      this.playbackIndex.set(next);
      if (next >= this.stepCount()) this.stopPlayback();
    }, 70);
  }

  protected stopPlayback(): void {
    if (this.playbackTimer) {
      clearInterval(this.playbackTimer);
      this.playbackTimer = null;
    }
    this.playing.set(false);
  }

  // ------------------------------------------------------------------ status + canvas

  private pollStatus(): void {
    void (async () => {
      const status = await this.api.status();
      this.modelStatus.set(status);
      if (status.status === 'loading' || status.status === 'training') {
        setTimeout(() => this.pollStatus(), 2000);
      }
    })();
  }

  private draw(): void {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return;

    const board = this.displayBoard();
    const size = PAD * 2 + 4 * CELL + 3 * GAP;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#1a1f2b';
    ctx.fillRect(0, 0, size, size);

    for (let i = 0; i < 16; i++) {
      const x = PAD + (i % 4) * (CELL + GAP);
      const y = PAD + Math.floor(i / 4) * (CELL + GAP);
      const exponent = board[i];

      ctx.fillStyle = exponent === 0 ? '#222938' : (TILE_COLORS[exponent] ?? '#4637a0');
      ctx.beginPath();
      ctx.roundRect(x, y, CELL, CELL, 10);
      ctx.fill();

      if (exponent > 0) {
        const value = 1 << exponent;
        ctx.fillStyle = '#ffffff';
        ctx.font = `bold ${value < 1000 ? 30 : value < 10000 ? 24 : 20}px system-ui`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(value), x + CELL / 2, y + CELL / 2 + 1);
      }
    }
    ctx.textAlign = 'start';
  }
}

function startingBoard(): Board {
  const board = new Array<number>(16).fill(0);
  board[5] = 1;
  board[10] = 1;
  return board;
}
