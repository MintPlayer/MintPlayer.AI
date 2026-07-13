import { DecimalPipe } from '@angular/common';
import { Component, DestroyRef, ElementRef, computed, inject, signal, viewChild } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { Game2048Api, SolveResponse2048, Status2048 } from './game-2048-api';
import { ClassicEngine, RenderTile } from './game-2048-classic';
import { Board, exponentOf } from './game-2048-logic';
import { pollModelStatus } from '../model-status';

type Mode = 'edit' | 'play' | 'playback';

/** Auto-play pacing: slightly above the 100 ms slide transition so every move reads. */
const PLAYBACK_INTERVAL = 120;

@Component({
  selector: 'app-game-2048',
  imports: [DecimalPipe, BsButtonTypeDirective],
  templateUrl: './game-2048.html',
  styleUrl: './game-2048.scss',
  host: { '(window:keydown)': 'onKey($event)' },
})
export class Game2048 {
  private readonly api = inject(Game2048Api);
  private readonly route = inject(ActivatedRoute);
  private readonly boardRef = viewChild<ElementRef<HTMLElement>>('board');

  protected readonly gridCells = Array.from({ length: 16 });
  protected readonly colors = Color;

  // Drawn board (exponents). Default: two starter tiles, like a fresh game.
  protected readonly drawn = signal<Board>(startingBoard());
  protected readonly mode = signal<Mode>('edit');
  protected readonly message = signal<string | null>(null);

  // The classic DOM board: tiles with identities, previous positions and merge/spawn
  // states so the original 2048 animations (slide / pop / appear) play.
  protected readonly tiles = signal<RenderTile[]>([]);
  private engine: ClassicEngine = new ClassicEngine();

  // Manual play.
  protected readonly playScore = signal(0);
  protected readonly playOver = signal(false);
  protected readonly playMax = signal(0);
  protected readonly scoreAdditions = signal<{ id: number; amount: number }[]>([]);
  private nextAdditionId = 1;

  // AI playback: exponent boards reconstructed once for instant scrubbing; forward
  // steps replay through the classic engine so they animate like real moves.
  protected readonly solution = signal<SolveResponse2048 | null>(null);
  private playbackStates: Board[] = [];
  private playbackScores: number[] = [];
  private engineIndex = 0;
  protected readonly playbackIndex = signal(0);
  protected readonly playing = signal(false);
  protected readonly busy = signal(false);

  protected readonly modelStatus = signal<Status2048 | null>(null);

  private playbackTimer: ReturnType<typeof setInterval> | null = null;
  private touchStart: { x: number; y: number } | null = null;

  protected readonly playbackScore = computed(() => this.playbackScores[this.playbackIndex()] ?? 0);
  protected readonly stepCount = computed(() => this.solution()?.steps.length ?? 0);

  constructor() {
    this.showStatic(this.drawn());
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

  // ------------------------------------------------------------------ rendering

  /** Rebuilds the board without animation (editing, seeking, resets). */
  private showStatic(board: Board): void {
    this.engine = ClassicEngine.fromExponents(board);
    this.tiles.set(this.engine.renderTiles());
  }

  protected tileClass(tile: RenderTile): string {
    const value = tile.value <= 2048 ? tile.value : 'super';
    let classes = `tile tile-${value} tile-position-${tile.x + 1}-${tile.y + 1}`;
    if (tile.state === 'new') classes += ' tile-new';
    else if (tile.state === 'merged') classes += ' tile-merged';
    return classes;
  }

  // ------------------------------------------------------------------ editing

  protected onBoardClick(event: PointerEvent): void {
    if (this.mode() !== 'edit') return;
    const index = this.cellAt(event);
    if (index < 0) return;
    const cells = [...this.drawn()];
    cells[index] = (cells[index] + 1) % 16; // cycle empty → 2 → 4 → … → 32768 → empty
    this.applyDrawing(cells);
  }

  protected onBoardRightClick(event: MouseEvent): void {
    event.preventDefault();
    if (this.mode() !== 'edit') return;
    const index = this.cellAt(event);
    if (index < 0) return;
    const cells = [...this.drawn()];
    cells[index] = (cells[index] + 15) % 16; // cycle down
    this.applyDrawing(cells);
  }

  private applyDrawing(cells: Board): void {
    this.drawn.set(cells);
    this.solution.set(null);
    this.message.set(null);
    this.showStatic(cells);
  }

  private cellAt(event: { clientX: number; clientY: number }): number {
    const board = this.boardRef()?.nativeElement;
    if (!board) return -1;
    const rect = board.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * 4);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * 4);
    return row >= 0 && row < 4 && col >= 0 && col < 4 ? row * 4 + col : -1;
  }

  protected clearBoard(): void {
    this.applyDrawing(startingBoard());
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
    this.showStatic(this.drawn());
    this.playScore.set(0);
    this.playOver.set(false);
    this.playMax.set(this.engine.maxTile());
  }

  protected backToEdit(): void {
    this.stopPlayback();
    this.mode.set('edit');
    this.showStatic(this.drawn());
  }

  protected onKey(event: KeyboardEvent): void {
    if (this.mode() !== 'play' || this.playOver()) return;
    const map: Record<string, number | undefined> =
      { ArrowLeft: 0, ArrowDown: 1, ArrowRight: 2, ArrowUp: 3 };
    const action = map[event.key];
    if (action === undefined) return;
    event.preventDefault();
    this.playMove(action);
  }

  // Touch swipes, like the original game.
  protected onTouchStart(event: TouchEvent): void {
    if (event.touches.length !== 1) return;
    this.touchStart = { x: event.touches[0].clientX, y: event.touches[0].clientY };
  }

  protected onTouchEnd(event: TouchEvent): void {
    if (!this.touchStart || this.mode() !== 'play' || this.playOver()) return;
    const dx = event.changedTouches[0].clientX - this.touchStart.x;
    const dy = event.changedTouches[0].clientY - this.touchStart.y;
    this.touchStart = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 10) return;
    event.preventDefault();
    // Server action ids: 0=left 1=down 2=right 3=up.
    const action = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 2 : 0) : (dy > 0 ? 1 : 3);
    this.playMove(action);
  }

  private playMove(action: number): void {
    const { moved, gained } = this.engine.move(action);
    if (!moved) return;
    this.engine.addRandomTile();
    this.tiles.set(this.engine.renderTiles());
    this.playScore.update(s => s + gained);
    this.playMax.set(this.engine.maxTile());
    if (gained > 0) this.floatScore(gained);
    if (!this.engine.movesAvailable()) this.playOver.set(true);
  }

  /** The original's "+4" score float. */
  private floatScore(amount: number): void {
    const id = this.nextAdditionId++;
    this.scoreAdditions.update(list => [...list, { id, amount }]);
    setTimeout(() => this.scoreAdditions.update(list => list.filter(a => a.id !== id)), 600);
  }

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
        case 'loading':
          this.modelStatus.set(result.status);
          this.message.set('The model is still loading — try again in a moment.');
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
    // Reconstruct every board state once (actions + spawn events are deterministic),
    // for instant scrubbing and as the replay checksum against finalCells.
    const states: Board[] = [];
    const scores: number[] = [];
    const replay = ClassicEngine.fromExponents(solution.initialCells.map(exponentOf));
    let score = 0;
    states.push(replay.toExponents());
    scores.push(0);
    for (const step of solution.steps) {
      replay.move(step.action);
      replay.addSpecificTile(step.spawnIndex, step.spawnValue);
      score += step.scoreGained;
      states.push(replay.toExponents());
      scores.push(score);
    }
    const finalExponents = solution.finalCells.map(exponentOf);
    if (states[states.length - 1].some((cell, i) => cell !== finalExponents[i])) {
      console.warn('2048 playback reconstruction does not match finalCells — engine parity bug?');
    }

    this.playbackStates = states;
    this.playbackScores = scores;
    this.solution.set(solution);
    this.playbackIndex.set(0);
    this.mode.set('playback');
    this.seek(0);
  }

  protected seek(index: number): void {
    const clamped = Math.max(0, Math.min(this.stepCount(), index));
    this.playbackIndex.set(clamped);
    this.showStatic(this.playbackStates[clamped] ?? this.drawn());
    this.engineIndex = clamped;
  }

  protected step(delta: number): void {
    this.stopPlayback();
    if (delta === 1) this.stepForwardAnimated();
    else this.seek(this.playbackIndex() + delta);
  }

  /** Forward steps replay the server's (action, spawn) through the classic engine — animated. */
  private stepForwardAnimated(): void {
    const solution = this.solution();
    const index = this.playbackIndex();
    if (!solution || index >= this.stepCount()) return;
    if (this.engineIndex !== index) this.seek(index);

    const step = solution.steps[index];
    this.engine.move(step.action);
    this.engine.addSpecificTile(step.spawnIndex, step.spawnValue);
    this.tiles.set(this.engine.renderTiles());
    this.engineIndex = index + 1;
    this.playbackIndex.set(index + 1);
  }

  protected togglePlay(): void {
    if (this.playing()) {
      this.stopPlayback();
      return;
    }
    if (this.playbackIndex() >= this.stepCount()) this.seek(0);
    this.playing.set(true);
    this.playbackTimer = setInterval(() => {
      this.stepForwardAnimated();
      if (this.playbackIndex() >= this.stepCount()) this.stopPlayback();
    }, PLAYBACK_INTERVAL);
  }

  protected stopPlayback(): void {
    if (this.playbackTimer) {
      clearInterval(this.playbackTimer);
      this.playbackTimer = null;
    }
    this.playing.set(false);
  }

  // ------------------------------------------------------------------ status

  private pollStatus(): void {
    pollModelStatus(() => this.api.status(), (s) => this.modelStatus.set(s));
  }

  protected playMaxTile = () => this.playMax();
}

function startingBoard(): Board {
  const board = new Array<number>(16).fill(0);
  board[5] = 1;
  board[10] = 1;
  return board;
}
