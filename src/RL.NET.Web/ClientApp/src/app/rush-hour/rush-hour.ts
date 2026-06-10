import { DecimalPipe } from '@angular/common';
import { Component, DestroyRef, ElementRef, computed, effect, inject, signal, viewChild } from '@angular/core';
import { AnalyzeResponse, RushHourApi, SolveResponse, StatusResponse, VehicleDto } from './rush-hour-api';
import { EXIT_ROW, SIZE, canMove, canPlace, initialPositions, isSolved, occupancy } from './rush-hour-logic';

type Mode = 'edit' | 'play' | 'playback';
type Tool = 'red' | 'car-h' | 'car-v' | 'truck-h' | 'truck-v' | 'erase';

const CELL = 72;
const PAD = 14;
const EXIT_W = 42;

const VEHICLE_COLORS = [
  '#e0245e', // red car
  '#3b82f6', '#22c55e', '#eab308', '#a855f7', '#14b8a6', '#f97316', '#64748b',
  '#84cc16', '#06b6d4', '#ec4899', '#8b5cf6', '#f59e0b', '#10b981', '#6366f1', '#94a3b8',
];

@Component({
  selector: 'app-rush-hour',
  imports: [DecimalPipe],
  templateUrl: './rush-hour.html',
  styleUrl: './rush-hour.scss',
  host: { '(window:keydown)': 'onKey($event)' },
})
export class RushHour {
  private readonly api = inject(RushHourApi);
  private readonly canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('board');

  // --- drawn puzzle (vehicle 0 = red car, always horizontal on the exit row) ---
  protected readonly vehicles = signal<VehicleDto[]>([{ row: EXIT_ROW, col: 0, length: 2, horizontal: true }]);
  protected readonly mode = signal<Mode>('edit');
  protected readonly tool = signal<Tool>('car-h');
  protected readonly analysis = signal<AnalyzeResponse | null>(null);
  protected readonly editMessage = signal<string | null>(null);

  // --- manual play ---
  protected readonly playPositions = signal<number[]>([]);
  protected readonly selected = signal<number | null>(null);
  protected readonly movesUsed = signal(0);
  protected readonly playWon = signal(false);

  // --- AI solution playback ---
  protected readonly solution = signal<SolveResponse | null>(null);
  protected readonly showOptimal = signal(false);
  protected readonly playbackIndex = signal(0);
  protected readonly playing = signal(false);
  protected readonly busy = signal(false);

  // --- model status ---
  protected readonly modelStatus = signal<StatusResponse | null>(null);

  private playbackTimer: ReturnType<typeof setInterval> | null = null;

  protected readonly initialPos = computed(() => initialPositions(this.vehicles()));

  protected readonly activeTrajectory = computed(() => {
    const solution = this.solution();
    if (!solution) return [];
    return this.showOptimal() ? solution.optimalTrajectory : solution.trajectory;
  });

  protected readonly displayPositions = computed(() => {
    switch (this.mode()) {
      case 'edit':
        return this.initialPos();
      case 'play':
        return this.playPositions();
      case 'playback': {
        const index = this.playbackIndex();
        return index === 0 ? this.initialPos() : this.activeTrajectory()[index - 1].positions;
      }
    }
  });

  protected readonly selectedCanMove = computed(() => {
    const index = this.selected();
    if (index === null || this.mode() !== 'play') return { back: false, forward: false };
    return {
      back: canMove(this.vehicles(), this.playPositions(), index, 0),
      forward: canMove(this.vehicles(), this.playPositions(), index, 1),
    };
  });

  constructor() {
    effect(() => this.draw());
    void this.refreshAnalysis();
    this.pollStatus();
    inject(DestroyRef).onDestroy(() => this.stopPlayback());
  }

  // ------------------------------------------------------------------ editing

  protected selectTool(tool: Tool): void {
    this.tool.set(tool);
    this.editMessage.set(null);
  }

  protected onCanvasClick(event: PointerEvent): void {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left - PAD) / CELL);
    const row = Math.floor((event.clientY - rect.top - PAD) / CELL);
    if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) return;

    if (this.mode() === 'edit') this.editCell(row, col);
    else if (this.mode() === 'play') this.playClick(row, col);
  }

  private editCell(row: number, col: number): void {
    const vehicles = this.vehicles();
    this.editMessage.set(null);

    switch (this.tool()) {
      case 'red': {
        if (row !== EXIT_ROW) {
          this.editMessage.set('The red car lives on row 3 — the exit row.');
          return;
        }
        const red: VehicleDto = { row: EXIT_ROW, col: Math.min(col, SIZE - 2), length: 2, horizontal: true };
        if (!canPlace(vehicles.slice(1), red)) {
          this.editMessage.set('Another vehicle is in the way.');
          return;
        }
        this.applyDrawing([red, ...vehicles.slice(1)]);
        break;
      }
      case 'erase': {
        const grid = occupancy(vehicles, this.initialPos());
        const index = grid[row * SIZE + col];
        if (index === 0) {
          this.editMessage.set('The red car cannot be removed — it is the goal.');
          return;
        }
        if (index > 0) this.applyDrawing(vehicles.filter((_, i) => i !== index));
        break;
      }
      default: {
        const truck = this.tool().startsWith('truck');
        const horizontal = this.tool().endsWith('-h');
        const vehicle: VehicleDto = { row, col, length: truck ? 3 : 2, horizontal };
        if (!canPlace(vehicles, vehicle)) {
          this.editMessage.set('No room there — the vehicle would overlap or leave the board.');
          return;
        }
        this.applyDrawing([...vehicles, vehicle]);
      }
    }
  }

  private applyDrawing(vehicles: VehicleDto[]): void {
    this.vehicles.set(vehicles);
    this.solution.set(null);
    void this.refreshAnalysis();
  }

  protected clearBoard(): void {
    this.applyDrawing([{ row: EXIT_ROW, col: 0, length: 2, horizontal: true }]);
    this.editMessage.set(null);
  }

  private async refreshAnalysis(): Promise<void> {
    this.analysis.set(await this.api.analyze(this.vehicles()));
  }

  // ------------------------------------------------------------------ manual play

  protected enterPlay(): void {
    this.stopPlayback();
    this.mode.set('play');
    this.resetPlay();
  }

  protected resetPlay(): void {
    this.playPositions.set(this.initialPos());
    this.movesUsed.set(0);
    this.playWon.set(false);
    this.selected.set(null);
  }

  protected backToEdit(): void {
    this.stopPlayback();
    this.mode.set('edit');
    this.selected.set(null);
  }

  private playClick(row: number, col: number): void {
    const grid = occupancy(this.vehicles(), this.playPositions());
    const index = grid[row * SIZE + col];
    if (index >= 0) this.selected.set(index);
  }

  protected tryMove(direction: number): void {
    const index = this.selected();
    if (index === null || this.playWon()) return;
    const vehicles = this.vehicles();
    const positions = this.playPositions();
    if (!canMove(vehicles, positions, index, direction)) return;

    const next = [...positions];
    next[index] += direction === 0 ? -1 : 1;
    this.playPositions.set(next);
    this.movesUsed.update(m => m + 1);
    if (isSolved(vehicles, next)) this.playWon.set(true);
  }

  protected onKey(event: KeyboardEvent): void {
    if (this.mode() !== 'play' || this.selected() === null) return;
    const horizontal = this.vehicles()[this.selected()!].horizontal;
    const map: Record<string, number | undefined> = horizontal
      ? { ArrowLeft: 0, ArrowRight: 1 }
      : { ArrowUp: 0, ArrowDown: 1 };
    const direction = map[event.key];
    if (direction !== undefined) {
      event.preventDefault();
      this.tryMove(direction);
    }
  }

  // ------------------------------------------------------------------ AI solve + playback

  protected async solve(): Promise<void> {
    this.busy.set(true);
    this.editMessage.set(null);
    try {
      const result = await this.api.solve(this.vehicles());
      switch (result.kind) {
        case 'solved':
          this.solution.set(result.value);
          this.showOptimal.set(false);
          this.playbackIndex.set(0);
          this.mode.set('playback');
          break;
        case 'training':
          this.modelStatus.set(result.status);
          this.editMessage.set('The model is still training — try again in a moment.');
          this.pollStatus();
          break;
        case 'invalid':
          this.editMessage.set(result.error);
          break;
      }
    } finally {
      this.busy.set(false);
    }
  }

  protected switchTrajectory(optimal: boolean): void {
    this.stopPlayback();
    this.showOptimal.set(optimal);
    this.playbackIndex.set(0);
  }

  protected step(delta: number): void {
    this.stopPlayback();
    this.seek(this.playbackIndex() + delta);
  }

  protected seek(index: number): void {
    const max = this.activeTrajectory().length;
    this.playbackIndex.set(Math.max(0, Math.min(max, index)));
  }

  protected togglePlay(): void {
    if (this.playing()) {
      this.stopPlayback();
      return;
    }
    if (this.playbackIndex() >= this.activeTrajectory().length) this.playbackIndex.set(0);
    this.playing.set(true);
    this.playbackTimer = setInterval(() => {
      const next = this.playbackIndex() + 1;
      if (next > this.activeTrajectory().length) {
        this.stopPlayback();
        return;
      }
      this.playbackIndex.set(next);
      if (next === this.activeTrajectory().length) this.stopPlayback();
    }, 380);
  }

  protected stopPlayback(): void {
    if (this.playbackTimer) {
      clearInterval(this.playbackTimer);
      this.playbackTimer = null;
    }
    this.playing.set(false);
  }

  // ------------------------------------------------------------------ model status

  private pollStatus(): void {
    void (async () => {
      const status = await this.api.status();
      this.modelStatus.set(status);
      if (status.status === 'loading' || status.status === 'training') {
        setTimeout(() => this.pollStatus(), 2000);
      }
    })();
  }

  // ------------------------------------------------------------------ canvas

  private draw(): void {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return;

    const positions = this.displayPositions();
    const vehicles = this.vehicles();
    const width = PAD * 2 + SIZE * CELL + EXIT_W;
    const height = PAD * 2 + SIZE * CELL;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Board background + cells.
    ctx.fillStyle = '#1a1f2b';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#222938';
    ctx.fillRect(PAD, PAD, SIZE * CELL, SIZE * CELL);
    ctx.strokeStyle = '#323b52';
    ctx.lineWidth = 1;
    for (let i = 0; i <= SIZE; i++) {
      ctx.beginPath();
      ctx.moveTo(PAD + i * CELL, PAD);
      ctx.lineTo(PAD + i * CELL, PAD + SIZE * CELL);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(PAD, PAD + i * CELL);
      ctx.lineTo(PAD + SIZE * CELL, PAD + i * CELL);
      ctx.stroke();
    }

    // Exit notch + arrow on the exit row.
    const exitY = PAD + EXIT_ROW * CELL;
    ctx.fillStyle = '#1a1f2b';
    ctx.fillRect(PAD + SIZE * CELL, exitY + 6, 4, CELL - 12);
    ctx.fillStyle = '#6ea8fe';
    ctx.font = 'bold 17px system-ui';
    ctx.textBaseline = 'middle';
    ctx.fillText('EXIT →', PAD + SIZE * CELL + 7, exitY + CELL / 2);

    // Vehicles.
    const lastStep = this.mode() === 'playback' && this.playbackIndex() > 0
      ? this.activeTrajectory()[this.playbackIndex() - 1]
      : null;

    vehicles.forEach((v, i) => {
      const row = v.horizontal ? v.row : positions[i];
      const col = v.horizontal ? positions[i] : v.col;
      const x = PAD + col * CELL + 5;
      const y = PAD + row * CELL + 5;
      const w = (v.horizontal ? v.length : 1) * CELL - 10;
      const h = (v.horizontal ? 1 : v.length) * CELL - 10;

      ctx.fillStyle = VEHICLE_COLORS[i % VEHICLE_COLORS.length];
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 12);
      ctx.fill();

      if (this.mode() === 'play' && this.selected() === i) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.stroke();
      } else if (lastStep?.vehicle === i) {
        ctx.strokeStyle = '#ffd166';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.font = 'bold 19px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(i === 0 ? 'R' : String.fromCharCode(64 + i), x + w / 2, y + h / 2 + 1);
      ctx.textAlign = 'start';
    });
  }
}
