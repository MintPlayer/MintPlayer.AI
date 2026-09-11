import { Component, DestroyRef, ElementRef, afterNextRender, computed, effect, inject, isDevMode, signal, viewChild } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { AnalyzeResponse, DeckLevel, RushHourApi, SolveResponse, StatusResponse, VehicleDto } from './rush-hour-api';
import { EXIT_ROW, SIZE, canMove, canPlace, clampRange, initialPositions, isSolved, occupancy } from './rush-hour-logic';
import { pollModelStatus } from '../model-status';
import { isTypingTarget } from '../keyboard-target';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';

type Mode = 'edit' | 'play' | 'playback';
type Tool = 'red' | 'red-truck' | 'car-h' | 'car-v' | 'truck-h' | 'truck-v' | 'erase';

const CELL = 72;
const PAD = 14;
const EXIT_W = 42;
const LOGICAL_W = PAD * 2 + SIZE * CELL + EXIT_W;
const LOGICAL_H = PAD * 2 + SIZE * CELL;

/** Axis travel, in cells, before a press becomes a drag rather than a tap. Shared with Lunar Lockout. */
const DRAG_MIN = 0.18;
/** Settling to the nearest cell covers at most half a cell, so a fixed duration beats a per-distance formula. */
const SETTLE_MS = 90;

/** A vehicle being dragged along its own axis. Plain fields, never signals — see `kick()`. */
interface Drag {
  pointerId: number;
  vehicle: number;
  horizontal: boolean;
  startPos: number;
  /** Where inside the vehicle the pointer grabbed it, in cells. Re-anchored whenever the vehicle is clamped. */
  offset: number;
  lo: number;
  hi: number;
  posF: number;
  moved: boolean;
}

const VEHICLE_COLORS = [
  '#e0245e', // red car
  '#3b82f6', '#22c55e', '#eab308', '#a855f7', '#14b8a6', '#f97316', '#64748b',
  '#84cc16', '#06b6d4', '#ec4899', '#8b5cf6', '#f59e0b', '#10b981', '#6366f1', '#94a3b8',
];

@Component({
  selector: 'app-rush-hour',
  templateUrl: './rush-hour.html',
  styleUrl: './rush-hour.scss',
  imports: [BsButtonTypeDirective],
  host: { '(window:keydown)': 'onKey($event)' },
})
export class RushHour {
  protected readonly colors = Color;
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

  // --- curated level deck ---
  protected readonly deck = signal<DeckLevel[]>([]);
  protected readonly levelName = signal('');
  protected readonly currentLevelId = signal<string | null>(null);
  protected readonly deckMessage = signal<string | null>(null);
  protected readonly canEditDeck = isDevMode(); // authoring UI shows only under `ng serve`

  private playbackTimer: ReturnType<typeof setInterval> | null = null;

  // Drag state is deliberately NOT signals. `effect(() => this.draw())` repaints the whole board on every signal
  // write, and a pointermove can fire far faster than a frame — so per-frame state lives in plain fields and a
  // local rAF drives the repaint. Signals are written exactly once, when the move commits.
  private drag: Drag | null = null;
  private settle: { vehicle: number; from: number; to: number; startedAt: number } | null = null;
  private frame = 0;

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

    // The backing store is sized from the element, so it must follow the element. Without this, any layout
    // change that does not also write a signal — switching to play mode, resizing the window, rotating a phone —
    // leaves a stale buffer that the browser simply scales up, which looks like a blurry, wrongly-sized board.
    const destroyRef = inject(DestroyRef);
    afterNextRender(() => {
      const canvas = this.canvasRef()?.nativeElement;
      if (!canvas || typeof ResizeObserver === 'undefined') return;
      const observer = new ResizeObserver(() => this.draw());
      observer.observe(canvas);
      destroyRef.onDestroy(() => observer.disconnect());
    });
    void this.refreshAnalysis();
    void this.loadDeck();
    this.pollStatus();
    destroyRef.onDestroy(() => this.stopPlayback());

    const replayId = inject(ActivatedRoute).snapshot.queryParamMap.get('replay');
    if (replayId) void this.loadGalleryEntry(replayId);
  }

  // ------------------------------------------------------------------ curated deck

  private async loadDeck(): Promise<void> {
    this.deck.set(await this.api.getDeck());
  }

  /** Load a saved level onto the board for play / solve; keeps its id so a re-save updates it. */
  protected loadLevel(level: DeckLevel): void {
    this.stopPlayback();
    this.vehicles.set(level.vehicles);
    this.currentLevelId.set(level.id);
    this.levelName.set(level.name);
    this.solution.set(null);
    this.deckMessage.set(null);
    this.mode.set('edit');
    void this.refreshAnalysis();
  }

  protected async saveToDeck(): Promise<void> {
    const result = await this.api.saveLevel(this.levelName(), this.vehicles(), this.currentLevelId() ?? undefined);
    switch (result.kind) {
      case 'saved':
        this.currentLevelId.set(result.level.id);
        this.deckMessage.set(`Saved “${result.level.name}” (optimal ${result.level.optimalMoves} moves).`);
        await this.loadDeck();
        break;
      case 'error':
        this.deckMessage.set(result.error);
        break;
      case 'unavailable':
        this.deckMessage.set('Saving is only available in development.');
        break;
    }
  }

  protected async deleteLevel(level: DeckLevel, event: Event): Promise<void> {
    event.stopPropagation();
    if (!await this.api.deleteLevel(level.id)) return;
    if (this.currentLevelId() === level.id) this.currentLevelId.set(null);
    await this.loadDeck();
  }

  private async loadGalleryEntry(id: string): Promise<void> {
    const response = await fetch(`/api/gallery/${id}`);
    if (!response.ok) return;
    const entry = await response.json();
    if (entry.game !== 'rushhour') return;
    this.vehicles.set(entry.request.vehicles);
    void this.refreshAnalysis();
    this.solution.set(entry.response);
    this.showOptimal.set(false);
    this.playbackIndex.set(0);
    this.mode.set('playback');
  }

  // ------------------------------------------------------------------ editing

  protected selectTool(tool: Tool): void {
    this.tool.set(tool);
    this.editMessage.set(null);
  }

  /**
   * Board coordinates in FRACTIONAL cells. Derived from the element's measured rect rather than the logical
   * constants, so it stays correct once the canvas is CSS-scaled — which it now is, on every viewport narrower
   * than 502px.
   */
  private locate(event: PointerEvent): { row: number; col: number; fx: number; fy: number } | null {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0) return null;

    const scale = rect.width / LOGICAL_W;
    const fx = ((event.clientX - rect.left) / scale - PAD) / CELL;
    const fy = ((event.clientY - rect.top) / scale - PAD) / CELL;
    return { row: Math.floor(fy), col: Math.floor(fx), fx, fy };
  }

  /**
   * Edit mode commits on the DOWN event (placing is a discrete act with nothing to preview); play mode begins a
   * drag. The two are separated by MODE, never by gesture — a "drag" on an empty cell in edit mode has no
   * meaningful reading, and the mode is already visible in the panel and the caption.
   */
  protected onPointerDown(event: PointerEvent): void {
    const at = this.locate(event);
    if (!at || at.row < 0 || at.row >= SIZE || at.col < 0 || at.col >= SIZE) return;

    if (this.mode() === 'edit') {
      this.editCell(at.row, at.col);
      return;
    }
    if (this.mode() !== 'play') return;   // playback owns the positions; a drag would fight the timer

    const vehicles = this.vehicles();
    const positions = this.playPositions();
    const index = occupancy(vehicles, positions)[at.row * SIZE + at.col];
    if (index < 0) {
      this.selected.set(null);
      return;
    }
    this.selected.set(index);
    if (this.playWon()) return;

    // A settle still in flight would have us read a fractional position as the start. Finish it first.
    this.finishSettle();

    const vehicle = vehicles[index];
    const { lo, hi } = clampRange(vehicles, positions, index);
    const axis = vehicle.horizontal ? at.fx : at.fy;
    this.drag = {
      pointerId: event.pointerId,
      vehicle: index,
      horizontal: vehicle.horizontal,
      startPos: positions[index],
      offset: axis - positions[index],
      lo,
      hi,
      posF: positions[index],
      moved: false,
    };
    this.canvasRef()?.nativeElement.setPointerCapture(event.pointerId);
  }

  protected onPointerMove(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const at = this.locate(event);
    if (!at) return;

    const axis = drag.horizontal ? at.fx : at.fy;
    let raw = axis - drag.offset;

    // Re-anchor while pinned against a blocker. Without this the overshoot is stored as phantom distance, and
    // dragging back moves nothing until that debt is repaid — which is what makes a hand-rolled drag feel dead.
    if (raw > drag.hi) {
      drag.offset = axis - drag.hi;
      raw = drag.hi;
    } else if (raw < drag.lo) {
      drag.offset = axis - drag.lo;
      raw = drag.lo;
    }

    if (Math.abs(raw - drag.posF) < 1e-4) return;
    if (Math.abs(raw - drag.startPos) >= DRAG_MIN) drag.moved = true;
    drag.posF = raw;
    this.kick();
  }

  protected onPointerUp(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    this.drag = null;
    this.canvasRef()?.nativeElement.releasePointerCapture?.(event.pointerId);

    if (!drag.moved) {
      this.kick();          // a tap: the selection is already set, nothing moved, nothing counted
      return;
    }
    const target = Math.max(drag.lo, Math.min(drag.hi, Math.round(drag.posF)));
    this.commitDrag(drag, target);
  }

  /** A revoked gesture is a cancelled move: the vehicle goes back where it was grabbed, and nothing is counted. */
  protected onPointerCancel(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    this.drag = null;
    this.commitDrag(drag, drag.startPos);
  }

  private commitDrag(drag: Drag, target: number): void {
    const vehicles = this.vehicles();
    const positions = [...this.playPositions()];
    positions[drag.vehicle] = target;
    this.playPositions.set(positions);

    // ONE MOVE PER CELL. RushHourSolver's BFS enumerates single-cell edges, and its depth is what the page shows
    // as "optimal". Counting a three-cell drag as one move would let a player beat a figure that is by
    // construction unbeatable, turning the comparison into a bug.
    const travelled = Math.abs(target - drag.startPos);
    if (travelled > 0) {
      this.movesUsed.update(m => m + travelled);
      if (isSolved(vehicles, positions)) this.playWon.set(true);
    }

    if (Math.abs(drag.posF - target) > 1e-3 && !this.reducedMotion()) {
      this.settle = { vehicle: drag.vehicle, from: drag.posF, to: target, startedAt: performance.now() };
    }
    this.kick();
  }

  private finishSettle(): void {
    this.settle = null;
  }

  private reducedMotion(): boolean {
    return matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /** Drives repaints while a drag or settle is in flight, then parks. */
  private kick(): void {
    if (this.frame) return;
    const step = () => {
      this.frame = 0;
      this.draw();
      if (this.drag || this.settle) this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  }

  private editCell(row: number, col: number): void {
    const vehicles = this.vehicles();
    this.editMessage.set(null);

    switch (this.tool()) {
      case 'red':
      case 'red-truck': {
        if (row !== EXIT_ROW) {
          this.editMessage.set('The red vehicle lives on row 3 — the exit row.');
          return;
        }
        const length = this.tool() === 'red-truck' ? 3 : 2;
        const red: VehicleDto = { row: EXIT_ROW, col: Math.min(col, SIZE - length), length, horizontal: true };
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
    this.currentLevelId.set(null); // a fresh board is a new level, not an edit of the loaded one
    this.levelName.set('');
    this.deckMessage.set(null);
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
    // Select the RED car by default. The arrow keys act on the selection, so starting with nothing selected made
    // the keyboard silently dead until the player happened to click a vehicle — and the red car is the one the
    // puzzle is about. Clicking or dragging any other vehicle still re-selects as before.
    this.selected.set(this.vehicles().length > 0 ? 0 : null);
  }

  protected backToEdit(): void {
    this.stopPlayback();
    this.mode.set('edit');
    this.selected.set(null);
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
    if (isTypingTarget(event.target)) return;
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
        case 'loading':
          this.modelStatus.set(result.status);
          this.editMessage.set('The model is still loading — try again in a moment.');
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
    pollModelStatus(() => this.api.status(), (s) => this.modelStatus.set(s));
  }

  // ------------------------------------------------------------------ canvas

  private draw(): void {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return;

    const positions = this.displayPositions();
    const vehicles = this.vehicles();
    const width = LOGICAL_W;
    const height = LOGICAL_H;

    // Sized from the element, not from the logical constants: the canvas is CSS-scaled so the board fits a
    // phone. Reassigning width/height CLEARS and reallocates the backing store, so it is guarded — during a drag
    // this runs on every pointer move, and an unguarded reallocation visibly stutters on low-end devices.
    const cssWidth = canvas.clientWidth || width;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const backingW = Math.round(cssWidth * dpr);
    const backingH = Math.round(cssWidth * (height / width) * dpr);
    if (canvas.width !== backingW || canvas.height !== backingH) {
      canvas.width = backingW;
      canvas.height = backingH;
    }

    const ctx = canvas.getContext('2d')!;
    const scale = backingW / width;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);

    // A vehicle mid-drag or mid-settle sits at a FRACTIONAL position; everything else is on its cell.
    let animated: { vehicle: number; pos: number } | null = null;
    if (this.drag) {
      animated = { vehicle: this.drag.vehicle, pos: this.drag.posF };
    } else if (this.settle) {
      const t = Math.min(1, (performance.now() - this.settle.startedAt) / SETTLE_MS);
      animated = { vehicle: this.settle.vehicle, pos: this.settle.from + (this.settle.to - this.settle.from) * t };
      if (t >= 1) this.settle = null;     // linear, no overshoot — the same policy as Lunar Lockout's slide
    }

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
      const pos = animated?.vehicle === i ? animated.pos : positions[i];
      const row = v.horizontal ? v.row : pos;
      const col = v.horizontal ? pos : v.col;
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
