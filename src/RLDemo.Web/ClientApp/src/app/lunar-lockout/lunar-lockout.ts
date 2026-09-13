import { Component, DestroyRef, ElementRef, afterNextRender, computed, inject, signal, viewChild } from '@angular/core';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { LUNAR_SIZE, LunarLockoutRenderer, LunarSnapshot } from './lunar-lockout-render';
import { isTypingTarget } from '../keyboard-target';
import { PgLunarBoard, PgLunarOracle } from './lunarlockout_solver';

interface LunarLevel {
  name: string;
  grid: string[];
  optimalMoves: number;
}

/** Where a pointer grabbed a rocket, so a gesture can be classified on release. */
interface Grab {
  pointerId: number;
  robot: number;
  startX: number;
  startY: number;
  startMs: number;
  touch: boolean;
}

// Aim geometry, in CELL UNITS from the cell centre. Both boundaries are sticky toward "keep what you have", so a
// pointer parked on a boundary never oscillates between two directions (PRD §12.3).
const HYST_IN = 0.16;        // radius to acquire a direction (mouse)
const HYST_OUT = 0.11;       // radius to fall back to rest (mouse)
const TOUCH_ARM = 0.22;      // fat-finger equivalents
const TOUCH_RELEASE = 0.15;
const ANGLE_IN = (40 * Math.PI) / 180;   // wedge half-width when acquiring
const ANGLE_OUT = (50 * Math.PI) / 180;  // wedge half-width when holding
const LATCH_R = 0.75;        // past this the direction latches and the wedge test stops
const TAP_MAX_R = 0.20;      // below this on release it was a tap, not a drag
const TAP_MAX_MS = 250;

/** Angle from the cell centre to each direction, in the same frame as the aim test. 0=up,1=right,2=down,3=left. */
const DIRECTION_BEARING = [Math.PI / 2, 0, -Math.PI / 2, Math.PI];

/**
 * Lunar Lockout page. Entirely client-side: the rules AND the exact BFS oracle come from the single-source `.pg`,
 * so the hint here is the same search that verified every shipped level's move count during the build.
 *
 * Input is direct manipulation (PRD §12): aim a rocket by where you point at it, and click or release to launch.
 * One Pointer Events stream serves mouse, pen and touch — `pointerType` selects the model per interaction, so a
 * touchscreen laptop needs no mode switch. An ILLEGAL aim never rotates the rocket, so it cannot look launchable
 * in a direction it has no blocker for.
 */
@Component({
  selector: 'app-lunar-lockout',
  templateUrl: './lunar-lockout.html',
  styleUrl: './lunar-lockout.scss',
  imports: [BsButtonTypeDirective],
  host: { '(document:keydown)': 'onKeyDown($event)' },
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

  /** Selection made by KEYBOARD or by a tap. Distinct from the pointer's transient aim. */
  protected readonly keyboardSelected = signal(-1);

  protected readonly level = computed(() => this.levels()[this.levelIndex()]);
  protected readonly canUndo = computed(() => this.history().length > 0);
  private readonly history = signal<PgLunarBoard[]>([]);

  /** Which directions the selected robot can actually fly, so the pad disables the rest instead of refusing them. */
  protected readonly legalDirections = computed<boolean[]>(() => {
    this.boardVersion();
    const robot = this.keyboardSelected();
    if (!this.board || robot < 0 || this.solved()) return [false, false, false, false];
    return [0, 1, 2, 3].map(d => this.board!.landing(robot, d) >= 0);
  });

  /** Bumped on every board mutation so `legalDirections` recomputes — the board itself is not a signal. */
  private readonly boardVersion = signal(0);

  private readonly hoverCapable = matchMedia('(hover: hover) and (pointer: fine)');
  private renderer: LunarLockoutRenderer | null = null;
  private board: PgLunarBoard | null = null;
  private grab: Grab | null = null;
  private aimRobot = -1;
  private aimDirection = -1;

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
    // Arm the TARGET rocket by default. Arrow keys act on the armed rocket, so leaving nothing armed made the
    // keyboard silently dead on arrival — and the red one is the rocket a player moves most anyway. Pressing a
    // number still switches, and the pointer path ignores this entirely.
    this.keyboardSelected.set(0);
    this.clearAim();
    this.hint.set(null);
    this.renderer?.resetAngles();
    this.status.set(`${level.name} — solvable in ${level.optimalMoves} move${level.optimalMoves === 1 ? '' : 's'}.`);
    this.bump();
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
    this.bump();
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

  // ── pointer ───────────────────────────────────────────────────────────────────────────────────────────────

  protected onPointerDown(event: PointerEvent): void {
    if (!this.board || this.solved()) return;
    const at = this.locate(event);
    if (!at || at.robot < 0) {
      this.keyboardSelected.set(-1);
      this.clearAim();
      return;
    }

    this.canvasRef().nativeElement.setPointerCapture(event.pointerId);
    this.grab = {
      pointerId: event.pointerId,
      robot: at.robot,
      startX: event.clientX,
      startY: event.clientY,
      startMs: event.timeStamp,
      touch: event.pointerType === 'touch' || event.pointerType === 'pen',
    };
    // Resolve from the DOWN point rather than trusting any prior hover, so a first click into an unfocused
    // window, a stylus, and a touch all aim correctly without ever having emitted a move.
    this.resolveAim(at.robot, at.u, at.v, this.grab.touch);
  }

  protected onPointerMove(event: PointerEvent): void {
    if (!this.board || this.solved()) return;

    if (this.grab) {
      const at = this.locate(event, this.grab.robot);
      if (at) this.resolveAim(this.grab.robot, at.u, at.v, this.grab.touch);
      return;
    }

    // Hover is mouse/pen only. A touch pointer that reaches here (some stacks emit a move before down) must not
    // leave a phantom aim behind.
    if (event.pointerType === 'touch' || !this.hoverCapable.matches) return;
    const at = this.locate(event);
    if (!at || at.robot < 0) {
      this.clearAim();
      return;
    }
    this.resolveAim(at.robot, at.u, at.v, false);
  }

  protected onPointerUp(event: PointerEvent): void {
    const grab = this.grab;
    if (!grab || grab.pointerId !== event.pointerId) return;
    this.grab = null;
    this.canvasRef().nativeElement.releasePointerCapture?.(event.pointerId);

    const travelled = Math.hypot(event.clientX - grab.startX, event.clientY - grab.startY) / this.cellCss();
    const elapsed = event.timeStamp - grab.startMs;
    const direction = this.aimRobot === grab.robot ? this.aimDirection : -1;

    if (direction >= 0 && this.board && this.board.landing(grab.robot, direction) >= 0) {
      this.launch(grab.robot, direction);
      return;
    }

    // A tap selects, so the keyboard and the pad remain available to anyone who taps rather than drags.
    if (travelled < TAP_MAX_R && elapsed < TAP_MAX_MS) {
      this.keyboardSelected.set(grab.robot);
      this.status.set('Aim by pointing at a side of the rocket, or use the arrow keys.');
    } else if (direction >= 0) {
      // Aimed somewhere it cannot fly: refuse, explain, and do not touch the history.
      this.status.set('That rocket has nothing to stop it that way.');
      this.renderer?.refuse(grab.robot, direction);
    }
    this.clearAim();
  }

  protected onPointerCancel(event: PointerEvent): void {
    if (this.grab?.pointerId === event.pointerId) this.grab = null;
    this.clearAim();
  }

  protected onPointerLeave(): void {
    if (this.grab) return;   // capture keeps an active drag alive outside the canvas
    this.clearAim();
  }

  /** Cell-local coordinates, in cell units from the centre of whichever cell the pointer is over. */
  private locate(event: PointerEvent, forceRobot?: number): { robot: number; u: number; v: number } | null {
    const board = this.board;
    if (!board) return null;
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    if (rect.width === 0) return null;

    const cell = rect.width / LUNAR_SIZE;
    const fx = (event.clientX - rect.left) / cell;
    const fy = (event.clientY - rect.top) / cell;

    if (forceRobot !== undefined) {
      const at = board.robots[forceRobot];
      return { robot: forceRobot, u: fx - (at % LUNAR_SIZE) - 0.5, v: fy - Math.floor(at / LUNAR_SIZE) - 0.5 };
    }

    const col = Math.floor(fx);
    const row = Math.floor(fy);
    if (col < 0 || col >= LUNAR_SIZE || row < 0 || row >= LUNAR_SIZE) return null;
    const robot = board.robots.indexOf(row * LUNAR_SIZE + col);
    return { robot, u: fx - col - 0.5, v: fy - row - 0.5 };
  }

  private cellCss(): number {
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    return (rect.width || LUNAR_SIZE) / LUNAR_SIZE;
  }

  /**
   * Two-axis hysteresis: a direction is acquired at a larger radius and wider wedge than it is released at, so a
   * pointer resting on a boundary keeps whatever it already had.
   */
  private resolveAim(robot: number, u: number, v: number, touch: boolean): void {
    const board = this.board;
    if (!board) return;

    const holding = this.aimRobot === robot && this.aimDirection >= 0;
    const radius = Math.hypot(u, v);
    const inner = touch ? (holding ? TOUCH_RELEASE : TOUCH_ARM) : (holding ? HYST_OUT : HYST_IN);

    let direction = -1;
    if (radius >= LATCH_R && holding) {
      direction = this.aimDirection;            // latched: a long drag keeps its direction
    } else if (radius >= inner) {
      const bearing = Math.atan2(-v, u);
      let best = -1;
      let bestDelta = Infinity;
      for (let d = 0; d < 4; d++) {
        const delta = Math.abs(normalise(bearing - DIRECTION_BEARING[d]));
        const halfWidth = holding && d === this.aimDirection ? ANGLE_OUT : ANGLE_IN;
        if (delta <= halfWidth && delta < bestDelta) {
          best = d;
          bestDelta = delta;
        }
      }
      direction = best >= 0 ? best : (holding ? this.aimDirection : -1);
    }

    const landing = direction >= 0 ? board.landing(robot, direction) : -1;
    this.aimRobot = robot;
    this.aimDirection = direction;
    this.renderer?.hover({ robot, direction, legal: landing >= 0, landing });

    const canvas = this.canvasRef().nativeElement;
    canvas.style.cursor = landing >= 0 ? 'pointer' : 'default';
  }

  private clearAim(): void {
    this.aimRobot = -1;
    this.aimDirection = -1;
    this.renderer?.hover(null);
    const canvas = this.canvasRef().nativeElement;
    if (canvas) canvas.style.cursor = 'default';
  }

  // ── keyboard ──────────────────────────────────────────────────────────────────────────────────────────────
  // Bound at DOCUMENT level so the board is playable the moment the page loads, with no click to focus it
  // first. `preventDefault` is called only for keys this page actually acts on, and never while the caret is in
  // a text field — otherwise arrow keys would be stolen from any input on the page.

  protected onKeyDown(event: KeyboardEvent): void {
    if (isTypingTarget(event.target)) return;

    const directions: Record<string, number> = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3 };
    if (event.key in directions) {
      event.preventDefault();
      this.slideSelected(directions[event.key]);
      return;
    }
    if (event.key === 'r' || event.key === 'R') this.reset();
    if (event.key === 'z' || event.key === 'Z') this.undo();
    if (event.key === 'Escape') this.keyboardSelected.set(-1);

    const digit = Number.parseInt(event.key, 10);
    if (!Number.isNaN(digit) && this.board && digit >= 1 && digit <= this.board.robots.length) {
      this.keyboardSelected.set(digit - 1);
      this.bump();
    }
  }

  /** The pad and the arrow keys act on the keyboard selection. */
  protected slideSelected(direction: number): void {
    const robot = this.keyboardSelected();
    if (robot < 0) {
      this.status.set('Pick a rocket first — tap it, or press its number.');
      return;
    }
    if (!this.board || this.solved()) return;
    if (this.board.landing(robot, direction) < 0) {
      this.status.set('That rocket has nothing to stop it that way.');
      this.renderer?.refuse(robot, direction);
      return;
    }
    this.launch(robot, direction);
  }

  private launch(robot: number, direction: number): void {
    const board = this.board;
    if (!board) return;

    const from = board.robots[robot];
    const previous = board;
    this.board = board.applyAction(robot * 4 + direction);
    this.history.update(stack => [...stack, previous]);
    this.moves.update(n => n + 1);
    this.hint.set(null);
    this.clearAim();

    if (this.board.isSolved()) {
      this.solved.set(true);
      const optimal = this.level()?.optimalMoves ?? 0;
      this.status.set(this.moves() === optimal
        ? `Solved in ${this.moves()} — that is optimal.`
        : `Solved in ${this.moves()}; the optimum is ${optimal}.`);
    }
    this.bump({ robot, from });
  }

  private bump(slide?: { robot: number; from: number }): void {
    this.boardVersion.update(n => n + 1);
    if (!this.renderer || !this.board) return;
    this.renderer.push(this.snapshot(), slide);
  }

  private snapshot(): LunarSnapshot {
    const board = this.board!;
    const selected = this.keyboardSelected();
    const keyboardHints: number[] = [];
    if (selected >= 0 && !this.solved()) {
      for (let direction = 0; direction < 4; direction++) {
        const landing = board.landing(selected, direction);
        if (landing >= 0) keyboardHints.push(landing);
      }
    }
    return {
      robots: [...board.robots],
      keyboardSelected: selected,
      keyboardHints,
      solved: this.solved(),
      moves: this.moves(),
    };
  }
}

function normalise(radians: number): number {
  let value = radians;
  while (value <= -Math.PI) value += Math.PI * 2;
  while (value > Math.PI) value -= Math.PI * 2;
  return value;
}

