import { Component, DestroyRef, ElementRef, afterNextRender, computed, inject, signal, viewChild } from '@angular/core';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { BlockDudeRenderer, BlockDudeSnapshot, FallingBlock } from './block-dude-render';
import { isTypingTarget } from '../keyboard-target';
import { PgBlockDudeBoard } from './blockdude_solver';
import { BlockDudeApi } from './block-dude-api';

interface BlockDudeLevel {
  name: string;
  grid: string[];
}

/** Actions, matching the engine's encoding: 0 left, 1 right, 2 climb, 3 grab. */
const LEFT = 0;
const RIGHT = 1;
const CLIMB = 2;
const GRAB = 3;

/**
 * Block Dude page. Rules come from the single-source `.pg`, so the board here behaves exactly as the training
 * environment does — including the quirks: you fall THROUGH a door but a block rests ON it, a climb never
 * triggers gravity, and only a block can be picked up, never stone.
 *
 * Gravity is never settled on load, which is why level 11 can ship with blocks floating in mid-air.
 */
@Component({
  selector: 'app-block-dude',
  templateUrl: './block-dude.html',
  styleUrl: './block-dude.scss',
  imports: [BsButtonTypeDirective],
  host: { '(document:keydown)': 'onKeyDown($event)' },
})
export class BlockDude {
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('bdCanvas');

  protected readonly colors = Color;
  protected readonly levels = signal<BlockDudeLevel[]>([]);
  protected readonly levelIndex = signal(0);
  protected readonly moves = signal(0);
  protected readonly won = signal(false);
  protected readonly status = signal('Loading levels…');
  protected readonly carrying = signal(false);

  /**
   * Every solution recorded so far, rendered as pasteable lines. Set on a win so the player can hand the whole
   * set over in one copy, rather than anyone having to dig it out of localStorage.
   */
  protected readonly solution = signal<string | null>(null);
  protected readonly copyLabel = signal('Copy');

  /**
   * The move list for the CURRENT attempt, live. Derived from <see cref="played"/> rather than accumulated
   * separately so it cannot drift from what actually happened: undo drops the last move from the list because
   * it drops it from the recording, and restart empties both. One source of truth, not two kept in step.
   */
  protected readonly attempt = computed(() => {
    const moves = this.played();
    const level = this.level();
    return moves.length === 0 || !level
      ? null
      : `${level.name} · ${moves.length} move${moves.length === 1 ? '' : 's'} · ${moves.join('')}`;
  });

  protected readonly level = computed(() => this.levels()[this.levelIndex()]);
  protected readonly canUndo = computed(() => this.history().length > 0);
  protected readonly aspect = computed(() => {
    const level = this.level();
    return level ? `${level.grid[0].length} / ${level.grid.length}` : '20 / 8';
  });

  private readonly history = signal<PgBlockDudeBoard[]>([]);

  /**
   * Actions taken since the level was (re)started — undo pops it, so it always mirrors what actually happened.
   * On a win the whole trajectory is saved to localStorage under SOLUTIONS_KEY.
   *
   * A human solution to a shipped level is training data this project cannot get any other way: the exact BFS
   * oracle cannot label boards this size (that is why the curriculum stops short of them), so these are the only
   * ground-truth solutions to the real content that exist. The ACTION SEQUENCE is kept, not just the final
   * board, because the sequence is strictly more: it yields a per-state action label, the final block
   * configuration, AND a true remaining-distance label for every state along the way — in the 1-to-several-hundred
   * range where the value head was measured to be badly compressed and has no training data at all.
   */
  protected readonly played = signal<number[]>([]);
  private renderer: BlockDudeRenderer | null = null;
  private board: PgBlockDudeBoard | null = null;

  // ── Watch the AI ────────────────────────────────────────────────────────────────────────────────────────
  // The net plays server-side (one POST per level) rather than in the browser: the checkpoint is 16 MB, which
  // is a lot to push at every visitor of a page that plays perfectly well without it.

  private readonly api = inject(BlockDudeApi);

  protected readonly aiPlaying = signal(false);
  protected readonly aiThinking = signal(false);

  /** Playback speed multiplier, cycled by the button. Level 11 is 826 moves — at 1× that is a minute and a
   * half of watching, so the faster steps are what make the long levels actually watchable. */
  protected readonly aiSpeed = signal(1);

  /** Matches the renderer's own WALK_MS, so at 1× a step lands as its animation finishes. */
  private static readonly STEP_MS = 110;

  private aiTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    afterNextRender(() => {
      this.renderer = new BlockDudeRenderer(this.canvasRef().nativeElement);
      void this.load();
    });
    inject(DestroyRef).onDestroy(() => {
      this.stopAi();               // a pending timer would keep firing into a destroyed component
      this.renderer?.dispose();
    });
  }

  /** The SAME JSON the training campaign embeds — one source of truth, no drift. */
  private async load(): Promise<void> {
    try {
      const response = await fetch('levels/blockdude-levels.json');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const document = await response.json() as { levels: BlockDudeLevel[] };
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
    // Not stopAi(): watchAi() calls reset() to put the net on the opening position, and clearing the flags here
    // would cancel the playback it is about to start. Only the pending STEP is dropped.
    if (this.aiTimer !== null) { clearTimeout(this.aiTimer); this.aiTimer = null; }
    this.board = PgBlockDudeBoard.fromGrid(level.grid);
    this.history.set([]);
    this.played.set([]);
    this.solution.set(null);   // restart clears the list; the saved recording in localStorage is untouched
    this.moves.set(0);
    this.won.set(false);
    this.carrying.set(false);
    this.status.set(`${level.name} — reach the door.`);
    this.renderer?.reset(this.snapshot());
  }

  protected changeLevel(delta: number): void {
    const next = this.levelIndex() + delta;
    if (next < 0 || next >= this.levels().length) return;
    this.stopAi();
    this.levelIndex.set(next);
    this.reset();
  }

  /** Where recorded human solutions accumulate. Read them with `localStorage.getItem(...)`. */
  private static readonly SOLUTIONS_KEY = 'blockdude.solutions.v1';

  /**
   * Appends the finished trajectory. Keyed by level NAME rather than index so a future change to the pack's
   * order cannot silently re-label old recordings, and the shortest solution per level wins — replaying a level
   * better should improve the record, not append a worse duplicate.
   */
  private recordSolution(): void {
    const level = this.level();
    if (!level) return;

    try {
      const raw = localStorage.getItem(BlockDude.SOLUTIONS_KEY);
      const all = raw ? JSON.parse(raw) as Record<string, { grid: string[]; moves: number[] }> : {};

      const existing = all[level.name];
      if (!existing || this.played().length < existing.moves.length)
        all[level.name] = { grid: level.grid, moves: [...this.played()] };

      localStorage.setItem(BlockDude.SOLUTIONS_KEY, JSON.stringify(all));
      this.solution.set(BlockDude.render(all));
    } catch {
      // A full or blocked localStorage must never break the game — the recording is a side benefit, not the
      // point. Still show THIS solution, which is the one the player just earned and would be annoyed to lose.
      const level = this.level();
      if (level) this.solution.set(BlockDude.render({ [level.name]: { grid: level.grid, moves: this.played() } }));
    }
  }

  /**
   * One line per level: name, move count, then the actions as digits — the same 0..3 encoding the engine and
   * the Lab's tests already use, so a pasted line needs no translation to be replayed.
   */
  private static render(all: Record<string, { grid: string[]; moves: number[] }>): string {
    return Object.entries(all)
      .map(([name, s]) => `${name} · ${s.moves.length} moves · ${s.moves.join('')}`)
      .join('\n');
  }

  protected async copySolution(text: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text);
      this.copyLabel.set('Copied');
    } catch {
      // Clipboard access is permission-gated and fails outright over plain http on some browsers. The text is
      // in a focusable, select-on-focus textarea precisely so this is a convenience, never the only way out.
      this.copyLabel.set('Select it and copy');
    }
    setTimeout(() => this.copyLabel.set('Copy'), 2500);
  }

  protected undo(): void {
    const stack = this.history();
    if (stack.length === 0) return;
    this.board = stack[stack.length - 1];
    this.history.set(stack.slice(0, -1));
    this.played.update(p => p.slice(0, -1));   // keep the recording honest: an undone move was never played
    this.moves.update(n => Math.max(0, n - 1));
    this.won.set(false);
    this.carrying.set(this.board.carrying);
    this.renderer?.reset(this.snapshot());
  }

  /**
   * Bound at DOCUMENT level so the board is playable the moment the page loads, with no click to focus it
   * first. `preventDefault` fires only for keys this page acts on, and never while the caret is in a text
   * field — otherwise arrow keys would be stolen from any input on the page.
   */
  protected onKeyDown(event: KeyboardEvent): void {
    if (isTypingTarget(event.target)) return;

    // Any deliberate input takes the board back. Silently ignoring keys while the AI plays would read as the
    // page being broken, and stopping is what someone reaching for the arrows actually wants.
    const wasAi = this.aiPlaying() || this.aiThinking();
    if (wasAi) {
      this.stopAi();
      if (event.key === 'Escape') {
        event.preventDefault();
        this.status.set('Stopped the AI — the board is yours.');
        return;
      }
    }

    const map: Record<string, number | undefined> = {
      ArrowLeft: LEFT,
      ArrowRight: RIGHT,
      ArrowUp: CLIMB,
      ArrowDown: GRAB,
      ' ': GRAB,
    };
    const action = map[event.key];
    if (action !== undefined) {
      event.preventDefault();
      this.act(action);   // act() sets its own status; the Stop button vanishing is the signal the AI ended
      return;
    }
    if (event.key === 'r' || event.key === 'R') this.reset();
    if (event.key === 'z' || event.key === 'Z') this.undo();
    else if (wasAi) this.status.set('Stopped the AI — the board is yours.');
  }

  /**
   * Tap the board left or right of the dude to step that way; tapping his own column climbs. Mirrors the
   * keyboard rather than inventing a second vocabulary, and needs no gesture literacy.
   */
  protected onPointerDown(event: PointerEvent): void {
    const board = this.board;
    if (!board || this.won()) return;
    if (this.aiPlaying() || this.aiThinking()) {
      this.stopAi();
      this.status.set('Stopped the AI — the board is yours.');
      return;                    // this tap stops it; the next one moves
    }
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    if (rect.width === 0) return;

    const cell = rect.width / board.width;
    const col = Math.floor((event.clientX - rect.left) / cell);
    if (col < board.px) this.act(LEFT);
    else if (col > board.px) this.act(RIGHT);
    else this.act(CLIMB);
  }

  protected act(action: number): void {
    const board = this.board;
    if (!board || this.won()) return;

    const before = board;
    const next = board.applyAction(action);

    // Nothing changed: an illegal move. Say why rather than failing silently, and do NOT record it — a refused
    // keypress is not a move, so it stays out of the move count, the undo history and the recorded solution.
    //
    // Asking the ENGINE rather than comparing fields here. The previous hand-rolled test compared block COUNT
    // rather than block positions, which is only safe as long as nothing can relocate a block without also
    // changing `carrying`. That happens to hold today, but it is the engine's rule to know, not this page's —
    // and `sameState` is the same predicate the training harness uses to detect a refused move, so the recording
    // now agrees with the trainer by construction.
    //
    // Note a turn IS a change: facing right when you were facing left counts as a move even if the way ahead is
    // blocked, faithful to the original. Pressing into a wall you already face does not.
    if (next.sameState(before)) {
      this.status.set(this.refusal(action));
      return;
    }

    // A block that left the player's hands falls from just above his head to wherever it settled.
    let fall: FallingBlock | undefined;
    if (before.carrying && !next.carrying) {
      const landed = next.blocks.find(c => !before.blocks.includes(c));
      if (landed !== undefined) {
        fall = {
          fromX: before.px + (before.facingRight ? 1 : -1),
          fromY: before.py - 1,
          toX: landed % next.width,
          toY: Math.floor(landed / next.width),
        };
      }
    }

    this.board = next;
    this.history.update(stack => [...stack, before]);
    this.moves.update(n => n + 1);
    this.carrying.set(next.carrying);

    const climbed = action === CLIMB;
    const moved = next.px !== before.px || next.py !== before.py;
    this.renderer?.push(
      this.snapshot(),
      moved ? { fromX: before.px, fromY: before.py, climb: climbed } : undefined,
      fall);

    // The AI's moves are deliberately kept OUT of the recording. `played` and the localStorage archive are the
    // human's solutions — the ground truth the net was trained on — and letting an autoplay overwrite a
    // hand-played line with the net's own would quietly corrupt the very data this game exists to collect.
    if (!this.aiPlaying()) this.played.update(p => [...p, action]);

    if (next.won) {
      this.won.set(true);
      if (this.aiPlaying()) {
        this.stopAi();
        this.status.set(`The AI solved ${this.level()?.name ?? 'it'} in ${this.moves()} moves.`);
        return;
      }
      this.recordSolution();
      this.status.set(`Level complete in ${this.moves()} move${this.moves() === 1 ? '' : 's'} — solution recorded.`);
    } else if (next.carrying !== before.carrying) {
      this.status.set(next.carrying ? 'Carrying a block.' : 'Block placed.');
    } else {
      this.status.set(`${this.level()?.name ?? ''} — reach the door.`);
    }
  }

  /**
   * Asks the net to play the current level, then replays its moves.
   *
   * Restarts the level first: the net is given the opening position, not whatever the player has already done
   * to the board — a half-played board is a different puzzle, and on an irreversible one it may be a lost one.
   */
  protected async watchAi(): Promise<void> {
    if (this.aiPlaying() || this.aiThinking()) { this.stopAi(); return; }

    const level = this.level();
    if (!level) return;

    this.reset();
    this.aiThinking.set(true);
    this.status.set('Asking the AI…');

    const result = await this.api.solve(level.grid);
    this.aiThinking.set(false);

    // The level may have been changed, restarted or played while the request was in flight.
    if (this.level() !== level || this.moves() > 0) return;

    if (result.kind === 'loading') { this.status.set('The AI is still loading — try again in a moment.'); return; }
    if (result.kind === 'failed') { this.status.set(`Could not reach the AI (${result.message}).`); return; }
    if (!result.value.solved) { this.status.set('The AI could not solve this level.'); return; }

    const { moves, tier } = result.value;
    this.status.set(tier === 'greedy'
      ? `The AI plays ${moves.length} moves — no search, one look per move.`
      : `The AI found ${moves.length} moves using search.`);

    this.aiPlaying.set(true);
    this.playAi(moves, 0);
  }

  /** One scheduled step per move. A timer chain rather than an interval, so a slow frame delays the next step
   * instead of letting steps pile up faster than the renderer can animate them. */
  private playAi(moves: number[], index: number): void {
    if (!this.aiPlaying() || index >= moves.length) { this.stopAi(); return; }

    const before = this.moves();
    this.act(moves[index]);
    if (!this.aiPlaying()) return;      // act() stops on the winning move

    // The engine refused a move the server said it played. That should be impossible — both sides run the same
    // single-source rules and the server only returns lines it walked to the door — so if it ever happens it is
    // a twin divergence, and continuing would replay a silently different game. Stop and say so.
    if (this.moves() === before) {
      this.stopAi();
      this.status.set(`The AI's move ${index + 1} was refused by the engine — stopping (the C# and browser rules disagree).`);
      return;
    }

    this.aiTimer = setTimeout(() => this.playAi(moves, index + 1),
                              BlockDude.STEP_MS / this.aiSpeed());
  }

  protected stopAi(): void {
    if (this.aiTimer !== null) { clearTimeout(this.aiTimer); this.aiTimer = null; }
    this.aiPlaying.set(false);
    this.aiThinking.set(false);
  }

  /** 1× → 2× → 4× → 1×. Takes effect on the next step, so it can be changed mid-playback. */
  protected cycleAiSpeed(): void {
    this.aiSpeed.update(s => (s >= 4 ? 1 : s * 2));
  }

  private refusal(action: number): string {
    switch (action) {
      case CLIMB: return 'Nothing to climb there — or no headroom above.';
      case GRAB: return this.board?.carrying
        ? 'No room to put the block down that way.'
        : 'Only a block can be picked up, and only with space above it.';
      default: return 'Blocked that way.';
    }
  }

  private snapshot(): BlockDudeSnapshot {
    const board = this.board!;
    return {
      width: board.width,
      height: board.height,
      tiles: board.tiles,
      blocks: board.blocks,
      px: board.px,
      py: board.py,
      facingRight: board.facingRight,
      carrying: board.carrying,
      won: board.won,
      moves: this.moves(),
    };
  }
}
