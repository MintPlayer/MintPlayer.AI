import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { BsButtonGroupComponent } from '@mintplayer/ng-bootstrap/button-group';
import { BsSelectComponent, BsSelectOption } from '@mintplayer/ng-bootstrap/select';
import { BsRangeComponent } from '@mintplayer/ng-bootstrap/range';
import { DraughtsDirector } from './draughts-director';
import { DraughtsDifficulty } from './draughts-net';

interface Cell {
  sq: number;      // engine square index (rank*N + file)
  piece: number;   // 0 empty, +1/+2 White man/king, −1/−2 Black
  light: boolean;
  selected: boolean;
  target: boolean; // the FINAL square of a legal (possibly multi-jump) move for the selected piece
  last: boolean;   // part of the last move (from, to, or a captured square)
}

type Mode = 'play' | 'watch';

const BETWEEN_MS = 550;   // pause between moves in AI-vs-AI so it's watchable
const GAMEOVER_MS = 2200; // pause on a finished AI-vs-AI board before auto-restarting

/**
 * Draughts (checkers) — play the self-taught AI, or watch it play itself, entirely in your browser. The
 * engine, the convolutional policy/value network, and the PUCT search are single-sourced from
 * `draughts_solver.pg` and transpiled to TypeScript; the browser downloads a trained checkpoint and thinks
 * locally, with no server move computation (the chess M40 pattern, M47.5). Captures are forced and a
 * multi-jump is one move — click the final landing square and the whole sequence plays.
 */
@Component({
  selector: 'app-draughts',
  imports: [
    FormsModule,
    BsButtonGroupComponent, BsButtonTypeDirective,
    BsSelectComponent, BsSelectOption,
    BsRangeComponent,
  ],
  template: `
    <div class="wrap">
      <h1>Draughts</h1>
      <p class="intro">
        A network that learned checkers from scratch by playing itself (AlphaZero-style self-play), running
        <strong>entirely in your browser</strong> — no server. Captures are forced; a multi-jump plays as one
        move. Pick a difficulty, then play it or watch it play itself.
      </p>

      <div class="toolbar">
        <bs-button-group>
          <button type="button" [color]="mode() === 'play' ? colors.primary : colors.secondary" (click)="setMode('play')">Play the AI</button>
          <button type="button" [color]="mode() === 'watch' ? colors.primary : colors.secondary" (click)="setMode('watch')">Watch AI vs AI</button>
        </bs-button-group>

        @if (difficulties().length > 1) {
          <label class="diff">
            <span>Difficulty</span>
            <bs-select [ngModel]="current()" (ngModelChange)="onDifficultyChange($event)" [disabled]="busy()" [identifier]="1">
              @for (d of difficulties(); track d.label) {
                <option [ngValue]="d">{{ d.label }}</option>
              }
            </bs-select>
          </label>
        }
      </div>

      <div class="status" [class.busy]="busy()">{{ statusText() }}</div>

      <div class="board">
        @for (cell of cells(); track cell.sq) {
          <button
            class="sq"
            [class.light]="cell.light"
            [class.dark]="!cell.light"
            [class.sel]="cell.selected"
            [class.last]="cell.last"
            [disabled]="locked()"
            (click)="onClick(cell.sq)">
            @if (cell.piece !== 0) {
              <span class="pc" [class.white]="cell.piece > 0" [class.black]="cell.piece < 0">
                @if (isKing(cell.piece)) { <span class="crown">♛</span> }
              </span>
            }
            @if (cell.target) { <span class="dot"></span> }
          </button>
        }
      </div>

      <div class="controls">
        @if (mode() === 'play') {
          <button [color]="colors.primary" (click)="newGame()">New game</button>
          <span class="hint" [class.on]="over()">
            {{ over() ? 'Game over — start a new game.' : hint() }}
          </span>
        } @else {
          <label class="speed">
            <span>Speed</span>
            <bs-range [min]="0" [max]="1900" [step]="50"
              [ngModel]="1900 - betweenMs()" (ngModelChange)="betweenMs.set(1900 - $event)"></bs-range>
            <span class="speed-end">{{ betweenMs() >= 1200 ? 'slow' : betweenMs() <= 250 ? 'fast' : '' }}</span>
          </label>
        }

        @if (captured().white + captured().black > 0) {
          <div class="tray" title="Captured pieces">
            @if (captured().black > 0) { <span class="cap"><span class="pc black mini"></span>× {{ captured().black }}</span> }
            @if (captured().white > 0) { <span class="cap"><span class="pc white mini"></span>× {{ captured().white }}</span> }
          </div>
        }
      </div>
    </div>
  `,
  styles: `
    .wrap { max-width: 560px; }
    .intro { color: #aab2c5; }

    .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem; margin: 1rem 0 0.25rem; }
    .diff { display: flex; align-items: center; gap: 0.5rem; color: #aab2c5; font-size: 0.9rem; }

    .status { margin: 0.5rem 0 0.75rem; font-weight: 600; min-height: 1.5em; }
    .status.busy { color: #6ea8fe; }

    .board {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      grid-template-rows: repeat(8, 1fr);
      width: min(90vw, 512px);
      aspect-ratio: 1;
      border: 3px solid #2b3245;
      border-radius: 8px;
      overflow: hidden;
      user-select: none;
    }

    .sq {
      position: relative;
      border: 0; padding: 0; margin: 0;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
    }
    .sq.light { background: #b9c2d6; }
    .sq.dark  { background: #6b7488; }
    .sq:disabled { cursor: default; }
    .sq.last::after {
      content: ''; position: absolute; inset: 0;
      background: rgba(232, 145, 42, 0.38); pointer-events: none;
    }
    .sq.sel::after {
      content: ''; position: absolute; inset: 0;
      background: rgba(232, 145, 42, 0.6); pointer-events: none;
    }

    .pc {
      position: relative; z-index: 1;
      width: 74%; height: 74%;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.45), inset 0 -3px 0 rgba(0, 0, 0, 0.25);
    }
    .pc.white { background: #f7f9fc; }
    .pc.black { background: #1a1f2b; }
    .pc.mini { width: 1.1rem; height: 1.1rem; display: inline-flex; vertical-align: -0.2em; }
    .crown { font-size: clamp(0.9rem, 4vw, 1.5rem); line-height: 1; }
    .pc.white .crown { color: #b8860b; }
    .pc.black .crown { color: #e8c34a; }

    .dot {
      position: absolute; width: 28%; height: 28%;
      border-radius: 50%; background: rgba(232, 145, 42, 0.7); z-index: 1;
    }

    .controls { margin-top: 1rem; display: flex; align-items: center; gap: 1rem; }
    .hint { color: #8891a5; font-size: 0.9rem; }
    .hint.on { color: #e0b050; }
    .speed { display: flex; align-items: center; gap: 0.6rem; color: #8891a5; font-size: 0.9rem; }
    .speed-end { min-width: 2.5em; color: #e8912a; }

    .tray { display: flex; align-items: center; gap: 0.9rem; color: #aab2c5; font-size: 0.95rem; }
    .cap { display: inline-flex; align-items: center; gap: 0.3rem; }
  `,
})
export class Draughts {
  protected readonly colors = Color;
  protected readonly director = new DraughtsDirector();
  protected readonly mode = signal<Mode>('play');
  protected readonly betweenMs = signal(BETWEEN_MS);

  private readonly board = signal<number[]>(this.director.board());
  private readonly selected = signal<number | null>(null);
  private readonly targets = signal<Set<number>>(new Set());
  private readonly last = signal<{ from: number; to: number; captures: number[] } | null>(null);
  private readonly tick = signal(0);

  private loopGen = 0; // cancels a stale watch loop on mode/difficulty change / destroy

  constructor() {
    void this.director.ready.then(() => this.tick.update(v => v + 1));
    inject(DestroyRef).onDestroy(() => { this.loopGen++; });
  }

  protected isKing(piece: number): boolean { return Math.abs(piece) === 2; }

  protected readonly difficulties = computed<DraughtsDifficulty[]>(() => { this.tick(); return this.director.difficulties; });
  protected readonly current = computed<DraughtsDifficulty>(() => { this.tick(); return this.director.current; });

  protected readonly cells = computed<Cell[]>(() => {
    const b = this.board();
    const sel = this.selected();
    const tg = this.targets();
    const lm = this.last();
    const n = this.director.size;
    const out: Cell[] = [];
    for (let row = 0; row < n; row++) {           // display top→bottom = rank N−1→0 (White at the bottom)
      const rank = n - 1 - row;
      for (let file = 0; file < n; file++) {
        const sq = rank * n + file;
        out.push({
          sq,
          piece: b[sq],
          light: ((rank + file) & 1) === 1,        // play is on the dark ((f+r)-even) squares
          selected: sel === sq,
          target: tg.has(sq),
          last: lm !== null && (lm.from === sq || lm.to === sq || lm.captures.includes(sq)),
        });
      }
    }
    return out;
  });

  protected readonly busy = computed(() => { this.tick(); return this.director.thinking; });
  protected readonly over = computed(() => { this.tick(); return this.director.outcome() !== 'ongoing'; });
  protected readonly captured = computed(() => { this.tick(); return this.director.captured(); });

  // tick() is the dependency that re-reads the director's plain fields (netReady/thinking) — without it
  // this computed caches the value from the first render, where the checkpoint is still downloading, and
  // the board stays disabled forever on slow connections.
  protected readonly locked = computed(() => {
    this.tick();
    return !this.director.netReady || this.director.thinking || this.over() || this.mode() === 'watch';
  });

  protected statusText(): string {
    this.tick();
    if (!this.director.netReady) return 'Loading the AI…';
    if (this.director.thinking) return 'The AI is thinking…';
    switch (this.director.outcome()) {
      case 'you-win': return 'You win! 🎉';
      case 'ai-wins': return 'The AI wins.';
      case 'draw': return 'Draw — no progress (king shuffling).';
    }
    const note = this.director.netMissing ? ' (no trained net found — playing random moves)' : '';
    if (this.mode() === 'watch') return `${this.director.whiteToMove ? 'White' : 'Black'} to move.${note}`;
    const forced = this.director.mustCapture() ? 'Capture! ' : '';
    return `${forced}Your move.${note}`;
  }

  protected hint(): string {
    this.tick();
    return this.director.mustCapture()
      ? 'Captures are forced — click a piece, then the landing square (a multi-jump plays in one go).'
      : 'Click a piece, then its destination.';
  }

  protected setMode(m: Mode): void {
    if (m === this.mode()) return;
    this.mode.set(m);
    this.loopGen++;
    this.director.reset();
    this.deselect();
    this.refresh();
    if (m === 'watch') void this.runWatch();
  }

  protected async onDifficultyChange(d: DraughtsDifficulty): Promise<void> {
    this.loopGen++;
    this.refresh();
    await this.director.setDifficulty(d);
    this.director.reset();
    this.deselect();
    this.refresh();
    if (this.mode() === 'watch') void this.runWatch();
  }

  protected newGame(): void {
    this.director.reset();
    this.deselect();
    this.refresh();
  }

  protected onClick(sq: number): void {
    if (this.locked()) return;
    const b = this.board();
    const sel = this.selected();
    if (sel === null) {
      if (b[sq] > 0) this.select(sq);              // only your own (White) pieces
      return;
    }
    // Target check FIRST: a capturing king can legally loop back to its own square (from === to).
    if (this.targets().has(sq)) { void this.play(sel, sq); return; }
    if (sq === sel) { this.deselect(); return; }
    if (b[sq] > 0) { this.select(sq); return; }    // switch selection
    this.deselect();
  }

  private select(sq: number): void {
    this.selected.set(sq);
    this.targets.set(new Set(this.director.legalTargets(sq)));
  }
  private deselect(): void {
    this.selected.set(null);
    this.targets.set(new Set());
  }

  // Play mode: apply the human's move, paint it, then let the AI reply.
  private async play(from: number, to: number): Promise<void> {
    if (!this.director.humanMove(from, to)) { this.deselect(); return; }
    this.deselect();
    this.refresh();
    if (this.director.outcome() !== 'ongoing') return;
    const reply = this.director.aiStep(); // sets thinking=true synchronously
    this.tick.update(v => v + 1);
    await reply;
    this.refresh();
  }

  // Watch mode: the AI plays both sides on a loop, restarting a finished game. Cancels via loopGen.
  private async runWatch(): Promise<void> {
    const gen = this.loopGen;
    while (gen === this.loopGen && this.mode() === 'watch') {
      if (this.director.outcome() !== 'ongoing') {
        await this.delay(GAMEOVER_MS);
        if (gen !== this.loopGen) return;
        this.director.reset();
        this.refresh();
        continue;
      }
      const step = this.director.aiStep();
      this.tick.update(v => v + 1);
      await step;
      if (gen !== this.loopGen) return;
      this.refresh();
      await this.delay(this.betweenMs());
    }
  }

  private delay(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)); }

  private refresh(): void {
    this.board.set(this.director.board());
    this.last.set(this.director.lastMove);
    this.tick.update(v => v + 1);
  }
}
