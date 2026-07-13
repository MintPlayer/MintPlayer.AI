import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { BsButtonGroupComponent } from '@mintplayer/ng-bootstrap/button-group';
import { BsSelectComponent, BsSelectOption } from '@mintplayer/ng-bootstrap/select';
import { BsRangeComponent } from '@mintplayer/ng-bootstrap/range';
import { ChessDirector } from './chess-director';
import { ChessDifficulty } from './chess-net';

interface Cell {
  sq: number;      // engine square index (rank*8 + file)
  piece: number;   // 0 empty, +1..+6 White, −1..−6 Black
  light: boolean;
  selected: boolean;
  target: boolean; // a legal destination for the selected piece
  last: boolean;   // part of the last move (from or to)
}

type Mode = 'play' | 'watch';

// Filled glyphs for both colours; colour is conveyed by CSS (white pieces light, black pieces dark) so they read
// on both square shades. Indexed by |piece|: 1 P … 6 K.
const GLYPH = ['', '♟', '♞', '♝', '♜', '♛', '♚'];

const BETWEEN_MS = 550;   // pause between moves in AI-vs-AI so it's watchable
const GAMEOVER_MS = 2200; // pause on a finished AI-vs-AI board before auto-restarting

/**
 * Chess — play the self-taught AI, or watch it play itself, entirely in your browser. The engine, the policy/value
 * network, and the PUCT search are single-sourced from `chess_solver.pg` and transpiled to TypeScript; the browser
 * downloads a trained checkpoint and thinks locally, with no server move computation (M40.3). The difficulty picker
 * (M40.4) chooses which tier checkpoint + search knobs to use, from the Lab-written manifest. Controls use
 * @mintplayer/ng-bootstrap; the app's dark-blue theme is preserved via Bootstrap CSS variables (styles.scss).
 */
@Component({
  selector: 'app-chess',
  imports: [
    FormsModule,
    BsButtonGroupComponent, BsButtonTypeDirective,
    BsSelectComponent, BsSelectOption,
    BsRangeComponent,
  ],
  template: `
    <div class="wrap">
      <h1>Chess</h1>
      <p class="intro">
        A network that learned chess from scratch by playing itself (AlphaZero-style self-play), running
        <strong>entirely in your browser</strong> — no server. It's a small, still-learning net, so it plays legal,
        beatable chess. Pick a difficulty, then play it or watch it play itself.
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
                <option [ngValue]="d">{{ d.label }}{{ d.winRateVsRandom != null ? ' · ' + pct(d.winRateVsRandom) + ' vs random' : '' }}</option>
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
              <span class="pc" [class.white]="cell.piece > 0" [class.black]="cell.piece < 0">{{ glyph(cell.piece) }}</span>
            }
            @if (cell.target) { <span class="dot" [class.capture]="cell.piece !== 0"></span> }
          </button>
        }
      </div>

      <div class="controls">
        @if (mode() === 'play') {
          <button [color]="colors.primary" (click)="newGame()">New game</button>
          <span class="hint" [class.on]="over()">
            {{ over() ? 'Game over — start a new game.' : 'Click a piece, then its destination.' }}
          </span>
        } @else {
          <label class="speed">
            <span>Speed</span>
            <bs-range [min]="0" [max]="1900" [step]="50"
              [ngModel]="1900 - betweenMs()" (ngModelChange)="betweenMs.set(1900 - $event)"></bs-range>
            <span class="speed-end">{{ betweenMs() >= 1200 ? 'slow' : betweenMs() <= 250 ? 'fast' : '' }}</span>
          </label>
        }

        @if (captured().length) {
          <div class="tray" title="Captured pieces">
            @for (p of captured(); track $index) {
              <span class="cap" [class.white]="p > 0" [class.black]="p < 0">{{ glyph(p) }}</span>
            }
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
      grid-template-rows: repeat(8, 1fr);   /* equal rows so empty squares don't collapse */
      width: min(90vw, 512px);
      aspect-ratio: 1;
      border: 3px solid #2b3245;
      border-radius: 8px;
      overflow: hidden;
      user-select: none;
    }

    /* Board squares are bespoke (a grid of buttons), not Bootstrap buttons — :not(.btn) keeps them off the .btn path. */
    .sq {
      position: relative;
      border: 0; padding: 0; margin: 0;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      font-size: clamp(1.6rem, 7vw, 2.6rem);
      line-height: 1;
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

    .pc { position: relative; z-index: 1; line-height: 1; }
    .pc.white { color: #f7f9fc; text-shadow: 0 0 2px #000, 0 1px 1px #000; }
    .pc.black { color: #1a1f2b; text-shadow: 0 0 1px #000; }

    .dot {
      position: absolute; width: 28%; height: 28%;
      border-radius: 50%; background: rgba(232, 145, 42, 0.7); z-index: 1;
    }
    .dot.capture {
      width: 82%; height: 82%; background: transparent;
      border: 4px solid rgba(232, 145, 42, 0.7);
    }

    .controls { margin-top: 1rem; display: flex; align-items: center; gap: 1rem; }
    .hint { color: #8891a5; font-size: 0.9rem; }
    .hint.on { color: #e0b050; }
    .speed { display: flex; align-items: center; gap: 0.6rem; color: #8891a5; font-size: 0.9rem; }
    .speed-end { min-width: 2.5em; color: #e8912a; }

    .tray {
      display: flex; align-items: center; flex-wrap: wrap; gap: 0.1rem;
      font-size: 1.5rem; line-height: 1;
    }
    .cap { position: relative; }
    .cap.white { color: #f7f9fc; text-shadow: 0 0 2px #000, 0 1px 1px #000; }
    .cap.black { color: #1a1f2b; text-shadow: 0 0 1px #000; }
    .cap::after {
      content: '✕';
      position: absolute; inset: 0;
      display: flex; align-items: center; justify-content: center;
      color: #e0403f; font-size: 1.1em; font-weight: 700;
      pointer-events: none;
    }
  `,
})
export class Chess {
  protected readonly colors = Color;
  protected readonly director = new ChessDirector();
  protected readonly mode = signal<Mode>('play');
  protected readonly betweenMs = signal(BETWEEN_MS); // watch-mode pause between moves (speed slider)

  private readonly board = signal<number[]>(this.director.board());
  private readonly selected = signal<number | null>(null);
  private readonly targets = signal<Set<number>>(new Set());
  private readonly last = signal<{ from: number; to: number } | null>(null);
  private readonly tick = signal(0); // bumped to refresh derived status (net-load, thinking, outcome, difficulties)

  private loopGen = 0; // cancels a stale watch loop on mode/difficulty change / destroy

  constructor() {
    // Leave the "loading" state once the manifest + default checkpoint have settled.
    void this.director.ready.then(() => this.tick.update(v => v + 1));
    inject(DestroyRef).onDestroy(() => { this.loopGen++; });
  }

  protected glyph(piece: number): string { return GLYPH[Math.abs(piece)]; }
  protected pct(x: number): string { return `${Math.round(x * 100)}%`; }

  protected readonly difficulties = computed<ChessDifficulty[]>(() => { this.tick(); return this.director.difficulties; });
  protected readonly current = computed<ChessDifficulty>(() => { this.tick(); return this.director.current; });

  protected readonly cells = computed<Cell[]>(() => {
    const b = this.board();
    const sel = this.selected();
    const tg = this.targets();
    const lm = this.last();
    const out: Cell[] = [];
    for (let row = 0; row < 8; row++) {           // display top→bottom = rank 7→0 (White at the bottom)
      const rank = 7 - row;
      for (let file = 0; file < 8; file++) {
        const sq = rank * 8 + file;
        out.push({
          sq,
          piece: b[sq],
          light: ((rank + file) & 1) === 1,        // a1 (0,0) is dark
          selected: sel === sq,
          target: tg.has(sq),
          last: lm !== null && (lm.from === sq || lm.to === sq),
        });
      }
    }
    return out;
  });

  protected readonly busy = computed(() => { this.tick(); return this.director.thinking; });
  protected readonly over = computed(() => { this.tick(); return this.director.outcome() !== 'ongoing'; });
  protected readonly captured = computed(() => { this.tick(); return this.director.capturedPieces(); });

  // Not clickable while the net loads, the AI thinks, the game is over, or we're watching AI-vs-AI.
  protected readonly locked = computed(() =>
    !this.director.netReady || this.director.thinking || this.over() || this.mode() === 'watch');

  protected statusText(): string {
    this.tick();
    if (!this.director.netReady) return 'Loading the AI…';
    if (this.director.thinking) return 'The AI is thinking…';
    switch (this.director.outcome()) {
      case 'you-win': return 'Checkmate — you win! 🎉';
      case 'ai-wins': return 'Checkmate — the AI wins.';
      case 'stalemate': return 'Stalemate — it\'s a draw.';
      case 'draw': return 'Draw.';
    }
    const note = this.director.netMissing ? ' (no trained net found — playing random moves)' : '';
    if (this.mode() === 'watch') return `${this.director.whiteToMove ? 'White' : 'Black'} to move.${note}`;
    const check = this.director.inCheck() ? 'Check! ' : '';
    return `${check}Your move.${note}`;
  }

  protected setMode(m: Mode): void {
    if (m === this.mode()) return;
    this.mode.set(m);
    this.loopGen++;            // cancel any running watch loop
    this.director.reset();
    this.deselect();
    this.refresh();
    if (m === 'watch') void this.runWatch();
  }

  protected async onDifficultyChange(d: ChessDifficulty): Promise<void> {
    this.loopGen++;                       // cancel any running watch loop while the new net loads
    this.refresh();                       // reflect "Loading the AI…" (setDifficulty flips netReady off)
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
      if (b[sq] > 0) this.select(sq);         // only your own (White) pieces
      return;
    }
    if (sq === sel) { this.deselect(); return; }
    if (this.targets().has(sq)) { void this.play(sel, sq); return; }
    if (b[sq] > 0) { this.select(sq); return; } // switch selection
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

  // Play mode: apply the human's move, paint it, then let the AI (Black) reply.
  private async play(from: number, to: number): Promise<void> {
    if (!this.director.humanMove(from, to)) { this.deselect(); return; }
    this.deselect();
    this.refresh();
    if (this.director.outcome() !== 'ongoing') return;
    const reply = this.director.aiStep(); // sets thinking=true synchronously
    this.tick.update(v => v + 1);         // reflect "thinking" now (before the blocking search)
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
      this.tick.update(v => v + 1);       // "thinking" / whose-move
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
