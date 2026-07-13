import { PgChessState, PgChessMcts, PgPolicyValueNet } from './chess_solver';
import { loadChessNet } from './chess-net';

// Client-side chess "play against the AI" director — the whole AI runs in the browser, single-sourced from
// chess_solver.pg (the same engine + net + MCTS the C# training uses). You are White; the AI is Black and replies
// with PgChessMcts over the net loaded from the shipped .ckpt. No server move computation (the FruitCake pattern).
//
// The engine speaks AlphaZero action indices (0..4671). The board UI speaks (from, to) squares, so the director
// resolves a human click to a legal index (auto-queen on promotion) and reads decoded moves back for highlighting.

// Start position as the engine's mailbox: index = rank*8 + file (a1 = 0); +1..+6 White P,N,B,R,Q,K, −1..−6 Black.
// Mirrors ChessFen.StartFen / ChessState.StartPosition (the engine has no FEN parser on the browser side).
const START_SQUARES: number[] = [
   4,  2,  3,  5,  6,  3,  2,  4,   // rank 1  a1..h1  R N B Q K B N R
   1,  1,  1,  1,  1,  1,  1,  1,   // rank 2  white pawns
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
  -1, -1, -1, -1, -1, -1, -1, -1,   // rank 7  black pawns
  -4, -2, -3, -5, -6, -3, -2, -4,   // rank 8  a8..h8
];
const CASTLE_ALL = 1 | 2 | 4 | 8; // WK|WQ|BK|BQ

export interface DecodedMove { index: number; from: number; to: number; promo: number; }

export class ChessDirector {
  private state = ChessDirector.startState();
  private net: PgPolicyValueNet | null = null;

  netReady = false;   // the checkpoint fetch has resolved (success or not)
  netMissing = false; // resolved but no usable net → AI falls back to a random legal move
  thinking = false;   // an AI search is in flight
  lastMove: { from: number; to: number } | null = null;

  /** Inference sims per AI move — the latency knob (browser MCTS). Modest for ~1–2 s/move. */
  sims = 96;
  cpuct = 1.5;

  /** Resolves when the checkpoint fetch has settled (the UI awaits this to leave the "loading" state). */
  readonly ready: Promise<void>;

  constructor() {
    this.ready = loadChessNet().then(net => {
      this.net = net;
      this.netMissing = net === null;
      this.netReady = true;
    });
  }

  private static startState(): PgChessState {
    return new PgChessState([...START_SQUARES], true, CASTLE_ALL, -1, 0);
  }

  reset(): void {
    this.state = ChessDirector.startState();
    this.lastMove = null;
    this.thinking = false;
  }

  /** A copy of the 64-cell board for rendering (never the engine's internal array). */
  board(): number[] { return [...this.state.squares]; }
  get whiteToMove(): boolean { return this.state.whiteToMove; }

  private legal(): DecodedMove[] {
    return this.state.legalMoveIndices().map(index => {
      const m = PgChessState.decode(index);
      return { index, from: m.from, to: m.to, promo: m.promotion };
    });
  }

  /** Destination squares for a piece on `from` (for move-target highlighting). */
  legalTargets(from: number): number[] {
    return this.legal().filter(m => m.from === from).map(m => m.to);
  }

  // Resolve a human (from,to) to a legal action index; prefer the queen-plane index on a promotion (auto-queen —
  // applyIndex infers Queen), matching the engine. Returns -1 if the move is illegal.
  private resolve(from: number, to: number): number {
    const matches = this.legal().filter(m => m.from === from && m.to === to);
    if (matches.length === 0) return -1;
    return (matches.find(m => m.promo === 0) ?? matches[0]).index;
  }

  /** Apply the human's (White) move if legal. Returns false (no-op) if it's not your turn or the move is illegal. */
  humanMove(from: number, to: number): boolean {
    if (this.thinking || !this.state.whiteToMove) return false;
    const index = this.resolve(from, to);
    if (index < 0) return false;
    this.state = this.state.applyIndex(index);
    this.lastMove = { from, to };
    return true;
  }

  /** Play the AI's move for whichever side is to move (Black's reply in play mode; either side in watch mode).
   *  Yields to the UI first (so the board paints the previous move + a "thinking" cue), then runs the search
   *  synchronously. Falls back to a random legal move if the checkpoint didn't load. No-op at a terminal position. */
  async aiStep(): Promise<void> {
    if (this.result() !== 0) return;
    this.thinking = true;
    await new Promise(resolve => setTimeout(resolve, 30));
    let index: number;
    if (this.net) {
      index = PgChessMcts.chooseMove(this.net, this.state, this.sims, this.cpuct);
    } else {
      const moves = this.state.legalMoveIndices();
      index = moves[Math.floor(Math.random() * moves.length)];
    }
    const m = PgChessState.decode(index);
    this.state = this.state.applyIndex(index);
    this.lastMove = { from: m.from, to: m.to };
    this.thinking = false;
  }

  // Full starting material as [signed piece code, count]: +1..+6 White P,N,B,R,Q,K, −1..−6 Black.
  private static readonly START_COUNTS: ReadonlyArray<readonly [number, number]> = [
    [-1, 8], [-2, 2], [-3, 2], [-4, 2], [-5, 1],   // Black (the AI's pieces you've captured)
    [1, 8], [2, 2], [3, 2], [4, 2], [5, 1],        // White (your pieces the AI has captured)
  ];

  /** Pieces no longer on the board (signed codes), i.e. the captured material — for a captured-pieces tray. Derived
   *  from the board vs the starting set (a promotion can make this off by the promoted pawn, which is rare here). */
  capturedPieces(): number[] {
    const current = new Map<number, number>();
    for (const p of this.state.squares) if (p !== 0) current.set(p, (current.get(p) ?? 0) + 1);
    const out: number[] = [];
    for (const [code, count] of ChessDirector.START_COUNTS) {
      const lost = count - (current.get(code) ?? 0);
      for (let i = 0; i < lost; i++) out.push(code);
    }
    return out;
  }

  /** Terminal result from the side-to-move's view: 0 ongoing, 1 loss (checkmated), 2 draw. */
  result(): number { return this.state.result(); }
  inCheck(): boolean { return this.state.inCheck(this.state.whiteToMove); }
  private hasLegalMoves(): boolean { return this.state.legalMoveIndices().length > 0; }

  /** Human-facing outcome. */
  outcome(): 'ongoing' | 'you-win' | 'ai-wins' | 'stalemate' | 'draw' {
    const r = this.result();
    if (r === 1) return this.state.whiteToMove ? 'ai-wins' : 'you-win'; // the mated side is the one to move
    if (r === 2) return !this.hasLegalMoves() && !this.inCheck() ? 'stalemate' : 'draw';
    return 'ongoing';
  }
}
