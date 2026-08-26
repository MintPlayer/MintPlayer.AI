// Canvas renderer for Tetris — pure view code: it reads the game's board, active piece, watch animation
// and flash state and draws; all rules live in the single-source engine. Stack cells are a neutral slate
// (row bitmasks carry no piece identity — a deliberate engine choice); the falling piece and the next-piece
// preview carry the classic per-piece colors.

import { TetrisGame, H, W } from './tetris-game';

// Logical space; the component scales the canvas to it (DPR handled there).
export const LOGICAL_W = 620;
export const LOGICAL_H = 760;
const BOARD_X = 16;
const BOARD_Y = 44;
const CELL = 34; // 10 × 34 = 340 wide, 20 × 34 = 680 tall
const SIDE_X = BOARD_X + W * CELL + 20;

const PIECE_COLORS = ['#4dd7e8', '#f2d24d', '#b06ee8', '#6ee87a', '#e86e6e', '#e8a34d', '#5d8aee'];
const PIECE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'L', 'J'];
const STACK_COLOR = '#5a6580';
const GARBAGE_TINT = '#4a4356';

export function pointToBoard(sx: number, sy: number, cssW: number): { x: number; y: number } | null {
  const scale = LOGICAL_W / cssW;
  const lx = sx * scale - BOARD_X;
  const ly = sy * scale - BOARD_Y;
  const x = Math.floor(lx / CELL);
  const y = Math.floor(ly / CELL);
  return x >= 0 && x < W && y >= 0 && y < H ? { x, y } : null;
}

/** One cell of horizontal drag in canvas CSS pixels. */
export function cellWidthCss(cssW: number): number {
  return CELL * (cssW / LOGICAL_W);
}

export function render(ctx: CanvasRenderingContext2D, game: TetrisGame, cssW: number, cssH: number,
  statusLine: string, mode: 'human' | 'watch'): void {
  const s = cssW / LOGICAL_W;
  ctx.save();
  ctx.scale(s, s);
  const b = game.board;

  ctx.fillStyle = '#141824';
  ctx.fillRect(0, 0, LOGICAL_W, LOGICAL_H);

  // Status line above the well.
  ctx.fillStyle = '#aab2c5';
  ctx.font = '15px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(statusLine, BOARD_X, 28);

  // The well.
  ctx.fillStyle = '#0d1017';
  ctx.fillRect(BOARD_X, BOARD_Y, W * CELL, H * CELL);
  ctx.strokeStyle = '#2a3040';
  ctx.strokeRect(BOARD_X - 0.5, BOARD_Y - 0.5, W * CELL + 1, H * CELL + 1);

  // Stack cells (garbage rows — full rows with one gap — get a slightly warmer tint).
  for (let y = 0; y < H; y++) {
    const row = b.rows[y];
    if (row === 0) continue;
    const isGarbage = popcount(row) === W - 1 && game.garbageEvery > 0;
    for (let x = 0; x < W; x++) {
      if (((row >> x) & 1) === 1) drawCell(ctx, x, y, isGarbage ? GARBAGE_TINT : STACK_COLOR);
    }
  }

  // Falling piece: human = the live micro piece + its ghost; watch = the animated drop.
  if (mode === 'human' && b.activeLive && !b.gameOver) {
    const gy = game.ghostY();
    if (gy >= 0 && gy !== b.activeY) drawPiece(ctx, b, b.current, b.activeRot, b.activeX, gy, null, true);
    drawPiece(ctx, b, b.current, b.activeRot, b.activeX, b.activeY, PIECE_COLORS[b.current], false);
  }
  const anim = game.anim;
  if (anim) {
    const yNow = Math.min(anim.yTo, anim.yTo * easeIn(anim.t));
    drawPiece(ctx, b, anim.piece, anim.rot, anim.x, yNow, PIECE_COLORS[anim.piece], false);
  }

  // Line-clear flash: a soft band pulse over the whole well (the rows are already gone in the engine).
  if (game.flashMs > 0) {
    ctx.fillStyle = `rgba(255, 244, 180, ${0.28 * (game.flashMs / 220)})`;
    ctx.fillRect(BOARD_X, BOARD_Y, W * CELL, H * CELL);
  }

  // Sidebar: next piece, score, lines, pieces, garbage countdown.
  ctx.fillStyle = '#aab2c5';
  ctx.font = '15px system-ui, sans-serif';
  ctx.fillText('Next', SIDE_X, BOARD_Y + 12);
  drawPreview(ctx, b, b.next, SIDE_X, BOARD_Y + 24);

  const stats: [string, string][] = [
    ['Score', `${b.score}`],
    ['Lines', `${b.lines}`],
    ['Tetrises', `${b.tetrises}`],
    ['Pieces', `${b.piecesPlaced}`],
  ];
  if (game.garbageEvery > 0) stats.push(['Garbage in', `${game.garbageIn()}`]);
  let sy = BOARD_Y + 150;
  for (const [label, value] of stats) {
    ctx.fillStyle = '#8b93a7';
    ctx.font = '13px system-ui, sans-serif';
    ctx.fillText(label, SIDE_X, sy);
    ctx.fillStyle = '#e8ecf4';
    ctx.font = 'bold 22px system-ui, sans-serif';
    ctx.fillText(value, SIDE_X, sy + 24);
    sy += 62;
  }

  // Game-over overlay.
  if (b.gameOver && !anim) {
    ctx.fillStyle = 'rgba(10, 12, 18, 0.72)';
    ctx.fillRect(BOARD_X, BOARD_Y, W * CELL, H * CELL);
    ctx.fillStyle = '#e8ecf4';
    ctx.textAlign = 'center';
    ctx.font = 'bold 30px system-ui, sans-serif';
    ctx.fillText('Game over', BOARD_X + W * CELL / 2, BOARD_Y + H * CELL / 2 - 18);
    ctx.font = '16px system-ui, sans-serif';
    ctx.fillStyle = '#aab2c5';
    ctx.fillText(`${b.lines} lines · ${b.score} points`, BOARD_X + W * CELL / 2, BOARD_Y + H * CELL / 2 + 12);
    if (mode === 'human')
      ctx.fillText('tap or press Enter to play again', BOARD_X + W * CELL / 2, BOARD_Y + H * CELL / 2 + 40);
    ctx.textAlign = 'left';
  }

  ctx.restore();
}

function popcount(v: number): number {
  let n = 0;
  while (v) { n += v & 1; v >>= 1; }
  return n;
}

function easeIn(t: number): number {
  return t * t;
}

function drawCell(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, outline = false): void {
  const px = BOARD_X + x * CELL;
  const py = BOARD_Y + y * CELL;
  if (outline) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(px + 2.5, py + 2.5, CELL - 5, CELL - 5);
    return;
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(px + 1.5, py + 1.5, CELL - 3, CELL - 3, 5);
  ctx.fill();
  // Subtle top highlight for depth.
  ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.beginPath();
  ctx.roundRect(px + 3, py + 3, CELL - 6, 7, 4);
  ctx.fill();
}

// Draws the 4 cells of a piece rotation at board position (x, y); fractional y supported for the watch
// glide. `ghost` = outline only.
function drawPiece(ctx: CanvasRenderingContext2D, b: { cellX: number[]; cellY: number[] },
  piece: number, rot: number, x: number, y: number, color: string | null, ghost: boolean): void {
  const ri = piece * 4 + rot;
  for (let k = 0; k < 4; k++) {
    const cx = x + b.cellX[ri * 4 + k];
    const cy = y + b.cellY[ri * 4 + k];
    if (ghost) drawCell(ctx, cx, cy, 'rgba(232, 236, 244, 0.35)', true);
    else drawCellAt(ctx, BOARD_X + cx * CELL, BOARD_Y + cy * CELL, color!);
  }
}

function drawCellAt(ctx: CanvasRenderingContext2D, px: number, py: number, color: string): void {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(px + 1.5, py + 1.5, CELL - 3, CELL - 3, 5);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
  ctx.beginPath();
  ctx.roundRect(px + 3, py + 3, CELL - 6, 7, 4);
  ctx.fill();
}

function drawPreview(ctx: CanvasRenderingContext2D, b: { cellX: number[]; cellY: number[]; rotW: number[]; rotH: number[] },
  piece: number, px: number, py: number): void {
  const ri = piece * 4; // spawn rotation
  const cell = 22;
  const w = b.rotW[ri] * cell;
  ctx.fillStyle = '#0d1017';
  ctx.fillRect(px, py, 4 * cell + 16, 2.5 * cell + 16);
  for (let k = 0; k < 4; k++) {
    const cx = px + 8 + b.cellX[ri * 4 + k] * cell + (4 * cell - w) / 2;
    const cy = py + 8 + b.cellY[ri * 4 + k] * cell;
    ctx.fillStyle = PIECE_COLORS[piece];
    ctx.beginPath();
    ctx.roundRect(cx + 1, cy + 1, cell - 2, cell - 2, 4);
    ctx.fill();
  }
  ctx.fillStyle = '#8b93a7';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText(PIECE_NAMES[piece], px + 4 * cell + 22, py + 14);
}
