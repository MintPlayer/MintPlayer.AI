// Canvas renderer for Crazy Fruits — the confirmed KidCity identity (cartoon fruit market stall: bunting
// flags, strawberry-red wordmark, saturated fruit art; original art, no KidCity/Belgacom branding). Pure
// view code: it reads the game's current animation step + progress and draws; all rules live in the engine.

import { drawFruit as drawFruitCakeArt } from '../fruit-cake/fruit-cake-art';
import { AnimStep, CrazyFruitsGame, SIZE } from './crazy-fruits-game';

// Logical space; the canvas scales it to its CSS size (device-pixel-ratio handled by the component).
export const LOGICAL_W = 620;
export const LOGICAL_H = 714;
const HEADER_H = 78;
const BOARD_X = 10;
const BOARD_Y = HEADER_H + 6;
const BOARD_SIZE = 600;
const CELL = BOARD_SIZE / SIZE; // 75

const BUNTING = ['#e74c3c', '#f1c40f', '#2ecc71', '#3498db', '#e67e22', '#9b59b6'];

/** Map a canvas-CSS-pixel point to a board cell index, or -1 outside the board. */
export function pointToCell(sx: number, sy: number, cssW: number): number {
  const s = LOGICAL_W / cssW;
  const lx = sx * s - BOARD_X;
  const ly = sy * s - BOARD_Y;
  if (lx < 0 || ly < 0 || lx >= BOARD_SIZE || ly >= BOARD_SIZE) return -1;
  return Math.floor(ly / CELL) * SIZE + Math.floor(lx / CELL);
}

/** Drag threshold in canvas CSS pixels for the given canvas width (0.35 cell). */
export function dragThreshold(cssW: number): number {
  return CELL * 0.35 * (cssW / LOGICAL_W);
}

export function render(ctx: CanvasRenderingContext2D, game: CrazyFruitsGame, cssW: number, cssH: number,
  statusLine: string, roundBars: readonly string[] = []): void {
  ctx.save();
  ctx.scale(cssW / LOGICAL_W, cssH / LOGICAL_H);

  drawBackdrop(ctx);
  drawHeader(ctx, game, statusLine);

  // Board plate
  roundRect(ctx, BOARD_X - 4, BOARD_Y - 4, BOARD_SIZE + 8, BOARD_SIZE + 8, 14);
  ctx.fillStyle = '#5d3a1a';
  ctx.fill();
  roundRect(ctx, BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE, 10);
  ctx.fillStyle = '#241a33';
  ctx.fill();

  // Checkerboard cells
  for (let r = 0; r < SIZE; r++)
    for (let c = 0; c < SIZE; c++) {
      ctx.fillStyle = (r + c) % 2 === 0 ? '#2c2140' : '#271d39';
      ctx.fillRect(BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, CELL);
    }

  const step = game.currentStep;
  if (!step) {
    drawGrid(ctx, game.board.grid, game.selected);
  } else {
    drawStep(ctx, step, game.progress);
  }

  if (game.roundOver) drawRoundOver(ctx, game, roundBars);

  ctx.restore();
}

// The 30-move round is over (SPECIALS PRD §3.8): scrim + score + the measured tier bars as challenge lines.
function drawRoundOver(ctx: CanvasRenderingContext2D, game: CrazyFruitsGame, roundBars: readonly string[]): void {
  ctx.fillStyle = 'rgba(15, 8, 26, 0.82)';
  ctx.fillRect(BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const cx = BOARD_X + BOARD_SIZE / 2;
  let y = BOARD_Y + BOARD_SIZE * 0.24;

  ctx.font = 'bold 34px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#f1c40f';
  ctx.fillText('Round over!', cx, y);
  y += 56;
  ctx.font = 'bold 46px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#f0e6cc';
  ctx.fillText(`${game.board.score}`, cx, y);
  y += 34;
  ctx.font = '15px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#aab2c5';
  ctx.fillText(`best ${game.best}`, cx, y);
  y += 44;
  for (const bar of roundBars) {
    ctx.fillStyle = '#8fd18f';
    ctx.fillText(bar, cx, y);
    y += 26;
  }
  y += 22;
  ctx.font = 'bold 18px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#6ea8fe';
  ctx.fillText('tap to play again', cx, y);
}

// ── Scene pieces ─────────────────────────────────────────────────────────────────────────────────────────

function drawBackdrop(ctx: CanvasRenderingContext2D): void {
  const sky = ctx.createLinearGradient(0, 0, 0, LOGICAL_H);
  sky.addColorStop(0, '#1e1430');
  sky.addColorStop(1, '#171022');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, LOGICAL_W, LOGICAL_H);

  // Bunting flags along the top — the one confirmed visual signature of the original.
  ctx.strokeStyle = '#c9a227';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 8);
  ctx.quadraticCurveTo(LOGICAL_W / 2, 22, LOGICAL_W, 8);
  ctx.stroke();
  const flags = 10;
  for (let i = 0; i < flags; i++) {
    const t0 = i / flags;
    const t1 = (i + 0.9) / flags;
    const x0 = t0 * LOGICAL_W;
    const x1 = t1 * LOGICAL_W;
    const xm = (x0 + x1) / 2;
    const y0 = 9 + 12 * Math.sin(Math.PI * t0);
    const y1 = 9 + 12 * Math.sin(Math.PI * t1);
    ctx.fillStyle = BUNTING[i % BUNTING.length];
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.lineTo(xm, Math.max(y0, y1) + 30);
    ctx.closePath();
    ctx.fill();
  }
}

function drawHeader(ctx: CanvasRenderingContext2D, game: CrazyFruitsGame, statusLine: string): void {
  ctx.textBaseline = 'middle';

  // Wordmark
  ctx.font = 'bold 26px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#e74c3c';
  ctx.textAlign = 'left';
  ctx.fillText('CRAZY', BOARD_X + 4, 52);
  ctx.fillStyle = '#f1c40f';
  ctx.fillText('FRUITS', BOARD_X + 92, 52);

  // Score / moves / best
  ctx.font = 'bold 22px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#f0e6cc';
  ctx.textAlign = 'center';
  ctx.fillText(`${game.board.score}`, LOGICAL_W / 2 + 30, 46);
  ctx.font = '13px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#aab2c5';
  ctx.fillText(`score · move ${game.board.movesMade}`, LOGICAL_W / 2 + 30, 66);

  ctx.textAlign = 'right';
  ctx.font = 'bold 16px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = '#8fd18f';
  ctx.fillText(`best ${game.best}`, LOGICAL_W - 14, 46);
  if (statusLine) {
    ctx.font = '13px "Segoe UI", system-ui, sans-serif';
    ctx.fillStyle = '#aab2c5';
    ctx.fillText(statusLine, LOGICAL_W - 14, 66);
  }
}

// ── Board states ─────────────────────────────────────────────────────────────────────────────────────────

function drawGrid(ctx: CanvasRenderingContext2D, grid: number[], selected: number): void {
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (grid[i] === 0) continue;
    const { x, y } = cellCenter(i);
    if (i === selected) {
      ctx.strokeStyle = '#f1c40f';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.arc(x, y, CELL * 0.47, 0, Math.PI * 2);
      ctx.stroke();
    }
    drawFruit(ctx, grid[i], x, y, CELL * 0.42);
  }
}

function drawStep(ctx: CanvasRenderingContext2D, step: AnimStep, t: number): void {
  const ease = t * t * (3 - 2 * t); // smoothstep
  switch (step.kind) {
    case 'swap': {
      for (let i = 0; i < SIZE * SIZE; i++) {
        if (i === step.a || i === step.b || step.grid[i] === 0) continue;
        const { x, y } = cellCenter(i);
        drawFruit(ctx, step.grid[i], x, y, CELL * 0.42);
      }
      const pa = cellCenter(step.a);
      const pb = cellCenter(step.b);
      drawFruit(ctx, step.fruitB, pb.x + (pa.x - pb.x) * ease, pb.y + (pa.y - pb.y) * ease, CELL * 0.42);
      drawFruit(ctx, step.fruitA, pa.x + (pb.x - pa.x) * ease, pa.y + (pb.y - pa.y) * ease, CELL * 0.44);
      break;
    }
    case 'pop': {
      // Blast glows UNDER the fruit, styled by what cleared each cell (2 striped beam · 3 blast · 4 bomb zap).
      for (let i = 0; i < SIZE * SIZE; i++) {
        const source = step.clearedBy[i];
        if (!step.cells[i] || source < 2) continue;
        const { x, y } = cellCenter(i);
        ctx.save();
        ctx.globalAlpha = 0.65 * (1 - ease);
        if (source === 2) {
          ctx.fillStyle = '#ffd64a';
          ctx.fillRect(x - CELL / 2, y - CELL * 0.18, CELL, CELL * 0.36);
          ctx.fillRect(x - CELL * 0.18, y - CELL / 2, CELL * 0.36, CELL);
        } else {
          ctx.fillStyle = source === 3 ? '#ff7b3d' : '#c05bff';
          ctx.beginPath();
          ctx.arc(x, y, CELL * (0.5 + 0.25 * ease), 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
      for (let i = 0; i < SIZE * SIZE; i++) {
        if (step.grid[i] === 0) continue;
        const { x, y } = cellCenter(i);
        if (step.cells[i]) {
          ctx.save();
          ctx.globalAlpha = 1 - ease;
          drawFruit(ctx, step.grid[i], x, y, CELL * 0.42 * (1 + 0.35 * ease));
          ctx.restore();
        } else {
          drawFruit(ctx, step.grid[i], x, y, CELL * 0.42);
        }
      }
      // Created specials sparkle IN while their match fades out.
      for (let p = 0; p + 1 < step.created.length; p += 2) {
        const { x, y } = cellCenter(step.created[p]);
        drawFruit(ctx, step.created[p + 1], x, y, CELL * 0.42 * ease);
        ctx.save();
        ctx.globalAlpha = 1 - ease;
        ctx.strokeStyle = '#fff7cc';
        ctx.lineWidth = 2.5;
        for (let s = 0; s < 4; s++) {
          const a = s * Math.PI / 2 + ease * Math.PI;
          ctx.beginPath();
          ctx.moveTo(x + Math.cos(a) * CELL * 0.3, y + Math.sin(a) * CELL * 0.3);
          ctx.lineTo(x + Math.cos(a) * CELL * (0.3 + 0.25 * ease), y + Math.sin(a) * CELL * (0.3 + 0.25 * ease));
          ctx.stroke();
        }
        ctx.restore();
      }
      // Floating score popup at the centroid of the cleared cells.
      let cx = 0, cy = 0, n = 0;
      for (let i = 0; i < SIZE * SIZE; i++)
        if (step.cells[i]) { const p = cellCenter(i); cx += p.x; cy += p.y; n++; }
      if (n > 0) {
        ctx.font = 'bold 24px "Segoe UI", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#f1c40f';
        ctx.strokeStyle = 'rgba(0,0,0,0.6)';
        ctx.lineWidth = 4;
        const y = cy / n - 26 * ease;
        ctx.strokeText(`+${step.points}`, cx / n, y);
        ctx.fillText(`+${step.points}`, cx / n, y);
      }
      break;
    }
    case 'fall': {
      const moving = new Set(step.moves.map(m => m.toRow * SIZE + m.col));
      for (let i = 0; i < SIZE * SIZE; i++) {
        if (moving.has(i) || step.grid[i] === 0) continue;
        const { x, y } = cellCenter(i);
        drawFruit(ctx, step.grid[i], x, y, CELL * 0.42);
      }
      // Clip so refills appear from under the header, not over it.
      ctx.save();
      ctx.beginPath();
      ctx.rect(BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE);
      ctx.clip();
      for (const m of step.moves) {
        const fromY = BOARD_Y + m.fromRow * CELL + CELL / 2;
        const toY = BOARD_Y + m.toRow * CELL + CELL / 2;
        const x = BOARD_X + m.col * CELL + CELL / 2;
        drawFruit(ctx, m.fruit, x, fromY + (toY - fromY) * ease, CELL * 0.42);
      }
      ctx.restore();
      break;
    }
    case 'reshuffle': {
      ctx.save();
      ctx.globalAlpha = 0.35 + 0.65 * ease;
      drawGrid(ctx, step.grid, -1);
      ctx.restore();
      ctx.font = 'bold 26px "Segoe UI", system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = `rgba(241, 196, 64, ${1 - ease})`;
      ctx.fillText('No more moves — reshuffling!', LOGICAL_W / 2, BOARD_Y + BOARD_SIZE / 2);
      break;
    }
  }
}

function cellCenter(cell: number): { x: number; y: number } {
  return {
    x: BOARD_X + (cell % SIZE) * CELL + CELL / 2,
    y: BOARD_Y + Math.floor(cell / SIZE) * CELL + CELL / 2,
  };
}

// ── Fruit art — FruitCake's cached vector clipart, scaled down, plus specials overlays ──────────────────
// Six visually distinct picks from the 11-tier FruitCake catalog (no two share a color):
// 1 strawberry · 2 grape · 3 dekopon (orange) · 4 pear · 5 pineapple · 6 watermelon.
// Grid values are PACKED (kind·16 + type): stripes are painted ALONG the blast axis so players can read
// the direction, wrapped fruit get a golden wrapper ring, and the colorless sugar bomb is its own sphere.
const FRUIT_TIER = [0, 2, 3, 4, 7, 9, 11];

const fruitOf = (v: number) => v % 16;
const kindOf = (v: number) => (v / 16) | 0;

export function drawFruit(ctx: CanvasRenderingContext2D, packed: number, x: number, y: number, r: number): void {
  const fruit = fruitOf(packed);
  const kind = kindOf(packed);
  if (kind === 4) {
    drawBomb(ctx, x, y, r);
    return;
  }
  const wrapped = kind === 3 || kind === 5;
  if (wrapped) drawWrapperBack(ctx, x, y, r);
  if (fruit >= 1 && fruit <= 6) drawFruitCakeArt(ctx, FRUIT_TIER[fruit], x, y, wrapped ? r * 0.8 : r);
  if (kind === 1 || kind === 2) drawStripes(ctx, x, y, r, kind === 1);
  if (wrapped) drawWrapperFront(ctx, x, y, r);
}

// Crisp candy stripes with a dark outline so they read on light fruit (pear/pineapple) too.
function drawStripes(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, horizontal: boolean): void {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r * 0.86, 0, Math.PI * 2);
  ctx.clip();
  for (const i of [-1, 0, 1]) {
    const bx = horizontal ? x - r : x + i * r * 0.52 - r * 0.05;
    const by = horizontal ? y + i * r * 0.52 - r * 0.05 : y - r;
    const bw = horizontal ? r * 2 : r * 0.1;
    const bh = horizontal ? r * 0.1 : r * 2;
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(bx, by, bw, bh);
  }
  ctx.restore();
}

// A SQUARE candy wrapper around the fruit: translucent filled square with folded corner tabs behind the
// fruit, then a solid rim + diagonal gloss in front — the fruit stays visible inside the wrapper.
function drawWrapperBack(ctx: CanvasRenderingContext2D, x: number, y: number, r: number): void {
  const half = r * 1.02;
  ctx.save();
  // Folded corner tabs poking out diagonally from under the square.
  ctx.fillStyle = 'rgba(255, 190, 40, 0.9)';
  for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    ctx.beginPath();
    ctx.moveTo(x + dx * half * 0.62, y + dy * half * 0.62);
    ctx.lineTo(x + dx * half * 1.18, y + dy * half * 0.86);
    ctx.lineTo(x + dx * half * 0.86, y + dy * half * 1.18);
    ctx.closePath();
    ctx.fill();
  }
  roundRect(ctx, x - half, y - half, half * 2, half * 2, r * 0.22);
  const g = ctx.createLinearGradient(x - half, y - half, x + half, y + half);
  g.addColorStop(0, 'rgba(255, 224, 130, 0.85)');
  g.addColorStop(1, 'rgba(255, 176, 32, 0.85)');
  ctx.fillStyle = g;
  ctx.fill();
  ctx.restore();
}

function drawWrapperFront(ctx: CanvasRenderingContext2D, x: number, y: number, r: number): void {
  const half = r * 1.02;
  ctx.save();
  // Diagonal gloss band across the wrapper, over the fruit — reads as translucent plastic.
  roundRect(ctx, x - half, y - half, half * 2, half * 2, r * 0.22);
  ctx.clip();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.28)';
  ctx.beginPath();
  ctx.moveTo(x - half, y - half * 0.1);
  ctx.lineTo(x - half * 0.1, y - half);
  ctx.lineTo(x + half * 0.45, y - half);
  ctx.lineTo(x - half, y + half * 0.45);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  // Solid wrapper rim.
  roundRect(ctx, x - half, y - half, half * 2, half * 2, r * 0.22);
  ctx.strokeStyle = '#e8a20c';
  ctx.lineWidth = r * 0.12;
  ctx.stroke();
  roundRect(ctx, x - half * 0.92, y - half * 0.92, half * 1.84, half * 1.84, r * 0.16);
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = r * 0.045;
  ctx.stroke();
}

const SPRINKLES: ReadonlyArray<readonly [number, number, string]> = [
  [-0.35, -0.2, '#e74c3c'], [0.3, -0.35, '#f1c40f'], [0.1, 0.15, '#2ecc71'],
  [-0.15, 0.4, '#3498db'], [0.42, 0.25, '#e67e22'], [-0.45, 0.15, '#9b59b6'], [0.05, -0.45, '#1abc9c'],
];

function drawBomb(ctx: CanvasRenderingContext2D, x: number, y: number, r: number): void {
  const g = ctx.createRadialGradient(x - r * 0.25, y - r * 0.3, r * 0.1, x, y, r * 0.95);
  g.addColorStop(0, '#6b4a86');
  g.addColorStop(0.6, '#43265c');
  g.addColorStop(1, '#2a1440');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(x, y, r * 0.85, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.5)';
  ctx.lineWidth = 2;
  ctx.stroke();
  for (const [sx, sy, color] of SPRINKLES) {
    ctx.save();
    ctx.translate(x + sx * r, y + sy * r);
    ctx.rotate((sx + sy) * 4);
    ctx.fillStyle = color;
    ctx.fillRect(-r * 0.1, -r * 0.035, r * 0.2, r * 0.07);
    ctx.restore();
  }
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.beginPath();
  ctx.arc(x - r * 0.3, y - r * 0.35, r * 0.14, 0, Math.PI * 2);
  ctx.fill();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, radius: number): void {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}
