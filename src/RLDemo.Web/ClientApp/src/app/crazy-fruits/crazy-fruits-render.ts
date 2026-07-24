// Canvas renderer for Crazy Fruits — the confirmed KidCity identity (cartoon fruit market stall: bunting
// flags, strawberry-red wordmark, saturated fruit art; original art, no KidCity/Belgacom branding). Pure
// view code: it reads the game's current animation step + progress and draws; all rules live in the engine.

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
  statusLine: string): void {
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

  ctx.restore();
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
      ctx.arc(x, y, CELL * 0.44, 0, Math.PI * 2);
      ctx.stroke();
    }
    drawFruit(ctx, grid[i], x, y, CELL * 0.36);
  }
}

function drawStep(ctx: CanvasRenderingContext2D, step: AnimStep, t: number): void {
  const ease = t * t * (3 - 2 * t); // smoothstep
  switch (step.kind) {
    case 'swap': {
      for (let i = 0; i < SIZE * SIZE; i++) {
        if (i === step.a || i === step.b || step.grid[i] === 0) continue;
        const { x, y } = cellCenter(i);
        drawFruit(ctx, step.grid[i], x, y, CELL * 0.36);
      }
      const pa = cellCenter(step.a);
      const pb = cellCenter(step.b);
      drawFruit(ctx, step.fruitB, pb.x + (pa.x - pb.x) * ease, pb.y + (pa.y - pb.y) * ease, CELL * 0.36);
      drawFruit(ctx, step.fruitA, pa.x + (pb.x - pa.x) * ease, pa.y + (pb.y - pa.y) * ease, CELL * 0.38);
      break;
    }
    case 'pop': {
      for (let i = 0; i < SIZE * SIZE; i++) {
        if (step.grid[i] === 0) continue;
        const { x, y } = cellCenter(i);
        if (step.cells[i]) {
          ctx.save();
          ctx.globalAlpha = 1 - ease;
          drawFruit(ctx, step.grid[i], x, y, CELL * 0.36 * (1 + 0.35 * ease));
          ctx.restore();
        } else {
          drawFruit(ctx, step.grid[i], x, y, CELL * 0.36);
        }
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
        drawFruit(ctx, step.grid[i], x, y, CELL * 0.36);
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
        drawFruit(ctx, m.fruit, x, fromY + (toY - fromY) * ease, CELL * 0.36);
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

// ── Fruit art (types 1..6: strawberry, banana, orange, grape, apple, lemon) ─────────────────────────────

export function drawFruit(ctx: CanvasRenderingContext2D, fruit: number, x: number, y: number, r: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.lineJoin = 'round';
  switch (fruit) {
    case 1: strawberry(ctx, r); break;
    case 2: banana(ctx, r); break;
    case 3: orange(ctx, r); break;
    case 4: grapes(ctx, r); break;
    case 5: apple(ctx, r); break;
    case 6: lemon(ctx, r); break;
  }
  ctx.restore();
}

function outline(ctx: CanvasRenderingContext2D): void {
  ctx.strokeStyle = 'rgba(0,0,0,0.45)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function strawberry(ctx: CanvasRenderingContext2D, r: number): void {
  ctx.beginPath();
  ctx.moveTo(0, r);
  ctx.bezierCurveTo(-r * 1.05, r * 0.25, -r * 0.85, -r * 0.8, 0, -r * 0.65);
  ctx.bezierCurveTo(r * 0.85, -r * 0.8, r * 1.05, r * 0.25, 0, r);
  ctx.fillStyle = '#e63946';
  ctx.fill();
  outline(ctx);
  ctx.fillStyle = '#ffd166';
  for (const [sx, sy] of [[-0.4, 0], [0, 0.15], [0.4, 0], [-0.2, 0.45], [0.2, 0.45], [0, -0.25]] as const) {
    ctx.beginPath();
    ctx.ellipse(sx * r, sy * r, r * 0.06, r * 0.09, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = '#2a9d34';
  ctx.beginPath();
  ctx.moveTo(0, -r * 0.6);
  for (let i = 0; i < 5; i++) {
    const a = -Math.PI / 2 + (i - 2) * 0.55;
    ctx.lineTo(Math.cos(a) * r * 0.55, -r * 0.6 + Math.sin(a) * r * 0.35 + r * 0.18);
    ctx.lineTo(Math.cos(a + 0.27) * r * 0.25, -r * 0.62);
  }
  ctx.closePath();
  ctx.fill();
}

function banana(ctx: CanvasRenderingContext2D, r: number): void {
  ctx.beginPath();
  ctx.moveTo(-r * 0.85, -r * 0.5);
  ctx.quadraticCurveTo(0, r * 1.05, r * 0.85, -r * 0.5);
  ctx.quadraticCurveTo(r * 0.7, -r * 0.2, r * 0.6, -r * 0.28);
  ctx.quadraticCurveTo(0, r * 0.55, -r * 0.6, -r * 0.28);
  ctx.quadraticCurveTo(-r * 0.7, -r * 0.2, -r * 0.85, -r * 0.5);
  ctx.fillStyle = '#ffd23f';
  ctx.fill();
  outline(ctx);
  ctx.fillStyle = '#8d6a1f';
  ctx.beginPath();
  ctx.arc(-r * 0.82, -r * 0.52, r * 0.09, 0, Math.PI * 2);
  ctx.arc(r * 0.82, -r * 0.52, r * 0.09, 0, Math.PI * 2);
  ctx.fill();
}

function orange(ctx: CanvasRenderingContext2D, r: number): void {
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.85, 0, Math.PI * 2);
  ctx.fillStyle = '#f77f00';
  ctx.fill();
  outline(ctx);
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.beginPath();
  ctx.arc(-r * 0.3, -r * 0.3, r * 0.22, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#2a9d34';
  ctx.beginPath();
  ctx.ellipse(r * 0.25, -r * 0.75, r * 0.28, r * 0.13, -0.5, 0, Math.PI * 2);
  ctx.fill();
}

function grapes(ctx: CanvasRenderingContext2D, r: number): void {
  ctx.fillStyle = '#7b2d8b';
  for (const [gx, gy, gr] of [[-0.35, -0.15, 0.38], [0.35, -0.15, 0.38], [0, 0.35, 0.4], [0, -0.45, 0.36]] as const) {
    ctx.beginPath();
    ctx.arc(gx * r, gy * r, gr * r, 0, Math.PI * 2);
    ctx.fill();
    outline(ctx);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.beginPath();
  ctx.arc(-r * 0.15, 0.15 * r, r * 0.1, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#5c8a2e';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, -r * 0.75);
  ctx.quadraticCurveTo(r * 0.15, -r * 0.95, r * 0.3, -r * 1.0);
  ctx.stroke();
}

function apple(ctx: CanvasRenderingContext2D, r: number): void {
  ctx.beginPath();
  ctx.arc(-r * 0.3, r * 0.05, r * 0.62, 0, Math.PI * 2);
  ctx.arc(r * 0.3, r * 0.05, r * 0.62, 0, Math.PI * 2);
  ctx.fillStyle = '#6fbf3f';
  ctx.fill();
  outline(ctx);
  ctx.strokeStyle = '#5d3a1a';
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.moveTo(0, -r * 0.45);
  ctx.quadraticCurveTo(r * 0.1, -r * 0.85, r * 0.05, -r * 1.0);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.beginPath();
  ctx.arc(-r * 0.45, -r * 0.2, r * 0.16, 0, Math.PI * 2);
  ctx.fill();
}

function lemon(ctx: CanvasRenderingContext2D, r: number): void {
  ctx.beginPath();
  ctx.ellipse(0, 0, r * 0.95, r * 0.62, -0.35, 0, Math.PI * 2);
  ctx.fillStyle = '#ffe74c';
  ctx.fill();
  outline(ctx);
  ctx.fillStyle = '#ffe74c';
  for (const s of [-1, 1] as const) {
    ctx.beginPath();
    ctx.ellipse(s * r * 0.88, -s * r * 0.35, r * 0.14, r * 0.1, -0.35, 0, Math.PI * 2);
    ctx.fill();
    outline(ctx);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.beginPath();
  ctx.arc(-r * 0.3, -r * 0.2, r * 0.15, 0, Math.PI * 2);
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
