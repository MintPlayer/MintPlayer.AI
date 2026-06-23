/**
 * Draws each of the 11 fruit as stylized vector art (cherry … watermelon) — recognizable shapes
 * with shading, stems, leaves, seeds, stripes. Asset-free: pure Canvas 2D, a faithful port of the
 * original SkiaSharp art.
 *
 * Each fruit is rendered once into a cached 256px offscreen canvas and then blitted scaled, so the
 * detail is a one-time cost and per-frame drawing is just a textured quad — cheap even with 100
 * fruit on screen. Fruit keep natural colors (a cherry is red regardless of skin); the theme
 * controls the backdrop, not the fruit.
 */

const REF = 256;
const CX = 128;
const CY = 128;
const R = 124; // fill the collider

const cache = new Map<number, HTMLCanvasElement>();

/** Draw fruit `tier` centered at (cx,cy) filling a circle of `radius` px, at `alpha` (0..255). */
export function drawFruit(
  ctx: CanvasRenderingContext2D,
  tier: number,
  cx: number,
  cy: number,
  radius: number,
  alpha = 255,
): void {
  const img = image(tier);
  if (alpha === 255) {
    ctx.drawImage(img, cx - radius, cy - radius, radius * 2, radius * 2);
  } else {
    const prev = ctx.globalAlpha;
    ctx.globalAlpha = alpha / 255;
    ctx.drawImage(img, cx - radius, cy - radius, radius * 2, radius * 2);
    ctx.globalAlpha = prev;
  }
}

function image(tier: number): HTMLCanvasElement {
  const cached = cache.get(tier);
  if (cached) return cached;
  const canvas = document.createElement('canvas');
  canvas.width = REF;
  canvas.height = REF;
  const ctx = canvas.getContext('2d')!;
  render(ctx, tier);
  cache.set(tier, canvas);
  return canvas;
}

function render(c: CanvasRenderingContext2D, tier: number): void {
  switch (tier) {
    case 1: cherry(c); break;
    case 2: strawberry(c); break;
    case 3: grape(c); break;
    case 4: dekopon(c); break;
    case 5: persimmon(c); break;
    case 6: apple(c); break;
    case 7: pear(c); break;
    case 8: peach(c); break;
    case 9: pineapple(c); break;
    case 10: melon(c); break;
    case 11: watermelon(c); break;
  }
}

// ── fruit ──────────────────────────────────────────────────────────────────────────────────

function cherry(c: CanvasRenderingContext2D): void {
  const red = 0xd32f2f;
  const ax = CX + R * 0.08;
  const ay = CY - R * 0.78;
  stroke(c, 0x5d4037, R * 0.05, () => {
    c.beginPath(); c.moveTo(ax, ay); c.quadraticCurveTo(CX - R * 0.3, CY - R * 0.25, CX - R * 0.42, CY + R * 0.05); c.stroke();
    c.beginPath(); c.moveTo(ax, ay); c.quadraticCurveTo(CX + R * 0.4, CY - R * 0.25, CX + R * 0.46, CY + R * 0.12); c.stroke();
  });
  leaf(c, ax, ay - R * 0.02, R * 0.34, -30);
  sphere(c, CX - R * 0.45, CY + R * 0.22, R * 0.54, red);
  sphere(c, CX + R * 0.45, CY + R * 0.2, R * 0.54, red);
}

function strawberry(c: CanvasRenderingContext2D): void {
  const red = 0xe5393b;
  c.beginPath();
  c.moveTo(CX - R * 0.72, CY - R * 0.22);
  c.bezierCurveTo(CX - R * 0.82, CY - R * 0.78, CX + R * 0.82, CY - R * 0.78, CX + R * 0.72, CY - R * 0.22);
  c.bezierCurveTo(CX + R * 0.55, CY + R * 0.48, CX + R * 0.2, CY + R * 0.86, CX, CY + R * 0.93);
  c.bezierCurveTo(CX - R * 0.2, CY + R * 0.86, CX - R * 0.55, CY + R * 0.48, CX - R * 0.72, CY - R * 0.22);
  c.closePath();
  c.fillStyle = radial(c, CX - R * 0.22, CY - R * 0.3, R * 1.35, lighten(red, 0.4), red, darken(red, 0.32));
  c.fill();

  c.fillStyle = rgb(0xffe082);
  for (let row = 0; row < 4; row++)
    for (let i = 0; i <= row; i++) {
      const sy = CY - R * 0.1 + row * R * 0.24;
      const spread = R * (0.12 + row * 0.14);
      const sx = CX + (i - row / 2) * spread;
      c.save(); c.translate(sx, sy); c.rotate(deg(18));
      ellipse(c, 0, 0, R * 0.05, R * 0.085); c.fill();
      c.restore();
    }

  calyx(c, CX, CY - R * 0.45, R * 0.55, 0x388e3c);
}

function grape(c: CanvasRenderingContext2D): void {
  const g = 0x7b3fb0;
  stroke(c, 0x5d4037, R * 0.05, () => {
    c.beginPath(); c.moveTo(CX, CY - R * 0.9); c.lineTo(CX, CY - R * 0.5); c.stroke();
  });
  leaf(c, CX + R * 0.05, CY - R * 0.78, R * 0.34, 25);
  const cluster: Array<[number, number]> = [
    [0, -0.5], [-0.3, -0.24], [0.3, -0.24], [0, -0.04],
    [-0.34, 0.16], [0.34, 0.16], [0, 0.3], [-0.2, 0.46], [0.2, 0.46], [0, 0.6],
  ];
  for (const [dx, dy] of cluster) sphere(c, CX + dx * R, CY + dy * R, R * 0.3, g);
}

function dekopon(c: CanvasRenderingContext2D): void {
  sphere(c, CX, CY, R * 0.95, 0xf57c00);
  sphere(c, CX, CY - R * 0.84, R * 0.16, 0xe65100); // top nub
  leaf(c, CX + R * 0.2, CY - R * 0.82, R * 0.28, 20);
}

function persimmon(c: CanvasRenderingContext2D): void {
  sphere(c, CX, CY, R * 0.95, 0xef6c00);
  calyx(c, CX, CY - R * 0.62, R * 0.55, 0x6d8c2a);
}

function apple(c: CanvasRenderingContext2D): void {
  stroke(c, 0x5d4037, R * 0.06, () => {
    c.beginPath(); c.moveTo(CX, CY - R * 0.5); c.quadraticCurveTo(CX + R * 0.06, CY - R * 0.82, CX + R * 0.02, CY - R * 0.92); c.stroke();
  });
  leaf(c, CX + R * 0.22, CY - R * 0.62, R * 0.34, 35);
  sphere(c, CX - R * 0.22, CY + R * 0.05, R * 0.72, 0xe53935);
  sphere(c, CX + R * 0.22, CY + R * 0.05, R * 0.72, 0xe53935);
}

function pear(c: CanvasRenderingContext2D): void {
  const p = 0xc0ca33;
  stroke(c, 0x5d4037, R * 0.05, () => {
    c.beginPath(); c.moveTo(CX, CY - R * 0.62); c.lineTo(CX, CY - R * 0.84); c.stroke();
  });
  sphere(c, CX, CY + R * 0.3, R * 0.64, p); // wide bottom
  sphere(c, CX, CY - R * 0.34, R * 0.42, p); // narrow top
}

function peach(c: CanvasRenderingContext2D): void {
  const p = 0xff8a65;
  leaf(c, CX + R * 0.1, CY - R * 0.8, R * 0.32, 20);
  sphere(c, CX - R * 0.2, CY, R * 0.74, p);
  sphere(c, CX + R * 0.2, CY, R * 0.74, p); // two lobes => cleft
}

function pineapple(c: CanvasRenderingContext2D): void {
  // crown
  c.fillStyle = rgb(0x4caf50);
  for (const a of [-36, -18, 0, 18, 36]) {
    c.save(); c.translate(CX, CY - R * 0.55); c.rotate(deg(a));
    c.beginPath(); c.moveTo(-R * 0.1, 0); c.lineTo(0, -R * 0.5); c.lineTo(R * 0.1, 0); c.closePath(); c.fill();
    c.restore();
  }
  // body
  const body = 0xf9a825;
  c.beginPath(); ellipse(c, CX, CY + R * 0.22, R * 0.62, R * 0.74);
  c.fillStyle = radial(c, CX - R * 0.2, CY - R * 0.1, R * 1.2, lighten(body, 0.35), body, darken(body, 0.3));
  c.fill();
  // crosshatch, clipped to body
  c.save();
  c.beginPath(); ellipse(c, CX, CY + R * 0.22, R * 0.62, R * 0.74); c.clip();
  c.strokeStyle = rgb(0x000000, 0x66); c.lineWidth = R * 0.03; c.lineCap = 'round';
  for (let t = -1.4; t <= 1.4 + 1e-6; t += 0.28) {
    c.beginPath(); c.moveTo(CX - R, CY + t * R); c.lineTo(CX + R, CY + t * R + R * 0.7); c.stroke();
    c.beginPath(); c.moveTo(CX - R, CY + t * R); c.lineTo(CX + R, CY + t * R - R * 0.7); c.stroke();
  }
  c.restore();
}

function melon(c: CanvasRenderingContext2D): void {
  sphere(c, CX, CY, R * 0.97, 0x9ccc65);
  c.save();
  c.beginPath(); c.arc(CX, CY, R * 0.97, 0, Math.PI * 2); c.clip();
  c.strokeStyle = rgb(0xefebcd, 0x80); c.lineWidth = R * 0.022; c.lineCap = 'round';
  for (let x = -1; x <= 1 + 1e-6; x += 0.33) { c.beginPath(); c.moveTo(CX + x * R, CY - R); c.lineTo(CX + x * R, CY + R); c.stroke(); }
  for (let y = -1; y <= 1 + 1e-6; y += 0.33) { c.beginPath(); c.moveTo(CX - R, CY + y * R); c.lineTo(CX + R, CY + y * R); c.stroke(); }
  c.restore();
}

function watermelon(c: CanvasRenderingContext2D): void {
  sphere(c, CX, CY, R * 0.97, 0x2e7d32);
  c.save();
  c.beginPath(); c.arc(CX, CY, R * 0.97, 0, Math.PI * 2); c.clip();
  c.strokeStyle = rgb(0x1b5e20); c.lineWidth = R * 0.12; c.lineCap = 'round';
  for (let x = -0.8; x <= 0.8 + 1e-6; x += 0.4) {
    c.beginPath(); c.moveTo(CX + x * R, CY - R); c.quadraticCurveTo(CX + x * R * 1.5, CY, CX + x * R, CY + R); c.stroke();
  }
  c.restore();
}

// ── helpers ───────────────────────────────────────────────────────────────────────────────

function deg(d: number): number {
  return (d * Math.PI) / 180;
}

/** A 24-bit 0xRRGGBB (plus optional 0..255 alpha) as a CSS color. */
function rgb(hex: number, alpha = 0xff): string {
  const r = (hex >>> 16) & 0xff;
  const g = (hex >>> 8) & 0xff;
  const b = hex & 0xff;
  return `rgba(${r},${g},${b},${(alpha / 255).toFixed(3)})`;
}

function lighten(hex: number, a: number): number {
  const r = (hex >>> 16) & 0xff, g = (hex >>> 8) & 0xff, b = hex & 0xff;
  return (pack(r + (255 - r) * a) << 16) | (pack(g + (255 - g) * a) << 8) | pack(b + (255 - b) * a);
}

function darken(hex: number, a: number): number {
  const r = (hex >>> 16) & 0xff, g = (hex >>> 8) & 0xff, b = hex & 0xff;
  return (pack(r * (1 - a)) << 16) | (pack(g * (1 - a)) << 8) | pack(b * (1 - a));
}

function pack(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function radial(c: CanvasRenderingContext2D, x: number, y: number, rad: number, a: number, b: number, d: number): CanvasGradient {
  const grad = c.createRadialGradient(x, y, 0, x, y, rad);
  grad.addColorStop(0, rgb(a));
  grad.addColorStop(0.5, rgb(b));
  grad.addColorStop(1, rgb(d));
  return grad;
}

function ellipse(c: CanvasRenderingContext2D, cx: number, cy: number, rx: number, ry: number): void {
  c.beginPath();
  c.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
}

function stroke(c: CanvasRenderingContext2D, hex: number, width: number, draw: () => void): void {
  c.strokeStyle = rgb(hex);
  c.lineWidth = width;
  c.lineCap = 'round';
  draw();
}

function sphere(c: CanvasRenderingContext2D, cx: number, cy: number, r: number, col: number): void {
  c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2);
  c.fillStyle = radial(c, cx - r * 0.3, cy - r * 0.35, r * 1.3, lighten(col, 0.45), col, darken(col, 0.32));
  c.fill();
  c.beginPath(); c.arc(cx - r * 0.32, cy - r * 0.38, r * 0.17, 0, Math.PI * 2);
  c.fillStyle = rgb(0xffffff, 90);
  c.fill();
}

function leaf(c: CanvasRenderingContext2D, x: number, y: number, len: number, degrees: number): void {
  c.save(); c.translate(x, y); c.rotate(deg(degrees));
  c.fillStyle = rgb(0x43a047);
  ellipse(c, 0, -len * 0.5, len * 0.28, len * 0.5); c.fill();
  c.restore();
}

function calyx(c: CanvasRenderingContext2D, x: number, y: number, size: number, color: number): void {
  c.fillStyle = rgb(color);
  for (const a of [-60, -20, 20, 60, 0]) {
    c.save(); c.translate(x, y); c.rotate(deg(a));
    c.beginPath();
    c.moveTo(0, 0);
    c.quadraticCurveTo(-size * 0.18, -size * 0.5, 0, -size);
    c.quadraticCurveTo(size * 0.18, -size * 0.5, 0, 0);
    c.closePath();
    c.fill();
    c.restore();
  }
}
