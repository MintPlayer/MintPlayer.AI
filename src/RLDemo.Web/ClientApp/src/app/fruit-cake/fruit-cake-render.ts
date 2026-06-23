import { drawFruit } from './fruit-cake-art';
import { PARTICLE_LIFE_SECONDS, POPUP_LIFE_SECONDS } from './fruit-cake-effects';
import { byTier, cssColor } from './fruit-cake-fruits';
import { FruitCakeGame, GamePhase } from './fruit-cake-game';
import { FruitWorld } from './fruit-cake-physics';

/**
 * Draws a {@link FruitCakeGame} to a 2D canvas — a faithful port of the original SkiaSharp
 * renderer. Works in CSS pixels (the caller pre-scales the context by devicePixelRatio). The
 * container is fit into the surface with a letterbox transform that {@link surfaceToContainerX}
 * inverts, so the host can map a pointer back to a drop position.
 */

const WALL = 0xff3d1f5c;
const HUD = 0xfff0e6cc;
const DIM = 0xffa89980;
const DANGER = 0xffe8503a;

const POP_SECONDS = 0.15;
const BUTTON_SIZE = 34;
const BUTTON_GAP = 8;
const FONT = 'system-ui, sans-serif';

/** Top-right toolbar buttons, hit-tested in surface (CSS) pixels. */
export enum HudButton {
  None,
  Mute,
  Labels,
  Theme,
  Music,
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function buttonRect(index: number, surfaceWidth: number): Rect {
  const x = surfaceWidth - (index + 1) * (BUTTON_SIZE + BUTTON_GAP);
  return { x, y: BUTTON_GAP, w: BUTTON_SIZE, h: BUTTON_SIZE };
}

function inRect(r: Rect, x: number, y: number): boolean {
  return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
}

/** Which toolbar button (if any) a tap at the given surface point hit. */
export function hitTest(sx: number, sy: number, surfaceWidth: number): HudButton {
  if (inRect(buttonRect(0, surfaceWidth), sx, sy)) return HudButton.Mute;
  if (inRect(buttonRect(1, surfaceWidth), sx, sy)) return HudButton.Labels;
  if (inRect(buttonRect(2, surfaceWidth), sx, sy)) return HudButton.Theme;
  if (inRect(buttonRect(3, surfaceWidth), sx, sy)) return HudButton.Music;
  return HudButton.None;
}

function easeOutBack(t: number): number {
  t = Math.min(1, Math.max(0, t));
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

function fitScale(w: number, h: number): number {
  return Math.min(w / FruitWorld.ContainerWidthPx, h / FruitWorld.ContainerHeightPx);
}

/** Map a surface-X (CSS px) back to a container-X — the inverse of the draw transform. */
export function surfaceToContainerX(surfaceX: number, surfaceWidth: number, surfaceHeight: number): number {
  const scale = fitScale(surfaceWidth, surfaceHeight);
  const offsetX = (surfaceWidth - FruitWorld.ContainerWidthPx * scale) / 2;
  return (surfaceX - offsetX) / scale;
}

export function render(ctx: CanvasRenderingContext2D, game: FruitCakeGame, surfaceWidth: number, surfaceHeight: number): void {
  const theme = game.theme;
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = cssColor(theme.background);
  ctx.fillRect(0, 0, surfaceWidth, surfaceHeight);

  const scale = fitScale(surfaceWidth, surfaceHeight);
  // Screen shake offsets the play area only (not the input mapping, which stays exact).
  const offsetX = (surfaceWidth - FruitWorld.ContainerWidthPx * scale) / 2 + game.effects.shakeDx;
  const offsetY = (surfaceHeight - FruitWorld.ContainerHeightPx * scale) / 2 + game.effects.shakeDy;

  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  drawDangerLine(ctx, game);
  drawHeldFruitAndGuide(ctx, game);

  for (const f of game.world.fruits) {
    const def = byTier(f.tier);
    // Merge-born fruit pop in: scale 0→1 with an ease-out-back overshoot.
    const popScale = f.mergeBorn && f.ageSeconds < POP_SECONDS ? easeOutBack(f.ageSeconds / POP_SECONDS) : 1;
    drawFruit(ctx, f.tier, f.xPx, f.yPx, def.radiusPx * popScale);
    if (game.colorblindLabels) drawTierLabel(ctx, f.tier, f.xPx, f.yPx, def.radiusPx);
  }

  drawParticles(ctx, game);

  ctx.strokeStyle = cssColor(theme.wall);
  ctx.lineWidth = 6;
  ctx.strokeRect(0, 0, FruitWorld.ContainerWidthPx, FruitWorld.ContainerHeightPx);

  drawNextPreview(ctx, game);
  drawScorePopups(ctx, game);
  ctx.restore();

  drawHud(ctx, game);
  drawButtons(ctx, game, surfaceWidth);

  const flash = game.effects.flashAlpha;
  if (flash > 0) {
    ctx.fillStyle = cssColor(0xffffffff, flash * 180);
    ctx.fillRect(0, 0, surfaceWidth, surfaceHeight);
  }

  if (game.phase === GamePhase.GameOver) drawGameOver(ctx, game, surfaceWidth, surfaceHeight);
}

function drawDangerLine(ctx: CanvasRenderingContext2D, game: FruitCakeGame): void {
  ctx.strokeStyle = game.dangerActive ? cssColor(DANGER) : cssColor(DANGER, 110);
  ctx.lineWidth = game.dangerActive ? 4 : 2;
  ctx.beginPath();
  ctx.moveTo(0, FruitWorld.DangerLineYPx);
  ctx.lineTo(FruitWorld.ContainerWidthPx, FruitWorld.DangerLineYPx);
  ctx.stroke();
}

function drawHeldFruitAndGuide(ctx: CanvasRenderingContext2D, game: FruitCakeGame): void {
  if (game.phase !== GamePhase.Playing) return;

  const def = byTier(game.currentTier);
  const x = game.aimXPx;
  const y = FruitCakeGame.heldYPx(game.currentTier);

  // Faint vertical drop guide (straight down, not a trajectory).
  ctx.strokeStyle = cssColor(HUD, 60);
  ctx.lineWidth = 1.5;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, FruitWorld.ContainerHeightPx);
  ctx.stroke();
  ctx.setLineDash([]);

  // The held fruit (dimmed slightly during cooldown).
  drawFruit(ctx, game.currentTier, x, y, def.radiusPx, game.cooldownRemaining > 0 ? 120 : 255);
}

function drawTierLabel(ctx: CanvasRenderingContext2D, tier: number, cx: number, cy: number, radius: number): void {
  const size = Math.max(12, radius * 0.7);
  ctx.font = `${size}px ${FONT}`;
  ctx.textAlign = 'center';
  const baseline = cy + size * 0.35;
  const text = `${tier}`;
  // White numeral with a dark halo so it reads on any fruit color.
  ctx.lineWidth = Math.max(2, size * 0.14);
  ctx.strokeStyle = cssColor(0xff000000, 190);
  ctx.strokeText(text, cx, baseline);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, cx, baseline);
  ctx.textAlign = 'start';
}

function drawParticles(ctx: CanvasRenderingContext2D, game: FruitCakeGame): void {
  for (const p of game.effects.particles) {
    const t = p.age / PARTICLE_LIFE_SECONDS;
    ctx.fillStyle = cssColor(p.color, 255 * (1 - t));
    ctx.beginPath();
    ctx.arc(p.xPx, p.yPx, p.radiusPx, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawScorePopups(ctx: CanvasRenderingContext2D, game: FruitCakeGame): void {
  ctx.textAlign = 'center';
  for (const p of game.effects.popups) {
    const t = p.age / POPUP_LIFE_SECONDS;
    ctx.fillStyle = cssColor(HUD, 255 * (1 - t));
    // Larger merges read a touch bigger, without a distracting number shower.
    ctx.font = `${20 + Math.min(p.points, 66) * 0.25}px ${FONT}`;
    ctx.fillText(`+${p.points}`, p.xPx, p.yPx);
  }
  ctx.textAlign = 'start';
}

function drawNextPreview(ctx: CanvasRenderingContext2D, game: FruitCakeGame): void {
  const def = byTier(game.nextTier);
  const cx = FruitWorld.ContainerWidthPx - 70;
  const cy = 60;
  // Clamp the preview glyph so very large fruit don't dominate the corner.
  drawFruit(ctx, game.nextTier, cx, cy, Math.min(def.radiusPx, 40));
  ctx.fillStyle = cssColor(DIM);
  ctx.font = `20px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.fillText('NEXT', cx, cy - 48);
  ctx.textAlign = 'start';
}

function drawHud(ctx: CanvasRenderingContext2D, game: FruitCakeGame): void {
  ctx.fillStyle = cssColor(HUD);
  ctx.textAlign = 'left';
  ctx.font = `30px ${FONT}`;
  ctx.fillText(`${game.score}`, 16, 36);
  ctx.font = `18px ${FONT}`;
  ctx.fillText(`Best ${game.highScore}`, 16, 60);
}

function drawButtons(ctx: CanvasRenderingContext2D, game: FruitCakeGame, surfaceWidth: number): void {
  drawButton(ctx, buttonRect(0, surfaceWidth), game.muted ? '🔇' : '🔊', !game.muted); // sound effects
  drawButton(ctx, buttonRect(1, surfaceWidth), '🔢', game.colorblindLabels); // tier-number labels
  drawButton(ctx, buttonRect(2, surfaceWidth), '🎨', game.themeIndex > 0); // theme
  drawButton(ctx, buttonRect(3, surfaceWidth), '🎵', game.musicOn); // music
}

function drawButton(ctx: CanvasRenderingContext2D, rect: Rect, icon: string, active: boolean): void {
  ctx.fillStyle = cssColor(WALL, active ? 220 : 90);
  ctx.beginPath();
  ctx.roundRect(rect.x, rect.y, rect.w, rect.h, 8);
  ctx.fill();
  ctx.fillStyle = active ? cssColor(HUD) : cssColor(DIM); // applies to symbol glyphs; emoji keep their own color
  ctx.font = `${Math.round(rect.h * 0.56)}px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(icon, rect.x + rect.w / 2, rect.y + rect.h / 2 + 1);
  ctx.textAlign = 'start';
  ctx.textBaseline = 'alphabetic';
}

function drawGameOver(ctx: CanvasRenderingContext2D, game: FruitCakeGame, w: number, h: number): void {
  ctx.fillStyle = cssColor(0xff000000, 170);
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = cssColor(HUD);
  ctx.textAlign = 'center';
  const cx = w / 2;
  const cy = h / 2;
  ctx.font = `48px ${FONT}`;
  ctx.fillText('Game Over', cx, cy - 30);
  ctx.font = `24px ${FONT}`;
  ctx.fillText(`Score ${game.score}   ·   Best ${game.highScore}`, cx, cy + 14);
  ctx.fillText('tap to play again', cx, cy + 54);
  ctx.textAlign = 'start';
}
