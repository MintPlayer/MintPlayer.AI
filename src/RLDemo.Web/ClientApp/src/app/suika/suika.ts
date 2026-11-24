import { AfterViewInit, Component, DestroyRef, ElementRef, NgZone, inject, signal, viewChild } from '@angular/core';
import { Color } from '@mintplayer/ng-bootstrap';
import { BsButtonTypeDirective } from '@mintplayer/ng-bootstrap/button-type';
import { SuikaAudio } from './suika-audio';
import { SuikaDirector } from './suika-director';
import { SuikaGame, GamePhase } from './suika-game';
import { HudButton, hitTest, render, renderFrame, surfaceToContainerX } from './suika-render';
import { ScreenWakeLock } from '../screen-wake-lock';

/**
 * Suika — a drop-and-merge physics game, ported from the original .NET MAUI/Blazor
 * app to a self-contained client-side Angular game (physics, art, audio, persistence all run in the
 * browser; no backend, no AI). Move the pointer to aim, tap/click to drop; two same-tier fruit that
 * touch merge into the next tier. Don't let fruit settle above the danger line.
 *
 * The HUD toolbar (sound / tier-labels / theme / music) is drawn on the canvas and tapped like the
 * original; the Fullscreen button is HTML so it stays keyboard-accessible and fullscreens just the
 * game stage.
 */
@Component({
  selector: 'app-suika',
  templateUrl: './suika.html',
  styleUrl: './suika.scss',
  imports: [BsButtonTypeDirective],
  host: {
    '(document:fullscreenchange)': 'onFullscreenChange()',
    '(document:webkitfullscreenchange)': 'onFullscreenChange()',
  },
})
export class Suika implements AfterViewInit {
  private readonly stageRef = viewChild.required<ElementRef<HTMLDivElement>>('stage');
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('board');
  private readonly zone = inject(NgZone);

  private readonly wakeLock = inject(ScreenWakeLock);
  private readonly audio = new SuikaAudio();
  private readonly game = new SuikaGame(this.audio);

  protected readonly colors = Color;
  protected readonly isFullscreen = signal(false);
  /** 'human' = play locally; 'watch' = the AI plays — physics + net + search all run in the browser (M32). */
  protected readonly mode = signal<'human' | 'watch'>('human');
  // The client-side AI game (created on first watch; the trained net loads itself). No server, no WebSocket.
  private director: SuikaDirector | null = null;

  /** "New game" is two-step: the first tap arms it, a second tap within {@link RestartConfirmMs} restarts. */
  protected readonly restartArmed = signal(false);
  private restartDisarmTimer: ReturnType<typeof setTimeout> | undefined;
  private static readonly RestartConfirmMs = 3000;

  private ctx: CanvasRenderingContext2D | null = null;
  private rafId = 0;
  private lastMs = 0;
  private accumulator = 0;
  private static readonly Step = 1 / 60;

  constructor() {
    inject(DestroyRef).onDestroy(() => this.teardown());
  }

  ngAfterViewInit(): void {
    this.ctx = this.canvasRef().nativeElement.getContext('2d');
    // Run the render loop outside Angular: the game draws itself to the canvas, so per-frame change
    // detection would be pure waste.
    this.zone.runOutsideAngular(() => {
      this.rafId = requestAnimationFrame(this.frame);
    });
  }

  private teardown(): void {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    clearTimeout(this.restartDisarmTimer);
    this.director?.dispose(); // release the AI worker
    void this.wakeLock.release();
    this.game.saveSnapshot();
    this.audio.dispose();
  }

  // ── AI "Watch" mode — the whole AI runs client-side (M32); no server, no WebSocket. ──────────
  protected setMode(mode: 'human' | 'watch'): void {
    if (this.mode() === mode) return;
    this.lastMs = 0;
    if (mode === 'watch') {
      this.mode.set('watch');
      (this.director ??= new SuikaDirector()).reset(); // lazily create; the net loads itself
      void this.wakeLock.acquire(); // keep the phone screen on so the game isn't frozen by an auto-lock
    } else {
      this.mode.set('human');
      this.accumulator = 0;
      void this.wakeLock.release();
    }
  }

  protected onRestartClick(): void {
    clearTimeout(this.restartDisarmTimer);
    if (!this.restartArmed()) {
      this.restartArmed.set(true);
      this.restartDisarmTimer = setTimeout(() => this.restartArmed.set(false), Suika.RestartConfirmMs);
      return;
    }
    this.restartArmed.set(false);
    this.game.unlockAudio();
    this.game.restart();
    this.accumulator = 0;
  }

  private readonly frame = (nowMs: number): void => {
    const dt = this.lastMs ? Math.min(0.25, (nowMs - this.lastMs) / 1000) : 0; // clamp to avoid a spiral after a stall
    this.lastMs = nowMs;
    if (this.mode() === 'human') {
      this.accumulator += dt;
      while (this.accumulator >= Suika.Step) {
        this.game.step(Suika.Step);
        this.accumulator -= Suika.Step;
      }
    } else {
      this.director?.update(dt); // watch mode: the browser runs the AI game locally
    }
    this.draw();
    this.rafId = requestAnimationFrame(this.frame);
  };

  private draw(): void {
    const canvas = this.canvasRef().nativeElement;
    if (!this.ctx) return;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.mode() === 'watch') {
      if (this.director) renderFrame(this.ctx, this.director.toFrame(), this.game.themeIndex, cssW, cssH);
      else {
        this.ctx.fillStyle = '#1c0b2b';
        this.ctx.fillRect(0, 0, cssW, cssH); // loading…
      }
    } else {
      render(this.ctx, this.game, cssW, cssH);
    }
  }

  protected onPointerMove(event: PointerEvent): void {
    if (this.mode() !== 'human' || this.game.phase !== GamePhase.Playing) return;
    const { sx, w, h } = this.toSurface(event);
    this.game.aimTo(surfaceToContainerX(sx, w, h));
  }

  protected onPointerDown(event: PointerEvent): void {
    if (this.mode() !== 'human') return;
    this.game.unlockAudio(); // first gesture: the only safe moment to start opted-in music
    const { sx, sy, w, h } = this.toSurface(event);

    if (this.game.phase === GamePhase.GameOver) {
      this.game.restart();
      return;
    }

    const button = hitTest(sx, sy, w);
    if (button !== HudButton.None) {
      this.handleButton(button);
      return;
    }

    this.game.aimTo(surfaceToContainerX(sx, w, h));
    this.game.drop();
  }

  private handleButton(button: HudButton): void {
    switch (button) {
      case HudButton.Mute:
        this.game.toggleMute();
        break;
      case HudButton.Labels:
        this.game.toggleColorblindLabels();
        break;
      case HudButton.Theme:
        this.game.cycleTheme();
        break;
      case HudButton.Music:
        this.game.toggleMusic();
        break;
    }
  }

  private toSurface(event: PointerEvent): { sx: number; sy: number; w: number; h: number } {
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    return { sx: event.clientX - rect.left, sy: event.clientY - rect.top, w: rect.width, h: rect.height };
  }

  protected toggleFullscreen(): void {
    const el = this.stageRef().nativeElement as HTMLElement & {
      webkitRequestFullscreen?: () => Promise<void> | void;
      msRequestFullscreen?: () => Promise<void> | void;
    };
    const doc = document as Document & {
      webkitFullscreenElement?: Element | null;
      webkitExitFullscreen?: () => Promise<void> | void;
      msExitFullscreen?: () => Promise<void> | void;
    };
    try {
      if (document.fullscreenElement ?? doc.webkitFullscreenElement) {
        const exit = document.exitFullscreen ?? doc.webkitExitFullscreen ?? doc.msExitFullscreen;
        exit?.call(document);
      } else {
        // Vendor fallbacks so older/WebKit browsers don't silently no-op (which would look like a dead button).
        const request = el.requestFullscreen ?? el.webkitRequestFullscreen ?? el.msRequestFullscreen;
        if (!request) {
          console.warn('[Suika] Fullscreen API is not available in this browser.');
          return;
        }
        const result = request.call(el);
        if (result && typeof (result as Promise<void>).catch === 'function') {
          (result as Promise<void>).catch(err => console.warn('[Suika] Fullscreen request was blocked:', err));
        }
      }
    } catch (err) {
      console.warn('[Suika] Fullscreen toggle failed:', err);
    }
  }

  protected onFullscreenChange(): void {
    const doc = document as Document & { webkitFullscreenElement?: Element | null };
    this.isFullscreen.set((document.fullscreenElement ?? doc.webkitFullscreenElement) === this.stageRef().nativeElement);
  }
}
