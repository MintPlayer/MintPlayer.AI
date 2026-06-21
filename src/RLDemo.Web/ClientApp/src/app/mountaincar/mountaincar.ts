import { Component, DestroyRef, ElementRef, afterNextRender, inject, signal, viewChild } from '@angular/core';
import { MountainCarApi, MountainCarStatus } from './mountaincar-api';
import { GOAL, MAX_POS, MIN_POS, MountainCarGame } from './mountaincar-logic';

/**
 * MountainCar page (PRD §7.1): **Watch AI** = principle B (backend drives the episode over a WebSocket,
 * this component renders frames), **Drive yourself** = client-side physics on a JS timer with ←/→.
 */
@Component({
  selector: 'app-mountaincar',
  templateUrl: './mountaincar.html',
  styleUrl: './mountaincar.scss',
  host: { '(window:keydown)': 'onKeyDown($event)', '(window:keyup)': 'onKeyUp($event)' },
})
export class MountainCar {
  private readonly api = inject(MountainCarApi);
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('mcCanvas');

  protected readonly mode = signal<'idle' | 'watch' | 'human'>('idle');
  protected readonly status = signal('Watch the PPO agent swing its way up, or drive it yourself.');
  protected readonly modelStatus = signal<MountainCarStatus | null>(null);

  private ctx: CanvasRenderingContext2D | null = null;
  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private game: MountainCarGame | null = null;
  private held = 1; // current human push: 0 left, 1 none, 2 right

  constructor() {
    this.pollStatus();
    afterNextRender(() => { this.ctx = this.canvasRef().nativeElement.getContext('2d'); this.draw(-0.5); });
    inject(DestroyRef).onDestroy(() => this.stop());
  }

  protected async watchAi(): Promise<void> {
    this.stop();
    const st = await this.api.status().catch(() => null);
    this.modelStatus.set(st);
    if (!st || st.status !== 'ready') {
      this.status.set(st?.status === 'loading' ? 'The agent is still loading — try again shortly.' : 'AI unavailable.');
      return;
    }
    this.mode.set('watch');
    this.status.set('Watching the PPO agent (the server drives the car)…');
    this.socket = this.api.connectLive(
      f => {
        this.draw(f.position);
        if (f.done) this.status.set(f.position >= GOAL ? 'Reached the flag! Restarting…' : 'Ran out of time — restarting…');
      },
      () => { if (this.mode() === 'watch') this.status.set('Stream closed.'); });
  }

  protected playHuman(): void {
    this.stop();
    this.mode.set('human');
    this.held = 1;
    this.status.set('Your turn — hold ← / → to push. Rock back and forth to build momentum!');
    const g = this.game = new MountainCarGame();
    this.draw(g.position);
    this.timer = setInterval(() => {
      g.step(this.held);
      this.draw(g.position);
      if (g.done) {
        this.status.set(g.reachedGoal ? `You made it in ${g.steps} steps!` : 'Out of time — try rocking back and forth first.');
        this.clearTimer();
      }
    }, 45);
  }

  protected onKeyDown(event: KeyboardEvent): void {
    if (this.mode() !== 'human') return;
    if (event.key === 'ArrowLeft' || event.key === 'a') { this.held = 0; event.preventDefault(); }
    else if (event.key === 'ArrowRight' || event.key === 'd') { this.held = 2; event.preventDefault(); }
  }

  protected onKeyUp(event: KeyboardEvent): void {
    if (this.mode() !== 'human') return;
    if (['ArrowLeft', 'ArrowRight', 'a', 'd'].includes(event.key)) this.held = 1;
  }

  protected stop(): void {
    this.socket?.close();
    this.socket = null;
    this.clearTimer();
    this.game = null;
    if (this.mode() !== 'idle') this.mode.set('idle');
  }

  private clearTimer(): void {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }

  private draw(position: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const cv = this.canvasRef().nativeElement;
    const w = cv.width, h = cv.height;
    const x2px = (x: number) => (x - MIN_POS) / (MAX_POS - MIN_POS) * w;
    const y2py = (y: number) => h - ((y + 1) / 2) * (h * 0.72) - h * 0.12; // y = sin(3x) ∈ [-1,1]
    const hill = (x: number) => Math.sin(3 * x);

    ctx.fillStyle = '#141823';
    ctx.fillRect(0, 0, w, h);

    // Hill curve.
    ctx.beginPath();
    ctx.strokeStyle = '#5b6678';
    ctx.lineWidth = 2;
    for (let px = 0; px <= w; px++) {
      const x = MIN_POS + (px / w) * (MAX_POS - MIN_POS);
      const py = y2py(hill(x));
      if (px === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Goal flag.
    const gx = x2px(GOAL), gy = y2py(hill(GOAL));
    ctx.strokeStyle = '#ffd479';
    ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(gx, gy - 34); ctx.stroke();
    ctx.fillStyle = '#ffd479';
    ctx.beginPath(); ctx.moveTo(gx, gy - 34); ctx.lineTo(gx + 16, gy - 28); ctx.lineTo(gx, gy - 22); ctx.fill();

    // Car.
    ctx.fillStyle = '#7cffb2';
    ctx.beginPath();
    ctx.arc(x2px(position), y2py(hill(position)) - 7, 7, 0, Math.PI * 2);
    ctx.fill();
  }

  private pollStatus(): void {
    void (async () => {
      try {
        const s = await this.api.status();
        this.modelStatus.set(s);
        if (s.status === 'loading') setTimeout(() => this.pollStatus(), 3000);
      } catch {
        // backend unreachable — leave status unknown
      }
    })();
  }
}
