import { Component, DestroyRef, ElementRef, afterNextRender, inject, signal, viewChild } from '@angular/core';
import { PendulumApi, PendulumStatus } from './pendulum-api';
import { MAX_TORQUE, PendulumGame } from './pendulum-logic';

/**
 * Pendulum page (PRD §7.1): **Watch AI** = principle B (backend drives the episode over a WebSocket, this
 * component renders frames), **Swing it yourself** = client-side physics on a JS timer with ←/→ applying a
 * continuous torque. The SDK's continuous-control showcase — the action is a real-valued torque, not a button.
 */
@Component({
  selector: 'app-pendulum',
  templateUrl: './pendulum.html',
  styleUrl: './pendulum.scss',
  host: { '(window:keydown)': 'onKeyDown($event)', '(window:keyup)': 'onKeyUp($event)' },
})
export class Pendulum {
  private readonly api = inject(PendulumApi);
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('penCanvas');

  protected readonly mode = signal<'idle' | 'watch' | 'human'>('idle');
  protected readonly status = signal('Watch the SAC agent swing the rod upright and balance it, or try it yourself.');
  protected readonly modelStatus = signal<PendulumStatus | null>(null);
  protected readonly torque = signal(0);

  private ctx: CanvasRenderingContext2D | null = null;
  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private game: PendulumGame | null = null;
  private held = 0; // current human torque: -MAX_TORQUE, 0, or +MAX_TORQUE

  constructor() {
    this.pollStatus();
    afterNextRender(() => { this.ctx = this.canvasRef().nativeElement.getContext('2d'); this.draw(-1, 0, 0); });
    inject(DestroyRef).onDestroy(() => this.stop());
  }

  protected async watchAi(): Promise<void> {
    this.stop();
    const st = await this.api.status().catch(() => null);
    this.modelStatus.set(st);
    if (!st || st.status !== 'ready') {
      this.status.set(st?.status === 'training' ? 'The agent is still training — try again shortly.' : 'AI unavailable.');
      return;
    }
    this.mode.set('watch');
    this.status.set('Watching the SAC agent (the server drives the rod)…');
    this.socket = this.api.connectLive(
      f => {
        this.torque.set(f.torque);
        this.draw(f.cosTheta, f.sinTheta, f.torque);
        if (f.done) this.status.set('Episode complete — restarting…');
      },
      () => { if (this.mode() === 'watch') this.status.set('Stream closed.'); });
  }

  protected playHuman(): void {
    this.stop();
    this.mode.set('human');
    this.held = 0;
    this.status.set('Your turn — hold ← / → to apply torque. Pump it to swing up, then ease off to balance!');
    const g = this.game = new PendulumGame();
    this.draw(Math.cos(g.theta), Math.sin(g.theta), 0);
    this.timer = setInterval(() => {
      g.step(this.held);
      this.torque.set(this.held);
      this.draw(Math.cos(g.theta), Math.sin(g.theta), this.held);
      if (g.done) {
        this.status.set(`Time up — best balance held: ${(g.upright * 100).toFixed(0)}% upright.`);
        this.clearTimer();
      }
    }, 50);
  }

  protected onKeyDown(event: KeyboardEvent): void {
    if (this.mode() !== 'human') return;
    if (event.key === 'ArrowLeft' || event.key === 'a') { this.held = -MAX_TORQUE; event.preventDefault(); }
    else if (event.key === 'ArrowRight' || event.key === 'd') { this.held = MAX_TORQUE; event.preventDefault(); }
  }

  protected onKeyUp(event: KeyboardEvent): void {
    if (this.mode() !== 'human') return;
    if (['ArrowLeft', 'ArrowRight', 'a', 'd'].includes(event.key)) this.held = 0;
  }

  protected stop(): void {
    this.socket?.close();
    this.socket = null;
    this.clearTimer();
    this.game = null;
    this.torque.set(0);
    if (this.mode() !== 'idle') this.mode.set('idle');
  }

  private clearTimer(): void {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }

  // cosTheta/sinTheta describe the rod angle (θ=0 is straight up); torque tints the applied-effort arc.
  private draw(cosTheta: number, sinTheta: number, torque: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const cv = this.canvasRef().nativeElement;
    const w = cv.width, h = cv.height;
    const cx = w / 2, cy = h / 2;
    const rodLen = Math.min(w, h) * 0.34;

    ctx.fillStyle = '#141823';
    ctx.fillRect(0, 0, w, h);

    // Reference circle the bob travels along.
    ctx.strokeStyle = '#262c3a';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, rodLen, 0, Math.PI * 2); ctx.stroke();

    // Bob position: θ=0 points up. x = sinθ (right), y = -cosθ (up on screen).
    const bx = cx + rodLen * sinTheta;
    const by = cy - rodLen * cosTheta;

    // Rod — green when the bob is in the upper half (balanced), grey otherwise.
    ctx.strokeStyle = cosTheta >= 0 ? '#7cffb2' : '#5b6678';
    ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(bx, by); ctx.stroke();

    // Torque arc at the pivot (direction + magnitude of effort).
    if (Math.abs(torque) > 1e-3) {
      ctx.strokeStyle = '#ffd479';
      ctx.lineWidth = 3;
      const span = (Math.abs(torque) / MAX_TORQUE) * Math.PI;
      ctx.beginPath();
      ctx.arc(cx, cy, 22, -Math.PI / 2, -Math.PI / 2 + (torque > 0 ? span : -span), torque < 0);
      ctx.stroke();
    }

    // Pivot + bob.
    ctx.fillStyle = '#9aa4b2';
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = cosTheta >= 0 ? '#7cffb2' : '#cdd5e0';
    ctx.beginPath(); ctx.arc(bx, by, 12, 0, Math.PI * 2); ctx.fill();
  }

  private pollStatus(): void {
    void (async () => {
      try {
        const s = await this.api.status();
        this.modelStatus.set(s);
        if (s.status === 'loading' || s.status === 'training') setTimeout(() => this.pollStatus(), 3000);
      } catch {
        // backend unreachable — leave status unknown
      }
    })();
  }
}
