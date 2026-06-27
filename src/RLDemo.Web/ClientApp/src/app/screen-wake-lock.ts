import { Injectable } from '@angular/core';

// Minimal local shapes for the Screen Wake Lock API — avoids depending on whether the TS DOM lib in this
// workspace ships the global WakeLockSentinel/Navigator.wakeLock types.
interface WakeLockSentinelLike {
  release(): Promise<void>;
  addEventListener(type: 'release', listener: () => void): void;
}
interface WakeLockLike {
  request(type: 'screen'): Promise<WakeLockSentinelLike>;
}

/**
 * Keeps the device screen awake while an AI "Watch" stream is running, via the Screen Wake Lock API.
 *
 * Why: on mobile (notably Android), the screen auto-dims/locks after a few minutes of no touch input.
 * That freezes the tab — the live-stream WebSocket drops and the AI appears to "stop playing." Holding a
 * screen wake lock for the duration of watch mode prevents the lock entirely.
 *
 * The OS releases the lock whenever the page is hidden (tab switch, lock screen), so we re-acquire on the
 * next `visibilitychange` back to visible as long as the caller still wants it. Fully self-contained and a
 * safe no-op where the API is unavailable (older browsers, non-secure contexts) — the site still works,
 * it just can't override the screen timeout there.
 */
@Injectable({ providedIn: 'root' })
export class ScreenWakeLock {
  private sentinel: WakeLockSentinelLike | null = null;
  private desired = false;
  private readonly onVisibility = (): void => {
    if (this.desired && document.visibilityState === 'visible') void this.request();
  };

  /** Start keeping the screen awake. Idempotent; pairs with {@link release}. */
  async acquire(): Promise<void> {
    if (this.desired) return;
    this.desired = true;
    document.addEventListener('visibilitychange', this.onVisibility);
    await this.request();
  }

  /** Allow the screen to sleep again. */
  async release(): Promise<void> {
    this.desired = false;
    document.removeEventListener('visibilitychange', this.onVisibility);
    const held = this.sentinel;
    this.sentinel = null;
    try {
      await held?.release();
    } catch {
      // Already released by the OS — nothing to do.
    }
  }

  private async request(): Promise<void> {
    const wakeLock = (navigator as Navigator & { wakeLock?: WakeLockLike }).wakeLock;
    if (this.sentinel || !wakeLock) return;
    try {
      const sentinel = await wakeLock.request('screen');
      if (!this.desired) {
        void sentinel.release(); // released while the request was in flight
        return;
      }
      this.sentinel = sentinel;
      sentinel.addEventListener('release', () => (this.sentinel = null));
    } catch {
      // Denied (e.g. the tab isn't focused) or unsupported — a later visibilitychange will retry.
    }
  }
}
