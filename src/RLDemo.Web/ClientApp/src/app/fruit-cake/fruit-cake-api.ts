import { Injectable } from '@angular/core';

/** One fruit in a streamed AI frame. */
export interface FruitFrameItem {
  x: number;
  y: number;
  angle: number;
  tier: number;
}

/** One server-streamed frame of an AI FruitCake game (PRD §4.6). */
export interface FruitCakeFrame {
  fruit: FruitFrameItem[];
  heldTier: number;
  nextTier: number;
  score: number;
  danger: boolean;
  done: boolean;
}

export interface FruitCakeStatus {
  status: 'loading' | 'ready' | 'failed';
  error: string | null;
}

@Injectable({ providedIn: 'root' })
export class FruitCakeApi {
  async status(): Promise<FruitCakeStatus> {
    const response = await fetch('/api/fruitcake/status');
    return response.json();
  }

  /**
   * Opens the server-authoritative live stream — the backend runs the C# physics + agent and pushes frames;
   * the caller just renders. Returns the socket so the caller can close it on teardown.
   */
  connectLive(onFrame: (frame: FruitCakeFrame) => void, onClose: () => void): WebSocket {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${proto}://${location.host}/api/fruitcake/live`);
    socket.onmessage = e => onFrame(JSON.parse(e.data) as FruitCakeFrame);
    socket.onclose = onClose;
    socket.onerror = onClose;
    return socket;
  }
}
