// Loads the trained Snake DuelingQNet checkpoint in the browser and builds the single-source PgSnakeNet
// (generated from snake_solver.pg). Mirrors the C# reader (DuelingQNetCheckpoint / CheckpointFormat) — same
// byte layout as fruit-cake/fruitcake-net.ts (kind "dueling-q"); the only difference is it constructs Snake's
// generated net type. (A shared ckpt.ts parser across games is a possible future refactor.)
//
//   uint32 magic "RLNC" · string kind "dueling-q" (7-bit-varint length + UTF-8) · int32 version
//   int32 inputSize · int32 hiddenCount + hidden[] · int32 actions · byte noisy(v>=2)
//   per parameter (trunk W/b…, value W/b, adv W/b): int32 count + count×float32
// float32 read into a JS number is the exact f64 of that float32, matching the C# `(double)f` load.

import { PgSnakeNet } from './snake_solver';

const MAGIC = 0x434e4c52; // "RLNC"
const KIND = 'dueling-q';

/** Fetch + parse the shipped Snake checkpoint. Returns null if missing/unreadable/incompatible. */
export async function loadSnakeNet(url = 'snake-net.ckpt'): Promise<PgSnakeNet | null> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    return parseSnakeNet(await resp.arrayBuffer());
  } catch {
    return null;
  }
}

export function parseSnakeNet(buffer: ArrayBuffer): PgSnakeNet {
  const dv = new DataView(buffer);
  let off = 0;
  const i32 = () => { const v = dv.getInt32(off, true); off += 4; return v; };
  const u32 = () => { const v = dv.getUint32(off, true); off += 4; return v; };
  const u8 = () => dv.getUint8(off++);
  const read7bit = () => {
    let result = 0, shift = 0, b: number;
    do { b = u8(); result |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
    return result;
  };
  const readString = () => {
    const len = read7bit();
    const bytes = new Uint8Array(buffer, off, len);
    off += len;
    return new TextDecoder().decode(bytes);
  };
  const readFloats = () => {
    const n = i32();
    const a = new Array<number>(n);
    for (let i = 0; i < n; i++) { a[i] = dv.getFloat32(off, true); off += 4; }
    return a;
  };

  if (u32() !== MAGIC) throw new Error('snake-net: not an RLNC checkpoint');
  const kind = readString();
  if (kind !== KIND) throw new Error(`snake-net: expected kind '${KIND}', got '${kind}'`);
  const version = i32();
  const inputSize = i32();
  const hiddenCount = i32();
  const hidden: number[] = [];
  for (let i = 0; i < hiddenCount; i++) hidden.push(i32());
  const actions = i32();
  const noisy = version >= 2 && u8() !== 0;

  const trunkWFlat: number[] = [];
  const trunkBFlat: number[] = [];
  for (let l = 0; l < hiddenCount; l++) {
    for (const f of readFloats()) trunkWFlat.push(f);
    for (const f of readFloats()) trunkBFlat.push(f);
  }
  // Heads. Plain: W, b. Noisy: MeanW, SigmaW, MeanB, SigmaB — keep the Mean tensors (inference has noise off).
  let valueW: number[], valueB: number[], advW: number[], advB: number[];
  if (!noisy) {
    valueW = readFloats(); valueB = readFloats();
    advW = readFloats(); advB = readFloats();
  } else {
    valueW = readFloats(); readFloats(); valueB = readFloats(); readFloats();
    advW = readFloats(); readFloats(); advB = readFloats(); readFloats();
  }

  return new PgSnakeNet(inputSize, actions, hidden, trunkWFlat, trunkBFlat, valueW, valueB, advW, advB);
}
