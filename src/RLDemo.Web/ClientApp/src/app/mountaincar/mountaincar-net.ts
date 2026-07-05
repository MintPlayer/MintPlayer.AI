// Loads the trained MountainCar PPO policy (an Mlp) checkpoint in the browser → the single-source PgMlpNet.
// Mirrors the C# MlpCheckpoint / CheckpointFormat reader:
//   uint32 magic "RLNC" · string kind "mlp" (7-bit-varint length + UTF-8) · int32 version
//   int32 sizesCount + sizes[] · byte hiddenActivation · per layer: int32 count+float32[] weight, then bias
// (PgMlpNet applies Tanh between hidden layers to match the shipped net's activation.)

import { PgMlpNet } from './mountaincar_solver';

const MAGIC = 0x434e4c52; // "RLNC"
const KIND = 'mlp';

/** Fetch + parse the shipped MountainCar policy. Returns null if missing/unreadable/incompatible. */
export async function loadMountainCarNet(url = '/models/mountaincar-net.ckpt'): Promise<PgMlpNet | null> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    return parseMlp(await resp.arrayBuffer());
  } catch {
    return null;
  }
}

export function parseMlp(buffer: ArrayBuffer): PgMlpNet {
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
  const readInts = () => {
    const n = i32();
    const a = new Array<number>(n);
    for (let i = 0; i < n; i++) a[i] = i32();
    return a;
  };
  const readFloats = () => {
    const n = i32();
    const a = new Array<number>(n);
    for (let i = 0; i < n; i++) { a[i] = dv.getFloat32(off, true); off += 4; }
    return a;
  };

  if (u32() !== MAGIC) throw new Error('mountaincar-net: not an RLNC checkpoint');
  const kind = readString();
  if (kind !== KIND) throw new Error(`mountaincar-net: expected kind '${KIND}', got '${kind}'`);
  i32();          // version
  const sizes = readInts();
  u8();           // hidden activation (Tanh for the PPO actor; PgMlpNet is tanh-only)

  const wFlat: number[] = [];
  const bFlat: number[] = [];
  const layers = sizes.length - 1;
  for (let l = 0; l < layers; l++) {
    for (const f of readFloats()) wFlat.push(f);
    for (const f of readFloats()) bFlat.push(f);
  }
  return new PgMlpNet(sizes, wFlat, bFlat);
}
