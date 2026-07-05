// Loads the trained FruitCake DuelingQNet checkpoint in the browser and builds the single-source PgDuelingNet
// (generated from fruitcake_solver.pg). This is the ONE piece that isn't single-sourced through Polyglot: parsing
// the .ckpt binary is I/O, not pure math. It mirrors the C# reader (DuelingQNetCheckpoint / CheckpointFormat):
//
//   uint32  magic  = "RLNC" (0x434E4C52, little-endian on disk 52 4C 4E 43)
//   string  kind   = "dueling-q"   (BinaryWriter: 7-bit-encoded-int length prefix + UTF-8 bytes)
//   int32   version (= 2)
//   int32   inputSize
//   int32   hiddenCount, then hiddenCount × int32
//   int32   actions
//   byte    noisy   (version >= 2)
//   per parameter (DuelingQNet.Parameters() order): int32 count + count × float32 (little-endian)
//
// Parameter order — plain net: trunk[l].W, trunk[l].b (per layer), value.W, value.b, adv.W, adv.b.
// Noisy net: each head is Mean*, Sigma* — noise is OFF at inference, so only the Mean tensors are kept.
// float32 read into a JS number is the exact f64 of that float32, matching the C# `(double)f` load — so the
// browser net is bit-identical to the C# PgDuelingNet built from the same checkpoint.

import { PgDuelingNet } from './fruitcake_solver';

const MAGIC = 0x434e4c52; // "RLNC"
const KIND = 'dueling-q';

/** Fetch + parse the shipped checkpoint. Returns null if it's missing/unreadable/incompatible (the caller then
 *  falls back to the heuristic leaf, exactly as the server did with no net). */
export async function loadFruitCakeNet(url = '/models/fruitcake-net.ckpt'): Promise<PgDuelingNet | null> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    return parseDuelingQNet(await resp.arrayBuffer());
  } catch {
    return null;
  }
}

export function parseDuelingQNet(buffer: ArrayBuffer): PgDuelingNet {
  const dv = new DataView(buffer);
  let off = 0;
  const i32 = () => { const v = dv.getInt32(off, true); off += 4; return v; };
  const u32 = () => { const v = dv.getUint32(off, true); off += 4; return v; };
  const u8 = () => dv.getUint8(off++);
  const read7bit = () => { // BinaryWriter's 7-bit-encoded length prefix
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

  if (u32() !== MAGIC) throw new Error('fruitcake-net: not an RLNC checkpoint');
  const kind = readString();
  if (kind !== KIND) throw new Error(`fruitcake-net: expected kind '${KIND}', got '${kind}'`);
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

  return new PgDuelingNet(inputSize, actions, hidden, trunkWFlat, trunkBFlat, valueW, valueB, advW, advB);
}
