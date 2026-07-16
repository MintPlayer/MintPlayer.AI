// Loads the self-play-trained draughts checkpoint in the browser and builds the single-source
// PgDraughtsNet (generated from draughts_solver.pg). Parsing the .ckpt binary is I/O, not pure math, so
// this is the one hand-written piece; it mirrors the C# writer (ConvResidualPolicyValueNet.Save /
// CheckpointFormat), kind "selfplay-pv-conv" — the conv branch chess never needed (its browser tier is
// an MLP; the draughts showcase net is the conv tower). Byte-level reference: DraughtsNetParityTests.
//
//   uint32  magic  = "RLNC" (0x434E4C52)
//   string  kind   = "selfplay-pv-conv"  (BinaryWriter: 7-bit-encoded length prefix + UTF-8 bytes)
//   int32   version (= 1)
//   int32   dimCount (= 5), then [planes, boardH, boardW, filters, blocks]
//   per parameter, in ConvResidualPolicyValueNet.Parameters() order:
//     int32 count + count × float32
//   order: stem conv (w,b), stem norm (γ,β), per block × [conv1 (w,b), norm1 (γ,β), conv2 (w,b),
//   norm2 (γ,β)], policy conv (w,b), policy norm (γ,β), policy head (w,b), value conv (w,b),
//   value norm (γ,β), value hidden (w,b), value head (w,b). Per-block layers CONCATENATE into the
//   PgDraughtsConvNet's flat per-role arrays.
//
// `actions` is NOT stored — it is the game's (N²/2)², derived here from the stored board size.

import { PgDraughtsConvNet, PgDraughtsNet } from './draughts_solver';

const MAGIC = 0x434e4c52; // "RLNC"
const KIND = 'selfplay-pv-conv';

/** A difficulty tier: which checkpoint to load + the search knobs (the chess manifest shape). */
export interface DraughtsDifficulty {
  label: string;
  ckpt: string;
  sims: number;
  temperature: number;
  cpuct: number;
  winRateVsRandom?: number;
}

// Used when the manifest is missing/unreadable.
const FALLBACK_DIFFICULTIES: DraughtsDifficulty[] = [
  { label: 'Default', ckpt: '/models/checkers8.az.ckpt', sims: 8, temperature: 0, cpuct: 1.5 },
];

/** Fetch + parse the difficulty manifest (ordered weakest→strongest). */
export async function loadDifficulties(url = '/models/draughts-difficulties.json'): Promise<DraughtsDifficulty[]> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return FALLBACK_DIFFICULTIES;
    const raw: unknown = await resp.json();
    if (!Array.isArray(raw) || raw.length === 0) return FALLBACK_DIFFICULTIES;
    return raw.map((d: any) => ({
      label: String(d.label),
      ckpt: String(d.ckpt),
      sims: Number(d.sims) || 8,
      temperature: Number(d.temperature) || 0,
      cpuct: Number(d.cpuct) || 1.5,
      winRateVsRandom: typeof d.winRateVsRandom === 'number' ? d.winRateVsRandom : undefined,
    }));
  } catch {
    return FALLBACK_DIFFICULTIES;
  }
}

/** Fetch + parse the shipped conv checkpoint. Returns null if missing/unreadable/incompatible (the
 *  caller then falls back to random legal moves, like the chess page). */
export async function loadDraughtsNet(url: string): Promise<PgDraughtsNet | null> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    return parseConvNet(await resp.arrayBuffer());
  } catch {
    return null;
  }
}

export function parseConvNet(buffer: ArrayBuffer): PgDraughtsNet {
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

  if (u32() !== MAGIC) throw new Error('draughts-net: not an RLNC checkpoint');
  const kind = readString();
  if (kind !== KIND) throw new Error(`draughts-net: expected kind '${KIND}', got '${kind}'`);
  const version = i32();
  if (version !== 1) throw new Error(`draughts-net: unsupported version ${version}`);
  const dimCount = i32();
  if (dimCount !== 5) throw new Error(`draughts-net: expected 5 dims, got ${dimCount}`);
  const planes = i32(), h = i32(), w = i32(), filters = i32(), blocks = i32();
  const half = (h * w) / 2;
  const actions = half * half; // (N²/2)² — 1024 on 8×8, 2500 on 10×10

  const conv = new PgDraughtsConvNet(planes, h, w, filters, blocks, actions);
  conv.stemW = readFloats(); conv.stemB = readFloats();
  conv.stemNG = readFloats(); conv.stemNB = readFloats();
  for (let i = 0; i < blocks; i++) {
    for (const f of readFloats()) conv.b1W.push(f);
    for (const f of readFloats()) conv.b1B.push(f);
    for (const f of readFloats()) conv.n1G.push(f);
    for (const f of readFloats()) conv.n1B.push(f);
    for (const f of readFloats()) conv.b2W.push(f);
    for (const f of readFloats()) conv.b2B.push(f);
    for (const f of readFloats()) conv.n2G.push(f);
    for (const f of readFloats()) conv.n2B.push(f);
  }
  conv.pConvW = readFloats(); conv.pConvB = readFloats();
  conv.pNG = readFloats(); conv.pNB = readFloats();
  conv.pHeadW = readFloats(); conv.pHeadB = readFloats();
  conv.vConvW = readFloats(); conv.vConvB = readFloats();
  conv.vNG = readFloats(); conv.vNB = readFloats();
  conv.vHidW = readFloats(); conv.vHidB = readFloats();
  conv.vHeadW = readFloats(); conv.vHeadB = readFloats();
  return PgDraughtsNet.withConv(conv);
}
