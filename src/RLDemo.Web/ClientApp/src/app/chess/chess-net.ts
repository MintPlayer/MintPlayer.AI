// Loads the self-play-trained chess policy/value checkpoint in the browser and builds the single-source
// PgPolicyValueNet (generated from chess_solver.pg). This is the ONE piece not single-sourced through Polyglot:
// parsing the .ckpt binary is I/O, not pure math. It mirrors the C# writer (PolicyValueNet.Save / CheckpointFormat),
// kind "selfplay-pv":
//
//   uint32  magic  = "RLNC" (0x434E4C52, little-endian on disk 52 4C 4E 43)
//   string  kind   = "selfplay-pv"   (BinaryWriter: 7-bit-encoded-int length prefix + UTF-8 bytes)
//   int32   version (= 2)
//   int32   trunkCount, then trunkCount × int32   (the trunk hidden widths)
//   per parameter, in PolicyValueNet.Parameters() order = [each trunk layer, then policyHead, then valueHead]:
//     int32 count + count × float32   (weight, row-major [inDim, outDim]; then bias)
//
// inputSize (1152) and actions (4672) are NOT stored — they come from ChessGame and are passed in.
// A float32 read into a JS number is the exact f64 of that float32, matching the C# `(double)f` load, so the
// browser net is the f64 twin of the training net (agrees to within an f32 tolerance — see ChessNetParityTests).

import { PgPolicyValueNet } from './chess_solver';

const MAGIC = 0x434e4c52; // "RLNC"
const KIND = 'selfplay-pv';

export const CHESS_INPUT_SIZE = 18 * 64; // 1152
export const CHESS_ACTIONS = 4672;       // 64 × 73 AlphaZero move encoding

/** Fetch + parse the shipped checkpoint. Returns null if it's missing/unreadable/incompatible (the caller can
 *  then disable the AI or fall back, exactly as the server did when no net was present). */
export async function loadChessNet(
  url = '/models/chess.az.ckpt',
  inputSize = CHESS_INPUT_SIZE,
  actions = CHESS_ACTIONS,
): Promise<PgPolicyValueNet | null> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    return parsePolicyValueNet(await resp.arrayBuffer(), inputSize, actions);
  } catch {
    return null;
  }
}

export function parsePolicyValueNet(
  buffer: ArrayBuffer,
  inputSize = CHESS_INPUT_SIZE,
  actions = CHESS_ACTIONS,
): PgPolicyValueNet {
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

  if (u32() !== MAGIC) throw new Error('chess-net: not an RLNC checkpoint');
  const kind = readString();
  if (kind !== KIND) throw new Error(`chess-net: expected kind '${KIND}', got '${kind}'`);
  const version = i32();
  if (version < 1 || version > 2) throw new Error(`chess-net: unsupported version ${version}`);

  const trunkCount = i32();
  const hidden: number[] = [];
  for (let i = 0; i < trunkCount; i++) hidden.push(i32());

  const trunkWFlat: number[] = [];
  const trunkBFlat: number[] = [];
  for (let l = 0; l < trunkCount; l++) {
    for (const f of readFloats()) trunkWFlat.push(f);
    for (const f of readFloats()) trunkBFlat.push(f);
  }
  const policyW = readFloats(); const policyB = readFloats();
  const valueW = readFloats(); const valueB = readFloats();

  return new PgPolicyValueNet(inputSize, actions, hidden, trunkWFlat, trunkBFlat, policyW, policyB, valueW, valueB);
}
