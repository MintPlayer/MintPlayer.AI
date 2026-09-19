import { describe, expect, it } from 'vitest';

import { parseConvNet } from './draughts-net';

// Draughts' half of the C#/TS checkpoint pair, kind "selfplay-pv-conv" — the only conv reader in the app
// (see `snake/snake-net.spec.ts` for the general rationale). Three things here are specific and easy to
// break silently:
//
//   * 34 float arrays are read back-to-back in one fixed order with no names and no lengths to re-check
//     them against, so a single inserted, dropped or swapped parameter lands every later tensor on the
//     wrong field. The fixture gives every parameter distinct values, which turns that into a failure.
//   * the per-block layers CONCATENATE into flat per-role arrays (all blocks' conv1 weights in `b1W`,
//     and so on) rather than nesting, so block-major vs role-major drift is visible in the lengths.
//   * `actions` is not stored: it is derived as (h·w/2)² from the board size in the header.
//
// Bytes are written from the format description, not from the reader's constants: the magic as the ASCII
// characters 'R','L','N','C' in on-disk order, the kind with BinaryWriter's 7-bit length prefix, scalars
// little-endian, and floats chosen to be exact in float32 so values compare without tolerance.

/** Writes a checkpoint by hand, from the documented layout rather than from the reader's constants. */
class CkptWriter {
  private readonly bytes: number[] = [];

  private scalar(v: number, put: (dv: DataView, value: number) => void): this {
    const u = new Uint8Array(4);
    put(new DataView(u.buffer), v);
    this.bytes.push(...u);
    return this;
  }

  magic(tag = 'RLNC'): this {
    for (const ch of tag) this.bytes.push(ch.charCodeAt(0));
    return this;
  }

  u8(v: number): this { this.bytes.push(v & 0xff); return this; }
  i32(v: number): this { return this.scalar(v, (dv, value) => dv.setInt32(0, value, true)); }

  str(s: string): this {
    const utf8 = new TextEncoder().encode(s);
    let n = utf8.length;
    do {
      const b = n & 0x7f;
      n >>>= 7;
      this.u8(n > 0 ? b | 0x80 : b);
    } while (n > 0);
    this.bytes.push(...utf8);
    return this;
  }

  floats(values: number[]): this {
    this.i32(values.length);
    for (const v of values) this.scalar(v, (dv, value) => dv.setFloat32(0, value, true));
    return this;
  }

  buffer(): ArrayBuffer { return new Uint8Array(this.bytes).buffer as ArrayBuffer; }
}

const PLANES = 2, BOARD_H = 4, BOARD_W = 4, FILTERS = 3, BLOCKS = 2;
const DIMS = [PLANES, BOARD_H, BOARD_W, FILTERS, BLOCKS];

// Every parameter gets its own value band, derived from its position in the stream, so no two tensors
// can be confused for one another. `param` appends to PARAMS, so declaration order IS write order.
const PARAMS: number[][] = [];
const param = (n: number) => {
  const a = Array.from({ length: n }, (_, i) => (PARAMS.length + 1) * 10 + i / 4);
  PARAMS.push(a);
  return a;
};

const STEM_W = param(4), STEM_B = param(2), STEM_NG = param(2), STEM_NB = param(2);
// Block 0, then block 1 — each contributing conv1 (w,b), norm1 (γ,β), conv2 (w,b), norm2 (γ,β).
const B0_1W = param(3), B0_1B = param(1), B0_N1G = param(1), B0_N1B = param(1);
const B0_2W = param(3), B0_2B = param(1), B0_N2G = param(1), B0_N2B = param(1);
const B1_1W = param(3), B1_1B = param(1), B1_N1G = param(1), B1_N1B = param(1);
const B1_2W = param(3), B1_2B = param(1), B1_N2G = param(1), B1_N2B = param(1);
const P_CONV_W = param(2), P_CONV_B = param(1), P_NG = param(1), P_NB = param(1);
const P_HEAD_W = param(2), P_HEAD_B = param(1);
const V_CONV_W = param(2), V_CONV_B = param(1), V_NG = param(1), V_NB = param(1);
const V_HID_W = param(2), V_HID_B = param(1), V_HEAD_W = param(2), V_HEAD_B = param(1);

interface CkptOptions {
  magic?: string;
  kind?: string;
  version?: number;
  dimCount?: number;
  dims?: number[];
  params?: number[][];
}

function convCkpt(o: CkptOptions = {}): ArrayBuffer {
  const dims = o.dims ?? DIMS;
  const w = new CkptWriter()
    .magic(o.magic)
    .str(o.kind ?? 'selfplay-pv-conv')
    .i32(o.version ?? 1)
    .i32(o.dimCount ?? dims.length);
  for (const d of dims) w.i32(d);
  for (const p of o.params ?? PARAMS) w.floats(p);
  return w.buffer();
}

describe('parseConvNet', () => {
  it('round-trips a hand-written selfplay-pv-conv checkpoint', () => {
    const net = parseConvNet(convCkpt());

    // The returned net is the conv tier, and its MLP fields are the empty ones `withConv` supplies.
    expect(net.conv).not.toBeNull();
    const conv = net.conv!;

    expect(conv.planes).toBe(PLANES);
    expect(conv.boardH).toBe(BOARD_H);
    expect(conv.boardW).toBe(BOARD_W);
    expect(conv.filters).toBe(FILTERS);
    expect(conv.blocks).toBe(BLOCKS);

    // Hand-computed from the stored board: half = 4·4/2 = 8 dark squares, actions = 8² = 64 (from,to)
    // pairs; the observation is planes · h · w = 2 · 4 · 4 = 32.
    expect(conv.actions).toBe(64);
    expect(net.actions).toBe(64);
    expect(net.inputSize).toBe(32);

    // Stem: conv weight, conv bias, then the norm's γ and β.
    expect(conv.stemW).toEqual(STEM_W);
    expect(conv.stemB).toEqual(STEM_B);
    expect(conv.stemNG).toEqual(STEM_NG);
    expect(conv.stemNB).toEqual(STEM_NB);

    // Per-block layers concatenate by ROLE across blocks, not by block.
    expect(conv.b1W).toEqual([...B0_1W, ...B1_1W]);
    expect(conv.b1B).toEqual([...B0_1B, ...B1_1B]);
    expect(conv.n1G).toEqual([...B0_N1G, ...B1_N1G]);
    expect(conv.n1B).toEqual([...B0_N1B, ...B1_N1B]);
    expect(conv.b2W).toEqual([...B0_2W, ...B1_2W]);
    expect(conv.b2B).toEqual([...B0_2B, ...B1_2B]);
    expect(conv.n2G).toEqual([...B0_N2G, ...B1_N2G]);
    expect(conv.n2B).toEqual([...B0_N2B, ...B1_N2B]);

    // Policy branch, then value branch — the value branch has an extra hidden layer before its head.
    expect(conv.pConvW).toEqual(P_CONV_W);
    expect(conv.pConvB).toEqual(P_CONV_B);
    expect(conv.pNG).toEqual(P_NG);
    expect(conv.pNB).toEqual(P_NB);
    expect(conv.pHeadW).toEqual(P_HEAD_W);
    expect(conv.pHeadB).toEqual(P_HEAD_B);
    expect(conv.vConvW).toEqual(V_CONV_W);
    expect(conv.vConvB).toEqual(V_CONV_B);
    expect(conv.vNG).toEqual(V_NG);
    expect(conv.vNB).toEqual(V_NB);
    expect(conv.vHidW).toEqual(V_HID_W);
    expect(conv.vHidB).toEqual(V_HID_B);
    expect(conv.vHeadW).toEqual(V_HEAD_W);
    expect(conv.vHeadB).toEqual(V_HEAD_B);
  });

  it('derives the action count from the stored board size', () => {
    // A blockless 3-plane 2×2 board: half = 2, so actions = 4, while the observation is 3·2·2 = 12.
    // Different arithmetic from the 4×4 case above, which is what distinguishes the (h·w/2)² rule from
    // any formula that happens to agree on one board.
    const flat = Array.from({ length: 4 + 14 }, () => [1]); // stem (4) + no blocks + the 14 head tensors
    const net = parseConvNet(convCkpt({ dims: [3, 2, 2, 3, 0], params: flat }));

    expect(net.actions).toBe(4);
    expect(net.inputSize).toBe(12);
    expect(net.conv!.b1W).toEqual([]);
  });

  it('rejects a version it was not written for', () => {
    expect(() => parseConvNet(convCkpt({ version: 2 }))).toThrow(/version/);
  });

  it('rejects a header that does not carry exactly five dimensions', () => {
    expect(() => parseConvNet(convCkpt({ dimCount: 4 }))).toThrow(/dims/);
  });

  it('rejects a buffer whose magic is not RLNC', () => {
    expect(() => parseConvNet(convCkpt({ magic: 'RLNX' }))).toThrow(/RLNC/);
  });

  it('rejects the non-conv self-play checkpoint kind', () => {
    // "selfplay-pv" is chess' MLP checkpoint: the same writer, a different tensor layout. Loading it
    // here would misread every tensor, so the kind guard is the thing standing between them.
    expect(() => parseConvNet(convCkpt({ kind: 'selfplay-pv' }))).toThrow(/kind/);
  });

  it('throws rather than returning a short net when the buffer is truncated', () => {
    const full = convCkpt();
    expect(() => parseConvNet(full.slice(0, full.byteLength - 8))).toThrow();
  });
});
