import { describe, expect, it } from 'vitest';

import { parseTetrisDuelingQNet } from './tetris-net';

// Tetris' half of the C#/TS checkpoint pair. The "dueling-q" layout is shared with snake, crazy-fruits
// and fruit-cake — `snake/snake-net.spec.ts` carries the full rationale for writing the bytes by hand
// from the format description rather than from the reader's own constants. What is specific here is the
// entry point and the generated net type it constructs, so the same contract is asserted against them.

/** Writes a checkpoint by hand, from the documented layout rather than from the reader's constants. */
class CkptWriter {
  private readonly bytes: number[] = [];

  private scalar(v: number, put: (dv: DataView, value: number) => void): this {
    const u = new Uint8Array(4);
    put(new DataView(u.buffer), v);
    this.bytes.push(...u);
    return this;
  }

  /** The magic, as the ASCII bytes it has on disk — NOT as a pre-swapped 32-bit constant. */
  magic(tag = 'RLNC'): this {
    for (const ch of tag) this.bytes.push(ch.charCodeAt(0));
    return this;
  }

  u8(v: number): this { this.bytes.push(v & 0xff); return this; }
  i32(v: number): this { return this.scalar(v, (dv, value) => dv.setInt32(0, value, true)); }

  /** BinaryWriter.Write(string): 7-bit-encoded length prefix, then UTF-8 bytes. */
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

  /** int32 count, then count little-endian float32. */
  floats(values: number[]): this {
    this.i32(values.length);
    for (const v of values) this.scalar(v, (dv, value) => dv.setFloat32(0, value, true));
    return this;
  }

  buffer(): ArrayBuffer { return new Uint8Array(this.bytes).buffer as ArrayBuffer; }
}

const INPUT = 3;
const HIDDEN = [4, 2];
const ACTIONS = 2;

// tag + i/4 is exact in float32 for these magnitudes, so parsed values compare with ===.
const seq = (tag: number, n: number) => Array.from({ length: n }, (_, i) => tag + i / 4);

const L0W = seq(10, INPUT * HIDDEN[0]);      // 3 × 4 = 12
const L0B = seq(20, HIDDEN[0]);              // 4
const L1W = seq(30, HIDDEN[0] * HIDDEN[1]);  // 4 × 2 = 8
const L1B = seq(40, HIDDEN[1]);              // 2
const VALUE_W = seq(50, HIDDEN[1] * 1);      // 2 × 1 = 2
const VALUE_B = seq(60, 1);
const ADV_W = seq(70, HIDDEN[1] * ACTIONS);  // 2 × 2 = 4
const ADV_B = seq(80, ACTIONS);
// Values AND length differ from every mean tensor, so keeping a sigma by mistake cannot look like a pass.
const SIGMA = seq(90, 3);

interface CkptOptions { magic?: string; kind?: string; version?: number; noisy?: boolean; }

function duelingCkpt(o: CkptOptions = {}): ArrayBuffer {
  const version = o.version ?? 2;
  const noisy = o.noisy ?? false;
  const w = new CkptWriter()
    .magic(o.magic)
    .str(o.kind ?? 'dueling-q')
    .i32(version)
    .i32(INPUT)
    .i32(HIDDEN.length);
  for (const h of HIDDEN) w.i32(h);
  w.i32(ACTIONS);
  if (version >= 2) w.u8(noisy ? 1 : 0);           // the noisy flag exists only from version 2 on
  w.floats(L0W).floats(L0B).floats(L1W).floats(L1B);
  if (!noisy) {
    w.floats(VALUE_W).floats(VALUE_B).floats(ADV_W).floats(ADV_B);
  } else {
    w.floats(VALUE_W).floats(SIGMA).floats(VALUE_B).floats(SIGMA)
      .floats(ADV_W).floats(SIGMA).floats(ADV_B).floats(SIGMA);
  }
  return w.buffer();
}

describe('parseTetrisDuelingQNet', () => {
  it('round-trips a hand-written dueling-q checkpoint', () => {
    const net = parseTetrisDuelingQNet(duelingCkpt());

    expect(net.inputSize).toBe(INPUT);
    expect(net.actions).toBe(ACTIONS);
    expect(net.hidden).toEqual(HIDDEN);

    // The stream alternates W,b per trunk layer; the reader must emit one flat W array and one flat b
    // array, each concatenated in layer order. Hand-computed: 3×4 + 4×2 weights, 4 + 2 biases.
    expect(net.trunkWFlat).toEqual([...L0W, ...L1W]);
    expect(net.trunkBFlat).toEqual([...L0B, ...L1B]);
    expect(net.trunkWFlat).toHaveLength(20);
    expect(net.trunkBFlat).toHaveLength(6);

    expect(net.valueW).toEqual(VALUE_W);
    expect(net.valueB).toEqual(VALUE_B);
    expect(net.advW).toEqual(ADV_W);
    expect(net.advB).toEqual(ADV_B);
  });

  it('keeps only the mean tensors of a noisy checkpoint', () => {
    const net = parseTetrisDuelingQNet(duelingCkpt({ noisy: true }));

    expect(net.valueW).toEqual(VALUE_W);
    expect(net.valueB).toEqual(VALUE_B);
    expect(net.advW).toEqual(ADV_W);
    expect(net.advB).toEqual(ADV_B);
    expect(net.trunkWFlat).toEqual([...L0W, ...L1W]);
  });

  it('reads a version-1 checkpoint, which carries no noisy flag at all', () => {
    // Reading a byte that is not in the stream would shift every float by one and produce garbage
    // weights rather than an error.
    const net = parseTetrisDuelingQNet(duelingCkpt({ version: 1 }));

    expect(net.trunkWFlat).toEqual([...L0W, ...L1W]);
    expect(net.valueW).toEqual(VALUE_W);
    expect(net.advB).toEqual(ADV_B);
  });

  it('rejects a buffer whose magic is not RLNC', () => {
    expect(() => parseTetrisDuelingQNet(duelingCkpt({ magic: 'RLNX' }))).toThrow(/RLNC/);
  });

  it('rejects a checkpoint written for a different net kind', () => {
    expect(() => parseTetrisDuelingQNet(duelingCkpt({ kind: 'mlp' }))).toThrow(/kind/);
  });

  it('throws rather than returning a short net when the buffer is truncated', () => {
    const full = duelingCkpt();
    expect(() => parseTetrisDuelingQNet(full.slice(0, full.byteLength - 8))).toThrow();
  });
});
