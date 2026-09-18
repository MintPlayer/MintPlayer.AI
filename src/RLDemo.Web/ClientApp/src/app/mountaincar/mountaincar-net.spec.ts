import { describe, expect, it } from 'vitest';

import { parseMlp } from './mountaincar-net';

// MountainCar's half of the C#/TS checkpoint pair, kind "mlp" — the PPO actor (see
// `snake/snake-net.spec.ts` for the general rationale). What is specific here:
//
//   * the layer COUNT is not stored. It is sizes.length − 1, so the shape array is the only thing
//     telling the reader how many (weight, bias) pairs follow; a drift in that arithmetic reads the
//     wrong number of tensors out of the stream.
//   * two header fields are consumed and deliberately ignored — the version and the hidden-activation
//     byte (the generated net is tanh-only). They still occupy bytes, so they must be skipped by
//     exactly the right width; a version gate added on the C# side would be invisible here.
//
// Bytes are written from the format description, not from the reader's constants: the magic as the
// ASCII characters 'R','L','N','C' in on-disk order, the kind with BinaryWriter's 7-bit length prefix,
// scalars little-endian, and floats chosen to be exact in float32 so values compare without tolerance.

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

  /** int32 count, then count little-endian int32. */
  ints(values: number[]): this {
    this.i32(values.length);
    for (const v of values) this.i32(v);
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

const SIZES = [2, 4, 3]; // two layers: 2→4, 4→3

const seq = (tag: number, n: number) => Array.from({ length: n }, (_, i) => tag + i / 4);

const W0 = seq(10, SIZES[0] * SIZES[1]); // 2 × 4 = 8
const B0 = seq(20, SIZES[1]);            // 4
const W1 = seq(30, SIZES[1] * SIZES[2]); // 4 × 3 = 12
const B1 = seq(40, SIZES[2]);            // 3

interface CkptOptions {
  magic?: string;
  kind?: string;
  version?: number;
  activation?: number;
  sizes?: number[];
  params?: number[][];
}

function mlpCkpt(o: CkptOptions = {}): ArrayBuffer {
  const w = new CkptWriter()
    .magic(o.magic)
    .str(o.kind ?? 'mlp')
    .i32(o.version ?? 1)
    .ints(o.sizes ?? SIZES)
    .u8(o.activation ?? 0);
  for (const p of o.params ?? [W0, B0, W1, B1]) w.floats(p);
  return w.buffer();
}

describe('parseMlp', () => {
  it('round-trips a hand-written mlp checkpoint', () => {
    const net = parseMlp(mlpCkpt());

    expect(net.sizes).toEqual(SIZES);

    // The stream alternates weight,bias per layer; the reader emits one flat weight array and one flat
    // bias array, each concatenated in layer order. Hand-computed: 2×4 + 4×3 weights, 4 + 3 biases.
    expect(net.wFlat).toEqual([...W0, ...W1]);
    expect(net.bFlat).toEqual([...B0, ...B1]);
    expect(net.wFlat).toHaveLength(20);
    expect(net.bFlat).toHaveLength(7);
  });

  it('reads sizes.length − 1 layers rather than a stored layer count', () => {
    // A single-layer net: one weight/bias pair follows the header, and nothing may be read past it.
    const w = seq(50, 2 * 3);
    const b = seq(60, 3);
    const net = parseMlp(mlpCkpt({ sizes: [2, 3], params: [w, b] }));

    expect(net.sizes).toEqual([2, 3]);
    expect(net.wFlat).toEqual(w);
    expect(net.bFlat).toEqual(b);
  });

  it('consumes but ignores the version and the hidden-activation byte', () => {
    // Neither is validated: an unseen version and a non-tanh activation code still parse, and the
    // weights land unshifted, which is what proves both fields were skipped at the right width.
    const net = parseMlp(mlpCkpt({ version: 7, activation: 3 }));

    expect(net.sizes).toEqual(SIZES);
    expect(net.wFlat).toEqual([...W0, ...W1]);
    expect(net.bFlat).toEqual([...B0, ...B1]);
  });

  it('rejects a buffer whose magic is not RLNC', () => {
    expect(() => parseMlp(mlpCkpt({ magic: 'RLNX' }))).toThrow(/RLNC/);
  });

  it('rejects a checkpoint written for a different net kind', () => {
    expect(() => parseMlp(mlpCkpt({ kind: 'dueling-q' }))).toThrow(/kind/);
  });

  it('throws rather than returning a short net when the buffer is truncated', () => {
    const full = mlpCkpt();
    expect(() => parseMlp(full.slice(0, full.byteLength - 8))).toThrow();
  });
});
