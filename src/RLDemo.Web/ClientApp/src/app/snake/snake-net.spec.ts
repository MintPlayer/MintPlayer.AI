import { describe, expect, it } from 'vitest';

import { parseSnakeNet } from './snake-net';

// The checkpoint readers are the ONE hand-written half of a C#/TS pair: the .ckpt bytes are produced
// by the C# writer (DuelingQNetCheckpoint / CheckpointFormat) and re-read here by hand. Nothing in the
// build ties the two together, so a byte-layout drift — a field added on one side, an endianness slip,
// a head read in the wrong order — is silent: the parse still "succeeds" and the browser net simply
// plays badly. That is the bug class these specs exist to catch.
//
// The trick that makes them meaningful: the test writes the checkpoint from the FORMAT DESCRIPTION,
// not from the reader's constants. The magic is emitted as the four ASCII bytes 'R','L','N','C' in
// that order, so if the reader ever read it big-endian the comparison would fail; lengths are emitted
// as BinaryWriter's 7-bit-encoded int; scalars are little-endian; every float is chosen to be exactly
// representable in float32, so values round-trip to `toBe`/`toEqual` equality with no tolerance.
//
// This file carries the full explanation for the four "dueling-q" readers (snake, tetris, crazy-fruits,
// fruit-cake), which share the layout byte for byte; their specs assert the same contract against their
// own entry point and their own generated net type.

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

// A deliberately tiny net, small enough that every weight count is hand-computable.
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
// The sigma tensors a noisy net interleaves. Both the values AND the length differ from every mean
// tensor, so keeping a sigma by mistake cannot look like a pass.
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

describe('parseSnakeNet', () => {
  it('round-trips a hand-written dueling-q checkpoint', () => {
    const net = parseSnakeNet(duelingCkpt());

    expect(net.inputSize).toBe(INPUT);
    expect(net.actions).toBe(ACTIONS);
    expect(net.hidden).toEqual(HIDDEN);

    // Trunk weights and biases are DE-interleaved: the stream alternates W,b per layer, but the reader
    // must produce one flat W array and one flat b array, each concatenated in layer order.
    expect(net.trunkWFlat).toEqual([...L0W, ...L1W]);
    expect(net.trunkBFlat).toEqual([...L0B, ...L1B]);
    // Hand-computed: 3×4 + 4×2 weights, 4 + 2 biases.
    expect(net.trunkWFlat).toHaveLength(20);
    expect(net.trunkBFlat).toHaveLength(6);

    // Head order in the stream is value.W, value.b, adv.W, adv.b — swapping any pair would land the
    // wrong array on the wrong field, which is exactly what these four assertions pin.
    expect(net.valueW).toEqual(VALUE_W);
    expect(net.valueB).toEqual(VALUE_B);
    expect(net.advW).toEqual(ADV_W);
    expect(net.advB).toEqual(ADV_B);
  });

  it('keeps only the mean tensors of a noisy checkpoint', () => {
    // Inference runs with noise off, so the four sigma tensors must be read and discarded, not stored
    // and not skipped over by the wrong number of bytes.
    const net = parseSnakeNet(duelingCkpt({ noisy: true }));

    expect(net.valueW).toEqual(VALUE_W);
    expect(net.valueB).toEqual(VALUE_B);
    expect(net.advW).toEqual(ADV_W);
    expect(net.advB).toEqual(ADV_B);
    expect(net.trunkWFlat).toEqual([...L0W, ...L1W]);
  });

  it('reads a version-1 checkpoint, which carries no noisy flag at all', () => {
    // Version 1 has no noisy byte in the stream. Reading one anyway would shift every float by one
    // byte and silently produce garbage weights instead of an error.
    const net = parseSnakeNet(duelingCkpt({ version: 1 }));

    expect(net.trunkWFlat).toEqual([...L0W, ...L1W]);
    expect(net.valueW).toEqual(VALUE_W);
    expect(net.advB).toEqual(ADV_B);
  });

  it('rejects a buffer whose magic is not RLNC', () => {
    expect(() => parseSnakeNet(duelingCkpt({ magic: 'RLNX' }))).toThrow(/RLNC/);
  });

  it('rejects a checkpoint written for a different net kind', () => {
    expect(() => parseSnakeNet(duelingCkpt({ kind: 'mlp' }))).toThrow(/kind/);
  });

  it('decodes a multi-byte 7-bit length prefix', () => {
    // Every kind string shipped today is short enough for a single length byte, so the continuation
    // loop is otherwise never exercised. A 200-char kind needs two bytes; the reader echoes the kind
    // it decoded into the error, so seeing it back intact proves the varint was decoded correctly.
    const longKind = 'x'.repeat(200);
    expect(() => parseSnakeNet(duelingCkpt({ kind: longKind }))).toThrow(longKind);
  });

  it('throws rather than returning a short net when the buffer is truncated', () => {
    const full = duelingCkpt();
    expect(() => parseSnakeNet(full.slice(0, full.byteLength - 8))).toThrow();
  });
});
