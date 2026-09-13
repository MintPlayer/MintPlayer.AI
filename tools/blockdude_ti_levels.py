"""Decode a TI-83+/84+ Block Dude level pack into this repo's ASCII grid format.

Merthsoft's calculator ports store their level packs as a TI-BASIC *appvar* (`.8xv`) holding one line per
level. Line 1 is the magic `BLOCKDUDELEVELS`, line 2 the list of level passwords, and every line after that
is one level, written as a run of TI-BASIC stores followed by the map string:

    20->W:12->H:0->U:0->V:18->I:1->J:"0101...0101"->Str1

    W,H  grid size in cells          U,V  initial camera scroll (presentation only, ignored here)
    I,J  player start (x, y)         Str1 two decimal digits per cell, row-major, TOP-first

Tile codes match the ports' `game.c`: 00 empty, 01 wall, 02 block, 03 door. The *first* entry of every pack
is the game's home screen rather than a level — it omits `W`/`H` (the program falls back to the 20x12 screen)
and draws the title in blocks, which is exactly how `is_home_screen` tells the two apart.

Both container formats are understood, because they differ only in encoding:

  - `.8xv` — the appvar itself, with the body tokenized (0x04 = STO, 0x2A = a quote, 0x3F = newline). Digits,
    letters and `>` survive as plain ASCII, so the same grammar reads straight through the token stream.
  - `.txt` — the Token IDE's text export of that appvar, already detokenized to `->` and `"`.

Standalone use, to inspect any pack before importing it:

    python tools/blockdude_ti_levels.py tools/blockdude/BLOCKLV2.8xv --render

Imported by `tools/blockdude_levels.py`, which writes the shipped level asset.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

TILE = {0: '.', 1: 'W', 2: 'B', 3: 'D'}

# The home screen omits W/H and the program falls back to the calculator's 20x12 screen.
DEFAULT_WIDTH = 20

# TI-83F variable entries start after the 55-byte file header: a 2-byte header length, then the entry.
_VAR_OFFSET = 55

# `<digits> STO <LETTER> :` — tokenized, STO is `\x04` and the statement separator `:` is `\x3e`; detokenized
# they read as `->` and `:`. The map string is quoted with `\x2a` tokenized and `"` detokenized.
_STORE = re.compile(r'(\d+)(?:\x04|->)([A-Z])[>:]')
_ENTRY = re.compile(r'((?:\d+(?:\x04|->)[A-Z][>:])+)(?:\x2a|")(\d+)(?:\x2a|")')


@dataclass(frozen=True)
class TiLevel:
    """One decoded pack entry. `rows` is TOP-first and does NOT carry the player: `start` holds it separately,
    exactly as the appvar does."""

    width: int
    height: int
    start: tuple[int, int]
    rows: list[str]
    is_home_screen: bool

    @property
    def grid(self) -> list[str]:
        """`rows` with the player start written in as `P` — this repo's asset format."""
        x, y = self.start
        out = list(self.rows)
        out[y] = out[y][:x] + 'P' + out[y][x + 1:]
        return out

    def counts(self, tile: str) -> int:
        return sum(r.count(tile) for r in self.rows)


def _body(path: str) -> str:
    """The pack's line-oriented body, as text. Tokenized bytes are passed through unchanged — the grammar only
    needs the ASCII that survives tokenization."""
    raw = open(path, 'rb').read()
    if not raw.startswith(b'**TI83F*'):
        return raw.decode('utf-8', errors='replace')

    header_length = int.from_bytes(raw[_VAR_OFFSET:_VAR_OFFSET + 2], 'little')
    data_length = int.from_bytes(raw[_VAR_OFFSET + 2:_VAR_OFFSET + 4], 'little')
    start = _VAR_OFFSET + 2 + header_length
    return raw[start:start + data_length].decode('latin-1')


def decode(path: str) -> list[TiLevel]:
    """Every entry of the pack, home screen first — callers that want only the playable levels skip
    `is_home_screen`."""
    body = _body(path)
    if 'BLOCKDUDELEVELS' not in body:
        raise ValueError(f'{path}: not a Block Dude level pack (no BLOCKDUDELEVELS magic)')

    levels: list[TiLevel] = []
    for stores, digits in _ENTRY.findall(body):
        values = {name: int(value) for value, name in _STORE.findall(stores)}
        width = values.get('W', DEFAULT_WIDTH)
        cells = [int(digits[i:i + 2]) for i in range(0, len(digits), 2)]

        if len(cells) % width:
            raise ValueError(f'{path}: entry {len(levels)} has {len(cells)} cells, not a multiple of {width}')
        height = len(cells) // width
        if 'H' in values and values['H'] != height:
            raise ValueError(f'{path}: entry {len(levels)} declares H={values["H"]} but holds {height} rows')

        rows = [''.join(TILE[cells[y * width + x]] for x in range(width)) for y in range(height)]
        start = (values['I'], values['J'])
        if rows[start[1]][start[0]] != '.':
            raise ValueError(f'{path}: entry {len(levels)} starts the player on '
                             f'{rows[start[1]][start[0]]!r}, expected empty space')

        levels.append(TiLevel(width, height, start, rows, is_home_screen='W' not in values))

    if not levels:
        raise ValueError(f'{path}: no entries decoded')
    return levels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pack', help='a .8xv appvar or its Token IDE .txt export')
    parser.add_argument('--render', action='store_true', help='print each grid')
    args = parser.parse_args()

    for i, level in enumerate(decode(args.pack)):
        kind = 'home screen' if level.is_home_screen else f'level {i}'
        print(f'entry {i} ({kind}): {level.width}x{level.height} start={level.start} '
              f'blocks={level.counts("B")} doors={level.counts("D")}')
        if args.render:
            for row in level.grid:
                print('   ', row)
    return 0


if __name__ == '__main__':
    sys.exit(main())
