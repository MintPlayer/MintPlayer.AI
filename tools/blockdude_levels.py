"""Import the 11 original Block Dude levels into this repo's ASCII grid format.

Provenance: the level layouts are Brandon Sterner's, from the 2001 TI-83+ PuzzPack game. They are read from
the TI-84+CE port by Shaun McFall (Merthsoft Creations), whose source is released under the Unlicense:

    https://github.com/merthsoft/blockdudece  ->  src/level.c, src/level.h

Tile codes come from that port's game.c: EMPTY 0, WALL 1, BLOCK 2, DOOR 3. The player start is stored
separately as (start_x, start_y) in each `struct level`.

Source rows are TOP-first and we keep that for the ASCII asset; the engine flips to its Y-up model on load.

    python tools/blockdude_levels.py

Requires network access. Re-running reproduces the committed file exactly.
"""
from __future__ import annotations

import re
import urllib.request

BASE = 'https://raw.githubusercontent.com/merthsoft/blockdudece/main/src/'

# Canonical, single source of truth: embedded into the Environments assembly for training AND linked into
# wwwroot/levels for the browser, so the C# campaign and the Angular app read the SAME bytes.
DEST = (r'C:\Repos\MintPlayer.AI\src\MintPlayer.AI.ReinforcementLearning.Environments'
        r'\BlockDude\levels\blockdude-levels.json')

TILE = {0: '.', 1: 'W', 2: 'B', 3: 'D'}


def fetch(name: str) -> str:
    with urllib.request.urlopen(BASE + name) as r:
        return r.read().decode('utf-8', errors='replace')


def main() -> None:
    level_c = fetch('level.c')
    level_h = fetch('level.h')

    expected = int(re.search(r'#define\s+NUM_BUILTIN_LEVELS\s+(\d+)', level_h).group(1))

    maps: dict[int, list[int]] = {}
    for m in re.finditer(r'uint8_t\s+map_(\d+)\s*\[\]\s*=\s*\{(.*?)\}\s*;', level_c, re.S):
        maps[int(m.group(1))] = [int(x) for x in re.findall(r'\d+', m.group(2))]

    metas: dict[int, list[int]] = {}
    for m in re.finditer(r'struct\s+level\s+level_(\d+)\s*=\s*\{(.*?)\}\s*;', level_c, re.S):
        metas[int(m.group(1))] = [int(x) for x in re.findall(r'\d+', m.group(2))][:4]

    assert len(maps) == expected, f'{len(maps)} maps, expected {expected}'
    assert len(metas) == expected, f'{len(metas)} level structs, expected {expected}'

    levels_out, report = [], []
    for i in range(expected):
        w, h, sx, sy = metas[i]
        cells = maps[i]
        assert len(cells) == w * h, f'level {i + 1}: {len(cells)} cells != {w}x{h}'

        grid = [[TILE[cells[y * w + x]] for x in range(w)] for y in range(h)]
        assert grid[sy][sx] in '.D', f'level {i + 1}: player start on {grid[sy][sx]}'
        grid[sy][sx] = 'P'
        rows = [''.join(r) for r in grid]

        blocks = sum(r.count('B') for r in rows)
        doors = sum(r.count('D') for r in rows)
        floating = sum(1 for y in range(h - 1) for x in range(w)
                       if rows[y][x] == 'B' and rows[y + 1][x] in '.D')
        assert doors == 1, f'level {i + 1} has {doors} doors'
        report.append((i + 1, w, h, blocks, floating))

        levels_out.append({'name': f'Level {i + 1}', 'grid': rows})

    doc = {
        '_generator': 'tools/blockdude_levels.py — do not hand-edit; re-run the tool instead.',
        '_source': 'https://github.com/merthsoft/blockdudece (src/level.c, Unlicense) — TI-84+CE port by '
                   'Shaun McFall of the original TI-83+ PuzzPack game and levels by Brandon Sterner (2001).',
        '_format': 'Rows are TOP-first. W = wall (immovable, climbable, never carryable), B = carryable block, '
                   'P = player start, D = door/exit, . = empty. No implicit floor or boundary — everything is '
                   'explicit.',
        '_notSettled': 'These grids are deliberately NOT gravity-settled: level 11 ships 14 blocks floating in '
                       'mid-air, which is authored content. The engine applies gravity only to the walking '
                       'player or a dropped block, never as a global settle pass (PRD §4.2).',
        '_noOptimalMoves': 'Omitted on purpose: most of these boards are far beyond any exact solver (level 11 '
                           'is 42 blocks on 551 cells). The oracle labels small GENERATED boards for training; '
                           'these shipped levels are the human content and the search benchmark (PRD §4.4).',
        'version': 1,
        'levels': levels_out,
    }

    import json
    import os
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, 'w', newline='\n', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print(f'{"lvl":>3} {"size":>8} {"blocks":>6} {"floating":>8}')
    for lvl, w, h, blocks, floating in report:
        print(f'{lvl:>3} {w:>3}x{h:<4} {blocks:>6} {floating:>8}')
    print(f'\nwrote {len(report)} levels -> {DEST}')


if __name__ == '__main__':
    main()
