"""Import the Block Dude levels into this repo's ASCII grid format: the 11 originals, then 4 bonus levels.

Levels 1-11 are Brandon Sterner's layouts from the 2001 TI-83+ PuzzPack game, read from the TI-84+CE port by
Shaun McFall (Merthsoft Creations), whose source is released under the Unlicense:

    https://github.com/merthsoft/blockdudece  ->  src/level.c, src/level.h

Tile codes come from that port's game.c: EMPTY 0, WALL 1, BLOCK 2, DOOR 3. The player start is stored
separately as (start_x, start_y) in each `struct level`.

Bonus 1-4 come from `BLOCKLV2`, the optional extra level set shipped with the same author's earlier TI-84+CSE
release — the appvar is vendored at tools/blockdude/ and decoded by `blockdude_ti_levels`, because unlike the CE
port it was never published as source. Its FIRST entry is the game's home screen (a title drawn in blocks),
which is skipped. These are genuinely new puzzles: the CSE release's *main* pack, `BLOCKLVL`, holds the same
11 originals we already import (7 of them byte-identical, 4 reframed for the narrower CSE screen), so there
is nothing to take from it.

Source rows are TOP-first and we keep that for the ASCII asset; the engine flips to its Y-up model on load.

    python tools/blockdude_levels.py

Requires network access for levels 1-11. Re-running reproduces the committed file exactly.
"""
from __future__ import annotations

import os
import re
import urllib.request

import blockdude_ti_levels

BASE = 'https://raw.githubusercontent.com/merthsoft/blockdudece/main/src/'

# Canonical, single source of truth: embedded into the Environments assembly for training AND linked into
# wwwroot/levels for the browser, so the C# campaign and the Angular app read the SAME bytes.
DEST = (r'C:\Repos\MintPlayer.AI\src\MintPlayer.AI.ReinforcementLearning.Environments'
        r'\BlockDude\levels\blockdude-levels.json')

# Vendored rather than fetched: the CSE release ships only the built appvar, so this file IS the source.
BONUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blockdude', 'BLOCKLV2.8xv')

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

    # The bonus pack. Unlike the CE originals it is not uniformly single-door: bonus 2 seals its exit behind a
    # row of seven door cells and bonus 4 offers two separate exits, so only "at least one" is asserted here.
    bonus = [lv for lv in blockdude_ti_levels.decode(BONUS) if not lv.is_home_screen]
    assert len(bonus) == 4, f'{len(bonus)} bonus levels, expected 4'
    for i, level in enumerate(bonus):
        rows = level.grid
        doors = sum(r.count('D') for r in rows)
        assert doors >= 1, f'bonus {i + 1} has no door'
        blocks = sum(r.count('B') for r in rows)
        floating = sum(1 for y in range(level.height - 1) for x in range(level.width)
                       if rows[y][x] == 'B' and rows[y + 1][x] in '.D')
        report.append((f'B{i + 1}', level.width, level.height, blocks, floating))
        levels_out.append({'name': f'Bonus {i + 1}', 'grid': rows})

    doc = {
        '_generator': 'tools/blockdude_levels.py — do not hand-edit; re-run the tool instead.',
        '_source': 'Levels 1-11: https://github.com/merthsoft/blockdudece (src/level.c, Unlicense) — TI-84+CE '
                   'port by Shaun McFall of the original TI-83+ PuzzPack game and levels by Brandon Sterner '
                   '(2001). Bonus 1-4: the BLOCKLV2 extra level set from the same author\'s TI-84+CSE release, '
                   'decoded from the vendored appvar at tools/blockdude/BLOCKLV2.8xv.',
        '_doors': 'Most levels have exactly one door, but not all: bonus 2 seals its exit behind a row of '
                  'seven door cells and bonus 4 offers two separate exits. Reaching ANY door wins.',
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
