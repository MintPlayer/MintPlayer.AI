using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube.Kociemba;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The internals of the Kociemba two-phase port (<c>RubiksCube/Kociemba/</c>) — the coordinate
/// encoders, the cubie-level group operations, the solvability checks and the nibble-packed pruning
/// tables — exercised directly rather than through a solve.
/// </summary>
/// <remarks>
/// <para>The only existing coverage of this port is <see cref="CubeApiTests"/> and
/// <c>RubiksCubeTests</c>, which solve a scrambled cube end to end. That path is a happy path: it
/// never sees a malformed cube, never asks for a solution that cannot be found, and only ever visits
/// the coordinate indices that one particular scramble happens to produce.</para>
/// <para>Everything here guards the same failure mode: this code is an almost-verbatim port of Java,
/// full of index arithmetic with no runtime checks, and every one of its mistakes is SILENT. A
/// coordinate encoder that is not the exact inverse of its decoder does not throw — it builds a
/// slightly wrong move table, and the solver then returns a move sequence that simply does not solve
/// the cube. A <c>verify()</c> branch that returns the wrong code turns "your photo of the cube has a
/// mis-scanned sticker" into an unrelated error message. And <c>setPruning</c>/<c>getPruning</c> pack
/// two 4-bit entries per byte: a single wrong shift corrupts a neighbouring entry, which lowers a
/// search bound and produces wrong answers on some cubes and not others.</para>
/// </remarks>
public class KociembaInternalsTests
{
    private const string Solved = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB";

    // Indices into CubieCube.moveCube, which is declared as { U, R, F, D, L, B }.
    private const int U = 0, R = 1, F = 2, D = 3, L = 4, B = 5;

    private static CubieCube Identity() => new();

    /// <summary>The cube reached from solved by applying <paramref name="moves"/> as quarter turns.</summary>
    private static CubieCube After(params int[] moves)
    {
        var cc = new CubieCube();
        foreach (int m in moves)
        {
            cc.cornerMultiply(CubieCube.moveCube[m]);
            cc.edgeMultiply(CubieCube.moveCube[m]);
        }
        return cc;
    }

    private static string FaceletsAfter(params int[] moves) => After(moves).toFaceCube().to_fc_String();

    private static Corner[] IdCp() =>
        [Corner.URF, Corner.UFL, Corner.ULB, Corner.UBR, Corner.DFR, Corner.DLF, Corner.DBL, Corner.DRB];

    private static Edge[] IdEp() =>
        [Edge.UR, Edge.UF, Edge.UL, Edge.UB, Edge.DR, Edge.DF, Edge.DL, Edge.DB, Edge.FR, Edge.FL, Edge.BL, Edge.BR];

    private static string WithFacelets(string s, params (int Index, char Colour)[] edits)
    {
        char[] chars = s.ToCharArray();
        foreach ((int index, char colour) in edits)
            chars[index] = colour;
        return new string(chars);
    }

    // getURFtoDLB and getURtoBR were private (no access modifier) while their setters were public, so
    // the set/get pair could not be asserted at all. Widened to internal -- InternalsVisibleTo already
    // covers this assembly -- in preference to reaching them by reflection, which would have pinned the
    // method NAMES rather than the behaviour and broken confusingly on any rename.
    private static int GetURFtoDLB(CubieCube c) => c.getURFtoDLB();

    private static int GetURtoBR(CubieCube c) => c.getURtoBR();

    // ── Coordinate encoders: set(x) then get() must give back x ──

    [Fact]
    public void twist_round_trips()
    {
        foreach (short idx in new short[] { 0, 1, 2, 3, 1093, 2185, 2186 })
        {
            var cc = Identity();
            cc.setTwist(idx);
            Assert.Equal(idx, cc.getTwist());
        }
    }

    [Fact]
    public void set_twist_leaves_the_eighth_corner_holding_the_orientation_sum()
    {
        // The coordinate only encodes 7 of the 8 corner orientations; the last is forced so the sum
        // stays a multiple of 3. If it were not, every cube built from a twist index would be
        // unsolvable and the whole phase-1 move table would be garbage.
        foreach (short idx in new short[] { 0, 1, 2, 1093, 2186 })
        {
            var cc = Identity();
            cc.setTwist(idx);

            int sum = 0;
            for (int i = 0; i < 8; i++)
                sum += cc.co[i];
            Assert.Equal(0, sum % 3);
        }
    }

    [Fact]
    public void flip_round_trips()
    {
        foreach (short idx in new short[] { 0, 1, 2, 1024, 2046, 2047 })
        {
            var cc = Identity();
            cc.setFlip(idx);
            Assert.Equal(idx, cc.getFlip());
        }
    }

    [Fact]
    public void set_flip_leaves_the_twelfth_edge_holding_the_flip_parity()
    {
        foreach (short idx in new short[] { 0, 1, 1024, 2047 })
        {
            var cc = Identity();
            cc.setFlip(idx);

            int sum = 0;
            for (int i = 0; i < 12; i++)
                sum += cc.eo[i];
            Assert.Equal(0, sum % 2);
        }
    }

    [Fact]
    public void fr_to_br_round_trips()
    {
        foreach (short idx in new short[] { 0, 1, 23, 24, 25, 5940, 11878, 11879 })
        {
            var cc = Identity();
            cc.setFRtoBR(idx);
            Assert.Equal(idx, cc.getFRtoBR());
        }
    }

    [Fact]
    public void urf_to_dlf_round_trips()
    {
        foreach (short idx in new short[] { 0, 1, 719, 720, 721, 10080, 20158, 20159 })
        {
            var cc = Identity();
            cc.setURFtoDLF(idx);
            Assert.Equal(idx, cc.getURFtoDLF());
        }
    }

    [Fact]
    public void ur_to_ul_round_trips()
    {
        foreach (short idx in new short[] { 0, 1, 5, 6, 7, 660, 1318, 1319 })
        {
            var cc = Identity();
            cc.setURtoUL(idx);
            Assert.Equal(idx, cc.getURtoUL());
        }
    }

    [Fact]
    public void ub_to_df_round_trips()
    {
        foreach (short idx in new short[] { 0, 1, 5, 6, 114, 660, 1318, 1319 })
        {
            var cc = Identity();
            cc.setUBtoDF(idx);
            Assert.Equal(idx, cc.getUBtoDF());
        }
    }

    [Fact]
    public void ur_to_df_round_trips()
    {
        foreach (int idx in new[] { 0, 1, 719, 720, 721, 10080, 20158, 20159 })
        {
            var cc = Identity();
            cc.setURtoDF(idx);
            Assert.Equal(idx, cc.getURtoDF());
        }
    }

    [Fact]
    public void urf_to_dlb_round_trips()
    {
        foreach (int idx in new[] { 0, 1, 2, 5039, 20160, 40318, 40319 })
        {
            var cc = Identity();
            cc.setURFtoDLB(idx);
            Assert.Equal(idx, GetURFtoDLB(cc));
        }
    }

    [Fact]
    public void ur_to_br_round_trips()
    {
        foreach (int idx in new[] { 0, 1, 11, 12, 239_500_800, 479_001_598, 479_001_599 })
        {
            var cc = Identity();
            cc.setURtoBR(idx);
            Assert.Equal(idx, GetURtoBR(cc));
        }
    }

    [Fact]
    public void the_solved_cube_sits_at_the_documented_coordinates()
    {
        // CoordCubeBuildTables' comment: "All coordinates are 0 for a solved cube except for
        // UBtoDF, which is 114". The tables are indexed by these, so a drift here is a silent
        // off-by-one across every lookup.
        var cc = Identity();

        Assert.Equal(0, cc.getTwist());
        Assert.Equal(0, cc.getFlip());
        Assert.Equal(0, cc.getFRtoBR());
        Assert.Equal(0, cc.getURFtoDLF());
        Assert.Equal(0, cc.getURtoUL());
        Assert.Equal(114, cc.getUBtoDF());
        Assert.Equal(0, cc.getURtoDF());
        Assert.Equal(0, GetURFtoDLB(cc));
        Assert.Equal(0, GetURtoBR(cc));
    }

    // ── Merging the two three-edge coordinates back into URtoDF ──

    [Fact]
    public void merging_ur_to_ul_with_ub_to_df_reproduces_ur_to_df()
    {
        // This is how phase 2 recovers its six-edge coordinate from the two cheap phase-1 ones; the
        // whole MergeURtoULandUBtoDF table is built from it.
        foreach (CubieCube cc in new[] { Identity(), After(U), After(U, U) })
            Assert.Equal(cc.getURtoDF(), CubieCube.getURtoDF(cc.getURtoUL(), cc.getUBtoDF()));
    }

    [Fact]
    public void merging_two_coordinates_that_claim_the_same_edge_slot_reports_a_collision()
    {
        // Index 0 of both coordinates puts its three edges in positions 0..2, so they overlap. The
        // merge must say -1 rather than quietly dropping one of the edges.
        Assert.Equal(-1, CubieCube.getURtoDF(0, 0));
    }

    // ── Cubie-level group operations ──

    [Fact]
    public void every_face_turn_has_order_four_on_the_corners()
    {
        for (int m = U; m <= B; m++)
        {
            var cc = Identity();
            cc.cornerMultiply(CubieCube.moveCube[m]);
            Assert.NotEqual(0, cc.getURFtoDLF() + GetURFtoDLB(cc) + cc.getTwist());

            for (int i = 1; i < 4; i++)
                cc.cornerMultiply(CubieCube.moveCube[m]);

            Assert.Equal(IdCp(), cc.cp);
            Assert.Equal(new byte[8], cc.co);
        }
    }

    [Fact]
    public void every_face_turn_has_order_four_on_the_edges()
    {
        for (int m = U; m <= B; m++)
        {
            var cc = Identity();
            cc.edgeMultiply(CubieCube.moveCube[m]);
            Assert.NotEqual(0, GetURtoBR(cc) + cc.getFlip());

            for (int i = 1; i < 4; i++)
                cc.edgeMultiply(CubieCube.moveCube[m]);

            Assert.Equal(IdEp(), cc.ep);
            Assert.Equal(new byte[12], cc.eo);
        }
    }

    [Fact]
    public void multiplying_never_mutates_the_move_it_is_multiplied_by()
    {
        // moveCube's six entries alias the static cpU/coU/... arrays, so a multiply that wrote
        // through its argument would permanently corrupt the shared move table for the whole process
        // — and the damage would only surface in whichever test ran next.
        Corner[] beforeCp = (Corner[])CubieCube.moveCube[R].cp.Clone();
        byte[] beforeCo = (byte[])CubieCube.moveCube[R].co.Clone();
        Edge[] beforeEp = (Edge[])CubieCube.moveCube[R].ep.Clone();
        byte[] beforeEo = (byte[])CubieCube.moveCube[R].eo.Clone();

        var cc = Identity();
        cc.cornerMultiply(CubieCube.moveCube[R]);
        cc.edgeMultiply(CubieCube.moveCube[R]);

        Assert.Equal(beforeCp, CubieCube.moveCube[R].cp);
        Assert.Equal(beforeCo, CubieCube.moveCube[R].co);
        Assert.Equal(beforeEp, CubieCube.moveCube[R].ep);
        Assert.Equal(beforeEo, CubieCube.moveCube[R].eo);
    }

    [Fact]
    public void a_single_quarter_turn_is_an_odd_permutation()
    {
        for (int m = U; m <= B; m++)
        {
            var cc = After(m);
            Assert.Equal(1, cc.cornerParity());
            Assert.Equal(1, cc.edgeParity());
        }
    }

    [Fact]
    public void corner_and_edge_parity_agree_on_every_cube_reachable_by_face_turns()
    {
        // The invariant phase 2 leans on: parityMove is indexed by the CORNER parity and used as if
        // it were the edge parity too.
        int[][] sequences =
        [
            [],
            [U],
            [U, R],
            [R, U, F, D, L, B],
            [F, F, R, U, U, B, L, D, R, R],
            [B, L, U, D, F, R, B, L, U, D, F, R],
        ];

        foreach (int[] moves in sequences)
        {
            var cc = After(moves);
            Assert.Equal(cc.cornerParity(), cc.edgeParity());
        }
    }

    [Fact]
    public void the_cubie_constructor_copies_rather_than_aliases_its_arguments()
    {
        Corner[] cp = IdCp();
        byte[] co = new byte[8];
        Edge[] ep = IdEp();
        byte[] eo = new byte[12];

        var cc = new CubieCube(cp, co, ep, eo);
        cp[0] = Corner.DRB;
        co[0] = 2;
        ep[0] = Edge.BR;
        eo[0] = 1;

        Assert.Equal(Corner.URF, cc.cp[0]);
        Assert.Equal(0, cc.co[0]);
        Assert.Equal(Edge.UR, cc.ep[0]);
        Assert.Equal(0, cc.eo[0]);
    }

    // ── CubieCube.verify(): every rejection branch ──

    [Fact]
    public void verify_accepts_a_cube_reachable_by_face_turns()
    {
        Assert.Equal(0, Identity().verify());
        Assert.Equal(0, After(R, U, F, D, L, B).verify());
    }

    [Fact]
    public void verify_rejects_a_duplicated_edge()
    {
        Edge[] ep = IdEp();
        ep[1] = Edge.UR;  // UR now appears twice and UF not at all

        Assert.Equal(-2, new CubieCube(IdCp(), new byte[8], ep, new byte[12]).verify());
    }

    [Fact]
    public void verify_rejects_an_odd_number_of_flipped_edges()
    {
        byte[] eo = new byte[12];
        eo[0] = 1;

        Assert.Equal(-3, new CubieCube(IdCp(), new byte[8], IdEp(), eo).verify());
    }

    [Fact]
    public void verify_rejects_a_duplicated_corner()
    {
        Corner[] cp = IdCp();
        cp[1] = Corner.URF;  // URF twice, UFL missing

        Assert.Equal(-4, new CubieCube(cp, new byte[8], IdEp(), new byte[12]).verify());
    }

    [Fact]
    public void verify_rejects_a_corner_twist_sum_that_is_not_a_multiple_of_three()
    {
        byte[] co = new byte[8];
        co[0] = 1;

        Assert.Equal(-5, new CubieCube(IdCp(), co, IdEp(), new byte[12]).verify());
    }

    [Fact]
    public void verify_accepts_a_corner_twist_sum_that_is_a_multiple_of_three()
    {
        // Guards the -5 test above against passing for the wrong reason: two corners twisted the
        // opposite way are legal, so the check must be the SUM and not "any corner is twisted".
        byte[] co = new byte[8];
        co[0] = 1;
        co[1] = 2;

        Assert.Equal(0, new CubieCube(IdCp(), co, IdEp(), new byte[12]).verify());
    }

    [Fact]
    public void verify_rejects_mismatched_corner_and_edge_parity()
    {
        // Two corners swapped and nothing else: every piece present, orientations fine, but the
        // corner permutation is odd while the edge permutation is even. This is the classic
        // "reassembled by hand" cube, and it is the only error the earlier checks cannot see.
        Corner[] cp = IdCp();
        (cp[0], cp[1]) = (cp[1], cp[0]);

        var cc = new CubieCube(cp, new byte[8], IdEp(), new byte[12]);
        Assert.Equal(1, cc.cornerParity());
        Assert.Equal(0, cc.edgeParity());
        Assert.Equal(-6, cc.verify());
    }

    // ── Tools.verify(string) ──

    [Fact]
    public void tools_verify_accepts_the_solved_cube()
    {
        Assert.Equal(0, Tools.verify(Solved));
        Assert.Equal(0, Tools.verify(FaceletsAfter(R, U, F, D, L, B)));
    }

    [Fact]
    public void tools_verify_rejects_a_colour_that_is_not_a_face_name()
    {
        Assert.Equal(-1, Tools.verify(WithFacelets(Solved, (0, 'G'))));
    }

    [Fact]
    public void tools_verify_rejects_a_repainted_sticker()
    {
        // All six letters are legal, but there are ten Rs and eight Us.
        Assert.Equal(-1, Tools.verify(WithFacelets(Solved, (0, 'R'))));
    }

    [Fact]
    public void tools_verify_rejects_a_single_flipped_edge()
    {
        // Swap the two stickers of the UF edge in place (U8 <-> F2). Colour counts are untouched and
        // the edge is still UF — only its orientation changed, so this reaches the flip check.
        Assert.Equal(-3, Tools.verify(WithFacelets(Solved, (7, 'F'), (19, 'U'))));
    }

    [Fact]
    public void tools_verify_rejects_a_single_twisted_corner()
    {
        // Rotate the three stickers of the URF corner in place (U9, R1, F3).
        Assert.Equal(-5, Tools.verify(WithFacelets(Solved, (8, 'F'), (9, 'U'), (20, 'R'))));
    }

    [Fact]
    public void tools_verify_rejects_two_swapped_corners()
    {
        // Put the UFL cubie in the URF slot and vice versa, both unrotated: URF gets (U,F,L) on
        // (U9,R1,F3) and UFL gets (U,R,F) on (U7,F1,L3).
        Assert.Equal(-6, Tools.verify(WithFacelets(Solved, (9, 'F'), (20, 'L'), (18, 'R'), (38, 'F'))));
    }

    [Fact]
    public void random_cubes_are_always_solvable()
    {
        // Tools.randomCube() uses an unseeded Random, so the only thing that can be asserted is the
        // property it promises: a uniformly drawn member of the cube group, i.e. one that verifies.
        for (int i = 0; i < 5; i++)
        {
            string cube = Tools.randomCube();

            Assert.Equal(54, cube.Length);
            Assert.Equal(0, Tools.verify(cube));
        }
    }

    [Fact]
    public void the_table_serializers_refuse_rather_than_writing_a_stale_file()
    {
        // These four are stubs left behind when BinaryFormatter was dropped; the tables are built in
        // memory now. They must keep throwing — a silent no-op would let a caller believe it had a
        // cached table on disk.
        Assert.Throws<NotSupportedException>(() => Tools.SerializeTable("twist", new short[1, 1]));
        Assert.Throws<NotSupportedException>(() => { _ = Tools.DeserializeTable("twist"); });
        Assert.Throws<NotSupportedException>(() => Tools.SerializeSbyteArray("prun", new sbyte[1]));
        Assert.Throws<NotSupportedException>(() => { _ = Tools.DeserializeSbyteArray("prun"); });
    }

    // ── Nibble-packed pruning tables ──

    [Fact]
    public void pruning_values_round_trip_through_both_halves_of_a_byte()
    {
        // Tables are pre-filled with -1 (both nibbles 0x0f) before anything is written, which is the
        // precondition setPruning is written against.
        for (sbyte value = 0; value <= 15; value++)
        {
            sbyte[] low = [-1, -1];
            CoordCubeBuildTables.setPruning(low, 2, value);
            Assert.Equal(value, CoordCubeBuildTables.getPruning(low, 2));
            Assert.Equal((sbyte)0x0f, CoordCubeBuildTables.getPruning(low, 3));   // the byte's other half
            Assert.Equal((sbyte)0x0f, CoordCubeBuildTables.getPruning(low, 0));   // the neighbouring byte
            Assert.Equal((sbyte)0x0f, CoordCubeBuildTables.getPruning(low, 1));

            sbyte[] high = [-1, -1];
            CoordCubeBuildTables.setPruning(high, 3, value);
            Assert.Equal(value, CoordCubeBuildTables.getPruning(high, 3));
            Assert.Equal((sbyte)0x0f, CoordCubeBuildTables.getPruning(high, 2));
            Assert.Equal((sbyte)0x0f, CoordCubeBuildTables.getPruning(high, 0));
            Assert.Equal((sbyte)0x0f, CoordCubeBuildTables.getPruning(high, 1));
        }
    }

    [Fact]
    public void pruning_entries_read_back_independently_of_each_other()
    {
        sbyte[] table = [-1, -1, -1, -1];
        for (int index = 0; index < 8; index++)
            CoordCubeBuildTables.setPruning(table, index, (sbyte)index);

        for (int index = 0; index < 8; index++)
            Assert.Equal((sbyte)index, CoordCubeBuildTables.getPruning(table, index));
    }

    [Fact]
    public void a_pruning_entry_can_only_be_written_once()
    {
        // setPruning is an AND, not an assignment: it can clear bits but never set them. The BFS that
        // fills the tables relies on this being safe by only ever writing an entry still reading
        // 0x0f. Anything else silently ANDs the two values together, which is why this is pinned.
        sbyte[] table = [-1];
        CoordCubeBuildTables.setPruning(table, 0, 5);
        Assert.Equal((sbyte)5, CoordCubeBuildTables.getPruning(table, 0));

        CoordCubeBuildTables.setPruning(table, 0, 15);
        Assert.Equal((sbyte)5, CoordCubeBuildTables.getPruning(table, 0));  // 0b0101 & 0b1111

        CoordCubeBuildTables.setPruning(table, 0, 2);
        Assert.Equal((sbyte)0, CoordCubeBuildTables.getPruning(table, 0));  // 0b0101 & 0b0010
    }

    // ── CoordCubeBuildTables ──

    [Fact]
    public void a_coordinate_cube_returns_to_its_start_after_four_identical_quarter_turns()
    {
        var coord = new CoordCubeBuildTables(Identity());
        short twist = coord.twist;
        short flip = coord.flip;
        short parity = coord.parity;
        short frToBr = coord.FRtoBR;
        short urfToDlf = coord.URFtoDLF;
        short urToUl = coord.URtoUL;
        short ubToDf = coord.UBtoDF;
        int urToDf = coord.URtoDF;

        for (int i = 0; i < 4; i++)
            coord.move(0);  // move index 3*axis + power-1; axis 0, quarter turn -> U

        Assert.Equal(twist, coord.twist);
        Assert.Equal(flip, coord.flip);
        Assert.Equal(parity, coord.parity);
        Assert.Equal(frToBr, coord.FRtoBR);
        Assert.Equal(urfToDlf, coord.URFtoDLF);
        Assert.Equal(urToUl, coord.URtoUL);
        Assert.Equal(ubToDf, coord.UBtoDF);
        Assert.Equal(urToDf, coord.URtoDF);
    }

    [Fact]
    public void asking_a_coordinate_cube_to_dump_its_tables_refuses()
    {
        Assert.Throws<NotSupportedException>(() => { _ = new CoordCubeBuildTables(Identity(), true); });
    }

    // ── SearchRunTime.solution: the branches a successful solve never takes ──

    [Fact]
    public void solving_an_already_solved_cube_returns_an_empty_move_list()
    {
        string result = SearchRunTime.solution(Solved, out string info);

        Assert.Equal("", result);
        Assert.False(string.IsNullOrEmpty(info));
    }

    [Fact]
    public void solving_reports_error_1_for_a_colour_that_is_not_a_face_name()
    {
        Assert.Equal("Error 1", SearchRunTime.solution(WithFacelets(Solved, (0, 'G')), out _));
    }

    [Fact]
    public void solving_reports_error_1_for_a_repainted_sticker()
    {
        Assert.Equal("Error 1", SearchRunTime.solution(WithFacelets(Solved, (0, 'R')), out _));
    }

    [Fact]
    public void solving_reports_the_verify_code_for_an_unsolvable_cube()
    {
        // The facelets parse and the colour counts are right, so these come back from cc.verify()
        // rather than from the input check — Error N for verify()'s -N.
        Assert.Equal("Error 3", SearchRunTime.solution(WithFacelets(Solved, (7, 'F'), (19, 'U')), out _));
        Assert.Equal("Error 5", SearchRunTime.solution(WithFacelets(Solved, (8, 'F'), (9, 'U'), (20, 'R')), out _));
        Assert.Equal("Error 6", SearchRunTime.solution(WithFacelets(Solved, (9, 'F'), (20, 'L'), (18, 'R'), (38, 'F')), out _));
    }

    [Fact]
    public void the_separator_is_the_only_difference_between_the_two_output_forms()
    {
        string facelets = FaceletsAfter(U);

        string plain = SearchRunTime.solution(facelets, out _);
        string separated = SearchRunTime.solution(facelets, out _, useSeparator: true);

        Assert.False(plain.StartsWith("Error"), plain);
        Assert.False(string.IsNullOrWhiteSpace(plain));
        Assert.DoesNotContain(".", plain);
        Assert.Contains(". ", separated);
        // The search is deterministic, so both calls describe the same maneuver; the separated form
        // must be the plain one with a phase marker inserted, not a differently formatted solution.
        Assert.Equal(plain, separated.Replace(". ", ""));
    }

    [Fact]
    public void a_max_depth_too_small_to_reach_the_cube_reports_error_7()
    {
        // maxDepth 1 allows one phase-1 move and zero phase-2 moves, which cannot reach a cube three
        // quarter turns from solved.
        Assert.Equal("Error 7", SearchRunTime.solution(FaceletsAfter(U, R, F), out _, maxDepth: 1));
    }

    [Fact]
    public void running_out_of_time_reports_error_8()
    {
        // A zero budget makes the deadline fire on the first elapsed millisecond. No timing is
        // asserted — only that the search gives up by the documented code rather than by looping or
        // returning a wrong answer. Twelve quarter turns is far past what a millisecond can solve.
        string result = SearchRunTime.solution(
            FaceletsAfter(R, F, U, L, B, D, R, F, U, L, B, D), out _, maxDepth: 22, timeOut: 0);

        Assert.Equal("Error 8", result);
    }
}
