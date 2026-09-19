using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69 (COVERAGE_90_PRD §19.5 item 1) — <see cref="CubePolicyTraining.TrainStep"/>, the single supervised step
/// BOTH cube policy campaigns (Kociemba imitation and teacher-free EfficientCube) train through. It is 26 lines of
/// index arithmetic that nothing exercised.
/// </summary>
/// <remarks>
/// <para><b>The silent failures this guards.</b> Two of the three lines that matter are indexing, and both fail
/// quietly:</para>
/// <list type="bullet">
/// <item>The one-hot weight is written at <c>i * ActionCount + s.Action</c>. A transposed index
/// (<c>s.Action * batch + i</c>) still produces a well-formed loss, still descends, and still reports a plausible
/// accuracy — it just trains the net toward the WRONG move for every sample. Nothing throws.</item>
/// <item>The Huber target is <c>DistanceToGo / CubePolicyNet.DistanceScale</c>, and
/// <see cref="CubePolicyNet.Evaluate"/> multiplies the scale back in. Drop either half and distance predictions
/// come back 30× off while the loss curve looks perfectly healthy.</item>
/// </list>
/// <para>The batch is hand-built: a <see cref="CubeOracle.LabeledState"/> is a plain record and a
/// <see cref="FaceletCube"/> plus one applied quarter turn gives its facelets, so no Kociemba tables and no oracle
/// are needed. A memorizable 4-sample batch makes both rules assertable: cross-entropy must fall and top-1
/// accuracy must reach 1.0 (only correct if the one-hot lines up with the argmax), and the predicted distances
/// must come back in MOVES near their labels (only correct if the scale round-trips).</para>
/// </remarks>
public class CubePolicyTrainStepTests
{
    /// <summary>Four distinct one-move-from-solved boards, each labeled with its own action and its own distance.</summary>
    private static List<CubeOracle.LabeledState> Batch()
    {
        var samples = new List<CubeOracle.LabeledState>();
        for (int a = 0; a < 4; a++)
        {
            var cube = new FaceletCube();
            cube.ApplyQuarterTurn(a);
            samples.Add(new CubeOracle.LabeledState(cube.Facelets.ToArray(), a, a + 1));
        }
        return samples;
    }

    [Fact]
    public void TrainStep_memorizes_the_labeled_move_of_every_sample_in_the_batch()
    {
        var net = new CubePolicyNet(new Xoshiro256StarStar(11), [16, 16]);
        var adam = new Adam(net.Parameters(), 2e-2f);
        var samples = Batch();

        var first = CubePolicyTraining.TrainStep(net, adam, samples, 0, samples.Count);
        (double Ce, double Huber, double Acc) last = first;
        for (int step = 0; step < 300; step++)
            last = CubePolicyTraining.TrainStep(net, adam, samples, 0, samples.Count);

        Assert.True(last.Ce < first.Ce, $"cross-entropy must fall on a memorizable batch: {first.Ce} → {last.Ce}");
        // 1.0, not "better than chance": with one distinct label per distinct board, anything less means the
        // one-hot and the argmax are not reading the same [sample, action] cell.
        Assert.Equal(1.0, last.Acc);
    }

    [Fact]
    public void The_argmax_after_training_is_each_samples_own_labeled_action()
    {
        // The direct form of the transposed-index guard: ask the net itself, not the accuracy the step reports.
        var net = new CubePolicyNet(new Xoshiro256StarStar(12), [16, 16]);
        var adam = new Adam(net.Parameters(), 2e-2f);
        var samples = Batch();
        for (int step = 0; step < 300; step++) CubePolicyTraining.TrainStep(net, adam, samples, 0, samples.Count);

        foreach (var s in samples)
        {
            var (logits, _) = net.Evaluate(FaceletCube.FromFacelets(s.Facelets));
            int argmax = 0;
            for (int a = 1; a < logits.Length; a++) if (logits[a] > logits[argmax]) argmax = a;
            Assert.Equal(s.Action, argmax);
        }
    }

    [Fact]
    public void The_distance_head_is_trained_in_scaled_units_and_read_back_in_moves()
    {
        // TrainStep regresses DistanceToGo / DistanceScale; Evaluate multiplies DistanceScale back in. Drop the
        // division and the labels become 1..4 in SCALED units, so Evaluate would hand back ~30..120 moves for
        // boards that are one to four quarter turns from solved -- with no exception anywhere.
        var net = new CubePolicyNet(new Xoshiro256StarStar(13), [16, 16]);
        var adam = new Adam(net.Parameters(), 2e-2f);
        var samples = Batch();
        for (int step = 0; step < 600; step++) CubePolicyTraining.TrainStep(net, adam, samples, 0, samples.Count);

        foreach (var s in samples)
        {
            var (_, distance) = net.Evaluate(FaceletCube.FromFacelets(s.Facelets));
            // The load-bearing bound: every board here is 1-4 quarter turns from solved, so a prediction anywhere
            // near 30-120 is the missing (or doubled) DistanceScale rather than an under-trained head.
            Assert.True(distance < 10f, $"distance {distance} is off by roughly the {CubePolicyNet.DistanceScale}× scale");
            Assert.True(Math.Abs(distance - s.DistanceToGo) < 2f, $"expected ~{s.DistanceToGo} moves, got {distance}");
        }
    }

    [Fact]
    public void The_offset_selects_the_window_the_caller_asked_for()
    {
        // The campaigns walk a shuffled sample list in [offset, offset+batch) windows. An offset that is ignored
        // (or applied to only one of the three arrays) trains the right count of samples against the wrong labels.
        var net = new CubePolicyNet(new Xoshiro256StarStar(14), [16, 16]);
        var adam = new Adam(net.Parameters(), 2e-2f);
        var samples = Batch();

        for (int step = 0; step < 300; step++) CubePolicyTraining.TrainStep(net, adam, samples, 2, 2);

        // Only samples 2 and 3 were ever shown, so only they need to be learned -- but they must be, exactly.
        Assert.Equal(1.0, CubePolicyTraining.TrainStep(net, adam, samples, 2, 2).Acc);
    }
}
