using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Shared live-viewer helpers for the cube campaigns. They train on shuffled scramble batches, not a running env,
/// so there's no "current observation" — instead the viewer forwards ONE fixed scramble (a depth-8 board from a
/// constant seed) each frame, so a viewer watches the net's move preferences + hidden activations for that single
/// board evolve as it learns. All forwards are read-only (no Backward) and, for a single row, stay on the CPU even
/// under the GPU backend (well below its MAC threshold) — so they never contend with training.
/// </summary>
public static class CubeViz
{
    private static float[] Probe(ref float[]? cache)
    {
        if (cache is null)
        {
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(new Xoshiro256StarStar(0xC0FFEE), 8, quarterTurnsOnly: true));
            var obs = new float[RubiksCubeEnv.ObservationSize];
            RubiksCubeEnv.WriteObservation(cube, obs);
            cache = obs;
        }
        return cache;
    }

    /// <summary>Policy net: the fixed board + its 12 move logits.</summary>
    public static (float[] Input, float[] Output)? SampleIo(CubePolicyNet? net, ref float[]? cache)
    {
        if (net is null) return null;
        try
        {
            var obs = Probe(ref cache);
            var (logits, _) = net.Forward(new Tensor((float[])obs.Clone(), 1, obs.Length));
            return ((float[])obs.Clone(), [.. logits.Data]);
        }
        catch { return null; }
    }

    public static float[][]? SampleActivations(CubePolicyNet? net, ref float[]? cache)
    {
        if (net is null) return null;
        try { var obs = Probe(ref cache); return net.LayerActivations(new Tensor((float[])obs.Clone(), 1, obs.Length)); }
        catch { return null; }
    }

    /// <summary>Value net (DAVI): the fixed board + its single cost-to-go estimate.</summary>
    public static (float[] Input, float[] Output)? SampleValueIo(IValueNet? net, ref float[]? cache)
    {
        if (net is null) return null;
        try
        {
            var obs = Probe(ref cache);
            var y = net.Forward(new Tensor((float[])obs.Clone(), 1, obs.Length));
            return ((float[])obs.Clone(), [.. y.Data]);
        }
        catch { return null; }
    }

    public static float[][]? SampleValueActivations(IValueNet? net, ref float[]? cache)
    {
        if (net is not ResidualMlp res) return null;
        try { var obs = Probe(ref cache); return res.LayerActivations(new Tensor((float[])obs.Clone(), 1, obs.Length)); }
        catch { return null; }
    }
}
