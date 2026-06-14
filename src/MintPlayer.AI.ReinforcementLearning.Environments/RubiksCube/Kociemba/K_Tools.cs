using System;
using System.IO;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube.Kociemba
{
    public class Tools
    {
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // Check if the cube string s represents a solvable cube.
        // 0: Cube is solvable
        // -1: There is not exactly one facelet of each colour
        // -2: Not all 12 edges exist exactly once
        // -3: Flip error: One edge has to be flipped
        // -4: Not all corners exist exactly once
        // -5: Twist error: One corner has to be twisted
        // -6: Parity error: Two corners or two edges have to be exchanged
        //
        /// <summary>
        /// Check if the cube definition string s represents a solvable cube.
        /// </summary>
        /// <param name="s"> is the cube definition string , see <seealso cref="Facelet"/> </param>
        /// <returns> 0: Cube is solvable<br/>
        ///         -1: There is not exactly one facelet of each colour<br/>
        ///         -2: Not all 12 edges exist exactly once<br/>
        ///         -3: Flip error: One edge has to be flipped<br/>
        ///         -4: Not all 8 corners exist exactly once<br/>
        ///         -5: Twist error: One corner has to be twisted<br/>
        ///         -6: Parity error: Two corners or two edges have to be exchanged </returns>
        public static int verify(string s)
        {
            int[] count = new int[6];
            try
            {
                for (int i = 0; i < 54; i++)
                {
                    // Parse the character at position i from string s (not i.ToString()!)
                    count[(int)(CubeColor)Enum.Parse(typeof(CubeColor), s[i].ToString())]++;
                }
            }
            catch (Exception)
            {
                return -1;
            }

            for (int i = 0; i < 6; i++)
            {
                if (count[i] != 9)
                {
                    return -1;
                }
            }

            FaceCube fc = new FaceCube(s);
            CubieCube cc = fc.toCubieCube();

            return cc.verify();
        }

        /// <summary>
        /// Generates a random cube. </summary>
        /// <returns> A random cube in the string representation. Each cube of the cube space has the same probability. </returns>
        public static string randomCube()
        {
            CubieCube cc = new CubieCube();
            Random gen = new Random();
            cc.setFlip((short)gen.Next(CoordCubeBuildTables.N_FLIP));
            cc.setTwist((short)gen.Next(CoordCubeBuildTables.N_TWIST));
            do
            {
                cc.setURFtoDLB(gen.Next(CoordCubeBuildTables.N_URFtoDLB));
                cc.setURtoBR(gen.Next(CoordCubeBuildTables.N_URtoBR));
            } while ((cc.edgeParity() ^ cc.cornerParity()) != 0);
            FaceCube fc = cc.toFaceCube();
            return fc.to_fc_String();
        }

        // Table serialization methods - not used in runtime mode
        // Disabled for .NET 10 compatibility (BinaryFormatter is obsolete)

        public static void SerializeTable(string filename, short[,] array)
        {
            // Not implemented - tables are built at runtime
            throw new NotSupportedException("Table serialization not supported. Use SearchRunTime instead.");
        }

        public static short[,] DeserializeTable(string filename)
        {
            // Not implemented - tables are built at runtime
            throw new NotSupportedException("Table deserialization not supported. Use SearchRunTime instead.");
        }

        public static void SerializeSbyteArray(string filename, sbyte[] array)
        {
            // Not implemented - tables are built at runtime
            throw new NotSupportedException("Table serialization not supported. Use SearchRunTime instead.");
        }

        public static sbyte[] DeserializeSbyteArray(string filename)
        {
            // Not implemented - tables are built at runtime
            throw new NotSupportedException("Table deserialization not supported. Use SearchRunTime instead.");
        }

        static void EnsureFolder(string path)
        {
            string? directoryName = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directoryName))
            {
                Directory.CreateDirectory(directoryName);
            }
        }
    }
}
