namespace RLDemo.Web.Services;

/// <summary>
/// Write a file so a concurrent reader never observes a half-written one: write to a sibling <c>.tmp</c> then
/// atomically rename it over the target (creating the directory if needed). The JSON stores (gallery, deck) use it
/// because a request may read the file while another request rewrites it.
/// </summary>
internal static class AtomicFile
{
    public static void Write(string path, string contents)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string temp = path + ".tmp";
        File.WriteAllText(temp, contents);
        File.Move(temp, path, overwrite: true);
    }
}
