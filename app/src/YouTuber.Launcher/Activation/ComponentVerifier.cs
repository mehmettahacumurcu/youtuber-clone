using System.Security.Cryptography;
using System.Text;
using System.IO;

namespace YouTuber.Launcher.Activation;

public static class ComponentVerifier
{
    public static async Task<ComponentInventory> VerifyAsync(string componentRoot, string expectedManifestHash, CancellationToken cancellationToken = default)
        => await VerifyAsync(componentRoot, expectedManifestHash, expectedInstallSize: null, cancellationToken);

    public static async Task<ComponentInventory> VerifyAsync(
        string componentRoot,
        string expectedManifestHash,
        long expectedInstallSize,
        CancellationToken cancellationToken = default)
        => await VerifyAsync(componentRoot, expectedManifestHash, (long?)expectedInstallSize, cancellationToken);

    private static async Task<ComponentInventory> VerifyAsync(
        string componentRoot,
        string expectedManifestHash,
        long? expectedInstallSize,
        CancellationToken cancellationToken)
    {
        if (!PathRules.IsSha256(expectedManifestHash)) throw new ComponentActivationException("The expected component manifest hash is invalid.");
        if (expectedInstallSize is <= 0) throw new ComponentActivationException("The signed component install size must be positive.");
        EnsureRealDirectory(componentRoot);
        var manifestPath = Path.Combine(componentRoot, "component-manifest.json");
        EnsureRegularFile(manifestPath);
        var manifestBytes = await File.ReadAllBytesAsync(manifestPath, cancellationToken);
        var manifestHash = Convert.ToHexString(SHA256.HashData(manifestBytes)).ToLowerInvariant();
        if (!CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(manifestHash), Encoding.ASCII.GetBytes(expectedManifestHash)))
        {
            throw new ComponentActivationException("The component manifest hash does not match the release manifest.");
        }

        var inventory = ComponentInventory.Parse(Encoding.UTF8.GetString(manifestBytes));
        if (expectedInstallSize is not null && inventory.InstallSize != expectedInstallSize.Value)
            throw new ComponentActivationException("The component inventory total does not match the signed install size.");
        var expected = inventory.Files.ToDictionary(file => PathRules.PathKey(PathRules.NormalizeRelativePath(file.Path, "component file")), StringComparer.OrdinalIgnoreCase);
        var found = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var foundDirectories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in EnumerateComponentPaths(componentRoot))
        {
            var relative = Path.GetRelativePath(componentRoot, item.Path).Replace(Path.DirectorySeparatorChar, '/');
            var key = PathRules.PathKey(PathRules.NormalizeRelativePath(relative, "component file"));
            if (item.IsDirectory)
            {
                if (!foundDirectories.Add(key)) throw new ComponentActivationException("The component tree contains case-colliding directories.");
            }
            else if (!found.TryAdd(key, item.Path))
            {
                throw new ComponentActivationException("The component tree contains case-colliding files.");
            }
        }

        var manifestKey = PathRules.PathKey("component-manifest.json");
        if (!found.Remove(manifestKey) || found.Count != expected.Count || found.Keys.Except(expected.Keys, StringComparer.OrdinalIgnoreCase).Any())
        {
            throw new ComponentActivationException("The component tree has missing or extra files.");
        }

        var expectedDirectories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var key in expected.Keys)
        {
            var parts = key.Split('/', StringSplitOptions.None);
            for (var index = 1; index < parts.Length; index++) expectedDirectories.Add(string.Join("/", parts[..index]));
        }
        if (!foundDirectories.SetEquals(expectedDirectories)) throw new ComponentActivationException("The component tree contains unlisted directories.");

        foreach (var (key, entry) in expected)
        {
            if (!found.TryGetValue(key, out var path)) throw new ComponentActivationException("The component tree has a missing inventoried file.");
            var metadata = new FileInfo(path);
            if (metadata.Length != entry.Size) throw new ComponentActivationException($"The component file size is invalid: {entry.Path}.");
            await using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
            var actual = Convert.ToHexString(await SHA256.HashDataAsync(input, cancellationToken)).ToLowerInvariant();
            if (!CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(actual), Encoding.ASCII.GetBytes(entry.Sha256))) throw new ComponentActivationException($"The component file hash is invalid: {entry.Path}.");
        }

        ApplyEntrypointPermissions(componentRoot, inventory.Entrypoint);
        return inventory;
    }

    private static IEnumerable<ComponentPath> EnumerateComponentPaths(string root)
    {
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            var directory = pending.Pop();
            EnsureRealDirectory(directory);
            foreach (var path in Directory.EnumerateFileSystemEntries(directory))
            {
                var attributes = File.GetAttributes(path);
                if ((attributes & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException("The component tree contains a symlink or reparse point.");
                if ((attributes & FileAttributes.Directory) != 0)
                {
                    yield return new ComponentPath(path, true);
                    pending.Push(path);
                }
                else
                {
                    if (!File.Exists(path)) throw new ComponentActivationException("The component tree contains a non-regular file.");
                    yield return new ComponentPath(path, false);
                }
            }
        }
    }

    private static void EnsureRealDirectory(string path)
    {
        if (!Directory.Exists(path) || (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException("The component root must be a real directory.");
    }

    private static void EnsureRegularFile(string path)
    {
        if (!File.Exists(path) || (File.GetAttributes(path) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0) throw new ComponentActivationException("The component manifest must be a regular file.");
    }

    private static void ApplyEntrypointPermissions(string componentRoot, string entrypoint)
    {
        var path = Path.Combine(componentRoot, entrypoint.Replace('/', Path.DirectorySeparatorChar));
        EnsureRegularFile(path);
        if (OperatingSystem.IsWindows())
        {
            File.SetAttributes(path, FileAttributes.Normal);
        }
        else
        {
            File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute | UnixFileMode.GroupRead | UnixFileMode.GroupExecute | UnixFileMode.OtherRead | UnixFileMode.OtherExecute);
        }
    }

    private sealed record ComponentPath(string Path, bool IsDirectory);
}
