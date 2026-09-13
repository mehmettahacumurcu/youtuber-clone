using System.IO;
using System.Security.Cryptography;
using YouTuber.Launcher.Activation;

namespace YouTuber.Launcher.Repair;

public enum RepairReason { Missing, Corrupt, UnsafePath, Unexpected }
public sealed record RepairFinding(string Path, RepairReason Reason, ComponentFile? Expected);

public sealed class RepairScanner
{
    public async Task<IReadOnlyList<RepairFinding>> ScanAsync(string componentRoot, ComponentInventory inventory, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(componentRoot);
        ArgumentNullException.ThrowIfNull(inventory);
        inventory.Validate();
        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(componentRoot));
        var findings = new List<RepairFinding>();
        if (HasExistingReparsePoint(root, root))
        {
            return inventory.Files.Select(file => new RepairFinding(file.Path, RepairReason.UnsafePath, file)).ToArray();
        }
        var manifestPath = Path.Combine(root, "component-manifest.json");
        if (HasExistingReparsePoint(root, manifestPath))
        {
            findings.Add(new RepairFinding("component-manifest.json", RepairReason.UnsafePath, null));
        }
        else if (!File.Exists(manifestPath))
        {
            findings.Add(new RepairFinding("component-manifest.json", RepairReason.Missing, null));
        }
        else
        {
            try
            {
                var onDisk = ComponentInventory.Parse(await File.ReadAllTextAsync(manifestPath, cancellationToken));
                if (!InventoriesMatch(inventory, onDisk)) findings.Add(new RepairFinding("component-manifest.json", RepairReason.Corrupt, null));
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or ComponentActivationException)
            {
                findings.Add(new RepairFinding("component-manifest.json", RepairReason.Corrupt, null));
            }
        }
        foreach (var file in inventory.Files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var target = Path.GetFullPath(Path.Combine(root, file.Path.Replace('/', Path.DirectorySeparatorChar)));
            if (!IsWithin(root, target) || HasExistingReparsePoint(root, target))
            {
                findings.Add(new RepairFinding(file.Path, RepairReason.UnsafePath, file));
                continue;
            }
            if (!File.Exists(target))
            {
                findings.Add(new RepairFinding(file.Path, RepairReason.Missing, file));
                continue;
            }
            var info = new FileInfo(target);
            if (info.Length != file.Size || !await HashMatchesAsync(target, file.Sha256, cancellationToken))
                findings.Add(new RepairFinding(file.Path, RepairReason.Corrupt, file));
        }
        findings.AddRange(FindUnexpectedEntries(root, inventory));
        return findings;
    }

    private static bool InventoriesMatch(ComponentInventory expected, ComponentInventory actual)
    {
        if (expected.Schema != actual.Schema || expected.Component != actual.Component || expected.Version != actual.Version
            || expected.Compatibility.RuntimeApi != actual.Compatibility.RuntimeApi || expected.Entrypoint != actual.Entrypoint
            || !expected.Healthcheck.SequenceEqual(actual.Healthcheck, StringComparer.Ordinal)
            || expected.Files.Count != actual.Files.Count)
        {
            return false;
        }
        var actualFiles = actual.Files.ToDictionary(
            file => PathRules.PathKey(PathRules.NormalizeRelativePath(file.Path, "component file")),
            StringComparer.OrdinalIgnoreCase);
        return expected.Files.All(file => actualFiles.TryGetValue(PathRules.PathKey(PathRules.NormalizeRelativePath(file.Path, "component file")), out var candidate)
                                          && candidate.Size == file.Size
                                          && string.Equals(candidate.Sha256, file.Sha256, StringComparison.Ordinal));
    }

    private static IReadOnlyList<RepairFinding> FindUnexpectedEntries(string root, ComponentInventory inventory)
    {
        var findings = new List<RepairFinding>();
        var expectedFiles = inventory.Files
            .Select(file => PathRules.PathKey(PathRules.NormalizeRelativePath(file.Path, "component file")))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var expectedDirectories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var path in expectedFiles)
        {
            var parts = path.Split('/', StringSplitOptions.None);
            for (var index = 1; index < parts.Length; index++) expectedDirectories.Add(string.Join('/', parts[..index]));
        }

        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            var directory = pending.Pop();
            IEnumerable<string> entries;
            try { entries = Directory.EnumerateFileSystemEntries(directory).ToArray(); }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                findings.Add(new RepairFinding(Path.GetRelativePath(root, directory).Replace(Path.DirectorySeparatorChar, '/'), RepairReason.UnsafePath, null));
                continue;
            }
            foreach (var entry in entries)
            {
                var relative = Path.GetRelativePath(root, entry).Replace(Path.DirectorySeparatorChar, '/');
                string key;
                try { key = PathRules.PathKey(PathRules.NormalizeRelativePath(relative, "component tree entry")); }
                catch (Exception exception) when (exception is UnsafeArchiveException or ArgumentException)
                {
                    findings.Add(new RepairFinding(relative, RepairReason.UnsafePath, null));
                    continue;
                }
                FileAttributes attributes;
                try { attributes = File.GetAttributes(entry); }
                catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
                {
                    findings.Add(new RepairFinding(relative, RepairReason.UnsafePath, null));
                    continue;
                }
                if ((attributes & FileAttributes.ReparsePoint) != 0)
                {
                    findings.Add(new RepairFinding(relative, RepairReason.UnsafePath, null));
                    continue;
                }
                if ((attributes & FileAttributes.Directory) != 0)
                {
                    if (!expectedDirectories.Contains(key))
                    {
                        findings.Add(new RepairFinding(relative, RepairReason.Unexpected, null));
                        continue;
                    }
                    pending.Push(entry);
                    continue;
                }
                if (!expectedFiles.Contains(key) && !string.Equals(key, PathRules.PathKey("component-manifest.json"), StringComparison.OrdinalIgnoreCase))
                    findings.Add(new RepairFinding(relative, RepairReason.Unexpected, null));
            }
        }
        return findings;
    }

    private static async Task<bool> HashMatchesAsync(string path, string expected, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        var actual = Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken)).ToLowerInvariant();
        return CryptographicOperations.FixedTimeEquals(System.Text.Encoding.ASCII.GetBytes(actual), System.Text.Encoding.ASCII.GetBytes(expected));
    }

    private static bool HasExistingReparsePoint(string root, string target)
    {
        var pathRoot = Path.GetPathRoot(root) ?? throw new InvalidDataException("Repair root is not absolute.");
        var relative = Path.GetRelativePath(pathRoot, target);
        var current = pathRoot;
        foreach (var part in relative.Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            if (!Exists(current)) break;
            if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) return true;
        }
        return false;
    }

    private static bool Exists(string path) => File.Exists(path) || Directory.Exists(path);
    private static bool IsWithin(string root, string target)
    {
        var relative = Path.GetRelativePath(root, target);
        return relative != ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) && !Path.IsPathRooted(relative);
    }
}
