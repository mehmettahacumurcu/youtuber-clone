using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using YouTuber.Launcher.Activation;

namespace YouTuber.Launcher.Uninstall;

public enum UninstallChoice { AppOnly, AppAndOwnedData }
public sealed record OwnedDirectoryDeletion(string Path, bool DeleteOnlyIfEmpty);
public sealed record OwnedDataDeletionPlan(
    IReadOnlyList<string> Files,
    IReadOnlyList<OwnedDirectoryDeletion> Directories,
    bool HasValidatedMatchingOwnership = false);
public sealed record OwnershipManifest(
    string Schema,
    string InstallId,
    IReadOnlyList<string> OwnedRelativePaths,
    IReadOnlyList<string>? OwnedTreeRoots = null);

public sealed class OwnedDataDeletionExecutor
{
    private static readonly JsonSerializerOptions ResidualJson = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
    };

    public void Execute(string root, OwnedDataDeletionPlan plan)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        ArgumentNullException.ThrowIfNull(plan);
        var fullRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
        var marker = Path.Combine(fullRoot, "state", "ownership.json");
        var files = plan.Files.Select(file => RequireOwnedTarget(fullRoot, file)).ToArray();
        var directories = plan.Directories.Select(directory => (Entry: directory, Target: RequireOwnedTarget(fullRoot, directory.Path))).ToArray();
        foreach (var target in files)
        {
            if (!File.Exists(target)) continue;
            if (HasReparsePoint(target) || (File.GetAttributes(target) & FileAttributes.Directory) != 0)
                throw new InvalidDataException("Owned file deletion target is no longer a regular file.");
        }
        foreach (var directory in directories)
        {
            if (!directory.Entry.DeleteOnlyIfEmpty) throw new InvalidDataException("Recursive owned-directory deletion is forbidden.");
            if (!Directory.Exists(directory.Target)) continue;
            if (HasReparsePoint(directory.Target))
                throw new InvalidDataException("Owned directory deletion target is a reparse point.");
        }
        foreach (var target in files.Where(file => !string.Equals(file, marker, StringComparison.OrdinalIgnoreCase)))
        {
            if (File.Exists(target)) File.Delete(target);
        }
        foreach (var directory in directories)
        {
            if (!Directory.Exists(directory.Target)) continue;
            try { Directory.Delete(directory.Target, recursive: false); }
            catch (IOException) { }
        }
        if (!files.Contains(marker, StringComparer.OrdinalIgnoreCase)) return;

        var residualReport = Path.Combine(fullRoot, "state", "uninstall-residual.json");
        WriteResidualReport(fullRoot, residualReport, FindResidualDirectories(directories, marker, residualReport));
        if (File.Exists(marker)) File.Delete(marker);

        RetryEmptyDirectoryDeletion(directories);
        var residuals = FindResidualDirectories(directories, marker, residualReport);
        if (residuals.Count > 0)
        {
            WriteResidualReport(fullRoot, residualReport, residuals);
            return;
        }

        if (File.Exists(residualReport)) File.Delete(residualReport);
        RetryEmptyDirectoryDeletion(directories);
    }

    private static void RetryEmptyDirectoryDeletion((OwnedDirectoryDeletion Entry, string Target)[] directories)
    {
        foreach (var directory in directories)
        {
            if (!Directory.Exists(directory.Target)) continue;
            try { Directory.Delete(directory.Target, recursive: false); }
            catch (IOException) { }
        }
    }

    private static IReadOnlyList<string> FindResidualDirectories(
        (OwnedDirectoryDeletion Entry, string Target)[] directories,
        string marker,
        string residualReport)
    {
        var residuals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var directory in directories)
        {
            if (!Directory.Exists(directory.Target)) continue;
            var hasUnexpectedEntry = Directory.EnumerateFileSystemEntries(directory.Target).Any(entry =>
                !string.Equals(entry, marker, StringComparison.OrdinalIgnoreCase)
                && !string.Equals(entry, residualReport, StringComparison.OrdinalIgnoreCase));
            if (hasUnexpectedEntry) residuals.Add(directory.Target);
        }
        return residuals.Order(StringComparer.OrdinalIgnoreCase).ToArray();
    }

    private static void WriteResidualReport(string root, string reportPath, IReadOnlyList<string> residualDirectories)
    {
        var stateRoot = Path.GetDirectoryName(reportPath) ?? throw new InvalidDataException("Residual report has no state directory.");
        Directory.CreateDirectory(stateRoot);
        var relativeDirectories = residualDirectories
            .Select(path => Path.GetRelativePath(root, path).Replace(Path.DirectorySeparatorChar, '/'))
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var temporaryPath = reportPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.WriteThrough))
            {
                JsonSerializer.Serialize(stream, new UninstallResidualReport("youtuber.uninstall-residual.v1", relativeDirectories), ResidualJson);
                stream.Flush(flushToDisk: true);
            }
            if (File.Exists(reportPath)) File.Replace(temporaryPath, reportPath, destinationBackupFileName: null);
            else File.Move(temporaryPath, reportPath);
        }
        finally
        {
            if (File.Exists(temporaryPath)) File.Delete(temporaryPath);
        }
    }

    private sealed record UninstallResidualReport(string Schema, IReadOnlyList<string> ResidualDirectories);

    private static string RequireOwnedTarget(string root, string target)
    {
        var fullTarget = Path.GetFullPath(target);
        var relative = Path.GetRelativePath(root, fullTarget);
        if (relative == "." || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) || Path.IsPathRooted(relative))
            throw new InvalidDataException("Owned deletion target escapes the data root.");
        return fullTarget;
    }

    private static bool HasReparsePoint(string target)
    {
        var root = Path.GetPathRoot(target) ?? throw new InvalidDataException("Owned deletion target is not absolute.");
        var current = root;
        foreach (var part in Path.GetRelativePath(root, target).Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            if (!File.Exists(current) && !Directory.Exists(current)) break;
            if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) return true;
        }
        return false;
    }
}

public sealed class OwnedDataInventory
{
    private const string Schema = "youtuber.ownership.v1";
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    public OwnedDataDeletionPlan Plan(string root, string installId, UninstallChoice choice)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        if (!Guid.TryParse(installId, out var expectedInstallId)) throw new ArgumentException("Install ID must be a GUID.", nameof(installId));
        if (choice == UninstallChoice.AppOnly) return Empty();
        var fullRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
        var marker = Path.Combine(fullRoot, "state", "ownership.json");
        if (!File.Exists(marker)) return Empty();
        if (HasReparsePoint(fullRoot, marker)) throw new InvalidDataException("Ownership state crosses a reparse point.");

        OwnershipManifest manifest;
        try { manifest = JsonSerializer.Deserialize<OwnershipManifest>(File.ReadAllText(marker), Json) ?? throw new InvalidDataException("Ownership state is empty."); }
        catch (JsonException exception) { throw new InvalidDataException("Ownership state is invalid.", exception); }
        if (manifest.Schema != Schema || !Guid.TryParse(manifest.InstallId, out var actualInstallId)) throw new InvalidDataException("Ownership identity is invalid.");
        if (actualInstallId != expectedInstallId) return Empty();
        if (manifest.OwnedRelativePaths is null) throw new InvalidDataException("Owned paths are missing.");

        var files = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var directories = new Dictionary<string, OwnedDirectoryDeletion>(StringComparer.OrdinalIgnoreCase);
        foreach (var raw in manifest.OwnedRelativePaths)
        {
            string normalized;
            try { normalized = PathRules.NormalizeRelativePath(raw, "owned path"); }
            catch (Exception exception) when (exception is UnsafeArchiveException or ArgumentException)
            {
                throw new InvalidDataException("Ownership state contains an unsafe path.", exception);
            }
            var target = Path.GetFullPath(Path.Combine(fullRoot, normalized.Replace('/', Path.DirectorySeparatorChar)));
            if (!IsWithin(fullRoot, target)) throw new InvalidDataException("Ownership state escapes the data root.");
            if (!File.Exists(target) && !Directory.Exists(target)) continue;
            if (HasReparsePoint(fullRoot, target))
                throw new InvalidDataException("Owned path crosses a reparse point.");
            if (File.Exists(target)) files.Add(target);
            else directories[target] = new OwnedDirectoryDeletion(target, DeleteOnlyIfEmpty: true);
        }
        foreach (var raw in manifest.OwnedTreeRoots ?? [])
        {
            string normalized;
            try { normalized = PathRules.NormalizeRelativePath(raw, "owned tree root"); }
            catch (Exception exception) when (exception is UnsafeArchiveException or ArgumentException)
            {
                throw new InvalidDataException("Ownership state contains an unsafe tree root.", exception);
            }
            var treeRoot = Path.GetFullPath(Path.Combine(fullRoot, normalized.Replace('/', Path.DirectorySeparatorChar)));
            if (!IsWithin(fullRoot, treeRoot)) throw new InvalidDataException("Owned tree root escapes the data root.");
            if (!Directory.Exists(treeRoot)) continue;
            if (HasReparsePoint(fullRoot, treeRoot)) throw new InvalidDataException("Owned tree root crosses a reparse point.");
            var pending = new Stack<string>();
            pending.Push(treeRoot);
            while (pending.Count > 0)
            {
                var directory = pending.Pop();
                directories[directory] = new OwnedDirectoryDeletion(directory, DeleteOnlyIfEmpty: true);
                foreach (var entry in Directory.EnumerateFileSystemEntries(directory, "*", SearchOption.TopDirectoryOnly))
                {
                    if ((File.GetAttributes(entry) & FileAttributes.ReparsePoint) != 0)
                        throw new InvalidDataException("Owned tree contains a reparse point.");
                    if (Directory.Exists(entry)) pending.Push(entry);
                    else files.Add(entry);
                }
            }
        }
        files.Add(marker);
        return new OwnedDataDeletionPlan(
            files.Order(StringComparer.OrdinalIgnoreCase).ToArray(),
            directories.Values.OrderByDescending(entry => entry.Path.Length).ToArray(),
            HasValidatedMatchingOwnership: true);
    }

    private static OwnedDataDeletionPlan Empty() => new([], []);
    private static bool HasReparsePoint(string root, string target)
    {
        var rootParent = Path.GetPathRoot(root) ?? throw new InvalidDataException("Ownership root is not absolute.");
        var current = rootParent;
        foreach (var part in Path.GetRelativePath(rootParent, target).Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            if (!File.Exists(current) && !Directory.Exists(current)) break;
            if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) return true;
        }
        return false;
    }
    private static bool IsWithin(string root, string target)
    {
        var relative = Path.GetRelativePath(root, target);
        return relative != ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) && !Path.IsPathRooted(relative);
    }
}
