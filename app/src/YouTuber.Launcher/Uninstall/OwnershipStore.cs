using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using YouTuber.Launcher.Activation;

namespace YouTuber.Launcher.Uninstall;

/// <summary>
/// Atomically records exact files and empty-only directories acquired by this installation.
/// It never infers ownership by recursively claiming an existing directory.
/// </summary>
public sealed class OwnershipStore
{
    private const string Schema = "youtuber.ownership.v1";
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        WriteIndented = true,
    };

    private readonly string _root;
    private readonly string _installId;
    private readonly string _stateRoot;
    private readonly string _manifestPath;
    private readonly string _lockPath;

    public OwnershipStore(string root, string installId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        if (!Guid.TryParse(installId, out var parsedInstallId))
            throw new ArgumentException("Install ID must be a GUID.", nameof(installId));
        _root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
        _installId = parsedInstallId.ToString("D");
        _stateRoot = Path.Combine(_root, "state");
        _manifestPath = Path.Combine(_stateRoot, "ownership.json");
        _lockPath = Path.Combine(_stateRoot, "ownership.lock");
    }

    public async Task RegisterExistingPathsAsync(IEnumerable<string> paths, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(paths);
        EnsureNoReparsePoint(_root);
        var additions = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var raw in paths)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentException.ThrowIfNullOrWhiteSpace(raw);
            var fullPath = Path.GetFullPath(raw);
            if (!IsWithin(_root, fullPath)) throw new InvalidDataException("Owned path escapes the data root.");
            if (!File.Exists(fullPath) && !Directory.Exists(fullPath))
                throw new FileNotFoundException("An owned path must exist before it can be registered.", fullPath);
            EnsureNoReparsePoint(fullPath);
            AddPathAndParents(additions, fullPath);
        }

        Directory.CreateDirectory(_stateRoot);
        EnsureNoReparsePoint(_stateRoot);
        await using var ownershipLock = await AcquireLockAsync(cancellationToken);
        additions.Add("state");
        additions.Add("state/ownership.lock");
        var existing = await LoadExistingAsync(cancellationToken);
        EnsureLockWasNotAdopted(ownershipLock, existing.Paths);
        existing.Paths.UnionWith(additions);
        var manifest = CreateManifest(existing.Paths, existing.TreeRoots);
        await AtomicWriteAsync(manifest, cancellationToken);
    }

    /// <summary>
    /// Durably declares a new, installation-exclusive directory before a child process can populate it.
    /// A later full-owned-data uninstall safely inventories that tree, including files written immediately before a crash.
    /// </summary>
    public async Task RegisterExclusiveTreeRootAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        EnsureNoReparsePoint(_root);
        var fullPath = Path.GetFullPath(path);
        if (!IsWithin(_root, fullPath)) throw new InvalidDataException("Owned tree root escapes the data root.");
        Directory.CreateDirectory(fullPath);
        EnsureNoReparsePoint(fullPath);
        Directory.CreateDirectory(_stateRoot);
        EnsureNoReparsePoint(_stateRoot);
        await using var ownershipLock = await AcquireLockAsync(cancellationToken);
        var existing = await LoadExistingAsync(cancellationToken);
        EnsureLockWasNotAdopted(ownershipLock, existing.Paths);
        var relative = PathRules.NormalizeRelativePath(Path.GetRelativePath(_root, fullPath).Replace(Path.DirectorySeparatorChar, '/'), "owned tree root");
        if (!existing.TreeRoots.Contains(relative) && Directory.EnumerateFileSystemEntries(fullPath).Any())
            throw new InvalidDataException("A new exclusive owned tree root must be empty before it is registered.");
        existing.TreeRoots.Add(relative);
        AddPathAndParents(existing.Paths, fullPath);
        existing.Paths.Add("state");
        existing.Paths.Add("state/ownership.lock");
        await AtomicWriteAsync(CreateManifest(existing.Paths, existing.TreeRoots), cancellationToken);
    }

    /// <summary>Durably reserves an absent exclusive tree so a subsequent atomic move is owned even if the process is lost immediately afterward.</summary>
    public async Task ReserveExclusiveTreeRootAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        EnsureNoReparsePoint(_root);
        var fullPath = Path.GetFullPath(path);
        if (!IsWithin(_root, fullPath)) throw new InvalidDataException("Reserved owned tree root escapes the data root.");
        var parent = Path.GetDirectoryName(fullPath) ?? throw new InvalidDataException("Reserved owned tree root has no parent.");
        Directory.CreateDirectory(parent);
        EnsureNoReparsePoint(parent);
        Directory.CreateDirectory(_stateRoot);
        EnsureNoReparsePoint(_stateRoot);
        await using var ownershipLock = await AcquireLockAsync(cancellationToken);
        var existing = await LoadExistingAsync(cancellationToken);
        EnsureLockWasNotAdopted(ownershipLock, existing.Paths);
        var relative = PathRules.NormalizeRelativePath(Path.GetRelativePath(_root, fullPath).Replace(Path.DirectorySeparatorChar, '/'), "reserved owned tree root");
        if (!existing.TreeRoots.Contains(relative) && (File.Exists(fullPath) || Directory.Exists(fullPath)))
            throw new InvalidDataException("A reserved exclusive owned tree root must not already exist.");
        existing.TreeRoots.Add(relative);
        AddPathAndParents(existing.Paths, fullPath);
        existing.Paths.Add("state");
        existing.Paths.Add("state/ownership.lock");
        await AtomicWriteAsync(CreateManifest(existing.Paths, existing.TreeRoots), cancellationToken);
    }

    /// <summary>Durably reserves one exact launcher download target before the downloader can create or replace it.</summary>
    public async Task ReserveOwnedFileAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        EnsureNoReparsePoint(_root);
        var fullPath = Path.GetFullPath(path);
        if (!IsWithin(_root, fullPath)) throw new InvalidDataException("Reserved owned file escapes the data root.");
        if (Directory.Exists(fullPath)) throw new InvalidDataException("A reserved owned file target cannot be a directory.");
        var parent = Path.GetDirectoryName(fullPath) ?? throw new InvalidDataException("Reserved owned file has no parent.");
        Directory.CreateDirectory(parent);
        EnsureNoReparsePoint(parent);
        Directory.CreateDirectory(_stateRoot);
        EnsureNoReparsePoint(_stateRoot);
        await using var ownershipLock = await AcquireLockAsync(cancellationToken);
        var existing = await LoadExistingAsync(cancellationToken);
        EnsureLockWasNotAdopted(ownershipLock, existing.Paths);
        var relative = PathRules.NormalizeRelativePath(Path.GetRelativePath(_root, fullPath).Replace(Path.DirectorySeparatorChar, '/'), "reserved owned file");
        if ((File.Exists(fullPath) || Directory.Exists(fullPath)) && !existing.Paths.Contains(relative))
            throw new InvalidDataException("A reserved owned file must not predate its ownership record.");
        if (File.Exists(fullPath)) EnsureNoReparsePoint(fullPath);
        AddPathAndParents(existing.Paths, fullPath);
        existing.Paths.Add("state");
        existing.Paths.Add("state/ownership.lock");
        await AtomicWriteAsync(CreateManifest(existing.Paths, existing.TreeRoots), cancellationToken);
    }

    private async Task<(HashSet<string> Paths, HashSet<string> TreeRoots)> LoadExistingAsync(CancellationToken cancellationToken)
    {
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var treeRoots = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!File.Exists(_manifestPath)) return (paths, treeRoots);
        EnsureNoReparsePoint(_manifestPath);
        OwnershipManifest manifest;
        try
        {
            await using var stream = new FileStream(_manifestPath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
            manifest = await JsonSerializer.DeserializeAsync<OwnershipManifest>(stream, Json, cancellationToken)
                ?? throw new InvalidDataException("Ownership state is empty.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("Ownership state is invalid.", exception);
        }

        if (manifest.Schema != Schema || !Guid.TryParse(manifest.InstallId, out var actual) || actual.ToString("D") != _installId)
            throw new InvalidDataException("Ownership identity does not match this installation.");
        if (manifest.OwnedRelativePaths is null) throw new InvalidDataException("Owned paths are missing.");
        foreach (var path in manifest.OwnedRelativePaths)
        {
            try { paths.Add(PathRules.NormalizeRelativePath(path, "owned path")); }
            catch (Exception exception) when (exception is UnsafeArchiveException or ArgumentException)
            {
                throw new InvalidDataException("Ownership state contains an unsafe path.", exception);
            }
        }
        foreach (var path in manifest.OwnedTreeRoots ?? [])
        {
            try { treeRoots.Add(PathRules.NormalizeRelativePath(path, "owned tree root")); }
            catch (Exception exception) when (exception is UnsafeArchiveException or ArgumentException)
            {
                throw new InvalidDataException("Ownership state contains an unsafe tree root.", exception);
            }
        }
        return (paths, treeRoots);
    }

    private OwnershipManifest CreateManifest(HashSet<string> paths, HashSet<string> treeRoots)
        => new(
            Schema,
            _installId,
            paths.Order(StringComparer.OrdinalIgnoreCase).ToArray(),
            treeRoots.Order(StringComparer.OrdinalIgnoreCase).ToArray());

    private async Task AtomicWriteAsync(OwnershipManifest manifest, CancellationToken cancellationToken)
    {
        var temporaryPath = _manifestPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            await using (var stream = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await JsonSerializer.SerializeAsync(stream, manifest, Json, cancellationToken);
                await stream.FlushAsync(cancellationToken);
                stream.Flush(flushToDisk: true);
            }
            if (File.Exists(_manifestPath)) File.Replace(temporaryPath, _manifestPath, destinationBackupFileName: null);
            else File.Move(temporaryPath, _manifestPath);
        }
        finally
        {
            if (File.Exists(temporaryPath)) File.Delete(temporaryPath);
        }
    }

    private async Task<AcquiredOwnershipLock> AcquireLockAsync(CancellationToken cancellationToken)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return new AcquiredOwnershipLock(
                    new FileStream(_lockPath, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.WriteThrough),
                    wasCreated: true);
            }
            catch (IOException) { }

            if (Directory.Exists(_lockPath))
                throw new InvalidDataException("The ownership lock target is not a regular file.");
            if (File.Exists(_lockPath) && (File.GetAttributes(_lockPath) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0)
                throw new InvalidDataException("The ownership lock target is not a regular file.");

            try
            {
                return new AcquiredOwnershipLock(
                    new FileStream(_lockPath, FileMode.Open, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.WriteThrough),
                    wasCreated: false);
            }
            catch (FileNotFoundException) { }
            catch (DirectoryNotFoundException) { }
            catch (IOException) { }
            await Task.Delay(TimeSpan.FromMilliseconds(25), cancellationToken);
        }
    }

    private static void EnsureLockWasNotAdopted(AcquiredOwnershipLock ownershipLock, HashSet<string> existingPaths)
    {
        if (!ownershipLock.WasCreated && !existingPaths.Contains("state/ownership.lock"))
            throw new InvalidDataException("A pre-existing ownership lock cannot be adopted.");
    }

    private sealed class AcquiredOwnershipLock(FileStream stream, bool wasCreated) : IAsyncDisposable
    {
        public bool WasCreated { get; } = wasCreated;
        public ValueTask DisposeAsync() => stream.DisposeAsync();
    }

    private void AddPathAndParents(HashSet<string> additions, string fullPath)
    {
        var current = fullPath;
        while (!string.Equals(current, _root, StringComparison.OrdinalIgnoreCase))
        {
            var relative = Path.GetRelativePath(_root, current).Replace(Path.DirectorySeparatorChar, '/');
            additions.Add(PathRules.NormalizeRelativePath(relative, "owned path"));
            current = Path.GetDirectoryName(current) ?? throw new InvalidDataException("Owned path has no parent.");
        }
    }

    private static bool IsWithin(string root, string target)
    {
        var relative = Path.GetRelativePath(root, target);
        return relative != ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) && !Path.IsPathRooted(relative);
    }

    private static void EnsureNoReparsePoint(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var root = Path.GetPathRoot(fullPath) ?? throw new InvalidDataException("Owned path has no filesystem root.");
        var current = root;
        foreach (var part in Path.GetRelativePath(root, fullPath).Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            if (!File.Exists(current) && !Directory.Exists(current)) break;
            if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                throw new InvalidDataException("Owned path crosses a reparse point.");
        }
    }
}
