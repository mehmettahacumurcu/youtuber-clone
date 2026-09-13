using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using YouTuber.Launcher.Uninstall;

namespace YouTuber.Launcher.Activation;

public sealed record DataAssetInstallRequest(
    string Component,
    string Version,
    string ArchivePath,
    string ArchiveRoot,
    string ManifestHash,
    string ArchiveSha256,
    string InstallRoot,
    long ExpandedSize,
    long InstallSize);

public sealed class DataAssetInstaller
{
    private readonly ActiveComponentsStore _store;
    private readonly OwnershipStore _ownership;

    public DataAssetInstaller(ActiveComponentsStore store, OwnershipStore? ownership = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _ownership = ownership ?? new OwnershipStore(store.DataRoot, UninstallPreparation.InstallId);
    }

    public async Task InstallAsync(DataAssetInstallRequest request, CancellationToken cancellationToken = default)
    {
        ValidateRequest(request);
        await using var activationLock = await _store.AcquireActivationLockAsync(cancellationToken);
        var relativeTarget = $"{request.InstallRoot}/{request.Component}/{request.Version}-{request.ManifestHash[..12]}";
        var target = Path.Combine(_store.DataRoot, relativeTarget.Replace('/', Path.DirectorySeparatorChar));
        var stagingRoot = Path.Combine(_store.DataRoot, "state", "asset-staging");
        var staging = Path.Combine(stagingRoot, $"{request.Component}.staging");
        EnsureNoReparseAncestors(_store.DataRoot, stagingRoot);
        Directory.CreateDirectory(stagingRoot);
        EnsureNoReparseAncestors(_store.DataRoot, stagingRoot);
        CleanupStaleStages(stagingRoot, request.Component);
        Directory.CreateDirectory(staging);
        await _ownership.RegisterExclusiveTreeRootAsync(staging, cancellationToken);
        var moved = false;
        try
        {
            using var archive = await ArchiveExtractor.OpenVerifiedArchiveAsync(request.ArchivePath, request.ArchiveSha256, cancellationToken);
            await ArchiveExtractor.ExtractAsync(archive, request.ArchiveRoot, staging, request.ExpandedSize, cancellationToken: cancellationToken);
            var inventory = await VerifyAsync(staging, request.ManifestHash, request.InstallSize, cancellationToken);
            if (!string.Equals(inventory.Component, request.Component, StringComparison.Ordinal) || !string.Equals(inventory.Version, request.Version, StringComparison.Ordinal))
                throw new ComponentActivationException("The data asset inventory does not match the requested component.");

            if (Directory.Exists(target))
            {
                var existing = await VerifyAsync(target, request.ManifestHash, request.InstallSize, cancellationToken);
                if (!string.Equals(existing.Component, request.Component, StringComparison.Ordinal) || !string.Equals(existing.Version, request.Version, StringComparison.Ordinal))
                    throw new ComponentActivationException("The installed data asset has the wrong identity.");
                DeleteTree(staging);
            }
            else
            {
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                await _ownership.ReserveExclusiveTreeRootAsync(target, cancellationToken);
                Directory.Move(staging, target);
                moved = true;
            }

            _ = await VerifyAsync(target, request.ManifestHash, request.InstallSize, cancellationToken);
            var active = await _store.LoadAsync(cancellationToken);
            var replacement = new Dictionary<string, ActiveComponentPointer>(active.Components, StringComparer.Ordinal)
            {
                [request.Component] = new(relativeTarget, request.ManifestHash),
            };
            await _store.SaveAsync(new ActiveComponents(replacement), cancellationToken);
            var owned = EnumerateTree(target).Append(target).Append(Path.Combine(_store.DataRoot, "state", "active-components.json")).ToList();
            if (IsWithinDataRoot(request.ArchivePath)) owned.Add(request.ArchivePath);
            await _ownership.RegisterExistingPathsAsync(owned, cancellationToken);
        }
        catch
        {
            if (!moved && Directory.Exists(staging)) DeleteTree(staging);
            throw;
        }
    }

    public static async Task<DataAssetInventory> VerifyAsync(string root, string expectedManifestHash, long expectedInstallSize, CancellationToken cancellationToken = default)
    {
        if (!PathRules.IsSha256(expectedManifestHash)) throw new ComponentActivationException("The expected data asset manifest hash is invalid.");
        if (expectedInstallSize <= 0) throw new ComponentActivationException("The signed data asset install size must be positive.");
        EnsureRealDirectory(root);
        var manifestPath = Path.Combine(root, "asset-manifest.json");
        EnsureRegularFile(manifestPath);
        var bytes = await File.ReadAllBytesAsync(manifestPath, cancellationToken);
        var actualManifestHash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        if (!CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(actualManifestHash), Encoding.ASCII.GetBytes(expectedManifestHash)))
            throw new ComponentActivationException("The data asset manifest hash does not match the distribution manifest.");
        var inventory = DataAssetInventory.Parse(Encoding.UTF8.GetString(bytes));
        if (inventory.InstallSize != expectedInstallSize)
            throw new ComponentActivationException("The data asset inventory total does not match the signed install size.");
        var expected = inventory.Files.ToDictionary(entry => PathRules.PathKey(PathRules.NormalizeRelativePath(entry.Path, "data asset file")), StringComparer.OrdinalIgnoreCase);
        var found = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var path in EnumerateTree(root).Where(File.Exists))
        {
            var relative = Path.GetRelativePath(root, path).Replace(Path.DirectorySeparatorChar, '/');
            var key = PathRules.PathKey(PathRules.NormalizeRelativePath(relative, "data asset file"));
            if (!found.TryAdd(key, path)) throw new ComponentActivationException("The data asset contains case-colliding files.");
        }
        if (!found.Remove(PathRules.PathKey("asset-manifest.json")) || !found.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase).SetEquals(expected.Keys))
            throw new ComponentActivationException("The data asset tree has missing or extra files.");
        foreach (var (key, entry) in expected)
        {
            var path = found[key];
            if (new FileInfo(path).Length != entry.Size) throw new ComponentActivationException($"The data asset file size is invalid: {entry.Path}.");
            await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
            var hash = Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken)).ToLowerInvariant();
            if (!CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(hash), Encoding.ASCII.GetBytes(entry.Sha256)))
                throw new ComponentActivationException($"The data asset file hash is invalid: {entry.Path}.");
        }
        return inventory;
    }

    private static void ValidateRequest(DataAssetInstallRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.InstallRoot is not ("models" or "rag") || string.IsNullOrWhiteSpace(request.Component) || !PathRules.IsSemVer(request.Version) ||
            !PathRules.IsSha256(request.ManifestHash) || !PathRules.IsSha256(request.ArchiveSha256) || !Path.IsPathFullyQualified(request.ArchivePath) ||
            request.ExpandedSize <= 0 || request.InstallSize <= 0)
            throw new ComponentActivationException("The data asset installation request is invalid.");
        _ = PathRules.NormalizeRelativePath(request.Component, "data asset component");
        _ = PathRules.NormalizeRelativePath(request.ArchiveRoot, "data asset archive root");
    }

    private bool IsWithinDataRoot(string path)
    {
        var relative = Path.GetRelativePath(_store.DataRoot, Path.GetFullPath(path));
        return relative != ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) && !Path.IsPathRooted(relative);
    }

    private static IEnumerable<string> EnumerateTree(string root)
    {
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            var directory = pending.Pop();
            EnsureRealDirectory(directory);
            foreach (var path in Directory.EnumerateFileSystemEntries(directory))
            {
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException("The data asset contains a reparse point.");
                yield return path;
                if (Directory.Exists(path)) pending.Push(path);
            }
        }
    }

    private static void DeleteTree(string root)
    {
        _ = EnumerateTree(root).ToArray();
        Directory.Delete(root, recursive: true);
    }

    private static void CleanupStaleStages(string stagingRoot, string component)
    {
        var deterministic = component + ".staging";
        var legacyPrefix = deterministic + "-";
        foreach (var candidate in Directory.EnumerateDirectories(stagingRoot, "*", SearchOption.TopDirectoryOnly))
        {
            var name = Path.GetFileName(candidate);
            var legacySuffix = name.StartsWith(legacyPrefix, StringComparison.Ordinal) ? name[legacyPrefix.Length..] : string.Empty;
            var isLegacyGuid = legacySuffix.Length == 32 && legacySuffix.All(Uri.IsHexDigit);
            if (string.Equals(name, deterministic, StringComparison.Ordinal) || isLegacyGuid)
                DeleteTree(candidate);
        }
    }

    private static void EnsureNoReparseAncestors(string root, string target)
    {
        try
        {
            var fullRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
            var fullTarget = Path.GetFullPath(target);
            var relative = Path.GetRelativePath(fullRoot, fullTarget);
            if (relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) || Path.IsPathRooted(relative))
                throw new ComponentActivationException("The data asset staging root escaped the selected data root.");
            var current = fullRoot;
            if (!Directory.Exists(current) || (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                throw new ComponentActivationException("The selected data root must be a real directory.");
            foreach (var part in relative.Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
            {
                current = Path.Combine(current, part);
                if (!Directory.Exists(current)) break;
                if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                    throw new ComponentActivationException("The data asset staging path cannot cross a reparse point.");
            }
        }
        catch (ComponentActivationException) { throw; }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            throw new ComponentActivationException("The data asset staging path could not be validated safely.", exception);
        }
    }

    private static void EnsureRealDirectory(string path)
    {
        if (!Directory.Exists(path) || (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
            throw new ComponentActivationException("The data asset root must be a real directory.");
    }

    private static void EnsureRegularFile(string path)
    {
        if (!File.Exists(path) || (File.GetAttributes(path) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0)
            throw new ComponentActivationException("The data asset manifest must be a regular file.");
    }
}

public sealed record DataAssetInventory(string Schema, string Component, string Version, IReadOnlyList<ComponentFile> Files)
{
    public long InstallSize => Files.Aggregate(0L, (total, file) => checked(total + file.Size));

    public static DataAssetInventory Parse(string json)
    {
        try
        {
            var inventory = JsonSerializer.Deserialize(json, DataAssetInventoryJsonContext.Default.DataAssetInventory)
                ?? throw new ComponentActivationException("The data asset inventory is empty.");
            if (inventory.Schema != "youtuber.asset.v1" || string.IsNullOrWhiteSpace(inventory.Component) || !PathRules.IsSemVer(inventory.Version) || inventory.Files is not { Count: > 0 })
                throw new ComponentActivationException("The data asset inventory contract is invalid.");
            var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var file in inventory.Files)
            {
                if (file is null || file.Size <= 0 || !PathRules.IsSha256(file.Sha256) || !paths.Add(PathRules.PathKey(PathRules.NormalizeRelativePath(file.Path, "data asset file"))))
                    throw new ComponentActivationException("The data asset inventory contains an invalid file.");
            }
            try { _ = inventory.InstallSize; }
            catch (OverflowException exception) { throw new ComponentActivationException("The data asset inventory size overflows Int64.", exception); }
            return inventory;
        }
        catch (ComponentActivationException) { throw; }
        catch (JsonException exception) { throw new ComponentActivationException("The data asset inventory is invalid.", exception); }
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
[JsonSerializable(typeof(DataAssetInventory))]
internal partial class DataAssetInventoryJsonContext : JsonSerializerContext { }
