using System.IO.Compression;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace YouTuber.Launcher.Activation;

public class ComponentActivationException : Exception
{
    public ComponentActivationException(string message, Exception? innerException = null) : base(message, innerException) { }
}

public sealed class UnsafeArchiveException : ComponentActivationException
{
    public UnsafeArchiveException(string message, Exception? innerException = null) : base(message, innerException) { }
}

public sealed record ArchiveExtractionLimits(int MaxEntries, double MaxCompressionRatio)
{
    public static ArchiveExtractionLimits Default { get; } = new(10_000, 200);

    public ArchiveExtractionLimits() : this(Default.MaxEntries, Default.MaxCompressionRatio) { }

    internal void Validate()
    {
        if (MaxEntries <= 0 || MaxCompressionRatio <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxEntries), "Archive extraction limits must be positive.");
        }
    }
}

public interface IArchiveExtractionHooks
{
    Task AfterArchiveVerifiedAsync(CancellationToken cancellationToken);
}

public static class ArchiveExtractor
{
    public static async Task VerifyOuterHashAsync(string archivePath, string expectedSha256, CancellationToken cancellationToken = default)
    {
        using var archive = await OpenVerifiedArchiveAsync(archivePath, expectedSha256, cancellationToken);
    }

    public static async Task<VerifiedArchive> OpenVerifiedArchiveAsync(string archivePath, string expectedSha256, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(archivePath);
        if (!File.Exists(archivePath) || !Path.IsPathFullyQualified(archivePath))
        {
            throw new ComponentActivationException("The component archive is unavailable.");
        }
        if ((File.GetAttributes(archivePath) & FileAttributes.ReparsePoint) != 0)
        {
            throw new ComponentActivationException("The component archive must not be a reparse point.");
        }

        if (!PathRules.IsSha256(expectedSha256))
        {
            throw new ComponentActivationException("The expected archive hash is invalid.");
        }

        var input = new FileStream(archivePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        var actual = Convert.ToHexString(await SHA256.HashDataAsync(input, cancellationToken)).ToLowerInvariant();
        if (!CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(actual), Encoding.ASCII.GetBytes(expectedSha256)))
        {
            await input.DisposeAsync();
            throw new ComponentActivationException("The component archive hash does not match the distribution manifest.");
        }

        input.Position = 0;
        try
        {
            return new VerifiedArchive(input, new ZipArchive(input, ZipArchiveMode.Read, leaveOpen: true));
        }
        catch (Exception exception) when (exception is InvalidDataException or IOException)
        {
            await input.DisposeAsync();
            throw new UnsafeArchiveException("The component archive is malformed.", exception);
        }
    }

    public static async Task ExtractAsync(
        string archivePath,
        string expectedSha256,
        string archiveRoot,
        string stagingDirectory,
        long expectedExpandedBytes,
        ArchiveExtractionLimits? limits = null,
        IArchiveExtractionHooks? hooks = null,
        CancellationToken cancellationToken = default)
    {
        using var archive = await OpenVerifiedArchiveAsync(archivePath, expectedSha256, cancellationToken);
        if (hooks is not null)
        {
            try { await hooks.AfterArchiveVerifiedAsync(cancellationToken); }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException) { throw new ComponentActivationException("The verified archive changed before extraction.", exception); }
        }
        await ExtractAsync(archive, archiveRoot, stagingDirectory, expectedExpandedBytes, limits, cancellationToken);
    }

    public static async Task ExtractAsync(
        VerifiedArchive archive,
        string archiveRoot,
        string stagingDirectory,
        long expectedExpandedBytes,
        ArchiveExtractionLimits? limits = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(archive);
        ArgumentException.ThrowIfNullOrWhiteSpace(stagingDirectory);
        limits ??= ArchiveExtractionLimits.Default;
        limits.Validate();
        if (expectedExpandedBytes <= 0) throw new ArgumentOutOfRangeException(nameof(expectedExpandedBytes), "The signed expanded size must be positive.");
        var normalizedRoot = PathRules.NormalizeRelativePath(archiveRoot, "archive root");
        EnsureSafeEmptyStagingDirectory(stagingDirectory);

        try
        {
            var entries = InspectEntries(archive.Entries, normalizedRoot, expectedExpandedBytes, limits);
            foreach (var entry in entries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var target = Path.Combine(stagingDirectory, entry.RelativePath.Replace('/', Path.DirectorySeparatorChar));
                EnsurePathRemainsWithin(stagingDirectory, target);
                if (entry.IsDirectory)
                {
                    Directory.CreateDirectory(target);
                    EnsureNotReparsePoint(target);
                    continue;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                EnsureNoReparseAncestors(Path.GetDirectoryName(target)!);
                await using var source = entry.Entry.Open();
                await using var destination = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1024 * 1024, FileOptions.Asynchronous | FileOptions.WriteThrough);
                await source.CopyToAsync(destination, 1024 * 1024, cancellationToken);
                await destination.FlushAsync(cancellationToken);
                destination.Flush(flushToDisk: true);
            }
        }
        catch (InvalidDataException exception)
        {
            throw new UnsafeArchiveException("The component archive is malformed.", exception);
        }
    }

    public sealed class VerifiedArchive(FileStream stream, ZipArchive archive) : IDisposable
    {
        private readonly FileStream _stream = stream;
        private readonly ZipArchive _archive = archive;

        internal IReadOnlyCollection<ZipArchiveEntry> Entries => _archive.Entries;

        public void Dispose()
        {
            _archive.Dispose();
            _stream.Dispose();
        }
    }

    private static IReadOnlyList<InspectedEntry> InspectEntries(
        IReadOnlyCollection<ZipArchiveEntry> zipEntries,
        string archiveRoot,
        long expectedExpandedBytes,
        ArchiveExtractionLimits limits)
    {
        var result = new List<InspectedEntry>();
        var normalizedEntries = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var files = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long expandedBytes = 0;
        var entryCount = 0;

        foreach (var entry in zipEntries)
        {
            var rawName = entry.FullName.TrimEnd('/');
            if (rawName.Length == 0) continue;
            if (++entryCount > limits.MaxEntries) throw new UnsafeArchiveException("The archive contains too many entries.");
            var normalizedName = PathRules.NormalizeRelativePath(rawName, "archive entry");
            if (!normalizedEntries.Add(normalizedName)) throw new UnsafeArchiveException("The archive contains duplicate Windows paths.");
            if (IsSymlinkOrReparseMetadata(entry)) throw new UnsafeArchiveException("The archive contains a symlink or reparse point.");
            if (!IsUnderArchiveRoot(normalizedName, archiveRoot, out var relativePath)) throw new UnsafeArchiveException("The archive contains an entry outside the declared archive root.");
            if (relativePath.Length == 0 && !entry.FullName.EndsWith("/", StringComparison.Ordinal)) throw new UnsafeArchiveException("The declared archive root must be a directory.");
            if (relativePath.Length == 0) continue;
            var isDirectory = entry.FullName.EndsWith("/", StringComparison.Ordinal);
            if (!isDirectory)
            {
                try { checked { expandedBytes += entry.Length; } }
                catch (OverflowException exception) { throw new UnsafeArchiveException("The archive expanded-size total overflows Int64.", exception); }
                if (expandedBytes > expectedExpandedBytes) throw new UnsafeArchiveException("The archive exceeds its signed expanded size.");
                if (entry.Length > 0 && (entry.CompressedLength == 0 || (double)entry.Length / entry.CompressedLength > limits.MaxCompressionRatio))
                {
                    throw new UnsafeArchiveException("The archive exceeds the compression-ratio limit.");
                }

                files.Add(normalizedName);
            }

            result.Add(new InspectedEntry(entry, relativePath, isDirectory));
        }

        if (result.Count == 0) throw new UnsafeArchiveException("The archive contains no files below its declared root.");
        if (expandedBytes != expectedExpandedBytes) throw new UnsafeArchiveException("The archive expanded size does not match the signed release size.");
        foreach (var entry in result)
        {
            var segments = PathRules.PathKey(entry.RelativePath).Split('/', StringSplitOptions.None);
            for (var index = 1; index < segments.Length; index++)
            {
                if (files.Contains(PathRules.PathKey(archiveRoot) + "/" + string.Join("/", segments[..index])))
                {
                    throw new UnsafeArchiveException("The archive contains a file ancestor conflict.");
                }
            }
        }

        return result;
    }

    private static bool IsUnderArchiveRoot(string name, string archiveRoot, out string relativePath)
    {
        if (string.Equals(name, archiveRoot, StringComparison.OrdinalIgnoreCase))
        {
            relativePath = string.Empty;
            return true;
        }

        var prefix = archiveRoot + "/";
        if (name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            relativePath = name[prefix.Length..];
            return true;
        }

        relativePath = string.Empty;
        return false;
    }

    private static bool IsSymlinkOrReparseMetadata(ZipArchiveEntry entry)
    {
        const int UnixFileTypeMask = 0xF000;
        const int UnixSymlink = 0xA000;
        const int DosReparsePoint = 0x0400;
        return (((entry.ExternalAttributes >> 16) & UnixFileTypeMask) == UnixSymlink) || ((entry.ExternalAttributes & 0xFFFF) & DosReparsePoint) != 0;
    }

    private static void EnsureSafeEmptyStagingDirectory(string stagingDirectory)
    {
        if (!Directory.Exists(stagingDirectory)) throw new UnsafeArchiveException("The staging directory must exist.");
        EnsureNoReparseAncestors(stagingDirectory);
        if (Directory.EnumerateFileSystemEntries(stagingDirectory).Any()) throw new UnsafeArchiveException("The staging directory must be empty.");
    }

    private static void EnsureNoReparseAncestors(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var root = Path.GetPathRoot(fullPath) ?? throw new UnsafeArchiveException("The staging path has no root.");
        var current = root;
        if (Directory.Exists(current) || File.Exists(current)) EnsureNotReparsePoint(current);
        foreach (var part in fullPath[root.Length..].Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            if (Directory.Exists(current) || File.Exists(current)) EnsureNotReparsePoint(current);
        }
    }

    private static void EnsureNotReparsePoint(string path)
    {
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new UnsafeArchiveException("The staging path contains a reparse point.");
    }

    private static void EnsurePathRemainsWithin(string stagingDirectory, string target)
    {
        var relative = Path.GetRelativePath(Path.GetFullPath(stagingDirectory), Path.GetFullPath(target));
        if (relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) || Path.IsPathRooted(relative))
        {
            throw new UnsafeArchiveException("The archive entry escapes the staging directory.");
        }
    }

    private sealed record InspectedEntry(ZipArchiveEntry Entry, string RelativePath, bool IsDirectory);
}
