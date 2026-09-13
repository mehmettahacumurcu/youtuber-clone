using System.IO;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace YouTuber.Launcher.Configuration;

/// <summary>
/// Stores the selected data root in a fixed per-user bootstrap location. The locator is deliberately
/// outside the selected tree so Repair, update, and uninstall can find custom roots after a restart.
/// </summary>
public sealed class DataRootLocator
{
    private const string Schema = "youtuber.data-root.v1";
    private const string BootstrapDirectoryName = "YouTuberStudioBootstrap";
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        WriteIndented = true,
    };

    private readonly string _stateRoot;
    private readonly string _lockFile;

    private DataRootLocator(string bootstrapRoot)
    {
        BootstrapRoot = AppPaths.NormalizeAndValidateDataRoot(bootstrapRoot);
        _stateRoot = Path.Combine(BootstrapRoot, "state");
        LocatorFile = Path.Combine(_stateRoot, "data-root.json");
        _lockFile = Path.Combine(_stateRoot, "data-root.lock");
    }

    public string BootstrapRoot { get; }

    public string LocatorFile { get; }

    public static DataRootLocator ForCurrentUser()
    {
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return ForBaseDirectory(localAppData);
    }

    public static DataRootLocator ForBaseDirectory(string localAppData)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(localAppData);
        if (!Path.IsPathFullyQualified(localAppData))
            throw new ArgumentException("The bootstrap base directory must be absolute.", nameof(localAppData));
        var normalized = Path.TrimEndingDirectorySeparator(Path.GetFullPath(localAppData));
        return new DataRootLocator(Path.Combine(normalized, BootstrapDirectoryName));
    }

    public string LoadRequired()
        => TryLoad() ?? throw new InvalidDataException("The data-root locator is missing.");

    public string? TryLoad()
    {
        EnsureBootstrapPathIsSafe();
        if (!File.Exists(LocatorFile))
        {
            if (Directory.Exists(LocatorFile))
                throw new InvalidDataException("The data-root locator is not a regular file.");
            return null;
        }

        if ((File.GetAttributes(LocatorFile) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0)
            throw new InvalidDataException("The data-root locator is not a regular file.");

        DataRootLocatorDocument document;
        try
        {
            using var stream = new FileStream(LocatorFile, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan);
            document = JsonSerializer.Deserialize<DataRootLocatorDocument>(stream, Json)
                ?? throw new InvalidDataException("The data-root locator is empty.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The data-root locator is invalid.", exception);
        }

        if (!string.Equals(document.Schema, Schema, StringComparison.Ordinal) || string.IsNullOrWhiteSpace(document.DataRoot))
            throw new InvalidDataException("The data-root locator schema is invalid.");

        var normalized = ValidateSelectedRoot(document.DataRoot, persisted: true);
        if (!string.Equals(document.DataRoot, normalized, StringComparison.Ordinal))
            throw new InvalidDataException("The persisted data root is not canonical.");
        return normalized;
    }

    public async Task SaveAsync(string dataRoot, CancellationToken cancellationToken = default)
    {
        var normalized = ValidateSelectedRoot(dataRoot, persisted: false);
        cancellationToken.ThrowIfCancellationRequested();

        EnsureBootstrapPathIsSafe();
        Directory.CreateDirectory(BootstrapRoot);
        RestrictDirectoryToCurrentUser(BootstrapRoot);
        EnsureBootstrapPathIsSafe();
        Directory.CreateDirectory(_stateRoot);
        RestrictDirectoryToCurrentUser(_stateRoot);
        EnsureBootstrapPathIsSafe();
        await using var locatorLock = await AcquireLockAsync(cancellationToken);
        var temporaryFile = LocatorFile + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            await using (var stream = new FileStream(
                temporaryFile,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                4096,
                FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await JsonSerializer.SerializeAsync(
                    stream,
                    new DataRootLocatorDocument(Schema, normalized),
                    Json,
                    cancellationToken);
                await stream.FlushAsync(cancellationToken);
                stream.Flush(flushToDisk: true);
            }

            EnsureBootstrapPathIsSafe();
            if (File.Exists(LocatorFile))
            {
                if ((File.GetAttributes(LocatorFile) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0)
                    throw new InvalidDataException("The data-root locator is not a regular file.");
                File.Replace(temporaryFile, LocatorFile, destinationBackupFileName: null);
            }
            else
                File.Move(temporaryFile, LocatorFile);
            RestrictFileToCurrentUser(LocatorFile);
        }
        finally
        {
            if (File.Exists(temporaryFile)) File.Delete(temporaryFile);
        }
    }

    public void Delete()
    {
        EnsureBootstrapPathIsSafe();
        DeleteRegularFileIfPresent(LocatorFile, "The data-root locator is not a regular file.");
        DeleteRegularFileIfPresent(_lockFile, "The data-root locator lock is not a regular file.");
        DeleteIfEmpty(_stateRoot);
        DeleteIfEmpty(BootstrapRoot);
    }

    private string ValidateSelectedRoot(string rawRoot, bool persisted)
    {
        try
        {
            var normalized = AppPaths.NormalizeAndValidateDataRoot(rawRoot);
            if (IsWithin(normalized, BootstrapRoot) || IsWithin(BootstrapRoot, normalized))
                throw new ArgumentException("The data root must be outside the fixed bootstrap locator tree.", nameof(rawRoot));
            return normalized;
        }
        catch (ArgumentException exception) when (persisted)
        {
            throw new InvalidDataException("The persisted data root is unsafe.", exception);
        }
    }

    private void EnsureBootstrapPathIsSafe()
    {
        try
        {
            _ = AppPaths.NormalizeAndValidateDataRoot(BootstrapRoot);
            _ = AppPaths.NormalizeAndValidateDataRoot(_stateRoot);
        }
        catch (ArgumentException exception)
        {
            throw new InvalidDataException("The bootstrap locator path is unsafe.", exception);
        }
    }

    private async Task<FileStream> AcquireLockAsync(CancellationToken cancellationToken)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return new FileStream(_lockFile, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.WriteThrough);
            }
            catch (IOException)
            {
                await Task.Delay(TimeSpan.FromMilliseconds(25), cancellationToken);
            }
        }
    }

    private static void DeleteRegularFileIfPresent(string path, string invalidMessage)
    {
        if (!File.Exists(path))
        {
            if (Directory.Exists(path)) throw new InvalidDataException(invalidMessage);
            return;
        }
        if ((File.GetAttributes(path) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0)
            throw new InvalidDataException(invalidMessage);
        File.Delete(path);
    }

    private static void DeleteIfEmpty(string path)
    {
        if (!Directory.Exists(path)) return;
        try { Directory.Delete(path, recursive: false); }
        catch (IOException) { }
    }

    private static void RestrictDirectoryToCurrentUser(string path)
    {
        if (!OperatingSystem.IsWindows()) return;
        var user = WindowsIdentity.GetCurrent().User
            ?? throw new InvalidOperationException("The current Windows user SID is unavailable.");
        var security = new DirectorySecurity();
        security.SetAccessRuleProtection(isProtected: true, preserveInheritance: false);
        security.AddAccessRule(new FileSystemAccessRule(
            user,
            FileSystemRights.FullControl,
            InheritanceFlags.ContainerInherit | InheritanceFlags.ObjectInherit,
            PropagationFlags.None,
            AccessControlType.Allow));
        new DirectoryInfo(path).SetAccessControl(security);
    }

    private static void RestrictFileToCurrentUser(string path)
    {
        if (!OperatingSystem.IsWindows()) return;
        var user = WindowsIdentity.GetCurrent().User
            ?? throw new InvalidOperationException("The current Windows user SID is unavailable.");
        var security = new FileSecurity();
        security.SetAccessRuleProtection(isProtected: true, preserveInheritance: false);
        security.AddAccessRule(new FileSystemAccessRule(user, FileSystemRights.FullControl, AccessControlType.Allow));
        new FileInfo(path).SetAccessControl(security);
    }

    private static bool IsWithin(string parent, string candidate)
    {
        var relative = Path.GetRelativePath(parent, candidate);
        return relative == "." || (!Path.IsPathRooted(relative)
            && relative != ".."
            && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal));
    }

    private sealed record DataRootLocatorDocument(string Schema, string DataRoot);
}
