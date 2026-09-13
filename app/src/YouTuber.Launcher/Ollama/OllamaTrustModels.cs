using System.Diagnostics;
using System.IO;
using System.ComponentModel;
using System.Net;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Distribution;

namespace YouTuber.Launcher.Ollama;

public sealed record OllamaPublication(
    InstallerArtifact Installer,
    string Publisher,
    string ReleaseVersion,
    string ReleaseIdentity,
    string ManifestIdentity,
    OllamaModelRolesMetadata Models)
{
    public void Validate()
    {
        Installer.Validate();
        var verifier = Models?.Verifier;
        if (!string.Equals(Publisher, OllamaInstallation.ExpectedPublisher, StringComparison.Ordinal) ||
            !string.Equals(Installer.ExpectedPublisher, Publisher, StringComparison.Ordinal) ||
            string.IsNullOrWhiteSpace(ReleaseVersion) || string.IsNullOrWhiteSpace(ReleaseIdentity) ||
            verifier is null || !string.Equals(verifier.Name, OllamaClient.VerifierModel, StringComparison.Ordinal) ||
            verifier.Source is null || verifier.Source.Size <= 0 || !IsSha256(verifier.Source.Sha256) ||
            verifier.Source.Revision is null || verifier.Source.Revision.Length != 40 || verifier.Source.Revision.Any(character => character is not (>= '0' and <= '9' or >= 'a' and <= 'f')) ||
            verifier.InstalledSize <= 0 || !IsSha256(verifier.InstalledDigest) ||
            Models!.Speaker is null || Models.Speaker.InstalledSize <= 0 ||
            ManifestIdentity.Length != 64 || ManifestIdentity.Any(character => character is not (>= '0' and <= '9' or >= 'a' and <= 'f')))
        {
            throw new InvalidOperationException("Ollama publication must be derived from one validated immutable manifest record.");
        }
    }

    private static bool IsSha256(string? value)
        => value is { Length: 64 } && value.Any(character => character != '0') &&
            value.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f');
}

public sealed record OllamaProcessReceipt(int ProcessId, DateTime StartTimeUtc, int SessionId, string ExecutablePath, string ReleaseIdentity);

public interface ITcpOwnerProbe
{
    int? GetListenerProcessId(Uri endpoint);
}

public sealed class WindowsTcpOwnerProbe : ITcpOwnerProbe
{
    public int? GetListenerProcessId(Uri endpoint)
    {
        OllamaInstallation.ValidateLoopbackEndpoint(endpoint);
        var size = 0;
        _ = GetExtendedTcpTable(IntPtr.Zero, ref size, true, 2, 3, 0);
        var table = Marshal.AllocHGlobal(size);
        try
        {
            if (GetExtendedTcpTable(table, ref size, true, 2, 3, 0) != 0) return null;
            var count = Marshal.ReadInt32(table);
            for (var index = 0; index < count; index++)
            {
                var row = IntPtr.Add(table, sizeof(int) + index * 24);
                var address = new IPAddress((uint)Marshal.ReadInt32(row, 4));
                var port = IPAddress.NetworkToHostOrder((short)Marshal.ReadInt32(row, 8));
                if (IPAddress.IsLoopback(address) && port == endpoint.Port) return Marshal.ReadInt32(row, 20);
            }
            return null;
        }
        finally { Marshal.FreeHGlobal(table); }
    }

    [DllImport("iphlpapi.dll", SetLastError = true)]
    private static extern uint GetExtendedTcpTable(IntPtr table, ref int size, bool order, int family, int tableClass, uint reserved);
}

public sealed class VerifiedOllamaExecutable : IAsyncDisposable
{
    private readonly NoFollowPathGuard _guard;
    private VerifiedOllamaExecutable(string path, NoFollowPathGuard guard, Version? signedFileVersion) { Path = path; _guard = guard; SignedFileVersion = signedFileVersion; }
    public string Path { get; }
    public Version? SignedFileVersion { get; }

    internal static VerifiedOllamaExecutable OpenForTest(string path)
        => new(System.IO.Path.GetFullPath(path), NoFollowPathGuard.Open(path), signedFileVersion: null);

    public static async Task<VerifiedOllamaExecutable> OpenAsync(string path, OllamaPublication publication, IWinVerifyTrust? trust = null, CancellationToken cancellationToken = default)
    {
        publication.Validate();
        var guard = NoFollowPathGuard.Open(path);
        try
        {
            var result = (trust ?? new WindowsWinVerifyTrust()).Verify(path);
            if (!result.IsTrusted || !string.Equals(result.Publisher, publication.Publisher, StringComparison.Ordinal)) throw new InvalidOperationException("Ollama executable publisher verification failed.");
            var info = FileVersionInfo.GetVersionInfo(path);
            var textVersion = info.ProductVersion ?? info.FileVersion;
            Version? version = null;
            if (!string.IsNullOrWhiteSpace(textVersion) && !Version.TryParse(textVersion.Split(' ', StringSplitOptions.RemoveEmptyEntries)[0].TrimStart('v'), out version)) throw new InvalidOperationException("Ollama executable signed file version is malformed.");
            if (version is not null && !string.Equals(version.ToString(), publication.ReleaseVersion, StringComparison.Ordinal)) throw new InvalidOperationException("Ollama executable release identity mismatched publication.");
            await Task.CompletedTask;
            return new VerifiedOllamaExecutable(path, guard, version);
        }
        catch { guard.Dispose(); throw; }
    }

    public Process StartServe(IReadOnlyDictionary<string, string> environment)
    {
        var start = CreateServeStartInfo(environment);
        var process = new Process { StartInfo = start };
        if (!process.Start()) throw new InvalidOperationException("Verified Ollama serve could not start.");
        return process;
    }

    internal ProcessStartInfo CreateServeStartInfo(IReadOnlyDictionary<string, string> environment)
    {
        _guard.EnsureUsable();
        var start = new ProcessStartInfo(Path) { UseShellExecute = false, CreateNoWindow = true, WindowStyle = ProcessWindowStyle.Hidden };
        start.ArgumentList.Add("serve");
        foreach (var entry in environment) start.Environment[entry.Key] = entry.Value;
        return start;
    }

    public async Task<string?> ReadVersionAsync(CancellationToken cancellationToken = default)
    {
        _guard.EnsureUsable();
        var start = new ProcessStartInfo(Path) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, WindowStyle = ProcessWindowStyle.Hidden };
        start.ArgumentList.Add("--version");
        using var process = new Process { StartInfo = start };
        if (!process.Start()) return null;
        var output = process.StandardOutput.ReadToEndAsync(cancellationToken);
        await process.WaitForExitAsync(cancellationToken);
        return process.ExitCode == 0 ? await output : null;
    }

    public ValueTask DisposeAsync() { _guard.Dispose(); return ValueTask.CompletedTask; }
}

internal sealed class NoFollowPathGuard : IDisposable
{
    private const uint GenericRead = 0x80000000;
    private const uint ShareRead = 0x00000001;
    private const uint OpenExisting = 3;
    private const uint FileFlagBackupSemantics = 0x02000000;
    private const uint FileFlagOpenReparsePoint = 0x00200000;
    private const uint FileAttributeReparsePoint = 0x00000400;
    private readonly List<SafeFileHandle> _handles;
    private NoFollowPathGuard(List<SafeFileHandle> handles) => _handles = handles;

    public static NoFollowPathGuard Open(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath) ?? throw new InvalidOperationException("Guarded path has no directory.");
        var handles = new List<SafeFileHandle>();
        try
        {
            var root = Path.GetPathRoot(directory) ?? throw new InvalidOperationException("Guarded path must be absolute.");
            var current = root;
            foreach (var part in directory[root.Length..].Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
            {
                current = Path.Combine(current, part);
                handles.Add(OpenHandle(current, FileFlagBackupSemantics | FileFlagOpenReparsePoint));
            }
            handles.Add(OpenHandle(fullPath, FileFlagOpenReparsePoint));
            return new NoFollowPathGuard(handles);
        }
        catch { foreach (var handle in handles) handle.Dispose(); throw; }
    }

    public static NoFollowPathGuard OpenDirectory(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var handles = new List<SafeFileHandle>();
        try
        {
            var root = Path.GetPathRoot(fullPath) ?? throw new InvalidOperationException("Guarded path must be absolute.");
            var current = root;
            foreach (var part in fullPath[root.Length..].Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
            {
                current = Path.Combine(current, part);
                handles.Add(OpenHandle(current, FileFlagBackupSemantics | FileFlagOpenReparsePoint));
            }
            return new NoFollowPathGuard(handles);
        }
        catch { foreach (var handle in handles) handle.Dispose(); throw; }
    }

    public static NoFollowPathGuard OpenAncestors(string path)
        => OpenDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? throw new InvalidOperationException("Guarded path has no directory."));

    private static SafeFileHandle OpenHandle(string path, uint flags)
    {
        var handle = CreateFile(path, GenericRead, ShareRead, IntPtr.Zero, OpenExisting, flags, IntPtr.Zero);
        if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to open guarded path.");
        if (GetFileAttributes(path) is uint attributes && (attributes & FileAttributeReparsePoint) != 0) { handle.Dispose(); throw new InvalidOperationException("Guarded paths may not traverse reparse points."); }
        return handle;
    }

    public void EnsureUsable() { if (_handles.Count == 0 || _handles.Any(handle => handle.IsClosed || handle.IsInvalid)) throw new ObjectDisposedException(nameof(NoFollowPathGuard)); }
    public void Dispose() { foreach (var handle in _handles) handle.Dispose(); _handles.Clear(); }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern SafeFileHandle CreateFile(string path, uint access, uint share, IntPtr securityAttributes, uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern uint GetFileAttributes(string path);
}
