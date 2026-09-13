using System.IO;
using System.Diagnostics;
using System.Net.Http;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;
using YouTuber.Launcher.Downloads;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Uninstall;
using YouTuber.Launcher.Distribution;

namespace YouTuber.Launcher.Ollama;

public sealed record OllamaImportArtifacts
{
    internal OllamaImportArtifacts(
        string verifierGgufPath,
        long verifierGgufSize,
        string verifierGgufSha256,
        string ggufPath,
        long ggufSize,
        string ggufSha256,
        string modelfilePath,
        long modelfileSize,
        string modelfileSha256)
    {
        VerifierGgufPath = verifierGgufPath;
        VerifierGgufSize = verifierGgufSize;
        VerifierGgufSha256 = verifierGgufSha256;
        GgufPath = ggufPath;
        GgufSize = ggufSize;
        GgufSha256 = ggufSha256;
        ModelfilePath = modelfilePath;
        ModelfileSize = modelfileSize;
        ModelfileSha256 = modelfileSha256;
    }

    public string VerifierGgufPath { get; init; }
    public long VerifierGgufSize { get; init; }
    public string VerifierGgufSha256 { get; init; }
    public string GgufPath { get; init; }
    public long GgufSize { get; init; }
    public string GgufSha256 { get; init; }
    public string ModelfilePath { get; init; }
    public long ModelfileSize { get; init; }
    public string ModelfileSha256 { get; init; }
}

public sealed class OllamaInstallerArtifact
{
    public OllamaInstallerArtifact(VerifiedInstaller installer, OllamaPublication publication)
    {
        ArgumentNullException.ThrowIfNull(installer);
        ArgumentNullException.ThrowIfNull(publication);
        publication.Validate();
        if (!string.Equals(installer.Artifact.ExpectedPublisher, publication.Installer.ExpectedPublisher, StringComparison.Ordinal) ||
            installer.Artifact.ExpectedSize != publication.Installer.ExpectedSize ||
            !string.Equals(installer.Artifact.ExpectedSha256, publication.Installer.ExpectedSha256, StringComparison.Ordinal) ||
            !string.Equals(installer.Artifact.Path, publication.Installer.Path, StringComparison.OrdinalIgnoreCase))
        {
            throw new InstallerVerificationException("The verified installer does not match the immutable Ollama publication record.");
        }
        Installer = installer;
        Publication = publication;
    }

    public VerifiedInstaller Installer { get; }
    public OllamaPublication Publication { get; }
}

public interface IOllamaModelsPathStore
{
    void Persist(string modelDirectory);
}

public sealed class UserEnvironmentOllamaModelsPathStore : IOllamaModelsPathStore
{
    public void Persist(string modelDirectory) => Environment.SetEnvironmentVariable("OLLAMA_MODELS", modelDirectory, EnvironmentVariableTarget.User);
}

public sealed class VerifiedOllamaServer : IAsyncDisposable
{
    internal VerifiedOllamaServer(VerifiedOllamaExecutable executable, Version version, string releaseIdentity) { Executable = executable; BinaryPath = executable.Path; Version = version; ReleaseIdentity = releaseIdentity; }
    public VerifiedOllamaServer(string binaryPath, Version version, string releaseIdentity) { BinaryPath = binaryPath; Version = version; ReleaseIdentity = releaseIdentity; }
    public string BinaryPath { get; }
    public Version Version { get; }
    public string ReleaseIdentity { get; }
    internal VerifiedOllamaExecutable? Executable { get; }
    public ValueTask DisposeAsync() => Executable?.DisposeAsync() ?? ValueTask.CompletedTask;
}

public interface IOllamaServerController
{
    Task SnapshotBeforeInstallAsync(CancellationToken cancellationToken = default);
    Task<OllamaProcessReceipt?> CaptureInstallerAutoStartAsync(VerifiedOllamaServer server, DateTime installerStartedUtc, DateTime installerFinishedUtc, CancellationToken cancellationToken = default);
    Task StopAutoStartedAsync(OllamaProcessReceipt receipt, CancellationToken cancellationToken = default);
    Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, CancellationToken cancellationToken = default);
    async Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, string requiredBinaryPath, CancellationToken cancellationToken = default)
    {
        var server = await LocateAndVerifyAsync(publication, cancellationToken);
        if (string.Equals(Path.GetFullPath(server.BinaryPath), Path.GetFullPath(requiredBinaryPath), StringComparison.OrdinalIgnoreCase)) return server;
        await server.DisposeAsync();
        throw new InvalidOperationException("The verified Ollama executable did not match the selected installation.");
    }
    Task StartServeAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default);
    Task WaitForHealthyAndConfirmStoreAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default);
    Task StopOwnedAsync(CancellationToken cancellationToken = default);
}

public interface IOllamaObservedProcess : IDisposable
{
    int Id { get; }
    DateTime StartTimeUtc { get; }
    int SessionId { get; }
    string ExecutablePath { get; }
    void Kill(bool entireProcessTree);
    Task WaitForExitAsync(CancellationToken cancellationToken);
}

public interface IOllamaProcessInventory
{
    IReadOnlyList<IOllamaObservedProcess> GetOllamaProcesses();
}

public sealed class WindowsOllamaProcessInventory : IOllamaProcessInventory
{
    public IReadOnlyList<IOllamaObservedProcess> GetOllamaProcesses()
        => Process.GetProcessesByName("ollama").Select(process => (IOllamaObservedProcess)new WindowsObservedProcess(process)).ToArray();

    private sealed class WindowsObservedProcess(Process process) : IOllamaObservedProcess
    {
        public int Id => process.Id;
        public DateTime StartTimeUtc => process.StartTime.ToUniversalTime();
        public int SessionId => process.SessionId;
        public string ExecutablePath => process.MainModule?.FileName ?? string.Empty;
        public void Kill(bool entireProcessTree) => process.Kill(entireProcessTree);
        public Task WaitForExitAsync(CancellationToken cancellationToken) => process.WaitForExitAsync(cancellationToken);
        public void Dispose() => process.Dispose();
    }
}

public interface IOllamaServeProcess : IJobProcess, IDisposable
{
    int Id { get; }
    bool HasExited { get; }
    DateTime StartTimeUtc { get; }
    int SessionId { get; }
    void Resume();
    void Kill(bool entireProcessTree);
    Task WaitForExitAsync(CancellationToken cancellationToken);
}

public interface IOllamaServeProcessLauncher
{
    IOllamaServeProcess Start(VerifiedOllamaServer server, IReadOnlyDictionary<string, string> environment);
}

public interface IOllamaBinaryLocator
{
    IReadOnlyList<string> FindCandidates();
}

/// <summary>Re-reads user and machine PATH on every lookup so a just-finished installer is visible to the parent launcher.</summary>
public sealed class OllamaBinaryLocator : IOllamaBinaryLocator
{
    private readonly Func<string?> _processPath;
    private readonly Func<string?> _userPath;
    private readonly Func<string?> _machinePath;
    private readonly Func<IEnumerable<string>> _fixedCandidates;

    public OllamaBinaryLocator(
        Func<string?>? processPath = null,
        Func<string?>? userPath = null,
        Func<string?>? machinePath = null,
        Func<IEnumerable<string>>? fixedCandidates = null)
    {
        _processPath = processPath ?? (() => Environment.GetEnvironmentVariable("PATH"));
        _userPath = userPath ?? (() => Environment.GetEnvironmentVariable("Path", EnvironmentVariableTarget.User));
        _machinePath = machinePath ?? (() => Environment.GetEnvironmentVariable("Path", EnvironmentVariableTarget.Machine));
        _fixedCandidates = fixedCandidates ?? KnownInstallCandidates;
    }

    public IReadOnlyList<string> FindCandidates()
    {
        var candidates = new List<string>();
        foreach (var path in new[] { _processPath(), _userPath(), _machinePath() })
        {
            if (string.IsNullOrWhiteSpace(path)) continue;
            foreach (var directory in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                candidates.Add(Path.Combine(directory, "ollama.exe"));
        }
        candidates.AddRange(_fixedCandidates());
        return candidates.Where(File.Exists).Select(Path.GetFullPath).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
    }

    private static IEnumerable<string> KnownInstallCandidates()
    {
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        if (!string.IsNullOrWhiteSpace(local))
        {
            yield return Path.Combine(local, "Programs", "Ollama", "ollama.exe");
            yield return Path.Combine(local, "Ollama", "ollama.exe");
        }
        if (!string.IsNullOrWhiteSpace(programFiles))
        {
            yield return Path.Combine(programFiles, "Ollama", "ollama.exe");
        }
    }
}

public sealed class ProcessOllamaServeProcessLauncher : IOllamaServeProcessLauncher
{
    public IOllamaServeProcess Start(VerifiedOllamaServer server, IReadOnlyDictionary<string, string> environment)
    {
        var executable = server.Executable ?? throw new InvalidOperationException("Ollama serve requires a held verified executable.");
        if (OperatingSystem.IsWindows()) return SuspendedWindowsOllamaServeProcess.Start(executable.CreateServeStartInfo(environment));
        return new ProcessOllamaServeProcess(executable.StartServe(environment));
    }

    private sealed class ProcessOllamaServeProcess(Process process) : IOllamaServeProcess
    {
        public int Id => process.Id;
        public bool HasExited => process.HasExited;
        public IntPtr NativeHandle => process.SafeHandle.DangerousGetHandle();
        public DateTime StartTimeUtc => process.StartTime.ToUniversalTime();
        public int SessionId => process.SessionId;
        public void Resume() { }
        public void Kill(bool entireProcessTree) => process.Kill(entireProcessTree);
        public Task WaitForExitAsync(CancellationToken cancellationToken) => process.WaitForExitAsync(cancellationToken);
        public void Dispose() => process.Dispose();
    }

    private sealed class SuspendedWindowsOllamaServeProcess(Process process, SafeFileHandle primaryThread) : IOllamaServeProcess
    {
        private readonly Process _process = process;
        private readonly SafeFileHandle _primaryThread = primaryThread;
        private bool _resumed;
        public int Id => _process.Id;
        public bool HasExited => _process.HasExited;
        public IntPtr NativeHandle => _process.SafeHandle.DangerousGetHandle();
        public DateTime StartTimeUtc => _process.StartTime.ToUniversalTime();
        public int SessionId => _process.SessionId;

        public static IOllamaServeProcess Start(ProcessStartInfo info)
        {
            var command = new StringBuilder(WindowsCommandLine.Quote(info.FileName) + string.Concat(info.ArgumentList.Select(argument => " " + WindowsCommandLine.Quote(argument))));
            var environment = string.Join('\0', info.Environment.OrderBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase).Select(pair => pair.Key + "=" + pair.Value)) + "\0\0";
            var environmentMemory = Marshal.StringToHGlobalUni(environment);
            try
            {
                var startup = new STARTUPINFO { cb = Marshal.SizeOf<STARTUPINFO>() };
                if (!CreateProcess(
                    info.FileName,
                    command,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    false,
                    CreateSuspended | CreateUnicodeEnvironment | CreateNoWindow,
                    environmentMemory,
                    Path.GetDirectoryName(info.FileName),
                    ref startup,
                    out var created))
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "Verified Ollama serve could not be created suspended.");
                }

                using var originalProcess = new SafeFileHandle(created.hProcess, ownsHandle: true);
                SafeFileHandle? primaryThread = new(created.hThread, ownsHandle: true);
                Process? process = null;
                try
                {
                    process = Process.GetProcessById(created.dwProcessId);
                    var result = new SuspendedWindowsOllamaServeProcess(process, primaryThread);
                    process = null;
                    primaryThread = null;
                    return result;
                }
                catch
                {
                    _ = TerminateProcess(originalProcess, 1);
                    _ = WaitForSingleObject(originalProcess, 5000);
                    process?.Dispose();
                    throw;
                }
                finally
                {
                    primaryThread?.Dispose();
                }
            }
            finally
            {
                Marshal.FreeHGlobal(environmentMemory);
            }
        }

        public void Resume()
        {
            if (_resumed) return;
            if (ResumeThread(_primaryThread) == uint.MaxValue)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Verified Ollama serve could not be resumed.");
            _resumed = true;
        }

        public void Kill(bool entireProcessTree)
        {
            if (!_process.HasExited) _process.Kill(entireProcessTree);
        }
        public Task WaitForExitAsync(CancellationToken cancellationToken) => _process.WaitForExitAsync(cancellationToken);
        public void Dispose() { _primaryThread.Dispose(); _process.Dispose(); }

        private const uint CreateSuspended = 0x00000004;
        private const uint CreateUnicodeEnvironment = 0x00000400;
        private const uint CreateNoWindow = 0x08000000;
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct STARTUPINFO
        {
            public int cb;
            public string? lpReserved;
            public string? lpDesktop;
            public string? lpTitle;
            public int dwX;
            public int dwY;
            public int dwXSize;
            public int dwYSize;
            public int dwXCountChars;
            public int dwYCountChars;
            public int dwFillAttribute;
            public int dwFlags;
            public short wShowWindow;
            public short cbReserved2;
            public IntPtr lpReserved2;
            public IntPtr hStdInput;
            public IntPtr hStdOutput;
            public IntPtr hStdError;
        }
        [StructLayout(LayoutKind.Sequential)]
        private struct PROCESS_INFORMATION { public IntPtr hProcess; public IntPtr hThread; public int dwProcessId; public int dwThreadId; }
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CreateProcess(string? applicationName, StringBuilder commandLine, IntPtr processAttributes, IntPtr threadAttributes, [MarshalAs(UnmanagedType.Bool)] bool inheritHandles, uint creationFlags, IntPtr environment, string? currentDirectory, ref STARTUPINFO startupInfo, out PROCESS_INFORMATION processInformation);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(SafeFileHandle thread);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool TerminateProcess(SafeFileHandle process, uint exitCode);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern uint WaitForSingleObject(SafeFileHandle handle, uint milliseconds);
    }
}

public sealed class OllamaServerController : IOllamaServerController, IDisposable
{
    private IOllamaServeProcess? _owned;
    private IWorkerJob? _ownedJob;
    private OllamaProcessReceipt? _ownedReceipt;
    private VerifiedOllamaServer? _ownedServer;
    private readonly HashSet<int> _preInstallPids = [];
    private readonly ITcpOwnerProbe _tcpOwnerProbe;
    private readonly IOllamaServeProcessLauncher _processLauncher;
    private readonly IOllamaProcessInventory _processInventory;
    private readonly IOllamaBinaryLocator _binaryLocator;
    private readonly Func<string, OllamaPublication, CancellationToken, Task<VerifiedOllamaServer>> _verifyServer;
    private readonly Func<IWorkerJob> _jobFactory;

    public OllamaServerController(
        ITcpOwnerProbe? tcpOwnerProbe = null,
        IOllamaServeProcessLauncher? processLauncher = null,
        IOllamaProcessInventory? processInventory = null,
        IOllamaBinaryLocator? binaryLocator = null,
        Func<string, OllamaPublication, CancellationToken, Task<VerifiedOllamaServer>>? verifyServer = null,
        Func<IWorkerJob>? jobFactory = null)
    {
        _tcpOwnerProbe = tcpOwnerProbe ?? new WindowsTcpOwnerProbe();
        _processLauncher = processLauncher ?? new ProcessOllamaServeProcessLauncher();
        _processInventory = processInventory ?? new WindowsOllamaProcessInventory();
        _binaryLocator = binaryLocator ?? new OllamaBinaryLocator();
        _verifyServer = verifyServer ?? VerifyServerAsync;
        _jobFactory = jobFactory ?? (() => new WindowsJob());
    }

    public Task SnapshotBeforeInstallAsync(CancellationToken cancellationToken = default)
    {
        _preInstallPids.Clear();
        foreach (var process in _processInventory.GetOllamaProcesses())
        {
            _preInstallPids.Add(process.Id);
            process.Dispose();
        }
        return Task.CompletedTask;
    }

    public Task<OllamaProcessReceipt?> CaptureInstallerAutoStartAsync(VerifiedOllamaServer server, DateTime installerStartedUtc, DateTime installerFinishedUtc, CancellationToken cancellationToken = default)
    {
        if (installerFinishedUtc < installerStartedUtc) throw new ArgumentOutOfRangeException(nameof(installerFinishedUtc));
        foreach (var process in _processInventory.GetOllamaProcesses())
        {
            try
            {
                if (_preInstallPids.Contains(process.Id) || process.SessionId != Process.GetCurrentProcess().SessionId ||
                    process.StartTimeUtc < installerStartedUtc || process.StartTimeUtc > installerFinishedUtc ||
                    !string.Equals(process.ExecutablePath, server.BinaryPath, StringComparison.OrdinalIgnoreCase)) continue;
                return Task.FromResult<OllamaProcessReceipt?>(new OllamaProcessReceipt(process.Id, process.StartTimeUtc, process.SessionId, process.ExecutablePath, server.ReleaseIdentity));
            }
            catch (Exception exception) when (exception is InvalidOperationException or Win32Exception) { }
            finally { process.Dispose(); }
        }
        return Task.FromResult<OllamaProcessReceipt?>(null);
    }

    public async Task StopAutoStartedAsync(OllamaProcessReceipt receipt, CancellationToken cancellationToken = default)
    {
        try
        {
            IOllamaObservedProcess? candidate = null;
            foreach (var process in _processInventory.GetOllamaProcesses())
            {
                if (process.Id == receipt.ProcessId) candidate = process;
                else process.Dispose();
            }
            using var observed = candidate;
            if (observed is null) return;
            if (observed.SessionId != receipt.SessionId || observed.StartTimeUtc != receipt.StartTimeUtc || !string.Equals(observed.ExecutablePath, receipt.ExecutablePath, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("The Ollama auto-start receipt no longer matches the candidate process.");
            }
            observed.Kill(entireProcessTree: true);
            await observed.WaitForExitAsync(cancellationToken);
        }
        catch (ArgumentException) { }
    }
    public async Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, CancellationToken cancellationToken = default)
    {
        publication.Validate();
        Exception? lastRejection = null;
        foreach (var binary in _binaryLocator.FindCandidates())
        {
            cancellationToken.ThrowIfCancellationRequested();
            try { return await _verifyServer(binary, publication, cancellationToken); }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }
            catch (Exception exception) when (exception is InvalidOperationException or IOException or UnauthorizedAccessException or Win32Exception)
            {
                lastRejection = exception;
            }
        }
        throw new InvalidOperationException("A manifest-authorized installed Ollama executable was not found in refreshed user, machine, or known install locations.", lastRejection);
    }

    public Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, string requiredBinaryPath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(requiredBinaryPath);
        if (!Path.IsPathFullyQualified(requiredBinaryPath)) throw new InvalidOperationException("The selected Ollama executable path must be absolute.");
        publication.Validate();
        return _verifyServer(Path.GetFullPath(requiredBinaryPath), publication, cancellationToken);
    }

    private static async Task<VerifiedOllamaServer> VerifyServerAsync(string binary, OllamaPublication publication, CancellationToken cancellationToken)
    {
        var executable = await VerifiedOllamaExecutable.OpenAsync(binary, publication, cancellationToken: cancellationToken);
        try
        {
            var version = await executable.ReadVersionAsync(cancellationToken) ?? throw new InvalidOperationException("Installed Ollama version was unavailable.");
            var token = version.Split(' ', StringSplitOptions.RemoveEmptyEntries).LastOrDefault()?.TrimStart('v');
            if (!Version.TryParse(token, out var parsed) || !string.Equals(parsed.ToString(), publication.ReleaseVersion, StringComparison.Ordinal)) throw new InvalidOperationException("Installed Ollama release version mismatched publication.");
            return new VerifiedOllamaServer(executable, parsed, publication.ReleaseIdentity);
        }
        catch { await executable.DisposeAsync(); throw; }
    }
    public async Task StartServeAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default)
    {
        if (_tcpOwnerProbe.GetListenerProcessId(installation.ApiEndpoint) is not null) throw new InvalidOperationException("The selected Ollama loopback port is already owned.");
        Directory.CreateDirectory(installation.ModelDirectory);
        File.WriteAllText(Marker(installation), server.ReleaseIdentity);
        IOllamaServeProcess? process = null;
        IWorkerJob? job = null;
        try
        {
            process = _processLauncher.Start(server, installation.Environment);
            job = _jobFactory();
            job.Assign(process);
            process.Resume();
            _owned = process;
            _ownedJob = job;
            _ownedReceipt = new OllamaProcessReceipt(process.Id, process.StartTimeUtc, process.SessionId, server.BinaryPath, server.ReleaseIdentity);
            _ownedServer = server;
        }
        catch
        {
            job?.Dispose();
            if (process is not null)
            {
                try
                {
                    if (!process.HasExited) process.Kill(true);
                    await process.WaitForExitAsync(CancellationToken.None);
                }
                catch { }
                process.Dispose();
            }
            throw;
        }
    }
    public async Task WaitForHealthyAndConfirmStoreAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default)
    {
        try
        {
            if (_owned is null || _ownedReceipt is null || _owned.HasExited || !File.Exists(Marker(installation)) || !string.Equals(await File.ReadAllTextAsync(Marker(installation), cancellationToken), server.ReleaseIdentity, StringComparison.Ordinal)) throw new InvalidOperationException("Controlled Ollama store confirmation failed.");
            using var client = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false }) { Timeout = TimeSpan.FromSeconds(2) };
            var uri = new Uri(installation.ApiEndpoint, "api/version");
            for (var attempt = 0; attempt < 120; attempt++)
            {
                try
                {
                    using var response = await client.GetAsync(uri, cancellationToken);
                    if (response.IsSuccessStatusCode && response.RequestMessage?.RequestUri == uri)
                    {
                        if (_tcpOwnerProbe.GetListenerProcessId(installation.ApiEndpoint) != _ownedReceipt.ProcessId) throw new InvalidOperationException("Ollama health endpoint is not owned by the started serve process.");
                        return;
                    }
                }
                catch (HttpRequestException) { }
                catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested) { }
                await Task.Delay(250, cancellationToken);
            }
            throw new TimeoutException("Controlled Ollama server did not become healthy.");
        }
        catch { await StopOwnedAsync(CancellationToken.None); throw; }
    }
    public async Task StopOwnedAsync(CancellationToken cancellationToken = default)
    {
        var process = _owned;
        var job = _ownedJob;
        var receipt = _ownedReceipt;
        _owned = null; _ownedJob = null; _ownedReceipt = null;
        try
        {
            if (process is not null && receipt is not null && !process.HasExited && process.Id == receipt.ProcessId && process.StartTimeUtc == receipt.StartTimeUtc)
            {
                process.Kill(true);
                await process.WaitForExitAsync(cancellationToken);
            }
        }
        finally
        {
            job?.Dispose();
            process?.Dispose();
            if (_ownedServer is not null) await _ownedServer.DisposeAsync();
            _ownedServer = null;
        }
    }
    public void Dispose() => StopOwnedAsync(CancellationToken.None).GetAwaiter().GetResult();
    private static string Marker(OllamaInstallation installation) => Path.Combine(installation.ModelDirectory, ".youtuber-ollama-store");
}

public sealed class OllamaManager
{
    private readonly Version _minimumVersion;
    private readonly string? _importRoot;
    private readonly IOllamaModelsPathStore _modelsPathStore;
    private readonly IOllamaServerController _serverController;
    private OwnershipStore? _ownership;
    private string? _controlledModelDirectory;
    private string? _controlledBinaryPath;
    private Version? _controlledVersion;

    public OllamaManager(Version? minimumVersion = null, string? importRoot = null, IOllamaModelsPathStore? modelsPathStore = null, IOllamaServerController? serverController = null, OwnershipStore? ownership = null)
    {
        _minimumVersion = minimumVersion ?? new Version(0, 1, 0);
        _importRoot = importRoot is null ? null : Path.GetFullPath(importRoot);
        _modelsPathStore = modelsPathStore ?? new UserEnvironmentOllamaModelsPathStore();
        _serverController = serverController ?? new OllamaServerController();
        _ownership = ownership;
    }

    public OllamaInstallation SelectInstallation(OllamaInstallation? existing, string dataRoot)
    {
        if (existing is null)
        {
            var selected = OllamaInstallation.New(dataRoot);
            ConfigureControlledOwnership(selected);
            return selected;
        }
        _controlledModelDirectory = null;
        if (!existing.IsExisting || existing.Version is null || existing.Version < _minimumVersion) throw new InvalidOperationException("The installed Ollama version is incompatible.");
        return existing;
    }

    public OllamaInstallation SelectInstallation(OllamaInstallation? existing, OllamaInstallation composed)
    {
        ArgumentNullException.ThrowIfNull(composed);
        if (composed.IsExisting) throw new InvalidOperationException("The composed Ollama runtime must use launcher-owned settings.");
        if (existing is null)
        {
            ConfigureControlledOwnership(composed);
            return composed;
        }
        if (!existing.IsExisting || !existing.BinaryTrusted || string.IsNullOrWhiteSpace(existing.BinaryPath) ||
            existing.Version is null || existing.Version < _minimumVersion)
        {
            throw new InvalidOperationException("The installed Ollama version is incompatible or untrusted.");
        }

        _controlledModelDirectory = null;
        return composed with
        {
            IsExisting = true,
            BinaryPath = existing.BinaryPath,
            Version = existing.Version,
            ApiReachable = false,
            BinaryTrusted = true,
        };
    }

    /// <summary>Starts the already-composed launcher-owned Ollama installation without selecting a default port.</summary>
    public async Task StartControlledServeAsync(VerifiedOllamaServer server, OllamaInstallation installation, PortReservation reservation, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(installation);
        ArgumentNullException.ThrowIfNull(reservation);
        if (installation.ApiEndpoint.Port != reservation.Port) throw new InvalidOperationException("The Ollama installation and its held loopback reservation differ.");
        cancellationToken.ThrowIfCancellationRequested();
        reservation.Release();
        await _serverController.StartServeAsync(server, installation, cancellationToken);
    }

    /// <summary>Stops only the launcher-owned controlled Ollama process.</summary>
    public Task StopControlledServeAsync(CancellationToken cancellationToken = default)
        => _serverController.StopOwnedAsync(cancellationToken);

    public async Task StartInstalledControlledAsync(OllamaPublication publication, OllamaInstallation installation, PortReservation reservation, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(publication);
        var server = installation.IsExisting
            ? await _serverController.LocateAndVerifyAsync(
                publication,
                installation.BinaryPath ?? throw new InvalidOperationException("The selected existing Ollama installation has no executable path."),
                cancellationToken)
            : await _serverController.LocateAndVerifyAsync(publication, cancellationToken);
        try
        {
            await StartControlledServeAsync(server, installation, reservation, cancellationToken);
            await _serverController.WaitForHealthyAndConfirmStoreAsync(server, installation, cancellationToken);
            RememberControlledServer(server);
        }
        catch
        {
            await _serverController.StopOwnedAsync(CancellationToken.None);
            await server.DisposeAsync();
            throw;
        }
    }

    public OllamaClient CreateControlledClient(OllamaInstallation installation, IWinVerifyTrust? trust = null)
    {
        if (_controlledBinaryPath is null || _controlledVersion is null)
            throw new InvalidOperationException("The launcher-owned Ollama server has not been verified and started.");
        var trusted = new OllamaInstallation(true, _controlledBinaryPath, installation.ApiEndpoint, _controlledVersion, installation.ModelDirectory, installation.Environment, true, true);
        return OllamaClient.FromInstallation(trusted, trust: trust);
    }

    public Task<InstallerRunResult> InstallNewAsync(OllamaInstallerArtifact artifact, OllamaInstallation installation, IInstallerRunner runner, CancellationToken cancellationToken = default)
        => InstallNewCoreAsync(artifact, installation, runner, null, cancellationToken);

    /// <summary>
    /// Installs and starts the launcher-owned Ollama instance from a single reserved runtime composition.
    /// The reservation is retained until every verification and installer step has succeeded.
    /// </summary>
    public Task<InstallerRunResult> InstallNewComposedAsync(
        OllamaInstallerArtifact artifact,
        OllamaInstallation installation,
        PortReservation reservation,
        IInstallerRunner runner,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(reservation);
        if (installation.ApiEndpoint.Port != reservation.Port)
        {
            throw new InvalidOperationException("The composed Ollama installation and reservation differ.");
        }
        return InstallNewCoreAsync(artifact, installation, runner, reservation, cancellationToken);
    }

    private async Task<InstallerRunResult> InstallNewCoreAsync(
        OllamaInstallerArtifact artifact,
        OllamaInstallation installation,
        IInstallerRunner runner,
        PortReservation? reservation,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        ArgumentNullException.ThrowIfNull(installation);
        ArgumentNullException.ThrowIfNull(runner);
        if (installation.IsExisting || !installation.Environment.TryGetValue("OLLAMA_MODELS", out var modelDirectory) || !string.Equals(modelDirectory, installation.ModelDirectory, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Only a new Ollama installation may set its initial model directory.");
        }

        ConfigureControlledOwnership(installation);
        var ownershipBaseline = SnapshotControlledModelTree(installation, cancellationToken);
        Directory.CreateDirectory(modelDirectory);
        await _ownership!.RegisterExclusiveTreeRootAsync(modelDirectory, cancellationToken);
        await RegisterControlledModelDeltaAsync(installation, ownershipBaseline, cancellationToken);
        await _serverController.SnapshotBeforeInstallAsync(cancellationToken);
        var installerStartedUtc = DateTime.UtcNow;
        var result = await runner.RunAsync(artifact.Installer, ["/VERYSILENT", "/SUPPRESSMSGBOXES"], TimeSpan.FromMinutes(5), installation.Environment, cancellationToken);
        var installerFinishedUtc = DateTime.UtcNow;
        if (result.TimedOut || result.ExitCode != 0) return result;
        var server = await _serverController.LocateAndVerifyAsync(artifact.Publication, cancellationToken);
        try
        {
            var autoStarted = await _serverController.CaptureInstallerAutoStartAsync(server, installerStartedUtc, installerFinishedUtc, cancellationToken);
            if (autoStarted is not null) await _serverController.StopAutoStartedAsync(autoStarted, cancellationToken);
            if (reservation is null)
            {
                await _serverController.StartServeAsync(server, installation, cancellationToken);
            }
            else
            {
                await StartControlledServeAsync(server, installation, reservation, cancellationToken);
            }
            await _serverController.WaitForHealthyAndConfirmStoreAsync(server, installation, cancellationToken);
            RememberControlledServer(server);
            await RegisterControlledModelDeltaAsync(installation, ownershipBaseline, cancellationToken);
        }
        catch { await _serverController.StopOwnedAsync(CancellationToken.None); await server.DisposeAsync(); throw; }
        return result;
    }

    private void RememberControlledServer(VerifiedOllamaServer server)
    {
        _controlledBinaryPath = server.BinaryPath;
        _controlledVersion = server.Version;
    }

    public async Task EnsureModelsAsync(OllamaClient client, OllamaInstallation installation, OllamaImportArtifacts import, OllamaModelRolesMetadata roles, Func<OllamaPullProgress, CancellationToken, Task>? reportProgress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(client);
        ArgumentNullException.ThrowIfNull(installation);
        ArgumentNullException.ThrowIfNull(roles);
        if (roles.Verifier is null || !string.Equals(roles.Verifier.Name, OllamaClient.VerifierModel, StringComparison.Ordinal) || roles.Speaker is null)
            throw new InvalidOperationException("Ollama model roles were not validated by the signed distribution contract.");
        EnsureImportMatchesRoles(import, roles);
        if (string.IsNullOrWhiteSpace(_importRoot)) throw new InvalidOperationException("Ollama import artifacts require a launcher-owned staging root.");
        ConfigureControlledOwnership(installation);
        var ownershipBaseline = SnapshotControlledModelTree(installation, cancellationToken);
        EnsureWithinImportRoot(import.VerifierGgufPath);
        EnsureWithinImportRoot(import.GgufPath);
        EnsureWithinImportRoot(import.ModelfilePath);
        using var guardedImports = await OpenVerifiedImportsAsync(import, cancellationToken);
        await client.EnsureHealthyAsync(cancellationToken);
        using (var verifierModelfile = GeneratedVerifierModelfile.Create(import.VerifierGgufPath))
        {
            await client.CreateModelAsync(roles.Verifier.Name, verifierModelfile.Path, installation.Environment, roles, cancellationToken);
        }
        await client.CreateModelAsync(roles.Speaker.Name, import.ModelfilePath, installation.Environment, roles, cancellationToken);
        var models = await client.ListModelDetailsAsync(cancellationToken);
        if (!ContainsExactModel(models, roles.Verifier.Name, roles.Verifier.InstalledDigest, roles.Verifier.InstalledSize, allowImplicitLatestTag: false) ||
            !ContainsExactModel(models, roles.Speaker.Name, expectedDigest: null, roles.Speaker.InstalledSize, allowImplicitLatestTag: true))
            throw new InvalidOperationException("Ollama did not report both required signed model identities.");
        await RegisterControlledModelDeltaAsync(installation, ownershipBaseline, cancellationToken);
        guardedImports.DeleteTemporaryGgufs();
    }

    private static bool ContainsExactModel(IReadOnlyList<OllamaModelInfo> models, string name, string? expectedDigest, long expectedSize, bool allowImplicitLatestTag)
    {
        var matches = models.Where(model => OllamaClient.MatchesCanonicalModelName(model.Name, name, allowImplicitLatestTag)).ToArray();
        return matches.Length == 1 && matches[0].Size == expectedSize &&
            (expectedDigest is null || string.Equals(matches[0].Digest, expectedDigest, StringComparison.Ordinal));
    }

    private static void EnsureImportMatchesRoles(OllamaImportArtifacts import, OllamaModelRolesMetadata roles)
    {
        ArgumentNullException.ThrowIfNull(import);
        if (import.VerifierGgufSize != roles.Verifier.Source.Size ||
            !string.Equals(import.VerifierGgufSha256, roles.Verifier.Source.Sha256, StringComparison.Ordinal) ||
            import.GgufSize != roles.Speaker.Gguf.Size ||
            !string.Equals(import.GgufSha256, roles.Speaker.Gguf.Sha256, StringComparison.Ordinal) ||
            import.ModelfileSize != roles.Speaker.Modelfile.Size ||
            !string.Equals(import.ModelfileSha256, roles.Speaker.Modelfile.Sha256, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Ollama import artifacts do not match the signed distribution contract.");
        }
    }

    private void ConfigureControlledOwnership(OllamaInstallation installation)
    {
        if (installation.IsExisting)
        {
            _controlledModelDirectory = null;
            return;
        }
        var modelDirectory = Path.TrimEndingDirectorySeparator(Path.GetFullPath(installation.ModelDirectory));
        var modelsDirectory = Directory.GetParent(modelDirectory);
        var dataRoot = modelsDirectory?.Parent;
        if (modelsDirectory is null || dataRoot is null ||
            !string.Equals(modelsDirectory.Name, "models", StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(Path.GetFileName(modelDirectory), "ollama", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("The controlled Ollama store must be under the YouTuber data root.");
        }
        _ownership ??= new OwnershipStore(dataRoot.FullName, UninstallPreparation.InstallId);
        _controlledModelDirectory = modelDirectory;
    }

    private HashSet<string> SnapshotControlledModelTree(OllamaInstallation installation, CancellationToken cancellationToken)
    {
        if (_ownership is null || _controlledModelDirectory is null || installation.IsExisting ||
            !string.Equals(Path.TrimEndingDirectorySeparator(Path.GetFullPath(installation.ModelDirectory)), _controlledModelDirectory, StringComparison.OrdinalIgnoreCase) ||
            !Directory.Exists(_controlledModelDirectory)) return new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        var snapshot = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { _controlledModelDirectory };
        var pending = new Stack<string>();
        pending.Push(_controlledModelDirectory);
        while (pending.Count > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var path in Directory.EnumerateFileSystemEntries(pending.Pop(), "*", SearchOption.TopDirectoryOnly))
            {
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidOperationException("The controlled Ollama store contains a reparse point.");
                snapshot.Add(path);
                if (Directory.Exists(path)) pending.Push(path);
            }
        }
        return snapshot;
    }

    private async Task RegisterControlledModelDeltaAsync(OllamaInstallation installation, HashSet<string> baseline, CancellationToken cancellationToken)
    {
        var current = SnapshotControlledModelTree(installation, cancellationToken);
        var acquired = current.Except(baseline, StringComparer.OrdinalIgnoreCase).ToArray();
        if (acquired.Length > 0) await _ownership!.RegisterExistingPathsAsync(acquired, cancellationToken);
    }

    private static async Task VerifyArtifactsAsync(OllamaImportArtifacts import, CancellationToken cancellationToken)
    {
        await using var gguf = new FileStream(import.GgufPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
        await using var modelfile = new FileStream(import.ModelfilePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
        if (gguf.Length != import.GgufSize || !await HashMatchesAsync(gguf, import.GgufSha256, cancellationToken)) throw new InvalidOperationException("The Speaker GGUF failed verification.");
        if (modelfile.Length != import.ModelfileSize || !await HashMatchesAsync(modelfile, import.ModelfileSha256, cancellationToken)) throw new InvalidOperationException("The Speaker Modelfile failed verification.");
    }

    private static async Task<bool> HashMatchesAsync(Stream stream, string expectedHash, CancellationToken cancellationToken)
    {
        stream.Position = 0;
        var actual = Convert.ToHexString(await System.Security.Cryptography.SHA256.HashDataAsync(stream, cancellationToken)).ToLowerInvariant();
        return System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(System.Text.Encoding.ASCII.GetBytes(actual), System.Text.Encoding.ASCII.GetBytes(expectedHash));
    }

    private async Task<GuardedImportArtifacts> OpenVerifiedImportsAsync(OllamaImportArtifacts import, CancellationToken cancellationToken)
    {
        EnsureNoReparsePoints(_importRoot!);
        var files = new GuardedImportArtifacts(_importRoot!, import.VerifierGgufPath, import.GgufPath, import.ModelfilePath);
        try
        {
            await files.VerifyAsync(import, cancellationToken);
            return files;
        }
        catch
        {
            files.Dispose();
            throw;
        }
    }

    private void EnsureWithinImportRoot(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var fullPath = Path.GetFullPath(path);
        var relative = Path.GetRelativePath(_importRoot!, fullPath);
        if (relative == ".." || relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) || Path.IsPathRooted(relative))
        {
            throw new InvalidOperationException("Ollama import artifacts must stay inside the launcher-owned staging root.");
        }
    }

    private static void EnsureNoReparsePoints(string path)
    {
        var root = Path.GetPathRoot(path) ?? throw new InvalidOperationException("Import root must be absolute.");
        var current = root;
        foreach (var component in path[root.Length..].Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, component);
            if (!Directory.Exists(current)) break;
            if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) throw new InvalidOperationException("Ollama import paths cannot traverse reparse points.");
        }
    }

    private sealed class GuardedImportArtifacts : IDisposable
    {
        private readonly NoFollowPathGuard _stagingRoot;
        private readonly NoFollowPathGuard _verifierPath;
        private readonly NoFollowPathGuard _ggufPath;
        private readonly NoFollowPathGuard _modelfilePath;
        private readonly GuardedFile _verifierGguf;
        private readonly GuardedFile _gguf;
        private readonly GuardedFile _modelfile;

        public GuardedImportArtifacts(string stagingRoot, string verifierGgufPath, string ggufPath, string modelfilePath)
        {
            var stagingGuard = NoFollowPathGuard.OpenDirectory(stagingRoot);
            NoFollowPathGuard? verifierPathGuard = null;
            NoFollowPathGuard? ggufPathGuard = null;
            NoFollowPathGuard? modelfilePathGuard = null;
            GuardedFile? verifierGguf = null;
            GuardedFile? gguf = null;
            GuardedFile? modelfile = null;
            try
            {
                verifierPathGuard = NoFollowPathGuard.OpenAncestors(verifierGgufPath);
                ggufPathGuard = NoFollowPathGuard.OpenAncestors(ggufPath);
                modelfilePathGuard = NoFollowPathGuard.OpenAncestors(modelfilePath);
                verifierGguf = new GuardedFile(verifierGgufPath);
                gguf = new GuardedFile(ggufPath);
                modelfile = new GuardedFile(modelfilePath);
            }
            catch
            {
                modelfile?.Dispose();
                gguf?.Dispose();
                verifierGguf?.Dispose();
                modelfilePathGuard?.Dispose();
                ggufPathGuard?.Dispose();
                verifierPathGuard?.Dispose();
                stagingGuard.Dispose();
                throw;
            }

            _stagingRoot = stagingGuard;
            _verifierPath = verifierPathGuard;
            _ggufPath = ggufPathGuard;
            _modelfilePath = modelfilePathGuard;
            _verifierGguf = verifierGguf;
            _gguf = gguf;
            _modelfile = modelfile;
        }

        public async Task VerifyAsync(OllamaImportArtifacts import, CancellationToken cancellationToken)
        {
            await _verifierGguf.VerifyAsync(import.VerifierGgufSize, import.VerifierGgufSha256, "verifier GGUF", cancellationToken);
            await _gguf.VerifyAsync(import.GgufSize, import.GgufSha256, "GGUF", cancellationToken);
            await _modelfile.VerifyAsync(import.ModelfileSize, import.ModelfileSha256, "Modelfile", cancellationToken);
        }

        public void DeleteTemporaryGgufs()
        {
            _verifierGguf.MarkForDeletion();
            _gguf.MarkForDeletion();
        }

        public void Dispose()
        {
            _modelfile.Dispose();
            _gguf.Dispose();
            _verifierGguf.Dispose();
            _modelfilePath.Dispose();
            _ggufPath.Dispose();
            _verifierPath.Dispose();
            _stagingRoot.Dispose();
        }
    }

    private sealed class GeneratedVerifierModelfile : IDisposable
    {
        private readonly NoFollowPathGuard _guard;

        private GeneratedVerifierModelfile(string path, NoFollowPathGuard guard)
        {
            Path = path;
            _guard = guard;
        }

        public string Path { get; }

        public static GeneratedVerifierModelfile Create(string verifierGgufPath)
        {
            var fileName = System.IO.Path.GetFileName(verifierGgufPath);
            if (fileName.Length == 0 || fileName.Any(character => !(char.IsAsciiLetterOrDigit(character) || character is '.' or '_' or '-')))
                throw new InvalidOperationException("The verifier GGUF staging name is invalid.");
            var directory = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(verifierGgufPath))
                ?? throw new InvalidOperationException("The verifier GGUF has no staging directory.");
            var path = System.IO.Path.Combine(directory, $".youtuber-verifier-{Guid.NewGuid():N}.Modelfile");
            using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                using var writer = new StreamWriter(stream, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false), 1024, leaveOpen: true);
                writer.Write($"FROM ./{fileName}\n");
                writer.Flush();
                stream.Flush(flushToDisk: true);
            }
            try
            {
                return new GeneratedVerifierModelfile(path, NoFollowPathGuard.Open(path));
            }
            catch
            {
                File.Delete(path);
                throw;
            }
        }

        public void Dispose()
        {
            _guard.Dispose();
            try { File.Delete(Path); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    private sealed class GuardedFile : IDisposable
    {
        private const uint GenericRead = 0x80000000;
        private const uint Delete = 0x00010000;
        private const uint ShareRead = 0x00000001;
        private const uint OpenExisting = 3;
        private const uint FileAttributeNormal = 0x00000080;
        private const uint FileFlagOpenReparsePoint = 0x00200000;
        private const int FileDispositionInfo = 4;
        private readonly SafeFileHandle _handle;

        public GuardedFile(string path)
        {
            EnsureNoReparsePoints(Path.GetDirectoryName(path) ?? throw new InvalidOperationException("Import path has no parent."));
            _handle = CreateFile(path, GenericRead | Delete, ShareRead, IntPtr.Zero, OpenExisting, FileAttributeNormal | FileFlagOpenReparsePoint, IntPtr.Zero);
            if (_handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to open import artifact safely.");
            if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) { _handle.Dispose(); throw new InvalidOperationException("Ollama import artifacts cannot be reparse points."); }
        }

        public async Task VerifyAsync(long expectedSize, string expectedHash, string name, CancellationToken cancellationToken)
        {
            var hasher = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
            var buffer = new byte[1024 * 1024];
            long offset = 0;
            while (offset < expectedSize)
            {
                var count = (int)Math.Min(buffer.Length, expectedSize - offset);
                var read = await RandomAccess.ReadAsync(_handle, buffer.AsMemory(0, count), offset, cancellationToken);
                if (read == 0) throw new InvalidOperationException($"The Speaker {name} failed verification.");
                hasher.AppendData(buffer, 0, read);
                offset += read;
            }
            if (RandomAccess.Read(_handle, buffer.AsSpan(0, 1), expectedSize) != 0) throw new InvalidOperationException($"The Speaker {name} failed verification.");
            var actual = Convert.ToHexString(hasher.GetHashAndReset()).ToLowerInvariant();
            if (!System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(System.Text.Encoding.ASCII.GetBytes(actual), System.Text.Encoding.ASCII.GetBytes(expectedHash))) throw new InvalidOperationException($"The Speaker {name} failed verification.");
        }

        public void MarkForDeletion()
        {
            var value = Marshal.AllocHGlobal(1);
            try
            {
                Marshal.WriteByte(value, 1);
                if (!SetFileInformationByHandle(_handle, FileDispositionInfo, value, 1)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to delete verified temporary GGUF by handle.");
            }
            finally { Marshal.FreeHGlobal(value); }
        }

        public void Dispose() => _handle.Dispose();

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateFile(string path, uint access, uint share, IntPtr securityAttributes, uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetFileInformationByHandle(SafeFileHandle handle, int fileInformationClass, IntPtr fileInformation, uint bufferSize);

    }
}
