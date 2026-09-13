using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Net;
using System.Net.Sockets;
using Microsoft.Win32.SafeHandles;

namespace YouTuber.Launcher.Processes;

public interface IWorkerProcess : IJobProcess, IAsyncDisposable
{
    int Id { get; }
    bool HasExited { get; }
    Task WaitForExitAsync(CancellationToken cancellationToken);
    Task KillTreeAsync(CancellationToken cancellationToken);
    void Resume();
}

public interface IWorkerProcessLauncher
{
    IWorkerProcess Start(WorkerDefinition definition, IReadOnlyDictionary<string, string> environment, Action<string> stdout, Action<string> stderr);
}

public sealed class WorkerPortHandoffException(string message) : IOException(message);

public sealed class ProcessWorkerProcessLauncher : IWorkerProcessLauncher
{
    private readonly Action<WorkerDefinition>? _afterPortReservationReleased;
    private readonly Func<int, Process> _processLookup;

    internal ProcessWorkerProcessLauncher(
        Action<WorkerDefinition>? afterPortReservationReleased = null,
        Func<int, Process>? processLookup = null)
    {
        _afterPortReservationReleased = afterPortReservationReleased;
        _processLookup = processLookup ?? Process.GetProcessById;
    }

    public IWorkerProcess Start(WorkerDefinition definition, IReadOnlyDictionary<string, string> environment, Action<string> stdout, Action<string> stderr)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(environment);
        ArgumentNullException.ThrowIfNull(stdout);
        ArgumentNullException.ThrowIfNull(stderr);
        if (!environment.TryGetValue("YOUTUBER_SESSION_SECRET", out var secret) || string.IsNullOrWhiteSpace(secret))
        {
            throw new InvalidOperationException("Worker launch requires a session credential in its environment.");
        }
        if (definition.Arguments.Any(argument => argument.Contains(secret, StringComparison.Ordinal)))
        {
            throw new InvalidOperationException("Worker launch arguments may not include session credentials.");
        }

        var startInfo = new ProcessStartInfo(definition.ExecutablePath)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = definition.WorkingDirectory ?? Path.GetDirectoryName(definition.ExecutablePath) ?? Environment.CurrentDirectory,
        };
        foreach (var argument in definition.Arguments) startInfo.ArgumentList.Add(argument);
        foreach (var key in startInfo.Environment.Keys.Where(key => key.StartsWith("YOUTUBER_", StringComparison.Ordinal)).ToArray()) startInfo.Environment.Remove(key);
        foreach (var setting in environment) startInfo.Environment[setting.Key] = setting.Value;

        if (OperatingSystem.IsWindows()) return SuspendedWindowsWorkerProcess.Start(definition, startInfo, stdout, stderr, _afterPortReservationReleased, _processLookup);

        var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        try
        {
            foreach (var reservation in definition.AllPortReservations) reservation.Release(); // Keep every listener reservation through complete start-info construction.
            if (!process.Start()) throw new InvalidOperationException("The worker process could not be started.");
            process.OutputDataReceived += (_, eventArgs) => { if (eventArgs.Data is not null) stdout(eventArgs.Data); };
            process.ErrorDataReceived += (_, eventArgs) => { if (eventArgs.Data is not null) stderr(eventArgs.Data); };
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            return new ProcessWorkerProcess(process);
        }
        catch
        {
            process.Dispose();
            throw;
        }
    }

    private sealed class SuspendedWindowsWorkerProcess(Process process, SafeFileHandle thread, SafeFileHandle stdout, SafeFileHandle stderr, Action<string> writeStdout, Action<string> writeStderr) : IWorkerProcess
    {
        private readonly Process _process = process;
        private readonly SafeFileHandle _thread = thread;
        private readonly SafeFileHandle _stdout = stdout;
        private readonly SafeFileHandle _stderr = stderr;
        private bool _resumed;
        private Task? _stdoutPump;
        private Task? _stderrPump;
        public int Id => _process.Id;
        public bool HasExited => _process.HasExited;
        public IntPtr NativeHandle => _process.SafeHandle.DangerousGetHandle();
        public static IWorkerProcess Start(
            WorkerDefinition definition,
            ProcessStartInfo info,
            Action<string> stdout,
            Action<string> stderr,
            Action<WorkerDefinition>? afterPortReservationReleased,
            Func<int, Process> processLookup)
        {
            var security = new SECURITY_ATTRIBUTES { nLength = Marshal.SizeOf<SECURITY_ATTRIBUTES>(), bInheritHandle = true };
            if (!CreatePipe(out var outRead, out var outWrite, ref security, 0) || !CreatePipe(out var errRead, out var errWrite, ref security, 0)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
            try
            {
                if (!SetHandleInformation(outRead, 1, 0) || !SetHandleInformation(errRead, 1, 0)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Unable to restrict inherited pipe handles.");
                using var stdin = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0, IntPtr.Zero);
                if (stdin.IsInvalid) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Unable to open worker standard input.");
                var startup = new STARTUPINFOEX { StartupInfo = new STARTUPINFO { cb = Marshal.SizeOf<STARTUPINFOEX>(), dwFlags = 0x100 | 0x1, wShowWindow = 0, hStdOutput = outWrite.DangerousGetHandle(), hStdError = errWrite.DangerousGetHandle(), hStdInput = stdin.DangerousGetHandle() } };
                var command = WindowsCommandLine.Quote(info.FileName) + string.Concat(info.ArgumentList.Select(argument => " " + WindowsCommandLine.Quote(argument)));
                var environment = string.Join('\0', info.Environment.OrderBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase).Select(pair => pair.Key + "=" + pair.Value)) + "\0\0";
                var environmentMemory = Marshal.StringToHGlobalUni(environment);
                IntPtr attributeList = IntPtr.Zero;
                IntPtr handleList = IntPtr.Zero;
                var attributeListInitialized = false;
                try
                {
                    nuint attributeSize = 0;
                    _ = InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref attributeSize);
                    if (attributeSize == 0) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Unable to size worker handle whitelist.");
                    attributeList = Marshal.AllocHGlobal(checked((int)attributeSize));
                    if (!InitializeProcThreadAttributeList(attributeList, 1, 0, ref attributeSize)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Unable to create worker handle whitelist.");
                    attributeListInitialized = true;
                    startup.lpAttributeList = attributeList;
                    handleList = Marshal.AllocHGlobal(IntPtr.Size * 3);
                    Marshal.WriteIntPtr(handleList, 0, stdin.DangerousGetHandle());
                    Marshal.WriteIntPtr(handleList, IntPtr.Size, outWrite.DangerousGetHandle());
                    Marshal.WriteIntPtr(handleList, IntPtr.Size * 2, errWrite.DangerousGetHandle());
                    if (!UpdateProcThreadAttribute(attributeList, 0, (IntPtr)0x00020002, handleList, (nuint)(IntPtr.Size * 3), IntPtr.Zero, IntPtr.Zero)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Unable to restrict worker handle inheritance.");
                    foreach (var reservation in definition.AllPortReservations) reservation.Release();
                    afterPortReservationReleased?.Invoke(definition);
                    if (definition.AllPortReservations.Any(reservation => !CanBindLoopback(reservation.Port)))
                    {
                        throw new WorkerPortHandoffException("The reserved worker port was claimed during process handoff.");
                    }
                    var workingDirectory = string.IsNullOrWhiteSpace(info.WorkingDirectory) ? null : info.WorkingDirectory;
                    if (!CreateProcess(null, command, IntPtr.Zero, IntPtr.Zero, true, 0x00000004 | 0x08000000 | 0x00000400 | 0x00080000, environmentMemory, workingDirectory, ref startup, out var created)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "The worker process could not be created.");
                    using var createdProcess = new SafeFileHandle(created.hProcess, ownsHandle: true);
                    SafeFileHandle? createdThread = new(created.hThread, ownsHandle: true);
                    Process? process = null;
                    try
                    {
                        process = processLookup(created.dwProcessId);
                        outWrite.Dispose();
                        errWrite.Dispose();
                        var worker = new SuspendedWindowsWorkerProcess(process, createdThread, outRead, errRead, stdout, stderr);
                        createdThread = null;
                        process = null;
                        return worker;
                    }
                    catch
                    {
                        // The primary thread is still suspended. Terminate by the original native
                        // handle so a managed Process lookup failure cannot orphan the child.
                        _ = TerminateProcess(createdProcess, 1);
                        _ = WaitForSingleObject(createdProcess, 5000);
                        process?.Dispose();
                        throw;
                    }
                    finally
                    {
                        createdThread?.Dispose();
                    }
                }
                finally { if (attributeListInitialized) DeleteProcThreadAttributeList(attributeList); if (attributeList != IntPtr.Zero) Marshal.FreeHGlobal(attributeList); if (handleList != IntPtr.Zero) Marshal.FreeHGlobal(handleList); Marshal.FreeHGlobal(environmentMemory); }
            }
            catch { outRead.Dispose(); outWrite.Dispose(); errRead.Dispose(); errWrite.Dispose(); throw; }
        }
        public void Resume()
        {
            if (_resumed) return;
            _stdoutPump = PumpAsync(_stdout, writeStdout); _stderrPump = PumpAsync(_stderr, writeStderr);
            if (ResumeThread(_thread) == uint.MaxValue) { KillTreeAsync(CancellationToken.None).GetAwaiter().GetResult(); throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "The worker process could not be resumed."); }
            _resumed = true;
        }
        private static async Task PumpAsync(SafeFileHandle handle, Action<string> sink)
        {
            try
            {
                using var stream = new FileStream(handle, FileAccess.Read, 4096, isAsync: false);
                using var reader = new StreamReader(stream);
                while (await reader.ReadLineAsync() is { } line) sink(line);
            }
            catch (IOException) { }
            catch (ObjectDisposedException) { }
        }
        private static bool CanBindLoopback(int port)
        {
            var listener = new TcpListener(IPAddress.Loopback, port);
            listener.Server.ExclusiveAddressUse = true;
            try
            {
                listener.Start();
                return true;
            }
            catch (SocketException)
            {
                return false;
            }
            finally
            {
                listener.Stop();
            }
        }
        public async Task WaitForExitAsync(CancellationToken cancellationToken) { await _process.WaitForExitAsync(cancellationToken); await DrainAsync(); }
        public async Task KillTreeAsync(CancellationToken cancellationToken) { if (!_process.HasExited) _process.Kill(true); await _process.WaitForExitAsync(cancellationToken); await DrainAsync(); }
        private async Task DrainAsync()
        {
            var pumps = Task.WhenAll([_stdoutPump ?? Task.CompletedTask, _stderrPump ?? Task.CompletedTask]);
            try { await pumps.WaitAsync(TimeSpan.FromSeconds(1)); }
            catch (TimeoutException)
            {
                _stdout.Dispose();
                _stderr.Dispose();
            }
        }
        public ValueTask DisposeAsync() { _thread.Dispose(); _stdout.Dispose(); _stderr.Dispose(); _process.Dispose(); return ValueTask.CompletedTask; }
        [StructLayout(LayoutKind.Sequential)] private struct SECURITY_ATTRIBUTES { public int nLength; public IntPtr lpSecurityDescriptor; [MarshalAs(UnmanagedType.Bool)] public bool bInheritHandle; }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] private struct STARTUPINFO { public int cb; public string? lpReserved; public string? lpDesktop; public string? lpTitle; public int dwX; public int dwY; public int dwXSize; public int dwYSize; public int dwXCountChars; public int dwYCountChars; public int dwFillAttribute; public int dwFlags; public short wShowWindow; public short cbReserved2; public IntPtr lpReserved2; public IntPtr hStdInput; public IntPtr hStdOutput; public IntPtr hStdError; }
        [StructLayout(LayoutKind.Sequential)] private struct STARTUPINFOEX { public STARTUPINFO StartupInfo; public IntPtr lpAttributeList; }
        [StructLayout(LayoutKind.Sequential)] private struct PROCESS_INFORMATION { public IntPtr hProcess; public IntPtr hThread; public int dwProcessId; public int dwThreadId; }
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CreateProcess(string? application, string command, IntPtr processAttributes, IntPtr threadAttributes, [MarshalAs(UnmanagedType.Bool)] bool inheritHandles, uint flags, IntPtr environment, string? currentDirectory, ref STARTUPINFOEX startup, out PROCESS_INFORMATION process);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CreatePipe(out SafeFileHandle read, out SafeFileHandle write, ref SECURITY_ATTRIBUTES attributes, int size);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool SetHandleInformation(SafeFileHandle handle, uint mask, uint flags);
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] private static extern SafeFileHandle CreateFile(string name, uint access, uint share, ref SECURITY_ATTRIBUTES attributes, uint creation, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, int flags, ref nuint size);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, IntPtr attribute, IntPtr value, nuint size, IntPtr previous, IntPtr returnedSize);
        [DllImport("kernel32.dll")] private static extern void DeleteProcThreadAttributeList(IntPtr list);
        [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool TerminateProcess(SafeFileHandle process, uint exitCode);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern uint WaitForSingleObject(SafeFileHandle handle, uint milliseconds);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(SafeFileHandle thread);
    }

    private sealed class ProcessWorkerProcess(Process process) : IWorkerProcess
    {
        public int Id => process.Id;
        public bool HasExited => process.HasExited;
        public IntPtr NativeHandle => process.SafeHandle.DangerousGetHandle();
        public Task WaitForExitAsync(CancellationToken cancellationToken) => process.WaitForExitAsync(cancellationToken);
        public void Resume() { }
        public async Task KillTreeAsync(CancellationToken cancellationToken)
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync(cancellationToken);
        }
        public ValueTask DisposeAsync() { process.Dispose(); return ValueTask.CompletedTask; }
    }
}
