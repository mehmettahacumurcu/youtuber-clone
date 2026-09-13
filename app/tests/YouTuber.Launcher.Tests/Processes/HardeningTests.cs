using System.Net;
using System.Net.Http;
using System.Diagnostics;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using YouTuber.Launcher.Processes;
using Xunit;

namespace YouTuber.Launcher.Tests.Processes;

public sealed class HardeningTests
{
    [Theory]
    [InlineData("C:\\Türkçe Yol\\worker.exe", "\"C:\\Türkçe Yol\\worker.exe\"")]
    [InlineData("ends with\\", "\"ends with\\\\\"")]
    [InlineData("a\"b", "\"a\\\"b\"")]
    public void Windows_argument_quoting_round_trips_special_paths(string argument, string expected)
        => Assert.Equal(expected, WindowsCommandLine.Quote(argument));

    [Fact]
    public void LauncherPorts_assigns_four_distinct_reserved_loopback_ports_including_RAG()
    {
        using var ports = LauncherPorts.Create();

        Assert.Equal(4, new[] { ports.OllamaPort, ports.RagPort, ports.VoicePort, ports.StudioPort }.Distinct().Count());
        Assert.Equal(4, ports.Reservations.Count);
        Assert.All(ports.Reservations, reservation => Assert.False(reservation.IsReleased));
    }

    [Fact]
    public void Production_runtime_factory_builds_the_complete_packaged_worker_contract()
    {
        using var root = new TemporaryDirectory();
        var voiceRoot = System.IO.Path.Combine(root.Path, "runtime", "voice");
        var studioRoot = System.IO.Path.Combine(root.Path, "runtime", "studio");
        var voice = new WorkerDefinition("voice", "voice.exe", new Uri("http://127.0.0.1:1/")) { WorkingDirectory = voiceRoot };
        var studio = new WorkerDefinition("studio", "studio.exe", new Uri("http://127.0.0.1:1/")) { WorkingDirectory = studioRoot };
        using var factory = new WorkerRuntimeFactory();

        var runtime = factory.Create(root.Path, voice, studio);

        Assert.Equal(runtime.OllamaReservation.Port, runtime.Ollama.ApiEndpoint.Port);
        Assert.Equal(runtime.Voice.PortReservation!.Port, runtime.Voice.HealthEndpoint.Port);
        Assert.Equal(runtime.Studio.PortReservation!.Port, runtime.Studio.HealthEndpoint.Port);
        Assert.Equal(runtime.RagReservation.Port.ToString(System.Globalization.CultureInfo.InvariantCulture), runtime.Settings["YOUTUBER_RAG_PORT"]);
        Assert.Equal(4, new[]
        {
            runtime.Ollama.ApiEndpoint.Port,
            runtime.RagReservation.Port,
            runtime.Voice.HealthEndpoint.Port,
            runtime.Studio.HealthEndpoint.Port,
        }.Distinct().Count());
        Assert.Equal(root.Path, runtime.Settings["YOUTUBER_DATA_ROOT"]);
        Assert.Equal(root.Path, runtime.Settings["YOUTUBER_CACHE_ROOT"]);
        Assert.Equal(runtime.Ollama.ApiEndpoint.GetLeftPart(UriPartial.Authority), runtime.Settings["YOUTUBER_OLLAMA_ORIGIN"]);
        Assert.Equal(voiceRoot, runtime.Voice.Environment["YOUTUBER_INSTALL_ROOT"]);
        Assert.Equal(studioRoot, runtime.Studio.Environment["YOUTUBER_INSTALL_ROOT"]);
        Assert.Contains(runtime.RagReservation, runtime.Studio.AdditionalPortReservations);
        Assert.All(new[] { runtime.OllamaReservation, runtime.RagReservation, runtime.Voice.PortReservation, runtime.Studio.PortReservation }, reservation => Assert.False(reservation!.IsReleased));
    }

    [Fact]
    public async Task Hanging_health_response_honors_the_startup_deadline()
    {
        using var health = new HttpWorkerHealthClient(new NeverRespondsHandler());
        var worker = new WorkerDefinition("voice", "voice-worker.exe", new Uri("http://127.0.0.1:31001/"));

        await Assert.ThrowsAsync<TimeoutException>(() => health.WaitForLiveAsync(worker, SessionSecret.Create(), TimeSpan.FromMilliseconds(50), CancellationToken.None));
    }

    [Fact]
    public async Task Fixture_streams_are_drained_and_session_credential_is_redacted_from_logs()
    {
        using var directory = new TemporaryDirectory();
        using var ports = LauncherPorts.Create();
        var template = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--port=" + ports.VoicePort],
            LogDirectory = directory.Path,
        };
        var composition = ports.Compose(template, template);
        var secret = SessionSecret.Create();
        await using var supervisor = new WorkerSupervisor();

        await supervisor.StartWorkerAsync(composition.Voice, secret, composition.Settings);
        for (var attempts = 0; attempts < 20 && !File.Exists(System.IO.Path.Combine(directory.Path, "voice.stdout.log")); attempts++) await Task.Delay(50);
        await supervisor.StopAsync();

        var logFiles = Directory.GetFiles(directory.Path, "voice.*.log");
        if (logFiles.Length > 0)
        {
            var logs = string.Concat(logFiles.Select(File.ReadAllText));
            Assert.DoesNotContain(secret, logs);
            Assert.Contains("[redacted]", logs);
        }
    }

    [Fact]
    public async Task Real_hanging_fixture_times_out_within_the_startup_bound_and_leaves_no_process()
    {
        using var directory = new TemporaryDirectory();
        using var ports = LauncherPorts.Create();
        var pidFile = System.IO.Path.Combine(directory.Path, "hang-pids.txt");
        var template = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--port-env=YOUTUBER_VOICE_PORT", "--hang-health"],
            StartupTimeout = TimeSpan.FromSeconds(1),
        };
        var composition = ports.Compose(template, template);
        var settings = new Dictionary<string, string>(composition.Settings) { ["YOUTUBER_FIXTURE_PID_FILE"] = pidFile };
        await using var supervisor = new WorkerSupervisor();
        var stopwatch = Stopwatch.StartNew();

        await Assert.ThrowsAsync<TimeoutException>(() => supervisor.StartWorkerAsync(composition.Voice, SessionSecret.Create(), settings));

        Assert.InRange(stopwatch.Elapsed, TimeSpan.Zero, TimeSpan.FromSeconds(3));
        var pids = await ReadPidsAsync(pidFile, expectedCount: 1);
        Assert.All(pids, AssertEventuallyExited);
    }

    [Fact]
    public async Task Real_crashing_fixture_restarts_once_then_requires_recovery_when_idle()
    {
        using var directory = new TemporaryDirectory();
        using var ports = LauncherPorts.Create();
        var startFile = System.IO.Path.Combine(directory.Path, "crash-pids.txt");
        var template = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--port-env=YOUTUBER_VOICE_PORT", "--crash"],
            StartupTimeout = TimeSpan.FromSeconds(2),
        };
        var composition = ports.Compose(template, template);
        var settings = new Dictionary<string, string>(composition.Settings) { ["YOUTUBER_FIXTURE_START_FILE"] = startFile };
        await using var supervisor = new WorkerSupervisor();
        var recovery = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        supervisor.RecoveryRequired += (_, _) => recovery.TrySetResult();

        await supervisor.StartWorkerAsync(composition.Voice, SessionSecret.Create(), settings);
        await recovery.Task.WaitAsync(TimeSpan.FromSeconds(5));

        var pids = await ReadPidsAsync(startFile, expectedCount: 2);
        await Task.Delay(250);
        Assert.Equal(2, File.ReadAllLines(startFile).Where(line => !string.IsNullOrWhiteSpace(line)).Distinct().Count());
        Assert.Equal(WorkerSupervisorState.RecoveryRequired, supervisor.State);
        Assert.All(pids, AssertEventuallyExited);
    }

    [Fact]
    public async Task Stolen_port_handoff_re_reserves_once_and_uses_the_new_fixture_port()
    {
        using var directory = new TemporaryDirectory();
        using var ports = LauncherPorts.Create();
        var portFile = System.IO.Path.Combine(directory.Path, "fixture-port.txt");
        TcpListener? squatter = null;
        var hookCalls = 0;
        var launcher = new ProcessWorkerProcessLauncher(definition =>
        {
            if (hookCalls++ != 0) return;
            squatter = new TcpListener(IPAddress.Loopback, definition.HealthEndpoint.Port);
            squatter.Start();
        });
        var template = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--port-env=YOUTUBER_VOICE_PORT"],
            StartupTimeout = TimeSpan.FromSeconds(1),
        };
        var composition = ports.Compose(template, template);
        var settings = new Dictionary<string, string>(composition.Settings) { ["YOUTUBER_FIXTURE_PORT_FILE"] = portFile };
        await using var supervisor = new WorkerSupervisor(launcher);
        try
        {
            await supervisor.StartWorkerAsync(composition.Voice, SessionSecret.Create(), settings);

            await WaitUntilAsync(() => File.Exists(portFile), TimeSpan.FromSeconds(2));
            var actualPort = int.Parse(File.ReadAllText(portFile), System.Globalization.CultureInfo.InvariantCulture);
            Assert.Equal(2, hookCalls);
            Assert.NotEqual(ports.VoicePort, actualPort);
        }
        finally
        {
            squatter?.Stop();
        }
    }

    [Fact]
    public async Task Real_burst_output_is_drained_and_redacted_before_dispose()
    {
        using var directory = new TemporaryDirectory();
        using var ports = LauncherPorts.Create();
        var template = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--port-env=YOUTUBER_VOICE_PORT", "--burst"],
            LogDirectory = directory.Path,
            StartupTimeout = TimeSpan.FromSeconds(2),
        };
        var composition = ports.Compose(template, template);
        var secret = SessionSecret.Create();
        await using var supervisor = new WorkerSupervisor();
        var recovery = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        supervisor.RecoveryRequired += (_, _) => recovery.TrySetResult();

        await supervisor.StartWorkerAsync(composition.Voice, secret, composition.Settings);
        await recovery.Task.WaitAsync(TimeSpan.FromSeconds(5));
        await supervisor.DisposeAsync();

        var stdout = File.ReadAllText(System.IO.Path.Combine(directory.Path, "voice.stdout.log"));
        var stderr = File.ReadAllText(System.IO.Path.Combine(directory.Path, "voice.stderr.log"));
        Assert.Contains("fixture final stdout marker [redacted]", stdout);
        Assert.Contains("fixture final stderr marker [redacted]", stderr);
        Assert.DoesNotContain(secret, stdout);
        Assert.DoesNotContain(secret, stderr);
    }

    [Fact]
    public async Task Inheritable_parent_sentinel_handle_is_excluded_from_the_native_worker_handle_list()
    {
        using var directory = new TemporaryDirectory();
        using SafeFileHandle sentinel = File.OpenHandle(System.IO.Path.Combine(directory.Path, "sentinel.txt"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite, FileOptions.None);
        Assert.True(SetHandleInformation(sentinel, HandleFlagInherit, HandleFlagInherit));
        var output = new ConcurrentQueue<string>();
        var secret = SessionSecret.Create();
        var definition = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--probe-handle=" + sentinel.DangerousGetHandle().ToInt64().ToString(System.Globalization.CultureInfo.InvariantCulture), "--exit-after-probe"],
        };
        var process = new ProcessWorkerProcessLauncher().Start(definition, new Dictionary<string, string> { ["YOUTUBER_SESSION_SECRET"] = secret }, output.Enqueue, _ => { });
        process.Resume();
        await process.WaitForExitAsync(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(3));
        await process.DisposeAsync();

        Assert.Contains("fixture probe handle inherited=false", output);
    }

    [Fact]
    public void Managed_process_lookup_failure_terminates_the_still_suspended_native_child()
    {
        using var directory = new TemporaryDirectory();
        var startFile = System.IO.Path.Combine(directory.Path, "must-not-start.pid");
        var definition = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--port=1"],
        };
        var launcher = new ProcessWorkerProcessLauncher(
            processLookup: _ => throw new InvalidOperationException("injected lookup failure"));

        Assert.Throws<InvalidOperationException>(() => launcher.Start(
            definition,
            new Dictionary<string, string>
            {
                ["YOUTUBER_SESSION_SECRET"] = SessionSecret.Create(),
                ["YOUTUBER_FIXTURE_START_FILE"] = startFile,
            },
            _ => { },
            _ => { }));

        Thread.Sleep(100);
        Assert.False(File.Exists(startFile));
    }

    [Fact]
    public async Task Primary_exit_with_a_descendant_holding_pipe_writers_drains_within_a_bound()
    {
        using var directory = new TemporaryDirectory();
        var childFile = System.IO.Path.Combine(directory.Path, "held-pipe-child.pid");
        var output = new ConcurrentQueue<string>();
        var definition = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [FixturePath(), "--port=1", "--spawn-child", "--exit-leaving-child"],
        };
        var process = new ProcessWorkerProcessLauncher().Start(definition, new Dictionary<string, string>
        {
            ["YOUTUBER_SESSION_SECRET"] = SessionSecret.Create(),
            ["YOUTUBER_CHILD_PID_FILE"] = childFile,
        }, output.Enqueue, _ => { });
        process.Resume();
        try
        {
            await process.WaitForExitAsync(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(2));
            Assert.Contains(output, line => line.StartsWith("fixture stdout", StringComparison.Ordinal));
        }
        finally
        {
            await process.DisposeAsync();
            if (File.Exists(childFile))
            {
                try { Process.GetProcessById(int.Parse(File.ReadAllText(childFile), System.Globalization.CultureInfo.InvariantCulture)).Kill(true); }
                catch (ArgumentException) { }
            }
        }
    }

    [Fact]
    public async Task Job_close_terminates_fixture_and_its_immediate_child()
    {
        using var directory = new TemporaryDirectory();
        using var ports = LauncherPorts.Create();
        var childFile = System.IO.Path.Combine(directory.Path, "child.pid");
        var template = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/")) { Arguments = [FixturePath(), "--port=" + ports.VoicePort, "--spawn-child"] };
        var composition = ports.Compose(template, template);
        await using var supervisor = new WorkerSupervisor();
        var settings = new Dictionary<string, string>(composition.Settings) { ["YOUTUBER_CHILD_PID_FILE"] = childFile };
        await supervisor.StartWorkerAsync(composition.Voice, SessionSecret.Create(), settings);
        for (var attempts = 0; attempts < 20 && !File.Exists(childFile); attempts++) await Task.Delay(50);
        var childId = int.Parse(File.ReadAllText(childFile), System.Globalization.CultureInfo.InvariantCulture);

        await supervisor.StopAsync();
        await Task.Delay(100);

        try { Assert.True(Process.GetProcessById(childId).HasExited); }
        catch (ArgumentException) { }
    }

    private static string FixturePath() => System.IO.Path.GetFullPath(System.IO.Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "YouTuber.WorkerFixture", "bin", "Release", "net8.0-windows", "YouTuber.WorkerFixture.dll"));

    private static async Task<IReadOnlyList<int>> ReadPidsAsync(string path, int expectedCount)
    {
        await WaitUntilAsync(() => File.Exists(path) && File.ReadAllLines(path).Count(line => !string.IsNullOrWhiteSpace(line)) >= expectedCount, TimeSpan.FromSeconds(3));
        return File.ReadAllLines(path).Where(line => !string.IsNullOrWhiteSpace(line)).Select(line => int.Parse(line, System.Globalization.CultureInfo.InvariantCulture)).Distinct().ToArray();
    }

    private static void AssertEventuallyExited(int pid)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(3);
        while (DateTime.UtcNow < deadline)
        {
            try
            {
                if (Process.GetProcessById(pid).HasExited) return;
            }
            catch (ArgumentException) { return; }
            Thread.Sleep(25);
        }
        Assert.Fail($"Fixture process {pid} is still running.");
    }

    private static async Task WaitUntilAsync(Func<bool> condition, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (condition()) return;
            await Task.Delay(25);
        }
        Assert.True(condition(), "Condition did not complete before its deadline.");
    }

    private const uint HandleFlagInherit = 1;
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetHandleInformation(SafeFileHandle handle, uint mask, uint flags);

    private sealed class NeverRespondsHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var completion = new TaskCompletionSource<HttpResponseMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
            cancellationToken.Register(() => completion.TrySetCanceled(cancellationToken));
            return completion.Task;
        }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory() { Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "YouTuberFixture-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(Path); }
        public string Path { get; }
        public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
    }
}
