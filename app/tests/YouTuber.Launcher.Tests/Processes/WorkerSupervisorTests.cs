using YouTuber.Launcher.Ollama;
using YouTuber.Launcher.Processes;
using Xunit;

namespace YouTuber.Launcher.Tests.Processes;

public sealed class WorkerSupervisorTests
{
    [Fact]
    public async Task Start_starts_voice_then_dependencies_then_studio_with_secret_only_in_environment()
    {
        var events = new List<string>();
        var launcher = new RecordingLauncher(events);
        var health = new RecordingHealth(events);
        var dependencies = new RecordingDependencies(events);
        await using var supervisor = new WorkerSupervisor(launcher, health, new NoopWorkerJob(), dependencies, new NoActiveJobs());
        var secret = SessionSecret.Create();
        var voice = new WorkerDefinition("voice", "voice-worker.exe", new Uri("http://127.0.0.1:31001/"));
        var studio = new WorkerDefinition("studio", "studio-worker.exe", new Uri("http://127.0.0.1:31002/"));

        await supervisor.StartAsync(voice, studio, secret, new Dictionary<string, string> { ["YOUTUBER_DATA_ROOT"] = "C:\\data" });

        Assert.Equal(["start:voice", "live:voice", "ready:voice", "live:voice", "dependencies", "live:voice", "start:studio", "live:studio", "ready:studio"], events);
        Assert.DoesNotContain(secret, launcher.Arguments);
        Assert.All(launcher.Environments, environment => Assert.Equal(secret, environment["YOUTUBER_SESSION_SECRET"]));
    }

    [Fact]
    public async Task Crashed_idle_worker_is_restarted_only_once()
    {
        var events = new List<string>();
        var launcher = new RecordingLauncher(events);
        await using var supervisor = new WorkerSupervisor(launcher, new RecordingHealth(events), new NoopWorkerJob(), new RecordingDependencies(events), new NoActiveJobs());
        var recovery = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        supervisor.RecoveryRequired += (_, _) => recovery.TrySetResult();
        var definition = new WorkerDefinition("voice", "voice-worker.exe", new Uri("http://127.0.0.1:31001/"));

        await supervisor.StartWorkerAsync(definition, SessionSecret.Create(), new Dictionary<string, string>());
        var first = launcher.Processes.Single();
        first.Crash();
        await first.Exited.Task;
        var second = await launcher.WaitForProcessAsync(2);
        second.Crash();
        await second.Exited.Task;
        await recovery.Task.WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Equal(2, launcher.Processes.Count);
        Assert.Equal(WorkerSupervisorState.RecoveryRequired, supervisor.State);
    }

    [Fact]
    public async Task Successful_auto_restart_notifies_consumers_before_and_after_worker_readiness()
    {
        var events = new List<string>();
        var launcher = new RecordingLauncher(events);
        await using var supervisor = new WorkerSupervisor(launcher, new RecordingHealth(events), new NoopWorkerJob(), new RecordingDependencies(events), new NoActiveJobs());
        var restarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        supervisor.WorkerRestarting += (_, args) => events.Add("restarting:" + args.WorkerName);
        supervisor.WorkerRestarted += (_, args) => { events.Add("restarted:" + args.WorkerName); restarted.TrySetResult(); };
        var definition = new WorkerDefinition("studio", "studio-worker.exe", new Uri("http://127.0.0.1:31002/"));

        await supervisor.StartWorkerAsync(definition, SessionSecret.Create());
        launcher.Processes.Single().Crash();
        await restarted.Task.WaitAsync(TimeSpan.FromSeconds(1));

        var restartingIndex = events.IndexOf("restarting:studio");
        var restartedIndex = events.IndexOf("restarted:studio");
        Assert.True(restartingIndex >= 0);
        Assert.True(restartedIndex > restartingIndex);
        Assert.Contains("ready:studio", events.Skip(restartingIndex + 1).Take(restartedIndex - restartingIndex - 1));
    }

    [Fact]
    public async Task Crash_cannot_restart_after_a_job_lease_is_admitted()
    {
        var events = new List<string>();
        var launcher = new RecordingLauncher(events);
        var activity = new WorkerActivityCoordinator();
        await using var supervisor = new WorkerSupervisor(launcher, new RecordingHealth(events), new NoopWorkerJob(), new RecordingDependencies(events), activity);
        var recovery = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        supervisor.RecoveryRequired += (_, _) => recovery.TrySetResult();
        await supervisor.StartWorkerAsync(new WorkerDefinition("voice", "voice-worker.exe", new Uri("http://127.0.0.1:31001/")), SessionSecret.Create());

        using var job = activity.AcquireJobLease();
        launcher.Processes.Single().Crash();
        await recovery.Task.WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Single(launcher.Processes);
        Assert.Equal(WorkerSupervisorState.RecoveryRequired, supervisor.State);
    }

    [Fact]
    public async Task Duplicate_worker_start_is_rejected_without_losing_the_owned_process()
    {
        var events = new List<string>();
        var launcher = new RecordingLauncher(events);
        await using var supervisor = new WorkerSupervisor(launcher, new RecordingHealth(events), new NoopWorkerJob(), new RecordingDependencies(events), new NoActiveJobs());
        var definition = new WorkerDefinition("voice", "voice-worker.exe", new Uri("http://127.0.0.1:31001/"));
        var secret = SessionSecret.Create();

        await supervisor.StartWorkerAsync(definition, secret);

        await Assert.ThrowsAsync<InvalidOperationException>(() => supervisor.StartWorkerAsync(definition, secret));
        Assert.Single(launcher.Processes);
    }

    [Fact]
    public async Task Runtime_start_failure_stops_launcher_owned_ollama_before_rethrowing()
    {
        var controller = new RecordingOllamaController();
        var manager = new OllamaManager(serverController: controller);
        await using var supervisor = new WorkerSupervisor(new ThrowingLauncher());
        var orchestrator = new WorkerRuntimeOrchestrator(manager, supervisor);
        using var factory = new WorkerRuntimeFactory();
        var runtime = factory.Create(
            Path.Combine(Path.GetTempPath(), $"youtuber-runtime-{Guid.NewGuid():N}"),
            new WorkerDefinition("voice", "voice-worker.exe", new Uri("http://127.0.0.1:31001/")),
            new WorkerDefinition("studio", "studio-worker.exe", new Uri("http://127.0.0.1:31002/")));
        var server = new VerifiedOllamaServer("ollama.exe", new Version(1, 0), "test-release");

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => orchestrator.StartAsync(server, runtime, SessionSecret.Create()));

        Assert.Equal(1, controller.StartCalls);
        Assert.Equal(1, controller.StopCalls);
    }

    private sealed class RecordingLauncher(List<string> events) : IWorkerProcessLauncher
    {
        public List<FakeProcess> Processes { get; } = [];
        private readonly TaskCompletionSource<FakeProcess> _secondProcess = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public List<IReadOnlyDictionary<string, string>> Environments { get; } = [];
        public string Arguments { get; private set; } = string.Empty;
        public IWorkerProcess Start(WorkerDefinition definition, IReadOnlyDictionary<string, string> environment, Action<string> stdout, Action<string> stderr)
        {
            events.Add($"start:{definition.Name}");
            Environments.Add(environment);
            Arguments = string.Join(' ', definition.Arguments);
            var process = new FakeProcess();
            Processes.Add(process);
            if (Processes.Count == 2) _secondProcess.TrySetResult(process);
            return process;
        }
        public Task<FakeProcess> WaitForProcessAsync(int count) => count == 2 ? _secondProcess.Task : Task.FromResult(Processes[count - 1]);
    }

    private sealed class ThrowingLauncher : IWorkerProcessLauncher
    {
        public IWorkerProcess Start(WorkerDefinition definition, IReadOnlyDictionary<string, string> environment, Action<string> stdout, Action<string> stderr)
            => throw new InvalidOperationException("worker startup failed");
    }

    private sealed class RecordingOllamaController : IOllamaServerController
    {
        public int StartCalls { get; private set; }
        public int StopCalls { get; private set; }
        public Task SnapshotBeforeInstallAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<OllamaProcessReceipt?> CaptureInstallerAutoStartAsync(VerifiedOllamaServer server, DateTime installerStartedUtc, DateTime installerFinishedUtc, CancellationToken cancellationToken = default)
            => Task.FromResult<OllamaProcessReceipt?>(null);
        public Task StopAutoStartedAsync(OllamaProcessReceipt receipt, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, CancellationToken cancellationToken = default)
            => throw new NotSupportedException();
        public Task StartServeAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default)
        {
            StartCalls++;
            return Task.CompletedTask;
        }
        public Task WaitForHealthyAndConfirmStoreAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default)
            => Task.CompletedTask;
        public Task StopOwnedAsync(CancellationToken cancellationToken = default)
        {
            StopCalls++;
            return Task.CompletedTask;
        }
    }

    private sealed class RecordingHealth(List<string> events) : IWorkerHealthClient
    {
        public Task WaitForLiveAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken) { events.Add($"live:{definition.Name}"); return Task.CompletedTask; }
        public Task WaitForReadyAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken) { events.Add($"ready:{definition.Name}"); return Task.CompletedTask; }
    }

    private sealed class RecordingDependencies(List<string> events) : IWorkerDependencyHealth
    {
        public Task WaitForReadyAsync(CancellationToken cancellationToken) { events.Add("dependencies"); return Task.CompletedTask; }
    }

    private sealed class NoopWorkerJob : IWorkerJob
    {
        public void Assign(IJobProcess process) { }
        public void Dispose() { }
    }

    [Fact]
    public void WindowsJobRejectsAZeroProcessHandleInsteadOfAllowingResumeWithoutContainment()
    {
        using var job = new WindowsJob();

        Assert.Throws<InvalidOperationException>(() => job.Assign(new ZeroHandleJobProcess()));
    }

    private sealed class ZeroHandleJobProcess : IJobProcess
    {
        public IntPtr NativeHandle => IntPtr.Zero;
    }

    private sealed class NoActiveJobs : IWorkerActivityCoordinator
    {
        public bool HasActiveChatOrVoiceJob => false;
        public IDisposable AcquireJobLease() => new Lease();
        public bool TryAcquireRestartLease(out IDisposable? lease) { lease = new Lease(); return true; }
        private sealed class Lease : IDisposable { public void Dispose() { } }
    }

    private sealed class FakeProcess : IWorkerProcess
    {
        public TaskCompletionSource Exited { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public int Id => 1;
        public bool HasExited { get; private set; }
        public IntPtr NativeHandle => IntPtr.Zero;
        public Task WaitForExitAsync(CancellationToken cancellationToken) => Exited.Task.WaitAsync(cancellationToken);
        public void Resume() { }
        public Task KillTreeAsync(CancellationToken cancellationToken) { Crash(); return Task.CompletedTask; }
        public void Crash() { HasExited = true; Exited.TrySetResult(); }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
