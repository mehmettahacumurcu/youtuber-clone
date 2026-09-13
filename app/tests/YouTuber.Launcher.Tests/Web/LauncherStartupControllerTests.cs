using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Setup;
using YouTuber.Launcher.Web;
using Xunit;

namespace YouTuber.Launcher.Tests.Web;

public sealed class LauncherStartupControllerTests
{
    [Fact]
    public void Production_startup_requires_the_durable_post_health_ready_checkpoint()
    {
        var completed = System.Collections.Immutable.ImmutableHashSet.Create(SetupStage.EndToEndHealth);
        Assert.True(LauncherSessionPlanFactory.IsSetupReady(new SetupState(SetupStage.Ready, true, false, null, [], true) { CompletedStages = completed }));
        Assert.False(LauncherSessionPlanFactory.IsSetupReady(new SetupState(SetupStage.Ready, true, false, null, [], false) { CompletedStages = completed }));
        Assert.False(LauncherSessionPlanFactory.IsSetupReady(new SetupState(SetupStage.Ready, false, false, null, [], true) { CompletedStages = completed }));
        Assert.False(LauncherSessionPlanFactory.IsSetupReady(new SetupState(SetupStage.EndToEndHealth, true, false, null, [], true) { CompletedStages = completed }));
    }
    [Fact]
    public async Task Application_startup_attaches_recovery_and_waits_for_both_workers_before_bootstrap()
    {
        var events = new List<string>();
        await using var workers = new RecordingWorkerSession(events);
        var host = new RecordingShellHost(events);
        var voice = new WorkerDefinition("voice", "voice.exe", new Uri("http://127.0.0.1:31001/"));
        var studio = new WorkerDefinition("studio", "studio.exe", new Uri("http://127.0.0.1:31002/"));
        var plan = new LauncherSessionPlan("C:\\data", voice, studio, new Dictionary<string, string>(), SessionSecret.Create());
        var controller = new LauncherStartupController(workers);

        await controller.StartAsync(host, plan);

        Assert.Equal(["attach", "workers:start", "workers:ready", "shell:bootstrap"], events);
        Assert.Equal(31002, host.Origin!.Port);
        Assert.Equal(plan.SessionSecret, host.Secret);
    }

    [Fact]
    public async Task Ready_setup_attaches_the_retained_running_session_without_starting_workers_twice()
    {
        var events = new List<string>();
        await using var workers = new RecordingWorkerSession(events, alreadyRunning: true);
        var host = new RecordingShellHost(events);
        var plan = new LauncherSessionPlan(
            "C:\\data",
            new WorkerDefinition("voice", "voice.exe", new Uri("http://127.0.0.1:31001/")),
            new WorkerDefinition("studio", "studio.exe", new Uri("http://127.0.0.1:31002/")),
            new Dictionary<string, string>(),
            SessionSecret.Create(),
            new Uri("http://127.0.0.1:32003/"));

        await new LauncherStartupController(workers).AttachRunningAsync(host, plan);

        Assert.Equal(["attach", "shell:bootstrap"], events);
    }

    private sealed class RecordingShellHost(List<string> events) : IStudioShellHost
    {
        public StudioOrigin? Origin { get; private set; }
        public string? Secret { get; private set; }
        public void AttachWorkerSupervisor(IWorkerSupervisor supervisor) => events.Add("attach");
        public Task StartStudioAsync(StudioOrigin origin, string dataRoot, string sessionSecret, CancellationToken cancellationToken = default)
        {
            Origin = origin;
            Secret = sessionSecret;
            events.Add("shell:bootstrap");
            return Task.CompletedTask;
        }
        public void ShowStartupFailure(string message) => events.Add("failure");
    }

    private sealed class RecordingWorkerSession : IWorkerSupervisor
    {
        private readonly List<string> _events;
        public RecordingWorkerSession(List<string> events, bool alreadyRunning = false) { _events = events; State = alreadyRunning ? WorkerSupervisorState.Running : WorkerSupervisorState.Stopped; }
        public WorkerSupervisorState State { get; private set; }
        public event EventHandler? RecoveryRequired { add { } remove { } }
        public event EventHandler<WorkerRestartEventArgs>? WorkerRestarting { add { } remove { } }
        public event EventHandler<WorkerRestartEventArgs>? WorkerRestarted { add { } remove { } }
        public Task StartAsync(WorkerDefinition voice, WorkerDefinition studio, string sessionSecret, IReadOnlyDictionary<string, string>? settings = null, CancellationToken cancellationToken = default)
        {
            _events.Add("workers:start");
            State = WorkerSupervisorState.Running;
            _events.Add("workers:ready");
            return Task.CompletedTask;
        }
        public Task StopAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
