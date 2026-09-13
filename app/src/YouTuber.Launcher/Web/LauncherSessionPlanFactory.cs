using System.IO;
using System.Net.Http;
using YouTuber.Launcher.Activation;
using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Setup;

namespace YouTuber.Launcher.Web;

public static class LauncherSessionPlanFactory
{
    public static async Task<LauncherSessionPlan> CreateAsync(IReadOnlyList<string> arguments, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        if (arguments.Contains("--fixture-workers", StringComparer.Ordinal))
        {
#if FIXTURE_BUILD
            return await CreateFixturePlan(arguments, cancellationToken);
#else
            throw new InvalidOperationException("Fixture workers are not available in release builds.");
#endif
        }

        var dataRootArgument = arguments.SingleOrDefault(argument => argument.StartsWith("--data-root=", StringComparison.Ordinal));
        var dataRoot = dataRootArgument is null ? null : dataRootArgument["--data-root=".Length..];
        var paths = AppPaths.ForCurrentUser(dataRoot);
        var setup = new AtomicFileSetupJournal(paths.SetupJournalFile).Load();
        if (!IsSetupReady(setup)) throw new InvalidOperationException("Setup has not recorded the durable Ready checkpoint.");
        var store = new ActiveComponentsStore(paths.DataRoot);
        var active = await store.LoadAsync(cancellationToken);
        var voice = await ResolveInstalledWorkerAsync(paths, active, "voice_runtime", "voice", cancellationToken);
        var studio = await ResolveInstalledWorkerAsync(paths, active, "studio_runtime", "studio", cancellationToken);
        var runtimeFactory = new WorkerRuntimeFactory();
        try
        {
            var runtime = runtimeFactory.Create(paths.DataRoot, voice, studio);
            var settings = new Dictionary<string, string>(runtime.Settings, StringComparer.Ordinal)
            {
                ["YOUTUBER_DATA_ROOT"] = paths.DataRoot,
                ["YOUTUBER_LOGS_ROOT"] = paths.LogsRoot,
            };
            return new LauncherSessionPlan(paths.DataRoot, runtime.Voice, runtime.Studio, settings, SessionSecret.Create(), runtime.Ollama.ApiEndpoint, runtimeFactory);
        }
        catch
        {
            runtimeFactory.Dispose();
            throw;
        }
    }

    public static bool IsSetupReady(SetupState? state)
        => state is { Stage: SetupStage.Ready, TermsAccepted: true, ReadyRecorded: true }
           && state.CompletedStages.Contains(SetupStage.EndToEndHealth);

#if FIXTURE_BUILD
    private static async Task<LauncherSessionPlan> CreateFixturePlan(IReadOnlyList<string> arguments, CancellationToken cancellationToken)
    {
        var dataRoot = arguments.SingleOrDefault(argument => argument.StartsWith("--data-root=", StringComparison.Ordinal))?["--data-root=".Length..] ?? throw new InvalidOperationException("Fixture build requires --data-root.");
        var manifestUrl = arguments.SingleOrDefault(argument => argument.StartsWith("--fixture-manifest-url=", StringComparison.Ordinal))?["--fixture-manifest-url=".Length..] ?? throw new InvalidOperationException("Fixture build requires --fixture-manifest-url.");
        var manifestHash = arguments.SingleOrDefault(argument => argument.StartsWith("--fixture-manifest-sha256=", StringComparison.Ordinal))?["--fixture-manifest-sha256=".Length..] ?? throw new InvalidOperationException("Fixture build requires --fixture-manifest-sha256.");
        if (!Uri.TryCreate(manifestUrl, UriKind.Absolute, out var parsed) || parsed.Scheme != Uri.UriSchemeHttp || !parsed.IsLoopback || !System.Text.RegularExpressions.Regex.IsMatch(manifestHash, "^[a-f0-9]{64}$")) throw new InvalidOperationException("Fixture manifest identity is invalid.");
        using (var client = new HttpClient(new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false }) { Timeout = TimeSpan.FromSeconds(5) })
        using (var response = await client.GetAsync(parsed, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
        {
            if (!response.IsSuccessStatusCode || response.RequestMessage?.RequestUri != parsed || response.Content.Headers.ContentLength is > 1024 * 1024) throw new InvalidOperationException("Fixture manifest request was rejected.");
            await response.Content.LoadIntoBufferAsync(1024 * 1024).WaitAsync(cancellationToken);
            var bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
            if (bytes.Length > 1024 * 1024 || !System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(System.Security.Cryptography.SHA256.HashData(bytes), Convert.FromHexString(manifestHash))) throw new InvalidOperationException("Fixture manifest hash mismatched.");
        }
        var fixture = FindFixtureAssembly();
        var logs = Path.Combine(dataRoot, "logs");
        Directory.CreateDirectory(logs);
        var readyMarker = Path.Combine(dataRoot, "state", "studio-ready.marker");
        var voicePid = Path.Combine(dataRoot, "state", "voice.pid"); var studioPid = Path.Combine(dataRoot, "state", "studio.pid");
        var voicePort = Path.Combine(dataRoot, "state", "voice.port"); var studioPort = Path.Combine(dataRoot, "state", "studio.port");
        Directory.CreateDirectory(Path.GetDirectoryName(readyMarker)!);
        var traceFile = Path.Combine(dataRoot, "state", "studio-trace.log");
        if (File.Exists(readyMarker)) File.Delete(readyMarker);
        if (File.Exists(traceFile)) File.Delete(traceFile);
        var voice = new WorkerDefinition("voice", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [fixture, "--port-env=YOUTUBER_VOICE_PORT"],
            WorkingDirectory = Path.GetDirectoryName(fixture),
            LogDirectory = logs,
            Environment = new Dictionary<string, string> { ["YOUTUBER_FIXTURE_PID_FILE"] = voicePid, ["YOUTUBER_FIXTURE_PORT_FILE"] = voicePort },
            StartupTimeout = TimeSpan.FromSeconds(10),
        };
        var studio = new WorkerDefinition("studio", "dotnet", new Uri("http://127.0.0.1:1/"))
        {
            Arguments = [fixture, "--port-env=YOUTUBER_STUDIO_PORT", "--studio"],
            WorkingDirectory = Path.GetDirectoryName(fixture),
            LogDirectory = logs,
            Environment = new Dictionary<string, string> { ["YOUTUBER_FIXTURE_PID_FILE"] = studioPid, ["YOUTUBER_FIXTURE_PORT_FILE"] = studioPort },
            StartupTimeout = TimeSpan.FromSeconds(10),
        };
        return Compose(dataRoot, logs, voice, studio, null, new Dictionary<string, string>
        {
            ["YOUTUBER_FIXTURE_STUDIO_READY_FILE"] = readyMarker,
            ["YOUTUBER_FIXTURE_STUDIO_TRACE_FILE"] = traceFile,
        });
    }

    private static string FindFixtureAssembly()
    {
        var direct = Path.Combine(AppContext.BaseDirectory, "YouTuber.WorkerFixture.dll");
        if (File.Exists(direct)) return direct;
        throw new FileNotFoundException("Fixture worker payload is missing beside the launcher.", direct);
    }
#endif

    private static async Task<WorkerDefinition> ResolveInstalledWorkerAsync(
        AppPaths paths,
        ActiveComponents active,
        string component,
        string workerName,
        CancellationToken cancellationToken)
    {
        if (!active.Components.TryGetValue(component, out var pointer)) throw new InvalidOperationException($"The active {component} component is missing. Complete setup or run Repair.");
        var root = Path.GetFullPath(Path.Combine(paths.DataRoot, pointer.RelativePath.Replace('/', Path.DirectorySeparatorChar)));
        var relative = Path.GetRelativePath(paths.DataRoot, root);
        if (Path.IsPathRooted(relative) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new InvalidOperationException("An active worker path escaped the selected data root.");
        var inventory = await ComponentVerifier.VerifyAsync(root, pointer.ManifestHash, cancellationToken);
        if (!string.Equals(inventory.Component, component, StringComparison.Ordinal)) throw new InvalidOperationException($"The active {component} inventory has the wrong identity.");
        var executable = Path.Combine(root, inventory.Entrypoint.Replace('/', Path.DirectorySeparatorChar));
        return new WorkerDefinition(workerName, executable, new Uri("http://127.0.0.1:1/"))
        {
            WorkingDirectory = root,
            LogDirectory = paths.LogsRoot,
        };
    }

    private static LauncherSessionPlan Compose(string dataRoot, string logsRoot, WorkerDefinition voice, WorkerDefinition studio, Uri? ollamaEndpoint = null, IReadOnlyDictionary<string, string>? additionalSettings = null)
    {
        var ports = new PortAllocator().Reserve(2);
        try
        {
            var composedVoice = voice.WithPort(ports[0], "YOUTUBER_VOICE_PORT");
            var composedStudio = studio.WithPort(ports[1], "YOUTUBER_STUDIO_PORT");
            var settings = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["YOUTUBER_DATA_ROOT"] = dataRoot,
                ["YOUTUBER_LOGS_ROOT"] = logsRoot,
            };
            if (additionalSettings is not null)
            {
                foreach (var setting in additionalSettings) settings[setting.Key] = setting.Value;
            }
            return new LauncherSessionPlan(dataRoot, composedVoice, composedStudio, settings, SessionSecret.Create(), ollamaEndpoint, ports);
        }
        catch
        {
            ports.Dispose();
            throw;
        }
    }
}
