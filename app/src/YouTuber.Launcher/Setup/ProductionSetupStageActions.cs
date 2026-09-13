using System.IO;
using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.SystemChecks;

namespace YouTuber.Launcher.Setup;

/// <summary>Production composition for every durable first-run checkpoint.</summary>
public sealed class ProductionSetupStageActions : ISetupStageActions, ISetupRunFailureCleanup, IAsyncDisposable
{
    private readonly AppPaths _paths;
    private readonly IDistributionManifestSource _manifestSource;
    private readonly ISetupPrerequisiteProvisioner _prerequisites;
    private readonly ISetupComponentProvisioner _components;
    private readonly ISetupOllamaProvisioner _ollama;
    private readonly ISetupEndToEndHealth _health;
    private readonly ISystemProbe _systemProbe;
    private readonly PreflightEvaluator _preflight;
    private ISetupComponentLifecycle? _lifecycle;
    private EventHandler<ComponentProgress>? _progressHandler;
    private bool _disposed;

    public ProductionSetupStageActions(AppPaths paths)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _manifestSource = new EmbeddedDistributionManifestSource();
        _prerequisites = new ProductionPrerequisiteProvisioner(paths);
        _components = new ProductionComponentProvisioner(paths);
        var runtime = new ProductionRuntimeSession(paths);
        _ollama = runtime;
        _health = runtime;
        _systemProbe = new WindowsSystemProbe();
        _preflight = new PreflightEvaluator();
    }

    public ProductionSetupStageActions(
        AppPaths paths,
        IDistributionManifestSource manifestSource,
        ISetupPrerequisiteProvisioner prerequisites,
        ISetupComponentProvisioner components,
        ISetupOllamaProvisioner ollama,
        ISetupEndToEndHealth health,
        ISystemProbe systemProbe,
        PreflightEvaluator preflight)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _manifestSource = manifestSource ?? throw new ArgumentNullException(nameof(manifestSource));
        _prerequisites = prerequisites ?? throw new ArgumentNullException(nameof(prerequisites));
        _components = components ?? throw new ArgumentNullException(nameof(components));
        _ollama = ollama ?? throw new ArgumentNullException(nameof(ollama));
        _health = health ?? throw new ArgumentNullException(nameof(health));
        _systemProbe = systemProbe ?? throw new ArgumentNullException(nameof(systemProbe));
        _preflight = preflight ?? throw new ArgumentNullException(nameof(preflight));
    }

    public bool HasRunningSession => _health.HasRunningSession;
    public ProductionRuntimeSession RunningSession
        => _health as ProductionRuntimeSession ?? throw new InvalidOperationException("The injected health verifier does not expose a production runtime session.");

    public async Task<SetupCoordinator> CreateCoordinatorAsync(
        ISetupJournal journal,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(journal);
        if (_progressHandler is not null) throw new InvalidOperationException("The production setup coordinator has already been created.");
        _lifecycle = _components as ISetupComponentLifecycle
            ?? throw new InvalidOperationException("Production setup components must expose recovery and operation control.");
        var fetched = await _manifestSource.GetAsync(cancellationToken);
        var coordinator = await SetupCoordinator.CreateAsync(journal, _lifecycle.CreateRecoveryProbe(fetched), _lifecycle, cancellationToken);
        _progressHandler = (_, progress) => coordinator.ReportComponent(progress);
        _lifecycle.ProgressChanged += _progressHandler;
        return coordinator;
    }

    public async Task PreflightAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        var fetched = await _manifestSource.GetAsync(cancellationToken);
        var report = _preflight.Evaluate(await _systemProbe.ProbeAsync(_paths, cancellationToken), fetched.Manifest, _paths);
        if (report.OverallStatus == PreflightStatus.Block)
            throw new InvalidOperationException(string.Join(" ", report.Checks.Where(check => check.Status == PreflightStatus.Block).Select(check => check.Action)));
    }

    public Task ChooseDataRootAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        Directory.CreateDirectory(_paths.DataRoot);
        return Task.CompletedTask;
    }

    public async Task PrerequisitesAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        await _prerequisites.EnsureAsync(await _manifestSource.GetAsync(cancellationToken), cancellationToken);
    }

    public async Task FetchCatalogAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        _ = await _manifestSource.GetAsync(cancellationToken);
    }

    public async Task DownloadAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        await _components.DownloadAsync(await _manifestSource.GetAsync(cancellationToken), cancellationToken);
    }

    public async Task VerifyAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        await _components.VerifyAsync(await _manifestSource.GetAsync(cancellationToken), cancellationToken);
    }

    public async Task ActivateAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        await _components.ActivateAsync(await _manifestSource.GetAsync(cancellationToken), cancellationToken);
    }

    public async Task OllamaModelsAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        await _ollama.EnsureModelsAsync(await _manifestSource.GetAsync(cancellationToken), cancellationToken);
    }

    public async Task EndToEndHealthAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        await _health.EnsureHealthyAsync(await _manifestSource.GetAsync(cancellationToken), cancellationToken);
        if (!_health.HasRunningSession) throw new InvalidOperationException("End-to-end health completed without a retained production runtime session.");
    }

    public async Task CleanupAfterFailedRunAsync()
    {
        if (_health is ISetupRunFailureCleanup healthCleanup) await healthCleanup.CleanupAfterFailedRunAsync();
        if (!ReferenceEquals(_health, _ollama) && _ollama is ISetupRunFailureCleanup ollamaCleanup)
            await ollamaCleanup.CleanupAfterFailedRunAsync();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ProductionSetupStageActions));
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        if (_lifecycle is not null && _progressHandler is not null) _lifecycle.ProgressChanged -= _progressHandler;
        try
        {
            if (ReferenceEquals(_ollama, _health)) await _ollama.DisposeAsync();
            else
            {
                try { await _health.DisposeAsync(); }
                finally { await _ollama.DisposeAsync(); }
            }
        }
        finally
        {
            if (_components is IDisposable components) components.Dispose();
            if (_prerequisites is IDisposable prerequisites) prerequisites.Dispose();
            _manifestSource.Dispose();
        }
    }
}
