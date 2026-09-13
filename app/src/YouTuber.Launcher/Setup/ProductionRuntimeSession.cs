using System.IO;
using YouTuber.Launcher.Activation;
using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.Ollama;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Web;
using YouTuber.Launcher.Uninstall;

namespace YouTuber.Launcher.Setup;

public interface IProductionRuntimeComposer : IDisposable
{
    Task<WorkerRuntime> ComposeAsync(CancellationToken cancellationToken);
}

public sealed class ProductionRuntimeComposer : IProductionRuntimeComposer
{
    private readonly AppPaths _paths;
    private readonly WorkerRuntimeFactory _factory;
    private WorkerRuntime? _runtime;

    public ProductionRuntimeComposer(AppPaths paths, WorkerRuntimeFactory? factory = null)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _factory = factory ?? new WorkerRuntimeFactory();
    }

    public async Task<WorkerRuntime> ComposeAsync(CancellationToken cancellationToken)
    {
        if (_runtime is not null) return _runtime;
        var store = new ActiveComponentsStore(_paths.DataRoot);
        var active = await store.LoadAsync(cancellationToken);
        var voice = await InstalledWorkerResolver.ResolveAsync(_paths, active, "voice_runtime", "voice", cancellationToken);
        var studio = await InstalledWorkerResolver.ResolveAsync(_paths, active, "studio_runtime", "studio", cancellationToken);
        return _runtime = _factory.Create(_paths.DataRoot, voice, studio);
    }

    public void Dispose() => _factory.Dispose();
}

public static class InstalledWorkerResolver
{
    public static async Task<WorkerDefinition> ResolveAsync(AppPaths paths, ActiveComponents active, string component, string workerName, CancellationToken cancellationToken)
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
}

public interface IControlledOllamaRuntime : IAsyncDisposable
{
    Task EnsureModelsAsync(FetchedDistributionManifest manifest, WorkerRuntime runtime, CancellationToken cancellationToken);
    Task StartExistingAsync(FetchedDistributionManifest manifest, WorkerRuntime runtime, CancellationToken cancellationToken);
    Task VerifyModelsAsync(FetchedDistributionManifest manifest, WorkerRuntime runtime, CancellationToken cancellationToken);
    Task StopAsync(CancellationToken cancellationToken);
}

public interface IOllamaRuntimeOperations
{
    OllamaInstallation SelectInstallation(OllamaInstallation? detected, OllamaInstallation composed);
    Task StartInstalledAsync(OllamaPublication publication, OllamaInstallation installation, PortReservation reservation, CancellationToken cancellationToken);
    Task<InstallerRunResult> InstallNewAsync(OllamaInstallerArtifact artifact, OllamaInstallation installation, PortReservation reservation, IInstallerRunner installer, CancellationToken cancellationToken);
    Task EnsureModelsAsync(OllamaInstallation installation, OllamaImportArtifacts artifacts, OllamaModelRolesMetadata roles, CancellationToken cancellationToken);
    Task VerifyModelsAsync(OllamaInstallation installation, OllamaModelRolesMetadata roles, CancellationToken cancellationToken);
    Task StopAsync(CancellationToken cancellationToken);
}

public sealed class OllamaRuntimeOperations(OllamaManager manager) : IOllamaRuntimeOperations
{
    public OllamaInstallation SelectInstallation(OllamaInstallation? detected, OllamaInstallation composed)
        => manager.SelectInstallation(detected, composed);

    public Task StartInstalledAsync(OllamaPublication publication, OllamaInstallation installation, PortReservation reservation, CancellationToken cancellationToken)
        => manager.StartInstalledControlledAsync(publication, installation, reservation, cancellationToken);

    public Task<InstallerRunResult> InstallNewAsync(OllamaInstallerArtifact artifact, OllamaInstallation installation, PortReservation reservation, IInstallerRunner installer, CancellationToken cancellationToken)
        => manager.InstallNewComposedAsync(artifact, installation, reservation, installer, cancellationToken);

    public async Task EnsureModelsAsync(OllamaInstallation installation, OllamaImportArtifacts artifacts, OllamaModelRolesMetadata roles, CancellationToken cancellationToken)
    {
        using var client = manager.CreateControlledClient(installation);
        await manager.EnsureModelsAsync(client, installation, artifacts, roles, cancellationToken: cancellationToken);
    }

    public async Task VerifyModelsAsync(OllamaInstallation installation, OllamaModelRolesMetadata roles, CancellationToken cancellationToken)
    {
        using var client = manager.CreateControlledClient(installation);
        await client.EnsureHealthyAsync(cancellationToken);
        await client.EnsureModelIdentityAsync(roles.Verifier, cancellationToken);
        var models = await client.ListModelDetailsAsync(cancellationToken);
        if (models.Count(model => OllamaClient.MatchesCanonicalModelName(model.Name, roles.Speaker.Name, allowImplicitLatestTag: true) && model.Size == roles.Speaker.InstalledSize) != 1)
            throw new InvalidOperationException("The launcher-owned Ollama server does not contain both manifest-authorized model roles.");
    }

    public Task StopAsync(CancellationToken cancellationToken) => manager.StopControlledServeAsync(cancellationToken);
}

public sealed class ControlledOllamaRuntime : IControlledOllamaRuntime
{
    private readonly AppPaths _paths;
    private readonly IOllamaRuntimeOperations _operations;
    private readonly IOllamaInstallationProbe _installationProbe;
    private readonly ISetupDownloadClient _downloads;
    private readonly AuthenticodeVerifier _authenticode;
    private readonly IInstallerRunner _installer;
    private readonly OwnershipStore _ownership;
    private OllamaInstallation? _selectedInstallation;
    private string? _preparedPublicationIdentity;
    private bool _started;

    public ControlledOllamaRuntime(
        AppPaths paths,
        OllamaManager? manager = null,
        ISetupDownloadClient? downloads = null,
        AuthenticodeVerifier? authenticode = null,
        IInstallerRunner? installer = null,
        IOllamaInstallationProbe? installationProbe = null,
        IOllamaRuntimeOperations? operations = null)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        if (manager is not null && operations is not null) throw new ArgumentException("Specify either an Ollama manager or runtime operations, not both.");
        _operations = operations ?? new OllamaRuntimeOperations(manager ?? new OllamaManager(importRoot: paths.DownloadsRoot));
        _installationProbe = installationProbe ?? new OllamaInstallationDetector();
        _downloads = downloads ?? new SetupDownloadClient();
        _authenticode = authenticode ?? new AuthenticodeVerifier();
        _installer = installer ?? new InstallerRunner();
        _ownership = new OwnershipStore(paths.DataRoot, UninstallPreparation.InstallId);
    }

    public async Task EnsureModelsAsync(FetchedDistributionManifest fetched, WorkerRuntime runtime, CancellationToken cancellationToken)
    {
        var metadata = fetched.Manifest.Ollama ?? throw new ManifestValidationException("Ollama publication metadata is missing.");
        var selected = await SelectInstallationAsync(runtime, cancellationToken);
        var publication = OllamaPublicationFactory.Create(fetched.Manifest, fetched.Sha256);
        EnsurePreparedPublicationMatches(publication);
        if (_started)
        {
            await _operations.VerifyModelsAsync(selected, metadata.Models, cancellationToken);
            return;
        }

        Directory.CreateDirectory(_paths.DownloadsRoot);
        var installerPath = Path.Combine(_paths.DownloadsRoot, "ollama-installer.exe");
        var verifierGgufPath = Path.Combine(_paths.DownloadsRoot, "qwen3-verifier.gguf");
        var ggufPath = Path.Combine(_paths.DownloadsRoot, "speaker-model.gguf");
        var modelfilePath = Path.Combine(_paths.DownloadsRoot, "Speaker.Modelfile");
        if (!selected.IsExisting && _preparedPublicationIdentity is null)
            await DownloadOwnedAsync(metadata.Installer, installerPath, cancellationToken);
        await DownloadOwnedAsync(metadata.Models.Verifier.Source, verifierGgufPath, cancellationToken);
        await DownloadOwnedAsync(metadata.Models.Speaker.Gguf, ggufPath, cancellationToken);
        await DownloadOwnedAsync(metadata.Models.Speaker.Modelfile, modelfilePath, cancellationToken);

        if (selected.IsExisting || _preparedPublicationIdentity is not null)
        {
            await _operations.StartInstalledAsync(publication, selected, runtime.OllamaReservation, cancellationToken);
            _started = true;
            RememberPreparedPublication(publication);
        }
        else
        {
            var localPublication = publication with { Installer = DistributionArtifactRequests.Installer(metadata.Installer, installerPath) };
            using var verified = await _authenticode.VerifyAsync(localPublication.Installer, cancellationToken);
            var result = await _operations.InstallNewAsync(new OllamaInstallerArtifact(verified, localPublication), selected, runtime.OllamaReservation, _installer, cancellationToken);
            if (result.TimedOut || result.ExitCode != 0) throw new InvalidOperationException("The verified Ollama installer did not complete successfully.");
            _started = true;
            RememberPreparedPublication(publication);
        }

        try
        {
            await _operations.EnsureModelsAsync(
                selected,
                new OllamaImportArtifacts(
                    verifierGgufPath, metadata.Models.Verifier.Source.Size, metadata.Models.Verifier.Source.Sha256,
                    ggufPath, metadata.Models.Speaker.Gguf.Size, metadata.Models.Speaker.Gguf.Sha256,
                    modelfilePath, metadata.Models.Speaker.Modelfile.Size, metadata.Models.Speaker.Modelfile.Sha256),
                metadata.Models,
                cancellationToken);
        }
        catch
        {
            await StopAsync(CancellationToken.None);
            throw;
        }
    }

    private async Task DownloadOwnedAsync(ImmutableArtifactMetadata metadata, string target, CancellationToken cancellationToken)
    {
        await _ownership.ReserveOwnedFileAsync(target, cancellationToken);
        await _downloads.DownloadAsync(DistributionArtifactRequests.Download(metadata, target), cancellationToken);
    }

    private Task DownloadOwnedAsync(SignedInstallerMetadata metadata, string target, CancellationToken cancellationToken)
        => DownloadOwnedAsync(new ImmutableArtifactMetadata(metadata.Repo, metadata.Revision, metadata.Path, metadata.Size, metadata.Sha256), target, cancellationToken);

    public async Task StartExistingAsync(FetchedDistributionManifest fetched, WorkerRuntime runtime, CancellationToken cancellationToken)
    {
        if (_started) return;
        var publication = OllamaPublicationFactory.Create(fetched.Manifest, fetched.Sha256);
        EnsurePreparedPublicationMatches(publication);
        var selected = await SelectInstallationAsync(runtime, cancellationToken);
        await _operations.StartInstalledAsync(publication, selected, runtime.OllamaReservation, cancellationToken);
        _started = true;
        RememberPreparedPublication(publication);
    }

    public async Task VerifyModelsAsync(FetchedDistributionManifest fetched, WorkerRuntime runtime, CancellationToken cancellationToken)
    {
        var roles = fetched.Manifest.Ollama?.Models ?? throw new ManifestValidationException("Ollama model roles are missing.");
        await _operations.VerifyModelsAsync(await SelectInstallationAsync(runtime, cancellationToken), roles, cancellationToken);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        try { await _operations.StopAsync(cancellationToken); }
        finally { _started = false; }
    }

    private async Task<OllamaInstallation> SelectInstallationAsync(WorkerRuntime runtime, CancellationToken cancellationToken)
    {
        if (_selectedInstallation is not null)
        {
            if (_selectedInstallation.ApiEndpoint != runtime.Ollama.ApiEndpoint ||
                !string.Equals(_selectedInstallation.ModelDirectory, runtime.Ollama.ModelDirectory, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("The composed Ollama runtime changed during the session.");
            return _selectedInstallation;
        }
        var detected = await _installationProbe.DetectAsync(cancellationToken);
        return _selectedInstallation = _operations.SelectInstallation(detected, runtime.Ollama);
    }

    private void RememberPreparedPublication(OllamaPublication publication)
        => _preparedPublicationIdentity ??= publication.ManifestIdentity + ":" + publication.ReleaseIdentity;

    private void EnsurePreparedPublicationMatches(OllamaPublication publication)
    {
        if (_preparedPublicationIdentity is not null &&
            !string.Equals(_preparedPublicationIdentity, publication.ManifestIdentity + ":" + publication.ReleaseIdentity, StringComparison.Ordinal))
            throw new InvalidOperationException("The prepared Ollama installation belongs to a different signed publication.");
    }

    public async ValueTask DisposeAsync()
    {
        try { await StopAsync(CancellationToken.None); }
        finally { _downloads.Dispose(); }
    }
}

public sealed class ProductionRuntimeSession : ISetupOllamaProvisioner, ISetupEndToEndHealth, ISetupRunFailureCleanup
{
    private readonly AppPaths _paths;
    private readonly IProductionRuntimeComposer _composer;
    private readonly IControlledOllamaRuntime _ollama;
    private IWorkerSupervisor? _workers;
    private readonly Func<Uri, IWorkerSupervisor> _workerFactory;
    private readonly IWorkerHealthClient _health;
    private WorkerRuntime? _runtime;
    private bool _modelsPrepared;
    private bool _disposed;

    public ProductionRuntimeSession(
        AppPaths paths,
        IProductionRuntimeComposer? composer = null,
        IControlledOllamaRuntime? ollama = null,
        IWorkerSupervisor? workers = null,
        IWorkerHealthClient? health = null,
        Func<Uri, IWorkerSupervisor>? workerFactory = null)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _composer = composer ?? new ProductionRuntimeComposer(paths);
        _ollama = ollama ?? new ControlledOllamaRuntime(paths);
        _workers = workers;
        _workerFactory = workerFactory ?? (endpoint => new WorkerSupervisor(dependencies: new LoopbackOllamaDependencyHealth(endpoint)));
        _health = health ?? new HttpWorkerHealthClient();
    }

    public bool HasRunningSession => SessionPlan is not null && _workers?.State == WorkerSupervisorState.Running;
    public LauncherSessionPlan? SessionPlan { get; private set; }
    public IWorkerSupervisor Workers => _workers ?? throw new InvalidOperationException("The production worker session has not been composed.");

    public async Task EnsureModelsAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        _runtime ??= await _composer.ComposeAsync(cancellationToken);
        await _ollama.EnsureModelsAsync(manifest, _runtime, cancellationToken);
        _modelsPrepared = true;
    }

    public async Task EnsureHealthyAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        if (HasRunningSession) return;
        _runtime ??= await _composer.ComposeAsync(cancellationToken);
        _workers ??= _workerFactory(_runtime.Ollama.ApiEndpoint);
        try
        {
            if (!_modelsPrepared) await _ollama.StartExistingAsync(manifest, _runtime, cancellationToken);
            await _ollama.VerifyModelsAsync(manifest, _runtime, cancellationToken);
            var roles = manifest.Manifest.Ollama!.Models;
            var settings = new Dictionary<string, string>(_runtime.Settings, StringComparer.Ordinal)
            {
                ["YOUTUBER_DATA_ROOT"] = _paths.DataRoot,
                ["YOUTUBER_LOGS_ROOT"] = _paths.LogsRoot,
                ["YOUTUBER_ANSWER_MODEL"] = roles.Speaker.Name,
                ["YOUTUBER_VERIFIER_MODEL"] = roles.Verifier.Name,
            };
            var secret = SessionSecret.Create();
            await _workers.StartAsync(_runtime.Voice, _runtime.Studio, secret, settings, cancellationToken);
            await _health.WaitForReadyAsync(_runtime.Voice, secret, _runtime.Voice.StartupTimeout, cancellationToken);
            await _health.WaitForReadyAsync(_runtime.Studio, secret, _runtime.Studio.StartupTimeout, cancellationToken);
            if (_workers.State != WorkerSupervisorState.Running) throw new InvalidOperationException("The production workers did not remain running after health verification.");
            SessionPlan = new LauncherSessionPlan(_paths.DataRoot, _runtime.Voice, _runtime.Studio, settings, secret, _runtime.Ollama.ApiEndpoint);
        }
        catch
        {
            await CleanupAfterFailedRunAsync();
            throw;
        }
    }

    public async Task CleanupAfterFailedRunAsync()
    {
        SessionPlan = null;
        var failedWorkers = _workers;
        _workers = null;
        try { if (failedWorkers is not null) await failedWorkers.DisposeAsync(); }
        catch { }
        try { await _ollama.StopAsync(CancellationToken.None); }
        catch { }
        _modelsPrepared = false;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ProductionRuntimeSession));
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        SessionPlan = null;
        try
        {
            if (_workers is not null) await _workers.DisposeAsync();
        }
        finally
        {
            try { await _ollama.DisposeAsync(); }
            finally
            {
                _composer.Dispose();
                if (_health is IDisposable disposable) disposable.Dispose();
            }
        }
    }
}
