using System.IO;
using YouTuber.Launcher.Activation;
using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.Downloads;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Uninstall;

namespace YouTuber.Launcher.Setup;

public interface IDistributionManifestSource : IDisposable
{
    Task<FetchedDistributionManifest> GetAsync(CancellationToken cancellationToken);
}

public sealed class EmbeddedDistributionManifestSource : IDistributionManifestSource
{
    private readonly DistributionManifestFetcher _fetcher;
    private FetchedDistributionManifest? _cached;

    public EmbeddedDistributionManifestSource(DistributionManifestFetcher? fetcher = null) => _fetcher = fetcher ?? new DistributionManifestFetcher();

    public async Task<FetchedDistributionManifest> GetAsync(CancellationToken cancellationToken)
        => _cached ??= await _fetcher.FetchAsync(BootstrapCatalogLoader.LoadEmbedded(), cancellationToken);

    public void Dispose() => _fetcher.Dispose();
}

public interface ISetupDownloadClient : IDisposable
{
    Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken);
}

public sealed class SetupDownloadClient : ISetupDownloadClient
{
    private readonly ResumableDownloader _downloader;
    public SetupDownloadClient(ResumableDownloader? downloader = null) => _downloader = downloader ?? new ResumableDownloader();
    public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken) => _downloader.DownloadAsync(request, cancellationToken: cancellationToken);
    public void Dispose() => _downloader.Dispose();
}

public static class DistributionArtifactRequests
{
    public static DownloadRequest Download(ImmutableArtifactMetadata artifact, string target)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        return new DownloadRequest(HuggingFaceUrl.Create(artifact.Repo, artifact.Revision, artifact.Path), target, artifact.Size, artifact.Sha256);
    }

    public static DownloadRequest Download(SignedInstallerMetadata artifact, string target)
        => Download(new ImmutableArtifactMetadata(artifact.Repo, artifact.Revision, artifact.Path, artifact.Size, artifact.Sha256), target);

    public static InstallerArtifact Installer(SignedInstallerMetadata artifact, string localPath)
        => new(localPath, artifact.Size, artifact.Sha256, artifact.Publisher);
}

public interface ISetupPrerequisiteProvisioner
{
    Task EnsureAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken);
}

public sealed class ProductionPrerequisiteProvisioner : ISetupPrerequisiteProvisioner, IDisposable
{
    private readonly AppPaths _paths;
    private readonly IWebView2Manager _webView2;
    private readonly ISetupDownloadClient _downloads;
    private readonly OwnershipStore _ownership;

    public ProductionPrerequisiteProvisioner(AppPaths paths, IWebView2Manager? webView2 = null, ISetupDownloadClient? downloads = null)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _webView2 = webView2 ?? new WebView2Manager();
        _downloads = downloads ?? new SetupDownloadClient();
        _ownership = new OwnershipStore(paths.DataRoot, UninstallPreparation.InstallId);
    }

    public async Task EnsureAsync(FetchedDistributionManifest fetched, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(fetched);
        if (_webView2.IsAvailable()) return;
        var metadata = fetched.Manifest.WebView2?.Installer ?? throw new ManifestValidationException("WebView2 publication metadata is missing.");
        Directory.CreateDirectory(_paths.DownloadsRoot);
        var target = Path.Combine(_paths.DownloadsRoot, "webview2-installer.exe");
        await _ownership.ReserveOwnedFileAsync(target, cancellationToken);
        await _downloads.DownloadAsync(DistributionArtifactRequests.Download(metadata, target), cancellationToken);
        try
        {
            await _webView2.EnsureAvailableAsync(DistributionArtifactRequests.Installer(metadata, target), cancellationToken);
        }
        catch (Exception exception) when (exception is InstallerVerificationException or InvalidOperationException)
        {
            throw new InvalidOperationException("Microsoft Edge WebView2 Runtime is required. Its manifest-pinned Microsoft-signed installer could not be verified or installed; retry setup or install WebView2 from Microsoft, then retry.", exception);
        }
    }

    public void Dispose() => _downloads.Dispose();
}

public interface ISetupComponentProvisioner
{
    Task DownloadAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken);
    Task VerifyAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken);
    Task ActivateAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken);
}

public interface ISetupComponentLifecycle : ISetupOperationControl
{
    event EventHandler<ComponentProgress>? ProgressChanged;
    ISetupRecoveryProbe CreateRecoveryProbe(FetchedDistributionManifest manifest);
}

public sealed class ProductionComponentProvisioner : ISetupComponentProvisioner, ISetupComponentLifecycle, IDisposable
{
    private readonly AppPaths _paths;
    private readonly ISetupDownloadClient _downloads;
    private readonly ComponentActivator _runtime;
    private readonly DataAssetInstaller _data;
    private readonly OwnershipStore _ownership;
    private readonly ActiveComponentsStore _store;
    private readonly object _operationGate = new();
    private readonly Dictionary<string, ControlledDownload> _controlledDownloads = new(StringComparer.Ordinal);
    private Task? _activationRun;
    private ActivationCloseResult _lastActivationResult = ActivationCloseResult.RolledBackSafe;

    public ProductionComponentProvisioner(AppPaths paths, ISetupDownloadClient? downloads = null, ComponentActivator? runtime = null, DataAssetInstaller? data = null)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _downloads = downloads ?? new SetupDownloadClient();
        _store = new ActiveComponentsStore(paths.DataRoot);
        _runtime = runtime ?? new ComponentActivator(_store);
        _data = data ?? new DataAssetInstaller(_store);
        _ownership = new OwnershipStore(paths.DataRoot, UninstallPreparation.InstallId);
    }

    public event EventHandler<ComponentProgress>? ProgressChanged;

    public ISetupRecoveryProbe CreateRecoveryProbe(FetchedDistributionManifest fetched)
    {
        ArgumentNullException.ThrowIfNull(fetched);
        var artifacts = fetched.Manifest.Components
            .OrderBy(entry => entry.Key, StringComparer.Ordinal)
            .Select(entry => new SetupDownloadArtifact(entry.Key, fetched.Manifest.Release, Request(entry.Key, entry.Value)))
            .ToArray();
        lock (_operationGate)
        {
            foreach (var artifact in artifacts)
            {
                if (_controlledDownloads.ContainsKey(artifact.Name)) continue;
                _controlledDownloads[artifact.Name] = new ControlledDownload(
                    artifact.Name,
                    artifact.Version,
                    artifact.Request,
                    new CancellationTokenSource());
            }
        }
        return new SetupRecoveryProbe(artifacts, _store, fetched.Manifest.Components.Keys);
    }

    public async Task DownloadAsync(FetchedDistributionManifest fetched, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_paths.DownloadsRoot);
        foreach (var (kind, component) in fetched.Manifest.Components.OrderBy(entry => entry.Key, StringComparer.Ordinal))
        {
            var request = Request(kind, component);
            await _ownership.ReserveOwnedFileAsync(request.TargetPath, cancellationToken);
            await StartDownloadAsync(kind, fetched.Manifest.Release, request, cancellationToken);
        }
    }

    public async Task VerifyAsync(FetchedDistributionManifest fetched, CancellationToken cancellationToken)
    {
        foreach (var (kind, component) in fetched.Manifest.Components.OrderBy(entry => entry.Key, StringComparer.Ordinal))
        {
            var request = Request(kind, component);
            if (!File.Exists(request.TargetPath) || new FileInfo(request.TargetPath).Length != request.ExpectedSize ||
                !await Sha256Verifier.MatchesAsync(request.TargetPath, request.ExpectedSha256, cancellationToken))
            {
                Publish(kind, fetched.Manifest.Release, request, "Failed", "The downloaded artifact did not match its signed release metadata.");
                throw new InvalidDataException($"Downloaded component '{kind}' does not match the signed distribution manifest.");
            }
        }
    }

    public async Task ActivateAsync(FetchedDistributionManifest fetched, CancellationToken cancellationToken)
    {
        Task run;
        lock (_operationGate)
        {
            if (_activationRun is not null) throw new InvalidOperationException("Component activation is already running.");
            run = ActivateCoreAsync(fetched, cancellationToken);
            _activationRun = run;
        }

        try { await run; }
        finally
        {
            lock (_operationGate)
            {
                if (ReferenceEquals(_activationRun, run)) _activationRun = null;
            }
        }
    }

    private async Task ActivateCoreAsync(FetchedDistributionManifest fetched, CancellationToken cancellationToken)
    {
        var recoveryCompleted = false;
        try
        {
            await _runtime.RecoverAsync(cancellationToken);
            recoveryCompleted = true;
            foreach (var (kind, component) in fetched.Manifest.Components.OrderBy(entry => entry.Key, StringComparer.Ordinal))
            {
                var request = Request(kind, component);
                if (component.PackageType == DistributionPackageType.RuntimeArchive)
                {
                    await _runtime.ActivateAsync(new ComponentActivationRequest(kind, DistributionInstallRoot.Runtime, fetched.Manifest.Release, request.TargetPath, component.ArchiveRoot, component.ManifestSha256!, component.Sha256, component.ExpandedSize, component.InstallSize), cancellationToken);
                }
                else if (component.PackageType == DistributionPackageType.DataArchive)
                {
                    await _data.InstallAsync(new DataAssetInstallRequest(kind, fetched.Manifest.Release, request.TargetPath, component.ArchiveRoot, component.ManifestSha256!, component.Sha256, component.InstallRoot!, component.ExpandedSize, component.InstallSize), cancellationToken);
                }
                else
                {
                    throw new ManifestValidationException($"Component '{kind}' has an unsupported package type.");
                }
            }
            lock (_operationGate) _lastActivationResult = ActivationCloseResult.AtomicCommitCompleted;
        }
        catch
        {
            lock (_operationGate) _lastActivationResult = recoveryCompleted ? ActivationCloseResult.RolledBackSafe : ActivationCloseResult.Unsafe;
            throw;
        }
    }

    public async Task<DownloadPauseResult> PauseDownloadsAsync(CancellationToken cancellationToken)
    {
        ControlledDownload[] active;
        lock (_operationGate) active = _controlledDownloads.Values.Where(value => !value.Task.IsCompleted).ToArray();
        foreach (var operation in active) operation.Cancellation.Cancel();
        foreach (var operation in active)
        {
            try { await operation.Task.WaitAsync(cancellationToken); }
            catch (OperationCanceledException) when (operation.Cancellation.IsCancellationRequested) { }
            catch { return DownloadPauseResult.Failed; }
        }

        return active.All(HasDurableEvidence) ? DownloadPauseResult.PausedAndPersisted : DownloadPauseResult.Failed;
    }

    public async Task<ActivationCloseResult> WaitForActivationSafePointAsync(CancellationToken cancellationToken)
    {
        Task? active;
        lock (_operationGate) active = _activationRun;
        if (active is not null)
        {
            try { await active.WaitAsync(cancellationToken); }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }
            catch { }
        }
        lock (_operationGate) return _lastActivationResult;
    }

    public Task<ComponentOperationResult> PauseComponentAsync(string component, CancellationToken cancellationToken) =>
        PauseComponentCoreAsync(component, cancellationToken);

    public Task<ComponentOperationResult> ResumeComponentAsync(string component, CancellationToken cancellationToken) =>
        RestartComponentAsync(component, requireResumeEvidence: true, cancellationToken);

    public Task<ComponentOperationResult> RetryComponentAsync(string component, CancellationToken cancellationToken) =>
        RestartComponentAsync(component, requireResumeEvidence: false, cancellationToken);

    private async Task<ComponentOperationResult> PauseComponentCoreAsync(string component, CancellationToken cancellationToken)
    {
        ControlledDownload? operation;
        lock (_operationGate) _controlledDownloads.TryGetValue(component, out operation);
        if (operation is null || operation.Task.IsCompleted) return ComponentOperationResult.Rejected;
        operation.Cancellation.Cancel();
        try { await operation.Task.WaitAsync(cancellationToken); }
        catch (OperationCanceledException) when (operation.Cancellation.IsCancellationRequested) { }
        catch { return ComponentOperationResult.Rejected; }
        return HasDurableEvidence(operation) ? ComponentOperationResult.ConfirmedDurable : ComponentOperationResult.Rejected;
    }

    private async Task<ComponentOperationResult> RestartComponentAsync(
        string component,
        bool requireResumeEvidence,
        CancellationToken cancellationToken)
    {
        ControlledDownload? previous;
        lock (_operationGate) _controlledDownloads.TryGetValue(component, out previous);
        if (previous is null || !previous.Task.IsCompleted || (requireResumeEvidence && !HasDurableEvidence(previous)))
            return ComponentOperationResult.Rejected;
        try
        {
            await StartDownloadAsync(previous.Component, previous.Version, previous.Request, cancellationToken);
            return ComponentOperationResult.ConfirmedDurable;
        }
        catch
        {
            return ComponentOperationResult.Rejected;
        }
    }

    private Task StartDownloadAsync(string component, string version, DownloadRequest request, CancellationToken cancellationToken)
    {
        lock (_operationGate)
        {
            if (_controlledDownloads.TryGetValue(component, out var existing) && !existing.Task.IsCompleted)
                return existing.Task;
            if (existing is not null) existing.Cancellation.Dispose();
            var cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var operation = new ControlledDownload(component, version, request, cancellation);
            _controlledDownloads[component] = operation;
            operation.Task = RunDownloadAsync(operation);
            return operation.Task;
        }
    }

    private async Task RunDownloadAsync(ControlledDownload operation)
    {
        Publish(operation, "Downloading");
        try
        {
            await _downloads.DownloadAsync(operation.Request, operation.Cancellation.Token);
            Publish(operation, "Completed");
        }
        catch (OperationCanceledException) when (operation.Cancellation.IsCancellationRequested)
        {
            Publish(operation, "Paused");
            throw;
        }
        catch (Exception exception)
        {
            Publish(operation, "Failed", exception.Message);
            throw;
        }
        finally
        {
            operation.Cancellation.Dispose();
        }
    }

    private void Publish(ControlledDownload operation, string status, string? error = null)
        => Publish(operation.Component, operation.Version, operation.Request, status, error);

    private void Publish(string component, string version, DownloadRequest request, string status, string? error = null)
    {
        var partialPath = request.TargetPath + ".partial";
        var transferred = File.Exists(request.TargetPath)
            ? new FileInfo(request.TargetPath).Length
            : File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0;
        ProgressChanged?.Invoke(this, new ComponentProgress(
            component,
            version,
            transferred,
            request.ExpectedSize,
            0,
            null,
            status,
            error));
    }

    private static bool HasDurableEvidence(ControlledDownload operation)
    {
        if (File.Exists(operation.Request.TargetPath) && new FileInfo(operation.Request.TargetPath).Length == operation.Request.ExpectedSize) return true;
        var partialPath = operation.Request.TargetPath + ".partial";
        var partialLength = File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0;
        var metadata = ResumeMetadata.Load(partialPath + ".json");
        return metadata is not null && metadata.Matches(operation.Request, partialLength);
    }

    private DownloadRequest Request(string kind, DistributionComponent component)
        => new(HuggingFaceUrl.Create(component.Repo, component.Revision, component.Path), Path.Combine(_paths.DownloadsRoot, kind + ".zip"), component.Size, component.Sha256);

    public void Dispose() => _downloads.Dispose();

    private sealed class ControlledDownload(string component, string version, DownloadRequest request, CancellationTokenSource cancellation)
    {
        public string Component { get; } = component;
        public string Version { get; } = version;
        public DownloadRequest Request { get; } = request;
        public CancellationTokenSource Cancellation { get; } = cancellation;
        public Task Task { get; set; } = Task.CompletedTask;
    }
}

public interface ISetupOllamaProvisioner : IAsyncDisposable
{
    Task EnsureModelsAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken);
}

public interface ISetupEndToEndHealth : IAsyncDisposable
{
    bool HasRunningSession { get; }
    Task EnsureHealthyAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken);
}
