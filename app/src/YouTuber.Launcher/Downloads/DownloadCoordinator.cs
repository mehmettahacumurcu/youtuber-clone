namespace YouTuber.Launcher.Downloads;

public sealed class DownloadCoordinator : IDisposable
{
    private readonly ResumableDownloader _downloader;
    private readonly object _gate = new();
    private ActiveRun? _activeRun;
    private DownloadRequest? _lastRequest;

    public DownloadCoordinator(ResumableDownloader downloader) => _downloader = downloader;
    public DownloadState State { get; private set; } = DownloadState.Queued;

    public Task<DownloadResult> StartAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            if (_activeRun is not null || State is not (DownloadState.Queued or DownloadState.Completed or DownloadState.Cancelled))
            {
                return Task.FromException<DownloadResult>(new InvalidOperationException("The download is not ready to start."));
            }

            return BeginRun(request, progress, cancellationToken);
        }
    }

    public Task<DownloadResult> ResumeAsync(IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            if (_activeRun is not null || State != DownloadState.Paused || _lastRequest is null)
            {
                return Task.FromException<DownloadResult>(new InvalidOperationException("No paused download is available."));
            }

            return BeginRun(_lastRequest, progress, cancellationToken);
        }
    }

    public Task<DownloadResult> RetryAsync(IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            if (_activeRun is not null || State != DownloadState.Failed || _lastRequest is null)
            {
                return Task.FromException<DownloadResult>(new InvalidOperationException("No failed download is available."));
            }

            return BeginRun(_lastRequest, progress, cancellationToken);
        }
    }

    public void Pause() => CancelActiveRun(cancelled: false);

    public void Cancel() => CancelActiveRun(cancelled: true);

    private Task<DownloadResult> BeginRun(DownloadRequest request, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        var cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var run = new ActiveRun(cancellation);
        _activeRun = run;
        _lastRequest = request;
        State = DownloadState.Downloading;
        _ = ExecuteAsync(run, request, progress);
        return run.Completion.Task;
    }

    private async Task ExecuteAsync(ActiveRun run, DownloadRequest request, IProgress<DownloadProgress>? progress)
    {
        try
        {
            var result = await _downloader.DownloadAsync(request, progress, run.Cancellation.Token);
            Complete(run, result.State, () => run.Completion.TrySetResult(result));
        }
        catch (OperationCanceledException) when (run.Cancellation.IsCancellationRequested)
        {
            Complete(run, run.Cancelled ? DownloadState.Cancelled : DownloadState.Paused, () => run.Completion.TrySetCanceled(run.Cancellation.Token));
        }
        catch (Exception exception)
        {
            Complete(run, DownloadState.Failed, () => run.Completion.TrySetException(exception));
        }
    }

    private void Complete(ActiveRun run, DownloadState state, Action completeTask)
    {
        lock (_gate)
        {
            if (!ReferenceEquals(_activeRun, run)) return;
            State = state;
            _activeRun = null;
            completeTask();
            run.Cancellation.Dispose();
        }
    }

    private void CancelActiveRun(bool cancelled)
    {
        ActiveRun? run;
        lock (_gate)
        {
            run = _activeRun;
            if (run is null) return;
            run.Cancelled |= cancelled;
            State = cancelled ? DownloadState.Cancelled : DownloadState.Paused;
        }

        try { run.Cancellation.Cancel(); }
        catch (ObjectDisposedException) { }
    }

    public void Dispose()
    {
        Cancel();
    }

    private sealed class ActiveRun(CancellationTokenSource cancellation)
    {
        public CancellationTokenSource Cancellation { get; } = cancellation;
        public TaskCompletionSource<DownloadResult> Completion { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Cancelled { get; set; }
    }
}
