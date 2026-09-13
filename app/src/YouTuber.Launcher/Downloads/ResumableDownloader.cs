using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text.Json;

namespace YouTuber.Launcher.Downloads;

public sealed record ResumeMetadata(string Source, long ExpectedSize, string ExpectedSha256, long BytesWritten, string? ETag)
{
    public static ResumeMetadata Create(DownloadRequest request, long bytesWritten, string? eTag) =>
        new(request.Source.AbsoluteUri, request.ExpectedSize, request.ExpectedSha256, bytesWritten, eTag?.Trim('"'));

    public static ResumeMetadata? Load(string path)
    {
        if (!File.Exists(path)) return null;
        try { return JsonSerializer.Deserialize<ResumeMetadata>(File.ReadAllText(path)); }
        catch (JsonException) { return null; }
    }

    public string ToJson() => JsonSerializer.Serialize(this);
    public void Save(string path)
    {
        var temporaryPath = path + ".tmp";
        using (var stream = new FileStream(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, FileOptions.WriteThrough))
        using (var writer = new StreamWriter(stream, System.Text.Encoding.UTF8, 4096, leaveOpen: true))
        {
            writer.Write(ToJson());
            writer.Flush();
            stream.Flush(flushToDisk: true);
        }

        File.Move(temporaryPath, path, overwrite: true);
    }

    public bool Matches(DownloadRequest request, long partialLength) =>
        MatchesRequest(request) &&
        BytesWritten == partialLength && partialLength >= 0 && partialLength <= request.ExpectedSize;

    public bool MatchesRequest(DownloadRequest request) =>
        string.Equals(Source, request.Source.AbsoluteUri, StringComparison.Ordinal) &&
        ExpectedSize == request.ExpectedSize &&
        string.Equals(ExpectedSha256, request.ExpectedSha256, StringComparison.Ordinal);
}

public sealed class ResumableDownloader : IDisposable
{
    private const int BufferSize = 1024 * 1024;
    private const long MetadataByteInterval = 16L * 1024 * 1024;
    private static readonly HashSet<string> HuggingFaceHosts = new(StringComparer.OrdinalIgnoreCase)
    {
        "huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
    };
    private readonly HttpClient _httpClient;
    private readonly int _maxAttempts;
    private readonly TimeSpan _retryDelay;

    public ResumableDownloader(int maxAttempts = 3, TimeSpan? retryDelay = null)
        : this(CreateProductionHandler(), maxAttempts, retryDelay)
    {
    }

    public ResumableDownloader(HttpMessageHandler handler, int maxAttempts = 3, TimeSpan? retryDelay = null)
    {
        ArgumentNullException.ThrowIfNull(handler);
        _httpClient = new HttpClient(handler, disposeHandler: false);
        _maxAttempts = Math.Clamp(maxAttempts, 1, 5);
        _retryDelay = retryDelay ?? TimeSpan.FromMilliseconds(150);
    }

    public async Task<DownloadResult> DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        request.Validate();
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(request.TargetPath))!);
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                return await DownloadAttemptAsync(request, progress, cancellationToken, allowRangeRestart: true);
            }
            catch (Exception exception) when (IsTransient(exception, cancellationToken) && attempt < _maxAttempts)
            {
                progress?.Report(new DownloadProgress(DownloadState.Retrying, PartialLength(request.TargetPath), request.ExpectedSize, 0, null, "Network retry."));
                var maximum = _retryDelay.TotalMilliseconds * Math.Pow(2, attempt - 1);
                var jitter = Random.Shared.NextDouble() * Math.Max(1, maximum * 0.25);
                await Task.Delay(TimeSpan.FromMilliseconds(maximum + jitter), cancellationToken);
            }
            catch (DownloadValidationException)
            {
                DeletePartial(request.TargetPath);
                throw;
            }
        }
    }

    private async Task<DownloadResult> DownloadAttemptAsync(DownloadRequest request, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken, bool allowRangeRestart)
    {
        var partialPath = request.TargetPath + ".partial";
        var metadataPath = partialPath + ".json";
        var partialLength = PartialLength(request.TargetPath);
        var metadata = ResumeMetadata.Load(metadataPath);
        if (partialLength > 0 && metadata is not null && metadata.MatchesRequest(request) && metadata.BytesWritten >= 0 && metadata.BytesWritten <= partialLength)
        {
            if (metadata.BytesWritten < partialLength)
            {
                using var partial = new FileStream(partialPath, FileMode.Open, FileAccess.Write, FileShare.None);
                partial.SetLength(metadata.BytesWritten);
            }

            partialLength = metadata.BytesWritten;
        }

        if (partialLength > 0 && (metadata is null || !metadata.Matches(request, partialLength)))
        {
            DeletePartial(request.TargetPath);
            partialLength = 0;
            metadata = null;
        }

        using var response = await SendFollowingRedirectsAsync(request.Source, partialLength, cancellationToken);
        if (partialLength > 0 && response.StatusCode == HttpStatusCode.OK)
        {
            if (!allowRangeRestart) throw new DownloadValidationException("The server repeatedly ignored the resume range.");
            DeletePartial(request.TargetPath);
            return await DownloadAttemptAsync(request, progress, cancellationToken, allowRangeRestart: false);
        }

        if (partialLength > 0)
        {
            if (response.StatusCode != HttpStatusCode.PartialContent) throw HttpFailure(response.StatusCode);
            ValidateContentRange(response, partialLength, request.ExpectedSize);
            ValidateEtag(response, metadata!.ETag);
        }
        else if (response.StatusCode != HttpStatusCode.OK)
        {
            throw HttpFailure(response.StatusCode);
        }

        var eTag = response.Headers.ETag?.Tag?.Trim('"');
        if (string.IsNullOrWhiteSpace(eTag))
        {
            throw new DownloadValidationException("The artifact response did not include an ETag for safe resume validation.");
        }
        var stopwatch = Stopwatch.StartNew();
        var bytesWritten = partialLength;
        var lastCheckpointBytes = bytesWritten;
        var lastCheckpointAt = stopwatch.Elapsed;
        var emaSpeed = 0d;
        var lastProgressAt = stopwatch.Elapsed;
        metadata = ResumeMetadata.Create(request, bytesWritten, eTag);
        metadata.Save(metadataPath);
        progress?.Report(new DownloadProgress(DownloadState.Downloading, bytesWritten, request.ExpectedSize, 0, null));

        {
            await using var input = await response.Content.ReadAsStreamAsync(cancellationToken);
            await using var output = new FileStream(partialPath, partialLength == 0 ? FileMode.Create : FileMode.Append, FileAccess.Write, FileShare.None, BufferSize, FileOptions.SequentialScan);
            try
            {
                var buffer = new byte[BufferSize];
                while (true)
                {
                    var read = await input.ReadAsync(buffer.AsMemory(), cancellationToken);
                    if (read == 0) break;
                    if (read > request.ExpectedSize - bytesWritten)
                    {
                        throw new DownloadValidationException("The artifact response exceeds the immutable manifest byte count.");
                    }

                    await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                    bytesWritten = checked(bytesWritten + read);
                    var elapsed = stopwatch.Elapsed;
                    var interval = elapsed - lastProgressAt;
                    if (interval >= TimeSpan.FromMilliseconds(100))
                    {
                        var instantaneous = (bytesWritten - partialLength) / Math.Max(elapsed.TotalSeconds, 0.001);
                        emaSpeed = emaSpeed == 0 ? instantaneous : (0.2 * instantaneous) + (0.8 * emaSpeed);
                        TimeSpan? remaining = emaSpeed > 0 ? TimeSpan.FromSeconds((request.ExpectedSize - bytesWritten) / emaSpeed) : null;
                        progress?.Report(new DownloadProgress(DownloadState.Downloading, bytesWritten, request.ExpectedSize, emaSpeed, remaining));
                        lastProgressAt = elapsed;
                    }

                    if (bytesWritten - lastCheckpointBytes >= MetadataByteInterval || elapsed - lastCheckpointAt >= TimeSpan.FromSeconds(5))
                    {
                        await output.FlushAsync(cancellationToken);
                        output.Flush(flushToDisk: true);
                        ResumeMetadata.Create(request, bytesWritten, eTag).Save(metadataPath);
                        lastCheckpointBytes = bytesWritten;
                        lastCheckpointAt = elapsed;
                    }
                }

                await output.FlushAsync(cancellationToken);
                output.Flush(flushToDisk: true);
                ResumeMetadata.Create(request, bytesWritten, eTag).Save(metadataPath);
            }
            catch (OperationCanceledException)
            {
                output.Flush(flushToDisk: true);
                ResumeMetadata.Create(request, bytesWritten, eTag).Save(metadataPath);
                progress?.Report(new DownloadProgress(DownloadState.Paused, bytesWritten, request.ExpectedSize, emaSpeed, null));
                throw;
            }
        }

        if (bytesWritten != request.ExpectedSize)
        {
            throw new DownloadValidationException("The downloaded byte count does not match the immutable manifest.");
        }

        progress?.Report(new DownloadProgress(DownloadState.Verifying, bytesWritten, request.ExpectedSize, emaSpeed, TimeSpan.Zero));
        if (!await Sha256Verifier.MatchesAsync(partialPath, request.ExpectedSha256, cancellationToken))
        {
            throw new DownloadValidationException("The downloaded SHA-256 does not match the immutable manifest.");
        }

        File.Move(partialPath, request.TargetPath, overwrite: true);
        File.Delete(metadataPath);
        progress?.Report(new DownloadProgress(DownloadState.Completed, bytesWritten, request.ExpectedSize, emaSpeed, TimeSpan.Zero));
        return new DownloadResult(DownloadState.Completed, request.TargetPath, bytesWritten);
    }

    private async Task<HttpResponseMessage> SendFollowingRedirectsAsync(Uri start, long rangeStart, CancellationToken cancellationToken)
    {
        Uri current = start;
        string? signedStorageHost = null;
        for (var hop = 0; hop <= 5; hop++)
        {
            using var message = new HttpRequestMessage(HttpMethod.Get, current);
            if (rangeStart > 0) message.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(rangeStart, null);
            var response = await _httpClient.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            if (response.RequestMessage?.RequestUri is Uri effectiveUri && !Uri.Equals(effectiveUri, current))
            {
                response.Dispose();
                throw new UnsafeRedirectException("The transport followed an unvalidated artifact redirect.");
            }

            if ((int)response.StatusCode is < 300 or > 399) return response;
            if (hop == 5 || response.Headers.Location is null)
            {
                response.Dispose();
                throw new UnsafeRedirectException("The artifact redirect chain is invalid or too long.");
            }

            var next = response.Headers.Location.IsAbsoluteUri ? response.Headers.Location : new Uri(current, response.Headers.Location);
            signedStorageHost = ValidateRedirect(current, next, signedStorageHost);
            response.Dispose();
            current = next;
        }

        throw new UnsafeRedirectException("The artifact redirect chain exceeded five hops.");
    }

    private static string? ValidateRedirect(Uri current, Uri next, string? signedStorageHost)
    {
        if (next.Scheme != Uri.UriSchemeHttps || next.Port != 443)
        {
            throw new UnsafeRedirectException("Artifact redirects must remain on standard HTTPS.");
        }

        if (signedStorageHost is not null)
        {
            if (string.Equals(signedStorageHost, next.Host, StringComparison.OrdinalIgnoreCase)) return signedStorageHost;
            throw new UnsafeRedirectException("Artifact redirects cannot leave the pinned signed storage host.");
        }

        if (HuggingFaceHosts.Contains(next.Host)) return null;
        if (HuggingFaceHosts.Contains(current.Host)) return next.Host;
        throw new UnsafeRedirectException("Artifact redirects must remain on an approved Hugging Face or signed storage host.");
    }

    private static void ValidateContentRange(HttpResponseMessage response, long start, long expectedSize)
    {
        var range = response.Content.Headers.ContentRange;
        if (range is null || range.Unit != "bytes" || range.From != start || range.To != expectedSize - 1 || range.Length != expectedSize)
        {
            throw new DownloadValidationException("The resume Content-Range does not match the requested artifact.");
        }
    }

    private static void ValidateEtag(HttpResponseMessage response, string? expected)
    {
        var actual = response.Headers.ETag?.Tag?.Trim('"');
        if (string.IsNullOrWhiteSpace(expected) || !string.Equals(expected, actual, StringComparison.Ordinal))
        {
            throw new DownloadValidationException("The artifact ETag changed before resume.");
        }
    }

    private static Exception HttpFailure(HttpStatusCode status) =>
        (int)status == 408 || (int)status == 429 || (int)status >= 500
            ? new HttpRequestException($"Transient artifact HTTP status {(int)status}.")
            : new DownloadValidationException($"Artifact HTTP status {(int)status} is not valid for this download.");

    private static bool IsTransient(Exception exception, CancellationToken cancellationToken) =>
        exception is HttpRequestException || (exception is TaskCanceledException && !cancellationToken.IsCancellationRequested);

    private static long PartialLength(string targetPath)
    {
        var partialPath = targetPath + ".partial";
        return File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0;
    }

    private static void DeletePartial(string targetPath)
    {
        File.Delete(targetPath + ".partial");
        File.Delete(targetPath + ".partial.json");
    }

    private static HttpMessageHandler CreateProductionHandler() => new HttpClientHandler { AllowAutoRedirect = false };

    public void Dispose() => _httpClient.Dispose();
}
