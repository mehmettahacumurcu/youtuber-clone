using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using YouTuber.Launcher.Downloads;
using Xunit;

namespace YouTuber.Launcher.Tests.Downloads;

public sealed class ResumableDownloaderTests
{
    private static readonly byte[] Payload = Encoding.UTF8.GetBytes("resumable artifact payload");

    [Fact]
    public async Task DownloadAsyncStreamsAndVerifiesACompleteArtifact()
    {
        using var directory = new TemporaryDirectory();
        var handler = new FixtureHandler(_ => Response(HttpStatusCode.OK, Payload, "etag-a"));
        var downloader = new ResumableDownloader(handler, maxAttempts: 1);

        var result = await downloader.DownloadAsync(Request(directory.Target));

        Assert.Equal(DownloadState.Completed, result.State);
        Assert.Equal(Payload, await File.ReadAllBytesAsync(directory.Target));
        Assert.False(File.Exists(directory.Target + ".partial"));
        Assert.False(File.Exists(directory.Target + ".partial.json"));
    }

    [Fact]
    public async Task DownloadAsyncResumesAValidatedPartialWithRange()
    {
        using var directory = new TemporaryDirectory();
        await File.WriteAllBytesAsync(directory.Target + ".partial", Payload[..8]);
        await File.WriteAllTextAsync(directory.Target + ".partial.json", ResumeMetadata.Create(Request(directory.Target), 8, "etag-a").ToJson());
        var handler = new FixtureHandler(request =>
        {
            Assert.Equal("bytes=8-", request.Headers.Range!.ToString());
            return Response(HttpStatusCode.PartialContent, Payload[8..], "etag-a", $"bytes 8-{Payload.Length - 1}/{Payload.Length}");
        });

        var result = await new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target));

        Assert.Equal(DownloadState.Completed, result.State);
        Assert.Equal(Payload, await File.ReadAllBytesAsync(directory.Target));
    }

    [Fact]
    public async Task DownloadAsyncTruncatesBytesBeyondLastDurableMetadataCheckpointBeforeResume()
    {
        using var directory = new TemporaryDirectory();
        await File.WriteAllBytesAsync(directory.Target + ".partial", Payload[..12]);
        await File.WriteAllTextAsync(directory.Target + ".partial.json", ResumeMetadata.Create(Request(directory.Target), 8, "etag-a").ToJson());
        var handler = new FixtureHandler(request =>
        {
            Assert.Equal("bytes=8-", request.Headers.Range!.ToString());
            return Response(HttpStatusCode.PartialContent, Payload[8..], "etag-a", $"bytes 8-{Payload.Length - 1}/{Payload.Length}");
        });

        var result = await new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target));

        Assert.Equal(DownloadState.Completed, result.State);
        Assert.Equal(Payload, await File.ReadAllBytesAsync(directory.Target));
    }

    [Fact]
    public async Task DownloadAsyncRestartsOnceWhenServerIgnoresRange()
    {
        using var directory = new TemporaryDirectory();
        await File.WriteAllBytesAsync(directory.Target + ".partial", Payload[..8]);
        await File.WriteAllTextAsync(directory.Target + ".partial.json", ResumeMetadata.Create(Request(directory.Target), 8, "etag-a").ToJson());
        var requests = 0;
        var handler = new FixtureHandler(request =>
        {
            requests++;
            return requests == 1
                ? Response(HttpStatusCode.OK, Payload, "etag-a")
                : Response(HttpStatusCode.OK, Payload, "etag-a");
        });

        var result = await new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target));

        Assert.Equal(DownloadState.Completed, result.State);
        Assert.Equal(2, requests);
        Assert.Equal(Payload, await File.ReadAllBytesAsync(directory.Target));
    }

    [Fact]
    public async Task DownloadAsyncRejectsWrongContentRangeAndKeepsNoCorruptPartial()
    {
        using var directory = new TemporaryDirectory();
        await File.WriteAllBytesAsync(directory.Target + ".partial", Payload[..8]);
        await File.WriteAllTextAsync(directory.Target + ".partial.json", ResumeMetadata.Create(Request(directory.Target), 8, "etag-a").ToJson());
        var handler = new FixtureHandler(_ => Response(HttpStatusCode.PartialContent, Payload[8..], "etag-a", $"bytes 7-{Payload.Length - 1}/{Payload.Length}"));

        await Assert.ThrowsAsync<DownloadValidationException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target)));

        Assert.False(File.Exists(directory.Target + ".partial"));
        Assert.False(File.Exists(directory.Target + ".partial.json"));
    }

    [Fact]
    public async Task DownloadAsyncRejectsChangedEtagBeforeAppending()
    {
        using var directory = new TemporaryDirectory();
        await File.WriteAllBytesAsync(directory.Target + ".partial", Payload[..8]);
        await File.WriteAllTextAsync(directory.Target + ".partial.json", ResumeMetadata.Create(Request(directory.Target), 8, "etag-a").ToJson());
        var handler = new FixtureHandler(_ => Response(HttpStatusCode.PartialContent, Payload[8..], "etag-b", $"bytes 8-{Payload.Length - 1}/{Payload.Length}"));

        await Assert.ThrowsAsync<DownloadValidationException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target)));

        Assert.False(File.Exists(directory.Target + ".partial"));
    }

    [Fact]
    public async Task DownloadAsyncRejectsUnsafeRedirect()
    {
        using var directory = new TemporaryDirectory();
        var handler = new FixtureHandler(_ => Redirect("http://evil.example/file"));

        await Assert.ThrowsAsync<UnsafeRedirectException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target)));
    }

    [Fact]
    public async Task DownloadAsyncRejectsAnAutoFollowedResponseWhoseEffectiveUriChanged()
    {
        using var directory = new TemporaryDirectory();
        var handler = new AutoFollowingHandler(Response(HttpStatusCode.OK, Payload, "etag-a"), new Uri("https://evil.example/artifact"));

        await Assert.ThrowsAsync<UnsafeRedirectException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target)));
        Assert.False(File.Exists(directory.Target));
    }

    [Fact]
    public async Task DownloadAsyncRejectsSignedStorageBounceThatChangesExternalHost()
    {
        using var directory = new TemporaryDirectory();
        var responses = new Queue<HttpResponseMessage>([
            Redirect("https://signed-one.example/object"),
            Redirect("https://huggingface.co/return"),
            Redirect("https://signed-two.example/object"),
        ]);
        var handler = new FixtureHandler(_ => responses.Dequeue());

        await Assert.ThrowsAsync<UnsafeRedirectException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target)));
    }

    [Fact]
    public async Task DownloadAsyncRejectsAnAllowlistedRedirectHostOnANonStandardPort()
    {
        using var directory = new TemporaryDirectory();
        var requests = 0;
        var handler = new FixtureHandler(_ =>
        {
            requests++;
            return Redirect("https://cdn-lfs.huggingface.co:444/file");
        });

        await Assert.ThrowsAsync<UnsafeRedirectException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target)));
        Assert.Equal(1, requests);
    }

    [Fact]
    public async Task DownloadAsyncDeletesArtifactWhenHashDoesNotMatch()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Target) with { ExpectedSha256 = new string('0', 64) };
        var handler = new FixtureHandler(_ => Response(HttpStatusCode.OK, Payload, "etag-a"));

        await Assert.ThrowsAsync<DownloadValidationException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(request));

        Assert.False(File.Exists(directory.Target));
        Assert.False(File.Exists(directory.Target + ".partial"));
    }

    [Fact]
    public async Task DownloadAsyncStopsBeforeWritingBytesPastExpectedSize()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Target) with
        {
            ExpectedSize = Payload.Length - 1,
            ExpectedSha256 = Convert.ToHexString(SHA256.HashData(Payload[..^1])).ToLowerInvariant(),
        };
        var handler = new FixtureHandler(_ => Response(HttpStatusCode.OK, Payload, "etag-a"));

        await Assert.ThrowsAsync<DownloadValidationException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(request));

        Assert.False(File.Exists(directory.Target + ".partial"));
        Assert.False(File.Exists(directory.Target + ".partial.json"));
    }

    [Fact]
    public async Task DownloadAsyncRequiresAnEtagForFutureSafeResume()
    {
        using var directory = new TemporaryDirectory();
        var handler = new FixtureHandler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Payload) });

        await Assert.ThrowsAsync<DownloadValidationException>(() => new ResumableDownloader(handler, maxAttempts: 1).DownloadAsync(Request(directory.Target)));

        Assert.False(File.Exists(directory.Target + ".partial"));
    }

    [Fact]
    public async Task DownloadAsyncRetriesTransientTransportFailures()
    {
        using var directory = new TemporaryDirectory();
        var attempts = 0;
        var handler = new FixtureHandler(_ =>
        {
            attempts++;
            if (attempts == 1) throw new HttpRequestException("fixture disconnect");
            return Response(HttpStatusCode.OK, Payload, "etag-a");
        });

        var result = await new ResumableDownloader(handler, maxAttempts: 2, retryDelay: TimeSpan.Zero).DownloadAsync(Request(directory.Target));

        Assert.Equal(DownloadState.Completed, result.State);
        Assert.Equal(2, attempts);
    }

    [Fact]
    public async Task DownloadAsyncRetriesAfterTheResponseDisconnectsHalfway()
    {
        using var directory = new TemporaryDirectory();
        var attempts = 0;
        var handler = new FixtureHandler(_ =>
        {
            attempts++;
            if (attempts == 1)
            {
                var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = new DisconnectingContent(Payload[..8]) };
                response.Headers.ETag = new EntityTagHeaderValue("\"etag-a\"");
                return response;
            }

            return Response(HttpStatusCode.OK, Payload, "etag-a");
        });

        var result = await new ResumableDownloader(handler, maxAttempts: 2, retryDelay: TimeSpan.Zero).DownloadAsync(Request(directory.Target));

        Assert.Equal(DownloadState.Completed, result.State);
        Assert.Equal(2, attempts);
        Assert.Equal(Payload, await File.ReadAllBytesAsync(directory.Target));
    }

    [Fact]
    public async Task DownloadAsyncRejectsCredentialBearingSourceBeforeWritingMetadata()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Target) with
        {
            Source = new Uri("https://token@huggingface.co/owner/model/resolve/1111111111111111111111111111111111111111/file.bin?download=true"),
        };

        await Assert.ThrowsAsync<ArgumentException>(() => new ResumableDownloader(new FixtureHandler(_ => Response(HttpStatusCode.OK, Payload, "etag-a")), maxAttempts: 1).DownloadAsync(request));

        Assert.False(File.Exists(directory.Target + ".partial.json"));
    }

    [Fact]
    public async Task DownloadAsyncRejectsMutableHuggingFaceSourceBeforeWritingMetadata()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Target) with
        {
            Source = new Uri("https://huggingface.co/owner/model/resolve/main/file.bin?download=true"),
        };

        await Assert.ThrowsAsync<ArgumentException>(() => new ResumableDownloader(new FixtureHandler(_ => Response(HttpStatusCode.OK, Payload, "etag-a")), maxAttempts: 1).DownloadAsync(request));

        Assert.False(File.Exists(directory.Target + ".partial.json"));
    }

    [Fact]
    public async Task CoordinatorRetriesTheSameFailedRequestOnDemand()
    {
        using var directory = new TemporaryDirectory();
        var calls = 0;
        var coordinator = new DownloadCoordinator(new ResumableDownloader(new FixtureHandler(_ =>
        {
            calls++;
            return calls == 1 ? Response(HttpStatusCode.BadRequest, [], "etag-a") : Response(HttpStatusCode.OK, Payload, "etag-a");
        }), maxAttempts: 1));

        await Assert.ThrowsAsync<DownloadValidationException>(() => coordinator.StartAsync(Request(directory.Target)));
        var result = await coordinator.RetryAsync();

        Assert.Equal(DownloadState.Completed, result.State);
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task CoordinatorRejectsStartUntilCancelledRunHasUnwound()
    {
        using var directory = new TemporaryDirectory();
        var handler = new BlockingHandler();
        using var coordinator = new DownloadCoordinator(new ResumableDownloader(handler, maxAttempts: 1));
        var running = coordinator.StartAsync(Request(directory.Target));
        await handler.RequestStarted.Task;

        coordinator.Cancel();
        using var alreadyCancelled = new CancellationTokenSource();
        alreadyCancelled.Cancel();
        await Assert.ThrowsAsync<InvalidOperationException>(() => coordinator.StartAsync(Request(directory.Target), cancellationToken: alreadyCancelled.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => running);
        Assert.Equal(DownloadState.Cancelled, coordinator.State);
    }

    [Fact]
    public async Task CoordinatorAllowsResumeAndRetryOnlyFromTheirMatchingTerminalStates()
    {
        using var coordinator = new DownloadCoordinator(new ResumableDownloader(new FixtureHandler(_ => Response(HttpStatusCode.OK, Payload, "etag-a")), maxAttempts: 1));

        await Assert.ThrowsAsync<InvalidOperationException>(() => coordinator.ResumeAsync());
        await Assert.ThrowsAsync<InvalidOperationException>(() => coordinator.RetryAsync());
    }

    private static DownloadRequest Request(string target) => new(
        new Uri("https://huggingface.co/owner/model/resolve/1111111111111111111111111111111111111111/file.bin?download=true"),
        target,
        Payload.Length,
        Convert.ToHexString(SHA256.HashData(Payload)).ToLowerInvariant());

    private static HttpResponseMessage Response(HttpStatusCode status, byte[] bytes, string etag, string? contentRange = null)
    {
        var response = new HttpResponseMessage(status) { Content = new ByteArrayContent(bytes) };
        response.Headers.ETag = new EntityTagHeaderValue($"\"{etag}\"");
        if (contentRange is not null) response.Content.Headers.TryAddWithoutValidation("Content-Range", contentRange);
        return response;
    }

    private static HttpResponseMessage Redirect(string location) => new(HttpStatusCode.Found)
    {
        Headers = { Location = new Uri(location) },
    };

    private sealed class FixtureHandler(Func<HttpRequestMessage, HttpResponseMessage> response) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var result = response(request);
            result.RequestMessage ??= request;
            return Task.FromResult(result);
        }
    }

    private sealed class AutoFollowingHandler(HttpResponseMessage response, Uri effectiveUri) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            response.RequestMessage = new HttpRequestMessage(HttpMethod.Get, effectiveUri);
            return Task.FromResult(response);
        }
    }

    private sealed class BlockingHandler : HttpMessageHandler
    {
        public TaskCompletionSource RequestStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            RequestStarted.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            throw new InvalidOperationException("Unreachable.");
        }
    }

    private sealed class DisconnectingContent(byte[] firstBytes) : HttpContent
    {
        protected override Task SerializeToStreamAsync(Stream stream, TransportContext? context) => throw new NotSupportedException();
        protected override bool TryComputeLength(out long length) { length = 0; return false; }
        protected override Task<Stream> CreateContentReadStreamAsync() => Task.FromResult<Stream>(new DisconnectingStream(firstBytes));
    }

    private sealed class DisconnectingStream(byte[] firstBytes) : Stream
    {
        private readonly MemoryStream _inner = new(firstBytes, writable: false);
        private bool _returnedData;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => _inner.Length;
        public override long Position { get => _inner.Position; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override Task FlushAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public override int Read(byte[] buffer, int offset, int count) => throw new HttpRequestException("fixture disconnect halfway");
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (_returnedData) throw new HttpRequestException("fixture disconnect halfway");
            _returnedData = true;
            return _inner.ReadAsync(buffer, cancellationToken);
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Root = Path.Combine(Path.GetTempPath(), "YouTuberDownloadsTests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Root);
        }

        public string Root { get; }
        public string Target => Path.Combine(Root, "artifact.bin");
        public void Dispose() { if (Directory.Exists(Root)) Directory.Delete(Root, recursive: true); }
    }
}
