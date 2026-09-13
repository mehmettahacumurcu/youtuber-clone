using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Distribution;

namespace YouTuber.Launcher.Ollama;

public sealed record OllamaPullProgress(string Status, long? Completed, long? Total);
public sealed record OllamaModelInfo(string Name, string? Digest, long? Size);

public interface IOllamaCommandRunner
{
    Task<int> RunAsync(string executable, IReadOnlyList<string> arguments, IReadOnlyDictionary<string, string> environment, CancellationToken cancellationToken = default);
}

public sealed class ProcessOllamaCommandRunner : IOllamaCommandRunner
{
    public async Task<int> RunAsync(string executable, IReadOnlyList<string> arguments, IReadOnlyDictionary<string, string> environment, CancellationToken cancellationToken = default)
    {
        var info = new ProcessStartInfo(executable) { UseShellExecute = false, CreateNoWindow = true, WindowStyle = ProcessWindowStyle.Hidden };
        foreach (var argument in arguments) info.ArgumentList.Add(argument);
        foreach (var pair in environment) info.Environment[pair.Key] = pair.Value;
        using var process = new Process { StartInfo = info };
        if (!process.Start()) throw new InvalidOperationException("Ollama could not be started.");
        try
        {
            await process.WaitForExitAsync(cancellationToken);
            return process.ExitCode;
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync(CancellationToken.None);
            throw;
        }
    }
}

public sealed class OllamaClient : IDisposable
{
    public const string SpeakerModel = "speaker-v5-a636";
    public const string VerifierModel = "qwen3:4b";
    private static readonly HashSet<string> OwnedModels = new(StringComparer.Ordinal) { SpeakerModel, VerifierModel };
    private readonly HttpClient _httpClient;
    private readonly string _binaryPath;
    private readonly Uri _endpoint;
    private readonly IOllamaCommandRunner _commandRunner;
    private readonly NoFollowPathGuard? _binaryGuard;

    private OllamaClient(HttpClient httpClient, string binaryPath, Uri endpoint, IOllamaCommandRunner? commandRunner = null, NoFollowPathGuard? binaryGuard = null)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        ArgumentException.ThrowIfNullOrWhiteSpace(binaryPath);
        OllamaInstallation.ValidateLoopbackEndpoint(endpoint);
        _httpClient = httpClient;
        _binaryPath = binaryPath;
        _endpoint = endpoint;
        _commandRunner = commandRunner ?? new ProcessOllamaCommandRunner();
        _binaryGuard = binaryGuard;
    }

    public static OllamaClient FromInstallation(OllamaInstallation installation, IOllamaCommandRunner? commandRunner = null, IWinVerifyTrust? trust = null)
    {
        ArgumentNullException.ThrowIfNull(installation);
        if (!installation.IsExisting || !installation.BinaryTrusted || string.IsNullOrWhiteSpace(installation.BinaryPath))
        {
            throw new InvalidOperationException("Ollama must be discovered and trusted before it can be executed.");
        }
        NoFollowPathGuard? guard = null;
        try
        {
            guard = NoFollowPathGuard.Open(installation.BinaryPath);
            var trustResult = (trust ?? new WindowsWinVerifyTrust()).Verify(installation.BinaryPath);
            if (!trustResult.IsTrusted || !string.Equals(trustResult.Publisher, OllamaInstallation.ExpectedPublisher, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("The Ollama binary does not match the official publisher trust policy.");
            }
            var handler = new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false };
            return new OllamaClient(new HttpClient(handler, disposeHandler: true), installation.BinaryPath, installation.ApiEndpoint, commandRunner, guard);
        }
        catch
        {
            guard?.Dispose();
            throw;
        }
    }

    internal static OllamaClient CreateForTest(HttpMessageHandler handler, string binaryPath, Uri endpoint, IOllamaCommandRunner? commandRunner = null)
    {
        if (handler is HttpClientHandler http)
        {
            http.AllowAutoRedirect = false;
            http.UseProxy = false;
        }
        return new(new HttpClient(handler, disposeHandler: true), binaryPath, endpoint, commandRunner);
    }

    public void Dispose() => _binaryGuard?.Dispose();

    public async Task EnsureHealthyAsync(CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, Api("api/version"));
        using var response = await SendAsync(request, HttpCompletionOption.ResponseContentRead, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public async Task<IReadOnlySet<string>> ListModelsAsync(CancellationToken cancellationToken = default)
        => (await ListModelDetailsAsync(cancellationToken)).Select(model => model.Name).ToHashSet(StringComparer.Ordinal);

    public async Task<IReadOnlyList<OllamaModelInfo>> ListModelDetailsAsync(CancellationToken cancellationToken = default)
        => await ListModelDetailsAsync("api/tags", cancellationToken);

    public async Task<IReadOnlySet<string>> ListResidentModelsAsync(CancellationToken cancellationToken = default)
        => await ListModelNamesAsync("api/ps", cancellationToken);

    private async Task<IReadOnlySet<string>> ListModelNamesAsync(string endpoint, CancellationToken cancellationToken)
        => (await ListModelDetailsAsync(endpoint, cancellationToken)).Select(model => model.Name).ToHashSet(StringComparer.Ordinal);

    private async Task<IReadOnlyList<OllamaModelInfo>> ListModelDetailsAsync(string endpoint, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, Api(endpoint));
        using var response = await SendAsync(request, HttpCompletionOption.ResponseContentRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        using var document = JsonDocument.Parse(await response.Content.ReadAsStreamAsync(cancellationToken));
        var details = new List<OllamaModelInfo>();
        if (document.RootElement.TryGetProperty("models", out var models))
        {
            foreach (var model in models.EnumerateArray())
            {
                if (model.TryGetProperty("name", out var name) && name.GetString() is { Length: > 0 } value)
                {
                    var digest = model.TryGetProperty("digest", out var digestValue) && digestValue.ValueKind == JsonValueKind.String
                        ? digestValue.GetString()
                        : null;
                    long? size = model.TryGetProperty("size", out var sizeValue) && sizeValue.TryGetInt64(out var parsedSize)
                        ? parsedSize
                        : null;
                    details.Add(new OllamaModelInfo(value, digest, size));
                }
            }
        }
        return details;
    }

    public async Task EnsureModelIdentityAsync(OllamaVerifierMetadata expected, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(expected);
        var matches = (await ListModelDetailsAsync(cancellationToken))
            .Where(model => string.Equals(model.Name, expected.Name, StringComparison.Ordinal))
            .ToArray();
        if (matches.Length != 1 ||
            !string.Equals(matches[0].Digest, expected.InstalledDigest, StringComparison.Ordinal) ||
            matches[0].Size != expected.InstalledSize)
        {
            throw new InvalidOperationException("The installed Ollama verifier does not match the signed digest and size.");
        }
    }

    internal static bool MatchesCanonicalModelName(string reportedName, string expectedName, bool allowImplicitLatestTag)
        => string.Equals(reportedName, expectedName, StringComparison.Ordinal) ||
            (allowImplicitLatestTag &&
             !expectedName.Contains(':', StringComparison.Ordinal) &&
             string.Equals(reportedName, expectedName + ":latest", StringComparison.Ordinal));

    public async Task PullModelAsync(string model, Func<OllamaPullProgress, CancellationToken, Task>? reportProgress = null, CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, Api("api/pull"))
        {
            Content = new StringContent(JsonSerializer.Serialize(new { name = model, stream = true }), Encoding.UTF8, "application/json"),
        };
        using var response = await SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, leaveOpen: false);
        while (await reader.ReadLineAsync(cancellationToken) is { Length: > 0 } line)
        {
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            if (reportProgress is not null)
            {
                await reportProgress(new OllamaPullProgress(
                    root.TryGetProperty("status", out var status) ? status.GetString() ?? "pulling" : "pulling",
                    root.TryGetProperty("completed", out var completed) ? completed.GetInt64() : null,
                    root.TryGetProperty("total", out var total) ? total.GetInt64() : null), cancellationToken);
            }
        }
    }

    public async Task CreateModelAsync(string name, string verifiedModelfile, IReadOnlyDictionary<string, string> environment, CancellationToken cancellationToken = default)
    {
        _binaryGuard?.EnsureUsable();
        if (!OwnedModels.Contains(name) || !File.Exists(verifiedModelfile)) throw new InvalidOperationException("Only verified YouTuber Modelfiles can create owned models.");
        await RunCreateAsync(name, verifiedModelfile, environment, cancellationToken);
    }

    internal async Task CreateModelAsync(string name, string verifiedModelfile, IReadOnlyDictionary<string, string> environment, OllamaModelRolesMetadata roles, CancellationToken cancellationToken = default)
    {
        _binaryGuard?.EnsureUsable();
        ArgumentNullException.ThrowIfNull(roles);
        if (roles.Verifier is null || roles.Speaker is null ||
            !(string.Equals(name, roles.Verifier.Name, StringComparison.Ordinal) || string.Equals(name, roles.Speaker.Name, StringComparison.Ordinal)) ||
            !File.Exists(verifiedModelfile))
            throw new InvalidOperationException("Only manifest-authorized Ollama models may be created.");
        await RunCreateAsync(name, verifiedModelfile, environment, cancellationToken);
    }

    private async Task RunCreateAsync(string name, string verifiedModelfile, IReadOnlyDictionary<string, string> environment, CancellationToken cancellationToken)
    {
        var exitCode = await _commandRunner.RunAsync(_binaryPath, ["create", name, "-f", verifiedModelfile], environment, cancellationToken);
        if (exitCode != 0) throw new InvalidOperationException($"Ollama model creation failed with exit code {exitCode}.");
    }

    public async Task DeleteOwnedModelAsync(string name, CancellationToken cancellationToken = default)
    {
        if (!OwnedModels.Contains(name)) throw new InvalidOperationException("The launcher may delete only YouTuber-owned Ollama models.");
        using var response = await PostAsync("api/delete", new { name }, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public async Task WaitForModelAbsentAsync(string model, TimeSpan timeout, TimeSpan pollInterval, CancellationToken cancellationToken = default)
    {
        var deadline = DateTimeOffset.UtcNow + timeout;
        while (DateTimeOffset.UtcNow <= deadline)
        {
            if (!(await ListResidentModelsAsync(cancellationToken)).Contains(model)) return;
            await Task.Delay(pollInterval, cancellationToken);
        }
        throw new TimeoutException($"Ollama model {model} remained resident beyond the allowed timeout.");
    }

    public async Task UnloadModelAsync(string model, CancellationToken cancellationToken = default)
    {
        using var response = await PostAsync("api/generate", new { model, prompt = string.Empty, stream = false, keep_alive = 0 }, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public async Task<string> GenerateAnswerAsync(string model, string prompt, CancellationToken cancellationToken = default)
    {
        using var response = await PostAsync("api/generate", new { model, prompt, stream = false, keep_alive = -1 }, cancellationToken);
        response.EnsureSuccessStatusCode();
        using var document = JsonDocument.Parse(await response.Content.ReadAsStreamAsync(cancellationToken));
        return document.RootElement.TryGetProperty("response", out var answer) ? answer.GetString() ?? string.Empty : string.Empty;
    }

    public async Task<string> GenerateSpeakerAfterVerifierReleasedAsync(string prompt, TimeSpan timeout, TimeSpan pollInterval, CancellationToken cancellationToken = default)
    {
        await UnloadModelAsync(VerifierModel, cancellationToken);
        await WaitForModelAbsentAsync(VerifierModel, timeout, pollInterval, cancellationToken);
        return await GenerateAnswerAsync(SpeakerModel, prompt, cancellationToken);
    }

    private async Task<HttpResponseMessage> PostAsync(string relativePath, object body, CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(body);
        using var request = new HttpRequestMessage(HttpMethod.Post, Api(relativePath)) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        return await SendAsync(request, HttpCompletionOption.ResponseContentRead, cancellationToken);
    }

    private Uri Api(string relativePath) => new(_endpoint, relativePath);

    private async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, HttpCompletionOption completionOption, CancellationToken cancellationToken)
    {
        var expected = request.RequestUri!;
        var response = await _httpClient.SendAsync(request, completionOption, cancellationToken);
        if ((int)response.StatusCode is >= 300 and < 400)
        {
            response.Dispose();
            throw new InvalidOperationException("Ollama redirects are forbidden.");
        }
        if (response.RequestMessage?.RequestUri is { } effective && effective != expected)
        {
            response.Dispose();
            throw new InvalidOperationException("Ollama response origin changed unexpectedly.");
        }
        return response;
    }
}
