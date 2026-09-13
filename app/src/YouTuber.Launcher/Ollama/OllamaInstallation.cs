using System.IO;
using System.Diagnostics;
using System.Net.Http;

namespace YouTuber.Launcher.Ollama;

public sealed record OllamaInstallation(
    bool IsExisting,
    string? BinaryPath,
    Uri ApiEndpoint,
    Version? Version,
    string ModelDirectory,
    IReadOnlyDictionary<string, string> Environment,
    bool ApiReachable,
    bool BinaryTrusted)
{
    public const string ExpectedPublisher = "Ollama, Inc.";
    public static OllamaInstallation Existing(string binaryPath, Uri apiEndpoint, string version, string modelDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(binaryPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelDirectory);
        ValidateLoopbackEndpoint(apiEndpoint);
        if (!Version.TryParse(version, out var parsedVersion)) throw new ArgumentException("Ollama reported an invalid version.", nameof(version));
        return new OllamaInstallation(true, binaryPath, apiEndpoint, parsedVersion, modelDirectory, new Dictionary<string, string>(), true, true);
    }

    public static OllamaInstallation New(string dataRoot)
        => New(dataRoot, 11434);

    public static OllamaInstallation New(string dataRoot, int port)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataRoot);
        var modelDirectory = Path.Combine(dataRoot, "models", "ollama");
        return new OllamaInstallation(false, null, new Uri($"http://127.0.0.1:{port}"), null, modelDirectory,
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) { ["OLLAMA_MODELS"] = modelDirectory, ["OLLAMA_HOST"] = $"127.0.0.1:{port}" }, false, false);
    }

    public static void ValidateLoopbackEndpoint(Uri endpoint)
    {
        ArgumentNullException.ThrowIfNull(endpoint);
        if (!endpoint.IsAbsoluteUri || endpoint.Scheme != Uri.UriSchemeHttp || endpoint.Host != "127.0.0.1" || endpoint.Port is <= 0 or > 65535 || !string.IsNullOrEmpty(endpoint.UserInfo) || !string.IsNullOrEmpty(endpoint.Query) || !string.IsNullOrEmpty(endpoint.Fragment))
        {
            throw new ArgumentException("Ollama must use an explicit HTTP 127.0.0.1 endpoint.", nameof(endpoint));
        }
    }
}

public interface IOllamaInstallationProbe
{
    Task<OllamaInstallation?> DetectAsync(CancellationToken cancellationToken = default);
}

public sealed class OllamaInstallationDetector : IOllamaInstallationProbe
{
    private readonly Func<string?> _findBinary;
    private readonly Func<string, string?> _environment;
    private readonly Func<string, CancellationToken, Task<string?>> _readVersion;
    private readonly Func<Uri, CancellationToken, Task<bool>> _apiHealth;
    private readonly Func<string, bool> _isTrusted;

    public OllamaInstallationDetector(
        Func<string?>? findBinary = null,
        Func<string, string?>? environment = null,
        Func<string, CancellationToken, Task<string?>>? readVersion = null,
        Func<Uri, CancellationToken, Task<bool>>? apiHealth = null,
        Func<string, bool>? isTrusted = null)
    {
        _findBinary = findBinary ?? FindBinaryOnPath;
        _environment = environment ?? Environment.GetEnvironmentVariable;
        _readVersion = readVersion ?? ReadVersionAsync;
        _apiHealth = apiHealth ?? CheckApiHealthAsync;
        _isTrusted = isTrusted ?? IsOfficialOllamaBinary;
    }

    public async Task<OllamaInstallation?> DetectAsync(CancellationToken cancellationToken = default)
    {
        var binary = _findBinary();
        if (string.IsNullOrWhiteSpace(binary)) return null;
        FileStream binaryGuard;
        try
        {
            binaryGuard = new FileStream(binary, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            throw new InvalidOperationException("A discovered Ollama binary could not be opened safely.", exception);
        }
        using (binaryGuard)
        {
        if (!_isTrusted(binary)) throw new InvalidOperationException("A discovered Ollama binary failed publisher trust verification.");
        var endpoint = ParseHost(_environment("OLLAMA_HOST"));
        var versionOutput = await _readVersion(binary, cancellationToken);
        if (string.IsNullOrWhiteSpace(versionOutput) || !TryExtractVersion(versionOutput, out var version))
            throw new InvalidOperationException("A discovered Ollama binary did not report a valid version.");
        var modelDirectory = _environment("OLLAMA_MODELS");
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            modelDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        }

        return new OllamaInstallation(true, binary, endpoint, version, modelDirectory, new Dictionary<string, string>(), await _apiHealth(endpoint, cancellationToken), true);
        }
    }

    private static Uri ParseHost(string? configuredHost)
    {
        var host = string.IsNullOrWhiteSpace(configuredHost) ? "127.0.0.1:11434" : configuredHost;
        if (!host.Contains("://", StringComparison.Ordinal)) host = $"http://{host}";
        if (!Uri.TryCreate(host, UriKind.Absolute, out var endpoint)) throw new InvalidOperationException("OLLAMA_HOST is invalid.");
        OllamaInstallation.ValidateLoopbackEndpoint(endpoint);
        return endpoint;
    }

    private static bool TryExtractVersion(string output, out Version version)
    {
        var candidate = output.Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries).LastOrDefault(value => Version.TryParse(value.TrimStart('v'), out _));
        if (candidate is not null && Version.TryParse(candidate.TrimStart('v'), out var parsed))
        {
            version = parsed;
            return true;
        }

        version = new Version(0, 0);
        return false;
    }

    private static string? FindBinaryOnPath()
    {
        var path = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrWhiteSpace(path)) return null;
        return path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(directory => Path.Combine(directory, "ollama.exe")).FirstOrDefault(File.Exists);
    }

    private static bool IsOfficialOllamaBinary(string path)
    {
        var trust = new YouTuber.Launcher.Prerequisites.WindowsWinVerifyTrust().Verify(path);
        return trust.IsTrusted && string.Equals(trust.Publisher, OllamaInstallation.ExpectedPublisher, StringComparison.OrdinalIgnoreCase);
    }

    private static async Task<string?> ReadVersionAsync(string binary, CancellationToken cancellationToken)
    {
        var info = new ProcessStartInfo(binary) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        info.ArgumentList.Add("--version");
        using var process = new Process { StartInfo = info };
        if (!process.Start()) return null;
        var output = process.StandardOutput.ReadToEndAsync(cancellationToken);
        await process.WaitForExitAsync(cancellationToken);
        return process.ExitCode == 0 ? await output : null;
    }

    private static async Task<bool> CheckApiHealthAsync(Uri endpoint, CancellationToken cancellationToken)
    {
        using var client = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false })
        {
            Timeout = TimeSpan.FromSeconds(3),
        };
        try
        {
            using var response = await client.GetAsync(new Uri(endpoint, "api/version"), cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (HttpRequestException) { return false; }
        catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested) { return false; }
    }
}
