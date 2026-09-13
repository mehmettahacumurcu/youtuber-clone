using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using System.IO;

namespace YouTuber.Launcher.Activation;

public sealed record ActiveComponents(IReadOnlyDictionary<string, ActiveComponentPointer> Components);
public sealed record ActiveComponentPointer(string RelativePath, string ManifestHash);

public enum ActivationPhase
{
    Downloaded,
    Verified,
    Extracted,
    Healthchecked,
    Activated,
    Cleaned,
}

public sealed record ActivationTransaction(
    string Component,
    string Kind,
    string Version,
    string ManifestHash,
    string TargetRelativePath,
    string StagingDirectory,
    ActivationPhase Phase,
    ActiveComponentPointer? Previous);

public sealed class ActiveComponentsStore
{
    private readonly string _dataRoot;
    private readonly string _stateRoot;
    private readonly string _activeComponentsFile;
    private readonly string _transactionsRoot;

    public ActiveComponentsStore(string dataRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataRoot);
        _dataRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(dataRoot));
        _stateRoot = Path.Combine(_dataRoot, "state");
        _activeComponentsFile = Path.Combine(_stateRoot, "active-components.json");
        _transactionsRoot = Path.Combine(_stateRoot, "activation-transactions");
    }

    public string DataRoot => _dataRoot;

    public string ActivationRoot => Path.Combine(_stateRoot, "activation-staging");

    public async Task<ActivationLock> AcquireActivationLockAsync(CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(_stateRoot);
        var lockPath = Path.Combine(_stateRoot, "activation.lock");
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                var handle = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.WriteThrough);
                return new ActivationLock(handle);
            }
            catch (IOException)
            {
                await Task.Delay(TimeSpan.FromMilliseconds(25), cancellationToken);
            }
        }
    }

    public async Task<ActiveComponents> MutateAsync(Func<ActiveComponents, ActiveComponents> mutation, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mutation);
        await using var activationLock = await AcquireActivationLockAsync(cancellationToken);
        var current = await LoadAsync(cancellationToken);
        var replacement = mutation(current) ?? throw new ComponentActivationException("The active-component mutation returned no value.");
        await SaveAsync(replacement, cancellationToken);
        return replacement;
    }

    public async Task<ActiveComponents> LoadAsync(CancellationToken cancellationToken = default)
    {
        if (!File.Exists(_activeComponentsFile)) return new ActiveComponents(new Dictionary<string, ActiveComponentPointer>(StringComparer.Ordinal));
        var value = await ReadAsync<ActiveComponents>(_activeComponentsFile, ActiveComponentsJsonContext.Default.ActiveComponents, cancellationToken);
        ValidateActiveComponents(value);
        return value;
    }

    public Task SaveAsync(ActiveComponents components, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(components);
        ValidateActiveComponents(components);
        return AtomicWriteAsync(_activeComponentsFile, components, ActiveComponentsJsonContext.Default.ActiveComponents, cancellationToken);
    }

    public Task SaveTransactionAsync(ActivationTransaction transaction, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transaction);
        ValidateTransaction(transaction);
        return AtomicWriteAsync(TransactionPath(transaction.Component), transaction, ActiveComponentsJsonContext.Default.ActivationTransaction, cancellationToken);
    }

    public async Task<IReadOnlyList<ActivationTransaction>> LoadTransactionsAsync(CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(_transactionsRoot)) return [];
        var transactions = new List<ActivationTransaction>();
        foreach (var path in Directory.EnumerateFiles(_transactionsRoot, "*.json", SearchOption.TopDirectoryOnly).OrderBy(Path.GetFileName, StringComparer.Ordinal))
        {
            var transaction = await ReadAsync<ActivationTransaction>(path, ActiveComponentsJsonContext.Default.ActivationTransaction, cancellationToken);
            ValidateTransaction(transaction);
            transactions.Add(transaction);
        }

        return transactions;
    }

    public Task DeleteTransactionAsync(string component, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var path = TransactionPath(component);
        if (File.Exists(path)) File.Delete(path);
        return Task.CompletedTask;
    }

    private async Task AtomicWriteAsync<T>(string path, T value, JsonTypeInfo<T> typeInfo, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporaryPath = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            await using (var stream = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await JsonSerializer.SerializeAsync(stream, value, typeInfo, cancellationToken);
                await stream.FlushAsync(cancellationToken);
                stream.Flush(flushToDisk: true);
            }

            if (File.Exists(path)) File.Replace(temporaryPath, path, destinationBackupFileName: null);
            else File.Move(temporaryPath, path);
        }
        finally
        {
            if (File.Exists(temporaryPath)) File.Delete(temporaryPath);
        }
    }

    private async Task<T> ReadAsync<T>(string path, JsonTypeInfo<T> typeInfo, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
        return await JsonSerializer.DeserializeAsync(stream, typeInfo, cancellationToken) ?? throw new ComponentActivationException("Activation state cannot be empty.");
    }

    private string TransactionPath(string component)
    {
        if (string.IsNullOrWhiteSpace(component) || component.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) throw new ComponentActivationException("The component name is invalid.");
        return Path.Combine(_transactionsRoot, component + ".json");
    }

    private static void ValidateActiveComponents(ActiveComponents components)
    {
        if (components.Components is null) throw new ComponentActivationException("Active components are missing.");
        foreach (var (component, pointer) in components.Components)
        {
            if (string.IsNullOrWhiteSpace(component) || pointer is null || !PathRules.IsSha256(pointer.ManifestHash)) throw new ComponentActivationException("An active component pointer is invalid.");
            _ = PathRules.NormalizeRelativePath(pointer.RelativePath, "active component path");
        }
    }

    private void ValidateTransaction(ActivationTransaction transaction)
    {
        if (string.IsNullOrWhiteSpace(transaction.Component) || string.IsNullOrWhiteSpace(transaction.Kind) || !PathRules.IsSemVer(transaction.Version) || !PathRules.IsSha256(transaction.ManifestHash))
        {
            throw new ComponentActivationException("The activation transaction is invalid.");
        }

        _ = PathRules.NormalizeRelativePath(transaction.TargetRelativePath, "activation target path");
        if (string.IsNullOrWhiteSpace(transaction.StagingDirectory) || !Path.IsPathFullyQualified(transaction.StagingDirectory)) throw new ComponentActivationException("The activation staging path is invalid.");
        var stagingPath = Path.GetFullPath(transaction.StagingDirectory);
        var relativeStagingPath = Path.GetRelativePath(ActivationRoot, stagingPath);
        if (Path.IsPathRooted(relativeStagingPath) || relativeStagingPath == ".." || relativeStagingPath.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) || !Path.GetFileName(stagingPath).Contains(".staging-", StringComparison.Ordinal))
        {
            throw new ComponentActivationException("The activation staging path escapes the data root.");
        }
        if (transaction.Previous is not null) ValidateActiveComponents(new ActiveComponents(new Dictionary<string, ActiveComponentPointer> { [transaction.Component] = transaction.Previous }));
    }

    public sealed class ActivationLock(FileStream handle) : IAsyncDisposable
    {
        private readonly FileStream _handle = handle;

        public ValueTask DisposeAsync() => _handle.DisposeAsync();
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
[JsonSerializable(typeof(ActiveComponents))]
[JsonSerializable(typeof(ActivationTransaction))]
internal partial class ActiveComponentsJsonContext : JsonSerializerContext { }
