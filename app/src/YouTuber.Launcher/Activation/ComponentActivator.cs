using System.IO;
using YouTuber.Launcher.Uninstall;

namespace YouTuber.Launcher.Activation;

public sealed record ComponentActivationRequest(
    string Component,
    string Kind,
    string Version,
    string ArchivePath,
    string ArchiveRoot,
    string ManifestHash,
    string ArchiveSha256,
    long ExpandedSize,
    long InstallSize);

public interface IActivationRaceHooks
{
    Task BeforeRecoveryAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    Task BeforeHealthcheckAsync(string stagingDirectory, CancellationToken cancellationToken);
    Task BeforeFinalMoveAsync(string stagingDirectory, CancellationToken cancellationToken);
    Task AfterFinalMoveBeforeHealthcheckAsync(string candidateDirectory, CancellationToken cancellationToken) => Task.CompletedTask;
    Task AfterAllCandidateGuardsAcquiredAsync(string candidateDirectory, CancellationToken cancellationToken) => Task.CompletedTask;
    Task BeforePointerCommitAsync(string candidateDirectory, CancellationToken cancellationToken) => Task.CompletedTask;
}

public sealed class ComponentActivator
{
    private readonly ActiveComponentsStore _store;
    private readonly IComponentHealthChecker _healthChecker;
    private readonly IActivationRaceHooks? _hooks;
    private readonly OwnershipStore _ownership;

    public ComponentActivator(ActiveComponentsStore store, IComponentHealthChecker? healthChecker = null, IActivationRaceHooks? hooks = null, OwnershipStore? ownership = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _healthChecker = healthChecker ?? new ComponentHealthChecker();
        _hooks = hooks;
        _ownership = ownership ?? new OwnershipStore(_store.DataRoot, UninstallPreparation.InstallId);
    }

    public async Task ActivateAsync(ComponentActivationRequest request, CancellationToken cancellationToken = default)
    {
        ValidateRequest(request);
        var targetRelativePath = $"{PathRules.NormalizeRelativePath(request.Kind, "component kind")}/{request.Version}-{request.ManifestHash[..12]}";
        var targetDirectory = GetTargetDirectory(targetRelativePath);
        var installParent = Path.GetDirectoryName(targetDirectory) ?? throw new ComponentActivationException("The component target has no install parent.");
        using var ancestorGuard = ComponentPathGuard.AcquireAncestors(_store.DataRoot, installParent);
        ancestorGuard.AssertIntact();
        await using var activationLock = await _store.AcquireActivationLockAsync(cancellationToken);
        ancestorGuard.AssertIntact();
        var active = await _store.LoadAsync(cancellationToken);
        active.Components.TryGetValue(request.Component, out var previous);
        var stagingDirectory = Path.Combine(_store.ActivationRoot, $"{request.Kind}-{request.Version}-{request.ManifestHash[..12]}.staging-{Guid.NewGuid():N}");
        var transaction = new ActivationTransaction(request.Component, request.Kind, request.Version, request.ManifestHash, targetRelativePath, stagingDirectory, ActivationPhase.Downloaded, previous);
        var activated = false;

        try
        {
            await _store.SaveTransactionAsync(transaction, cancellationToken);
            using var archive = await ArchiveExtractor.OpenVerifiedArchiveAsync(request.ArchivePath, request.ArchiveSha256, cancellationToken);
            transaction = transaction with { Phase = ActivationPhase.Verified };
            await _store.SaveTransactionAsync(transaction, cancellationToken);

            ancestorGuard.AssertIntact();
            Directory.CreateDirectory(Path.GetDirectoryName(stagingDirectory)!);
            Directory.CreateDirectory(stagingDirectory);
            await ArchiveExtractor.ExtractAsync(archive, request.ArchiveRoot, stagingDirectory, request.ExpandedSize, cancellationToken: cancellationToken);
            var inventory = await ComponentVerifier.VerifyAsync(stagingDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
            if (!string.Equals(inventory.Component, request.Component, StringComparison.Ordinal) || !string.Equals(inventory.Version, request.Version, StringComparison.Ordinal))
            {
                throw new ComponentActivationException("The extracted component identity does not match the requested activation.");
            }

            transaction = transaction with { Phase = ActivationPhase.Extracted };
            await _store.SaveTransactionAsync(transaction, cancellationToken);
            if (_hooks is not null) await _hooks.BeforeHealthcheckAsync(stagingDirectory, cancellationToken);
            inventory = await ComponentVerifier.VerifyAsync(stagingDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
            if (_hooks is not null) await _hooks.BeforeFinalMoveAsync(stagingDirectory, cancellationToken);
            _ = await ComponentVerifier.VerifyAsync(stagingDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
            ancestorGuard.AssertIntact();

            if (Directory.Exists(targetDirectory))
            {
                await ComponentVerifier.VerifyAsync(targetDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
                Directory.Delete(stagingDirectory, recursive: true);
            }
            else
            {
                Directory.CreateDirectory(Path.GetDirectoryName(targetDirectory)!);
                Directory.Move(stagingDirectory, targetDirectory);
            }

            inventory = await ComponentVerifier.VerifyAsync(targetDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
            if (_hooks is not null) await RunHookAsync(() => _hooks.AfterFinalMoveBeforeHealthcheckAsync(targetDirectory, cancellationToken));
            ancestorGuard.AssertIntact();
            inventory = await ComponentVerifier.VerifyAsync(targetDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
            var executablePath = Path.Combine(targetDirectory, inventory.Entrypoint.Replace('/', Path.DirectorySeparatorChar));
            using var pathGuard = ComponentPathGuard.Acquire(targetDirectory, inventory);
            pathGuard.AssertIntact();
            if (_hooks is not null) await RunHookAsync(() => _hooks.AfterAllCandidateGuardsAcquiredAsync(targetDirectory, cancellationToken));
            inventory = await ComponentVerifier.VerifyAsync(targetDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
            pathGuard.AssertIntact();
            await _healthChecker.CheckAsync(new ComponentHealthCheckRequest(executablePath, inventory.Healthcheck, targetDirectory), cancellationToken);
            transaction = transaction with { Phase = ActivationPhase.Healthchecked };
            await _store.SaveTransactionAsync(transaction, cancellationToken);
            if (_hooks is not null) await RunHookAsync(() => _hooks.BeforePointerCommitAsync(targetDirectory, cancellationToken));
            ancestorGuard.AssertIntact();
            pathGuard.AssertIntact();
            _ = await ComponentVerifier.VerifyAsync(targetDirectory, request.ManifestHash, request.InstallSize, cancellationToken);
            pathGuard.AssertIntact();
            var replacement = new Dictionary<string, ActiveComponentPointer>(active.Components, StringComparer.Ordinal)
            {
                [request.Component] = new ActiveComponentPointer(targetRelativePath, request.ManifestHash),
            };
            await _store.SaveAsync(new ActiveComponents(replacement), cancellationToken);
            ancestorGuard.AssertIntact();
            var ownedPaths = inventory.Files
                .Select(file => Path.Combine(targetDirectory, file.Path.Replace('/', Path.DirectorySeparatorChar)))
                .Append(Path.Combine(targetDirectory, "component-manifest.json"))
                .Append(targetDirectory)
                .Append(Path.Combine(_store.DataRoot, "state", "active-components.json"))
                .ToList();
            AddIfExistingInsideDataRoot(ownedPaths, request.ArchivePath);
            AddIfExistingInsideDataRoot(ownedPaths, Path.Combine(_store.DataRoot, "state", "activation.lock"));
            AddIfExistingInsideDataRoot(ownedPaths, Path.Combine(_store.DataRoot, "state", "activation-transactions"));
            AddIfExistingInsideDataRoot(ownedPaths, _store.ActivationRoot);
            await _ownership.RegisterExistingPathsAsync(ownedPaths, cancellationToken);
            ancestorGuard.AssertIntact();
            activated = true;
            transaction = transaction with { Phase = ActivationPhase.Activated };
            await _store.SaveTransactionAsync(transaction, cancellationToken);
            transaction = transaction with { Phase = ActivationPhase.Cleaned };
            await _store.SaveTransactionAsync(transaction, cancellationToken);
            await _store.DeleteTransactionAsync(request.Component, cancellationToken);
        }
        catch
        {
            if (!activated)
            {
                await RestorePointerAsync(request.Component, previous, CancellationToken.None);
                DeleteStagingDirectory(stagingDirectory);
                await DeleteUnactivatedCandidateAsync(targetDirectory, request.ManifestHash, request.Component, CancellationToken.None);
                await _store.DeleteTransactionAsync(request.Component, CancellationToken.None);
            }

            throw;
        }
    }

    private void AddIfExistingInsideDataRoot(List<string> paths, string candidate)
    {
        var fullPath = Path.GetFullPath(candidate);
        var relative = Path.GetRelativePath(_store.DataRoot, fullPath);
        if ((File.Exists(fullPath) || Directory.Exists(fullPath)) && relative != ".." &&
            !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) && !Path.IsPathRooted(relative))
        {
            paths.Add(fullPath);
        }
    }

    public async Task RecoverAsync(CancellationToken cancellationToken = default)
    {
        using var dataRootGuard = ComponentPathGuard.AcquireAncestors(_store.DataRoot, _store.DataRoot);
        await using var activationLock = await _store.AcquireActivationLockAsync(cancellationToken);
        dataRootGuard.AssertIntact();
        if (_hooks is not null) await _hooks.BeforeRecoveryAsync(cancellationToken);
        foreach (var transaction in await _store.LoadTransactionsAsync(cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var targetDirectory = GetTargetDirectory(transaction.TargetRelativePath);
            var installParent = Path.GetDirectoryName(targetDirectory) ?? throw new ComponentActivationException("The recovery target has no install parent.");
            using var ancestorGuard = ComponentPathGuard.AcquireAncestors(_store.DataRoot, installParent);
            ancestorGuard.AssertIntact();
            if (transaction.Phase < ActivationPhase.Activated)
            {
                var active = await _store.LoadAsync(cancellationToken);
                var replacement = new Dictionary<string, ActiveComponentPointer>(active.Components, StringComparer.Ordinal);
                if (transaction.Previous is null) replacement.Remove(transaction.Component);
                else replacement[transaction.Component] = transaction.Previous;
                await _store.SaveAsync(new ActiveComponents(replacement), cancellationToken);
                ancestorGuard.AssertIntact();
                DeleteStagingDirectory(transaction.StagingDirectory);
                await DeleteUnactivatedCandidateAsync(targetDirectory, transaction.ManifestHash, transaction.Component, cancellationToken);
                ancestorGuard.AssertIntact();
            }

            await _store.DeleteTransactionAsync(transaction.Component, cancellationToken);
        }
    }

    private static void ValidateRequest(ComponentActivationRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (string.IsNullOrWhiteSpace(request.Component) || !PathRules.IsSemVer(request.Version) || !PathRules.IsSha256(request.ManifestHash) || !PathRules.IsSha256(request.ArchiveSha256) || request.ExpandedSize <= 0 || request.InstallSize <= 0)
        {
            throw new ComponentActivationException("The component activation request is invalid.");
        }

        _ = PathRules.NormalizeRelativePath(request.Kind, "component kind");
        _ = PathRules.NormalizeRelativePath(request.ArchiveRoot, "archive root");
        if (!Path.IsPathFullyQualified(request.ArchivePath)) throw new ComponentActivationException("The component archive path must be absolute.");
    }

    private async Task RestorePointerAsync(string component, ActiveComponentPointer? previous, CancellationToken cancellationToken)
    {
        var current = await _store.LoadAsync(cancellationToken);
        var replacement = new Dictionary<string, ActiveComponentPointer>(current.Components, StringComparer.Ordinal);
        if (previous is null) replacement.Remove(component);
        else replacement[component] = previous;
        await _store.SaveAsync(new ActiveComponents(replacement), cancellationToken);
    }

    private async Task DeleteUnactivatedCandidateAsync(string targetDirectory, string manifestHash, string component, CancellationToken cancellationToken)
    {
        if (!Directory.Exists(targetDirectory)) return;
        var active = await _store.LoadAsync(cancellationToken);
        if (active.Components.TryGetValue(component, out var pointer) && string.Equals(pointer.RelativePath, Path.GetRelativePath(_store.DataRoot, targetDirectory).Replace(Path.DirectorySeparatorChar, '/'), StringComparison.OrdinalIgnoreCase)) return;
        _ = await ComponentVerifier.VerifyAsync(targetDirectory, manifestHash, cancellationToken);
        DeleteCandidateDirectory(targetDirectory);
    }

    private static async Task RunHookAsync(Func<Task> action)
    {
        try { await action(); }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            throw new ComponentActivationException("The candidate component changed during activation.", exception);
        }
    }

    private string GetTargetDirectory(string targetRelativePath)
    {
        var target = Path.GetFullPath(Path.Combine(_store.DataRoot, targetRelativePath.Replace('/', Path.DirectorySeparatorChar)));
        var relative = Path.GetRelativePath(_store.DataRoot, target);
        if (Path.IsPathRooted(relative) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)) throw new ComponentActivationException("The component target escapes its data root.");
        return target;
    }

    private static void DeleteStagingDirectory(string stagingDirectory)
    {
        var name = Path.GetFileName(stagingDirectory);
        if (!name.Contains(".staging-", StringComparison.Ordinal) || !Directory.Exists(stagingDirectory)) return;
        var pending = new Stack<string>();
        pending.Push(stagingDirectory);
        while (pending.Count > 0)
        {
            foreach (var path in Directory.EnumerateFileSystemEntries(pending.Pop()))
            {
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException("Refusing to remove a staging tree containing a reparse point.");
                if (Directory.Exists(path)) pending.Push(path);
            }
        }
        if ((File.GetAttributes(stagingDirectory) & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException("Refusing to remove a reparse-point staging directory.");
        Directory.Delete(stagingDirectory, recursive: true);
    }

    private static void DeleteCandidateDirectory(string candidateDirectory)
    {
        if ((File.GetAttributes(candidateDirectory) & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException("Refusing to remove a reparse-point candidate directory.");
        var pending = new Stack<string>();
        pending.Push(candidateDirectory);
        while (pending.Count > 0)
        {
            foreach (var path in Directory.EnumerateFileSystemEntries(pending.Pop()))
            {
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException("Refusing to remove a candidate tree containing a reparse point.");
                if (Directory.Exists(path)) pending.Push(path);
            }
        }
        Directory.Delete(candidateDirectory, recursive: true);
    }

}
