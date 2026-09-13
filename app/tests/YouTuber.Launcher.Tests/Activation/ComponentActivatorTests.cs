using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using YouTuber.Launcher.Activation;
using YouTuber.Launcher.Uninstall;
using Xunit;

namespace YouTuber.Launcher.Tests.Activation;

public sealed class ComponentActivatorTests
{
    [Fact]
    public async Task ActivateAsyncKeepsTheExistingActiveVersionWhenTheCandidateHealthcheckFails()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive();
        var activator = fixture.CreateActivator(new FailingHealthChecker());

        await Assert.ThrowsAsync<ComponentHealthCheckException>(() => activator.ActivateAsync(fixture.Request));

        var active = await fixture.Store.LoadAsync();
        Assert.Equal(fixture.V1RelativePath, active.Components[fixture.Component].RelativePath);
        Assert.True(File.Exists(Path.Combine(fixture.V1Target, "worker.exe")));
        Assert.False(Directory.Exists(fixture.V2Target));
        Assert.Empty(Directory.EnumerateFiles(Path.Combine(fixture.DataRoot, "state", "activation-transactions"), "*.json"));
    }

    [Fact]
    public async Task ActivateAsyncAtomicallySwitchesThePointerOnlyAfterVerificationAndHealthcheckWhileRetainingV1()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive();
        var checker = new RecordingHealthChecker();
        var activator = fixture.CreateActivator(checker);

        await activator.ActivateAsync(fixture.Request);

        var active = await fixture.Store.LoadAsync();
        Assert.Equal(fixture.V2RelativePath, active.Components[fixture.Component].RelativePath);
        Assert.Equal(fixture.ManifestHash, active.Components[fixture.Component].ManifestHash);
        Assert.Equal(["worker.exe", "--healthcheck"], checker.ArgumentVector);
        Assert.True(File.Exists(Path.Combine(fixture.V1Target, "worker.exe")));
        Assert.True(File.Exists(Path.Combine(fixture.V2Target, "worker.exe")));
        Assert.Empty(Directory.EnumerateFiles(Path.Combine(fixture.DataRoot, "state"), "*.tmp", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task ActivateAsyncRejectsAnExpandedTotalThatDoesNotMatchTheSignedReleaseSize()
    {
        using var fixture = new ActivationFixture();
        fixture.WriteV2Archive();
        var health = new RecordingHealthChecker();

        await Assert.ThrowsAnyAsync<ComponentActivationException>(() => fixture.CreateActivator(health).ActivateAsync(
            fixture.Request with { ExpandedSize = fixture.Request.ExpandedSize + 1 }));

        Assert.Empty(health.ArgumentVector);
        Assert.False(Directory.Exists(fixture.V2Target));
    }

    [Fact]
    public async Task ActivateAsyncRejectsAnInventoryTotalThatDoesNotMatchTheSignedInstallSize()
    {
        using var fixture = new ActivationFixture();
        fixture.WriteV2Archive();
        var health = new RecordingHealthChecker();

        await Assert.ThrowsAsync<ComponentActivationException>(() => fixture.CreateActivator(health).ActivateAsync(
            fixture.Request with { InstallSize = fixture.Request.InstallSize + 1 }));

        Assert.Empty(health.ArgumentVector);
        Assert.False(Directory.Exists(fixture.V2Target));
    }

    [Fact]
    public async Task ActivateAsyncRegistersVerifiedComponentAndDownloadedArchiveForOwnedDataRemoval()
    {
        using var fixture = new ActivationFixture();
        fixture.WriteV2Archive(includeDependency: true);

        await fixture.CreateActivator(new RecordingHealthChecker()).ActivateAsync(fixture.Request);

        var plan = new OwnedDataInventory().Plan(
            fixture.DataRoot,
            UninstallPreparation.InstallId,
            UninstallChoice.AppAndOwnedData);
        Assert.Contains(fixture.Archive, plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.Contains(Path.Combine(fixture.V2Target, "worker.exe"), plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.Contains(Path.Combine(fixture.V2Target, "lib", "dependency.dll"), plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.Contains(Path.Combine(fixture.V2Target, "component-manifest.json"), plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.Contains(Path.Combine(fixture.DataRoot, "state", "active-components.json"), plan.Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RecoverAsyncRollsBackAnIncompleteTransactionAndTheNextActivationRetrySucceeds()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        var staging = Path.Combine(fixture.Store.ActivationRoot, "runtime-2.0.0-staging.staging-deadbeef");
        Directory.CreateDirectory(staging);
        await fixture.Store.SaveTransactionAsync(new ActivationTransaction(
            fixture.Component,
            "runtime",
            "2.0.0",
            fixture.ManifestHash,
            fixture.V2RelativePath,
            staging,
            ActivationPhase.Extracted,
            new ActiveComponentPointer(fixture.V1RelativePath, fixture.V1Hash)));

        var activator = fixture.CreateActivator(new RecordingHealthChecker());
        await activator.RecoverAsync();

        var active = await fixture.Store.LoadAsync();
        Assert.Equal(fixture.V1RelativePath, active.Components[fixture.Component].RelativePath);
        Assert.True(File.Exists(Path.Combine(fixture.V1Target, "worker.exe")));
        Assert.False(Directory.Exists(staging));

        fixture.WriteV2Archive();
        await activator.ActivateAsync(fixture.Request);

        active = await fixture.Store.LoadAsync();
        Assert.Equal(fixture.V2RelativePath, active.Components[fixture.Component].RelativePath);
        Assert.Empty(await fixture.Store.LoadTransactionsAsync());
    }

    [Fact]
    public async Task ActivateAsyncRejectsAnUninventoriedEmptyDirectory()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive(includeUnexpectedDirectory: true);

        await Assert.ThrowsAsync<ComponentActivationException>(() => fixture.CreateActivator(new RecordingHealthChecker()).ActivateAsync(fixture.Request));

        var active = await fixture.Store.LoadAsync();
        Assert.Equal(fixture.V1RelativePath, active.Components[fixture.Component].RelativePath);
        Assert.False(Directory.Exists(fixture.V2Target));
    }

    [Fact]
    public async Task SaveTransactionAsyncRejectsAStagingPathOutsideTheDataRoot()
    {
        using var fixture = new ActivationFixture();
        var outsideStaging = Path.Combine(Path.GetTempPath(), "outside.staging-" + Guid.NewGuid().ToString("N"));

        await Assert.ThrowsAsync<ComponentActivationException>(() => fixture.Store.SaveTransactionAsync(new ActivationTransaction(
            fixture.Component,
            "runtime",
            "2.0.0",
            fixture.ManifestHash,
            fixture.V2RelativePath,
            outsideStaging,
            ActivationPhase.Extracted,
            null)));
    }

    [Fact]
    public async Task ActivateAsyncRollbackPreservesPointersWrittenByAnotherComponentTransaction()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive();
        var healthChecker = new ConcurrentUpdateThenFailHealthChecker(fixture.Store, fixture.Component);

        await Assert.ThrowsAsync<ComponentHealthCheckException>(() => fixture.CreateActivator(healthChecker).ActivateAsync(fixture.Request));

        var active = await fixture.Store.LoadAsync();
        Assert.Equal(fixture.V1RelativePath, active.Components[fixture.Component].RelativePath);
        Assert.Equal("models/1.0.0-cccccccccccc", active.Components["llm"].RelativePath);
    }

    [Fact]
    public async Task MutateAsyncAcrossTwoStoreInstancesPreservesConcurrentPointersForDifferentComponents()
    {
        using var fixture = new ActivationFixture();
        var otherStore = new ActiveComponentsStore(fixture.DataRoot);

        await Task.WhenAll(
            fixture.Store.MutateAsync(active => WithPointer(active, "studio_runtime", "runtime/2.0.0-aaaaaaaaaaaa", new string('a', 64))),
            otherStore.MutateAsync(active => WithPointer(active, "voice_runtime", "runtime/2.0.0-bbbbbbbbbbbb", new string('b', 64))));

        var active = await fixture.Store.LoadAsync();
        Assert.Equal("runtime/2.0.0-aaaaaaaaaaaa", active.Components["studio_runtime"].RelativePath);
        Assert.Equal("runtime/2.0.0-bbbbbbbbbbbb", active.Components["voice_runtime"].RelativePath);
    }

    [Fact]
    public async Task ActivateAsyncRejectsAStagingDirectorySwappedToAJunctionBeforeHealthcheck()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive();
        var health = new RecordingHealthChecker();
        var activator = fixture.CreateActivator(health, new JunctionSwapHooks(fixture.OutsideDirectory, swapBeforeHealthcheck: true));

        await Assert.ThrowsAsync<ComponentActivationException>(() => activator.ActivateAsync(fixture.Request));

        Assert.Empty(health.ArgumentVector);
        Assert.False(File.Exists(Path.Combine(fixture.OutsideDirectory, "worker.exe")));
        Assert.Equal(fixture.V1RelativePath, (await fixture.Store.LoadAsync()).Components[fixture.Component].RelativePath);
    }

    [Fact]
    public async Task ActivateAsyncRejectsAStagingDirectorySwappedToAJunctionBeforeFinalMove()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive();
        var health = new RecordingHealthChecker();
        var activator = fixture.CreateActivator(health, new JunctionSwapHooks(fixture.OutsideDirectory, swapBeforeHealthcheck: false));

        await Assert.ThrowsAsync<ComponentActivationException>(() => activator.ActivateAsync(fixture.Request));

        Assert.Empty(health.ArgumentVector);
        Assert.False(File.Exists(Path.Combine(fixture.OutsideDirectory, "worker.exe")));
        Assert.Equal(fixture.V1RelativePath, (await fixture.Store.LoadAsync()).Components[fixture.Component].RelativePath);
    }

    [Fact]
    public async Task ActivateAsyncRejectsAFinalCandidateSwappedToAJunctionBeforeHealthcheck()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive();
        var health = new RecordingHealthChecker();
        var activator = fixture.CreateActivator(health, new PostMoveJunctionSwapHooks(fixture.OutsideDirectory, beforePointerCommit: false));

        await Assert.ThrowsAsync<ComponentActivationException>(() => activator.ActivateAsync(fixture.Request));

        Assert.Empty(health.ArgumentVector);
        Assert.False(File.Exists(Path.Combine(fixture.OutsideDirectory, "worker.exe")));
        Assert.Equal(fixture.V1RelativePath, (await fixture.Store.LoadAsync()).Components[fixture.Component].RelativePath);
    }

    [Fact]
    public async Task ActivateAsyncRejectsAFinalCandidateSwappedToAJunctionBeforePointerCommit()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive();
        var health = new RecordingHealthChecker();
        var activator = fixture.CreateActivator(health, new PostMoveJunctionSwapHooks(fixture.OutsideDirectory, beforePointerCommit: true));

        await Assert.ThrowsAsync<ComponentActivationException>(() => activator.ActivateAsync(fixture.Request));

        Assert.Equal(["worker.exe", "--healthcheck"], health.ArgumentVector);
        Assert.False(File.Exists(Path.Combine(fixture.OutsideDirectory, "worker.exe")));
        Assert.Equal(fixture.V1RelativePath, (await fixture.Store.LoadAsync()).Components[fixture.Component].RelativePath);
    }

    [Fact]
    public async Task ActivateAsyncRejectsAPreexistingInstallParentJunctionBeforeExtraction()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var fixture = new ActivationFixture();
        fixture.WriteV2Archive();
        Directory.CreateDirectory(fixture.OutsideDirectory);
        JunctionSwapHooks.CreateJunction(
            Path.Combine(fixture.DataRoot, "runtime"), fixture.OutsideDirectory);
        var health = new RecordingHealthChecker();

        await Assert.ThrowsAsync<ComponentActivationException>(() =>
            fixture.CreateActivator(health).ActivateAsync(fixture.Request));

        Assert.Empty(health.ArgumentVector);
        Assert.False(File.Exists(Path.Combine(fixture.OutsideDirectory, "worker.exe")));
    }

    [Fact]
    public async Task ActivateAsyncRetainsInstallParentIdentityThroughTheFinalMove()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var fixture = new ActivationFixture();
        fixture.WriteV2Archive();
        var health = new RecordingHealthChecker();
        var hooks = new InstallParentReplacementHooks(
            Path.Combine(fixture.DataRoot, "runtime"), fixture.OutsideDirectory);

        await Assert.ThrowsAsync<ComponentActivationException>(() =>
            fixture.CreateActivator(health, hooks).ActivateAsync(fixture.Request));

        Assert.True(hooks.ReplacementBlocked);
        Assert.Empty(health.ArgumentVector);
        Assert.False(File.Exists(Path.Combine(fixture.OutsideDirectory, "worker.exe")));
    }

    [Fact]
    public async Task RecoverAsyncRemovesAMovedButUnactivatedCandidate()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2CandidateDirectory();
        var staging = Path.Combine(fixture.Store.ActivationRoot, "runtime-2.0.0-staging.staging-recovery");
        await fixture.Store.SaveTransactionAsync(new ActivationTransaction(
            fixture.Component,
            "runtime",
            "2.0.0",
            fixture.ManifestHash,
            fixture.V2RelativePath,
            staging,
            ActivationPhase.Extracted,
            new ActiveComponentPointer(fixture.V1RelativePath, fixture.V1Hash)));

        await fixture.CreateActivator(new RecordingHealthChecker()).RecoverAsync();

        Assert.False(Directory.Exists(fixture.V2Target));
        Assert.Equal(fixture.V1RelativePath, (await fixture.Store.LoadAsync()).Components[fixture.Component].RelativePath);
    }

    [Fact]
    public async Task RecoverAsyncRejectsAnInstallParentJunctionWithoutDeletingItsTarget()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var fixture = new ActivationFixture();
        Directory.CreateDirectory(fixture.OutsideDirectory);
        JunctionSwapHooks.CreateJunction(
            Path.Combine(fixture.DataRoot, "runtime"), fixture.OutsideDirectory);
        fixture.WriteV2CandidateDirectory();
        var staging = Path.Combine(
            fixture.Store.ActivationRoot,
            "runtime-2.0.0-staging.staging-junction-recovery");
        await fixture.Store.SaveTransactionAsync(new ActivationTransaction(
            fixture.Component,
            "runtime",
            "2.0.0",
            fixture.ManifestHash,
            fixture.V2RelativePath,
            staging,
            ActivationPhase.Extracted,
            null));
        var externalWorker = Path.Combine(
            fixture.OutsideDirectory,
            $"2.0.0-{fixture.ManifestHash[..12]}",
            "worker.exe");

        await Assert.ThrowsAsync<ComponentActivationException>(() =>
            fixture.CreateActivator(new RecordingHealthChecker()).RecoverAsync());

        Assert.True(File.Exists(externalWorker));
    }

    [Fact]
    public async Task ActivateAsyncBlocksDependencyAndAncestorDirectoryReplacementAfterAllGuardsAreAcquired()
    {
        using var fixture = new ActivationFixture();
        await fixture.SetActiveV1Async();
        fixture.WriteV2Archive(includeDependency: true);
        var health = new RecordingHealthChecker();
        var hooks = new DependencyReplacementHooks();

        await Assert.ThrowsAsync<ComponentActivationException>(() => fixture.CreateActivator(health, hooks).ActivateAsync(fixture.Request));

        Assert.True(hooks.FileReplacementBlocked);
        Assert.True(hooks.DirectoryRenameBlocked);
        Assert.Empty(health.ArgumentVector);
        Assert.Equal(fixture.V1RelativePath, (await fixture.Store.LoadAsync()).Components[fixture.Component].RelativePath);
    }

    [Theory]
    [InlineData("COM¹.exe")]
    [InlineData("COM²")]
    [InlineData("COM³.txt")]
    [InlineData("LPT¹.exe")]
    [InlineData("LPT²")]
    [InlineData("LPT³.txt")]
    [InlineData("CONIN$.txt")]
    [InlineData("CONOUT$.exe")]
    public void ComponentInventoryRejectsCanonicalWindowsDeviceNames(string path)
    {
        var json = $$"""{"schema":"youtuber.component.v1","component":"studio_runtime","version":"2.0.0","compatibility":{"runtime_api":1},"entrypoint":"{{path}}","healthcheck":["--healthcheck"],"files":[{"path":"{{path}}","size":2,"sha256":"fb04dcb6970e4c3d1873de51fd5a50d7bb46b3383113602665c350ec40b5f990"}]}""";

        Assert.ThrowsAny<ComponentActivationException>(() => ComponentInventory.Parse(json));
    }

    [Fact]
    public void ComponentInventoryRejectsAnOverflowingFileSizeTotal()
    {
        var json = $$"""{"schema":"youtuber.component.v1","component":"studio_runtime","version":"2.0.0","compatibility":{"runtime_api":1},"entrypoint":"worker.exe","healthcheck":["--healthcheck"],"files":[{"path":"worker.exe","size":{{long.MaxValue}},"sha256":"{{new string('a', 64)}}"},{"path":"second.bin","size":1,"sha256":"{{new string('b', 64)}}"}]}""";

        Assert.Throws<ComponentActivationException>(() => ComponentInventory.Parse(json));
    }

    private static ActiveComponents WithPointer(ActiveComponents active, string component, string relativePath, string hash)
    {
        var replacement = new Dictionary<string, ActiveComponentPointer>(active.Components, StringComparer.Ordinal)
        {
            [component] = new(relativePath, hash),
        };
        return new ActiveComponents(replacement);
    }

    private sealed class ActivationFixture : IDisposable
    {
        private const string WorkerManifestJson = "{\"schema\":\"youtuber.component.v1\",\"component\":\"studio_runtime\",\"version\":\"2.0.0\",\"compatibility\":{\"runtime_api\":1},\"entrypoint\":\"worker.exe\",\"healthcheck\":[\"--healthcheck\"],\"files\":[{\"path\":\"worker.exe\",\"size\":2,\"sha256\":\"fb04dcb6970e4c3d1873de51fd5a50d7bb46b3383113602665c350ec40b5f990\"}]}";
        private const string DependencyManifestJson = "{\"schema\":\"youtuber.component.v1\",\"component\":\"studio_runtime\",\"version\":\"2.0.0\",\"compatibility\":{\"runtime_api\":1},\"entrypoint\":\"worker.exe\",\"healthcheck\":[\"--healthcheck\"],\"files\":[{\"path\":\"worker.exe\",\"size\":2,\"sha256\":\"fb04dcb6970e4c3d1873de51fd5a50d7bb46b3383113602665c350ec40b5f990\"},{\"path\":\"lib/dependency.dll\",\"size\":3,\"sha256\":\"8ce3e71ef8635d2bf27913bb680d7f88ad0238ea42c779c4b30f4e587d07da8e\"}]}";
        private string _manifestJson = WorkerManifestJson;

        public ActivationFixture()
        {
            DataRoot = Path.Combine(Path.GetTempPath(), "YouTuberActivationTests", Guid.NewGuid().ToString("N"));
            Archive = Path.Combine(DataRoot, "download.zip");
            OutsideDirectory = Path.Combine(DataRoot, "outside");
            Component = "studio_runtime";
            ManifestHash = HashManifest(_manifestJson);
            Store = new ActiveComponentsStore(DataRoot);
            Request = new ComponentActivationRequest(Component, "runtime", "2.0.0", Archive, "runtime", ManifestHash, string.Empty, 1, 1);
        }

        public string DataRoot { get; }
        public string Archive { get; }
        public string OutsideDirectory { get; }
        public string Component { get; }
        public string ManifestHash { get; private set; }
        public string V1Hash { get; } = new string('a', 64);
        public string V1RelativePath => $"runtime/1.0.0-{V1Hash[..12]}";
        public string V1Target => Path.Combine(DataRoot, "runtime", $"1.0.0-{V1Hash[..12]}");
        public string V2RelativePath => $"runtime/2.0.0-{ManifestHash[..12]}";
        public string V2Target => Path.Combine(DataRoot, "runtime", $"2.0.0-{ManifestHash[..12]}");
        public ActiveComponentsStore Store { get; }
        public ComponentActivationRequest Request { get; private set; }

        public async Task SetActiveV1Async()
        {
            var v1 = V1Target;
            Directory.CreateDirectory(v1);
            await File.WriteAllTextAsync(Path.Combine(v1, "worker.exe"), "v1");
            await Store.SaveAsync(new ActiveComponents(new Dictionary<string, ActiveComponentPointer>
            {
                [Component] = new(V1RelativePath, V1Hash),
            }));
        }

        public void WriteV2Archive(bool includeUnexpectedDirectory = false, bool includeDependency = false)
        {
            _manifestJson = includeDependency ? DependencyManifestJson : WorkerManifestJson;
            ManifestHash = HashManifest(_manifestJson);
            Directory.CreateDirectory(DataRoot);
            using (var stream = new FileStream(Archive, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var archive = new ZipArchive(stream, ZipArchiveMode.Create))
            {
                WriteEntry(archive, "runtime/worker.exe", "v2");
                WriteEntry(archive, "runtime/component-manifest.json", _manifestJson);
                if (includeUnexpectedDirectory) archive.CreateEntry("runtime/unexpected/");
                if (includeDependency) WriteEntry(archive, "runtime/lib/dependency.dll", "dep");
            }

            using var written = ZipFile.OpenRead(Archive);
            var expandedSize = written.Entries.Where(entry => !entry.FullName.EndsWith("/", StringComparison.Ordinal))
                .Aggregate(0L, (total, entry) => checked(total + entry.Length));
            Request = Request with
            {
                ManifestHash = ManifestHash,
                ArchiveSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Archive))).ToLowerInvariant(),
                ExpandedSize = expandedSize,
                InstallSize = includeDependency ? 5 : 2,
            };
        }

        public void WriteV2CandidateDirectory()
        {
            Directory.CreateDirectory(V2Target);
            File.WriteAllText(Path.Combine(V2Target, "worker.exe"), "v2", new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            File.WriteAllText(Path.Combine(V2Target, "component-manifest.json"), _manifestJson, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }

        public ComponentActivator CreateActivator(IComponentHealthChecker healthChecker, IActivationRaceHooks? hooks = null) => new(Store, healthChecker, hooks);

        public void Dispose()
        {
            if (!Directory.Exists(DataRoot)) return;
            var pending = new Stack<string>();
            pending.Push(DataRoot);
            while (pending.Count > 0)
            {
                foreach (var path in Directory.EnumerateFileSystemEntries(pending.Pop()))
                {
                    if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                    {
                        Directory.Delete(path);
                    }
                    else if (Directory.Exists(path))
                    {
                        pending.Push(path);
                    }
                }
            }
            Directory.Delete(DataRoot, recursive: true);
        }

        private static void WriteEntry(ZipArchive archive, string path, string content)
        {
            var entry = archive.CreateEntry(path);
            using var writer = new StreamWriter(entry.Open(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: false), leaveOpen: false);
            writer.Write(content);
        }

        private static string HashManifest(string manifest) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(manifest))).ToLowerInvariant();
    }

    private sealed class FailingHealthChecker : IComponentHealthChecker
    {
        public Task CheckAsync(ComponentHealthCheckRequest request, CancellationToken cancellationToken = default) =>
            Task.FromException(new ComponentHealthCheckException("fixture healthcheck failure"));
    }

    private sealed class RecordingHealthChecker : IComponentHealthChecker
    {
        public IReadOnlyList<string> ArgumentVector { get; private set; } = [];

        public Task CheckAsync(ComponentHealthCheckRequest request, CancellationToken cancellationToken = default)
        {
            ArgumentVector = [Path.GetFileName(request.ExecutablePath), .. request.Arguments];
            return Task.CompletedTask;
        }
    }

    private sealed class ConcurrentUpdateThenFailHealthChecker(ActiveComponentsStore store, string component) : IComponentHealthChecker
    {
        public async Task CheckAsync(ComponentHealthCheckRequest request, CancellationToken cancellationToken = default)
        {
            var active = await store.LoadAsync(cancellationToken);
            var replacement = new Dictionary<string, ActiveComponentPointer>(active.Components, StringComparer.Ordinal)
            {
                ["llm"] = new("models/1.0.0-cccccccccccc", new string('c', 64)),
            };
            await store.SaveAsync(new ActiveComponents(replacement), cancellationToken);
            throw new ComponentHealthCheckException($"{component} health-check failure");
        }
    }

    private sealed class JunctionSwapHooks(string outsideDirectory, bool swapBeforeHealthcheck) : IActivationRaceHooks
    {
        public Task BeforeHealthcheckAsync(string stagingDirectory, CancellationToken cancellationToken)
        {
            if (swapBeforeHealthcheck) Swap(stagingDirectory, outsideDirectory);
            return Task.CompletedTask;
        }

        public Task BeforeFinalMoveAsync(string stagingDirectory, CancellationToken cancellationToken)
        {
            if (!swapBeforeHealthcheck) Swap(stagingDirectory, outsideDirectory);
            return Task.CompletedTask;
        }

        public static void Swap(string stagingDirectory, string outsideDirectory)
        {
            Directory.CreateDirectory(outsideDirectory);
            Directory.Delete(stagingDirectory, recursive: true);
            CreateJunction(stagingDirectory, outsideDirectory);
        }

        public static void CreateJunction(string junction, string outsideDirectory)
        {
            Directory.CreateDirectory(outsideDirectory);
            var process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "cmd.exe",
                UseShellExecute = false,
                CreateNoWindow = true,
                ArgumentList = { "/d", "/c", "mklink", "/J", junction, outsideDirectory },
            })!;
            process.WaitForExit();
            if (process.ExitCode != 0) throw new InvalidOperationException("The fixture junction could not be created.");
        }
    }

    private sealed class InstallParentReplacementHooks(
        string installParent,
        string outsideDirectory) : IActivationRaceHooks
    {
        public bool ReplacementBlocked { get; private set; }

        public Task BeforeHealthcheckAsync(string stagingDirectory, CancellationToken cancellationToken) =>
            Task.CompletedTask;

        public Task BeforeFinalMoveAsync(string stagingDirectory, CancellationToken cancellationToken)
        {
            try
            {
                Directory.Delete(installParent);
                JunctionSwapHooks.CreateJunction(installParent, outsideDirectory);
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                ReplacementBlocked = true;
                throw new ComponentActivationException("The guarded install parent could not be replaced.");
            }
            throw new ComponentActivationException("The install parent was replaced while activation was in flight.");
        }
    }

    private sealed class PostMoveJunctionSwapHooks(string outsideDirectory, bool beforePointerCommit) : IActivationRaceHooks
    {
        public Task BeforeHealthcheckAsync(string stagingDirectory, CancellationToken cancellationToken) => Task.CompletedTask;
        public Task BeforeFinalMoveAsync(string stagingDirectory, CancellationToken cancellationToken) => Task.CompletedTask;

        public Task AfterFinalMoveBeforeHealthcheckAsync(string candidateDirectory, CancellationToken cancellationToken)
        {
            if (!beforePointerCommit) JunctionSwapHooks.Swap(candidateDirectory, outsideDirectory);
            return Task.CompletedTask;
        }

        public Task BeforePointerCommitAsync(string candidateDirectory, CancellationToken cancellationToken)
        {
            if (beforePointerCommit) JunctionSwapHooks.Swap(candidateDirectory, outsideDirectory);
            return Task.CompletedTask;
        }
    }

    private sealed class DependencyReplacementHooks : IActivationRaceHooks
    {
        public bool FileReplacementBlocked { get; private set; }
        public bool DirectoryRenameBlocked { get; private set; }

        public Task BeforeHealthcheckAsync(string stagingDirectory, CancellationToken cancellationToken) => Task.CompletedTask;
        public Task BeforeFinalMoveAsync(string stagingDirectory, CancellationToken cancellationToken) => Task.CompletedTask;

        public Task AfterAllCandidateGuardsAcquiredAsync(string candidateDirectory, CancellationToken cancellationToken)
        {
            try { File.WriteAllText(Path.Combine(candidateDirectory, "lib", "dependency.dll"), "evil"); }
            catch (IOException) { FileReplacementBlocked = true; }
            try { Directory.Move(Path.Combine(candidateDirectory, "lib"), Path.Combine(candidateDirectory, "renamed-lib")); }
            catch (IOException) { DirectoryRenameBlocked = true; }
            if (!FileReplacementBlocked || !DirectoryRenameBlocked) throw new ComponentActivationException("The fixture replaced an unguarded candidate dependency.");
            throw new ComponentActivationException("The guarded replacement attempt was blocked.");
        }
    }
}
