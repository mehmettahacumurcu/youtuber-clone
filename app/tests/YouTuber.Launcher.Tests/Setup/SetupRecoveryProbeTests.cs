using System.Collections.Immutable;
using YouTuber.Launcher.Activation;
using YouTuber.Launcher.Downloads;
using YouTuber.Launcher.Setup;
using Xunit;

namespace YouTuber.Launcher.Tests.Setup;

public sealed class SetupRecoveryProbeTests
{
    private const string Revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private const string Hash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    private const string DownloadHash = "9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a";

    [Fact]
    public async Task Fresh_setup_does_not_skip_preflight_when_artifacts_are_absent()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Path);
        var probe = new SetupRecoveryProbe([request], new ActiveComponentsStore(Path.Combine(directory.Path, "data")), ["voice"]);

        var coordinator = await SetupCoordinator.CreateAsync(new MemoryJournal(), probe);

        Assert.Equal(SetupStage.Preflight, coordinator.State.Stage);
    }

    [Fact]
    public async Task Matching_partial_download_evidence_recovers_at_download_with_durable_byte_count()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Path);
        Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
        await File.WriteAllBytesAsync(request.TargetPath + ".partial", [1, 2]);
        ResumeMetadata.Create(request, 2, "etag").Save(request.TargetPath + ".partial.json");
        var state = StateAt(SetupStage.Verify);
        var journal = new MemoryJournal { State = state };
        var probe = new SetupRecoveryProbe([request], new ActiveComponentsStore(Path.Combine(directory.Path, "data")), ["voice"]);

        var coordinator = await SetupCoordinator.CreateAsync(journal, probe);

        Assert.Equal(SetupStage.Download, coordinator.State.Stage);
        Assert.Equal(2, coordinator.State.Components.Single(component => component.Name == "artifact.zip").BytesReceived);
    }

    [Fact]
    public async Task Same_size_corrupt_download_rewinds_a_post_download_checkpoint()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Path);
        Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
        await File.WriteAllBytesAsync(request.TargetPath, [4, 3, 2, 1]);
        var journal = new MemoryJournal { State = StateAt(SetupStage.Verify) };
        var probe = new SetupRecoveryProbe([request], new ActiveComponentsStore(Path.Combine(directory.Path, "data")), ["voice"]);

        var coordinator = await SetupCoordinator.CreateAsync(journal, probe);

        Assert.Equal(SetupStage.Download, coordinator.State.Stage);
        Assert.DoesNotContain(SetupStage.Download, coordinator.State.CompletedStages);
    }

    [Fact]
    public async Task Missing_download_after_a_durable_checkpoint_rewinds_to_download()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Path);
        var journal = new MemoryJournal { State = StateAt(SetupStage.Verify) };
        var probe = new SetupRecoveryProbe(
            [request],
            new ActiveComponentsStore(Path.Combine(directory.Path, "data")),
            ["voice"]);

        var coordinator = await SetupCoordinator.CreateAsync(journal, probe);

        Assert.Equal(SetupStage.Download, coordinator.State.Stage);
        Assert.DoesNotContain(SetupStage.Download, coordinator.State.CompletedStages);
        Assert.Equal(SetupStage.Download, journal.State!.Stage);
    }

    [Fact]
    public async Task Unfinished_real_activation_transaction_recovers_at_activate()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Path);
        Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
        await File.WriteAllBytesAsync(request.TargetPath, [1, 2, 3, 4]);
        var store = new ActiveComponentsStore(Path.Combine(directory.Path, "data"));
        var staging = Path.Combine(store.ActivationRoot, "voice.staging-test");
        await store.SaveTransactionAsync(new ActivationTransaction(
            "voice", "archive", "1.0.0", Hash, "components/voice/1.0.0", staging, ActivationPhase.Extracted, null));
        var journal = new MemoryJournal { State = StateAt(SetupStage.OllamaModels) };
        var probe = new SetupRecoveryProbe([request], store, ["voice"]);

        var coordinator = await SetupCoordinator.CreateAsync(journal, probe);

        Assert.Equal(SetupStage.Activate, coordinator.State.Stage);
    }

    [Fact]
    public async Task Real_active_pointer_allows_setup_to_continue_after_activation()
    {
        using var directory = new TemporaryDirectory();
        var request = Request(directory.Path);
        Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
        await File.WriteAllBytesAsync(request.TargetPath, [1, 2, 3, 4]);
        var store = new ActiveComponentsStore(Path.Combine(directory.Path, "data"));
        await store.SaveAsync(new ActiveComponents(new Dictionary<string, ActiveComponentPointer>
        {
            ["voice"] = new("components/voice/1.0.0", Hash),
        }));
        var journal = new MemoryJournal { State = StateAt(SetupStage.OllamaModels) };
        var probe = new SetupRecoveryProbe([request], store, ["voice"]);

        var coordinator = await SetupCoordinator.CreateAsync(journal, probe);

        Assert.Equal(SetupStage.OllamaModels, coordinator.State.Stage);
    }

    [Fact]
    public void Failed_atomic_journal_replacement_preserves_the_last_valid_snapshot()
    {
        using var directory = new TemporaryDirectory();
        var path = Path.Combine(directory.Path, "setup.json");
        var journal = new AtomicFileSetupJournal(path);
        var original = SetupState.Initial with { IsPaused = true, Revision = 1 };
        journal.Save(original);
        using var guard = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);

        Assert.Throws<IOException>(() => journal.Save(SetupState.Initial with { IsPaused = false, Revision = 2 }));
        var restored = Assert.IsType<SetupState>(journal.Load());
        Assert.Equal(original.Stage, restored.Stage);
        Assert.Equal(original.IsPaused, restored.IsPaused);
        Assert.Equal(original.CompletedStages, restored.CompletedStages);
        Assert.Empty(Directory.EnumerateFiles(directory.Path, "*.tmp"));
    }

    [Fact]
    public void Atomic_journal_rejects_a_stale_revision_instead_of_committing_it_last()
    {
        using var directory = new TemporaryDirectory();
        var journal = new AtomicFileSetupJournal(Path.Combine(directory.Path, "setup.json"));
        journal.Save(SetupState.Initial with { IsPaused = true, Revision = 1 });

        Assert.Throws<SetupJournalConflictException>(() => journal.Save(SetupState.Initial with { IsPaused = false, Revision = 1 }));
        Assert.True(journal.Load()!.IsPaused);
    }

    [Fact]
    public async Task Two_coordinators_merge_racing_checkpoints_through_revision_conflict_retry()
    {
        using var directory = new TemporaryDirectory();
        var journal = new AtomicFileSetupJournal(Path.Combine(directory.Path, "setup.json"));
        journal.Save(SetupState.Initial with { Revision = 1 });
        var first = new SetupCoordinator(journal);
        var second = new SetupCoordinator(journal);

        await Task.WhenAll(
            Task.Run(() => first.ReportComponent(new ComponentProgress("voice", "1", 1, 2, 1, null, "Downloading"))),
            Task.Run(() => second.ReportComponent(new ComponentProgress("studio", "1", 1, 2, 1, null, "Downloading"))));

        var restored = journal.Load()!;
        Assert.Equal(3, restored.Revision);
        Assert.Equal(["studio", "voice"], restored.Components.Select(value => value.Name).Order().ToArray());
    }

    private static DownloadRequest Request(string root) => new(
        new Uri($"https://huggingface.co/owner/repo/resolve/{Revision}/artifact.zip?download=true"),
        Path.Combine(root, "downloads", "artifact.zip"),
        4,
        DownloadHash);

    private static SetupState StateAt(SetupStage stage) => SetupState.Initial with
    {
        Stage = stage,
        TermsAccepted = true,
        CompletedStages = Enum.GetValues<SetupStage>().Where(value => value < stage).ToImmutableHashSet(),
    };

    private sealed class MemoryJournal : ISetupJournal
    {
        public SetupState? State { get; set; }
        public SetupState? Load() => State;
        public void Save(SetupState state) => State = state;
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"youtuber-setup-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }
        public void Dispose() => Directory.Delete(Path, recursive: true);
    }
}
