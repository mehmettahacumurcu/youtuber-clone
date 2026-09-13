using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using YouTuber.Launcher.Ollama;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Uninstall;
using YouTuber.Launcher.Distribution;
using Xunit;

namespace YouTuber.Launcher.Tests.Ollama;

public sealed class OllamaManagerTests
{
    [Fact]
    public async Task Fresh_install_discovers_Ollama_from_refreshed_user_path_when_inherited_path_is_clean()
    {
        using var installer = new TemporaryBinary();
        var directory = Path.Combine(Path.GetTempPath(), "youtuber-ollama-path-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var binary = Path.Combine(directory, "ollama.exe");
        File.WriteAllText(binary, "installed after launcher start");
        try
        {
            var verified = new List<string>();
            var locator = new OllamaBinaryLocator(
                processPath: () => string.Empty,
                userPath: () => directory,
                machinePath: () => string.Empty,
                fixedCandidates: () => []);
            using var controller = new OllamaServerController(
                binaryLocator: locator,
                verifyServer: (candidate, publication, _) =>
                {
                    verified.Add(candidate);
                    return Task.FromResult(new VerifiedOllamaServer(candidate, new Version(publication.ReleaseVersion), publication.ReleaseIdentity));
                });

            var server = await controller.LocateAndVerifyAsync(Publication(installer.Path, "0.9.1"));

            Assert.Equal(binary, server.BinaryPath);
            Assert.Equal([binary], verified);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task Controlled_runtime_serve_releases_its_composed_reservation_only_at_start()
    {
        using var ports = new PortAllocator().Reserve(1);
        var serve = new ReservationAssertingLauncher(ports[0]);
        using var controller = new OllamaServerController(new SequenceTcpOwnerProbe((int?)null), serve, new FakeProcessInventory(), jobFactory: () => new NoopProcessJob());
        var manager = new OllamaManager(serverController: controller);
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")), ports[0].Port);

        await manager.StartControlledServeAsync(new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "test"), installation, ports[0]);

        Assert.True(serve.ReservationWasReleasedAtStart);
        Assert.True(ports[0].IsReleased);
    }

    [Fact]
    public void DetectExistingPreservesItsCustomModelDirectoryAndConfiguredLoopbackPort()
    {
        var installation = OllamaInstallation.Existing(
            @"D:\Tools\ollama.exe",
            new Uri("http://127.0.0.1:22114"),
            "0.9.1",
            @"F:\My Ollama Models");

        var manager = new OllamaManager();
        var selected = manager.SelectInstallation(installation, @"D:\YouTuberData");

        Assert.True(selected.IsExisting);
        Assert.Equal(@"F:\My Ollama Models", selected.ModelDirectory);
        Assert.Equal(22114, selected.ApiEndpoint.Port);
    }

    [Fact]
    public void SelectInstallationUsesDataRootOnlyForANewOllamaInstall()
    {
        var selected = new OllamaManager().SelectInstallation(null, @"D:\YouTuberData");

        Assert.False(selected.IsExisting);
        Assert.Equal(@"D:\YouTuberData\models\ollama", selected.ModelDirectory);
        Assert.Equal(@"D:\YouTuberData\models\ollama", selected.Environment["OLLAMA_MODELS"]);
    }

    [Fact]
    public void SelectedExistingBinaryUsesTheComposedEndpointAndChildOnlyOwnedModelStore()
    {
        var detected = OllamaInstallation.Existing(
            @"D:\Tools\ollama.exe",
            new Uri("http://127.0.0.1:22114"),
            "0.9.1",
            @"F:\User Ollama Models");
        var composed = OllamaInstallation.New(@"D:\YouTuberData", 32123);

        var selected = new OllamaManager().SelectInstallation(detected, composed);

        Assert.True(selected.IsExisting);
        Assert.Equal(detected.BinaryPath, selected.BinaryPath);
        Assert.Equal(detected.Version, selected.Version);
        Assert.Equal(composed.ApiEndpoint, selected.ApiEndpoint);
        Assert.Equal(composed.ModelDirectory, selected.ModelDirectory);
        Assert.Equal(composed.ModelDirectory, selected.Environment["OLLAMA_MODELS"]);
        Assert.Equal("127.0.0.1:32123", selected.Environment["OLLAMA_HOST"]);
    }

    [Fact]
    public async Task ExistingStartVerifiesAndLaunchesTheExactDetectedBinary()
    {
        using var root = new TemporaryDirectory();
        using var installer = new TemporaryBinary();
        using var ports = new PortAllocator().Reserve(1);
        var detectedPath = @"C:\SelectedOllama\ollama.exe";
        var detected = OllamaInstallation.Existing(
            detectedPath,
            new Uri("http://127.0.0.1:11434"),
            "0.9.1",
            @"C:\Users\person\.ollama\models");
        var composed = OllamaInstallation.New(root.Path, ports[0].Port);
        var controller = new SelectedPathServerController();
        var manager = new OllamaManager(serverController: controller);
        var selected = manager.SelectInstallation(detected, composed);

        await manager.StartInstalledControlledAsync(Publication(installer.Path, "0.9.1"), selected, ports[0]);

        Assert.Equal(detectedPath, controller.VerifiedPath, ignoreCase: true);
        Assert.Equal(detectedPath, controller.ServedPath, ignoreCase: true);
    }

    [Fact]
    public void ExistingInstallationRequiresACompatibleVersion()
    {
        var old = OllamaInstallation.Existing(@"D:\Tools\ollama.exe", new Uri("http://127.0.0.1:11434"), "0.0.9", @"D:\models");

        Assert.Throws<InvalidOperationException>(() => new OllamaManager(minimumVersion: new Version(0, 1, 0)).SelectInstallation(old, @"D:\YouTuberData"));
    }

    [Fact]
    public async Task DetectorFindsConfiguredExistingOllamaWithoutChangingItsStorage()
    {
        using var binary = new TemporaryBinary();
        var detector = new OllamaInstallationDetector(
            () => binary.Path,
            name => name switch { "OLLAMA_HOST" => "127.0.0.1:22114", "OLLAMA_MODELS" => @"G:\CustomModels", _ => null },
            (_, _) => Task.FromResult<string?>("ollama version is 0.9.1"),
            (_, _) => Task.FromResult(true),
            _ => true);

        var detected = await detector.DetectAsync();

        Assert.NotNull(detected);
        Assert.True(detected.ApiReachable);
        Assert.Equal(@"G:\CustomModels", detected.ModelDirectory);
        Assert.Equal(22114, detected.ApiEndpoint.Port);
    }

    [Fact]
    public async Task DetectorFailsClosedWhenAFoundOllamaBinaryIsNotTrusted()
    {
        using var binary = new TemporaryBinary();
        var detector = new OllamaInstallationDetector(
            () => binary.Path,
            _ => null,
            (_, _) => Task.FromResult<string?>("ollama version is 0.9.1"),
            (_, _) => Task.FromResult(false),
            _ => false);

        await Assert.ThrowsAsync<InvalidOperationException>(() => detector.DetectAsync());
    }

    [Fact]
    public async Task ClientUsesResidencyKeepAliveValuesAndWaitsForVerifierToDisappear()
    {
        var handler = new OllamaServerHandler();
        var client = Client(handler);

        await client.UnloadModelAsync(OllamaClient.VerifierModel);
        await client.GenerateAnswerAsync(OllamaClient.SpeakerModel, "merhaba");
        await client.WaitForModelAbsentAsync(OllamaClient.VerifierModel, TimeSpan.FromSeconds(1), TimeSpan.Zero);

        Assert.Equal([0, -1], handler.KeepAlives);
        Assert.Equal(2, handler.ProcessRequests);
    }

    [Fact]
    public async Task ClientDoesNotGenerateWithSpeakerUntilVerifierAbsenceIsConfirmed()
    {
        var handler = new OllamaServerHandler();
        var client = Client(handler);

        await client.GenerateSpeakerAfterVerifierReleasedAsync("merhaba", TimeSpan.FromSeconds(1), TimeSpan.Zero);

        Assert.Equal(["generate:qwen3:4b", "ps", "ps", "generate:speaker-v5-a636"], handler.Events);
    }

    [Fact]
    public async Task DeleteOwnedModelRejectsAnyNonYouTuberIdentity()
    {
        var client = Client(new OllamaServerHandler());

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.DeleteOwnedModelAsync("llama3:latest"));
    }

    [Fact]
    public void ClientFactoryRefusesAnInstallationWhoseBinaryFailsPublisherTrust()
    {
        using var binary = new TemporaryBinary();
        var installation = OllamaInstallation.Existing(binary.Path, new Uri("http://127.0.0.1:11434"), "0.9.1", @"C:\models");

        Assert.Throws<InvalidOperationException>(() => OllamaClient.FromInstallation(installation, trust: new FakeTrust(false)));
    }

    [Fact]
    public async Task PullModelReportsEachStreamingOllamaProgressEvent()
    {
        var progress = new List<OllamaPullProgress>();
        var client = Client(new StreamingPullHandler());

        await client.PullModelAsync(OllamaClient.VerifierModel, (item, _) => { progress.Add(item); return Task.CompletedTask; });

        Assert.Equal([1L, 2L], progress.Select(item => item.Completed));
        Assert.Equal("success", progress[1].Status);
    }

    [Theory]
    [InlineData(HttpStatusCode.TemporaryRedirect)]
    [InlineData(HttpStatusCode.PermanentRedirect)]
    public async Task ClientRejectsExternalRedirectsBeforeAnyOllamaOperation(HttpStatusCode redirect)
    {
        var client = Client(new RedirectingHandler(redirect));

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.EnsureHealthyAsync());
    }

    [Fact]
    public async Task ClientDoesNotFollowARealLocalRedirectOrSendThePromptToItsTarget()
    {
        await using var target = await LocalHttpServer.StartAsync(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("{\"response\":\"unexpected\"}", Encoding.UTF8, "application/json"),
        }));
        await using var source = await LocalHttpServer.StartAsync(request =>
        {
            var response = new HttpResponseMessage(HttpStatusCode.TemporaryRedirect);
            response.Headers.Location = new Uri(target.Endpoint.AbsoluteUri + "api/generate");
            return Task.FromResult(response);
        });
        using var handler = new HttpClientHandler { AllowAutoRedirect = true };
        using var client = OllamaClient.CreateForTest(handler, @"C:\Ollama\ollama.exe", source.Endpoint);

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.GenerateAnswerAsync(OllamaClient.SpeakerModel, "do not disclose this prompt"));

        Assert.Equal(0, target.RequestCount);
        Assert.Equal(0, target.BodyBytes);
    }

    [Fact]
    public async Task ImportDeletesVerifiedTemporaryGgufOnlyAfterBothModelsAreReported()
    {
        using var files = new ImportFiles();
        var handler = new OllamaServerHandler { IncludeSpeakerInTags = true };
        var runner = new RecordingRunner();
        var installation = new OllamaManager().SelectInstallation(null, files.Root);
        var client = Client(handler, runner);

        await new OllamaManager(importRoot: files.Root).EnsureModelsAsync(client, installation, files.Artifacts, ModelRoles());

        Assert.False(File.Exists(files.VerifierGguf));
        Assert.False(File.Exists(files.Gguf));
        Assert.Equal(@"create speaker-v5-a636 -f", string.Join(' ', runner.Arguments.Take(3)));
        Assert.Equal(Path.Combine(files.Root, "models", "ollama"), runner.Environment["OLLAMA_MODELS"]);
    }

    [Fact]
    public async Task Signed_model_roles_import_the_immutable_verifier_source_without_pulling_a_mutable_tag()
    {
        using var files = new ImportFiles();
        var runner = new RecordingRunner();
        var roles = ModelRoles("speaker-release-candidate");
        var handler = new OllamaServerHandler { IncludeSpeakerInTags = true, SpeakerModel = roles.Speaker.Name };
        var installation = new OllamaManager().SelectInstallation(null, files.Root);

        await new OllamaManager(importRoot: files.Root).EnsureModelsAsync(Client(handler, runner), installation, files.Artifacts, roles);

        Assert.Equal(@"create speaker-release-candidate -f", string.Join(' ', runner.Arguments.Take(3)));
        Assert.Contains(runner.Calls, call => call.Count >= 2 && call[0] == "create" && call[1] == roles.Verifier.Name);
        Assert.Empty(handler.PulledModels);
    }

    [Fact]
    public async Task ImportAcceptsOllamasCanonicalLatestTagForAnUntaggedSpeakerAlias()
    {
        using var files = new ImportFiles();
        var handler = new OllamaServerHandler
        {
            IncludeSpeakerInTags = true,
            ReportSpeakerWithLatestTag = true,
        };

        await new OllamaManager(importRoot: files.Root).EnsureModelsAsync(
            Client(handler, new RecordingRunner()),
            OllamaInstallation.New(files.Root),
            files.Artifacts,
            ModelRoles());

        Assert.False(File.Exists(files.VerifierGguf));
        Assert.False(File.Exists(files.Gguf));
    }

    [Fact]
    public async Task ImportRejectsAnInstalledVerifierWhoseDigestDoesNotMatchTheSignedIdentity()
    {
        using var files = new ImportFiles();
        var handler = new OllamaServerHandler
        {
            IncludeSpeakerInTags = true,
            VerifierDigest = new string('9', 64),
        };

        await Assert.ThrowsAsync<InvalidOperationException>(() => new OllamaManager(importRoot: files.Root).EnsureModelsAsync(
            Client(handler, new RecordingRunner()),
            OllamaInstallation.New(files.Root),
            files.Artifacts,
            ModelRoles()));

        Assert.True(File.Exists(files.VerifierGguf));
        Assert.True(File.Exists(files.Gguf));
    }

    [Fact]
    public async Task ImportRejectsAnInstalledVerifierWhoseSizeDoesNotMatchTheSignedIdentity()
    {
        using var files = new ImportFiles();
        var handler = new OllamaServerHandler
        {
            IncludeSpeakerInTags = true,
            VerifierSize = 999,
        };

        await Assert.ThrowsAsync<InvalidOperationException>(() => new OllamaManager(importRoot: files.Root).EnsureModelsAsync(
            Client(handler, new RecordingRunner()),
            OllamaInstallation.New(files.Root),
            files.Artifacts,
            ModelRoles()));
    }

    [Fact]
    public async Task ImportRejectsVerifierBytesThatDoNotMatchTheSignedSourceHashBeforeCreatingModels()
    {
        using var files = new ImportFiles();
        File.WriteAllText(files.VerifierGguf, "tampered verifier");
        var runner = new RecordingRunner();

        await Assert.ThrowsAsync<InvalidOperationException>(() => new OllamaManager(importRoot: files.Root).EnsureModelsAsync(
            Client(new OllamaServerHandler { IncludeSpeakerInTags = true }, runner),
            OllamaInstallation.New(files.Root),
            files.Artifacts,
            ModelRoles()));

        Assert.Empty(runner.Calls);
    }

    [Fact]
    public async Task Controlled_model_import_registers_exact_ollama_store_files_for_uninstall()
    {
        using var files = new ImportFiles();
        var manager = new OllamaManager(importRoot: files.Root);
        var installation = manager.SelectInstallation(null, files.Root);
        Directory.CreateDirectory(installation.ModelDirectory);
        var ownedBlob = Path.Combine(installation.ModelDirectory, "blobs", "sha256-owned");
        var unrelated = Path.Combine(installation.ModelDirectory, "preexisting-user-model.bin");
        File.WriteAllText(unrelated, "user");

        await manager.EnsureModelsAsync(
            Client(new OllamaServerHandler { IncludeSpeakerInTags = true }, new CreatingModelFileRunner(ownedBlob)),
            installation,
            files.Artifacts,
            ModelRoles());

        var plan = new OwnedDataInventory().Plan(files.Root, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData);
        Assert.Contains(ownedBlob, plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain(unrelated, plan.Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ImportRetainsTemporaryGgufWhenOllamaDoesNotReportSpeaker()
    {
        using var files = new ImportFiles();
        var client = Client(new OllamaServerHandler(), new RecordingRunner());

        await Assert.ThrowsAsync<InvalidOperationException>(() => new OllamaManager(importRoot: files.Root).EnsureModelsAsync(client, OllamaInstallation.New(files.Root), files.Artifacts, ModelRoles()));

        Assert.True(File.Exists(files.Gguf));
    }

    [Fact]
    public async Task ImportRefusesToDeleteAGgufOutsideTheOwnedStagingDirectory()
    {
        using var files = new ImportFiles();
        using var outside = new ImportFiles();
        var unsafeArtifacts = files.Artifacts with { GgufPath = outside.Gguf, GgufSize = new FileInfo(outside.Gguf).Length, GgufSha256 = Hash(outside.Gguf) };

        await Assert.ThrowsAsync<InvalidOperationException>(() => new OllamaManager(importRoot: files.Root).EnsureModelsAsync(Client(new OllamaServerHandler(), new RecordingRunner()), OllamaInstallation.New(files.Root), unsafeArtifacts, ModelRoles()));

        Assert.True(File.Exists(outside.Gguf));
    }

    [Fact]
    public async Task ImportPinsTheStagingAncestorThroughModelCreation()
    {
        using var files = new ImportFiles();
        var runner = new AncestorRenameRunner(files.Root);

        await new OllamaManager(importRoot: files.Root).EnsureModelsAsync(
            Client(new OllamaServerHandler { IncludeSpeakerInTags = true }, runner),
            OllamaInstallation.New(files.Root),
            files.Artifacts,
            ModelRoles());

        Assert.True(runner.RenameRejected);
    }

    [Fact]
    public async Task ImportPinsANestedModelfileParentThroughModelCreation()
    {
        using var files = new ImportFiles(nestedModelfile: true);
        var runner = new AncestorRenameRunner(Path.GetDirectoryName(files.Artifacts.ModelfilePath)!);

        await new OllamaManager(importRoot: files.Root).EnsureModelsAsync(
            Client(new OllamaServerHandler { IncludeSpeakerInTags = true }, runner),
            OllamaInstallation.New(files.Root),
            files.Artifacts,
            ModelRoles());

        Assert.True(runner.RenameRejected);
    }

    [Fact]
    public Task ServerControllerRefusesToStartWhenTheLoopbackPortAlreadyHasAnOwner()
    {
        using var controller = new OllamaServerController(new FixedTcpOwnerProbe(42));
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")));

        Assert.Throws<InvalidOperationException>(() => controller.StartServeAsync(new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release"), installation).GetAwaiter().GetResult());
        return Task.CompletedTask;
    }

    [Fact]
    public async Task ServerControllerRejectsAHealthyListenerOwnedByAnotherProcessAndCleansOnlyItsServeReceipt()
    {
        await using var health = await LocalHttpServer.StartAsync(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)));
        var process = new FakeServeProcess(101);
        var launcher = new FakeServeProcessLauncher(process);
        using var controller = new OllamaServerController(new SequenceTcpOwnerProbe(null, 202), launcher, jobFactory: () => new NoopProcessJob());
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))) with { ApiEndpoint = health.Endpoint };
        var server = new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release");

        await controller.StartServeAsync(server, installation);
        await Assert.ThrowsAsync<InvalidOperationException>(() => controller.WaitForHealthyAndConfirmStoreAsync(server, installation));

        Assert.True(process.KilledTree);
        Assert.True(process.Waited);
        Assert.True(process.Disposed);
        Assert.Equal(1, launcher.Starts);
    }

    [Fact]
    public async Task ServerControllerAcceptsAHealthyListenerOwnedByItsServeReceipt()
    {
        await using var health = await LocalHttpServer.StartAsync(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)));
        var process = new FakeServeProcess(101);
        using var controller = new OllamaServerController(new SequenceTcpOwnerProbe(null, process.Id), new FakeServeProcessLauncher(process), jobFactory: () => new NoopProcessJob());
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))) with { ApiEndpoint = health.Endpoint };
        var server = new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release");

        await controller.StartServeAsync(server, installation);
        await controller.WaitForHealthyAndConfirmStoreAsync(server, installation);

        Assert.False(process.KilledTree);
        await controller.StopOwnedAsync();
        Assert.True(process.KilledTree);
    }

    [Fact]
    public async Task ServerControllerRetriesTransientTransportFailureUntilOwnedServeBecomesHealthy()
    {
        using var root = new TemporaryDirectory();
        var process = new FakeServeProcess(101);
        await using var endpoint = await LocalHttpServer.StartAsync(
            _ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)),
            dropConnections: 10);
        using var controller = new OllamaServerController(
            new SequenceTcpOwnerProbe(null, process.Id),
            new FakeServeProcessLauncher(process),
            jobFactory: () => new NoopProcessJob());
        var installation = OllamaInstallation.New(root.Path) with { ApiEndpoint = endpoint.Endpoint };
        var server = new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release");
        await controller.StartServeAsync(server, installation);

        await controller.WaitForHealthyAndConfirmStoreAsync(server, installation).WaitAsync(TimeSpan.FromSeconds(3));

        Assert.Equal(11, endpoint.RequestCount);
        Assert.False(process.KilledTree);
    }

    [Fact]
    public async Task ServerControllerHealthRetryHonorsCallerCancellationAndCleansOwnedTree()
    {
        using var root = new TemporaryDirectory();
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        var process = new FakeServeProcess(101);
        using var controller = new OllamaServerController(
            new SequenceTcpOwnerProbe((int?)null),
            new FakeServeProcessLauncher(process),
            jobFactory: () => new NoopProcessJob());
        var installation = OllamaInstallation.New(root.Path, port);
        var server = new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release");
        await controller.StartServeAsync(server, installation);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            controller.WaitForHealthyAndConfirmStoreAsync(server, installation, cancellation.Token));

        Assert.True(process.KilledTree);
        Assert.True(process.Waited);
        Assert.True(process.Disposed);
    }

    [Fact]
    public async Task ServerControllerAssignsSuspendedServeToKillOnCloseJobBeforeResume()
    {
        var events = new List<string>();
        var process = new LifecycleServeProcess(events);
        var job = new RecordingProcessJob(events);
        using var controller = new OllamaServerController(
            new SequenceTcpOwnerProbe((int?)null),
            new LifecycleServeLauncher(process, events),
            jobFactory: () => job);
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")));

        await controller.StartServeAsync(new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release"), installation);

        Assert.Equal(["start-suspended", "assign-job", "resume"], events);
        Assert.False(job.Disposed);
    }

    [Fact]
    public async Task ServerControllerKillsSuspendedServeWhenJobAssignmentFails()
    {
        var events = new List<string>();
        var process = new LifecycleServeProcess(events);
        var job = new RecordingProcessJob(events) { FailAssignment = true };
        using var controller = new OllamaServerController(
            new SequenceTcpOwnerProbe((int?)null),
            new LifecycleServeLauncher(process, events),
            jobFactory: () => job);
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")));

        await Assert.ThrowsAsync<InvalidOperationException>(() => controller.StartServeAsync(
            new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release"), installation));

        Assert.Equal(["start-suspended", "assign-job", "job-dispose", "kill-tree", "wait", "process-dispose"], events);
        Assert.False(process.Resumed);
    }

    [Fact]
    public async Task ServerControllerDisposesAssignedJobAndKillsServeWhenResumeFails()
    {
        var events = new List<string>();
        var process = new LifecycleServeProcess(events) { FailResume = true };
        var job = new RecordingProcessJob(events);
        using var controller = new OllamaServerController(
            new SequenceTcpOwnerProbe((int?)null),
            new LifecycleServeLauncher(process, events),
            jobFactory: () => job);
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")));

        await Assert.ThrowsAsync<InvalidOperationException>(() => controller.StartServeAsync(
            new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), "release"), installation));

        Assert.Equal(["start-suspended", "assign-job", "resume", "job-dispose", "kill-tree", "wait", "process-dispose"], events);
    }

    [Fact]
    public async Task ProductionServeLauncherDoesNotExecuteOllamaUntilResume()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var directory = new TemporaryDirectory();
        var marker = Path.Combine(directory.Path, "serve-started.pid");
        await using var executable = VerifiedOllamaExecutable.OpenForTest(WorkerFixtureExecutable());
        await using var server = new VerifiedOllamaServer(executable, new Version(0, 9, 1), "release");
        using var process = new ProcessOllamaServeProcessLauncher().Start(
            server,
            new Dictionary<string, string> { ["YOUTUBER_FIXTURE_START_FILE"] = marker });

        await Task.Delay(100);
        Assert.False(File.Exists(marker));

        process.Resume();
        for (var attempt = 0; attempt < 60 && !File.Exists(marker); attempt++) await Task.Delay(50);
        Assert.True(File.Exists(marker));
        if (!process.HasExited) process.Kill(entireProcessTree: true);
        await process.WaitForExitAsync(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(3));
    }

    [Fact]
    public async Task AutoStartAttributionStopsOnlyTheNewVerifiedInstallerProcess()
    {
        var installerStart = DateTime.UtcNow.AddSeconds(-2);
        var installerEnd = DateTime.UtcNow.AddSeconds(-1);
        var attributed = new FakeObservedProcess(11, installerStart.AddMilliseconds(500), @"C:\Program Files\Ollama\ollama.exe");
        var unrelated = new FakeObservedProcess(12, installerStart.AddMilliseconds(500), @"C:\Users\me\bin\ollama.exe");
        var inventory = new FakeProcessInventory(attributed, unrelated);
        using var controller = new OllamaServerController(processInventory: inventory);
        await controller.SnapshotBeforeInstallAsync();
        inventory.IncludeNewProcesses = true;
        var server = new VerifiedOllamaServer(@"C:\Program Files\Ollama\ollama.exe", new Version(0, 9, 1), "release");

        var receipt = await controller.CaptureInstallerAutoStartAsync(server, installerStart, installerEnd);
        Assert.NotNull(receipt);
        await controller.StopAutoStartedAsync(receipt!);

        Assert.True(attributed.KilledTree);
        Assert.True(attributed.Waited);
        Assert.False(unrelated.KilledTree);
    }

    [Fact]
    public async Task StopAutoStartedKeepsTheObservedProcessHandleUntilExitCompletes()
    {
        var observed = new BlockingObservedProcess(11, DateTime.UtcNow, @"C:\Program Files\Ollama\ollama.exe");
        var inventory = new FakeProcessInventory(observed) { IncludeNewProcesses = true };
        using var controller = new OllamaServerController(processInventory: inventory);
        var receipt = new OllamaProcessReceipt(observed.Id, observed.StartTimeUtc, observed.SessionId, observed.ExecutablePath, "release");

        var stopping = controller.StopAutoStartedAsync(receipt);

        Assert.True(observed.KilledTree);
        Assert.False(observed.Disposed);
        observed.CompleteExit();
        await stopping;
        Assert.True(observed.Disposed);
    }

    [Fact]
    public async Task NewInstallPassesOwnedModelsPathOnlyToChildWithoutPersistingUserEnvironment()
    {
        using var installerFile = new TemporaryBinary();
        using var verified = new VerifiedInstaller(new InstallerArtifact(installerFile.Path, 0, Hash(installerFile.Path), OllamaInstallation.ExpectedPublisher), new FileStream(installerFile.Path, FileMode.Open, FileAccess.Read, FileShare.Read));
        var events = new List<string>();
        var persisted = new List<string>();
        var runner = new CapturingInstallerRunner(() => persisted.Count == 0, events);
        var manager = new OllamaManager(modelsPathStore: new RecordingModelsPathStore(persisted, events), serverController: new SequenceServerController(events));
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")));
        var artifact = new OllamaInstallerArtifact(verified, Publication(installerFile.Path, "0.9.1"));

        await manager.InstallNewAsync(artifact, installation, runner);

        Assert.Empty(persisted);
        Assert.Equal(installation.ModelDirectory, runner.Environment["OLLAMA_MODELS"]);
        Assert.Equal(["snapshot", "install", "verify", "capture", "serve", "confirm"], events);
    }

    [Fact]
    public async Task ComposedNewInstallRetainsItsOllamaPortUntilControlledServeStarts()
    {
        using var installerFile = new TemporaryBinary();
        using var verified = new VerifiedInstaller(
            new InstallerArtifact(installerFile.Path, 0, Hash(installerFile.Path), OllamaInstallation.ExpectedPublisher),
            new FileStream(installerFile.Path, FileMode.Open, FileAccess.Read, FileShare.Read));
        using var allocation = new PortAllocator().Reserve(1);
        var reservation = allocation[0];
        var events = new List<string>();
        var manager = new OllamaManager(
            modelsPathStore: new RecordingModelsPathStore([], events),
            serverController: new SequenceServerController(events, reservation));
        var installation = OllamaInstallation.New(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")),
            reservation.Port);
        var artifact = new OllamaInstallerArtifact(verified, Publication(installerFile.Path, "0.9.1"));

        Assert.False(reservation.IsReleased);
        await manager.InstallNewComposedAsync(
            artifact,
            installation,
            reservation,
            new CapturingInstallerRunner(() => true, events));

        Assert.True(reservation.IsReleased);
        Assert.Equal(["snapshot", "install", "verify", "capture", "serve", "confirm"], events);
    }

    [Fact]
    public async Task NewInstallWaitsForAutoStartedProcessExitBeforeStartingControlledServe()
    {
        using var installerFile = new TemporaryBinary();
        using var verified = new VerifiedInstaller(new InstallerArtifact(installerFile.Path, 0, Hash(installerFile.Path), OllamaInstallation.ExpectedPublisher), new FileStream(installerFile.Path, FileMode.Open, FileAccess.Read, FileShare.Read));
        var controller = new BlockingAutoStartServerController();
        var manager = new OllamaManager(modelsPathStore: new RecordingModelsPathStore([], []), serverController: controller);
        var installation = OllamaInstallation.New(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")));
        var installing = manager.InstallNewAsync(new OllamaInstallerArtifact(verified, Publication(installerFile.Path, "0.9.1")), installation, new CapturingInstallerRunner(() => true, []));

        await controller.StopEntered.Task;
        Assert.False(controller.ControlledServeStarted);
        controller.AllowAutoStartExit.SetResult();
        await installing;

        Assert.True(controller.ControlledServeStarted);
    }

    [Fact]
    public void OllamaInstallerArtifactRejectsAWrongButOtherwiseVerifiedPublisher()
    {
        using var installerFile = new TemporaryBinary();
        using var verified = new VerifiedInstaller(new InstallerArtifact(installerFile.Path, 0, Hash(installerFile.Path), "Other Trusted Publisher"), new FileStream(installerFile.Path, FileMode.Open, FileAccess.Read, FileShare.Read));

        Assert.Throws<InstallerVerificationException>(() => new OllamaInstallerArtifact(verified, Publication(installerFile.Path, "0.9.1")));
    }

    [Fact]
    public void OllamaInstallerArtifactRejectsAnUnpinnedManifestIdentityBeforeInstallerExecution()
    {
        using var installerFile = new TemporaryBinary();
        using var verified = new VerifiedInstaller(new InstallerArtifact(installerFile.Path, 0, Hash(installerFile.Path), OllamaInstallation.ExpectedPublisher), new FileStream(installerFile.Path, FileMode.Open, FileAccess.Read, FileShare.Read));
        var publication = Publication(installerFile.Path, "0.9.1") with { ManifestIdentity = "same-publisher-but-wrong-release" };

        Assert.Throws<InvalidOperationException>(() => new OllamaInstallerArtifact(verified, publication));
    }

    [Fact]
    public async Task ExistingClientPinsExecutableAncestorsBeforeCreateModel()
    {
        using var files = new ImportFiles();
        var binary = Path.Combine(files.Root, "ollama.exe");
        File.WriteAllText(binary, "trusted ollama");
        var installation = OllamaInstallation.Existing(binary, new Uri("http://127.0.0.1:11434"), "0.9.1", files.Root);
        var runner = new AncestorRenameRunner(files.Root);
        using var client = OllamaClient.FromInstallation(installation, runner, new FakeTrust(true));

        await client.CreateModelAsync(OllamaClient.SpeakerModel, files.Artifacts.ModelfilePath, new Dictionary<string, string>());

        Assert.True(runner.RenameRejected);
    }

    private sealed class OllamaServerHandler : HttpMessageHandler
    {
        public List<int> KeepAlives { get; } = [];
        public List<string> Events { get; } = [];
        public int TagRequests { get; private set; }
        public int ProcessRequests { get; private set; }
        public bool IncludeSpeakerInTags { get; init; }
        public bool ReportSpeakerWithLatestTag { get; init; }
        public string SpeakerModel { get; init; } = "speaker-v5-a636";
        public string VerifierDigest { get; init; } = "8888888888888888888888888888888888888888888888888888888888888888";
        public long VerifierSize { get; init; } = 4;
        public long SpeakerSize { get; init; } = 5;
        public List<string> PulledModels { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (request.RequestUri!.AbsolutePath == "/api/generate")
            {
                using var document = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
                Events.Add($"generate:{document.RootElement.GetProperty("model").GetString()}");
                KeepAlives.Add(document.RootElement.GetProperty("keep_alive").GetInt32());
                return Json("{\"response\":\"ok\"}");
            }

            if (request.RequestUri.AbsolutePath == "/api/tags")
            {
                TagRequests++;
                Events.Add("tags");
                var reportedSpeakerModel = ReportSpeakerWithLatestTag ? SpeakerModel + ":latest" : SpeakerModel;
                return Json(IncludeSpeakerInTags
                    ? JsonSerializer.Serialize(new { models = new[] { new { name = "qwen3:4b", digest = VerifierDigest, size = VerifierSize }, new { name = reportedSpeakerModel, digest = new string('7', 64), size = SpeakerSize } } })
                    : TagRequests == 1 ? JsonSerializer.Serialize(new { models = new[] { new { name = "qwen3:4b", digest = VerifierDigest, size = VerifierSize } } }) : "{\"models\":[]}");
            }

            if (request.RequestUri.AbsolutePath == "/api/pull")
            {
                using var document = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
                PulledModels.Add(document.RootElement.GetProperty("name").GetString()!);
                return Json(string.Empty);
            }

            if (request.RequestUri.AbsolutePath == "/api/ps")
            {
                ProcessRequests++;
                Events.Add("ps");
                return Json(ProcessRequests == 1
                    ? "{\"models\":[{\"name\":\"qwen3:4b\"}]}"
                    : "{\"models\":[]}");
            }

            return Json("{}");
        }

        private static HttpResponseMessage Json(string payload) => new(HttpStatusCode.OK)
        {
            Content = new StringContent(payload, Encoding.UTF8, "application/json"),
        };
    }

    private static OllamaClient Client(HttpMessageHandler handler, IOllamaCommandRunner? runner = null) =>
        OllamaClient.CreateForTest(handler, @"C:\Ollama\ollama.exe", new Uri("http://127.0.0.1:11434"), runner);

    private sealed class FakeTrust(bool trusted) : IWinVerifyTrust
    {
        public AuthenticodeTrustResult Verify(string path) => new(trusted, trusted ? "Ollama, Inc." : null, trusted ? null : "untrusted");
    }

    private sealed class StreamingPullHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("{\"status\":\"pulling\",\"completed\":1,\"total\":2}\n{\"status\":\"success\",\"completed\":2,\"total\":2}\n", Encoding.UTF8, "application/json"),
        });
    }

    private sealed class RedirectingHandler(HttpStatusCode statusCode) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => Task.FromResult(new HttpResponseMessage(statusCode)
        {
            Headers = { Location = new Uri("https://evil.example/ollama") },
        });
    }

    private class RecordingRunner : IOllamaCommandRunner
    {
        public IReadOnlyList<string> Arguments { get; private set; } = [];
        public IReadOnlyDictionary<string, string> Environment { get; private set; } = new Dictionary<string, string>();
        public List<IReadOnlyList<string>> Calls { get; } = [];

        public virtual Task<int> RunAsync(string executable, IReadOnlyList<string> arguments, IReadOnlyDictionary<string, string> environment, CancellationToken cancellationToken = default)
        {
            Arguments = arguments;
            Environment = environment;
            Calls.Add(arguments.ToArray());
            return Task.FromResult(0);
        }
    }

    private sealed class AncestorRenameRunner(string root) : RecordingRunner
    {
        public bool RenameRejected { get; private set; }
        public override Task<int> RunAsync(string executable, IReadOnlyList<string> arguments, IReadOnlyDictionary<string, string> environment, CancellationToken cancellationToken = default)
        {
            try { Directory.Move(root, root + "-replaced"); }
            catch (IOException) { RenameRejected = true; }
            return Task.FromResult(0);
        }
    }

    private sealed class CreatingModelFileRunner(string path) : RecordingRunner
    {
        public override Task<int> RunAsync(string executable, IReadOnlyList<string> arguments, IReadOnlyDictionary<string, string> environment, CancellationToken cancellationToken = default)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, "owned");
            return base.RunAsync(executable, arguments, environment, cancellationToken);
        }
    }

    private sealed class FixedTcpOwnerProbe(int processId) : ITcpOwnerProbe
    {
        public int? GetListenerProcessId(Uri endpoint) => processId;
    }

    private sealed class SequenceTcpOwnerProbe(params int?[] processIds) : ITcpOwnerProbe
    {
        private int _index;
        public int? GetListenerProcessId(Uri endpoint) => processIds[Math.Min(_index++, processIds.Length - 1)];
    }

    private sealed class FakeServeProcessLauncher(FakeServeProcess process) : IOllamaServeProcessLauncher
    {
        public int Starts { get; private set; }
        public IOllamaServeProcess Start(VerifiedOllamaServer server, IReadOnlyDictionary<string, string> environment) { Starts++; return process; }
    }

    private sealed class ReservationAssertingLauncher(PortReservation reservation) : IOllamaServeProcessLauncher
    {
        public bool ReservationWasReleasedAtStart { get; private set; }
        public IOllamaServeProcess Start(VerifiedOllamaServer server, IReadOnlyDictionary<string, string> environment)
        {
            ReservationWasReleasedAtStart = reservation.IsReleased;
            return new FakeServeProcess(99);
        }
    }

    private sealed class FakeServeProcess(int id) : IOllamaServeProcess
    {
        public int Id => id;
        public bool HasExited { get; private set; }
        public IntPtr NativeHandle => new(id);
        public DateTime StartTimeUtc { get; } = DateTime.UtcNow;
        public int SessionId => 1;
        public bool KilledTree { get; private set; }
        public bool Waited { get; private set; }
        public bool Disposed { get; private set; }
        public void Resume() { }
        public void Kill(bool entireProcessTree) { KilledTree = entireProcessTree; HasExited = true; }
        public Task WaitForExitAsync(CancellationToken cancellationToken) { Waited = true; return Task.CompletedTask; }
        public void Dispose() => Disposed = true;
    }

    private sealed class LifecycleServeLauncher(LifecycleServeProcess process, List<string> events) : IOllamaServeProcessLauncher
    {
        public IOllamaServeProcess Start(VerifiedOllamaServer server, IReadOnlyDictionary<string, string> environment)
        {
            events.Add("start-suspended");
            return process;
        }
    }

    private sealed class LifecycleServeProcess(List<string> events) : IOllamaServeProcess
    {
        public int Id => 701;
        public bool HasExited { get; private set; }
        public IntPtr NativeHandle => new(701);
        public DateTime StartTimeUtc { get; } = DateTime.UtcNow;
        public int SessionId => 1;
        public bool FailResume { get; init; }
        public bool Resumed { get; private set; }
        public void Resume()
        {
            events.Add("resume");
            if (FailResume) throw new InvalidOperationException("resume failed");
            Resumed = true;
        }
        public void Kill(bool entireProcessTree) { events.Add("kill-tree"); HasExited = true; }
        public Task WaitForExitAsync(CancellationToken cancellationToken) { events.Add("wait"); return Task.CompletedTask; }
        public void Dispose() => events.Add("process-dispose");
    }

    private sealed class RecordingProcessJob(List<string> events) : IWorkerJob
    {
        public bool FailAssignment { get; init; }
        public bool Disposed { get; private set; }
        public void Assign(IJobProcess process)
        {
            events.Add("assign-job");
            if (FailAssignment) throw new InvalidOperationException("assignment failed");
        }
        public void Dispose() { Disposed = true; events.Add("job-dispose"); }
    }

    private sealed class NoopProcessJob : IWorkerJob
    {
        public void Assign(IJobProcess process) { }
        public void Dispose() { }
    }

    private sealed class FakeObservedProcess(int id, DateTime startTimeUtc, string executablePath) : IOllamaObservedProcess
    {
        public int Id => id;
        public DateTime StartTimeUtc => startTimeUtc;
        public int SessionId => 1;
        public string ExecutablePath => executablePath;
        public bool KilledTree { get; private set; }
        public bool Waited { get; private set; }
        public void Kill(bool entireProcessTree) => KilledTree = entireProcessTree;
        public Task WaitForExitAsync(CancellationToken cancellationToken) { Waited = true; return Task.CompletedTask; }
        public void Dispose() { }
    }

    private sealed class BlockingObservedProcess(int id, DateTime startTimeUtc, string executablePath) : IOllamaObservedProcess
    {
        private readonly TaskCompletionSource _exited = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public int Id => id;
        public DateTime StartTimeUtc => startTimeUtc;
        public int SessionId => 1;
        public string ExecutablePath => executablePath;
        public bool KilledTree { get; private set; }
        public bool Disposed { get; private set; }
        public void Kill(bool entireProcessTree) => KilledTree = entireProcessTree;
        public Task WaitForExitAsync(CancellationToken cancellationToken) => _exited.Task.WaitAsync(cancellationToken);
        public void CompleteExit() => _exited.SetResult();
        public void Dispose() => Disposed = true;
    }

    private sealed class FakeProcessInventory(params IOllamaObservedProcess[] processes) : IOllamaProcessInventory
    {
        public bool IncludeNewProcesses { get; set; }
        public IReadOnlyList<IOllamaObservedProcess> GetOllamaProcesses() => IncludeNewProcesses ? processes : [];
    }

    private sealed class RecordingModelsPathStore(List<string> values, List<string> events) : IOllamaModelsPathStore
    {
        public void Persist(string modelDirectory) { values.Add(modelDirectory); events.Add("persist"); }
    }

    private sealed class CapturingInstallerRunner(Func<bool> canRun, List<string> events) : IInstallerRunner
    {
        public IReadOnlyDictionary<string, string> Environment { get; private set; } = new Dictionary<string, string>();
        public Task<InstallerRunResult> RunAsync(VerifiedInstaller installer, IEnumerable<string> arguments, TimeSpan timeout, IReadOnlyDictionary<string, string>? environment = null, CancellationToken cancellationToken = default)
        {
            if (!canRun()) throw new InvalidOperationException("Models path was not persisted before installer execution.");
            events.Add("install");
            Environment = environment ?? new Dictionary<string, string>();
            return Task.FromResult(new InstallerRunResult(0, string.Empty, string.Empty, false));
        }
    }

    private sealed class SequenceServerController(List<string> events, PortReservation? expectedReleasedReservation = null) : IOllamaServerController
    {
        public Task SnapshotBeforeInstallAsync(CancellationToken cancellationToken = default) { events.Add("snapshot"); return Task.CompletedTask; }
        public Task<OllamaProcessReceipt?> CaptureInstallerAutoStartAsync(VerifiedOllamaServer server, DateTime installerStartedUtc, DateTime installerFinishedUtc, CancellationToken cancellationToken = default) { events.Add("capture"); return Task.FromResult<OllamaProcessReceipt?>(null); }
        public Task StopAutoStartedAsync(OllamaProcessReceipt receipt, CancellationToken cancellationToken = default) { events.Add("stop"); return Task.CompletedTask; }
        public Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, CancellationToken cancellationToken = default) { events.Add("verify"); return Task.FromResult(new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), publication.ReleaseIdentity)); }
        public Task StartServeAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default)
        {
            if (expectedReleasedReservation is not null) Assert.True(expectedReleasedReservation.IsReleased);
            events.Add("serve");
            return Task.CompletedTask;
        }
        public Task WaitForHealthyAndConfirmStoreAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default) { events.Add("confirm"); return Task.CompletedTask; }
        public Task StopOwnedAsync(CancellationToken cancellationToken = default) { events.Add("stop-owned"); return Task.CompletedTask; }
    }

    private sealed class SelectedPathServerController : IOllamaServerController
    {
        public string? VerifiedPath { get; private set; }
        public string? ServedPath { get; private set; }
        public Task SnapshotBeforeInstallAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<OllamaProcessReceipt?> CaptureInstallerAutoStartAsync(VerifiedOllamaServer server, DateTime installerStartedUtc, DateTime installerFinishedUtc, CancellationToken cancellationToken = default) => Task.FromResult<OllamaProcessReceipt?>(null);
        public Task StopAutoStartedAsync(OllamaProcessReceipt receipt, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, CancellationToken cancellationToken = default)
        {
            VerifiedPath = @"C:\OtherOllama\ollama.exe";
            return Task.FromResult(new VerifiedOllamaServer(VerifiedPath, new Version(0, 9, 1), publication.ReleaseIdentity));
        }
        public Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, string requiredBinaryPath, CancellationToken cancellationToken = default)
        {
            VerifiedPath = requiredBinaryPath;
            return Task.FromResult(new VerifiedOllamaServer(VerifiedPath, new Version(0, 9, 1), publication.ReleaseIdentity));
        }
        public Task StartServeAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default) { ServedPath = server.BinaryPath; return Task.CompletedTask; }
        public Task WaitForHealthyAndConfirmStoreAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task StopOwnedAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    }

    private sealed class BlockingAutoStartServerController : IOllamaServerController
    {
        public TaskCompletionSource StopEntered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource AllowAutoStartExit { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool ControlledServeStarted { get; private set; }
        public Task SnapshotBeforeInstallAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<OllamaProcessReceipt?> CaptureInstallerAutoStartAsync(VerifiedOllamaServer server, DateTime installerStartedUtc, DateTime installerFinishedUtc, CancellationToken cancellationToken = default)
            => Task.FromResult<OllamaProcessReceipt?>(new OllamaProcessReceipt(1, DateTime.UtcNow, 1, server.BinaryPath, server.ReleaseIdentity));
        public async Task StopAutoStartedAsync(OllamaProcessReceipt receipt, CancellationToken cancellationToken = default) { StopEntered.SetResult(); await AllowAutoStartExit.Task.WaitAsync(cancellationToken); }
        public Task<VerifiedOllamaServer> LocateAndVerifyAsync(OllamaPublication publication, CancellationToken cancellationToken = default)
            => Task.FromResult(new VerifiedOllamaServer(@"C:\Ollama\ollama.exe", new Version(0, 9, 1), publication.ReleaseIdentity));
        public Task StartServeAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default) { ControlledServeStarted = true; return Task.CompletedTask; }
        public Task WaitForHealthyAndConfirmStoreAsync(VerifiedOllamaServer server, OllamaInstallation installation, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task StopOwnedAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    }

    private sealed class ImportFiles : IDisposable
    {
        public ImportFiles(bool nestedModelfile = false)
        {
            Root = Path.Combine(Path.GetTempPath(), $"YouTuberOllama-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
            VerifierGguf = Path.Combine(Root, "qwen3-verifier.gguf");
            Gguf = Path.Combine(Root, "speaker.gguf");
            var modelfileDirectory = nestedModelfile ? Path.Combine(Root, "subdir") : Root;
            Directory.CreateDirectory(modelfileDirectory);
            var modelfile = Path.Combine(modelfileDirectory, "Modelfile");
            File.WriteAllText(VerifierGguf, "verifier");
            File.WriteAllText(Gguf, "gguf");
            File.WriteAllText(modelfile, "FROM ./speaker.gguf");
            Artifacts = new OllamaImportArtifacts(
                VerifierGguf, new FileInfo(VerifierGguf).Length, Hash(VerifierGguf),
                Gguf, new FileInfo(Gguf).Length, Hash(Gguf),
                modelfile, new FileInfo(modelfile).Length, Hash(modelfile));
        }

        public string Root { get; }
        public string VerifierGguf { get; }
        public string Gguf { get; }
        public OllamaImportArtifacts Artifacts { get; }

        public void Dispose() { if (Directory.Exists(Root)) Directory.Delete(Root, true); }

    }

    private static string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    private static string WorkerFixtureExecutable() => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "YouTuber.WorkerFixture", "bin", "Release", "net8.0-windows", "YouTuber.WorkerFixture.exe"));
    private static OllamaModelRolesMetadata ModelRoles(string speaker = "speaker-v5-a636") => new(
        new OllamaVerifierMetadata(
            "qwen3:4b",
            new ImmutableArtifactMetadata("owner/models", new string('1', 40), "qwen3-verifier.gguf", 8, "88c9eae68eb300b2971a2bec9e5a26ff4179fd661d6b7d861e4c6557b9aaee14"),
            "8888888888888888888888888888888888888888888888888888888888888888",
            4),
        new SpeakerModelMetadata(
            speaker,
            new ImmutableArtifactMetadata("owner/models", new string('1', 40), "speaker.gguf", 4, "1cb1b7e0f8b96cee3445e317b8064d8805bf35c7dc7de82cddcb9f78d4c95e0e"),
            new ImmutableArtifactMetadata("owner/models", new string('1', 40), "Modelfile", 19, "3e9ceb67dd6a3611c9f481d2acfda702b548b934f41ad8f48845e24d10ca2247"),
            5));
    private static OllamaPublication Publication(string installerPath, string version) => new(
        new InstallerArtifact(installerPath, new FileInfo(installerPath).Length, Hash(installerPath), OllamaInstallation.ExpectedPublisher),
        OllamaInstallation.ExpectedPublisher,
        version,
        $"ollama/ollama@v{version}",
        Hash(installerPath),
        ModelRoles());

    private sealed class TemporaryBinary : IDisposable
    {
        public TemporaryBinary() { Path = System.IO.Path.GetTempFileName(); }
        public string Path { get; }
        public void Dispose() => File.Delete(Path);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-ollama-lifecycle-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }
        public string Path { get; }
        public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
    }

    private sealed class LocalHttpServer : IAsyncDisposable
    {
        private readonly TcpListener _listener;
        private readonly Task _serve;
        private readonly Func<HttpRequestMessage, Task<HttpResponseMessage>> _response;
        private readonly CancellationTokenSource _stop = new();
        private readonly int _dropConnections;

        private LocalHttpServer(TcpListener listener, Func<HttpRequestMessage, Task<HttpResponseMessage>> response, int dropConnections)
        {
            _listener = listener;
            _response = response;
            _dropConnections = dropConnections;
            _serve = ServeAsync();
        }

        public Uri Endpoint { get; private init; } = null!;
        public int RequestCount { get; private set; }
        public int BodyBytes { get; private set; }

        public static Task<LocalHttpServer> StartAsync(
            Func<HttpRequestMessage, Task<HttpResponseMessage>> response,
            int port = 0,
            int dropConnections = 0)
        {
            var listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();
            return Task.FromResult(new LocalHttpServer(listener, response, dropConnections)
            {
                Endpoint = new Uri($"http://127.0.0.1:{((IPEndPoint)listener.LocalEndpoint).Port}/"),
            });
        }

        private async Task ServeAsync()
        {
            try
            {
                while (!_stop.IsCancellationRequested)
                {
                    using var connection = await _listener.AcceptTcpClientAsync(_stop.Token);
                    await using var stream = connection.GetStream();
                    using var reader = new StreamReader(stream, Encoding.ASCII, leaveOpen: true);
                    var requestLine = await reader.ReadLineAsync(_stop.Token) ?? throw new InvalidOperationException("Missing HTTP request line.");
                    var contentLength = 0;
                    string? header;
                    while (!string.IsNullOrEmpty(header = await reader.ReadLineAsync(_stop.Token)))
                    {
                        if (header.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase)) contentLength = int.Parse(header[15..].Trim());
                    }
                    var body = new char[contentLength];
                    var read = 0;
                    while (read < body.Length) read += await reader.ReadAsync(body.AsMemory(read), _stop.Token);
                    RequestCount++;
                    BodyBytes += Encoding.UTF8.GetByteCount(body);
                    if (RequestCount <= _dropConnections) continue;
                    using var request = new HttpRequestMessage { RequestUri = new Uri(Endpoint, requestLine.Split(' ')[1]) };
                    using var response = await _response(request);
                    var payload = response.Content is null ? string.Empty : await response.Content.ReadAsStringAsync(_stop.Token);
                    var headers = $"HTTP/1.1 {(int)response.StatusCode} {response.ReasonPhrase ?? response.StatusCode.ToString()}\r\nContent-Length: {Encoding.UTF8.GetByteCount(payload)}\r\nConnection: close\r\n";
                    if (response.Headers.Location is not null) headers += $"Location: {response.Headers.Location}\r\n";
                    headers += "\r\n";
                    await stream.WriteAsync(Encoding.ASCII.GetBytes(headers), _stop.Token);
                    await stream.WriteAsync(Encoding.UTF8.GetBytes(payload), _stop.Token);
                }
            }
            catch (OperationCanceledException) { }
        }

        public async ValueTask DisposeAsync()
        {
            _stop.Cancel();
            _listener.Stop();
            await _serve;
            _stop.Dispose();
        }
    }
}
