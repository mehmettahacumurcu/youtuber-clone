using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.SystemChecks;
using Xunit;

namespace YouTuber.Launcher.Tests.SystemChecks;

public sealed class PreflightEvaluatorTests
{
    private const long MinimumVram = 12_000_000_000;
    private const long MinimumRam = 32_000_000_000;
    private const long FortyGb = 40_000_000_000;

    [Theory]
    [InlineData(19045, PreflightStatus.Pass)]
    [InlineData(19044, PreflightStatus.Block)]
    public void EvaluateEnforcesTheExactMinimumWindowsBuild(int build, PreflightStatus expectedStatus)
    {
        var paths = Paths();
        var report = new PreflightEvaluator().Evaluate(Profile(paths, windowsBuild: build), Manifest(), paths);

        Assert.Equal(expectedStatus, report.Checks[0].Status);
        Assert.Equal("Windows", report.Checks[0].Name);
    }

    [Theory]
    [InlineData(MinimumVram, PreflightStatus.Pass)]
    [InlineData(MinimumVram - 1, PreflightStatus.Block)]
    public void EvaluateEnforcesTheExactMinimumVram(long vram, PreflightStatus expectedStatus)
    {
        var paths = Paths();
        var report = new PreflightEvaluator().Evaluate(Profile(paths, vramBytes: vram), Manifest(), paths);

        var check = Assert.Single(report.Checks, check => check.Name == "NVIDIA VRAM");
        Assert.Equal(expectedStatus, check.Status);
    }

    [Fact]
    public void EvaluateAcceptsTheExactMinimumPhysicalRam()
    {
        var paths = Paths();
        var report = new PreflightEvaluator().Evaluate(Profile(paths, ramBytes: MinimumRam), Manifest(), paths);

        var check = Assert.Single(report.Checks, check => check.Name == "Physical RAM");
        Assert.Equal(PreflightStatus.Pass, check.Status);
        Assert.Equal(MinimumRam, check.ActualValue);
    }

    [Fact]
    public void EvaluateBlocksWhenDataDriveIsBelowTheGreaterOfFortyGbAndManifestPeak()
    {
        const long manifestPeak = 50_000_000_000;
        var paths = Paths();
        var profile = Profile(paths, dataDriveFreeBytes: manifestPeak - 1);
        var manifest = Manifest(diskBytes: manifestPeak, componentPeakSpace: 1);

        var report = new PreflightEvaluator().Evaluate(profile, manifest, paths);

        var check = Assert.Single(report.Checks, check => check.Name == "Data drive space");
        Assert.Equal(PreflightStatus.Block, check.Status);
        Assert.Equal(manifestPeak, check.RequiredValue);
        Assert.Equal(manifestPeak - 1, check.ActualValue);
    }

    [Fact]
    public void EvaluateIncludesASeparateOllamaDriveRequirementWhenTheStoreIsOnAnotherDrive()
    {
        var paths = Paths();
        var profile = Profile(
            paths,
            dataDriveFreeBytes: 200_000_000_000,
            ollamaStorePath: @"D:\",
            ollamaDriveFreeBytes: 1);
        var report = new PreflightEvaluator().Evaluate(profile, Manifest(includeLlm: true), paths);

        var check = Assert.Single(report.Checks, check => check.Name == "Ollama drive space");
        Assert.Equal(PreflightStatus.Block, check.Status);
        Assert.Equal(1, check.ActualValue);
        Assert.True(check.RequiredValue > 1);
    }

    [Fact]
    public void EvaluateStillReportsTheExistingOllamaDriveWhenTheManifestHasNoLlmComponent()
    {
        var paths = Paths();
        var profile = Profile(
            paths,
            dataDriveFreeBytes: 200_000_000_000,
            ollamaStorePath: @"D:\",
            ollamaDriveFreeBytes: 200_000_000_000);

        var report = new PreflightEvaluator().Evaluate(profile, Manifest(), paths);

        Assert.Equal(PreflightStatus.Pass, Assert.Single(report.Checks, check => check.Name == "Ollama drive space").Status);
    }

    [Fact]
    public void EvaluateBlocksTheDataDriveForLlmDownloadAndStagingWhenOllamaUsesAnotherDrive()
    {
        var paths = Paths();
        const long dataFreeBytes = 40_000_000_000;
        const long ollamaFreeBytes = 200_000_000_000;
        var profile = Profile(
            paths,
            dataDriveFreeBytes: dataFreeBytes,
            ollamaStorePath: @"D:\",
            ollamaDriveFreeBytes: ollamaFreeBytes);

        var report = new PreflightEvaluator().Evaluate(
            profile,
            Manifest(diskBytes: 1, componentPeakSpace: 1_000_000_000, componentSize: 20_000_000_000, componentInstallSize: 20_000_000_000, includeVoice: false, includeLlm: true),
            paths);

        var data = Assert.Single(report.Checks, check => check.Name == "Data drive space");
        var ollama = Assert.Single(report.Checks, check => check.Name == "Ollama drive space");
        Assert.Equal(PreflightStatus.Block, data.Status);
        Assert.Equal(dataFreeBytes, data.ActualValue);
        Assert.True(data.RequiredValue > dataFreeBytes);
        Assert.Equal(PreflightStatus.Pass, ollama.Status);
        Assert.Equal(ollamaFreeBytes, ollama.ActualValue);
        Assert.Equal(DiskProbe.DriveRoot(paths.DataRoot), profile.DataDriveRoot);
        Assert.Equal(DiskProbe.DriveRoot(@"D:\"), profile.OllamaStoreDriveRoot);
    }

    [Fact]
    public void EvaluateKeepsChecksOrderedAndMakesEveryFailureActionable()
    {
        var paths = Paths();
        var report = new PreflightEvaluator().Evaluate(
            Profile(paths, windowsBuild: 19044, nvidia: NvidiaProbeResult.Failure("nvidia-smi was not found.")),
            Manifest(),
            paths);

        Assert.Equal(["Windows", "NVIDIA GPU", "NVIDIA VRAM", "Physical RAM", "Data drive space", "Ollama", "WebView2"], report.Checks.Select(check => check.Name));
        Assert.All(report.Checks, check => Assert.False(string.IsNullOrWhiteSpace(check.Action)));
        Assert.Equal(PreflightStatus.Block, report.OverallStatus);
    }

    [Fact]
    public void CalculateDiskRequirementsIncludesDownloadsArchivesExtractionActivationAndSafetyMargin()
    {
        var requirements = DiskProbe.CalculateRequirements(Manifest(
            diskBytes: 1,
            componentPeakSpace: 1_000_000_000,
            componentSize: 5_000_000_000,
            componentExpandedSize: 7_000_000_000,
            componentInstallSize: 10_000_000_000),
            DiskProbe.DriveRoot(@"C:\"),
            DiskProbe.DriveRoot(@"C:\"),
            DiskProbe.DriveRoot(@"C:\"),
            new PrerequisiteStatus(true, true));

        var data = Assert.Single(requirements.PerDrive);
        Assert.Equal(DiskProbe.DriveRoot(@"C:\"), data.DriveRoot);
        Assert.Equal(41_900_000_013, data.RequiredBytes);
    }

    [Fact]
    public void CalculateDiskRequirementsAccountsForPrerequisitesAndOllamaModelsOnTheirActualDrives()
    {
        var requirements = DiskProbe.CalculateRequirements(
            Manifest(
                diskBytes: 1,
                componentPeakSpace: 4_000_000_000,
                componentSize: 1_000_000_000,
                componentExpandedSize: 2_000_000_000,
                componentInstallSize: 3_000_000_000,
                installerSize: 1_000_000_000,
                installerInstallSize: 2_000_000_000,
                modelSourceSize: 1_000_000_000,
                modelInstallSize: 2_000_000_000),
            DiskProbe.DriveRoot(@"C:\"),
            DiskProbe.DriveRoot(@"D:\"),
            DiskProbe.DriveRoot(@"E:\"),
            new PrerequisiteStatus(false, false));

        Assert.Collection(
            requirements.PerDrive,
            data =>
            {
                Assert.Equal(DiskProbe.DriveRoot(@"C:\"), data.DriveRoot);
                Assert.Equal(40_000_000_000, data.RequiredBytes);
            },
            program =>
            {
                Assert.Equal(DiskProbe.DriveRoot(@"D:\"), program.DriveRoot);
                Assert.Equal(6_200_000_000, program.RequiredBytes);
            },
            ollama =>
            {
                Assert.Equal(DiskProbe.DriveRoot(@"E:\"), ollama.DriveRoot);
                Assert.Equal(14_600_000_000, ollama.RequiredBytes);
            });
    }

    [Fact]
    public void CalculateDiskRequirementsCoalescesEveryCopyWhenAllStorageUsesOneDrive()
    {
        var drive = DiskProbe.DriveRoot(@"C:\");
        var requirements = DiskProbe.CalculateRequirements(
            Manifest(
                diskBytes: 1,
                componentPeakSpace: 4_000_000_000,
                componentSize: 1_000_000_000,
                componentExpandedSize: 2_000_000_000,
                componentInstallSize: 3_000_000_000,
                installerSize: 1_000_000_000,
                installerInstallSize: 2_000_000_000,
                modelSourceSize: 1_000_000_000,
                modelInstallSize: 2_000_000_000),
            drive,
            drive,
            drive,
            new PrerequisiteStatus(false, false));

        Assert.Equal(44_000_000_000, Assert.Single(requirements.PerDrive).RequiredBytes);
    }

    [Fact]
    public void CalculateDiskRequirementsFailsClosedOnOverflow()
    {
        Assert.Throws<OverflowException>(() => DiskProbe.CalculateRequirements(Manifest(
            diskBytes: long.MaxValue,
            componentPeakSpace: long.MaxValue,
            componentSize: long.MaxValue,
            componentExpandedSize: long.MaxValue,
            componentInstallSize: long.MaxValue),
            DiskProbe.DriveRoot(@"C:\"),
            DiskProbe.DriveRoot(@"C:\"),
            DiskProbe.DriveRoot(@"C:\"),
            new PrerequisiteStatus(false, false)));
    }

    private static AppPaths Paths() => AppPaths.ForBaseDirectory(@"C:\");

    private static SystemProfile Profile(
        AppPaths paths,
        int windowsBuild = 19045,
        long vramBytes = MinimumVram,
        long ramBytes = MinimumRam,
        long dataDriveFreeBytes = 200_000_000_000,
        string? ollamaStorePath = null,
        long? ollamaDriveFreeBytes = null,
        NvidiaProbeResult? nvidia = null)
    {
        var dataDrive = DiskProbe.DriveRoot(paths.DataRoot);
        var ollamaDrive = DiskProbe.DriveRoot(ollamaStorePath ?? dataDrive);
        return new(
            windowsBuild,
            nvidia ?? NvidiaProbeResult.Success([new NvidiaGpu("NVIDIA GPU", vramBytes, "555.85")]),
            ramBytes,
            new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase)
            {
                [dataDrive] = dataDriveFreeBytes,
                [ollamaDrive] = ollamaDriveFreeBytes ?? dataDriveFreeBytes,
            },
            dataDrive,
            ollamaDrive,
            new PrerequisiteStatus(true, true));
    }

    private static DistributionManifest Manifest(
        long diskBytes = FortyGb,
        long componentPeakSpace = 1,
        long componentSize = 1,
        long componentExpandedSize = 1,
        long componentInstallSize = 1,
        long installerSize = 1,
        long installerInstallSize = 1,
        long modelSourceSize = 1,
        long modelInstallSize = 1,
        bool includeVoice = true,
        bool includeLlm = false)
    {
        var components = new Dictionary<string, DistributionComponent>();
        if (includeVoice)
        {
            components["voice_runtime"] = new("owner/repo", new string('a', 40), "runtime.zip", componentSize, new string('b', 64), "runtime", componentExpandedSize, componentInstallSize, componentPeakSpace, "runtime.exe", ["--healthcheck"], new Compatibility(1), ["LICENSE"]);
        }

        if (includeLlm)
        {
            components["llm"] = new("owner/repo", new string('c', 40), "model.gguf", componentSize, new string('d', 64), "model", componentExpandedSize, componentInstallSize, componentPeakSpace, "model.gguf", ["--healthcheck"], new Compatibility(1), ["LICENSE"]);
        }

        var artifact = new ImmutableArtifactMetadata("owner/models", new string('e', 40), "model.gguf", modelSourceSize, new string('f', 64));
        return new DistributionManifest(
            "youtuber.distribution.v1",
            "1.0.0",
            new MinimumRequirements("10.0.19045", MinimumVram, MinimumRam, diskBytes, 1),
            components)
        {
            Ollama = new OllamaPublicationMetadata(
                new SignedInstallerMetadata("owner/ollama", new string('1', 40), "ollama.exe", installerSize, new string('2', 64), "Ollama, Inc.", installerInstallSize),
                "1.0.0",
                "ollama-1.0.0",
                new OllamaModelRolesMetadata(
                    new OllamaVerifierMetadata("qwen3:4b", artifact, new string('3', 64), modelInstallSize),
                    new SpeakerModelMetadata("speaker-v5-a636", artifact, artifact, modelInstallSize))),
            WebView2 = new WebView2PublicationMetadata(
                new SignedInstallerMetadata("owner/webview2", new string('4', 40), "webview2.exe", installerSize, new string('5', 64), "Microsoft Corporation", installerInstallSize)),
        };
    }
}
