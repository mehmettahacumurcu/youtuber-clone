using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using System.IO;

namespace YouTuber.Launcher.SystemChecks;

public sealed record DriveSpaceRequirement(string DriveRoot, long RequiredBytes);

public sealed record DiskRequirements(IReadOnlyList<DriveSpaceRequirement> PerDrive);

public sealed class DiskProbe
{
    private const long TwoGigabytes = 2_000_000_000;

    public IReadOnlyDictionary<string, long> GetAvailableBytes(AppPaths paths, string ollamaStoreDriveRoot)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var roots = new[] { DriveRoot(paths.DataRoot), DriveRoot(paths.ProgramRoot), NormalizeDriveRoot(ollamaStoreDriveRoot) }
            .Distinct(StringComparer.OrdinalIgnoreCase);
        var available = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        foreach (var root in roots)
        {
            available[root] = new DriveInfo(root).AvailableFreeSpace;
        }

        return available;
    }

    public static DiskRequirements CalculateRequirements(
        DistributionManifest manifest,
        string dataDriveRoot,
        string programDriveRoot,
        string ollamaStoreDriveRoot,
        PrerequisiteStatus prerequisites)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(prerequisites);
        var dataRoot = NormalizeDriveRoot(dataDriveRoot);
        var programRoot = NormalizeDriveRoot(programDriveRoot);
        var ollamaRoot = NormalizeDriveRoot(ollamaStoreDriveRoot);
        var perDriveBase = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase)
        {
            [dataRoot] = 0,
        };

        foreach (var component in manifest.Components)
        {
            var componentRequirement = checked(
                checked(component.Value.Size + component.Value.Size) +
                component.Value.ExpandedSize +
                checked(component.Value.InstallSize + component.Value.InstallSize) +
                component.Value.PeakSpace);
            AddRequirement(perDriveBase, dataRoot, componentRequirement);
        }

        var ollama = manifest.Ollama ?? throw new ManifestValidationException("Ollama footprint metadata is required.");
        var webView2 = manifest.WebView2 ?? throw new ManifestValidationException("WebView2 footprint metadata is required.");
        if (!prerequisites.OllamaAvailable) AddInstaller(perDriveBase, dataRoot, programRoot, ollama.Installer);
        if (!prerequisites.WebView2Available) AddInstaller(perDriveBase, dataRoot, programRoot, webView2.Installer);

        var verifier = ollama.Models.Verifier ?? throw new ManifestValidationException("Verifier footprint metadata is required.");
        var speaker = ollama.Models.Speaker ?? throw new ManifestValidationException("Speaker footprint metadata is required.");
        var acquisitionBytes = checked(
            checked(checked(speaker.Gguf.Size + speaker.Gguf.Size) + checked(speaker.Modelfile.Size + speaker.Modelfile.Size)) +
            checked(verifier.Source.Size + verifier.Source.Size));
        AddRequirement(perDriveBase, dataRoot, acquisitionBytes);

        // Ollama may need an import/work copy while retaining current and rollback versions.
        var ollamaModelBytes = checked(
            checked(checked(speaker.InstalledSize + speaker.InstalledSize) + speaker.InstalledSize) +
            checked(checked(verifier.InstalledSize + verifier.InstalledSize) + verifier.InstalledSize));
        AddRequirement(perDriveBase, ollamaRoot, ollamaModelBytes);

        var minimumDataRequirement = Math.Max(40_000_000_000, manifest.Minimum.DiskBytes);
        var result = new List<DriveSpaceRequirement>();
        foreach (var (drive, baseRequirement) in perDriveBase)
        {
            var withMargin = AddSafetyMargin(baseRequirement);
            var required = string.Equals(drive, dataRoot, StringComparison.OrdinalIgnoreCase)
                ? Math.Max(minimumDataRequirement, Math.Max(FindLargestComponentPeak(manifest), withMargin))
                : withMargin;
            result.Add(new DriveSpaceRequirement(drive, required));
        }

        return new DiskRequirements(result.OrderBy(requirement => requirement.DriveRoot, StringComparer.OrdinalIgnoreCase).ToArray());
    }

    public static string DriveRoot(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return NormalizeDriveRoot(Path.GetPathRoot(Path.GetFullPath(path)) ?? throw new ArgumentException("Path has no drive root.", nameof(path)));
    }

    private static long FindLargestComponentPeak(DistributionManifest manifest) => manifest.Components.Values.Aggregate(
        0L,
        (largest, component) => Math.Max(largest, component.PeakSpace));

    private static void AddInstaller(
        Dictionary<string, long> requirements,
        string dataRoot,
        string programRoot,
        SignedInstallerMetadata installer)
    {
        AddRequirement(requirements, dataRoot, checked(installer.Size + installer.Size));
        AddRequirement(requirements, programRoot, installer.InstallSize);
    }

    private static void AddRequirement(Dictionary<string, long> requirements, string driveRoot, long bytes)
    {
        if (bytes < 0) throw new OverflowException("Disk-space accounting produced a negative byte count.");
        requirements[driveRoot] = checked(requirements.GetValueOrDefault(driveRoot) + bytes);
    }

    private static long AddSafetyMargin(long baseRequirement)
    {
        var fivePercent = checked((checked(baseRequirement * 5) + 99) / 100);
        return checked(checked(baseRequirement + fivePercent) + TwoGigabytes);
    }

    private static string NormalizeDriveRoot(string driveRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(driveRoot);
        var root = Path.GetPathRoot(driveRoot);
        if (string.IsNullOrEmpty(root))
        {
            throw new ArgumentException("A local drive root is required.", nameof(driveRoot));
        }

        return Path.TrimEndingDirectorySeparator(root) + Path.DirectorySeparatorChar;
    }
}
