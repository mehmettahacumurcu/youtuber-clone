using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;

namespace YouTuber.Launcher.SystemChecks;

public sealed class PreflightEvaluator
{
    public PreflightReport Evaluate(SystemProfile profile, DistributionManifest manifest, AppPaths paths)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(paths);

        var checks = new List<PreflightCheck>();
        var minimumBuild = ParseWindowsBuild(manifest.Minimum.Windows);
        checks.Add(BuildCheck(profile.WindowsBuild, minimumBuild));
        checks.Add(NvidiaCheck(profile.Nvidia));
        checks.Add(VramCheck(profile.Nvidia, manifest.Minimum.VramBytes));
        checks.Add(PhysicalMemoryCheck(profile.PhysicalMemoryBytes, manifest.Minimum.RamBytes));

        var dataDrive = DiskProbe.DriveRoot(paths.DataRoot);
        var programDrive = DiskProbe.DriveRoot(paths.ProgramRoot);
        var ollamaDrive = DiskProbe.DriveRoot(profile.OllamaStoreDriveRoot);
        var requirements = DiskProbe.CalculateRequirements(manifest, dataDrive, programDrive, ollamaDrive, profile.Prerequisites).PerDrive;
        checks.Add(DriveSpaceCheck("Data drive space", dataDrive, requirements, profile.AvailableBytesByDrive));
        if (string.Equals(programDrive, ollamaDrive, StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(dataDrive, programDrive, StringComparison.OrdinalIgnoreCase))
        {
            checks.Add(DriveSpaceCheck("Program and Ollama drive space", programDrive, requirements, profile.AvailableBytesByDrive));
        }
        else
        {
            if (!string.Equals(dataDrive, programDrive, StringComparison.OrdinalIgnoreCase) && HasRequirement(programDrive, requirements))
            {
                checks.Add(DriveSpaceCheck("Program drive space", programDrive, requirements, profile.AvailableBytesByDrive));
            }
            if (!string.Equals(dataDrive, ollamaDrive, StringComparison.OrdinalIgnoreCase) && HasRequirement(ollamaDrive, requirements))
            {
                checks.Add(DriveSpaceCheck("Ollama drive space", ollamaDrive, requirements, profile.AvailableBytesByDrive));
            }
        }

        checks.Add(PrerequisiteCheck("Ollama", profile.Prerequisites.OllamaAvailable, profile.Prerequisites.OllamaDiagnostic, "Install Ollama, then return to this launcher and run the preflight again."));
        checks.Add(PrerequisiteCheck("WebView2", profile.Prerequisites.WebView2Available, profile.Prerequisites.WebView2Diagnostic, "Install the Microsoft Edge WebView2 Runtime, then run the preflight again."));
        return new PreflightReport(checks);
    }

    private static PreflightCheck BuildCheck(int actual, int required) => actual >= required
        ? new("Windows", PreflightStatus.Pass, actual, required, "Windows meets the minimum build requirement.")
        : new("Windows", PreflightStatus.Block, actual, required, "Update Windows to version 10 22H2 (build 19045) or newer.");

    private static PreflightCheck NvidiaCheck(NvidiaProbeResult nvidia) => nvidia.IsAvailable
        ? new("NVIDIA GPU", PreflightStatus.Pass, nvidia.Gpus.Count, null, "NVIDIA GPU detection succeeded: " + nvidia.Diagnostic)
        : new("NVIDIA GPU", PreflightStatus.Block, null, null, "Install or repair the NVIDIA driver. " + nvidia.Diagnostic);

    private static PreflightCheck VramCheck(NvidiaProbeResult nvidia, long required)
    {
        var actual = nvidia.SelectedGpu?.VramBytes;
        if (actual is null)
        {
            return new PreflightCheck("NVIDIA VRAM", PreflightStatus.Block, null, required, "A healthy NVIDIA driver and GPU with sufficient VRAM are required.");
        }

        return actual >= required
            ? new("NVIDIA VRAM", PreflightStatus.Pass, actual, required, "Detected NVIDIA VRAM meets the release requirement.")
            : new("NVIDIA VRAM", PreflightStatus.Block, actual, required, "Use an NVIDIA GPU with at least the required VRAM; product names are not used for compatibility decisions.");
    }

    private static PreflightCheck PhysicalMemoryCheck(long actual, long required) => actual >= required
        ? new("Physical RAM", PreflightStatus.Pass, actual, required, "Physical memory meets the release requirement.")
        : new("Physical RAM", PreflightStatus.Block, actual, required, "Install at least the required physical RAM before running YouTuber Studio.");

    private static PreflightCheck DriveSpaceCheck(
        string name,
        string drive,
        IReadOnlyList<DriveSpaceRequirement> requirements,
        IReadOnlyDictionary<string, long> availableBytes)
    {
        var requirement = requirements.Single(item => string.Equals(item.DriveRoot, drive, StringComparison.OrdinalIgnoreCase));
        var actual = availableBytes.TryGetValue(drive, out var freeBytes) ? freeBytes : -1;
        return actual >= requirement.RequiredBytes
            ? new(name, PreflightStatus.Pass, actual, requirement.RequiredBytes, $"{drive} has sufficient free space.")
            : new(name, PreflightStatus.Block, actual, requirement.RequiredBytes, $"Free space on {drive} or select a different local NTFS data drive before downloading components.");
    }

    private static bool HasRequirement(string drive, IReadOnlyList<DriveSpaceRequirement> requirements) =>
        requirements.Any(item => string.Equals(item.DriveRoot, drive, StringComparison.OrdinalIgnoreCase));

    private static PreflightCheck PrerequisiteCheck(string name, bool available, string diagnostic, string action) => available
        ? new(name, PreflightStatus.Pass, null, null, string.IsNullOrWhiteSpace(diagnostic) ? $"{name} is available." : diagnostic)
        : new(name, PreflightStatus.Warning, null, null, action);

    private static int ParseWindowsBuild(string windows)
    {
        var parts = windows.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length != 3 || !int.TryParse(parts[2], out var build) || build <= 0)
        {
            throw new InvalidOperationException("The release manifest has an invalid minimum Windows version.");
        }

        return build;
    }
}
