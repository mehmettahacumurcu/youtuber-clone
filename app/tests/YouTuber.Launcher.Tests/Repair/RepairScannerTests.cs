using System.Security.Cryptography;
using System.Text.Json;
using YouTuber.Launcher.Activation;
using YouTuber.Launcher.Repair;
using Xunit;

namespace YouTuber.Launcher.Tests.Repair;

public sealed class RepairScannerTests
{
    [Fact]
    public async Task Only_missing_or_corrupt_inventoried_files_are_selected_for_repair()
    {
        using var directory = new TemporaryDirectory();
        var good = System.Text.Encoding.UTF8.GetBytes("good");
        var expected = System.Text.Encoding.UTF8.GetBytes("expected");
        File.WriteAllBytes(Path.Combine(directory.Path, "worker.exe"), good);
        File.WriteAllText(Path.Combine(directory.Path, "corrupt.bin"), "wrong");
        var inventory = Inventory(
            FileEntry("worker.exe", good),
            FileEntry("missing.bin", expected),
            FileEntry("corrupt.bin", expected));
        WriteManifest(directory.Path, inventory);

        var findings = await new RepairScanner().ScanAsync(directory.Path, inventory);

        Assert.Equal(["corrupt.bin", "missing.bin"], findings.Select(finding => finding.Path).Order(StringComparer.Ordinal));
    }

    [Fact]
    public async Task Reparse_points_are_never_followed_during_repair_hashing()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var directory = new TemporaryDirectory();
        using var outside = new TemporaryDirectory();
        File.WriteAllText(Path.Combine(outside.Path, "outside.bin"), "outside");
        try { Directory.CreateSymbolicLink(Path.Combine(directory.Path, "linked"), outside.Path); }
        catch (UnauthorizedAccessException) { return; }
        var bytes = System.Text.Encoding.UTF8.GetBytes("outside");
        var inventory = Inventory(FileEntry("worker.exe", [1]), FileEntry("linked/outside.bin", bytes));
        File.WriteAllBytes(Path.Combine(directory.Path, "worker.exe"), [1]);
        WriteManifest(directory.Path, inventory);

        var findings = await new RepairScanner().ScanAsync(directory.Path, inventory);

        Assert.Contains(findings, finding => finding.Path == "linked/outside.bin" && finding.Reason == RepairReason.UnsafePath);
    }

    [Fact]
    public async Task Reparse_point_ancestors_above_the_component_root_are_never_followed()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var directory = new TemporaryDirectory();
        using var outside = new TemporaryDirectory();
        var component = Directory.CreateDirectory(Path.Combine(outside.Path, "component")).FullName;
        File.WriteAllBytes(Path.Combine(component, "worker.exe"), [1]);
        var linked = Path.Combine(directory.Path, "linked");
        try { Directory.CreateSymbolicLink(linked, outside.Path); }
        catch (Exception exception) when (exception is UnauthorizedAccessException or IOException) { return; }

        var inventory = Inventory(FileEntry("worker.exe", [1]));
        WriteManifest(component, inventory);
        var findings = await new RepairScanner().ScanAsync(Path.Combine(linked, "component"), inventory);

        Assert.Contains(findings, finding => finding.Path == "worker.exe" && finding.Reason == RepairReason.UnsafePath);
    }

    [Fact]
    public async Task Extra_files_and_unlisted_directories_make_the_component_unhealthy()
    {
        using var directory = new TemporaryDirectory();
        File.WriteAllBytes(Path.Combine(directory.Path, "worker.exe"), [1]);
        File.WriteAllText(Path.Combine(directory.Path, "injected.dll"), "injected");
        Directory.CreateDirectory(Path.Combine(directory.Path, "unlisted"));

        var inventory = Inventory(FileEntry("worker.exe", [1]));
        WriteManifest(directory.Path, inventory);
        var findings = await new RepairScanner().ScanAsync(directory.Path, inventory);

        Assert.Contains(findings, finding => finding.Path == "injected.dll" && finding.Reason == RepairReason.Unexpected);
        Assert.Contains(findings, finding => finding.Path == "unlisted" && finding.Reason == RepairReason.Unexpected);
    }

    [Fact]
    public async Task Missing_or_mismatched_on_disk_inventory_is_never_reported_healthy()
    {
        using var directory = new TemporaryDirectory();
        var inventory = Inventory(FileEntry("worker.exe", [1]));
        File.WriteAllBytes(Path.Combine(directory.Path, "worker.exe"), [1]);

        var missing = await new RepairScanner().ScanAsync(directory.Path, inventory);
        WriteManifest(directory.Path, inventory with { Version = "2.0.0" });
        var mismatched = await new RepairScanner().ScanAsync(directory.Path, inventory);

        Assert.Contains(missing, finding => finding.Path == "component-manifest.json" && finding.Reason == RepairReason.Missing);
        Assert.Contains(mismatched, finding => finding.Path == "component-manifest.json" && finding.Reason == RepairReason.Corrupt);
    }

    private static ComponentInventory Inventory(params ComponentFile[] files)
        => new(ComponentInventory.ExpectedSchema, "voice_runtime", "1.0.0", new ComponentCompatibility(1), "worker.exe", ["--healthcheck"], files);

    private static ComponentFile FileEntry(string path, byte[] bytes)
        => new(path, bytes.Length, Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant());

    private static void WriteManifest(string root, ComponentInventory inventory)
    {
        var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
        File.WriteAllText(Path.Combine(root, "component-manifest.json"), JsonSerializer.Serialize(inventory, options));
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory() { Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-repair-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(Path); }
        public string Path { get; }
        public void Dispose() { try { Directory.Delete(Path, true); } catch { } }
    }
}
