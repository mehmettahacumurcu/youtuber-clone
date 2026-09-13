using System.Text.Json;
using System.Security.AccessControl;
using System.Security.Principal;
using YouTuber.Launcher.Configuration;
using Xunit;

namespace YouTuber.Launcher.Tests.Configuration;

public sealed class DataRootLocatorTests
{
    [Fact]
    public async Task Persists_canonical_custom_root_outside_data_tree_without_creating_it()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var selected = Path.Combine(directory.Path, "selected", "youtuber-data");
        var locator = DataRootLocator.ForBaseDirectory(localAppData);

        await locator.SaveAsync(selected);

        Assert.False(Directory.Exists(selected));
        Assert.Equal(Path.GetFullPath(selected), locator.LoadRequired());
        Assert.StartsWith(Path.GetFullPath(localAppData), locator.LocatorFile, StringComparison.OrdinalIgnoreCase);
        Assert.False(IsWithin(selected, locator.LocatorFile));
    }

    [Fact]
    public async Task No_cli_restart_resolves_the_same_persisted_custom_root()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var selected = Path.Combine(directory.Path, "custom-data");

        await DataRootLocator.ForBaseDirectory(localAppData).SaveAsync(selected);

        var restarted = DataRootLocator.ForBaseDirectory(localAppData);
        Assert.Equal(Path.GetFullPath(selected), restarted.LoadRequired());
    }

    [Fact]
    public async Task Valid_override_replaces_locator_only_after_validation()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var first = Path.Combine(directory.Path, "first-data");
        var second = Path.Combine(directory.Path, "second-data");
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        await locator.SaveAsync(first);

        await locator.SaveAsync(second);

        Assert.Equal(Path.GetFullPath(second), locator.LoadRequired());
    }

    [Fact]
    public async Task Invalid_override_does_not_replace_existing_locator()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var first = Path.Combine(directory.Path, "first-data");
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        await locator.SaveAsync(first);

        await Assert.ThrowsAsync<ArgumentException>(() => locator.SaveAsync(Path.GetPathRoot(first)!));

        Assert.Equal(Path.GetFullPath(first), locator.LoadRequired());
    }

    [Fact]
    public void Missing_locator_is_distinct_from_corrupt_locator()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        Assert.Null(locator.TryLoad());

        Directory.CreateDirectory(Path.GetDirectoryName(locator.LocatorFile)!);
        File.WriteAllText(locator.LocatorFile, "{not-json");

        Assert.Throws<InvalidDataException>(() => locator.TryLoad());
        Assert.Throws<InvalidDataException>(() => locator.LoadRequired());
    }

    [Theory]
    [InlineData("relative-data")]
    [InlineData("C:\\")]
    public async Task Relative_or_filesystem_root_selection_is_rejected(string selected)
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);

        await Assert.ThrowsAsync<ArgumentException>(() => DataRootLocator.ForBaseDirectory(localAppData).SaveAsync(selected));
    }

    [Fact]
    public async Task Selection_cannot_contain_or_be_contained_by_fixed_bootstrap_root()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var locator = DataRootLocator.ForBaseDirectory(localAppData);

        await Assert.ThrowsAsync<ArgumentException>(() => locator.SaveAsync(localAppData));
        await Assert.ThrowsAsync<ArgumentException>(() => locator.SaveAsync(Path.Combine(locator.BootstrapRoot, "data")));
    }

    [Fact]
    public void Noncanonical_or_unknown_locator_content_fails_closed()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        Directory.CreateDirectory(Path.GetDirectoryName(locator.LocatorFile)!);
        var selected = Path.Combine(directory.Path, "selected-data") + Path.DirectorySeparatorChar;
        File.WriteAllText(locator.LocatorFile, JsonSerializer.Serialize(new
        {
            schema = "youtuber.data-root.v1",
            data_root = selected,
            ignored = true,
        }));

        Assert.Throws<InvalidDataException>(() => locator.TryLoad());
    }

    [Fact]
    public void Unsafe_persisted_filesystem_root_fails_closed_instead_of_falling_back()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        Directory.CreateDirectory(Path.GetDirectoryName(locator.LocatorFile)!);
        File.WriteAllText(locator.LocatorFile, JsonSerializer.Serialize(new
        {
            schema = "youtuber.data-root.v1",
            data_root = Path.GetPathRoot(directory.Path),
        }));

        Assert.Throws<InvalidDataException>(() => locator.LoadRequired());
    }

    [Fact]
    public async Task Bootstrap_path_replaced_by_reparse_point_after_construction_fails_before_following_it()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var directory = new TemporaryDirectory();
        using var outside = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        try { Directory.CreateSymbolicLink(locator.BootstrapRoot, outside.Path); }
        catch (UnauthorizedAccessException) { return; }

        Assert.Throws<InvalidDataException>(() => locator.TryLoad());
        await Assert.ThrowsAsync<InvalidDataException>(() => locator.SaveAsync(Path.Combine(directory.Path, "selected-data")));
        Assert.False(Directory.Exists(Path.Combine(outside.Path, "state")));
    }

    [Fact]
    public async Task Locator_directory_and_file_have_explicit_current_user_only_acl()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var directory = new TemporaryDirectory();
        var localAppData = Path.Combine(directory.Path, "local-app-data");
        Directory.CreateDirectory(localAppData);
        var locator = DataRootLocator.ForBaseDirectory(localAppData);

        await locator.SaveAsync(Path.Combine(directory.Path, "selected-data"));

        var currentUser = WindowsIdentity.GetCurrent().User ?? throw new InvalidOperationException("Current user SID is unavailable.");
        AssertRestrictedToCurrentUser(new DirectoryInfo(locator.BootstrapRoot).GetAccessControl(), currentUser);
        AssertRestrictedToCurrentUser(new DirectoryInfo(Path.GetDirectoryName(locator.LocatorFile)!).GetAccessControl(), currentUser);
        AssertRestrictedToCurrentUser(new FileInfo(locator.LocatorFile).GetAccessControl(), currentUser);
    }

    private static void AssertRestrictedToCurrentUser(FileSystemSecurity security, SecurityIdentifier currentUser)
    {
        Assert.True(security.AreAccessRulesProtected);
        var rules = security.GetAccessRules(includeExplicit: true, includeInherited: false, typeof(SecurityIdentifier))
            .Cast<FileSystemAccessRule>()
            .ToArray();
        Assert.Contains(rules, rule => currentUser.Equals(rule.IdentityReference)
            && rule.AccessControlType == AccessControlType.Allow
            && (rule.FileSystemRights & FileSystemRights.FullControl) == FileSystemRights.FullControl);
        var broad = new[]
        {
            new SecurityIdentifier(WellKnownSidType.WorldSid, null),
            new SecurityIdentifier(WellKnownSidType.AuthenticatedUserSid, null),
            new SecurityIdentifier(WellKnownSidType.BuiltinUsersSid, null),
        };
        Assert.DoesNotContain(rules, rule => broad.Contains(rule.IdentityReference)
            && rule.AccessControlType == AccessControlType.Allow
            && (rule.FileSystemRights & (FileSystemRights.Write | FileSystemRights.Modify | FileSystemRights.FullControl)) != 0);
    }

    private static bool IsWithin(string parent, string candidate)
    {
        var relative = Path.GetRelativePath(Path.GetFullPath(parent), Path.GetFullPath(candidate));
        return relative == "." || (!Path.IsPathRooted(relative)
            && relative != ".."
            && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal));
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-locator-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); }
            catch { }
        }
    }
}
