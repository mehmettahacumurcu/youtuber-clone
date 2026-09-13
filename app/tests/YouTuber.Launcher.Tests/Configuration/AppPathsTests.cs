using YouTuber.Launcher.Configuration;
using Xunit;

namespace YouTuber.Launcher.Tests.Configuration;

public sealed class AppPathsTests
{
    [Fact]
    public void DefaultPathsUseSeparateProgramAndDataRoots()
    {
        var paths = AppPaths.ForBaseDirectory(@"C:\Users\Developer\AppData\Local");

        Assert.Equal(@"C:\Users\Developer\AppData\Local\Programs\YouTuberStudio", paths.ProgramRoot);
        Assert.Equal(@"C:\Users\Developer\AppData\Local\YouTuberStudio", paths.DataRoot);
        Assert.Equal(@"C:\Users\Developer\AppData\Local\YouTuberStudio\runtime", paths.RuntimeRoot);
        Assert.Equal(@"C:\Users\Developer\AppData\Local\YouTuberStudio\state\launcher-settings.json", paths.SettingsFile);
        Assert.Equal(@"C:\Users\Developer\AppData\Local\YouTuberStudio\state\setup.json", paths.SetupJournalFile);
        Assert.Equal(@"C:\Users\Developer\AppData\Local\YouTuberStudio\state\active-components.json", paths.ActiveComponentsFile);
    }

    [Theory]
    [InlineData(@"\\server\share")]
    [InlineData(@"C:\Windows")]
    public void InvalidDataRootsAreRejected(string root)
    {
        var exception = Assert.Throws<ArgumentException>(() => AppPaths.ValidateDataRoot(root));

        Assert.Equal("root", exception.ParamName);
    }

    [Fact]
    public void DataRootUnderExistingReparsePointIsRejected()
    {
        var testRoot = Path.Combine(Path.GetTempPath(), "YouTuberStudioTests", Guid.NewGuid().ToString("N"));
        var target = Path.Combine(testRoot, "target");
        var link = Path.Combine(testRoot, "redirect");

        try
        {
            Directory.CreateDirectory(target);
            Directory.CreateSymbolicLink(link, target);

            var exception = Assert.Throws<ArgumentException>(() => AppPaths.ForBaseDirectory(testRoot, Path.Combine(link, "data")));

            Assert.Equal("root", exception.ParamName);
        }
        finally
        {
            if (Directory.Exists(link))
            {
                Directory.Delete(link);
            }

            if (Directory.Exists(testRoot))
            {
                Directory.Delete(testRoot, recursive: true);
            }
        }
    }

    [Theory]
    [InlineData("payload:stream")]
    [InlineData("CON")]
    [InlineData("nul.txt")]
    [InlineData("trailing.")]
    [InlineData("trailing ")]
    public void Win32_alias_device_and_stream_components_are_rejected(string component)
    {
        var root = Path.Combine(Path.GetTempPath(), "YouTuberStudioTests", Guid.NewGuid().ToString("N"));
        var exception = Assert.Throws<ArgumentException>(() => AppPaths.ValidateDataRoot(Path.Combine(root, component)));
        Assert.Equal("root", exception.ParamName);
    }
}
