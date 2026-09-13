using YouTuber.Launcher.Downloads;
using Xunit;

namespace YouTuber.Launcher.Tests.Downloads;

public sealed class HuggingFaceUrlTests
{
    [Fact]
    public void CreateEscapesEveryPathSegmentWithoutEscapingTheRepositorySeparators()
    {
        var uri = HuggingFaceUrl.Create(
            "speaker/konuşma modeli",
            "1111111111111111111111111111111111111111",
            "weights/İstanbul final.bin");

        Assert.Equal(
            "https://huggingface.co/speaker/konu%C5%9Fma%20modeli/resolve/1111111111111111111111111111111111111111/weights/%C4%B0stanbul%20final.bin?download=true",
            uri.AbsoluteUri);
    }

    [Theory]
    [InlineData("main")]
    [InlineData("refs/pr/17")]
    [InlineData("111111111111111111111111111111111111111")]
    [InlineData("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")]
    public void CreateRejectsNonImmutableRevisions(string revision)
    {
        Assert.Throws<ArgumentException>(() => HuggingFaceUrl.Create("owner/name", revision, "file.bin"));
    }

    [Theory]
    [InlineData("owner")]
    [InlineData("owner/name/extra")]
    [InlineData("owner//name")]
    public void CreateRequiresExactlyOwnerAndName(string repo)
    {
        Assert.Throws<ArgumentException>(() => HuggingFaceUrl.Create(repo, "1111111111111111111111111111111111111111", "file.bin"));
    }
}
