using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using YouTuber.Launcher.Activation;
using Xunit;

namespace YouTuber.Launcher.Tests.Activation;

public sealed class ArchiveExtractorTests
{
    [Theory]
    [InlineData("../escape.txt")]
    [InlineData("/rooted.txt")]
    [InlineData("C:drive-relative.txt")]
    [InlineData("runtime/CON")]
    [InlineData("runtime/COM¹.txt")]
    [InlineData("runtime/COM².exe")]
    [InlineData("runtime/COM³")]
    [InlineData("runtime/LPT¹.txt")]
    [InlineData("runtime/LPT².exe")]
    [InlineData("runtime/LPT³")]
    [InlineData("runtime/file.txt:stream")]
    public async Task ExtractAsyncRejectsUnsafeNamesBeforeWritingOutsideStaging(string entryName)
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([(entryName, Bytes("unsafe"), null)]);

        await Assert.ThrowsAsync<UnsafeArchiveException>(() => fixture.ExtractAsync());

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
        Assert.False(File.Exists(Path.Combine(fixture.Root, "escape.txt")));
    }

    [Fact]
    public async Task ExtractAsyncRejectsCaseCollisionsBeforeWritingStaging()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([
            ("runtime/Worker.exe", Bytes("first"), null),
            ("runtime/worker.exe", Bytes("second"), null),
        ]);

        await Assert.ThrowsAsync<UnsafeArchiveException>(() => fixture.ExtractAsync());

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    [Fact]
    public async Task ExtractAsyncRejectsSymlinkMetadataBeforeWritingStaging()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([("runtime/link", Bytes("target"), (UnixFileMode)0xA1FF)]);

        await Assert.ThrowsAsync<UnsafeArchiveException>(() => fixture.ExtractAsync());

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    [Fact]
    public async Task ExtractAsyncRejectsAnEntryOutsideTheDeclaredArchiveRoot()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([("other/file.txt", Bytes("outside"), null)]);

        await Assert.ThrowsAsync<UnsafeArchiveException>(() => fixture.ExtractAsync());

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    [Fact]
    public async Task ExtractAsyncRejectsTooManyEntriesBeforeWritingStaging()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([
            ("runtime/one.txt", Bytes("one"), null),
            ("runtime/two.txt", Bytes("two"), null),
            ("runtime/three.txt", Bytes("three"), null),
        ]);

        await Assert.ThrowsAsync<UnsafeArchiveException>(() => fixture.ExtractAsync(new ArchiveExtractionLimits(2, 100)));

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    [Fact]
    public async Task ExtractAsyncRejectsAnExtremeCompressionRatioBeforeWritingStaging()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([("runtime/payload.bin", new byte[10_000], null)], CompressionLevel.SmallestSize);

        await Assert.ThrowsAsync<UnsafeArchiveException>(() => fixture.ExtractAsync(new ArchiveExtractionLimits(10, 2)));

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    public async Task ExtractAsyncRejectsAnArchiveWhoseExpandedTotalDiffersFromTheSignedSize(long delta)
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([("runtime/payload.bin", Bytes("signed payload"), null)]);

        await Assert.ThrowsAsync<UnsafeArchiveException>(() => fixture.ExtractAsync(
            expectedExpandedBytes: checked(fixture.ExpandedBytes + delta)));

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    [Fact]
    public async Task ExtractAsyncVerifiesTheOuterArchiveHashBeforeCreatingPayloadFiles()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([("runtime/worker.exe", Bytes("safe"), null)]);

        await Assert.ThrowsAsync<ComponentActivationException>(() => ArchiveExtractor.ExtractAsync(
            fixture.Archive,
            new string('0', 64),
            "runtime",
            fixture.Staging,
            fixture.ExpandedBytes));

        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    [Fact]
    public async Task ExtractAsyncExtractsOnlyTheDeclaredRootIntoStaging()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([("runtime/bin/worker.exe", Bytes("safe"), null)]);

        await fixture.ExtractAsync();

        Assert.Equal("safe", await File.ReadAllTextAsync(Path.Combine(fixture.Staging, "bin", "worker.exe")));
        Assert.False(Directory.Exists(Path.Combine(fixture.Staging, "runtime")));
    }

    [Fact]
    public async Task ExtractAsyncKeepsUsingTheVerifiedOpenArchiveWhenTheArchivePathIsSwappedAtTheRaceHook()
    {
        using var fixture = new ArchiveFixture();
        fixture.WriteZip([("runtime/worker.exe", Bytes("verified"), null)]);
        var expectedHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(fixture.Archive))).ToLowerInvariant();
        var replacement = Path.Combine(fixture.Root, "replacement.zip");
        using (var stream = new FileStream(replacement, FileMode.Create, FileAccess.Write, FileShare.None))
        using (var archive = new ZipArchive(stream, ZipArchiveMode.Create))
        {
            using var output = archive.CreateEntry("runtime/worker.exe").Open();
            output.Write(Bytes("unverified"));
        }

        var hook = new ReplacingArchiveHook(fixture.Archive, replacement);
        await Assert.ThrowsAsync<ComponentActivationException>(() => ArchiveExtractor.ExtractAsync(
            fixture.Archive,
            expectedHash,
            "runtime",
            fixture.Staging,
            fixture.ExpandedBytes,
            hooks: hook));

        Assert.True(hook.Attempted);
        Assert.Empty(Directory.EnumerateFileSystemEntries(fixture.Staging));
    }

    private static byte[] Bytes(string value) => Encoding.UTF8.GetBytes(value);

    private sealed class ArchiveFixture : IDisposable
    {
        public ArchiveFixture()
        {
            Root = Path.Combine(Path.GetTempPath(), "YouTuberActivationTests", Guid.NewGuid().ToString("N"));
            Archive = Path.Combine(Root, "component.zip");
            Staging = Path.Combine(Root, "staging");
            Directory.CreateDirectory(Staging);
        }

        public string Root { get; }
        public string Archive { get; }
        public string Staging { get; }

        public void WriteZip((string Name, byte[] Payload, UnixFileMode? Mode)[] entries, CompressionLevel compression = CompressionLevel.Optimal)
        {
            Directory.CreateDirectory(Root);
            using var stream = new FileStream(Archive, FileMode.Create, FileAccess.Write, FileShare.None);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Create);
            foreach (var (name, payload, mode) in entries)
            {
                var entry = archive.CreateEntry(name, compression);
                if (mode is not null) entry.ExternalAttributes = ((int)mode.Value) << 16;
                using var output = entry.Open();
                output.Write(payload);
            }
        }

        public long ExpandedBytes
        {
            get
            {
                using var archive = ZipFile.OpenRead(Archive);
                return archive.Entries.Where(entry => !entry.FullName.EndsWith("/", StringComparison.Ordinal))
                    .Aggregate(0L, (total, entry) => checked(total + entry.Length));
            }
        }

        public Task ExtractAsync(ArchiveExtractionLimits? limits = null, long? expectedExpandedBytes = null) => ArchiveExtractor.ExtractAsync(
            Archive,
            Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Archive))).ToLowerInvariant(),
            "runtime",
            Staging,
            expectedExpandedBytes ?? ExpandedBytes,
            limits);

        public void Dispose()
        {
            if (Directory.Exists(Root)) Directory.Delete(Root, recursive: true);
        }
    }

    private sealed class ReplacingArchiveHook(string archive, string replacement) : IArchiveExtractionHooks
    {
        public bool Attempted { get; private set; }

        public Task AfterArchiveVerifiedAsync(CancellationToken cancellationToken)
        {
            Attempted = true;
            File.Move(replacement, archive, overwrite: true);
            return Task.CompletedTask;
        }
    }
}
