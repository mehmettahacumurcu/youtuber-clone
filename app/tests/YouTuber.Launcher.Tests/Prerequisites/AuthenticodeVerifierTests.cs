using System.Security.Cryptography;
using System.Text;
using YouTuber.Launcher.Prerequisites;
using Xunit;

namespace YouTuber.Launcher.Tests.Prerequisites;

public sealed class AuthenticodeVerifierTests
{
    [Fact]
    public async Task VerifyAsyncRejectsHashMismatchBeforeTrustVerification()
    {
        using var file = new TemporaryFile("installer");
        var trust = new FakeTrust(true, "Microsoft Corporation");
        var verifier = new AuthenticodeVerifier(trust);

        await Assert.ThrowsAsync<InstallerVerificationException>(() => verifier.VerifyAsync(Artifact(file.Path, new string('0', 64))));

        Assert.Equal(0, trust.Calls);
    }

    [Theory]
    [InlineData(false, "Microsoft Corporation")]
    [InlineData(true, "Unexpected Publisher")]
    public async Task VerifyAsyncRejectsUntrustedOrWrongPublisherInstallers(bool trusted, string publisher)
    {
        using var file = new TemporaryFile("installer");
        var verifier = new AuthenticodeVerifier(new FakeTrust(trusted, publisher));

        await Assert.ThrowsAsync<InstallerVerificationException>(() => verifier.VerifyAsync(Artifact(file.Path)));
    }

    [Fact]
    public async Task VerifyAsyncReturnsAnOpaqueVerifiedInstallerForAValidArtifact()
    {
        using var file = new TemporaryFile("installer");
        using var verified = await new AuthenticodeVerifier(new FakeTrust(true, "Microsoft Corporation")).VerifyAsync(Artifact(file.Path));

        Assert.Equal(file.Path, verified.Path);
    }

    [Fact]
    public async Task VerifyAsyncHoldsAnExclusiveReadGuardUntilTheInstallerIsConsumed()
    {
        using var file = new TemporaryFile("installer");
        using var verified = await new AuthenticodeVerifier(new FakeTrust(true, "Microsoft Corporation")).VerifyAsync(Artifact(file.Path));

        Assert.Throws<IOException>(() => new FileStream(file.Path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite));
    }

    [Fact]
    public void WindowsWinVerifyTrustAcceptsACopiedMicrosoftSignedSystemExecutableAndRejectsUnsignedFixture()
    {
        if (!OperatingSystem.IsWindows()) return;
        var systemExecutable = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "appverif.exe");
        Assert.True(File.Exists(systemExecutable), "Windows test host must supply Microsoft-signed appverif.exe.");
        using var directory = new TemporaryDirectory();
        var signedCopy = Path.Combine(directory.Path, "appverif.exe");
        File.Copy(systemExecutable, signedCopy);
        var unsigned = Path.Combine(directory.Path, "unsigned.exe");
        File.WriteAllText(unsigned, "not a signed executable");

        var trust = new WindowsWinVerifyTrust();
        var signed = trust.Verify(signedCopy);
        var unsignedResult = trust.Verify(unsigned);

        Assert.True(signed.IsTrusted, signed.FailureReason);
        Assert.Contains("Microsoft", signed.Publisher, StringComparison.OrdinalIgnoreCase);
        Assert.False(unsignedResult.IsTrusted);
    }

    private static InstallerArtifact Artifact(string path, string? hash = null) => new(
        path,
        new FileInfo(path).Length,
        hash ?? Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant(),
        "Microsoft Corporation");

    private sealed class FakeTrust(bool trusted, string publisher) : IWinVerifyTrust
    {
        public int Calls { get; private set; }

        public AuthenticodeTrustResult Verify(string path)
        {
            Calls++;
            return new AuthenticodeTrustResult(trusted, publisher, trusted ? null : "invalid chain");
        }
    }

    private sealed class TemporaryFile : IDisposable
    {
        public TemporaryFile(string contents)
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"YouTuberAuth-{Guid.NewGuid():N}.exe");
            File.WriteAllText(Path, contents, Encoding.UTF8);
        }

        public string Path { get; }

        public void Dispose() => File.Delete(Path);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"YouTuberAuthenticode-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Path);
        }
        public string Path { get; }
        public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, true); }
    }
}
