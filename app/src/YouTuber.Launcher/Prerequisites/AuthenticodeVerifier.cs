using System.Runtime.InteropServices;
using System.IO;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using YouTuber.Launcher.Downloads;

namespace YouTuber.Launcher.Prerequisites;

public sealed record InstallerArtifact(string Path, long ExpectedSize, string ExpectedSha256, string ExpectedPublisher)
{
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(ExpectedPublisher);
        if (ExpectedSize < 0 || ExpectedSha256.Length != 64 || ExpectedSha256.Any(character => character is not (>= '0' and <= '9' or >= 'a' and <= 'f')))
        {
            throw new ArgumentException("Installer artifacts require an exact non-negative size and lowercase SHA-256.");
        }
    }
}

public sealed record AuthenticodeTrustResult(bool IsTrusted, string? Publisher, string? FailureReason);

public interface IWinVerifyTrust
{
    AuthenticodeTrustResult Verify(string path);
}

public sealed class InstallerVerificationException(string message) : InvalidOperationException(message);

public sealed class VerifiedInstaller : IDisposable
{
    private readonly FileStream _guard;

    internal VerifiedInstaller(InstallerArtifact artifact, FileStream guard)
    {
        Artifact = artifact;
        _guard = guard;
    }

    internal InstallerArtifact Artifact { get; }
    public string Path => Artifact.Path;
    internal void EnsureUsable()
    {
        if (_guard.SafeFileHandle.IsClosed) throw new InvalidOperationException("The verified installer guard has already been released.");
    }
    public void Dispose() => _guard.Dispose();
}

public sealed class AuthenticodeVerifier(IWinVerifyTrust? trust = null)
{
    private readonly IWinVerifyTrust _trust = trust ?? new WindowsWinVerifyTrust();

    public async Task<VerifiedInstaller> VerifyAsync(InstallerArtifact artifact, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        artifact.Validate();
        FileStream? guard = null;
        try
        {
            guard = new FileStream(artifact.Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
            if (guard.Length != artifact.ExpectedSize)
            {
                throw new InstallerVerificationException("The installer size does not match the signed manifest.");
            }

            var actualHash = Convert.ToHexString(await System.Security.Cryptography.SHA256.HashDataAsync(guard, cancellationToken)).ToLowerInvariant();
            if (!System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(System.Text.Encoding.ASCII.GetBytes(actualHash), System.Text.Encoding.ASCII.GetBytes(artifact.ExpectedSha256)))
            {
                throw new InstallerVerificationException("The installer hash does not match the signed manifest.");
            }

            var result = _trust.Verify(artifact.Path);
            if (!result.IsTrusted)
            {
                throw new InstallerVerificationException($"Authenticode trust verification failed: {result.FailureReason ?? "unknown error"}.");
            }

            if (!string.Equals(result.Publisher, artifact.ExpectedPublisher, StringComparison.OrdinalIgnoreCase))
            {
                throw new InstallerVerificationException("The Authenticode publisher does not match the signed manifest.");
            }

            return new VerifiedInstaller(artifact, guard);
        }
        catch
        {
            guard?.Dispose();
            throw;
        }
    }
}

public sealed class WindowsWinVerifyTrust : IWinVerifyTrust
{
    public AuthenticodeTrustResult Verify(string path)
    {
        try
        {
            if (!VerifyWindowsTrust(path))
            {
                return new AuthenticodeTrustResult(false, null, "invalid signature chain");
            }

            using var certificate = new X509Certificate2(X509Certificate.CreateFromSignedFile(path));
            return new AuthenticodeTrustResult(true, certificate.GetNameInfo(X509NameType.SimpleName, false), null);
        }
        catch (Exception exception) when (exception is CryptographicException or IOException or UnauthorizedAccessException)
        {
            return new AuthenticodeTrustResult(false, null, "unsigned or unreadable installer");
        }
    }

    private static bool VerifyWindowsTrust(string path)
    {
        var filePath = Marshal.StringToHGlobalUni(path);
        var fileInfoPointer = IntPtr.Zero;
        var action = WinTrustActionGenericVerifyV2;
        var data = new WinTrustData();
        try
        {
            var fileInfo = new WinTrustFileInfo
            {
                cbStruct = (uint)Marshal.SizeOf<WinTrustFileInfo>(),
                pcwszFilePath = filePath,
            };
            fileInfoPointer = Marshal.AllocHGlobal(Marshal.SizeOf<WinTrustFileInfo>());
            Marshal.StructureToPtr(fileInfo, fileInfoPointer, false);
            data = new WinTrustData
            {
                cbStruct = (uint)Marshal.SizeOf<WinTrustData>(),
                dwUIChoice = WtdUiNone,
                fdwRevocationChecks = WtdRevokeWholeChain,
                dwUnionChoice = WtdChoiceFile,
                pFile = fileInfoPointer,
                dwStateAction = WtdStateActionVerify,
                dwProvFlags = WtdRevocationCheckChainExcludeRoot,
            };
            return WinVerifyTrust(IntPtr.Zero, ref action, ref data) == 0;
        }
        finally
        {
            if (data.hWVTStateData != IntPtr.Zero)
            {
                data.dwStateAction = WtdStateActionClose;
                _ = WinVerifyTrust(IntPtr.Zero, ref action, ref data);
            }
            if (fileInfoPointer != IntPtr.Zero) Marshal.FreeHGlobal(fileInfoPointer);
            if (filePath != IntPtr.Zero) Marshal.FreeHGlobal(filePath);
        }
    }

    private static readonly Guid WinTrustActionGenericVerifyV2 = new("00AAC56B-CD44-11D0-8CC2-00C04FC295EE");

    private const uint WtdUiNone = 2;
    private const uint WtdRevokeWholeChain = 1;
    private const uint WtdChoiceFile = 1;
    private const uint WtdStateActionVerify = 1;
    private const uint WtdStateActionClose = 2;
    private const uint WtdRevocationCheckChainExcludeRoot = 0x80;

    [DllImport("wintrust.dll", ExactSpelling = true)]
    private static extern int WinVerifyTrust(IntPtr hwnd, ref Guid actionId, ref WinTrustData data);

    [StructLayout(LayoutKind.Sequential)]
    private struct WinTrustFileInfo
    {
        public uint cbStruct;
        public IntPtr pcwszFilePath;
        public IntPtr hFile;
        public IntPtr pgKnownSubject;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct WinTrustData
    {
        public uint cbStruct;
        public IntPtr pPolicyCallbackData;
        public IntPtr pSIPClientData;
        public uint dwUIChoice;
        public uint fdwRevocationChecks;
        public uint dwUnionChoice;
        public IntPtr pFile;
        public uint dwStateAction;
        public IntPtr hWVTStateData;
        public IntPtr pwszURLReference;
        public uint dwProvFlags;
        public uint dwUIContext;
        public IntPtr pSignatureSettings;
    }
}
