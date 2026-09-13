using System.Security.Cryptography;
using System.IO;

namespace YouTuber.Launcher.Downloads;

public static class Sha256Verifier
{
    public static async Task<bool> MatchesAsync(string path, string expectedSha256, CancellationToken cancellationToken = default)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
        var actual = Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken)).ToLowerInvariant();
        return CryptographicOperations.FixedTimeEquals(System.Text.Encoding.ASCII.GetBytes(actual), System.Text.Encoding.ASCII.GetBytes(expectedSha256));
    }
}
