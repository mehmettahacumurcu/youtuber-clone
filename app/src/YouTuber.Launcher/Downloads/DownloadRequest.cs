namespace YouTuber.Launcher.Downloads;

public sealed record DownloadRequest(Uri Source, string TargetPath, long ExpectedSize, string ExpectedSha256)
{
    public void Validate()
    {
        ArgumentNullException.ThrowIfNull(Source);
        ArgumentException.ThrowIfNullOrWhiteSpace(TargetPath);
        if (Source.Scheme != Uri.UriSchemeHttps ||
            !string.Equals(Source.Host, "huggingface.co", StringComparison.OrdinalIgnoreCase) ||
            Source.Port != 443 ||
            !string.IsNullOrEmpty(Source.UserInfo) ||
            !string.IsNullOrEmpty(Source.Fragment) ||
            !string.Equals(Source.Query, "?download=true", StringComparison.Ordinal))
        {
            throw new ArgumentException("Downloads must begin with a credential-free immutable HTTPS huggingface.co URL.", nameof(Source));
        }

        var pathSegments = Source.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (pathSegments.Length < 5 ||
            pathSegments[2] != "resolve" ||
            !IsImmutableRevision(pathSegments[3]))
        {
            throw new ArgumentException("Downloads must use an immutable 40-character Hugging Face resolve revision.", nameof(Source));
        }

        if (ExpectedSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ExpectedSize));
        }

        if (ExpectedSha256.Length != 64 || ExpectedSha256.Any(character => !Uri.IsHexDigit(character)) || ExpectedSha256.Any(char.IsUpper))
        {
            throw new ArgumentException("Expected SHA-256 must be lowercase hexadecimal.", nameof(ExpectedSha256));
        }
    }

    private static bool IsImmutableRevision(string value) =>
        value.Length == 40 && value.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f');
}
