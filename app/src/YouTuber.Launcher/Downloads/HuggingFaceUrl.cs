using System.Text.RegularExpressions;

namespace YouTuber.Launcher.Downloads;

public static partial class HuggingFaceUrl
{
    private const string Host = "huggingface.co";
    private const string ImmutableRevisionPattern = "^[0-9a-f]{40}$";

    public static Uri Create(string repo, string revision, string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repo);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var repoParts = repo.Split('/', StringSplitOptions.None);
        if (repoParts.Length != 2 || repoParts.Any(string.IsNullOrWhiteSpace))
        {
            throw new ArgumentException("The Hugging Face repository must be exactly owner/name.", nameof(repo));
        }

        if (!ImmutableRevision().IsMatch(revision))
        {
            throw new ArgumentException("The Hugging Face revision must be a 40-character lowercase hexadecimal commit.", nameof(revision));
        }

        var escapedRepo = string.Join('/', repoParts.Select(Uri.EscapeDataString));
        var escapedPath = string.Join('/', path.Split('/', StringSplitOptions.None).Select(EscapePathSegment));
        return new Uri($"https://{Host}/{escapedRepo}/resolve/{revision}/{escapedPath}?download=true", UriKind.Absolute);
    }

    public static bool IsCanonicalImmutableResolveUri(Uri uri, bool requireDownloadQuery = false)
    {
        ArgumentNullException.ThrowIfNull(uri);
        if (!uri.IsAbsoluteUri ||
            !string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.Ordinal) ||
            !string.Equals(uri.Host, Host, StringComparison.OrdinalIgnoreCase) ||
            !uri.IsDefaultPort ||
            uri.IsLoopback ||
            uri.UserInfo.Length != 0 ||
            uri.Fragment.Length != 0 ||
            !string.Equals(uri.Query, requireDownloadQuery ? "?download=true" : string.Empty, StringComparison.Ordinal))
        {
            return false;
        }

        var escapedPath = uri.GetComponents(UriComponents.Path, UriFormat.UriEscaped);
        var schemeDelimiter = uri.OriginalString.IndexOf("://", StringComparison.Ordinal);
        var originalPathStart = schemeDelimiter < 0 ? -1 : uri.OriginalString.IndexOf('/', schemeDelimiter + 3);
        if (originalPathStart < 0 ||
            !string.Equals(uri.OriginalString[originalPathStart..], "/" + escapedPath, StringComparison.Ordinal))
        {
            return false;
        }

        var segments = escapedPath.Split('/', StringSplitOptions.None);
        if (segments.Length < 5 ||
            !IsCanonicalRepositorySegment(segments[0]) ||
            !IsCanonicalRepositorySegment(segments[1]) ||
            !string.Equals(segments[2], "resolve", StringComparison.Ordinal) ||
            !ImmutableRevision().IsMatch(segments[3]))
        {
            return false;
        }

        return segments[4..].All(IsCanonicalPathSegment);
    }

    private static string EscapePathSegment(string segment)
    {
        if (string.IsNullOrWhiteSpace(segment) || segment is "." or "..")
        {
            throw new ArgumentException("The Hugging Face artifact path must contain non-empty relative segments.", "path");
        }

        return Uri.EscapeDataString(segment);
    }

    private static bool IsCanonicalRepositorySegment(string segment)
    {
        if (!IsCanonicalPathSegment(segment)) return false;
        return RepositorySegment().IsMatch(Uri.UnescapeDataString(segment));
    }

    private static bool IsCanonicalPathSegment(string segment)
    {
        if (string.IsNullOrWhiteSpace(segment)) return false;
        try
        {
            var decoded = Uri.UnescapeDataString(segment);
            return decoded is not "." and not ".." &&
                !decoded.Contains('/') &&
                !decoded.Contains('\\') &&
                string.Equals(Uri.EscapeDataString(decoded), segment, StringComparison.Ordinal);
        }
        catch (UriFormatException)
        {
            return false;
        }
    }

    [GeneratedRegex(ImmutableRevisionPattern, RegexOptions.CultureInvariant)]
    private static partial Regex ImmutableRevision();

    [GeneratedRegex("^[A-Za-z0-9_.-]+$", RegexOptions.CultureInvariant)]
    private static partial Regex RepositorySegment();
}
