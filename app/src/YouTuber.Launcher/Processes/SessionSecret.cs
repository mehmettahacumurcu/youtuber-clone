using System.Security.Cryptography;

namespace YouTuber.Launcher.Processes;

/// <summary>Creates per-launch bearer credentials. Values are intended only for child environments and HTTP headers.</summary>
public static class SessionSecret
{
    public const int ByteLength = 32;

    public static string Create()
    {
        Span<byte> bytes = stackalloc byte[ByteLength];
        RandomNumberGenerator.Fill(bytes);
        return Convert.ToBase64String(bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_');
    }

    public static byte[] Decode(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (value.Length != 43 || value.Any(character => !char.IsAsciiLetterOrDigit(character) && character is not '-' and not '_'))
        {
            throw new ArgumentException("The session credential is not valid base64url.", nameof(value));
        }

        return Convert.FromBase64String(value.Replace('-', '+').Replace('_', '/') + "=");
    }
}
