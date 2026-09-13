using System.Text.Json.Serialization;

namespace YouTuber.Launcher.Configuration;

public sealed record LauncherSettings(
    int Schema,
    string DataRoot,
    string ReleaseChannel,
    bool LicenseAccepted,
    bool DisclosureAccepted,
    DateTimeOffset? LastUpdateCheckUtc)
{
    public const int CurrentSchema = 1;

    [JsonIgnore]
    public bool CanDownload => Schema == CurrentSchema
        && string.Equals(ReleaseChannel, "stable", StringComparison.Ordinal)
        && LicenseAccepted
        && DisclosureAccepted;
}
