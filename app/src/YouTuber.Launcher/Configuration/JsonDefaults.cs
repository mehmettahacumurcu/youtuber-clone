using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;

namespace YouTuber.Launcher.Configuration;

public static class JsonDefaults
{
    public static JsonTypeInfo<LauncherSettings> LauncherSettings => LauncherJsonContext.Default.LauncherSettings;
}

[JsonSourceGenerationOptions(
    GenerationMode = JsonSourceGenerationMode.Metadata,
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    WriteIndented = true)]
[JsonSerializable(typeof(LauncherSettings))]
internal partial class LauncherJsonContext : JsonSerializerContext
{
}
