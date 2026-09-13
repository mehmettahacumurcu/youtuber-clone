using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.IO;

namespace YouTuber.Launcher.Activation;

public sealed record ComponentInventory(
    string Schema,
    string Component,
    string Version,
    ComponentCompatibility Compatibility,
    string Entrypoint,
    IReadOnlyList<string> Healthcheck,
    IReadOnlyList<ComponentFile> Files)
{
    public const string ExpectedSchema = "youtuber.component.v1";

    public long InstallSize => Files.Aggregate(0L, (total, file) => checked(total + file.Size));

    public static ComponentInventory Parse(string json)
    {
        try
        {
            var inventory = JsonSerializer.Deserialize(json, ComponentInventoryJsonContext.Default.ComponentInventory)
                ?? throw new ComponentActivationException("The component inventory is empty.");
            inventory.Validate();
            return inventory;
        }
        catch (ComponentActivationException) { throw; }
        catch (JsonException exception) { throw new ComponentActivationException("The component inventory is invalid.", exception); }
    }

    public void Validate()
    {
        if (Schema != ExpectedSchema || string.IsNullOrWhiteSpace(Component) || !PathRules.IsSemVer(Version) || Compatibility is null || Compatibility.RuntimeApi != 1)
        {
            throw new ComponentActivationException("The component inventory contract is invalid.");
        }

        var entrypoint = PathRules.NormalizeRelativePath(Entrypoint, "component entrypoint");
        if (Healthcheck is not { Count: > 0 } || Healthcheck.Any(argument => string.IsNullOrWhiteSpace(argument) || argument.IndexOf('\0') >= 0))
        {
            throw new ComponentActivationException("The component health-check argv is invalid.");
        }

        if (Files is not { Count: > 0 }) throw new ComponentActivationException("The component inventory has no files.");
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var hasEntrypoint = false;
        foreach (var file in Files)
        {
            if (file is null || file.Size <= 0 || !PathRules.IsSha256(file.Sha256)) throw new ComponentActivationException("The component inventory file is invalid.");
            var path = PathRules.NormalizeRelativePath(file.Path, "component file");
            if (!paths.Add(path)) throw new ComponentActivationException("The component inventory contains duplicate Windows paths.");
            hasEntrypoint |= string.Equals(path, entrypoint, StringComparison.OrdinalIgnoreCase);
        }

        if (!hasEntrypoint) throw new ComponentActivationException("The component entrypoint is not inventoried.");
        try { _ = InstallSize; }
        catch (OverflowException exception) { throw new ComponentActivationException("The component inventory size overflows Int64.", exception); }
    }
}

public sealed record ComponentCompatibility(int RuntimeApi);
public sealed record ComponentFile(string Path, long Size, string Sha256);

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
[JsonSerializable(typeof(ComponentInventory))]
internal partial class ComponentInventoryJsonContext : JsonSerializerContext { }

internal static partial class PathRules
{
    private static readonly HashSet<string> ReservedDevices = new(StringComparer.OrdinalIgnoreCase)
    {
        "CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    };

    internal static string NormalizeRelativePath(string value, string description)
    {
        if (string.IsNullOrWhiteSpace(value) || Path.IsPathRooted(value) || value.Contains('\\') || value.Contains(':') || value.StartsWith("/", StringComparison.Ordinal))
        {
            throw new UnsafeArchiveException($"The {description} must be a strict relative POSIX path.");
        }

        var parts = value.Split('/', StringSplitOptions.None);
        if (parts.Length == 0 || parts.Any(part => part.Length == 0 || part is "." or "..")) throw new UnsafeArchiveException($"The {description} traverses outside its root.");
        foreach (var part in parts)
        {
            if (part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || part.Any(char.IsControl)) throw new UnsafeArchiveException($"The {description} contains an invalid filename.");
            var trimmed = part.TrimEnd(' ', '.');
            if (trimmed.Length == 0 || ReservedDevices.Contains(CanonicalizeDeviceName(trimmed.Split('.', StringSplitOptions.None)[0]))) throw new UnsafeArchiveException($"The {description} contains a Windows device name.");
        }

        return string.Join('/', parts.Select(part => part.TrimEnd(' ', '.')));
    }

    internal static string PathKey(string value) => string.Join('/', value.Split('/', StringSplitOptions.None).Select(part => part.TrimEnd(' ', '.').ToUpperInvariant()));
    internal static bool IsSha256(string? value) => value is not null && Sha256Pattern().IsMatch(value);
    internal static bool IsSemVer(string? value) => value is not null && SemVerPattern().IsMatch(value);

    [GeneratedRegex("^[0-9a-f]{64}$", RegexOptions.CultureInvariant)]
    private static partial Regex Sha256Pattern();
    [GeneratedRegex("^[0-9]+\\.[0-9]+\\.[0-9]+$", RegexOptions.CultureInvariant)]
    private static partial Regex SemVerPattern();

    private static string CanonicalizeDeviceName(string value) => value.Replace('¹', '1').Replace('²', '2').Replace('³', '3');
}
