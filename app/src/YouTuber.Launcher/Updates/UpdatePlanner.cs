using YouTuber.Launcher.Distribution;

namespace YouTuber.Launcher.Updates;

public sealed record InstalledComponent(string Revision, string Sha256, int RuntimeApi);

public sealed record UpdatePlan(IReadOnlySet<string> Components, long RequiredRollbackBytes, string? Rejection = null)
{
    public bool IsRejected => Rejection is not null;
}

public sealed class UpdatePlanner
{
    private static readonly IReadOnlySet<string> RuntimeComponents = new HashSet<string>(["studio_runtime", "voice_runtime"], StringComparer.Ordinal);

    public UpdatePlan Plan(DistributionManifest manifest, IReadOnlyDictionary<string, InstalledComponent> installed, long freeBytes)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(installed);
        if (freeBytes < 0) throw new ArgumentOutOfRangeException(nameof(freeBytes));
        var changed = manifest.Components
            .Where(pair => !installed.TryGetValue(pair.Key, out var current)
                           || !string.Equals(current.Revision, pair.Value.Revision, StringComparison.Ordinal)
                           || !string.Equals(current.Sha256, pair.Value.Sha256, StringComparison.Ordinal)
                           || current.RuntimeApi != pair.Value.Compatibility.RuntimeApi)
            .Select(pair => pair.Key)
            .ToHashSet(StringComparer.Ordinal);

        var requiredApis = changed
            .Where(installed.ContainsKey)
            .Where(name => installed[name].RuntimeApi != manifest.Components[name].Compatibility.RuntimeApi)
            .Select(name => manifest.Components[name].Compatibility.RuntimeApi)
            .Distinct()
            .ToArray();
        if (requiredApis.Length > 1)
            return new UpdatePlan(new HashSet<string>(StringComparer.Ordinal), 0, "The release requests conflicting runtime APIs.");
        if (requiredApis.Length == 1)
        {
            if (RuntimeComponents.Any(runtime => !manifest.Components.TryGetValue(runtime, out var component)
                                                  || component.Compatibility.RuntimeApi != requiredApis[0]))
            {
                return new UpdatePlan(new HashSet<string>(StringComparer.Ordinal), 0, "The release does not contain a complete compatible runtime set.");
            }
            foreach (var runtime in RuntimeComponents)
            {
                changed.Add(runtime);
            }
        }

        var required = changed.Sum(name => manifest.Components[name].PeakSpace);
        if (freeBytes < required)
            return new UpdatePlan(new HashSet<string>(StringComparer.Ordinal), required, "Insufficient space for a verified rollback copy.");
        return new UpdatePlan(changed, required);
    }
}
