using YouTuber.Launcher.Activation;
using Xunit;

namespace YouTuber.Launcher.Tests.Activation;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ComponentHealthCheckerEnvironmentCollection
{
    public const string Name = "ComponentHealthChecker process environment";
}

[Collection(ComponentHealthCheckerEnvironmentCollection.Name)]
public sealed class ComponentHealthCheckerTests
{
    [Fact]
    public async Task Starts_the_actual_frozen_studio_worker_with_the_isolated_production_healthcheck_environment()
    {
        if (!OperatingSystem.IsWindows()) return;
        var executable = RepositoryPath("dist", "workers", "studio-worker", "studio-worker.exe");
        if (!File.Exists(executable)) return;
        var root = Path.GetDirectoryName(executable)!;

        await new ComponentHealthChecker().CheckAsync(
            new ComponentHealthCheckRequest(executable, ["--healthcheck"], root));
    }

    [Fact]
    public async Task Supplies_only_required_Windows_bootstrap_and_explicit_healthcheck_variables()
    {
        if (!OperatingSystem.IsWindows()) return;
        var executable = WorkerFixtureExecutable();
        var root = Path.GetDirectoryName(executable)!;
        const string forbidden = "YOUTUBER_TEST_PARENT_SECRET";
        var previous = Environment.GetEnvironmentVariable(forbidden);
        Environment.SetEnvironmentVariable(forbidden, "must-not-reach-child");
        try
        {
            await new ComponentHealthChecker().CheckAsync(new ComponentHealthCheckRequest(
                executable,
                ["--component-healthcheck", $"--expected-root={root}", $"--forbidden-env={forbidden}"],
                root));
        }
        finally
        {
            Environment.SetEnvironmentVariable(forbidden, previous);
        }
    }

    private static string WorkerFixtureExecutable() => Path.GetFullPath(Path.Combine(
        AppContext.BaseDirectory,
        "..", "..", "..", "..", "YouTuber.WorkerFixture", "bin", "Release", "net8.0-windows", "YouTuber.WorkerFixture.exe"));

    private static string RepositoryPath(params string[] parts)
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null && !File.Exists(Path.Combine(current.FullName, "AGENTS.md"))) current = current.Parent;
        Assert.NotNull(current);
        return Path.Combine([current!.FullName, .. parts]);
    }
}
