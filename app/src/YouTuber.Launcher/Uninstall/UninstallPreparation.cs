using System.IO;
using YouTuber.Launcher.Configuration;

namespace YouTuber.Launcher.Uninstall;

public static class UninstallPreparation
{
    public const string InstallId = "f324e3a6-7c5d-4bdf-92dd-f00a307b3f1e";

    public static bool IsRequested(IReadOnlyList<string> arguments)
        => arguments.Contains("--prepare-uninstall", StringComparer.Ordinal);

    public static Task RunAsync(IReadOnlyList<string> arguments, CancellationToken cancellationToken = default)
        => RunAsync(arguments, DataRootLocator.ForCurrentUser(), cancellationToken);

    public static Task RunAsync(
        IReadOnlyList<string> arguments,
        DataRootLocator locator,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        ArgumentNullException.ThrowIfNull(locator);
        cancellationToken.ThrowIfCancellationRequested();
        if (!IsRequested(arguments)) throw new ArgumentException("Uninstall preparation was not requested.", nameof(arguments));
        var allowed = arguments.Where(argument => argument == "--prepare-uninstall"
                                                   || argument.StartsWith("--uninstall-choice=", StringComparison.Ordinal)
                                                   || argument == "--uninstall-choice-file"
                                                   || argument.StartsWith("--data-root=", StringComparison.Ordinal)).ToArray();
        if (allowed.Length != arguments.Count || arguments.Count(argument => argument == "--prepare-uninstall") != 1)
            throw new ArgumentException("Uninstall preparation arguments are invalid.", nameof(arguments));

        var choiceArguments = arguments.Where(argument => argument.StartsWith("--uninstall-choice=", StringComparison.Ordinal)).ToArray();
        var choiceFileArguments = arguments.Where(argument => argument == "--uninstall-choice-file").ToArray();
        if (choiceArguments.Length + choiceFileArguments.Length != 1) throw new ArgumentException("Exactly one uninstall choice is required.", nameof(arguments));
        var choiceValue = choiceArguments.Length == 1
            ? choiceArguments[0]["--uninstall-choice=".Length..]
            : LoadChoiceFile();
        var choice = choiceValue switch
        {
            "app-only" => UninstallChoice.AppOnly,
            "app-and-owned-data" => UninstallChoice.AppAndOwnedData,
            _ => throw new ArgumentException("Uninstall choice is invalid.", nameof(arguments)),
        };
        var dataRootArgument = arguments.SingleOrDefault(argument => argument.StartsWith("--data-root=", StringComparison.Ordinal));
        var dataRoot = dataRootArgument is null
            ? locator.LoadRequired()
            : AppPaths.NormalizeAndValidateDataRoot(dataRootArgument["--data-root=".Length..]);
        var paths = AppPaths.ForCurrentUser(dataRoot);
        var plan = new OwnedDataInventory().Plan(paths.DataRoot, InstallId, choice);
        new OwnedDataDeletionExecutor().Execute(paths.DataRoot, plan);
        var residualReport = Path.Combine(paths.StateRoot, "uninstall-residual.json");
        if (choice == UninstallChoice.AppAndOwnedData
            && plan.HasValidatedMatchingOwnership
            && !File.Exists(residualReport)
            && !Directory.Exists(residualReport))
        {
            var locatedRoot = locator.TryLoad();
            if (locatedRoot is not null && string.Equals(locatedRoot, paths.DataRoot, StringComparison.OrdinalIgnoreCase))
                locator.Delete();
        }
        return Task.CompletedTask;
    }

    private static string LoadChoiceFile()
    {
        return File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "uninstall-choice.txt"));
    }
}
