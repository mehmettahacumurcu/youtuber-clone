using System.IO;
using System.Text.Json;

namespace YouTuber.Launcher.Configuration;

public sealed class SettingsStore
{
    private readonly AppPaths _paths;

    public SettingsStore(AppPaths paths)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
    }

    public async Task<LauncherSettings?> LoadAsync(CancellationToken cancellationToken = default)
    {
        if (!File.Exists(_paths.SettingsFile))
        {
            return null;
        }

        await using var stream = new FileStream(
            _paths.SettingsFile,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 4096,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        var settings = await JsonSerializer.DeserializeAsync(stream, JsonDefaults.LauncherSettings, cancellationToken);

        if (settings is null)
        {
            throw new JsonException("Launcher settings cannot be null.");
        }

        ValidateSettings(settings);
        return settings;
    }

    public async Task SaveAsync(LauncherSettings settings, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ValidateSettings(settings);

        Directory.CreateDirectory(_paths.StateRoot);
        var temporaryFile = Path.Combine(_paths.StateRoot, $"{Path.GetFileName(_paths.SettingsFile)}.{Guid.NewGuid():N}.tmp");

        try
        {
            await using (var stream = new FileStream(
                temporaryFile,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 4096,
                FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await JsonSerializer.SerializeAsync(stream, settings, JsonDefaults.LauncherSettings, cancellationToken);
                await stream.FlushAsync(cancellationToken);
                stream.Flush(flushToDisk: true);
            }

            if (File.Exists(_paths.SettingsFile))
            {
                File.Replace(temporaryFile, _paths.SettingsFile, destinationBackupFileName: null);
            }
            else
            {
                File.Move(temporaryFile, _paths.SettingsFile);
            }
        }
        finally
        {
            if (File.Exists(temporaryFile))
            {
                File.Delete(temporaryFile);
            }
        }
    }

    private void ValidateSettings(LauncherSettings settings)
    {
        if (settings.Schema != LauncherSettings.CurrentSchema)
        {
            throw new JsonException($"Unsupported launcher settings schema: {settings.Schema}.");
        }

        if (!string.Equals(settings.ReleaseChannel, "stable", StringComparison.Ordinal))
        {
            throw new JsonException("Only the stable release channel is supported.");
        }

        AppPaths.ValidateDataRoot(settings.DataRoot);
        if (!string.Equals(Path.TrimEndingDirectorySeparator(Path.GetFullPath(settings.DataRoot)), _paths.DataRoot, StringComparison.OrdinalIgnoreCase))
        {
            throw new JsonException("Launcher settings data root does not match the configured data root.");
        }
    }
}
