using System.Collections.Generic;
using System.IO;

namespace YouTuber.Launcher.Configuration;

public sealed class AppPaths
{
    private const string ApplicationName = "YouTuberStudio";

    private AppPaths(string programRoot, string dataRoot)
    {
        ProgramRoot = programRoot;
        DataRoot = dataRoot;
        RuntimeRoot = Path.Combine(DataRoot, "runtime");
        ModelsRoot = Path.Combine(DataRoot, "models");
        RagRoot = Path.Combine(DataRoot, "rag");
        DownloadsRoot = Path.Combine(DataRoot, "downloads");
        StateRoot = Path.Combine(DataRoot, "state");
        LogsRoot = Path.Combine(DataRoot, "logs");
        GeneratedAudioRoot = Path.Combine(DataRoot, "generated-audio");
        SettingsFile = Path.Combine(StateRoot, "launcher-settings.json");
        SetupJournalFile = Path.Combine(StateRoot, "setup.json");
        ActiveComponentsFile = Path.Combine(StateRoot, "active-components.json");
    }

    public string ProgramRoot { get; }

    public string DataRoot { get; }

    public string RuntimeRoot { get; }

    public string ModelsRoot { get; }

    public string RagRoot { get; }

    public string DownloadsRoot { get; }

    public string StateRoot { get; }

    public string LogsRoot { get; }

    public string GeneratedAudioRoot { get; }

    public string SettingsFile { get; }

    public string SetupJournalFile { get; }

    public string ActiveComponentsFile { get; }

    public static AppPaths ForCurrentUser(string? dataRootOverride = null)
    {
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return ForBaseDirectory(localAppData, dataRootOverride);
    }

    public static AppPaths ForBaseDirectory(string baseDirectory, string? dataRootOverride = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDirectory);

        var normalizedBaseDirectory = NormalizeAbsolutePath(baseDirectory, nameof(baseDirectory));
        var programRoot = Path.Combine(normalizedBaseDirectory, "Programs", ApplicationName);
        var dataRoot = dataRootOverride is null
            ? Path.Combine(normalizedBaseDirectory, ApplicationName)
            : NormalizeAndValidateDataRoot(dataRootOverride);

        ValidateDataRoot(dataRoot);
        return new AppPaths(programRoot, dataRoot);
    }

    public static void ValidateDataRoot(string root)
    {
        _ = NormalizeAndValidateDataRoot(root);
    }

    public static string NormalizeAndValidateDataRoot(string root)
    {
        if (!Path.IsPathFullyQualified(root))
            throw new ArgumentException("The path must be absolute.", nameof(root));
        var rawDriveRoot = Path.GetPathRoot(root)
            ?? throw new ArgumentException("The data root must have a stable drive.", nameof(root));
        RejectUnsafeWindowsComponents(root, rawDriveRoot, nameof(root));
        var normalizedRoot = NormalizeAbsolutePath(root, nameof(root));

        if (normalizedRoot.StartsWith(@"\\", StringComparison.Ordinal) || normalizedRoot.StartsWith(@"\\?\", StringComparison.Ordinal))
        {
            throw new ArgumentException("The data root must be a local drive path.", nameof(root));
        }

        RejectExistingReparsePoints(normalizedRoot, nameof(root));

        var driveRoot = Path.GetPathRoot(normalizedRoot);
        if (string.IsNullOrEmpty(driveRoot) || !Path.IsPathFullyQualified(driveRoot))
        {
            throw new ArgumentException("The data root must have a stable drive.", nameof(root));
        }

        var normalizedDriveRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(driveRoot));
        if (string.Equals(normalizedRoot, normalizedDriveRoot, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("The data root cannot be a filesystem root.", nameof(root));
        }

        RejectUnsafeWindowsComponents(normalizedRoot, driveRoot, nameof(root));

        DriveInfo drive;
        try
        {
            drive = new DriveInfo(driveRoot);
        }
        catch (ArgumentException exception)
        {
            throw new ArgumentException("The data root must have a stable drive.", nameof(root), exception);
        }

        if (!drive.IsReady || drive.DriveType != DriveType.Fixed || !string.Equals(drive.DriveFormat, "NTFS", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("The data root must be on a local NTFS fixed drive.", nameof(root));
        }

        foreach (var protectedRoot in EnumerateProtectedRoots())
        {
            if (IsWithin(normalizedRoot, protectedRoot))
            {
                throw new ArgumentException("The data root cannot be inside a protected Windows directory.", nameof(root));
            }
        }

        return normalizedRoot;
    }

    private static void RejectUnsafeWindowsComponents(string path, string driveRoot, string parameterName)
    {
        var reserved = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        };
        var invalid = Path.GetInvalidFileNameChars();
        foreach (var component in path[driveRoot.Length..].Split(
                     [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar],
                     StringSplitOptions.RemoveEmptyEntries))
        {
            var stem = component.Split('.', 2)[0];
            if (component.EndsWith(' ') || component.EndsWith('.') || component.IndexOfAny(invalid) >= 0 || reserved.Contains(stem))
                throw new ArgumentException("The data root contains a Win32-unsafe path component.", parameterName);
        }
    }

    private static void RejectExistingReparsePoints(string path, string parameterName)
    {
        var driveRoot = Path.GetPathRoot(path);
        if (string.IsNullOrEmpty(driveRoot))
        {
            throw new ArgumentException("The data root must have a stable drive.", parameterName);
        }

        var current = driveRoot;
        EnsureExistingPathIsSafe(current, parameterName);

        var relativePath = path[driveRoot.Length..];
        foreach (var component in relativePath.Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, component);
            if (!EnsureExistingPathIsSafe(current, parameterName))
            {
                return;
            }
        }
    }

    private static bool EnsureExistingPathIsSafe(string path, string parameterName)
    {
        FileAttributes attributes;
        try
        {
            attributes = File.GetAttributes(path);
        }
        catch (FileNotFoundException)
        {
            return false;
        }
        catch (DirectoryNotFoundException)
        {
            return false;
        }
        catch (UnauthorizedAccessException exception)
        {
            throw new ArgumentException("The data root cannot be inspected safely.", parameterName, exception);
        }
        catch (IOException exception)
        {
            throw new ArgumentException("The data root cannot be inspected safely.", parameterName, exception);
        }

        if ((attributes & FileAttributes.ReparsePoint) != 0)
        {
            throw new ArgumentException("The data root cannot contain a reparse point.", parameterName);
        }

        if (!Directory.Exists(path))
        {
            throw new ArgumentException("The data root cannot be beneath a file.", parameterName);
        }

        return true;
    }

    private static IEnumerable<string> EnumerateProtectedRoots()
    {
        var windowsDirectory = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);

        if (!string.IsNullOrWhiteSpace(windowsDirectory))
        {
            yield return NormalizeAbsolutePath(windowsDirectory, nameof(windowsDirectory));
        }

        if (!string.IsNullOrWhiteSpace(programFiles))
        {
            yield return NormalizeAbsolutePath(programFiles, nameof(programFiles));
        }

        if (!string.IsNullOrWhiteSpace(programFilesX86))
        {
            yield return NormalizeAbsolutePath(programFilesX86, nameof(programFilesX86));
        }
    }

    private static bool IsWithin(string candidate, string parent)
    {
        var relative = Path.GetRelativePath(parent, candidate);
        return relative == "." || (!relative.StartsWith("..", StringComparison.Ordinal) && !Path.IsPathRooted(relative));
    }

    private static string NormalizeAbsolutePath(string path, string parameterName)
    {
        if (!Path.IsPathFullyQualified(path))
        {
            throw new ArgumentException("The path must be absolute.", parameterName);
        }

        return Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));
    }
}
