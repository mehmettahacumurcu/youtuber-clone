using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace YouTuber.Launcher.Activation;

public sealed class ComponentPathGuard : IDisposable
{
    private const uint GenericRead = 0x80000000;
    private const uint FileShareRead = 0x00000001;
    private const uint FileShareWrite = 0x00000002;
    private const uint OpenExisting = 3;
    private const uint FileFlagBackupSemantics = 0x02000000;
    private const uint FileFlagOpenReparsePoint = 0x00200000;
    private const uint FileAttributeDirectory = 0x00000010;
    private const uint FileAttributeReparsePoint = 0x00000400;

    private readonly string _root;
    private readonly IReadOnlyList<GuardedPath> _paths;

    private ComponentPathGuard(string root, IReadOnlyList<GuardedPath> paths)
    {
        _root = root;
        _paths = paths;
    }

    public static ComponentPathGuard Acquire(string componentRoot, ComponentInventory inventory)
    {
        ArgumentNullException.ThrowIfNull(inventory);
        var root = Path.GetFullPath(componentRoot);
        EnsureDirectory(root, "component root");
        var directories = CollectDirectories(inventory);
        var files = inventory.Files.Select(file => PathRules.NormalizeRelativePath(file.Path, "component file"))
            .Append("component-manifest.json")
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();
        var guarded = new List<GuardedPath>(directories.Length + files.Length);
        try
        {
            foreach (var directory in directories)
            {
                guarded.Add(OpenPath(root, directory, isDirectory: true));
            }
            foreach (var file in files)
            {
                guarded.Add(OpenPath(root, file, isDirectory: false));
            }
            return new ComponentPathGuard(root, guarded);
        }
        catch
        {
            DisposeReverse(guarded);
            throw;
        }
    }

    public static ComponentPathGuard AcquireAncestors(string dataRoot, string installParent)
    {
        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(dataRoot));
        var parent = Path.TrimEndingDirectorySeparator(Path.GetFullPath(installParent));
        EnsureWithinRoot(root, parent);
        EnsureDirectory(root, "component data root");
        var relative = Path.GetRelativePath(root, parent);
        var parts = relative == "."
            ? []
            : relative.Split(
                [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar],
                StringSplitOptions.RemoveEmptyEntries);
        var guarded = new List<GuardedPath>(parts.Length + 1);
        try
        {
            guarded.Add(OpenPath(root, string.Empty, isDirectory: true, shareMode: FileShareRead | FileShareWrite));
            var current = root;
            var relativeParts = new List<string>(parts.Length);
            foreach (var part in parts)
            {
                current = Path.Combine(current, part);
                relativeParts.Add(part);
                if (File.Exists(current) && !Directory.Exists(current))
                    throw new ComponentActivationException("A component install ancestor is not a directory.");
                if (!Directory.Exists(current)) Directory.CreateDirectory(current);
                guarded.Add(OpenPath(root, string.Join('/', relativeParts), isDirectory: true, shareMode: FileShareRead | FileShareWrite));
            }
            return new ComponentPathGuard(root, guarded);
        }
        catch
        {
            DisposeReverse(guarded);
            throw;
        }
    }

    public void AssertIntact()
    {
        EnsureDirectory(_root, "component root");
        foreach (var guarded in _paths)
        {
            if (guarded.IsDirectory) EnsureDirectory(guarded.Path, "component directory");
            else EnsureFile(guarded.Path, "component file");

            if (guarded.NativeHandle is not null)
            {
                var current = GetInformation(guarded.NativeHandle, guarded.Path);
                if (!SameIdentity(current, guarded.Identity)) throw new ComponentActivationException("A guarded component path changed identity during activation.");
            }
        }
    }

    public void Dispose() => DisposeReverse(_paths);

    private static string[] CollectDirectories(ComponentInventory inventory)
    {
        var directories = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { string.Empty };
        foreach (var file in inventory.Files)
        {
            var parts = PathRules.NormalizeRelativePath(file.Path, "component file").Split('/', StringSplitOptions.None);
            for (var index = 1; index < parts.Length; index++) directories.Add(string.Join("/", parts[..index]));
        }

        return directories.OrderBy(path => path.Count(character => character == '/')).ThenBy(path => path, StringComparer.Ordinal).ToArray();
    }

    private static GuardedPath OpenPath(
        string root,
        string relativePath,
        bool isDirectory,
        uint shareMode = FileShareRead)
    {
        var path = relativePath.Length == 0
            ? root
            : Path.GetFullPath(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)));
        EnsureWithinRoot(root, path);
        if (isDirectory) EnsureDirectory(path, "component directory");
        else EnsureFile(path, "component file");

        if (!OperatingSystem.IsWindows())
        {
            var portable = isDirectory ? null : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            return new GuardedPath(path, isDirectory, null, portable, default);
        }

        var flags = FileFlagOpenReparsePoint | (isDirectory ? FileFlagBackupSemantics : 0);
        var handle = OpenNoFollow(path, flags, shareMode);
        try
        {
            var information = GetInformation(handle, path);
            var handleIsDirectory = (information.FileAttributes & FileAttributeDirectory) != 0;
            if (handleIsDirectory != isDirectory) throw new ComponentActivationException("A guarded component path has the wrong file type.");
            return new GuardedPath(path, isDirectory, handle, null, information);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    private static void EnsureWithinRoot(string root, string path)
    {
        var relative = Path.GetRelativePath(root, path);
        if (Path.IsPathRooted(relative) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)) throw new ComponentActivationException("A guarded component path escapes its candidate root.");
    }

    private static void EnsureDirectory(string path, string description)
    {
        if (!Directory.Exists(path) || (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new ComponentActivationException($"The {description} is missing or a reparse point.");
    }

    private static void EnsureFile(string path, string description)
    {
        if (!File.Exists(path) || (File.GetAttributes(path) & (FileAttributes.Directory | FileAttributes.ReparsePoint)) != 0) throw new ComponentActivationException($"The {description} is missing or a reparse point.");
    }

    private static SafeFileHandle OpenNoFollow(string path, uint flags, uint shareMode)
    {
        var handle = CreateFile(path, GenericRead, shareMode, IntPtr.Zero, OpenExisting, flags, IntPtr.Zero);
        if (handle.IsInvalid) throw new ComponentActivationException("The component path could not be guarded.", new Win32Exception(Marshal.GetLastWin32Error()));
        return handle;
    }

    private static ByHandleFileInformation GetInformation(SafeFileHandle handle, string path)
    {
        if (!GetFileInformationByHandle(handle, out var information)) throw new ComponentActivationException($"The guarded component path could not be inspected: {path}.", new Win32Exception(Marshal.GetLastWin32Error()));
        if ((information.FileAttributes & FileAttributeReparsePoint) != 0) throw new ComponentActivationException("A guarded component path is a reparse point.");
        return information;
    }

    private static bool SameIdentity(ByHandleFileInformation left, ByHandleFileInformation right) =>
        left.VolumeSerialNumber == right.VolumeSerialNumber && left.FileIndexHigh == right.FileIndexHigh && left.FileIndexLow == right.FileIndexLow;

    private static void DisposeReverse(IEnumerable<GuardedPath> paths)
    {
        foreach (var path in paths.Reverse())
        {
            path.PortableHandle?.Dispose();
            path.NativeHandle?.Dispose();
        }
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFile(string fileName, uint desiredAccess, uint shareMode, IntPtr securityAttributes, uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetFileInformationByHandle(SafeFileHandle file, out ByHandleFileInformation information);

    private sealed record GuardedPath(string Path, bool IsDirectory, SafeFileHandle? NativeHandle, FileStream? PortableHandle, ByHandleFileInformation Identity);

    [StructLayout(LayoutKind.Sequential)]
    private struct ByHandleFileInformation
    {
        public uint FileAttributes;
        public uint CreationTimeLow;
        public uint CreationTimeHigh;
        public uint LastAccessTimeLow;
        public uint LastAccessTimeHigh;
        public uint LastWriteTimeLow;
        public uint LastWriteTimeHigh;
        public uint VolumeSerialNumber;
        public uint FileSizeHigh;
        public uint FileSizeLow;
        public uint NumberOfLinks;
        public uint FileIndexHigh;
        public uint FileIndexLow;
    }
}
