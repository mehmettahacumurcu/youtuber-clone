using System.Text.Json;
using System.IO;
using System.Collections.Concurrent;
namespace YouTuber.Launcher.Setup;
public sealed class SetupJournalConflictException(string message) : IOException(message);
public sealed class AtomicFileSetupJournal(string path) : ISetupJournal
{
    private static readonly ConcurrentDictionary<string, object> Gates = new(StringComparer.OrdinalIgnoreCase);
    private readonly string _fullPath = Path.GetFullPath(path);

    public SetupState? Load()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(_fullPath) ?? ".");
        return WithExclusiveLock(LoadCore);
    }
    public void Save(SetupState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        Directory.CreateDirectory(Path.GetDirectoryName(_fullPath) ?? ".");
        WithExclusiveLock(() => { SaveCore(state); return true; });
    }

    private SetupState? LoadCore()
    {
        if (!File.Exists(_fullPath)) return null;
        try { return JsonSerializer.Deserialize<SetupState>(File.ReadAllText(_fullPath)) ?? throw new IOException("Setup journal is empty."); }
        catch (JsonException exception) { throw new IOException("Setup journal is corrupt.", exception); }
    }

    private void SaveCore(SetupState state)
    {
        var current = LoadCore();
        if (current is not null && state.Revision <= current.Revision)
        {
            throw new SetupJournalConflictException("The setup journal changed before this checkpoint could be committed.");
        }

        var temporary = _fullPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.WriteThrough))
            {
                JsonSerializer.Serialize(stream, state);
                stream.Flush(flushToDisk: true);
            }
            if (File.Exists(_fullPath)) File.Replace(temporary, _fullPath, destinationBackupFileName: null);
            else File.Move(temporary, _fullPath);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
        }
    }

    private T WithExclusiveLock<T>(Func<T> action)
    {
        var gate = Gates.GetOrAdd(_fullPath, static _ => new object());
        lock (gate)
        {
            var lockPath = _fullPath + ".lock";
            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
            FileStream processLock;
            while (true)
            {
                try
                {
                    processLock = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.WriteThrough);
                    break;
                }
                catch (IOException) when (DateTime.UtcNow < deadline)
                {
                    Thread.Sleep(10);
                }
            }
            using (processLock) return action();
        }
    }
}
