using System.Text;
using System.IO;

namespace YouTuber.Launcher.Processes;

/// <summary>Thread-safe bounded UTF-8 append log. Rotation is performed before a record would exceed the limit.</summary>
public sealed class RotatingLogWriter : IDisposable
{
    public const long DefaultMaximumBytes = 10L * 1024 * 1024;
    public const int DefaultRotations = 5;
    private readonly object _gate = new();
    private readonly string _path;
    private readonly long _maximumBytes;
    private readonly int _rotations;
    private bool _disposed;

    public RotatingLogWriter(string directory, string fileName, long maximumBytes = DefaultMaximumBytes, int rotations = DefaultRotations)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);
        if (maximumBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maximumBytes));
        if (rotations < 1) throw new ArgumentOutOfRangeException(nameof(rotations));
        Directory.CreateDirectory(directory);
        _path = System.IO.Path.Combine(directory, fileName);
        _maximumBytes = maximumBytes;
        _rotations = rotations;
    }

    public string Path => _path;
    public void WriteLine(string value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var bytes = EncodeBoundedLine(value ?? string.Empty);
        lock (_gate)
        {
            RotateIfNeeded(bytes.Length);
            using var stream = new FileStream(_path, FileMode.Append, FileAccess.Write, FileShare.Read);
            stream.Write(bytes);
            stream.Flush(flushToDisk: true);
        }
    }

    private byte[] EncodeBoundedLine(string value)
    {
        var newline = Encoding.UTF8.GetBytes(Environment.NewLine);
        if (_maximumBytes < newline.Length) return [];
        var valueBytes = Encoding.UTF8.GetBytes(value);
        var length = Math.Min(valueBytes.Length, checked((int)(_maximumBytes - newline.Length)));
        while (length > 0 && length < valueBytes.Length && (valueBytes[length] & 0b1100_0000) == 0b1000_0000) length--;
        var result = new byte[length + newline.Length];
        Buffer.BlockCopy(valueBytes, 0, result, 0, length);
        Buffer.BlockCopy(newline, 0, result, length, newline.Length);
        return result;
    }

    private void RotateIfNeeded(int nextWriteBytes)
    {
        if (File.Exists(_path) && new FileInfo(_path).Length + nextWriteBytes > _maximumBytes)
        {
            var oldest = _path + "." + _rotations;
            if (File.Exists(oldest)) File.Delete(oldest);
            for (var index = _rotations - 1; index >= 1; index--)
            {
                var source = _path + "." + index;
                if (File.Exists(source)) File.Move(source, _path + "." + (index + 1), overwrite: true);
            }
            File.Move(_path, _path + ".1", overwrite: true);
        }
    }
    public void Dispose() => _disposed = true;
}
