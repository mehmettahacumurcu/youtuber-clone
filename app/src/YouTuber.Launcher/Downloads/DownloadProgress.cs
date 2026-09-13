using System.IO;

namespace YouTuber.Launcher.Downloads;

public enum DownloadState
{
    Queued,
    Downloading,
    Paused,
    Cancelled,
    Retrying,
    Verifying,
    Completed,
    Failed,
}

public sealed record DownloadProgress(DownloadState State, long BytesTransferred, long TotalBytes, double BytesPerSecond, TimeSpan? EstimatedRemaining, string? Message = null);

public sealed record DownloadResult(DownloadState State, string TargetPath, long BytesTransferred);

public sealed class DownloadValidationException(string message) : IOException(message);

public sealed class UnsafeRedirectException(string message) : IOException(message);
