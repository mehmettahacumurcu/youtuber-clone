using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace YouTuber.Launcher.Processes;

public interface IJobProcess
{
    IntPtr NativeHandle { get; }
}

public interface IWorkerJob : IDisposable
{
    void Assign(IJobProcess process);
}

/// <summary>Owns worker descendants even if the launcher exits unexpectedly.</summary>
public sealed class WindowsJob : IWorkerJob
{
    private readonly SafeFileHandle? _handle;
    public WindowsJob()
    {
        if (!OperatingSystem.IsWindows()) return;
        _handle = CreateJobObject(IntPtr.Zero, null);
        if (_handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to create worker job.");
        var information = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            BasicLimitInformation = new JOBOBJECT_BASIC_LIMIT_INFORMATION { LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE },
        };
        var length = Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
        var buffer = Marshal.AllocHGlobal(length);
        try
        {
            Marshal.StructureToPtr(information, buffer, false);
            if (!SetInformationJobObject(_handle, JobObjectExtendedLimitInformation, buffer, (uint)length))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to configure worker job.");
            }
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }

    public void Assign(IJobProcess process)
    {
        ArgumentNullException.ThrowIfNull(process);
        if (_handle is null) throw new InvalidOperationException("A Windows kill-on-close job is unavailable.");
        if (process.NativeHandle == IntPtr.Zero) throw new InvalidOperationException("A process cannot be resumed without a valid job-assignment handle.");
        if (!AssignProcessToJobObject(_handle, process.NativeHandle))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to assign worker to job.");
        }
    }

    public void Dispose() => _handle?.Dispose();

    private const uint JobObjectExtendedLimitInformation = 9;
    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    [StructLayout(LayoutKind.Sequential)] private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)] private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }
    [StructLayout(LayoutKind.Sequential)] private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] private static extern SafeFileHandle CreateJobObject(IntPtr attributes, string? name);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool SetInformationJobObject(SafeFileHandle job, uint informationClass, IntPtr information, uint length);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool AssignProcessToJobObject(SafeFileHandle job, IntPtr process);
}
