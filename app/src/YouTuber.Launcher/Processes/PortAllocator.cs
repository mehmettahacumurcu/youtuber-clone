using System.Net;
using System.Net.Sockets;

namespace YouTuber.Launcher.Processes;

/// <summary>Reserves loopback ports until each associated worker is about to start.</summary>
public sealed class PortAllocator
{
    public PortAllocation Reserve(int count = 3)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        var reservations = new List<PortReservation>(count);
        try
        {
            for (var index = 0; index < count; index++) reservations.Add(PortReservation.Create());
            return new PortAllocation(reservations);
        }
        catch
        {
            foreach (var reservation in reservations) reservation.Dispose();
            throw;
        }
    }
}

public sealed class PortAllocation : IDisposable
{
    private readonly IReadOnlyList<PortReservation> _reservations;
    internal PortAllocation(IReadOnlyList<PortReservation> reservations) => _reservations = reservations;
    public IReadOnlyList<int> Ports => _reservations.Select(reservation => reservation.Port).ToArray();
    public PortReservation this[int index] => _reservations[index];
    public void Dispose()
    {
        foreach (var reservation in _reservations) reservation.Dispose();
    }
}

public sealed class PortReservation : IDisposable
{
    private TcpListener? _listener;
    private PortReservation(TcpListener listener) { _listener = listener; Port = ((IPEndPoint)listener.LocalEndpoint).Port; }
    public int Port { get; }
    public bool IsReleased => _listener is null;
    internal static PortReservation Create()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        return new PortReservation(listener);
    }
    /// <summary>Releases the bound socket immediately before Process.Start.</summary>
    public void Release() => Interlocked.Exchange(ref _listener, null)?.Stop();
    public void Dispose() => Release();
}
