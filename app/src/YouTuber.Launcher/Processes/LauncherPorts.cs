namespace YouTuber.Launcher.Processes;

/// <summary>Single-launch port composition. Reservations are owned until their respective native worker starts.</summary>
public sealed class LauncherPorts : IDisposable
{
    private readonly PortAllocation _allocation;
    private LauncherPorts(PortAllocation allocation) => _allocation = allocation;
    public int OllamaPort => _allocation[0].Port;
    public int RagPort => _allocation[1].Port;
    public int VoicePort => _allocation[2].Port;
    public int StudioPort => _allocation[3].Port;
    public IReadOnlyList<PortReservation> Reservations => [_allocation[0], _allocation[1], _allocation[2], _allocation[3]];
    public PortReservation OllamaReservation => _allocation[0];
    public PortReservation RagReservation => _allocation[1];
    public PortReservation VoiceReservation => _allocation[2];
    public PortReservation StudioReservation => _allocation[3];
    public static LauncherPorts Create(PortAllocator? allocator = null) => new((allocator ?? new PortAllocator()).Reserve(4));
    public LauncherWorkerComposition Compose(WorkerDefinition voice, WorkerDefinition studio)
    {
        ArgumentNullException.ThrowIfNull(voice);
        ArgumentNullException.ThrowIfNull(studio);
        return new LauncherWorkerComposition(
            voice.WithPort(VoiceReservation, "YOUTUBER_VOICE_PORT"),
            studio.WithPort(StudioReservation, "YOUTUBER_STUDIO_PORT").WithAdditionalPort(RagReservation, "YOUTUBER_RAG_PORT"),
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["YOUTUBER_OLLAMA_PORT"] = OllamaPort.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["YOUTUBER_OLLAMA_ORIGIN"] = $"http://127.0.0.1:{OllamaPort}",
                ["YOUTUBER_RAG_PORT"] = RagPort.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["YOUTUBER_VOICE_PORT"] = VoicePort.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["YOUTUBER_STUDIO_PORT"] = StudioPort.ToString(System.Globalization.CultureInfo.InvariantCulture),
            });
    }
    public void Dispose() => _allocation.Dispose();
}

public sealed record LauncherWorkerComposition(WorkerDefinition Voice, WorkerDefinition Studio, IReadOnlyDictionary<string, string> Settings);
