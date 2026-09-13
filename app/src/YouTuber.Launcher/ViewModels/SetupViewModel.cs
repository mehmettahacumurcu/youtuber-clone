using System.Collections.ObjectModel;
using YouTuber.Launcher.Setup;

namespace YouTuber.Launcher.ViewModels;
public sealed class SetupViewModel : ObservableObject
{
    private readonly SetupCoordinator _coordinator;
    private readonly IUiDispatcher _dispatcher;
    private long _lastAppliedRevision = -1;
    public ObservableCollection<ComponentProgressViewModel> Components { get; } = [];
    public SetupStage Stage => _coordinator.State.Stage;
    public string? Error => _coordinator.State.BlockingError;
    public bool CanOpenChat => _coordinator.State.ReadyRecorded;
    public bool CanAcceptTerms => _coordinator.State.Stage == SetupStage.AcceptTerms && !_coordinator.State.TermsAccepted;
    public SetupViewModel(SetupCoordinator coordinator, IUiDispatcher? dispatcher = null)
    {
        _coordinator = coordinator;
        _dispatcher = dispatcher ?? new WpfUiDispatcher();
        _coordinator.Changed += (_, state) => _dispatcher.Post(() => Apply(state));
        Apply(_coordinator.State);
    }
    public void AcceptTerms()
    {
        if (CanAcceptTerms) _coordinator.AcceptTerms();
    }
    public Task<bool> RequestCloseAsync(CancellationToken cancellationToken = default) => _coordinator.RequestCloseAsync(cancellationToken);
    private void Apply(SetupState state)
    {
        if (state.Revision <= _lastAppliedRevision) return;
        _lastAppliedRevision = state.Revision;
        foreach (var removed in Components.Where(value => state.Components.All(component => component.Name != value.Name)).ToArray()) Components.Remove(removed);
        foreach (var component in state.Components)
        {
            var existing = Components.FirstOrDefault(value => value.Name == component.Name);
            if (existing is null) Components.Add(new ComponentProgressViewModel(component, _coordinator));
            else existing.Update(component);
        }
        Raise(nameof(Stage)); Raise(nameof(Error)); Raise(nameof(CanOpenChat)); Raise(nameof(CanAcceptTerms));
    }
}
