using System.Windows.Input;
namespace YouTuber.Launcher.ViewModels;
public sealed class RelayCommand(Action execute, Func<bool>? canExecute = null) : ICommand
{
    public event EventHandler? CanExecuteChanged;
    public bool CanExecute(object? parameter) => canExecute?.Invoke() ?? true;
    public void Execute(object? parameter) => execute();
    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}

public sealed class AsyncRelayCommand(Func<CancellationToken, Task> execute, Func<bool>? canExecute = null) : ICommand
{
    private int _executing;
    public event EventHandler? CanExecuteChanged;
    public bool CanExecute(object? parameter) => Volatile.Read(ref _executing) == 0 && (canExecute?.Invoke() ?? true);
    public async void Execute(object? parameter) => await ExecuteAsync();
    public async Task ExecuteAsync(CancellationToken cancellationToken = default)
    {
        if (!CanExecute(null) || Interlocked.Exchange(ref _executing, 1) != 0) return;
        RaiseCanExecuteChanged();
        try { await execute(cancellationToken); }
        finally { Volatile.Write(ref _executing, 0); RaiseCanExecuteChanged(); }
    }
    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
