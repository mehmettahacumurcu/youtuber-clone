using System.Windows;
using System.Windows.Threading;
using YouTuber.Launcher.Setup;

namespace YouTuber.Launcher.ViewModels;

public sealed class WpfUiDispatcher : IUiDispatcher
{
    private readonly Dispatcher? _dispatcher = Application.Current?.Dispatcher ?? Dispatcher.FromThread(Thread.CurrentThread);

    public void Post(Action action)
    {
        ArgumentNullException.ThrowIfNull(action);
        if (_dispatcher is null || _dispatcher.CheckAccess()) action();
        else _dispatcher.BeginInvoke(action);
    }
}
