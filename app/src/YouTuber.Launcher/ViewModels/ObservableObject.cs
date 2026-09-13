using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace YouTuber.Launcher.ViewModels;
public abstract class ObservableObject : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;
    protected void Raise([CallerMemberName] string? name = null) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
