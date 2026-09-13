using System.Windows.Controls;
using YouTuber.Launcher.ViewModels;

namespace YouTuber.Launcher.Views;

public partial class DataRootSelectionView : UserControl
{
    public DataRootSelectionView(DataRootSelectionViewModel viewModel)
    {
        ArgumentNullException.ThrowIfNull(viewModel);
        InitializeComponent();
        DataContext = viewModel;
    }
}
