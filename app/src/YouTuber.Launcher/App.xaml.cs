using System.Windows;
using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Setup;
using YouTuber.Launcher.ViewModels;
using YouTuber.Launcher.Uninstall;
using YouTuber.Launcher.Web;

namespace YouTuber.Launcher;

public partial class App : Application
{
    private Mutex? _singleInstance;
    private IWorkerSupervisor? _workers;
    private LauncherSessionPlan? _plan;
    private SetupViewModel? _setup;
    private ProductionSetupStageActions? _setupActions;
    private Task? _setupRun;
    private readonly CancellationTokenSource _lifetime = new();
    private bool _closing;
    private bool _allowClose;

    protected override async void OnStartup(StartupEventArgs eventArgs)
    {
        base.OnStartup(eventArgs);
        if (UninstallPreparation.IsRequested(eventArgs.Args))
        {
            ShutdownMode = ShutdownMode.OnExplicitShutdown;
            try
            {
                await UninstallPreparation.RunAsync(eventArgs.Args, _lifetime.Token);
                Shutdown(0);
            }
            catch
            {
                Shutdown(1);
            }
            return;
        }
        _singleInstance = new Mutex(initiallyOwned: true, name: "YouTuberStudio.Launcher", createdNew: out var firstInstance);
        if (!firstInstance)
        {
            _singleInstance.Dispose();
            _singleInstance = null;
            Shutdown(0);
            return;
        }
        var window = new MainWindow();
        MainWindow = window;
        window.Closing += HandleWindowClosing;
        window.Show();
        try
        {
            if (!eventArgs.Args.Contains("--fixture-workers", StringComparer.Ordinal))
            {
                var locator = DataRootLocator.ForCurrentUser();
                var dataRootArgument = eventArgs.Args.SingleOrDefault(argument => argument.StartsWith("--data-root=", StringComparison.Ordinal));
                string dataRoot;
                if (dataRootArgument is not null)
                {
                    await locator.SaveAsync(dataRootArgument["--data-root=".Length..], _lifetime.Token);
                    dataRoot = locator.LoadRequired();
                }
                else if (locator.TryLoad() is { } persistedDataRoot)
                {
                    dataRoot = persistedDataRoot;
                }
                else
                {
                    var defaultDataRoot = AppPaths.ForCurrentUser().DataRoot;
                    var selection = new DataRootSelectionViewModel(locator, defaultDataRoot, new WindowsDataRootPicker());
                    window.ShowDataRootSelection(selection);
                    dataRoot = await selection.Selection.WaitAsync(_lifetime.Token);
                }

                var paths = AppPaths.ForCurrentUser(dataRoot);
                await ProductionOwnershipBootstrap.InitializeAsync(paths, _lifetime.Token);
                _setupActions = new ProductionSetupStageActions(paths);
                var coordinator = await _setupActions.CreateCoordinatorAsync(new AtomicFileSetupJournal(paths.SetupJournalFile), _lifetime.Token);
                var runner = new FirstRunSetupRunner(coordinator, _setupActions);
                async Task ContinueSetupAsync(CancellationToken token)
                {
                    if (_setupRun is not null) return;
                    using var linked = CancellationTokenSource.CreateLinkedTokenSource(token, _lifetime.Token);
                    var run = runner.RunAsync(linked.Token);
                    _setupRun = run;
                    try
                    {
                        var result = await run;
                        if (!result.IsReady) return;
                        await StartRetainedProductionSessionAsync(window, _setupActions, linked.Token);
                    }
                    finally
                    {
                        if (ReferenceEquals(_setupRun, run)) _setupRun = null;
                    }
                }
                void ShowRepairableSetup()
                {
                    _setup = new SetupViewModel(coordinator);
                    window.ShowSetup(_setup, new Views.SetupViewActions(ContinueSetupAsync, ContinueSetupAsync));
                }

                if (!LauncherSessionPlanFactory.IsSetupReady(coordinator.State))
                {
                    ShowRepairableSetup();
                    return;
                }

                try
                {
                    await runner.RevalidateReadyAsync(_lifetime.Token);
                }
                catch (OperationCanceledException) when (_lifetime.IsCancellationRequested) { throw; }
                catch
                {
                    ShowRepairableSetup();
                    return;
                }
                await StartRetainedProductionSessionAsync(window, _setupActions, _lifetime.Token);
                return;
            }
            _plan = await LauncherSessionPlanFactory.CreateAsync(eventArgs.Args, _lifetime.Token);
            IWorkerDependencyHealth dependencies = _plan.OllamaEndpoint is null
                ? new NoopWorkerDependencyHealth()
                : new LoopbackOllamaDependencyHealth(_plan.OllamaEndpoint);
            _workers = new WorkerSupervisor(dependencies: dependencies);
            await new LauncherStartupController(_workers).StartAsync(window, _plan, _lifetime.Token);
        }
        catch (OperationCanceledException) when (_lifetime.IsCancellationRequested) { }
        catch (Exception exception)
        {
            _ = exception;
#if FIXTURE_BUILD
            if (eventArgs.Args.Contains("--fixture-workers", StringComparer.Ordinal))
            {
                var dataRoot = eventArgs.Args.SingleOrDefault(argument => argument.StartsWith("--data-root=", StringComparison.Ordinal))?["--data-root=".Length..];
                if (!string.IsNullOrWhiteSpace(dataRoot))
                {
                    var state = System.IO.Path.Combine(dataRoot, "state");
                    System.IO.Directory.CreateDirectory(state);
                    System.IO.File.WriteAllText(System.IO.Path.Combine(state, "fixture-startup-error.log"), exception.GetType().FullName + ": " + exception.Message);
                }
                Shutdown(1);
                return;
            }
#endif
#if DEBUG
            if (_plan is not null)
            {
                var state = System.IO.Path.Combine(_plan.DataRoot, "state");
                System.IO.Directory.CreateDirectory(state);
                System.IO.File.WriteAllText(System.IO.Path.Combine(state, "startup-error.log"), exception.GetType().FullName + ": " + exception.Message);
            }
#endif
            window.ShowStartupFailure("YouTuber Studio could not start its local services. Complete setup or run Repair, then retry.");
        }
    }

    private async Task StartRetainedProductionSessionAsync(MainWindow window, ProductionSetupStageActions actions, CancellationToken cancellationToken)
    {
        if (!actions.HasRunningSession) throw new InvalidOperationException("Setup reached Ready without retaining its verified production runtime.");
        var session = actions.RunningSession;
        _plan = session.SessionPlan ?? throw new InvalidOperationException("The verified production runtime did not provide a launch plan.");
        _workers = session.Workers;
        _setup = null;
        window.ShowStudioShell();
        await new LauncherStartupController(_workers).AttachRunningAsync(window, _plan, cancellationToken);
    }

    private async void HandleWindowClosing(object? sender, System.ComponentModel.CancelEventArgs eventArgs)
    {
        if (_allowClose) return;
        eventArgs.Cancel = true;
        if (_closing) return;
        _closing = true;
        if (_setup is not null && sender is MainWindow setupWindow && !setupWindow.ConsumeSetupCloseApproval())
        {
            if (!await _setup.RequestCloseAsync())
            {
                _closing = false;
                return;
            }
        }
        _lifetime.Cancel();
        try
        {
            if (_setupRun is not null)
            {
                try { await _setupRun; }
                catch (OperationCanceledException) when (_lifetime.IsCancellationRequested) { }
                finally { _setupRun = null; }
            }
            if (_setupActions is not null)
            {
                await _setupActions.DisposeAsync();
                _setupActions = null;
                _workers = null;
            }
            else if (_workers is not null)
            {
                await _workers.DisposeAsync();
                _workers = null;
            }
        }
        finally
        {
            _plan?.Dispose();
            _plan = null;
            _setup = null;
            _allowClose = true;
            if (sender is Window window) window.Close();
        }
    }

    protected override void OnExit(ExitEventArgs eventArgs)
    {
        _lifetime.Cancel();
        _lifetime.Dispose();
        _plan?.Dispose();
        if (_singleInstance is not null)
        {
            _singleInstance.ReleaseMutex();
            _singleInstance.Dispose();
            _singleInstance = null;
        }
        base.OnExit(eventArgs);
    }
}
