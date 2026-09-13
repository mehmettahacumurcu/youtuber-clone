using Microsoft.Web.WebView2.Core;

namespace YouTuber.Launcher.Web;

public sealed class WebViewPolicy(StudioOrigin origin)
{
    public StudioOrigin Origin { get; } = origin;
    public WebViewRestrictions Restrictions { get; } = WebViewRestrictions.LockedDown;
    public void Apply(CoreWebView2Settings settings)
    {
        settings.AreDefaultContextMenusEnabled = Restrictions.ContextMenus;
        settings.AreDevToolsEnabled = Restrictions.DevTools;
        settings.IsStatusBarEnabled = Restrictions.StatusBar;
        settings.IsPasswordAutosaveEnabled = Restrictions.PasswordAutosave;
        settings.IsGeneralAutofillEnabled = Restrictions.GeneralAutofill;
        settings.AreHostObjectsAllowed = Restrictions.HostObjects;
        settings.IsWebMessageEnabled = Restrictions.WebMessages;
        settings.AreDefaultScriptDialogsEnabled = Restrictions.ScriptDialogs;
        settings.IsZoomControlEnabled = false;
    }
    public bool Allows(string uri) => Origin.AllowsNavigation(uri);
}

public sealed record WebViewRestrictions(
    bool ContextMenus,
    bool DevTools,
    bool StatusBar,
    bool PasswordAutosave,
    bool GeneralAutofill,
    bool HostObjects,
    bool WebMessages,
    bool ScriptDialogs,
    bool ExternalDrop)
{
    public static WebViewRestrictions LockedDown { get; } = new(false, false, false, false, false, false, false, false, false);
}
