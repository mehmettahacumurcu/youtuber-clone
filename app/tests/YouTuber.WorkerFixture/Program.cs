using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Net;

var options = args.Select(argument => argument.Split('=', 2)).ToDictionary(parts => parts[0], parts => parts.Length == 2 ? parts[1] : "true", StringComparer.OrdinalIgnoreCase);
if (options.ContainsKey("--component-healthcheck"))
{
    var expectedRoot = options["--expected-root"];
    var forbidden = options["--forbidden-env"];
    var valid = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("SystemRoot"))
        && !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("WINDIR"))
        && string.Equals(Environment.GetEnvironmentVariable("YOUTUBER_COMPONENT_ROOT"), expectedRoot, StringComparison.OrdinalIgnoreCase)
        && string.Equals(Environment.GetEnvironmentVariable("YOUTUBER_HEALTHCHECK"), "1", StringComparison.Ordinal)
        && Environment.GetEnvironmentVariable(forbidden) is null;
    Environment.Exit(valid ? 0 : 23);
}
if (options.ContainsKey("--child"))
{
    await Task.Delay(Timeout.InfiniteTimeSpan);
    return;
}
var secret = Environment.GetEnvironmentVariable("YOUTUBER_SESSION_SECRET") ?? string.Empty;
if (Environment.GetEnvironmentVariable("YOUTUBER_FIXTURE_PID_FILE") is { Length: > 0 } pidFile)
    File.AppendAllText(pidFile, Environment.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture) + Environment.NewLine);
if (Environment.GetEnvironmentVariable("YOUTUBER_FIXTURE_START_FILE") is { Length: > 0 } startFile)
    File.AppendAllText(startFile, Environment.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture) + Environment.NewLine);
Console.WriteLine("fixture stdout " + secret);
Console.Error.WriteLine("fixture stderr " + secret);
if (options.TryGetValue("--probe-handle", out var handleText))
{
    var inherited = GetHandleInformation(new IntPtr(long.Parse(handleText, System.Globalization.CultureInfo.InvariantCulture)), out _);
    Console.WriteLine("fixture probe handle inherited=" + inherited.ToString().ToLowerInvariant());
    if (options.ContainsKey("--exit-after-probe")) Environment.Exit(0);
}
if (options.ContainsKey("--spawn-child"))
{
    var child = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false };
    child.ArgumentList.Add(Environment.ProcessPath!.EndsWith("dotnet.exe", StringComparison.OrdinalIgnoreCase) ? AppContext.BaseDirectory + "YouTuber.WorkerFixture.dll" : "--child");
    child.ArgumentList.Add("--child");
    var childProcess = Process.Start(child)!;
    if (Environment.GetEnvironmentVariable("YOUTUBER_CHILD_PID_FILE") is { Length: > 0 } childPidFile) File.WriteAllText(childPidFile, childProcess.Id.ToString(System.Globalization.CultureInfo.InvariantCulture));
    if (options.ContainsKey("--exit-leaving-child")) Environment.Exit(0);
}
var port = options.TryGetValue("--port-env", out var portEnvironmentVariable)
    ? int.Parse(Environment.GetEnvironmentVariable(portEnvironmentVariable) ?? throw new InvalidOperationException($"Missing {portEnvironmentVariable}."), System.Globalization.CultureInfo.InvariantCulture)
    : int.Parse(options["--port"], System.Globalization.CultureInfo.InvariantCulture);
using var listener = new HttpListener();
listener.Prefixes.Add($"http://127.0.0.1:{port}/");
listener.Start();
if (Environment.GetEnvironmentVariable("YOUTUBER_FIXTURE_PORT_FILE") is { Length: > 0 } portFile)
    File.WriteAllText(portFile, port.ToString(System.Globalization.CultureInfo.InvariantCulture));
var healthRequests = 0;
while (true)
{
    var context = await listener.GetContextAsync();
    if (options.ContainsKey("--hang-health")) continue;
    var delay = options.TryGetValue("--delay-ms", out var value) ? int.Parse(value, System.Globalization.CultureInfo.InvariantCulture) : 0;
    if (delay > 0) await Task.Delay(delay);
    var expected = "Bearer " + secret;
    var authorized = string.Equals(context.Request.Headers["Authorization"], expected, StringComparison.Ordinal);
    if (options.ContainsKey("--studio")) await RespondAsStudioAsync(context, authorized);
    else
    {
        context.Response.StatusCode = authorized ? 200 : 401;
        context.Response.Close();
    }
    healthRequests++;
    if (options.ContainsKey("--crash") && healthRequests >= 2) Environment.Exit(17);
    if (options.ContainsKey("--burst") && healthRequests >= 2)
    {
        for (var index = 0; index < 256; index++)
        {
            Console.WriteLine($"fixture burst stdout {index:D4} {secret}");
            Console.Error.WriteLine($"fixture burst stderr {index:D4} {secret}");
        }
        Console.WriteLine("fixture final stdout marker " + secret);
        Console.Error.WriteLine("fixture final stderr marker " + secret);
        Environment.Exit(0);
    }
}

static async Task RespondAsStudioAsync(HttpListenerContext context, bool bearerAuthorized)
{
    var path = context.Request.Url?.AbsolutePath ?? string.Empty;
    if (Environment.GetEnvironmentVariable("YOUTUBER_FIXTURE_STUDIO_TRACE_FILE") is { Length: > 0 } trace)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(trace)!);
        File.AppendAllText(trace, $"{context.Request.HttpMethod} {path} bearer={bearerAuthorized} cookie={context.Request.Cookies["youtuber_fixture"]?.Value ?? "-"}\n");
    }
    if (path is "/health/live" or "/health/ready")
    {
        context.Response.StatusCode = bearerAuthorized ? 200 : 401;
        context.Response.Close();
        return;
    }
    if (path == "/bootstrap" && context.Request.HttpMethod == "POST")
    {
        if (!bearerAuthorized) { context.Response.StatusCode = 401; context.Response.Close(); return; }
        context.Response.AppendHeader("Set-Cookie", "youtuber_fixture=ready; Path=/; HttpOnly; SameSite=Strict");
        context.Response.StatusCode = 303;
        context.Response.RedirectLocation = "/";
        context.Response.Close();
        return;
    }
    if (path == "/v1/browser-bootstrap" && context.Request.HttpMethod == "POST")
    {
        if (!bearerAuthorized) { context.Response.StatusCode = 401; context.Response.Close(); return; }
        await WriteAsync(context.Response, "application/json", "{\"url\":\"http://127.0.0.1:" + context.Request.LocalEndPoint!.Port.ToString(System.Globalization.CultureInfo.InvariantCulture) + "/v1/browser-exchange?nonce=AAAAAAAAAAAAAAAAAAAAAA\"}");
        return;
    }
    if (path == "/v1/browser-exchange" && context.Request.QueryString["nonce"] == "AAAAAAAAAAAAAAAAAAAAAA")
    {
        context.Response.AppendHeader("Set-Cookie", "youtuber_fixture=ready; Path=/; HttpOnly; SameSite=Strict");
        context.Response.StatusCode = 303;
        context.Response.RedirectLocation = "/";
        context.Response.Close();
        return;
    }
    if (path == "/" && context.Request.Cookies["youtuber_fixture"]?.Value == "ready")
    {
        if (Environment.GetEnvironmentVariable("YOUTUBER_FIXTURE_STUDIO_READY_FILE") is { Length: > 0 } marker)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(marker)!);
            File.WriteAllText(marker, "ready\n");
        }
        await WriteAsync(context.Response, "text/html; charset=utf-8", "<!doctype html><title>YouTuber Studio fixture</title><main>Fixture Studio ready</main>");
        return;
    }
    context.Response.StatusCode = 404;
    context.Response.Close();
}

static async Task WriteAsync(HttpListenerResponse response, string contentType, string body)
{
    var bytes = System.Text.Encoding.UTF8.GetBytes(body);
    response.StatusCode = 200;
    response.ContentType = contentType;
    response.ContentLength64 = bytes.Length;
    await response.OutputStream.WriteAsync(bytes);
    response.Close();
}

[DllImport("kernel32.dll", SetLastError = true)]
[return: MarshalAs(UnmanagedType.Bool)]
static extern bool GetHandleInformation(IntPtr handle, out uint flags);
