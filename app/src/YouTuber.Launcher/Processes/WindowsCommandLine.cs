using System.Text;

namespace YouTuber.Launcher.Processes;

public static class WindowsCommandLine
{
    public static string Quote(string argument)
    {
        ArgumentNullException.ThrowIfNull(argument);
        if (argument.Length > 0 && !argument.Any(char.IsWhiteSpace) && !argument.Contains('"')) return argument;
        var result = new StringBuilder("\"");
        var slashes = 0;
        foreach (var character in argument)
        {
            if (character == '\\') { slashes++; continue; }
            if (character == '"') result.Append('\\', slashes * 2 + 1).Append(character);
            else result.Append('\\', slashes).Append(character);
            slashes = 0;
        }
        result.Append('\\', slashes * 2).Append('"');
        return result.ToString();
    }
}
