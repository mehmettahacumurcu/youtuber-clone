#ifndef AppVersion
  #error AppVersion must be supplied by build-installer.ps1
#endif
#ifndef PublishDir
  #error PublishDir must be supplied by build-installer.ps1
#endif
#ifndef AssetsDir
  #error AssetsDir must be supplied by build-installer.ps1
#endif
#ifndef BootstrapCatalog
  #error BootstrapCatalog must be supplied by build-installer.ps1
#endif
#ifndef InstallerOutputDir
  #error InstallerOutputDir must be supplied by build-installer.ps1
#endif

[Setup]
AppId={{F324E3A6-7C5D-4BDF-92DD-F00A307B3F1E}
AppName=YouTuber Studio
AppVersion={#AppVersion}
AppVerName=YouTuber Studio {#AppVersion}
AppPublisher=YouTuber Studio Contributors
AppPublisherURL=https://github.com/mehmettahacumurcu/youtuber-clone
AppSupportURL=https://github.com/mehmettahacumurcu/youtuber-clone/issues
DefaultDirName={localappdata}\Programs\YouTuberStudio
DefaultGroupName=YouTuber Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19045
WizardStyle=modern dynamic
SetupIconFile={#AssetsDir}\youtuber.ico
UninstallDisplayIcon={app}\YouTuberStudio.exe
LicenseFile={#AssetsDir}\LICENSE-code.txt
InfoBeforeFile={#AssetsDir}\AI-DISCLOSURE.txt
InfoAfterFile={#AssetsDir}\README-SMARTSCREEN.txt
OutputDir={#InstallerOutputDir}
OutputBaseFilename=YouTuberStudio-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
CloseApplications=yes
CloseApplicationsFilter=YouTuberStudio.exe
RestartApplications=no
AppMutex=YouTuberStudio.Launcher
#ifdef FixtureMode
UsePreviousAppDir=no
#else
UsePreviousAppDir=yes
#endif
ChangesAssociations=no
ChangesEnvironment=no
Uninstallable=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#PublishDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BootstrapCatalog}"; DestDir: "{app}\resources"; DestName: "bootstrap-catalog.json"; Flags: ignoreversion
Source: "{#AssetsDir}\LICENSE-code.txt"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "{#AssetsDir}\LICENSE-data.txt"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "{#AssetsDir}\AI-DISCLOSURE.txt"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "{#AssetsDir}\THIRD-PARTY-NOTICES.txt"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "{#AssetsDir}\README-SMARTSCREEN.txt"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "{#AssetsDir}\youtuber.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
#ifdef FixtureMode
Source: "{#PublishDir}\YouTuber.WorkerFixture*"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\YouTuber Studio"; Filename: "{app}\YouTuberStudio.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\YouTuber Studio"; Filename: "{app}\YouTuberStudio.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\YouTuberStudio.exe"; Description: "Launch YouTuber Studio"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
#ifdef FixtureMode
; The {code:GetFixtureDataRoot} value is expanded while Setup records this UninstallRun entry,
; so the harness sets YOUTUBER_FIXTURE_DATA_ROOT only around the installer process. The resulting
; data-root argument is persisted in the uninstaller and is intentionally independent of its environment.
Filename: "{app}\YouTuberStudio.exe"; Parameters: "--prepare-uninstall --uninstall-choice-file --data-root=""{code:GetFixtureDataRoot}"""; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "YouTuberPrepareUninstall"
#else
Filename: "{app}\YouTuberStudio.exe"; Parameters: "--prepare-uninstall --uninstall-choice-file"; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "YouTuberPrepareUninstall"
#endif

[UninstallDelete]
Type: files; Name: "{app}\uninstall-choice.txt"

[Code]
var
  RemoveOwnedData: Boolean;

function CmdLineParamExists(Param: String): Boolean;
begin
  Result := Pos(' /' + Uppercase(Param) + ' ', ' ' + Uppercase(GetCmdTail) + ' ') > 0;
end;

function GetUninstallChoice(Param: String): String;
begin
  if RemoveOwnedData then
    Result := 'app-and-owned-data'
  else
    Result := 'app-only';
end;

function InitializeUninstall(): Boolean;
var
  UninstallChoice: AnsiString;
begin
  if CmdLineParamExists('REMOVEOWNEDDATA') then
    RemoveOwnedData := True
  else
    RemoveOwnedData := SuppressibleMsgBox(
    'Remove YouTuber-owned downloaded models, RAG data, voice files, and cache too?' + #13#10 + #13#10 +
    'Choose No to remove only the application and keep downloaded data for a future reinstall.',
    mbConfirmation, MB_YESNO, IDNO) = IDYES;
  UninstallChoice := GetUninstallChoice('');
  if not SaveStringToFile(ExpandConstant('{app}\uninstall-choice.txt'), UninstallChoice, False) then
    RaiseException('Unable to save the uninstall data choice.');
  Result := True;
end;

#ifdef FixtureMode
function GetFixtureDataRoot(Param: String): String;
begin
  Result := Trim(GetEnv('YOUTUBER_FIXTURE_DATA_ROOT'));
  if Result = '' then
    RaiseException('Fixture uninstall data root is missing.');
end;
#endif
