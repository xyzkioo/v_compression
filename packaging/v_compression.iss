[Setup]
AppId={{C0BBE697-96BC-4D69-97D9-D5497D22D8A1}
AppName=v_compression
AppVersion={#AppVersion}
AppPublisher=xyzkioo
DefaultDirName={localappdata}\Programs\v_compression
DefaultGroupName=v_compression
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=v_compression-{#AppVersion}-windows-x64-setup
VersionInfoVersion={#AppVersion}
VersionInfoProductVersion={#AppVersion}
UninstallDisplayIcon={app}\v_compression.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked

[Dirs]
Name: "{app}\inputs"; Flags: uninsneveruninstall
Name: "{app}\user-data"; Flags: uninsneveruninstall

[INI]
Filename: "{app}\.windows-install"; Section: "Installation"; Key: "Platform"; String: "Windows"; Flags: uninsdeletesection

[Files]
Source: "{#SourceDir}\v_compression.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\component.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion recursesubdirs
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\v_compression"; Filename: "{app}\v_compression.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\v_compression"; Filename: "{app}\v_compression.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\v_compression.exe"; Description: "{cm:LaunchProgram,v_compression}"; Flags: nowait postinstall skipifsilent
