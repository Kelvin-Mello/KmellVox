; ==============================================================================
; Inno Setup Script para KmellVox Studio (Windows x64)
; ==============================================================================

#define MyAppName "KmellVox Studio"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "KmellVox AI Team"
#define MyAppURL "https://github.com/KmellVox"
#define MyAppExeName "KmellVox.exe"

[Setup]
AppId={{C8E2B5A0-1A79-4FD1-8A15-983C4B82F10E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\KmellVox
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=dist\installer
OutputBaseFilename=KmellVox_Setup_v1.0.0
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\KmellVox\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Na primeira execução pós-instalação, abre o app com --first-run para download inicial dos modelos
Filename: "{app}\{#MyAppExeName}"; Parameters: "--first-run"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}} (abrir gerenciador de modelos)"; Flags: nowait postinstall skipifsilent
