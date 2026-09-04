# build_release.ps1
# Script de build seguro do KmellVox - protege pastas de runtime (models/, python_env/, voices/).
#
# USO: .\build_release.ps1
#      .\build_release.ps1 -SkipModels  (nao copia modelos, util em CI sem GPU)

param(
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$DistDir     = Join-Path $ProjectRoot "dist\KmellVox"
$ModelsInDist = Join-Path $DistDir "models"
$ModelsInRepo = Join-Path $ProjectRoot "models"
$ModelsBackup = Join-Path $ProjectRoot "dist\_models_backup"

# Pastas e arquivos de runtime que devem ser preservados durante o rebuild
$RuntimeDirs = @(
    @{ Name = "python_env";  Path = (Join-Path $DistDir "python_env");  Backup = (Join-Path $ProjectRoot "dist\_python_env_backup") },
    @{ Name = "voices";      Path = (Join-Path $DistDir "voices");      Backup = (Join-Path $ProjectRoot "dist\_voices_backup") },
    @{ Name = "config.yaml"; Path = (Join-Path $DistDir "config.yaml"); Backup = (Join-Path $ProjectRoot "dist\_config_backup.yaml") }
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  KmellVox Studio - Build Release"                           -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Fecha processos em execução do KmellVox para evitar erro de arquivo travado
Get-Process -Name "KmellVox" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Fechando instancia em execucao do KmellVox (PID $($_.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force
    Start-Sleep -Milliseconds 500
}

# Sincroniza vozes, modelos e config salvos no dist para a raiz antes de qualquer build
if (Test-Path (Join-Path $DistDir "voices")) {
    robocopy (Join-Path $DistDir "voices") (Join-Path $ProjectRoot "voices") /E /XO /NP /NFL /NDL /NJH /NJS | Out-Null
}
if (Test-Path $ModelsInDist) {
    robocopy $ModelsInDist $ModelsInRepo /E /XO /NP /NFL /NDL /NJH /NJS | Out-Null
}
if (Test-Path (Join-Path $DistDir "config.yaml")) {
    Copy-Item (Join-Path $DistDir "config.yaml") (Join-Path $ProjectRoot "config.yaml") -Force
}

# 1. Faz backup dos modelos e pastas de runtime antes do --clean
if ((Test-Path $ModelsInDist) -and -not (Test-Path $ModelsBackup)) {
    Write-Host "[1/4] Fazendo backup de models/ antes do build..." -ForegroundColor Yellow
    Move-Item -Path $ModelsInDist -Destination $ModelsBackup -Force
    Write-Host "      Backup criado em: $ModelsBackup" -ForegroundColor Green
} else {
    Write-Host "[1/4] Nenhum backup necessario para models/." -ForegroundColor Gray
}

foreach ($dir in $RuntimeDirs) {
    if ((Test-Path $dir.Path) -and -not (Test-Path $dir.Backup)) {
        Write-Host "      Fazendo backup de $($dir.Name)..." -ForegroundColor Yellow
        Move-Item -Path $dir.Path -Destination $dir.Backup -Force
        Write-Host "      Backup de $($dir.Name) criado." -ForegroundColor Green
    }
}


# 2. Executa o PyInstaller
Write-Host "[2/4] Compilando com PyInstaller..." -ForegroundColor Yellow
& "$ProjectRoot\.venv\Scripts\pyinstaller.exe" --clean --noconfirm "$ProjectRoot\packaging\pyinstaller.spec"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO: PyInstaller falhou (codigo $LASTEXITCODE). Restaurando backup se existir..." -ForegroundColor Red
    if (Test-Path $ModelsBackup) {
        Copy-Item -Path $ModelsBackup -Destination $ModelsInDist -Recurse -Force
        Remove-Item $ModelsBackup -Recurse -Force
        Write-Host "      models/ restaurados do backup." -ForegroundColor Green
    }
    exit $LASTEXITCODE
}
Write-Host "      Build concluido com sucesso." -ForegroundColor Green

# 3. Restaura/vincula os modelos no dist
if (-not $SkipModels) {
    Write-Host "[3/4] Sincronizando modelos para dist\KmellVox\models\..." -ForegroundColor Yellow

    # Preferencia: restaurar backup (se existir e tiver conteudo real)
    $BackupHasContent = (Test-Path $ModelsBackup) -and ((Get-ChildItem $ModelsBackup -Recurse -File | Measure-Object).Count -gt 0)
    $RepoHasContent   = (Test-Path $ModelsInRepo) -and ((Get-ChildItem $ModelsInRepo -Recurse -File | Where-Object { $_.Extension -ne "" } | Measure-Object).Count -gt 0)

    if ($BackupHasContent) {
        if (Test-Path $ModelsInDist) { Remove-Item $ModelsInDist -Recurse -Force }
        Move-Item -Path $ModelsBackup -Destination $ModelsInDist -Force
        Write-Host "      models/ restaurados do backup." -ForegroundColor Green
    } elseif ($RepoHasContent) {
        if (Test-Path $ModelsInDist) { Remove-Item $ModelsInDist -Recurse -Force }
        Copy-Item -Path $ModelsInRepo -Destination $ModelsInDist -Recurse -Force
        Write-Host "      models/ copiados de $ModelsInRepo." -ForegroundColor Green
    } else {
        Write-Host "      AVISO: Nenhum modelo encontrado. Execute o downloader dentro do app." -ForegroundColor DarkYellow
        New-Item -ItemType Directory -Force -Path $ModelsInDist | Out-Null
    }
} else {
    Write-Host "[3/4] -SkipModels ativo: pasta models/ deixada vazia." -ForegroundColor Gray
}

# 3.5. Restaura pastas de runtime (python_env/, voices/)
foreach ($dir in $RuntimeDirs) {
    if (Test-Path $dir.Backup) {
        if (Test-Path $dir.Path) { Remove-Item $dir.Path -Recurse -Force }
        Move-Item -Path $dir.Backup -Destination $dir.Path -Force
        Write-Host "      $($dir.Name)/ restaurado do backup." -ForegroundColor Green
    }
}

# 3.6. Garante que as ferramentas e DLLs do FFmpeg estão sincronizadas em dist
$FFmpegSrc = Join-Path $ProjectRoot "tools\ffmpeg\bin"
if (Test-Path $FFmpegSrc) {
    $DistFFmpeg = Join-Path $DistDir "tools\ffmpeg\bin"
    $InternalFFmpeg = Join-Path $DistDir "_internal\tools\ffmpeg\bin"
    New-Item -ItemType Directory -Force -Path $DistFFmpeg | Out-Null
    New-Item -ItemType Directory -Force -Path $InternalFFmpeg | Out-Null
    Copy-Item -Path "$FFmpegSrc\*" -Destination $DistFFmpeg -Force -Recurse
    Copy-Item -Path "$FFmpegSrc\*" -Destination $InternalFFmpeg -Force -Recurse
    Write-Host "      FFmpeg binaries e DLLs sincronizados com sucesso." -ForegroundColor Green
}

# 4. Exibe resultado final
Write-Host "[4/4] Resultado final:" -ForegroundColor Yellow
$Exe = Get-Item (Join-Path $DistDir "KmellVox.exe") -ErrorAction SilentlyContinue
if ($Exe) {
    $SizeMB = [math]::Round($Exe.Length / 1MB, 1)
    Write-Host "      KmellVox.exe - $SizeMB MB - $($Exe.LastWriteTime)" -ForegroundColor Green
}
$ModelCount = (Get-ChildItem $ModelsInDist -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "      Modelos em dist\KmellVox\models\: $ModelCount arquivo(s)" -ForegroundColor Green
$PythonEnvDir = Join-Path $DistDir "python_env"
if (Test-Path $PythonEnvDir) {
    $EnvSizeMB = [math]::Round(((Get-ChildItem $PythonEnvDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB), 0)
    Write-Host "      python_env/: ${EnvSizeMB} MB (dependencias de voz)" -ForegroundColor Green
}
Write-Host ""
Write-Host "  Pronto! Executavel em: $DistDir" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
