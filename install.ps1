# NALA-Colibri-RS v2.2.0-nala.6.6.6 — Native Windows x86_64 PowerShell Installer (SHA-256 Verified)
$ErrorActionPreference = "Stop"
$Repo = "Master-MD/NALA-Colibri-RS"
$Version = if ($env:NALA_COLIBRI_VERSION) { $env:NALA_COLIBRI_VERSION } else { "v2.2.0-nala.6.6.6" }
$InstallDir = if ($env:NALA_INSTALL_DIR) { $env:NALA_INSTALL_DIR } else { Join-Path $HOME ".local\bin" }
$Asset = "nala-colibri-rs-windows-x86_64.exe"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("colibri_rs_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
    $BaseUrl = "https://github.com/$Repo/releases/download/$Version"
    Invoke-WebRequest -Uri "$BaseUrl/$Asset" -OutFile (Join-Path $Tmp $Asset)
    Invoke-WebRequest -Uri "$BaseUrl/SHA256SUMS.txt" -OutFile (Join-Path $Tmp "SHA256SUMS.txt")

    $Expected = (Select-String -Path (Join-Path $Tmp "SHA256SUMS.txt") -Pattern "  $Asset$").Line.Split(" ")[0].Trim().ToLower()
    $Actual = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $Tmp $Asset)).Hash.ToLower()
    if (-not $Expected -or ($Expected -ne $Actual)) {
        throw "SHA-256 verification failed! Expected $Expected, got $Actual"
    }

    $Dest = Join-Path $InstallDir "nala-colibri-rs.exe"
    Copy-Item -Force (Join-Path $Tmp $Asset) $Dest
    Copy-Item -Force (Join-Path $Tmp $Asset) (Join-Path $InstallDir "coli-rs.exe")
    Write-Host "[OK] Installed NALA-Colibri-RS to $Dest"
    & $Dest --version
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
