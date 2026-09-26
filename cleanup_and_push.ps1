# cleanup_and_push.ps1
# Script to clean up repo, remove logs/cache, untrack obsolete files, and push to refactor-clean branch

param (
    [string]$BranchName = "refactor-clean",
    [switch]$NoPush = $false
)

$ErrorActionPreference = "Continue"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       START CLEANUP REPO AND PUSH TO NEW BRANCH        " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Remove .log files inside Sign Language Translator
Write-Host "[1/5] Removing .log files..." -ForegroundColor Yellow
$logFiles = Get-ChildItem -Path "Sign Language Translator" -Filter "*.log" -File -Recurse -ErrorAction SilentlyContinue
foreach ($file in $logFiles) {
    Write-Host "  [-] Deleted log: $($file.Name)" -ForegroundColor Gray
    Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
}

# 2. Remove __pycache__ directories
Write-Host "[2/5] Removing __pycache__ directories..." -ForegroundColor Yellow
$pycacheDirs = Get-ChildItem -Path . -Filter "__pycache__" -Directory -Recurse -ErrorAction SilentlyContinue
foreach ($dir in $pycacheDirs) {
    Write-Host "  [-] Deleted cache: $($dir.FullName)" -ForegroundColor Gray
    Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

# 3. Clean tmp_clips folder
Write-Host "[3/5] Cleaning tmp_clips directory..." -ForegroundColor Yellow
$tmpClips = Join-Path (Get-Location) "Sign Language Translator\tmp_clips"
if (Test-Path $tmpClips) {
    Get-ChildItem -Path $tmpClips -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  [OK] Cleaned tmp_clips" -ForegroundColor Green
}

# 4. Untrack obsolete Data files from Git index
Write-Host "[4/5] Untracking obsolete files from Git index..." -ForegroundColor Yellow
git rm -r --cached "Sign Language Translator/Data" --ignore-unmatch 2>$null

# 5. Create or switch to target branch
Write-Host "[5/5] Switching to branch $BranchName..." -ForegroundColor Yellow
$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne $BranchName) {
    $existing = git branch --list $BranchName
    if ($existing) {
        git checkout $BranchName
    } else {
        git checkout -b $BranchName
    }
}

# 6. Add changes and commit
Write-Host "Staging changes and committing..." -ForegroundColor Yellow
git add .gitignore
git add -A
git commit -m "chore: cleanup logs, pycache, untrack obsolete data files and update .gitignore"

# 7. Push to remote
if (-not $NoPush) {
    Write-Host "Pushing branch $BranchName to origin..." -ForegroundColor Cyan
    git push -u origin $BranchName
    if ($LASTEXITCODE -eq 0) {
        Write-Host "========================================================" -ForegroundColor Green
        Write-Host "    SUCCESS: Pushed to branch $BranchName on GitHub!     " -ForegroundColor Green
        Write-Host "========================================================" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Git push failed. Please verify GitHub credentials." -ForegroundColor Red
    }
} else {
    Write-Host "Skipped git push because -NoPush was specified." -ForegroundColor Yellow
}
