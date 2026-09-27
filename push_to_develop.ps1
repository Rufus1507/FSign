# push_to_develop.ps1
# Script to switch to/create 'develop' branch, stage all changes, commit, and push to origin.

$ErrorActionPreference = "Continue"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         CREATE & PUSH TO BRANCH 'develop'              " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Chuyen huong vao thu muc repository Git
$repoDir = "d:\Desktop\5\DPL302m\project\Sign-Language-Translator"
Set-Location -Path $repoDir
Write-Host "[1/6] Thu muc lam viec: $(Get-Location)" -ForegroundColor Yellow

# Giai phong index.lock neu bi ket do tien trinh cu
$lockFile = Join-Path $repoDir ".git\index.lock"
if (Test-Path $lockFile) {
    Write-Host "  [-] Phat hien .git/index.lock con sot lai, dang giai phong..." -ForegroundColor Yellow
    Remove-Item -Path $lockFile -Force -ErrorAction SilentlyContinue
}

# 2. Kiem tra va chuyen sang nhanh develop
Write-Host "`n[2/6] Tao hoac chuyen sang nhanh 'develop'..." -ForegroundColor Yellow
$currentBranch = (git branch --show-current).Trim()
Write-Host "  Nhanh hien tai: $currentBranch" -ForegroundColor Gray

if ($currentBranch -ne "develop") {
    $branchExists = git branch --list "develop"
    if ($branchExists) {
        Write-Host "  Nhanh 'develop' da ton tai. Chuyen sang 'develop'..." -ForegroundColor Gray
        git checkout develop
    } else {
        Write-Host "  Tao nhanh moi 'develop' tu $currentBranch..." -ForegroundColor Gray
        git checkout -b develop
    }
} else {
    Write-Host "  Da o san tren nhanh 'develop'." -ForegroundColor Green
}

# 3. Loai bo thu muc Data/ khoi Git index (chi quan ly code, khong commit 216k file .npy)
Write-Host "`n[3/6] Bo theo doi thu muc Data/ thue (untrack Data/)..." -ForegroundColor Yellow
git rm -r --cached "Sign Language Translator/Data" --ignore-unmatch 2>$null

# 4. Stage cac file code, docs, models, configs (da bao ve boi .gitignore)
Write-Host "`n[4/6] Dua cac thay doi vao Stage (git add)..." -ForegroundColor Yellow
git add .gitignore
git add -A
Write-Host "  Danh sach cac thay doi duoc stage:" -ForegroundColor Gray
git status -s

# 5. Commit changes
Write-Host "`n[5/6] Tao Commit..." -ForegroundColor Yellow
$status = (git status --porcelain)
if ($status) {
    $commitMsg = "feat: complete tasks A-E (optimization, consensus, idle tolerance, presence flag) and cleanup redundant codebase"
    git commit -m $commitMsg
    Write-Host "  [OK] Da commit thanh cong!" -ForegroundColor Green
} else {
    Write-Host "  [.] Khong co thay doi moi can commit." -ForegroundColor DarkGray
}

# 6. Push to remote origin develop
Write-Host "`n[6/6] Day len GitHub (git push -u origin develop)..." -ForegroundColor Cyan
git push -u origin develop

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================================" -ForegroundColor Green
    Write-Host "    THANH CONG: Da day len nhanh 'develop' tren GitHub! " -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
} else {
    Write-Host "`n[CANH BAO] Git push gap loi hoac can nhap credentials. Vui long kiem tra terminal." -ForegroundColor Red
}
