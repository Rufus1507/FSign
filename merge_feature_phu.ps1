# merge_feature_phu.ps1
# Script to fetch and merge branch origin/feature/Phu into current branch

$ErrorActionPreference = "Continue"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       FETCH VA MERGE VOI NHANH origin/feature/Phu       " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Fetch from remote
Write-Host "[1/3] Dang fetch cac nhanh tu GitHub..." -ForegroundColor Yellow
git fetch origin

# 2. Check if branch exists
Write-Host "`n[2/3] Kiem tra nhanh origin/feature/Phu..." -ForegroundColor Yellow
$remoteBranch = git branch -r --list "origin/feature/Phu"
$allPhu = git branch -r --list "*Phu*"

if (-not $remoteBranch) {
    Write-Host "[CANH BAO] Khong tim thay chinh xac 'origin/feature/Phu'." -ForegroundColor Yellow
    if ($allPhu) {
        Write-Host "Cac nhanh lien quan tim thay tren remote:" -ForegroundColor Green
        $allPhu
    } else {
        Write-Host "Danh sach toan bo nhanh remote hien co:" -ForegroundColor Gray
        git branch -r
    }
    exit 1
}

# 3. Merge into current branch
$current = (git branch --show-current).Trim()
Write-Host "`n[3/3] Tien hanh merge origin/feature/Phu vao nhanh hien tai ($current)..." -ForegroundColor Yellow
git merge origin/feature/Phu

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================================" -ForegroundColor Green
    Write-Host "    MERGE THANH CONG! DANG PUSH LEN GITHUB...           " -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    git push origin $current
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Da push thanh cong len origin/$current!" -ForegroundColor Green
    }
} else {
    Write-Host "`n[CANH BAO] Co xung dot (CONFLICT) giua 2 nhanh!" -ForegroundColor Red
    Write-Host "Vui long kiem tra cac file bi conflict duoi day:" -ForegroundColor Yellow
    git status -s
}
