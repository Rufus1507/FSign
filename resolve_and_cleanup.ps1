# resolve_and_cleanup.ps1
# Script to resolve merge conflicts with feature/Phu and clean up directory tree

$ErrorActionPreference = "Continue"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       RESOLVE MERGE CONFLICTS VA DON DEP CAY THU MUC    " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Luu code cua Phu thanh RunModel_phu_159.py
Write-Host "`n[1/6] Luu ban RunModel cua Phu thanh RunModel_phu_159.py de tham khao..." -ForegroundColor Yellow
$phuCode = git show origin/feature/Phu:"Sign Language Translator/RunModel.py" 2>$null
if ($phuCode) {
    [System.IO.File]::WriteAllText("Sign Language Translator\RunModel_phu_159.py", ($phuCode -join "`r`n"), [System.Text.Encoding]::UTF8)
    Write-Host "  [OK] Da tao: Sign Language Translator\RunModel_phu_159.py" -ForegroundColor Green
}

# 2. Khoi phuc RunModel.py cua HEAD (TFLite + Threaded Camera C3)
Write-Host "`n[2/6] Giu nguyen RunModel.py goc cua HEAD (TFLite + Threaded Camera)..." -ForegroundColor Yellow
git checkout --ours "Sign Language Translator/RunModel.py"
Write-Host "  [OK] Da checkout --ours cho RunModel.py" -ForegroundColor Green

# 3. Resolve cac file cua Phu trong Structure 5 (final)
Write-Host "`n[3/6] Resolve cac file fsign_159classes.h5 tu origin/feature/Phu..." -ForegroundColor Yellow
git add "Sign Language Translator/legacy/Structure/Structure 5 (final)" 2>$null

# Copy model 159 classes va label_map.json vao Models/ de tap trung quan ly
$h5Source = "Sign Language Translator\legacy\Structure\Structure 5 (final)\fsign_159classes.h5"
if (Test-Path $h5Source) {
    Copy-Item $h5Source "Sign Language Translator\Models\fsign_159classes.h5" -Force
    Write-Host "  [+] Da copy fsign_159classes.h5 vao Sign Language Translator\Models\" -ForegroundColor Green
}
$labelSource = "Sign Language Translator\label_map.json"
if (Test-Path $labelSource) {
    Copy-Item $labelSource "Sign Language Translator\Models\label_map_159.json" -Force
    Write-Host "  [+] Da copy label_map.json vao Sign Language Translator\Models\label_map_159.json" -ForegroundColor Green
}

# 4. Don dep cac file log nang va file test thua
Write-Host "`n[4/6] Xoa cac file log TensorBoard nang va script test thua..." -ForegroundColor Yellow
$logsFsign = "Sign Language Translator\Logs\fsign159"
if (Test-Path $logsFsign) {
    Remove-Item -Path $logsFsign -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  [-] Da xoa thu muc: $logsFsign" -ForegroundColor Gray
}
$logsFsignOpt = "Sign Language Translator\Logs\fsign159_opt"
if (Test-Path $logsFsignOpt) {
    Remove-Item -Path $logsFsignOpt -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  [-] Da xoa thu muc: $logsFsignOpt" -ForegroundColor Gray
}
$testInference = "Sign Language Translator\test_model_inference.py"
if (Test-Path $testInference) {
    Remove-Item -Path $testInference -Force -ErrorAction SilentlyContinue
    Write-Host "  [-] Da xoa file test thua: $testInference" -ForegroundColor Gray
}

# 5. Gom cac file bao cao markdown o root vao thu muc docs/
Write-Host "`n[5/6] Gom cac file bao cao o thu muc goc vao thu muc docs/..." -ForegroundColor Yellow
if (-not (Test-Path "docs")) {
    New-Item -ItemType Directory -Path "docs" -Force | Out-Null
}
$docFiles = @(
    "camera_ui_and_model_evaluation.md",
    "dataset_compatibility_report.md",
    "model_evaluation_report.md",
    "training_report_159classes.md"
)
foreach ($doc in $docFiles) {
    if (Test-Path $doc) {
        Move-Item -Path $doc -Destination "docs\" -Force
        Write-Host "  [->] Da di chuyen $doc vao docs\" -ForegroundColor Gray
    }
}

# Di chuyen ca thu muc dataset (neu chi co label_mapping.pkl) vao docs/
if (Test-Path "dataset\label_mapping.pkl") {
    if (-not (Test-Path "docs\dataset")) {
        New-Item -ItemType Directory -Path "docs\dataset" -Force | Out-Null
    }
    Move-Item -Path "dataset\label_mapping.pkl" -Destination "docs\dataset\" -Force
    Remove-Item -Path "dataset" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  [->] Da di chuyen dataset\label_mapping.pkl vao docs\dataset\" -ForegroundColor Gray
}

# 6. Commit va push
Write-Host "`n[6/6] Staging toan bo thay doi, commit va push len GitHub..." -ForegroundColor Yellow
git add -A
git commit -m "merge: resolve conflicts with feature/Phu, keep HEAD RunModel, archive Phu RunModel, cleanup logs and organize docs"

$currentBranch = (git branch --show-current).Trim()
Write-Host "Dang push len origin/$currentBranch..." -ForegroundColor Cyan
git push origin $currentBranch

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================================" -ForegroundColor Green
    Write-Host "    HOAN TAT! DA RESOLVE CONFLICT, DON DEP VA PUSH XONG!" -ForegroundColor Green
    Write-Host "========================================================`n" -ForegroundColor Green
} else {
    Write-Host "`n[LUU Y] Push gap loi hoac can kiem tra lai status." -ForegroundColor Red
}
