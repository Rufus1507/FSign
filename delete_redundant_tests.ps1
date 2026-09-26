# delete_redundant_tests.ps1
# Script to delete one-off test and verification scripts, commit and push to refactor-clean

$ErrorActionPreference = "Continue"

$targetDir = "Sign Language Translator"
$filesToDelete = @(
    "test_b1_pipeline.py",
    "test_b2_idle.py",
    "test_b3_consensus.py",
    "test_b4_cooldown.py",
    "test_b5_dropout.py",
    "test_c3_threaded_camera.py",
    "run_all_b_tests.py",
    "run_collect_test.py",
    "verify_pipeline.py",
    "verify_run_and_backup.py",
    "verify_tflite_threads.py",
    "inspect_h5.py",
    "extract_train_log.py"
)

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "     DELETING REDUNDANT TEST & VERIFICATION SCRIPTS     " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

foreach ($file in $filesToDelete) {
    $fullPath = Join-Path $targetDir $file
    if (Test-Path $fullPath) {
        Remove-Item -Path $fullPath -Force
        Write-Host "  [-] Deleted: $file" -ForegroundColor Gray
    } else {
        Write-Host "  [?] Not found: $file" -ForegroundColor DarkGray
    }
}

# Also ensure any lingering logs or cache are removed
Get-ChildItem -Path $targetDir -Filter "*.log" -File -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path . -Filter "__pycache__" -Directory -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

Write-Host "`nStaging deletions and committing..." -ForegroundColor Yellow
git add -A
git commit -m "chore: remove redundant test and verification scripts"

Write-Host "Pushing changes to branch refactor-clean..." -ForegroundColor Cyan
git push origin refactor-clean

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "     DONE! REDUNDANT SCRIPTS REMOVED AND PUSHED         " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
