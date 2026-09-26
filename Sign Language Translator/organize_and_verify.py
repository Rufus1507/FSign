import os
import shutil
import zipfile
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print("=" * 80)
print("=== FSIGN: KỊCH BẢN DỌN DẸP & DI CHUYỂN AN TOÀN (TỰ ĐỘNG & CÓ XÁC MINH) ===")
print("=" * 80)

# 1. DANH SÁCH 4 FILE TRUNG GIAN TRONG archive/ CẦN XÓA
archive_dir = os.path.join(SCRIPT_DIR, "archive")
intermediate_files = [
    "RunModel_pre_b3.py",
    "RunModel_pre_b4.py",
    "RunModel_pre_b5.py",
    "RunModel_pre_c2_merge.py"
]

keep_files = [
    "ActionDetection_pre_ga_verified.py",
    "RunModel_pre_gb_verified.py",
    "RunModel_pre_gc_verified.py",
    "RunModel_pre_c3.py"
]

print("\n[BƯỚC 1] KIỂM TRA FILE TRONG archive/:")
for f in intermediate_files:
    p = os.path.join(archive_dir, f)
    if os.path.exists(p):
        print(f"  [-] File trung gian sẽ xóa: {f:<30} ({os.path.getsize(p):>6} bytes)")
    else:
        print(f"  [!] File không tồn tại:     {f}")

for f in keep_files:
    p = os.path.join(archive_dir, f)
    if os.path.exists(p):
        print(f"  [+] File mốc sẽ GIỮ LẠI:    {f:<30} ({os.path.getsize(p):>6} bytes)")
    else:
        print(f"  [!] File mốc thiếu:        {f}")

# 2. BACKUP TOÀN BỘ archive/ (8 FILES) RA NGOÀI REPO TRƯỚC KHI XÓA
# Đường dẫn ngoài repo: d:\Desktop\5\DPL302m\archive_backup_full_8files.zip
external_backup_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
backup_zip_path = os.path.join(external_backup_dir, "archive_backup_full_8files.zip")

print(f"\n[BƯỚC 2] TIẾN HÀNH SAO LƯU TOÀN BỘ archive/ SANG: {backup_zip_path}")
with zipfile.ZipFile(backup_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(archive_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, archive_dir)
            zipf.write(file_path, arcname)
            print(f"  * Đã nén vào zip: {arcname} ({os.path.getsize(file_path)} bytes)")

if os.path.exists(backup_zip_path):
    print(f"=> XÁC NHẬN: Tạo file backup zip thành công ({os.path.getsize(backup_zip_path)} bytes).")
else:
    raise RuntimeError(f"Lỗi: Không tạo được file backup {backup_zip_path}!")

# 3. TẠO THƯ MỤC legacy/ VÀ DI CHUYỂN NHÓM (c) BẰNG shutil.move
legacy_dir = os.path.join(SCRIPT_DIR, "legacy")
legacy_models_dir = os.path.join(legacy_dir, "Models_old")
os.makedirs(legacy_dir, exist_ok=True)
os.makedirs(legacy_models_dir, exist_ok=True)

print(f"\n[BƯỚC 3] DI CHUYỂN CÁC THƯ MỤC VÀ FILE DI SẢN LOOK & TELL VÀO: {legacy_dir}")

# Danh sách thư mục ở gốc "Sign Language Translator" cần di chuyển vào legacy/
folders_to_move = [
    "Structure",
    "backup",
    "Module",
    "Demo",
    "Videos",
    "server",
    "release",
    "Data"
]

for folder in folders_to_move:
    src = os.path.join(SCRIPT_DIR, folder)
    dst = os.path.join(legacy_dir, folder)
    if os.path.exists(src):
        if os.path.exists(dst):
            print(f"  [!] Đích đã tồn tại, bỏ qua: {dst}")
        else:
            shutil.move(src, dst)
            print(f"  [MOVE THƯ MỤC] {folder} -> legacy/{folder}")
    else:
        print(f"  [-] Không tìm thấy thư mục: {folder} (có thể đã di chuyển trước đó)")

# Danh sách script lẻ ở gốc cần di chuyển vào legacy/
scripts_to_move = [
    "ActionDetection.py",
    "CheckData.py",
    "CollectData.py"
]

for script in scripts_to_move:
    src = os.path.join(SCRIPT_DIR, script)
    dst = os.path.join(legacy_dir, script)
    if os.path.exists(src):
        if os.path.exists(dst):
            print(f"  [!] Đích đã tồn tại, bỏ qua: {dst}")
        else:
            shutil.move(src, dst)
            print(f"  [MOVE FILE] {script} -> legacy/{script}")

# Di chuyển các model cũ trong Models/ vào legacy/Models_old/
models_dir = os.path.join(SCRIPT_DIR, "Models")
models_subfolders_to_move = [
    "model cũ",
    "model (1)",
    "model (2)",
    "model (3)",
    "model (4)",
    "model (5)",
    "model 50 câu 95,33",
    "model 50 câu gốc",
    "model 50 câu sc 89.17"
]

for m_folder in models_subfolders_to_move:
    src = os.path.join(models_dir, m_folder)
    dst = os.path.join(legacy_models_dir, m_folder)
    if os.path.exists(src):
        if os.path.exists(dst):
            print(f"  [!] Đích đã tồn tại, bỏ qua: {dst}")
        else:
            shutil.move(src, dst)
            print(f"  [MOVE MODEL CŨ] Models/{m_folder} -> legacy/Models_old/{m_folder}")

# 4. XÓA 4 FILE TRUNG GIAN TRONG archive/
print(f"\n[BƯỚC 4] XÓA 4 FILE TRUNG GIAN TRONG archive/:")
for f in intermediate_files:
    p = os.path.join(archive_dir, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"  [ĐÃ XÓA] {f}")
    else:
        print(f"  [-] File đã không còn: {f}")

# Xóa __pycache__ trong archive nếu có
archive_pycache = os.path.join(archive_dir, "__pycache__")
if os.path.exists(archive_pycache):
    shutil.rmtree(archive_pycache)

# Kiểm tra lại archive/
print("\n=> DANH SÁCH FILE CÒN LẠI TRONG archive/:")
remaining = sorted(os.listdir(archive_dir))
for f in remaining:
    p = os.path.join(archive_dir, f)
    print(f"  * {f:<35} ({os.path.getsize(p)} bytes)")

assert len(remaining) == 4, f"Lỗi: archive/ phải có đúng 4 file mốc, hiện có {len(remaining)}!"

# 5. CHẠY LẠI CÁC BÀI TEST TỰ ĐỘNG ĐỂ XÁC MINH TOÀN VẸN
print("\n" + "=" * 80)
print("[BƯỚC 5] CHẠY LẠI TOÀN BỘ TEST SUITES ĐỂ XÁC MINH KHÔNG BỊ GÃY ĐƯỜNG DẪN")
print("=" * 80)

print("\n--- 5.1. CHẠY run_all_b_tests.py ---")
res_b = subprocess.run([sys.executable, "run_all_b_tests.py"], capture_output=True, text=True, encoding="utf-8")
print(res_b.stdout)
if res_b.stderr:
    print("STDERR (nếu có):", res_b.stderr)
assert res_b.returncode == 0, f"run_all_b_tests.py THẤT BẠI với mã lỗi {res_b.returncode}!"

print("\n--- 5.2. CHẠY test_c3_threaded_camera.py ---")
res_c3 = subprocess.run([sys.executable, "test_c3_threaded_camera.py"], capture_output=True, text=True, encoding="utf-8")
print(res_c3.stdout)
if res_c3.stderr:
    print("STDERR (nếu có):", res_c3.stderr)
assert res_c3.returncode == 0, f"test_c3_threaded_camera.py THẤT BẠI với mã lỗi {res_c3.returncode}!"

# 6. KIỂM TRA GIT STATUS
print("\n" + "=" * 80)
print("[BƯỚC 6] KIỂM TRA GIT STATUS SAU DI CHUYỂN")
print("=" * 80)
try:
    git_res = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, encoding="utf-8")
    print(git_res.stdout if git_res.stdout else "Git clean / không có thay đổi chưa tracked.")
except Exception as e:
    print("Không thể gọi git status:", e)

print("\n" + "=" * 80)
print("=== HOÀN TẤT DỌN DẸP & DI CHUYỂN AN TOÀN 100% ===")
print("=" * 80)
