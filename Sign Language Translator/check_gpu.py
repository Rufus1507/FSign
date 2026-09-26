# -*- coding: utf-8 -*-
"""
check_gpu.py — Kiểm tra phần cứng GPU, CUDA Toolkit, cuDNN và trạng thái TensorFlow
===================================================================================
Hỗ trợ chẩn đoán chính xác lý do TensorFlow nhận diện hoặc không nhận diện GPU.
"""
import os
import sys
import subprocess

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

print("=" * 75)
print("=== [1] KIỂM TRA PHẦN CỨNG NVIDIA GPU VÀ DRIVER ===")
print("=" * 75)

has_nvidia_gpu = False
try:
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if res.returncode == 0:
        print("=> [OK] Đã tìm thấy lệnh nvidia-smi:")
        print(res.stdout)
        has_nvidia_gpu = True
    else:
        print("=> nvidia-smi trả về mã lỗi:", res.stderr)
except FileNotFoundError:
    print("=> Lệnh 'nvidia-smi' không có sẵn trong PATH hệ thống.")
    # Thử tìm trực tiếp trong DriverStore
    nvsmi_default = r"C:\Windows\System32\DriverStore\FileRepository"
    found_smi = False
    if os.path.exists(nvsmi_default):
        for root, dirs, files in os.walk(nvsmi_default):
            if "nvidia-smi.exe" in files:
                smi_path = os.path.join(root, "nvidia-smi.exe")
                print(f"=> Tìm thấy nvidia-smi tại: {smi_path}")
                try:
                    r = subprocess.run([smi_path], capture_output=True, text=True, timeout=5)
                    print(r.stdout)
                    has_nvidia_gpu = True
                    found_smi = True
                    break
                except Exception:
                    pass
    if not found_smi:
        print("=> Không tìm thấy nvidia-smi.exe. Máy có thể không có GPU rời NVIDIA hoặc chưa cài Driver NVIDIA.")
except Exception as e:
    print(f"=> Lỗi kiểm tra nvidia-smi: {e}")

print("\n" + "=" * 75)
print("=== [2] KIỂM TRA CUDA TOOLKIT VÀ CÁC FILE DLL CẦN THIẾT CHO TF 2.10 ===")
print("=" * 75)

REQUIRED_DLLS = [
    ("cudart64_110.dll", "CUDA Runtime 11.x"),
    ("cublas64_11.dll", "cuBLAS 11.x"),
    ("cudnn64_8.dll", "cuDNN 8.x"),
    ("cusolver64_11.dll", "cuSOLVER 11.x"),
]

loaded_directories = []
found_dll_map = {name: None for name, _ in REQUIRED_DLLS}

# 1. Tìm trong thư mục cài chuẩn CUDA
cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if os.path.exists(cuda_base):
    installed_versions = os.listdir(cuda_base)
    print(f"=> Thư mục CUDA cài đặt: {cuda_base}")
    print(f"=> Các phiên bản phát hiện: {installed_versions}")
    for ver in installed_versions:
        for sub in ["bin", "libnvvp"]:
            bin_dir = os.path.join(cuda_base, ver, sub)
            if os.path.exists(bin_dir):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_dir)
                        loaded_directories.append(bin_dir)
                    except Exception:
                        pass
                for name, _ in REQUIRED_DLLS:
                    target_path = os.path.join(bin_dir, name)
                    if os.path.exists(target_path):
                        found_dll_map[name] = target_path
else:
    print(f"=> Không tìm thấy thư mục cài đặt CUDA mặc định tại: {cuda_base}")

# 2. Tìm trong các đường dẫn PATH
for p in os.environ.get("PATH", "").split(os.pathsep):
    if p and os.path.exists(p):
        if "cuda" in p.lower():
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(p)
                    loaded_directories.append(p)
                except Exception:
                    pass
        for name, _ in REQUIRED_DLLS:
            if not found_dll_map[name]:
                target = os.path.join(p, name)
                if os.path.exists(target):
                    found_dll_map[name] = target

print("\n--- BẢNG KIỂM TRA CÁC DLL QUAN TRỌNG CHO TENSORFLOW 2.10 (WINDOWS) ---")
all_dlls_found = True
for name, desc in REQUIRED_DLLS:
    loc = found_dll_map[name]
    if loc:
        print(f"  [OK]    {name:<18} ({desc}): Tìm thấy tại {loc}")
    else:
        print(f"  [THIẾU] {name:<18} ({desc}): CHƯA TÌM THẤY")
        all_dlls_found = False

print(f"\n=> Đã nạp {len(loaded_directories)} thư mục vào Windows DLL search path.")

print("\n" + "=" * 75)
print("=== [3] KIỂM TRA TENSORFLOW NHẬN DIỆN GPU THỰC TẾ ===")
print("=" * 75)

import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n=> [THÀNH CÔNG] TensorFlow đã nhận diện {len(gpus)} GPU:")
    for i, gpu in enumerate(gpus):
        print(f"   [{i}] Device: {gpu.name} (Type: {gpu.device_type})")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("       -> Bật dynamic memory growth: THÀNH CÔNG")
        except Exception as e:
            print(f"       -> Memory growth error: {e}")
    
    # Test tính toán thử trên GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("=> [TEST] Tính toán ma trận trên GPU: THÀNH CÔNG!")
        print("=> BẠN CÓ THỂ BẮT ĐẦU TRAIN TRÊN GPU NGAY BẰNG LỆNH:")
        print("   python train_model.py --device gpu --epochs 50 --batch_size 32")
    except Exception as e:
        print(f"=> [TEST] Lỗi khi thực thi phép toán trên GPU: {e}")

else:
    print("\n=> [CHƯA KÍCH HOẠT] TensorFlow hiện KHÔNG nhận diện được GPU.")
    print("\n--- HƯỚNG DẪN KÍCH HOẠT GPU CHO TENSORFLOW 2.10 TRÊN WINDOWS ---")
    if not has_nvidia_gpu:
        print("1. Máy tính của bạn không phát hiện GPU NVIDIA (hoặc Driver chưa cài đặt).")
        print("   Nếu máy chỉ có Intel/AMD CPU hoặc card tích hợp: việc huấn luyện trên CPU")
        print("   là hoàn toàn bình thường (chỉ mất ~4.5 phút cho 50 epochs nhờ dataset_cache.npz).")
    else:
        print("1. Máy CÓ GPU NVIDIA nhưng thiếu bộ thư viện CUDA Toolkit và cuDNN tương thích:")
        print("   - Cần cài đặt CUDA Toolkit 11.2.2 (hoặc 11.8):")
        print("     https://developer.nvidia.com/cuda-11.2.2-download-archive")
        print("   - Tải cuDNN v8.1.0 (cho CUDA 11.x):")
        print("     https://developer.nvidia.com/rdp/cudnn-archive")
        print("   - Giải nén cuDNN và copy các file trong bin/, include/, lib/ vào:")
        print(r"     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2")
        print("   - Sau khi copy xong, chạy lại 'python check_gpu.py' để xác nhận.")

print("=" * 75)
