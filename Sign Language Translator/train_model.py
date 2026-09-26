# -*- coding: utf-8 -*-
"""
train_model.py — Pipeline huấn luyện và benchmark FSign trên Data_normalized/
=============================================================================
Tính năng:
  - Tự động backup checkpoint cũ trước khi train mới.
  - Sử dụng kiến trúc chuẩn nhất quán từ model_def.py.
  - Đồng bộ 100% metric theo dõi: EarlyStopping, ReduceLROnPlateau và ModelCheckpoint
    cùng dùng monitor='val_categorical_accuracy', mode='max'.
  - Nạp dữ liệu siêu tốc bằng ThreadPool đa luồng + tự động lưu cache dataset_cache.npz.
  - Tách Train / Validation / Test (Holdout 15%) phân tầng (stratified).
  - Tự động kiểm tra chéo độ khớp 100% giữa model trong RAM và file .h5 lưu trên đĩa.
  - Đánh giá chi tiết bằng classification_report (đầy đủ 61 nhãn).
  - Benchmark đối đầu với checkpoint cũ release/94,58.h5 trên cùng tập Test.
  - Thử nghiệm độ bền (robustness test) với các biến đổi tỷ lệ khoảng cách/góc.
"""

import os
import sys
import shutil
import argparse
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def configure_windows_cuda_dlls():
    """Tự động tìm kiếm và nạp các đường dẫn CUDA/cuDNN vào DLL search path của Windows (Python 3.8+)."""
    if sys.platform != 'win32' or not hasattr(os, 'add_dll_directory'):
        return []
    
    loaded = []
    # 1. Thư mục cài đặt chuẩn của CUDA Toolkit
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(cuda_base):
        try:
            for ver in os.listdir(cuda_base):
                for sub in ["bin", "libnvvp"]:
                    p = os.path.join(cuda_base, ver, sub)
                    if os.path.exists(p):
                        try:
                            os.add_dll_directory(p)
                            loaded.append(p)
                        except Exception:
                            pass
        except Exception:
            pass

    # 2. Thư mục trong PATH
    for ep in os.environ.get("PATH", "").split(os.pathsep):
        if "cuda" in ep.lower() and os.path.exists(ep):
            try:
                os.add_dll_directory(ep)
                loaded.append(ep)
            except Exception:
                pass

    return loaded

_loaded_dll_dirs = configure_windows_cuda_dlls()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model_def import build_model as build_unified_model


def setup_device(device_choice='auto'):
    """
    Cấu hình GPU / CPU cho TensorFlow:
      - 'auto': Ưu tiên GPU nếu có (bật dynamic memory growth); nếu không thì tự động dùng CPU.
      - 'gpu': Bắt buộc dùng GPU. Báo lỗi chi tiết nếu TensorFlow không tìm thấy GPU.
      - 'cpu': Bắt buộc dùng CPU (vô hiệu hóa GPU đối với TensorFlow).
    """
    if device_choice == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        print("=> [DEVICE] Đã cấu hình bắt buộc chạy trên CPU (--device cpu).\n", flush=True)
        return 'CPU'

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"=> [DEVICE] ĐÃ KÍCH HOẠT GPU THÀNH CÔNG! Tìm thấy {len(gpus)} GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   [{i}] {gpu.name} ({gpu.device_type}) - Đã bật Dynamic Memory Growth")
            print("=> Quá trình huấn luyện sẽ chạy trực tiếp trên GPU!\n", flush=True)
            return 'GPU'
        except Exception as e:
            print(f"=> [DEVICE] Cảnh báo khi cấu hình GPU: {e}")
            if device_choice == 'gpu':
                raise
            return 'CPU'
    else:
        if device_choice == 'gpu':
            raise RuntimeError(
                "LỖI: Bạn chọn '--device gpu' nhưng TensorFlow không nhận diện được GPU nào!\n"
                "Nguyên nhân thường gặp trên Windows:\n"
                "1. Máy không có GPU rời NVIDIA hoặc Driver NVIDIA chưa cài đặt.\n"
                "2. Thiếu CUDA Toolkit 11.2 hoặc cuDNN 8.1.0 (TensorFlow 2.10 yêu cầu cudart64_110.dll).\n"
                "-> Hãy chạy 'python check_gpu.py' trong terminal để kiểm tra chi tiết trạng thái phần cứng và DLL."
            )
        print("=> [DEVICE] Không tìm thấy GPU hợp lệ hoặc thiếu CUDA DLLs. Đang chạy trên CPU.\n", flush=True)
        return 'CPU'


def _load_single_sequence(args_tuple):
    """Hàm helper đọc 1 sequence 60 frame (dùng cho ThreadPool)."""
    seq_dir, sequence_length = args_tuple
    window = []
    for frame_idx in range(sequence_length):
        fpath = os.path.join(seq_dir, f"{frame_idx}.npy")
        try:
            kp = np.load(fpath)
            window.append(kp)
        except Exception:
            return None
    if len(window) == sequence_length:
        return np.array(window, dtype=np.float32)
    return None


def fast_load_dataset_normalized(data_path='Data_normalized', sequence_length=60, cache_name='dataset_cache.npz'):
    """
    Nạp dữ liệu Data_normalized:
    1. Kiểm tra cache .npz: nếu có -> nạp tức thì trong 0.5s!
    2. Nếu chưa có -> đọc song song đa luồng (ThreadPool 16 workers) kèm log tiến độ, sau đó tự động lưu cache.
    """
    cache_path = os.path.join(data_path, cache_name)
    if os.path.exists(cache_path):
        print(f"=> [CACHE] Tìm thấy file cache: {cache_path}")
        print("=> Đang giải nén nhanh vào RAM...")
        t0 = time.time()
        c = np.load(cache_path, allow_pickle=True)
        X = c['X']
        y = c['y']
        label_map = c['label_map'].item()
        print(f"=> Nạp xong từ Cache trong {time.time()-t0:.2f}s! X.shape={X.shape}, y.shape={y.shape}\n", flush=True)
        return X, y, label_map

    print(f"=> Không tìm thấy cache. Bắt đầu đọc đa luồng (16 threads) từ: {data_path}")
    actions = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_map = {label: idx for idx, label in enumerate(actions)}

    all_sequences = []
    all_labels = []

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for idx, label in enumerate(actions, 1):
            t_label = time.time()
            label_dir = os.path.join(data_path, label)
            seq_dirs = sorted([os.path.join(label_dir, d) for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))])
            
            tasks = [(s_dir, sequence_length) for s_dir in seq_dirs]
            results = list(pool.map(_load_single_sequence, tasks))
            
            valid_seqs = [res for res in results if res is not None]
            all_sequences.extend(valid_seqs)
            all_labels.extend([label_map[label]] * len(valid_seqs))

            print(f"  [{idx:2d}/{len(actions)}] {label:<40}: {len(valid_seqs):>3} seqs ({time.time()-t_label:.2f}s)", flush=True)

    X = np.array(all_sequences, dtype=np.float32)
    y = to_categorical(all_labels, num_classes=len(actions)).astype(np.float32)
    print(f"=> Đọc xong toàn bộ trong {time.time()-t_start:.1f}s! X.shape={X.shape}, y.shape={y.shape}", flush=True)

    print(f"=> Đang lưu cache vào: {cache_path}...")
    np.savez_compressed(cache_path, X=X, y=y, label_map=label_map)
    print("=> Đã lưu cache thành công! Những lần chạy sau sẽ không cần đọc lại 216k file con.\n", flush=True)
    return X, y, label_map


def build_legacy_model(input_shape=(60, 126), num_classes=60) -> Sequential:
    """Kiến trúc mô hình cũ release/94,58.h5 (60 nhãn)."""
    model = Sequential([
        LSTM(32, return_sequences=True, activation='relu', input_shape=input_shape, name='lstm'),
        LSTM(128, return_sequences=True, activation='relu', name='lstm_1'),
        LSTM(64, return_sequences=False, activation='relu', name='lstm_2'),
        Dense(64, activation='relu', name='dense'),
        Dense(32, activation='relu', name='dense_1'),
        Dense(num_classes, activation='softmax', name='dense_2')
    ], name='FSign_Legacy_9458')
    return model


def run_robustness_test(model, X_test, y_test, label_names, scale_factors=[0.7, 0.85, 1.0, 1.15, 1.3]):
    """
    Thử nghiệm độ bền: Co/giãn khoảng cách tọa độ (mô phỏng người đứng xa/gần camera).
    """
    print("\n" + "="*70)
    print("=== [A3.2] THỬ NGHIỆM ĐỘ BỀN (ROBUSTNESS TEST KHI THAY ĐỔI SCALE) ===")
    print("="*70)
    y_true_indices = np.argmax(y_test, axis=1)

    print(f"{'Scale Factor':<15} {'Mô phỏng':<30} {'Accuracy':<12}")
    print("-" * 60)
    for scale in scale_factors:
        X_scaled = X_test * scale
        preds = model.predict(X_scaled, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        acc = accuracy_score(y_true_indices, pred_indices)
        desc = "Chuẩn (1.0x)" if scale == 1.0 else f"Thu nhỏ ({scale}x)" if scale < 1.0 else f"Phóng to ({scale}x)"
        print(f"{scale:<15.2f} {desc:<30} {acc*100:<10.2f}%")
    print()


def evaluate_legacy_comparison(new_model, X_test, y_test, label_map, legacy_weights_path='release/94,58.h5'):
    """
    So sánh đối đầu giữa Model Mới (61 nhãn) và Checkpoint Cũ 94,58.h5 (60 nhãn)
    trên cùng tập test holdout.
    """
    print("\n" + "="*70)
    print("=== [A3.1] BENCHMARK ĐỐI ĐẦU: MODEL MỚI VS CHECKPOINT CŨ (94,58.h5) ===")
    print("="*70)

    y_test_indices = np.argmax(y_test, axis=1)
    inv_label_map = {v: k for k, v in label_map.items()}

    y_pred_new = new_model.predict(X_test, verbose=0)
    y_pred_new_idx = np.argmax(y_pred_new, axis=1)
    acc_new = accuracy_score(y_test_indices, y_pred_new_idx)
    print(f"\n=> [1] Model Mới (train trên Data_normalized/ - 61 nhãn):")
    print(f"   Accuracy trên toàn bộ tập test: {acc_new * 100:.2f}% ({np.sum(y_pred_new_idx == y_test_indices)}/{len(y_test_indices)} samples)")

    if not os.path.exists(legacy_weights_path):
        legacy_weights_path = os.path.join('Structure', 'Structure 5 (final)', '94,58.h5')

    if os.path.exists(legacy_weights_path):
        print(f"\n=> [2] Checkpoint Cũ (94,58.h5 - 60 nhãn từ Data/ cũ):")
        print(f"   Đang tải weights từ: {legacy_weights_path}")
        legacy_model = build_legacy_model(num_classes=60)
        try:
            legacy_model.load_weights(legacy_weights_path)
            print("   Tải weights thành công!")

            valid_mask = [inv_label_map[idx] != 'xin loi' for idx in y_test_indices]
            X_test_shared = X_test[valid_mask]
            y_test_shared_indices = y_test_indices[valid_mask]

            from ActionDetection import DEFAULT_ACTIONS
            legacy_action_to_idx = {act: i for i, act in enumerate(DEFAULT_ACTIONS)}
            y_test_legacy_mapped = np.array([legacy_action_to_idx[inv_label_map[idx]] for idx in y_test_shared_indices])

            y_pred_legacy = legacy_model.predict(X_test_shared, verbose=0)
            y_pred_legacy_idx = np.argmax(y_pred_legacy, axis=1)
            acc_legacy = accuracy_score(y_test_legacy_mapped, y_pred_legacy_idx)

            print(f"   Accuracy của model cũ trên tập test normalized (60 nhãn chung): {acc_legacy * 100:.2f}%")
            print(f"   -> Chênh lệch hiệu năng: {acc_new * 100 - acc_legacy * 100:+.2f}%")

            y_pred_new_shared = y_pred_new_idx[valid_mask]
            acc_new_on_shared = accuracy_score(y_test_shared_indices, y_pred_new_shared)
            print(f"   Accuracy của model mới trên 60 nhãn chung: {acc_new_on_shared * 100:.2f}%")

        except Exception as e:
            print(f"   [Cảnh báo] Lỗi khi benchmark model cũ: {e}")
    else:
        print(f"   [Bỏ qua] Không tìm thấy file checkpoint cũ tại: {legacy_weights_path}")


def main():
    p = argparse.ArgumentParser(description='Train và Benchmark FSign trên Data_normalized/')
    p.add_argument('--data_path', default='Data_normalized', help='Đường dẫn Data_normalized')
    p.add_argument('--epochs', type=int, default=50, help='Số epoch huấn luyện (mặc định: 50)')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size (mặc định: 32)')
    p.add_argument('--lr', type=float, default=1e-3, help='Tốc độ học Learning rate (mặc định: 0.001)')
    p.add_argument('--device', default='auto', choices=['auto', 'gpu', 'cpu'],
                   help='Thiết bị huấn luyện: auto, gpu, cpu (mặc định: auto)')
    p.add_argument('--output_model', default=os.path.join('Models', 'model_normalized_v1.h5'),
                   help='Đường dẫn lưu model mới (mặc định: Models/model_normalized_v1.h5)')
    p.add_argument('--eval_only', action='store_true', help='Chỉ đánh giá model đã lưu, không train lại')
    args = p.parse_args()

    print("=" * 70)
    print("=== HUẤN LUYỆN & BENCHMARK MÔ HÌNH FSIGN TRÊN DATA_NORMALIZED ===")
    print("=" * 70)

    # 0. Cấu hình thiết bị (GPU / CPU)
    active_device = setup_device(args.device)

    # 1. Nạp dữ liệu siêu tốc bằng ThreadPool + Cache
    print(f"\n[1/5] Đang nạp dataset từ: {args.data_path}")
    X, y, label_map = fast_load_dataset_normalized(data_path=args.data_path, sequence_length=60)
    num_classes = len(label_map)
    label_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    print(f"=> Tổng cộng: {X.shape[0]} sequences ({num_classes} nhãn)")

    # 2. Phân chia Train / Val / Test (Holdout 15%) phân tầng
    print("\n[2/5] Phân chia Train / Validation / Test (Holdout 15%)...")
    y_int = np.argmax(y, axis=1)
    
    unique, counts = np.unique(y_int, return_counts=True)
    rare_classes = set(unique[counts < 2])
    if rare_classes:
        print(f"   (Nhãn hiếm < 2 mẫu: {[label_names[c] for c in rare_classes]} -> đưa vào Train set)")
        regular_mask = ~np.isin(y_int, list(rare_classes))
        X_reg, y_reg = X[regular_mask], y[regular_mask]
        X_rare, y_rare = X[~regular_mask], y[~regular_mask]
        
        X_train_reg, X_temp, y_train_reg, y_temp = train_test_split(
            X_reg, y_reg, test_size=0.30, random_state=42, stratify=np.argmax(y_reg, axis=1)
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        X_train = np.concatenate([X_train_reg, X_rare], axis=0)
        y_train = np.concatenate([y_train_reg, y_rare], axis=0)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y_int
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1)
        )

    print(f"   Train set:      {X_train.shape[0]} mẫu ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"   Validation set: {X_val.shape[0]} mẫu ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"   Test (Holdout): {X_test.shape[0]} mẫu ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

    # 3. Khởi tạo mô hình kiến trúc chuẩn từ model_def.py
    print("\n[3/5] Khởi tạo mô hình kiến trúc chuẩn nhất quán (Unified LSTM)...")
    model = build_unified_model(input_shape=(60, 126), num_classes=num_classes, lr=args.lr, clipnorm=1.0)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    # 4. Huấn luyện (hoặc nạp model đã lưu nếu --eval_only)
    if not args.eval_only:
        # Tự động backup checkpoint cũ trước khi ghi đè
        if os.path.exists(args.output_model):
            backup_target = os.path.splitext(args.output_model)[0] + '_backup_run1.h5'
            if not os.path.exists(backup_target):
                shutil.copy2(args.output_model, backup_target)
                print(f"=> [BACKUP] Đã sao lưu checkpoint cũ sang: {backup_target}")

        model.summary()
        print(f"\n[4/5] Bắt đầu huấn luyện ({args.epochs} epochs, batch_size={args.batch_size})...")
        log_dir = os.path.join('Logs', 'train_normalized_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

        # ĐỒNG BỘ 100% METRIC: EarlyStopping và ModelCheckpoint cùng theo dõi val_categorical_accuracy
        callbacks = [
            EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
            ModelCheckpoint(args.output_model, monitor='val_categorical_accuracy', mode='max', save_best_only=True, verbose=1),
            TensorBoard(log_dir=log_dir)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print(f"\n=> Đã lưu mô hình tốt nhất tại: {args.output_model}")
        print("=> Đang đồng bộ mô hình trong RAM với checkpoint tốt nhất đã lưu trên đĩa...")
        model.load_weights(args.output_model)
        print("=> Đã nạp thành công weights tốt nhất vào RAM!\n")
    else:
        print(f"\n[4/5] Chế độ Eval-Only: Đang nạp model đã train từ {args.output_model}...")
        model.load_weights(args.output_model)
        print("=> Tải trọng số thành công!")

    # 5. Đánh giá toàn diện trên Holdout Test Set
    print("\n" + "="*70)
    print("=== [5/5] KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP HOLDOUT TEST SET ===")
    print("="*70)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n>> Test Loss:     {test_loss:.4f}")
    print(f">> Test Accuracy: {test_acc * 100:.2f}%\n")

    # Kiểm tra chéo đồng nhất RAM vs Đĩa khi vừa train xong
    if not args.eval_only:
        print("--- [KIỂM TRA CHÉO ĐỒNG NHẤT RAM VS ĐĨA] ---")
        disk_model = build_unified_model(input_shape=(60, 126), num_classes=num_classes, lr=args.lr)
        disk_model.load_weights(args.output_model)
        _, disk_acc = disk_model.evaluate(X_test, y_test, verbose=0)
        print(f"  Model trong RAM:     {test_acc * 100:.4f}%")
        print(f"  Model load lại từ đĩa: {disk_acc * 100:.4f}%")
        if np.isclose(test_acc, disk_acc, atol=1e-5):
            print("  => XÁC NHẬN: Model trong RAM và File trên đĩa KHỚP NHAU 100% HOÀN HẢO!\n")
        else:
            print("  => [CẢNH BÁO] Có sự sai lệch giữa RAM và đĩa!\n")

    # Dự đoán trên tập test
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("--- CLASSIFICATION REPORT CHI TIẾT THEO TỪNG NHÃN ---")
    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=label_names,
        digits=4,
        zero_division=0
    )
    print(report)

    # 6. Benchmark so sánh đối đầu với model cũ & kiểm tra độ bền
    evaluate_legacy_comparison(model, X_test, y_test, label_map)
    run_robustness_test(model, X_test, y_test, label_names)

    print("\nHoàn tất toàn bộ pipeline!")


if __name__ == '__main__':
    main()
