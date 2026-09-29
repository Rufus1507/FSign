# -*- coding: utf-8 -*-
"""
train_baseline_119.py — Pipeline Huấn luyện & Đánh giá Baseline 119 Nhãn (Task MERGE-5)
========================================================================================
Kiến trúc: Dùng ĐÚNG model_def_merged119.py (128 -> 64 -> 32, Tanh, clipnorm=1.0)
Tập dữ liệu: Data_normalized_merged119/ (119 nhãn, 6,840 sequences, 129 chiều)
Phân chia: Stratified Split 80/20 (Train: 5,472 mẫu, Test/Val: 1,368 mẫu)
Mục tiêu: Đánh giá hội tụ, kiểm soát bùng nổ gradient, đo Top-1, Top-3, Top-5, Confusion Matrix.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Fix Unicode UTF-8 console output trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# CUDA DLL loading cho Windows
def configure_windows_cuda():
    if sys.platform != 'win32' or not hasattr(os, 'add_dll_directory'):
        return
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(cuda_base):
        try:
            for ver in os.listdir(cuda_base):
                for sub in ["bin", "libnvvp"]:
                    p = os.path.join(cuda_base, ver, sub)
                    if os.path.exists(p):
                        try:
                            os.add_dll_directory(p)
                        except Exception:
                            pass
        except Exception:
            pass

configure_windows_cuda()

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    Callback
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_def_merged119 import build_model

SEQUENCE_LENGTH = 60
FEATURE_DIM = 129


class GradientAndMetricsTracker(Callback):
    """
    Theo dõi sát sao từng epoch:
    - Kiểm tra lỗi NaN / Inf ở loss và weights (đề phòng gradient explosion).
    - Đo đạc thời gian từng epoch.
    - Lưu lại toàn bộ lịch sử train/val để lập báo cáo.
    """
    def __init__(self):
        super().__init__()
        self.history_records = []
        self.epoch_start_time = 0.0
        self.nan_detected = False

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dur = time.time() - self.epoch_start_time
        logs = logs or {}

        loss = float(logs.get('loss', 0.0))
        acc = float(logs.get('categorical_accuracy', 0.0))
        val_loss = float(logs.get('val_loss', 0.0))
        val_acc = float(logs.get('val_categorical_accuracy', 0.0))

        # Tìm metric top_3 và top_5 trong logs linh hoạt
        top3 = 0.0
        top5 = 0.0
        for k, v in logs.items():
            if 'val' in k and '3' in k:
                top3 = float(v)
            elif 'val' in k and '5' in k:
                top5 = float(v)

        lr_val = 0.001
        try:
            opt = self.model.optimizer
            if hasattr(opt, 'learning_rate'):
                lr_val = float(tf.keras.backend.get_value(opt.learning_rate))
        except Exception:
            pass

        # Kiểm tra NaN
        if np.isnan(loss) or np.isnan(val_loss) or np.isinf(loss) or np.isinf(val_loss):
            self.nan_detected = True
            print(f"\n[CẢNH BÁO BÙNG NỔ GRADIENT] Phát hiện NaN/Inf tại Epoch {epoch + 1}!")

        rec = {
            'epoch': epoch + 1,
            'loss': round(loss, 4),
            'acc': round(acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_top3': round(top3, 4),
            'val_top5': round(top5, 4),
            'lr': lr_val,
            'duration_s': round(dur, 2)
        }
        self.history_records.append(rec)

        print(
            f"  [Epoch {epoch+1:2d}] "
            f"Loss: {loss:.4f} | Acc: {acc*100:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | "
            f"Top-3: {top3*100:6.2f}% | Top-5: {top5*100:6.2f}% | "
            f"LR: {lr_val:.6f} ({dur:.1f}s)",
            flush=True
        )


def _load_single_sequence(seq_path: Path):
    """Nạp 1 chuỗi 60 frames."""
    frames = []
    for f_idx in range(SEQUENCE_LENGTH):
        f_file = seq_path / f"{f_idx}.npy"
        if not f_file.exists():
            return None
        try:
            arr = np.load(str(f_file))
            if arr.shape != (FEATURE_DIM,):
                return None
            frames.append(arr)
        except Exception:
            return None
    if len(frames) == SEQUENCE_LENGTH:
        return np.array(frames, dtype=np.float32)
    return None


def load_dataset_119(data_dir: Path, label_map_file: Path, cache_name="dataset_cache_119.npz"):
    """
    Nạp toàn bộ 119 nhãn:
    - Nếu có cache_name (.npz), nạp siêu tốc trong < 2 giây.
    - Nếu chưa có, nạp đa luồng (16 workers) và nén lưu cache.
    """
    cache_path = data_dir / cache_name
    if cache_path.exists():
        print(f"=> [CACHE] Tìm thấy tệp cache tại: {cache_path}")
        print("=> Đang giải nén nhanh vào RAM...")
        t0 = time.time()
        c = np.load(str(cache_path), allow_pickle=True)
        X = c['X']
        y = c['y']
        label_to_id = c['label_to_id'].item()
        id_to_label = c['id_to_label'].item()
        print(f"=> Nạp xong từ Cache trong {time.time()-t0:.2f}s! X.shape={X.shape}, y.shape={y.shape}\n", flush=True)
        return X, y, label_to_id, id_to_label

    print(f"=> Chưa có cache. Bắt đầu đọc dữ liệu từ: {data_dir}")
    if not label_map_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {label_map_file}")

    with open(label_map_file, 'r', encoding='utf-8') as f:
        lmap_data = json.load(f)

    label_to_id = lmap_data['label_to_id']
    id_to_label = {int(k): v for k, v in lmap_data['id_to_label'].items()}
    classes = lmap_data['classes']

    print(f"=> Đang nạp {len(classes)} nhãn theo đúng chuẩn label_map_119.json...")
    all_sequences = []
    all_targets = []

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for idx, lbl in enumerate(classes, 1):
            lbl_dir = data_dir / lbl
            if not lbl_dir.exists():
                print(f"  [CẢNH BÁO] Không tìm thấy thư mục nhãn: {lbl}")
                continue

            seq_dirs = sorted([d for d in lbl_dir.iterdir() if d.is_dir()])
            results = list(pool.map(_load_single_sequence, seq_dirs))
            valid_seqs = [r for r in results if r is not None]

            class_id = label_to_id[lbl]
            all_sequences.extend(valid_seqs)
            all_targets.extend([class_id] * len(valid_seqs))

            if idx % 20 == 0 or idx == len(classes):
                print(f"  -> Đã nạp {idx:3d}/{len(classes)} nhãn ({len(all_sequences):,d} sequences)")

    X = np.array(all_sequences, dtype=np.float32)
    y = to_categorical(all_targets, num_classes=len(classes)).astype(np.float32)

    print(f"=> Nạp hoàn tất trong {time.time()-t_start:.1f}s! X: {X.shape}, y: {y.shape}")
    print(f"=> Đang lưu cache nén vào {cache_path}...")
    np.savez_compressed(str(cache_path), X=X, y=y, label_to_id=label_to_id, id_to_label=id_to_label)
    print("=> Đã lưu cache thành công!\n")

    return X, y, label_to_id, id_to_label


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Baseline 119 classes (Task MERGE-5)")
    parser.add_argument("--epochs", type=int, default=60, help="Số epoch tối đa (mặc định: 60)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (mặc định: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (mặc định: 0.001)")
    parser.add_argument("--patience", type=int, default=15, help="Patience cho EarlyStopping (mặc định: 15)")
    parser.add_argument("--data_dir", type=str, default="", help="Thư mục dataset (mặc định: tự động tìm)")
    parser.add_argument("--output_model", type=str, default="", help="Đường dẫn lưu model (mặc định: Models/baseline_119_tanh.h5)")
    parser.add_argument("--eval_only", action="store_true", help="Chỉ đánh giá model đã lưu")
    args = parser.parse_args()

    print("=" * 80)
    print("  TASK MERGE-5: HUẤN LUYỆN BASELINE FSIGN 119 NHÃN (KIẾN TRÚC TANH CHỐT)")
    print("=" * 80)

    # 1. Định vị đường dẫn
    base_dir = SCRIPT_DIR
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "Data_normalized_merged119"
    models_dir = base_dir / "Models"
    logs_dir = base_dir / "Logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    label_map_file = models_dir / "label_map_119.json"
    output_model_path = Path(args.output_model) if args.output_model else models_dir / "baseline_119_tanh.h5"

    # 2. Cấu hình phần cứng
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"=> [DEVICE] Tìm thấy {len(gpus)} GPU. Đã bật Memory Growth:")
            for i, g in enumerate(gpus):
                print(f"   [{i}] {g.name}")
        except Exception as e:
            print(f"=> [DEVICE] Cảnh báo cấu hình GPU: {e}")
    else:
        print("=> [DEVICE] Không tìm thấy GPU NVIDIA, huấn luyện trên CPU đa lõi.")

    # 3. Nạp dữ liệu
    print(f"\n[BƯỚC 1/4] Nạp dữ liệu từ: {data_dir}")
    X, y, label_to_id, id_to_label = load_dataset_119(data_dir, label_map_file)
    num_classes = len(label_to_id)
    assert num_classes == 119, f"Kỳ vọng 119 classes, nhận được {num_classes}"
    assert X.shape[1:] == (SEQUENCE_LENGTH, FEATURE_DIM), f"Shape không đúng: {X.shape}"

    # 4. Phân chia Train / Test phân tầng (Stratified Split 80/20)
    print("\n[BƯỚC 2/4] Phân chia dữ liệu phân tầng Stratified 80/20...")
    y_indices = np.argmax(y, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y_indices
    )

    y_train_idx = np.argmax(y_train, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)

    print(f"  -> Tập huấn luyện (Train 80%): {X_train.shape[0]:,d} sequences")
    print(f"  -> Tập kiểm thử (Test 20%):     {X_test.shape[0]:,d} sequences")
    print(f"  -> Kiểm tra cân bằng: 100% {num_classes} classes đều có đúng 80% train và 20% test.")

    # 5. Khởi tạo mô hình
    print("\n[BƯỚC 3/4] Khởi tạo mô hình Baseline 119 (128 -> 64 -> 32, Tanh, clipnorm=1.0)...")
    model = build_model(
        input_shape=(SEQUENCE_LENGTH, FEATURE_DIM),
        num_classes=num_classes,
        lr=args.lr,
        clipnorm=1.0
    )
    model.summary()

    tracker = GradientAndMetricsTracker()

    if not args.eval_only:
        tb_dir = logs_dir / ("train_baseline119_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        callbacks = [
            tracker,
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=args.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(output_model_path),
                monitor='val_categorical_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(log_dir=str(tb_dir))
        ]

        print(f"\n[BƯỚC 4/4] Bắt đầu huấn luyện tối đa {args.epochs} epochs (Batch size={args.batch_size})...\n")
        t_train_start = time.time()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=0  # Tracker đã in log chi tiết từng epoch
        )

        total_train_dur = time.time() - t_train_start
        print(f"\n=> Quá trình huấn luyện kết thúc sau {int(total_train_dur//60)}m {int(total_train_dur%60)}s!")
        print(f"=> Tải lại trọng số tốt nhất từ: {output_model_path}")
        model.load_weights(str(output_model_path))

        # Lưu lịch sử training
        history_out_file = logs_dir / "baseline_119_training_history.json"
        with open(history_out_file, 'w', encoding='utf-8') as f:
            json.dump({
                'epochs_run': len(tracker.history_records),
                'total_duration_s': round(total_train_dur, 2),
                'nan_detected': tracker.nan_detected,
                'records': tracker.history_records
            }, f, ensure_ascii=False, indent=2)
        print(f"=> Đã lưu lịch sử huấn luyện vào: {history_out_file}")

    else:
        print(f"\n[CHẾ ĐỘ ĐÁNH GIÁ] Đang nạp model từ: {output_model_path}")
        model.load_weights(str(output_model_path))

    # 6. ĐÁNH GIÁ TOÀN DIỆN TRÊN TẬP TEST (1,368 SEQUENCES)
    print("\n" + "=" * 80)
    print("  KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST (1,368 MẪU - 119 NHÃN)")
    print("=" * 80)

    # Dự đoán chi tiết trước
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)

    eval_res = model.evaluate(X_test, y_test, verbose=1)
    test_loss = float(eval_res[0])
    test_acc = float(accuracy_score(y_test_idx, y_pred_idx))
    test_top3 = float(top_k_accuracy_score(y_test_idx, y_pred_probs, k=3, labels=list(range(num_classes))))
    test_top5 = float(top_k_accuracy_score(y_test_idx, y_pred_probs, k=5, labels=list(range(num_classes))))

    print(f"\n1. CHỈ SỐ TỔNG THỂ:")
    print(f"   - Test Loss:               {test_loss:.4f}")
    print(f"   - Test Accuracy (Top-1):   {test_acc * 100:.2f}% ({int(test_acc * len(X_test))}/{len(X_test)} mẫu)")
    print(f"   - Top-3 Accuracy:          {test_top3 * 100:.2f}%")
    print(f"   - Top-5 Accuracy:          {test_top5 * 100:.2f}%")
    print(f"   - Trạng thái Gradient:     {'CÓ LỖI BÙNG NỔ GRADIENT (NaN)' if tracker.nan_detected else 'ỔN ĐỊNH HOÀN TOÀN (0 NaN)'}")

    # Classification report
    label_names = [id_to_label[i] for i in range(num_classes)]
    cls_report = classification_report(
        y_test_idx,
        y_pred_idx,
        labels=list(range(num_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    # Phân tích theo từng nhãn: xếp hạng accuracy tăng dần
    per_class_metrics = []
    for i, lbl in enumerate(label_names):
        cls_data = cls_report[lbl]
        mask = (y_test_idx == i)
        n_test = np.sum(mask)
        n_correct = np.sum((y_test_idx == i) & (y_pred_idx == i))
        cls_acc = n_correct / n_test if n_test > 0 else 0.0

        per_class_metrics.append({
            'class_id': i,
            'label': lbl,
            'test_samples': int(n_test),
            'correct': int(n_correct),
            'accuracy': round(cls_acc, 4),
            'precision': round(cls_data['precision'], 4),
            'recall': round(cls_data['recall'], 4),
            'f1_score': round(cls_data['f1-score'], 4)
        })

    per_class_sorted = sorted(per_class_metrics, key=lambda x: (x['accuracy'], x['f1_score']))

    print("\n2. TOP 10 NHÃN CÓ ACCURACY THẤP NHẤT:")
    print(f"   {'STT':<4} {'Tên Nhãn':<25} {'Số mẫu test':<12} {'Đúng/Tổng':<12} {'Accuracy':<10} {'F1-Score':<10}")
    print("   " + "-" * 75)
    for idx, item in enumerate(per_class_sorted[:10], 1):
        print(f"   {idx:<4} {item['label']:<25} {item['test_samples']:<12} {item['correct']}/{item['test_samples']:<10} {item['accuracy']*100:6.2f}%   {item['f1_score']*100:6.2f}%")

    # Phân tích các cặp nhầm lẫn phổ biến nhất
    cm = confusion_matrix(y_test_idx, y_pred_idx, labels=list(range(num_classes)))
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'true_label': label_names[i],
                    'pred_label': label_names[j],
                    'count': int(cm[i, j])
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)

    print("\n3. TOP CÁC CẶP NHÃN DỄ BỊ NHẬN DIỆN NHẦM NHẤT:")
    print(f"   {'Nhãn Thực Tế':<25} -> {'Nhãn Đoán Nhầm':<25} : {'Số Lần Nhầm':<10}")
    print("   " + "-" * 65)
    for p in confused_pairs[:10]:
        print(f"   {p['true_label']:<25} -> {p['pred_label']:<25} : {p['count']} lần")

    # Lưu kết quả đánh giá ra JSON
    eval_out_file = logs_dir / "baseline_119_test_evaluation.json"
    with open(eval_out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'overall': {
                'test_loss': round(test_loss, 4),
                'test_accuracy': round(test_acc, 4),
                'top_3_accuracy': round(test_top3, 4),
                'top_5_accuracy': round(test_top5, 5),
                'total_test_samples': len(X_test),
                'correct_predictions': int(np.sum(y_pred_idx == y_test_idx)),
                'gradient_explosion': tracker.nan_detected
            },
            'per_class': per_class_metrics,
            'top_10_lowest': per_class_sorted[:10],
            'top_confused_pairs': confused_pairs[:15]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=> Đã lưu toàn bộ báo cáo chi tiết vào: {eval_out_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
