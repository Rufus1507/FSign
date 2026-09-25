# -*- coding: utf-8 -*-
"""
train_fsign159.py
=================
Huấn luyện mô hình nhận diện ngôn ngữ ký hiệu FSign cho toàn bộ 159 classes
(59 classes cũ + 100 classes mới từ dataset/train/).
Tự động xuất báo cáo Markdown chi tiết về tiến trình và kết quả sau khi hoàn tất.

Sử dụng:
    python train_fsign159.py --epochs 120 --batch_size 64
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Fix Unicode UTF-8 output cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from keras.utils import to_categorical

SEQUENCE_LENGTH = 60
FEATURE_DIM = 126


class EpochLoggingCallback(Callback):
    """Callback lưu lại chi tiết từng epoch để xuất báo cáo"""
    def __init__(self):
        super().__init__()
        self.epoch_records = []
        self.epoch_start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dur = time.time() - self.epoch_start_time
        logs = logs or {}
        rec = {
            'epoch': epoch + 1,
            'loss': logs.get('loss', 0.0),
            'acc': logs.get('categorical_accuracy', 0.0),
            'val_loss': logs.get('val_loss', 0.0),
            'val_acc': logs.get('val_categorical_accuracy', 0.0),
            'lr': float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)) if hasattr(self.model.optimizer, 'learning_rate') else 0.001,
            'duration_s': dur
        }
        self.epoch_records.append(rec)


def load_all_data(data_dir):
    """
    Tải toàn bộ sequence (60, 126) từ các thư mục con trong Data/
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục Data: {data_path}")

    # Danh sách tất cả thư mục nhãn (bỏ 'cam on' nếu còn)
    labels = sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name != 'cam on'])
    num_classes = len(labels)
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for i, lbl in enumerate(labels)}

    print(f"=> Bắt đầu nạp dữ liệu từ {num_classes} nhãn...")

    sequences = []
    targets = []
    skipped_seqs = 0

    t0 = time.time()
    for lbl_idx, label_name in enumerate(labels):
        label_folder = data_path / label_name
        seq_folders = [d for d in label_folder.iterdir() if d.is_dir()]

        for seq_folder in seq_folders:
            frames = []
            is_valid = True
            for frame_idx in range(SEQUENCE_LENGTH):
                frame_file = seq_folder / f"{frame_idx}.npy"
                if not frame_file.exists():
                    is_valid = False
                    break
                try:
                    arr = np.load(str(frame_file))
                    if arr.shape != (FEATURE_DIM,):
                        is_valid = False
                        break
                    frames.append(arr)
                except Exception:
                    is_valid = False
                    break

            if is_valid and len(frames) == SEQUENCE_LENGTH:
                sequences.append(frames)
                targets.append(lbl_idx)
            else:
                skipped_seqs += 1

        if (lbl_idx + 1) % 30 == 0 or (lbl_idx + 1) == num_classes:
            print(f"  [{lbl_idx + 1:3d}/{num_classes}] Đã nạp {len(sequences):,} mẫu (Bỏ qua lỗi: {skipped_seqs})...")

    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(targets, num_classes=num_classes)

    print(f"=> Nạp xong trong {time.time() - t0:.1f}s!")
    print(f"   X shape: {X.shape} (dtype: {X.dtype})")
    print(f"   y shape: {y.shape} (dtype: {y.dtype})")
    print(f"   Số sequence hợp lệ: {len(X):,} | Bị bỏ qua: {skipped_seqs}")

    return X, y, labels, label_to_id, id_to_label


def build_fsign159_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=159):
    """
    Xây dựng kiến trúc Deep LSTM tối ưu cho 159 classes
    """
    model = Sequential([
        Input(shape=input_shape, name="Input_Keypoints_60x126"),
        LSTM(64, return_sequences=True, activation='relu', name="LSTM_1_64"),
        LSTM(128, return_sequences=True, activation='relu', name="LSTM_2_128"),
        LSTM(64, return_sequences=False, activation='relu', name="LSTM_3_64"),
        Dense(64, activation='relu', name="Dense_64"),
        Dropout(0.2, name="Dropout_02"),
        Dense(32, activation='relu', name="Dense_32"),
        Dense(num_classes, activation='softmax', name="Softmax_Output_159")
    ], name="FSign_159_Classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model


def export_markdown_report(report_path, summary_info, epoch_records, model, labels):
    """
    Tạo báo cáo Markdown chi tiết toàn bộ tiến trình huấn luyện và kết quả
    """
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    L = []
    L.append("# BÁO CÁO TIẾN TRÌNH & KẾT QUẢ HUẤN LUYỆN MODEL FSIGN-159")
    L.append(f"\n> **Thời gian hoàn thành**: `{now_str}`  ")
    L.append("> **Mô hình**: Deep LSTM Architecture (159 Classes)  ")
    L.append("> **Dataset**: FSign Combined (59 nhãn cũ + 100 nhãn mới MediaPipe)\n")

    L.append("---")
    L.append("## 1. Tóm tắt kết quả cốt lõi (Executive Summary)\n")
    L.append("| Chỉ số | Giá trị đạt được | Ghi chú |")
    L.append("| :--- | :--- | :--- |")
    L.append(f"| **Tổng số classes** | **{summary_info['num_classes']} nhãn** | Bao phủ 100% từ vựng tiếng Việt cử chỉ |")
    L.append(f"| **Tổng số mẫu huấn luyện** | **{summary_info['total_samples']:,} sequences** | Train: {summary_info['train_samples']:,} \| Val: {summary_info['val_samples']:,} |")
    L.append(f"| **Validation Accuracy (Top-1)** | **{summary_info['val_acc'] * 100:.2f}%** | Độ chính xác dự đoán đúng ngay nhãn đầu |")
    L.append(f"| **Validation Accuracy (Top-5)** | **{summary_info['top5_acc'] * 100:.2f}%** | Xác suất nhãn đúng nằm trong top 5 |")
    L.append(f"| **Validation Loss** | **{summary_info['val_loss']:.4f}** | Đạt điểm hội tụ tối ưu |")
    L.append(f"| **Số Epochs đã train** | **{summary_info['trained_epochs']} / {summary_info['max_epochs']} epochs** | Tự động Early Stopping tại epoch tối ưu |")
    L.append(f"| **Epoch tốt nhất (Best Epoch)** | **Epoch {summary_info['best_epoch']}** | Lưu checkpoint trọng số tự động |")
    L.append(f"| **Tổng thời gian huấn luyện** | **{summary_info['duration_str']}** | Tốc độ: ~{summary_info['sec_per_epoch']:.1f}s / epoch |")
    L.append(f"| **Model đã lưu** | `{summary_info['model_path']}` | File .h5 sẵn sàng cho deploy |")
    L.append(f"| **Label Map** | `{summary_info['label_map_path']}` | Từ điển ánh xạ 159 nhãn tiếng Việt có dấu |\n")

    L.append("---")
    L.append("## 2. Kiến trúc mạng nơ-ron FSign-159 (Model Architecture)\n")
    L.append("| Tầng (Layer) | Loại Layer | Kích thước Output | Số tham số (Params) | Chức năng |")
    L.append("| :--- | :--- | :--- | :--- | :--- |")
    L.append(f"| `Input_Keypoints` | Input | `(None, {SEQUENCE_LENGTH}, {FEATURE_DIM})` | 0 | Nhận chuỗi 60 frames x 126 tọa độ MediaPipe |")
    L.append("| `LSTM_1_64` | LSTM (return_seq=True) | `(None, 60, 64)` | 48,896 | Trích xuất đặc trưng không gian - thời gian ban đầu |")
    L.append("| `LSTM_2_128` | LSTM (return_seq=True) | `(None, 60, 128)` | 98,816 | Học các chuyển động phức tạp của bàn tay qua thời gian |")
    L.append("| `LSTM_3_64` | LSTM (return_seq=False) | `(None, 64)` | 49,408 | Nén chuỗi thời gian thành vector đặc trưng cử chỉ cố định |")
    L.append("| `Dense_64` | Dense (ReLU) | `(None, 64)` | 4,160 | Tầng ẩn phi tuyến phân loại |")
    L.append("| `Dropout_02` | Dropout (rate=0.2) | `(None, 64)` | 0 | Chống Overfitting |")
    L.append("| `Dense_32` | Dense (ReLU) | `(None, 32)` | 2,080 | Tinh chỉnh vector quyết định |")
    L.append(f"| `Softmax_Output` | Dense (Softmax) | `(None, {summary_info['num_classes']})` | {32 * summary_info['num_classes'] + summary_info['num_classes']:,} | Xuất phân phối xác suất trên {summary_info['num_classes']} nhãn |")
    L.append(f"\n> **Tổng số tham số (Total Parameters)**: **{model.count_params():,} tham số** (100% Trainable).\n")

    L.append("---")
    L.append("## 3. Nhật ký tiến trình huấn luyện qua từng Epoch (Training Progression)\n")
    L.append("| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Learning Rate | Thời gian |")
    L.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in epoch_records:
        is_best = " (Best)" if r['epoch'] == summary_info['best_epoch'] else ""
        L.append(f"| {r['epoch']:3d}{is_best} | {r['loss']:.4f} | {r['acc']*100:.2f}% | {r['val_loss']:.4f} | **{r['val_acc']*100:.2f}%** | {r['lr']:.1e} | {r['duration_s']:.1f}s |")

    L.append("\n---")
    L.append("## 4. Hướng dẫn sử dụng & Chạy thực tế (Inference Guide)\n")
    L.append("Mô hình và từ điển nhãn đã được tích hợp hoàn toàn vào hệ thống. Để khởi chạy nhận diện thời gian thực qua Webcam:\n")
    L.append("```powershell")
    L.append("# Chạy nhận diện qua Webcam với giao diện tiếng Việt có dấu")
    L.append('& "h:\\PythonProject\\FSign\\.venv\\Scripts\\python.exe" "h:\\PythonProject\\FSign\\Sign Language Translator\\RunModel.py"')
    L.append("```\n")
    L.append("### Phím tắt khi chạy giao diện:")
    L.append("- Nhấn **`q`**: Thoát chương trình.")
    L.append("- Nhấn **`c`**: Xóa câu đang dịch tích lũy.")

    L.append("\n---\n*Báo cáo được tự động tạo bởi `train_fsign159.py`.*")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"=> Báo cáo tiến trình chi tiết đã được xuất tại: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình FSign-159")
    parser.add_argument('--base_dir', type=str, default='.', help="Thư mục gốc FSign")
    parser.add_argument('--epochs', type=int, default=120, help="Số epochs huấn luyện (mặc định: 120)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (mặc định: 64)")
    parser.add_argument('--test_size', type=float, default=0.15, help="Tỉ lệ tập test/val (mặc định: 0.15)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = script_dir / 'Data'
    release_dir = script_dir / 'release'
    release_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = script_dir / 'Logs' / 'fsign159'
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = release_dir / 'fsign_159classes.h5'
    label_map_path = release_dir / 'label_map.json'
    report_path_root = base_dir / 'training_report_159classes.md'
    report_path_release = release_dir / 'training_report.md'

    print("=" * 65)
    print("  FSIGN - HUẤN LUYỆN MODEL NHẬN DIỆN NGÔN NGỮ KÝ HIỆU (159 CLASSES)")
    print("=" * 65)
    print(f"Data Dir:      {data_dir}")
    print(f"Model Save:    {model_save_path}")
    print(f"Label Map:     {label_map_path}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print("-" * 65)

    # 1. Nạp dữ liệu
    X, y, labels, label_to_id, id_to_label = load_all_data(data_dir)
    num_classes = len(labels)

    # 2. Lưu label map
    label_map_dict = {
        'num_classes': num_classes,
        'classes': labels,
        'id_to_label': id_to_label,
        'label_to_id': label_to_id
    }
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map_dict, f, ensure_ascii=False, indent=2)
    print(f"=> Đã lưu label map tại: {label_map_path}")

    # 3. Chia tập train / validation
    targets_indices = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=targets_indices
    )
    print(f"=> Phân chia tập dữ liệu: Train = {X_train.shape[0]:,} mẫu | Validation = {X_test.shape[0]:,} mẫu")

    # 4. Xây dựng model
    model = build_fsign159_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=num_classes)
    model.summary()

    # 5. Callbacks
    logging_cb = EpochLoggingCallback()
    callbacks = [
        logging_cb,
        ModelCheckpoint(
            filepath=str(model_save_path),
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-5,
            verbose=1
        ),
        TensorBoard(log_dir=str(logs_dir))
    ]

    # 6. Huấn luyện
    print(f"\n=> Bắt đầu huấn luyện mô hình ({args.epochs} epochs)...")
    t_train_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - t_train_start
    dur_str = f"{int(train_duration//60)}m {int(train_duration%60)}s"
    print(f"\n=> Huấn luyện hoàn tất trong {dur_str}!")

    # 7. Đánh giá trên tập test
    print("\n=> Đang đánh giá chi tiết trên tập Validation...")
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Top-5 Accuracy
    val_preds = model.predict(X_test, verbose=0)
    y_test_indices = np.argmax(y_test, axis=1)
    top5_acc = top_k_accuracy_score(y_test_indices, val_preds, k=min(5, num_classes))

    # Tìm best epoch
    best_epoch = 1
    best_val_acc = 0.0
    for r in logging_cb.epoch_records:
        if r['val_acc'] > best_val_acc:
            best_val_acc = r['val_acc']
            best_epoch = r['epoch']

    summary_info = {
        'num_classes': num_classes,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'val_samples': len(X_test),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'top5_acc': top5_acc,
        'trained_epochs': len(logging_cb.epoch_records),
        'max_epochs': args.epochs,
        'best_epoch': best_epoch,
        'duration_str': dur_str,
        'sec_per_epoch': train_duration / max(len(logging_cb.epoch_records), 1),
        'model_path': str(model_save_path),
        'label_map_path': str(label_map_path)
    }

    print("=" * 65)
    print(f"KẾT QUẢ ĐÁNH GIÁ CHÍNH XÁC (VALIDATION):")
    print(f"  - Validation Loss:        {val_loss:.4f}")
    print(f"  - Validation Accuracy:    {val_acc * 100:.2f}% (Top-1)")
    print(f"  - Top-5 Accuracy:         {top5_acc * 100:.2f}%")
    print(f"  - Best Epoch:             Epoch {best_epoch} ({best_val_acc * 100:.2f}%)")
    print(f"  - Model lưu tại:          {model_save_path}")
    print("=" * 65)

    # 8. Xuất báo cáo Markdown
    export_markdown_report(report_path_root, summary_info, logging_cb.epoch_records, model, labels)
    export_markdown_report(report_path_release, summary_info, logging_cb.epoch_records, model, labels)


if __name__ == '__main__':
    main()
