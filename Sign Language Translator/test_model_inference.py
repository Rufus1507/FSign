# -*- coding: utf-8 -*-
"""
test_model_inference.py
=======================
Kiểm tra đánh giá nhanh mô hình FSign 159 classes trên các mẫu dữ liệu thực tế
(cả từ file .npy trong Data/ và video .mp4 trong dataset/train/).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Fix Unicode UTF-8 output cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import keras
from keras.models import load_model

SEQUENCE_LENGTH = 60
FEATURE_DIM = 126


def load_label_map(label_map_path):
    with open(label_map_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if 'id_to_display' in data:
            return {int(k): v for k, v in data['id_to_display'].items()}
        elif 'id_to_label' in data:
            return {int(k): v for k, v in data['id_to_label'].items()}
        elif 'classes' in data:
            return {i: c for i, c in enumerate(data['classes'])}
    return {}


def test_random_samples(data_dir, model, id_to_label, num_samples=10):
    data_path = Path(data_dir)
    label_folders = [d for d in data_path.iterdir() if d.is_dir() and d.name != 'cam on']

    if not label_folders:
        print("[LỖI] Không tìm thấy thư mục nhãn trong Data/")
        return

    np.random.seed(42)
    selected_labels = np.random.choice(label_folders, size=min(num_samples, len(label_folders)), replace=False)

    print("\n" + "=" * 75)
    print(f"  FSIGN - KIỂM THỬ MÔ HÌNH TRÊN {len(selected_labels)} MẪU DỮ LIỆU THỰC TẾ")
    print("=" * 75)

    correct_count = 0
    top5_count = 0
    latencies = []

    for idx, l_folder in enumerate(selected_labels, 1):
        true_label = l_folder.name
        seq_dirs = [s for s in l_folder.iterdir() if s.is_dir()]
        if not seq_dirs:
            continue

        seq_dir = np.random.choice(seq_dirs)
        frames = []
        for f_idx in range(SEQUENCE_LENGTH):
            f_path = seq_dir / f"{f_idx}.npy"
            if f_path.exists():
                frames.append(np.load(str(f_path)))
            else:
                break

        if len(frames) != SEQUENCE_LENGTH:
            continue

        input_data = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)

        t0 = time.time()
        preds = model.predict(input_data, verbose=0)[0]
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        top_indices = np.argsort(preds)[::-1][:5]
        top1_idx = top_indices[0]
        top1_label = id_to_label.get(top1_idx, f"Class {top1_idx}")
        top1_conf = preds[top1_idx] * 100

        is_top1 = (top1_label == true_label)
        if is_top1:
            correct_count += 1
            result_str = "[ĐÚNG]"
        else:
            result_str = "[CHƯA ĐÚNG]"

        top5_labels = [id_to_label.get(i, f"Class {i}") for i in top_indices]
        is_top5 = true_label in top5_labels
        if is_top5:
            top5_count += 1

        top5_preview = " | ".join([f"{id_to_label.get(i,'?')}: {preds[i]*100:.1f}%" for i in top_indices[:3]])

        print(f"[{idx:2d}] Nhãn thực tế:  '{true_label}'")
        print(f"     Dự đoán Top-1: '{top1_label}' ({top1_conf:.1f}%) -> {result_str}")
        print(f"     Top-3:         {top5_preview}")
        print(f"     Độ trễ xử lý:  {latency_ms:.1f} ms")
        print("-" * 75)

    avg_latency = sum(latencies) / max(len(latencies), 1)
    print(f"\n=> TỔNG KẾT KIỂM THỬ:")
    print(f"  - Top-1 Accuracy: {correct_count}/{len(selected_labels)} ({correct_count/max(len(selected_labels),1)*100:.1f}%)")
    print(f"  - Top-5 Accuracy: {top5_count}/{len(selected_labels)} ({top5_count/max(len(selected_labels),1)*100:.1f}%)")
    print(f"  - Độ trễ trung bình: {avg_latency:.1f} ms / sequence (~{1000/avg_latency:.0f} FPS)")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Kiểm thử mô hình FSign-159")
    parser.add_argument('--samples', type=int, default=10, help="Số mẫu kiểm thử")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    release_dir = script_dir / 'release'
    data_dir = script_dir / 'Data'
    model_path = release_dir / 'fsign_159classes.h5'
    label_map_path = release_dir / 'label_map.json'

    if not model_path.exists():
        print(f"[LỖI] Không tìm thấy model tại: {model_path}")
        return

    print(f"=> Đang tải model từ: {model_path}...")
    model = load_model(str(model_path), compile=False)
    id_to_label = load_label_map(label_map_path)

    test_random_samples(data_dir, model, id_to_label, num_samples=args.samples)


if __name__ == '__main__':
    main()
