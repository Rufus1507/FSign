# -*- coding: utf-8 -*-
"""
extract_all_new_dataset.py
==========================
Pipeline trích xuất MediaPipe Holistic tự động cho toàn bộ 100 nhãn mới từ dataset/train/
và lưu vào Sign Language Translator/Data/<Tên_Nhãn>/<seq_idx>/<0..59>.npy

Hỗ trợ:
- Đa luồng (Multi-threading / Multi-worker) tăng tốc xử lý
- Resumable: Tự động bỏ qua các video/nhãn đã trích xuất hoàn chỉnh
- Linear interpolation (resample) chuẩn 60 frames mỗi sequence
- Unicode UTF-8 chuẩn cho tên nhãn tiếng Việt có dấu
"""

import os
import sys
import types
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix Unicode UTF-8 output cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Protobuf & TF compatibility shims ──────────────────────────────────────────
try:
    import google.protobuf.message_factory as mf
    if not hasattr(mf, 'GetMessageClass'):
        _factory = mf.MessageFactory()
        mf.GetMessageClass = _factory.GetPrototype
except Exception:
    pass

if 'tensorflow.tools.docs.doc_controls' not in sys.modules:
    m = types.ModuleType('tensorflow.tools.docs.doc_controls')
    m.do_not_generate_docs = lambda x: x
    m.doc_private = lambda x: x
    sys.modules['tensorflow.tools.docs.doc_controls'] = m
    sys.modules['tensorflow.tools.docs'] = types.ModuleType('tensorflow.tools.docs')
    sys.modules['tensorflow.tools'] = types.ModuleType('tensorflow.tools')
    sys.modules['tensorflow'] = types.ModuleType('tensorflow')

import cv2
import numpy as np
import mediapipe as mp

SEQUENCE_LENGTH = 60
FEATURE_DIM = 126
HAND_DIM = 63


def resample_sequence(frames, target_length=SEQUENCE_LENGTH):
    """
    Nội suy tuyến tính N frames (126,) thành target_length (60,) frames.
    """
    frames_arr = np.array(frames, dtype=np.float32)
    N, D = frames_arr.shape
    if N == target_length:
        return frames_arr
    if N == 0:
        return np.zeros((target_length, FEATURE_DIM), dtype=np.float32)

    x_old = np.linspace(0.0, 1.0, N)
    x_new = np.linspace(0.0, 1.0, target_length)

    resampled = np.zeros((target_length, D), dtype=np.float32)
    for d in range(D):
        resampled[:, d] = np.interp(x_new, x_old, frames_arr[:, d])
    return resampled


def extract_video_keypoints(video_path, holistic):
    """
    Đọc 1 video mp4 và trích xuất keypoints [lh(63) + rh(63)] cho từng frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        lh = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
              if results.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32))
        rh = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
              if results.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32))

        keypoints = np.concatenate([lh, rh])
        raw_frames.append(keypoints)

    cap.release()

    if not raw_frames:
        return None

    # Resample thành đúng SEQUENCE_LENGTH = 60 frames
    seq_60 = resample_sequence(raw_frames, target_length=SEQUENCE_LENGTH)
    return seq_60


def process_single_label(label_dir, output_data_dir):
    """
    Xử lý tất cả video của 1 nhãn (folder).
    """
    label_name = label_dir.name
    target_label_dir = output_data_dir / label_name
    target_label_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(label_dir.glob('*.mp4')))
    extracted_count = 0
    skipped_count = 0
    failed_count = 0

    # Khởi tạo instance Holistic riêng cho từng thread/worker
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        for seq_idx, video_path in enumerate(videos):
            seq_dir = target_label_dir / str(seq_idx)
            
            # Kiểm tra nếu sequence đã được trích xuất hoàn chỉnh (đủ 60 frames)
            if seq_dir.exists():
                existing_npys = list(seq_dir.glob('*.npy'))
                if len(existing_npys) == SEQUENCE_LENGTH:
                    skipped_count += 1
                    continue

            seq_dir.mkdir(parents=True, exist_ok=True)
            seq_keypoints = extract_video_keypoints(video_path, holistic)

            if seq_keypoints is None:
                failed_count += 1
                continue

            # Lưu 60 file npy
            for frame_idx in range(SEQUENCE_LENGTH):
                np.save(str(seq_dir / f"{frame_idx}.npy"), seq_keypoints[frame_idx])

            extracted_count += 1

    return {
        'label': label_name,
        'total': len(videos),
        'extracted': extracted_count,
        'skipped': skipped_count,
        'failed': failed_count
    }


def main():
    parser = argparse.ArgumentParser(description="Trích xuất MediaPipe dataset mới vào Data/")
    parser.add_argument('--base_dir', type=str, default='h:/PythonProject/FSign', help="Thư mục gốc FSign")
    parser.add_argument('--workers', type=int, default=4, help="Số worker song song (mặc định: 4)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    train_dir = base_dir / 'dataset' / 'train'
    output_data_dir = base_dir / 'Sign Language Translator' / 'Data'

    if not train_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục dataset: {train_dir}")
        return

    label_dirs = [d for d in sorted(train_dir.iterdir()) if d.is_dir()]
    total_labels = len(label_dirs)
    total_videos = sum(len(list(d.glob('*.mp4'))) for d in label_dirs)

    print("=" * 65)
    print("  FSIGN - PIPELINE TRÍCH XUẤT ĐẶC TRƯNG MEDIAPIPE HOLISTIC")
    print("=" * 65)
    print(f"Dataset nguồn:       {train_dir}")
    print(f"Tổng số nhãn mới:    {total_labels} nhãn")
    print(f"Tổng số video:       {total_videos:,} files (.mp4)")
    print(f"Thư mục lưu Data:    {output_data_dir}")
    print(f"Số worker song song: {args.workers}")
    print(f"Độ dài sequence:     {SEQUENCE_LENGTH} frames")
    print("-" * 65)

    t_start = time.time()
    completed_labels = 0
    total_extracted = 0
    total_skipped = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_label, l_dir, output_data_dir): l_dir
            for l_dir in label_dirs
        }

        for future in as_completed(futures):
            res = future.result()
            completed_labels += 1
            total_extracted += res['extracted']
            total_skipped += res['skipped']
            total_failed += res['failed']

            elapsed = time.time() - t_start
            pct = (completed_labels / total_labels) * 100
            speed = completed_labels / elapsed if elapsed > 0 else 0
            eta_s = (total_labels - completed_labels) / speed if speed > 0 else 0

            print(
                f"[{completed_labels:3d}/{total_labels}] ({pct:5.1f}%) "
                f"Nhãn: '{res['label']:<20}' | "
                f"Mới: +{res['extracted']:2d} | Bỏ qua: {res['skipped']:2d} | Lỗi: {res['failed']:2d} | "
                f"ETA: {int(eta_s//60):02d}m{int(eta_s%60):02d}s"
            )

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print("HOÀN TẤT TRÍCH XUẤT ĐẶC TRƯNG MEDIAPIPE!")
    print(f"- Tổng nhãn đã xử lý:       {completed_labels}/{total_labels}")
    print(f"- Tổng sequences mới tạo:    {total_extracted:,}")
    print(f"- Sequences đã có sẵn:      {total_skipped:,}")
    print(f"- Video không đọc được:     {total_failed}")
    print(f"- Tổng thời gian:            {int(total_time//60)}m {int(total_time%60)}s")
    print(f"- Dữ liệu đã sẵn sàng tại:  {output_data_dir}")
    print("=" * 65)


if __name__ == '__main__':
    main()
