# -*- coding: utf-8 -*-
"""
verify_pipeline.py — Bộ kiểm tra toàn diện pipeline chuẩn hóa và tính toàn vẹn dataset
========================================================================================
Kiểm tra Data_normalized/:
  1. Kiểm tra cấu trúc: số nhãn, số sequence, số frame mỗi sequence.
  2. Kiểm tra shape: xác nhận 100% frame đạt chuẩn (129,).
  3. Kiểm tra tính toàn vẹn số học: NaN count = 0, Inf count = 0.
  4. Kiểm tra logic chuẩn hóa: centering cổ tay về gốc (0,0,0) và vector tương đối 3 chiều cuối.
  5. Kiểm tra tương thích với dataset_loader: nạp thử nghiệm qua load_dataset_normalized().
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# Fix Unicode console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_DIR))

from dataset_loader import load_dataset_normalized


def parse_args():
    p = argparse.ArgumentParser(description="Kiểm tra tính toàn vẹn của Data_normalized/")
    p.add_argument('--data_path', default=str(PROJECT_DIR / "Data_normalized"),
                   help="Đường dẫn tới Data_normalized")
    p.add_argument('--expected_dims', type=int, default=129,
                   help="Số chiều kỳ vọng của mỗi frame (mặc định: 129)")
    p.add_argument('--seq_length', type=int, default=60,
                   help="Số frame mỗi sequence (mặc định: 60)")
    p.add_argument('--fast_sample', type=int, default=0,
                   help="Nếu > 0, chỉ kiểm tra ngẫu nhiên N sequences mỗi nhãn để tăng tốc")
    return p.parse_args()


def verify_dataset(data_path: str, expected_dims: int = 129, seq_length: int = 60, sample_per_label: int = 0):
    data_dir = Path(data_path)
    print("=" * 68)
    print("       KIỂM TRA TÍNH TOÀN VẸN PIPELINE & DATASET NORMALIZED       ")
    print("=" * 68)
    print(f"  • Thư mục kiểm tra:      {data_dir}")
    print(f"  • Kích thước frame kỳ vọng: ({expected_dims},)")
    print(f"  • Độ dài sequence kỳ vọng:  {seq_length} frames")
    print(f"  • Chế độ quét:           {'Toàn bộ 100% dataset' if sample_per_label <= 0 else f'Lấy mẫu {sample_per_label} seq/nhãn'}")
    print("=" * 68)

    if not data_dir.is_dir():
        print(f"[ERROR] Thư mục không tồn tại: {data_dir}")
        return False, {}

    labels = sorted([d for d in os.listdir(data_dir) if (data_dir / d).is_dir()])
    print(f"\n[1] Cấu trúc tổng quan:")
    print(f"    • Số lượng nhãn phát hiện: {len(labels)} nhãn")

    total_seqs_checked = 0
    total_frames_checked = 0
    shape_errors = 0
    nan_errors = 0
    inf_errors = 0
    missing_frame_seqs = 0
    wrist_centering_ok = True
    rel_wrist_active_count = 0

    label_seq_counts = {}

    print("\n[2] Đang quét kiểm tra chi tiết từng sequence...")
    for idx, label in enumerate(labels, 1):
        label_dir = data_dir / label
        seq_dirs = sorted([d for d in os.listdir(label_dir) if (label_dir / d).is_dir()])
        label_seq_counts[label] = len(seq_dirs)

        targets = seq_dirs if sample_per_label <= 0 else seq_dirs[:sample_per_label]

        for seq_name in targets:
            seq_path = label_dir / seq_name
            total_seqs_checked += 1

            frames = []
            for f_idx in range(seq_length):
                f_path = seq_path / f"{f_idx}.npy"
                if not f_path.exists():
                    missing_frame_seqs += 1
                    break
                arr = np.load(f_path)
                frames.append(arr)

            if len(frames) != seq_length:
                continue

            seq_arr = np.stack(frames)  # (seq_length, dims)
            total_frames_checked += len(frames)

            # Kiểm tra shape
            if seq_arr.shape != (seq_length, expected_dims):
                shape_errors += 1

            # Kiểm tra NaN / Inf
            if np.isnan(seq_arr).any():
                nan_errors += 1
            if np.isinf(seq_arr).any():
                inf_errors += 1

            # Kiểm tra logic: centering cổ tay (wrist idx 0 -> (0,0,0) khi có tay)
            for t in range(seq_length):
                lh = seq_arr[t, :63]
                rh = seq_arr[t, 63:126]
                rel = seq_arr[t, 126:] if expected_dims == 129 else None

                if not np.all(lh == 0.0):
                    if np.max(np.abs(lh[:3])) > 1e-5:
                        wrist_centering_ok = False
                if not np.all(rh == 0.0):
                    if np.max(np.abs(rh[:3])) > 1e-5:
                        wrist_centering_ok = False

                if rel is not None and not np.all(rel == 0.0):
                    rel_wrist_active_count += 1

        if idx % 10 == 0 or idx == len(labels):
            print(f"    • Đã quét {idx:2d}/{len(labels)} nhãn ({total_seqs_checked:,} sequences, {total_frames_checked:,} frames)...", flush=True)

    # Đánh giá độ cân bằng số sequence giữa các nhãn
    counts_list = list(label_seq_counts.values())
    min_count = min(counts_list) if counts_list else 0
    max_count = max(counts_list) if counts_list else 0
    is_balanced = (min_count == max_count)

    print("\n[3] Kết quả kiểm tra tính toàn vẹn:")
    print(f"    • Tổng sequences đã quét:     {total_seqs_checked:,}")
    print(f"    • Tổng frames đã quét:        {total_frames_checked:,}")
    print(f"    • Số sequence thiếu frame:    {missing_frame_seqs}")
    print(f"    • Lỗi kích thước (Shape err): {shape_errors}")
    print(f"    • Lỗi giá trị NaN:            {nan_errors}")
    print(f"    • Lỗi giá trị vô cùng (Inf):  {inf_errors}")
    print(f"    • Cổ tay centering gốc (0,0,0): {'ĐẠT (100% chuẩn)' if wrist_centering_ok else 'KHÔNG ĐẠT'}")
    print(f"    • Số frame phát hiện vector tương đối 2 tay: {rel_wrist_active_count:,} frames")
    print(f"    • Phân bố số sequence mỗi nhãn: Min={min_count}, Max={max_count} ({'Hoàn toàn cân bằng' if is_balanced else 'Có sự chênh lệch'})")

    # Kiểm tra tương thích với dataset_loader
    print("\n[4] Kiểm tra tương thích với dataset_loader (load_dataset_normalized)...")
    loader_success = False
    try:
        sample_labels = labels[:2]
        X, y, lmap = load_dataset_normalized(data_path=str(data_dir), actions=sample_labels, verbose=False)
        print(f"    • Nạp thành công tập mẫu ({len(sample_labels)} nhãn):")
        print(f"        - X.shape: {X.shape} (Kỳ vọng: ({len(sample_labels)*60}, 60, {expected_dims}))")
        print(f"        - y.shape: {y.shape} (One-hot {len(sample_labels)} lớp)")
        loader_success = (X.shape[1:] == (seq_length, expected_dims))
    except Exception as e:
        print(f"    • [ERROR] dataset_loader lỗi: {e}")

    all_passed = (
        shape_errors == 0 and
        nan_errors == 0 and
        inf_errors == 0 and
        missing_frame_seqs == 0 and
        wrist_centering_ok and
        loader_success
    )

    print("\n" + "=" * 68)
    print(f"  TỔNG KẾT KIỂM ĐỊNH: {'TẤT CẢ TIÊU CHÍ ĐỀU ĐẠT CHUẨN (PASS)' if all_passed else 'PHÁT HIỆN LỖI (FAIL)'}")
    print("=" * 68)

    summary = {
        'total_labels': len(labels),
        'total_seqs': total_seqs_checked,
        'total_frames': total_frames_checked,
        'shape_errors': shape_errors,
        'nan_errors': nan_errors,
        'inf_errors': inf_errors,
        'missing_frame_seqs': missing_frame_seqs,
        'wrist_centering_ok': wrist_centering_ok,
        'rel_wrist_active_count': rel_wrist_active_count,
        'min_count_per_label': min_count,
        'max_count_per_label': max_count,
        'is_balanced': is_balanced,
        'loader_success': loader_success,
        'all_passed': all_passed
    }
    return all_passed, summary


if __name__ == '__main__':
    args = parse_args()
    verify_dataset(
        data_path=args.data_path,
        expected_dims=args.expected_dims,
        seq_length=args.seq_length,
        sample_per_label=args.fast_sample
    )
