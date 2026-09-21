# -*- coding: utf-8 -*-
"""
dataset_loader.py — Normalized Dataset Loader for FSign
=========================================================
Thay thế load_dataset() của ActionDetection.py khi training trên
Data_normalized/ (chứa cả webcam_* và yt_* sequences).

Lý do cần file này:
  load_dataset() gốc dùng range(no_sequences) với số nguyên → chỉ đọc
  thư mục tên số ('0', '1', ...), KHÔNG đọc được 'webcam_0', 'yt_0'.
  Hàm load_dataset_normalized() ở đây dùng os.listdir() để đọc MỌI
  thư mục con, không phân biệt tên.

KHÔNG sửa ActionDetection.py — chỉ import hàm mới này khi cần train
trên Data_normalized/.

Ví dụ dùng trong script training:
    from dataset_loader import load_dataset_normalized
    X, y, label_map = load_dataset_normalized(
        data_path='Data_normalized',
        sequence_length=60
    )
    # X.shape == (N, 60, 126)
    # y.shape == (N, num_classes)
"""

import os
import sys
import logging
import argparse
from typing import Optional

import numpy as np
from tensorflow.keras.utils import to_categorical

# Fix Unicode output on Windows terminals (cp1252 -> utf-8)
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(_sys.stderr, 'reconfigure'):
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')

log = logging.getLogger(__name__)


def load_dataset_normalized(
    data_path: str = 'Data_normalized',
    sequence_length: int = 60,
    actions: Optional[list] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Tải dữ liệu từ Data_normalized/, đọc mọi thư mục con (webcam_*, yt_*, v.v.)
    theo từng nhãn.

    Args:
        data_path       : thư mục gốc (mặc định 'Data_normalized')
        sequence_length : số frame mỗi sequence (mặc định 60)
        actions         : list nhãn cần load; None = tự động từ os.listdir
        verbose         : in log tiến trình

    Returns:
        X         : ndarray shape (N, sequence_length, 126)
        y         : ndarray shape (N, num_classes) — one-hot
        label_map : dict {label_str: class_index}

    Notes:
        - Sequence bị thiếu frame sẽ bị bỏ qua (không raise lỗi).
        - metadata.csv trong data_path bị bỏ qua khi liệt kê nhãn.
    """
    data_path = os.path.abspath(data_path)

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_path}")

    # ── Tự động phát hiện nhãn nếu không truyền vào ──────────────────────
    if actions is None:
        actions = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ])

    if not actions:
        raise ValueError(f"Không tìm thấy nhãn nào trong {data_path}")

    label_map = {label: idx for idx, label in enumerate(actions)}

    if verbose:
        print(f"=> Đang tải dữ liệu từ '{data_path}' ({len(actions)} nhãn)...")

    sequences, labels = [], []
    skipped_labels    = []
    skipped_seqs      = 0

    for label in actions:
        label_dir = os.path.join(data_path, label)
        if not os.path.isdir(label_dir):
            if verbose:
                log.warning(f"  Nhãn không có thư mục: '{label}' — bỏ qua")
            skipped_labels.append(label)
            continue

        # Liệt kê MỌI thư mục con (webcam_*, yt_*, tên số, ...)
        # Bỏ qua file (ví dụ metadata.csv sẽ không xuất hiện ở đây)
        seq_dirs = sorted([
            d for d in os.listdir(label_dir)
            if os.path.isdir(os.path.join(label_dir, d))
        ])

        for seq_name in seq_dirs:
            seq_dir = os.path.join(label_dir, seq_name)
            window  = []
            valid   = True

            for frame_idx in range(sequence_length):
                frame_path = os.path.join(seq_dir, f"{frame_idx}.npy")
                if os.path.exists(frame_path):
                    kp = np.load(frame_path)
                    window.append(kp)
                else:
                    valid = False
                    break

            if valid and len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[label])
            else:
                skipped_seqs += 1
                if verbose:
                    log.debug(f"  Bỏ qua sequence thiếu frame: {seq_dir}")

    if not sequences:
        raise ValueError(
            f"Không load được sequence nào từ '{data_path}'. "
            f"Hãy kiểm tra cấu trúc thư mục và sequence_length={sequence_length}."
        )

    X = np.array(sequences, dtype=np.float32)             # (N, 60, 126)
    y = to_categorical(labels, num_classes=len(actions))   # (N, num_classes)

    if verbose:
        print(f"=> Tải xong! X.shape={X.shape}, y.shape={y.shape}")
        if skipped_seqs:
            print(f"   (Bỏ qua {skipped_seqs} sequence thiếu frame)")
        if skipped_labels:
            print(f"   (Nhãn không có thư mục: {skipped_labels})")

    return X, y, label_map


def inspect_dataset(data_path: str = 'Data_normalized') -> None:
    """
    In thống kê chi tiết cấu trúc Data_normalized/:
    - Số sequence per nhãn
    - Phân loại theo nguồn (webcam_* / yt_*)
    """
    data_path = os.path.abspath(data_path)
    print(f"\n=== THỐNG KÊ DATASET: {data_path} ===\n")
    print(f"{'Nhãn':<40} {'webcam':>8} {'youtube':>8} {'khác':>6} {'tổng':>6}")
    print("-" * 70)

    total_webcam = total_yt = total_other = 0

    if not os.path.isdir(data_path):
        print(f"[Lỗi] Thư mục không tồn tại: {data_path}")
        return

    labels = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    for label in labels:
        label_dir = os.path.join(data_path, label)
        seq_dirs  = [d for d in os.listdir(label_dir)
                     if os.path.isdir(os.path.join(label_dir, d))]
        n_webcam  = sum(1 for d in seq_dirs if d.startswith('webcam_'))
        n_yt      = sum(1 for d in seq_dirs if d.startswith('yt_'))
        n_other   = len(seq_dirs) - n_webcam - n_yt
        print(f"  {label:<38} {n_webcam:>8} {n_yt:>8} {n_other:>6} {len(seq_dirs):>6}")
        total_webcam += n_webcam
        total_yt     += n_yt
        total_other  += n_other

    print("-" * 70)
    total = total_webcam + total_yt + total_other
    print(f"  {'TỔNG':<38} {total_webcam:>8} {total_yt:>8} {total_other:>6} {total:>6}")
    print()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Tải và kiểm tra dataset normalized cho FSign'
    )
    p.add_argument('--data_path', default='Data_normalized',
                   help='Thư mục dataset (mặc định: Data_normalized)')
    p.add_argument('--sequence_length', type=int, default=60)
    p.add_argument('--inspect_only', action='store_true',
                   help='Chỉ in thống kê cấu trúc, không load toàn bộ dữ liệu')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    if args.inspect_only:
        inspect_dataset(args.data_path)
    else:
        inspect_dataset(args.data_path)
        X, y, label_map = load_dataset_normalized(
            data_path=args.data_path,
            sequence_length=args.sequence_length,
        )
        print(f"\nX.shape = {X.shape}")
        print(f"y.shape = {y.shape}")
        print(f"Số nhãn = {len(label_map)}")
        print(f"Nhãn: {list(label_map.keys())[:5]}{'...' if len(label_map) > 5 else ''}")
