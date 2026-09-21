# -*- coding: utf-8 -*-
"""
normalize_existing_data.py — Batch Normalize Webcam Dataset
=============================================================
Đọc toàn bộ dữ liệu .npy RAW trong Data/ (từ webcam),
áp dụng normalize_keypoints() từ preprocessing.py,
và ghi kết quả ra Data_normalized/<label>/webcam_<id>/<frame>.npy.

KHÔNG sửa hoặc xóa Data/ gốc.
KHÔNG chạy lại MediaPipe — normalize thuần túy trên vector (126,) đã lưu.

Chạy 1 lần duy nhất trước khi bắt đầu thu thập YouTube.

Ví dụ:
    # Xem trước (không ghi file)
    python normalize_existing_data.py --src_path Data --dst_path Data_normalized --dry_run

    # Chạy thật
    python normalize_existing_data.py --src_path Data --dst_path Data_normalized

    # Cho phép ghi đè nếu đã chạy trước
    python normalize_existing_data.py --src_path Data --dst_path Data_normalized --overwrite
"""

import os
import sys
import csv
import argparse
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

# Fix Unicode output on Windows terminals (cp1252 -> utf-8)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Import từ module dùng chung ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from preprocessing import normalize_keypoints

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description='Normalize toàn bộ dataset webcam (Data/) → Data_normalized/'
    )
    p.add_argument('--src_path', default='Data',
                   help='Thư mục nguồn chứa dữ liệu raw (mặc định: Data)')
    p.add_argument('--dst_path', default='Data_normalized',
                   help='Thư mục đích chứa dữ liệu đã normalize (mặc định: Data_normalized)')
    p.add_argument('--dry_run', action='store_true',
                   help='Chỉ mô phỏng, không ghi file nào')
    p.add_argument('--overwrite', action='store_true',
                   help='Cho phép ghi đè nếu thư mục đích đã tồn tại')
    return p.parse_args()


def classify_frame(kp_126: np.ndarray) -> str:
    """Phân loại frame: both_hands / left_only / right_only / no_hands."""
    has_left  = not np.all(kp_126[:63] == 0)
    has_right = not np.all(kp_126[63:] == 0)
    if has_left and has_right:
        return 'both_hands'
    elif has_left:
        return 'left_only'
    elif has_right:
        return 'right_only'
    else:
        return 'no_hands'


def main():
    args = parse_args()

    # Resolve absolute paths
    src = os.path.abspath(args.src_path)
    dst = os.path.abspath(args.dst_path)

    if not os.path.isdir(src):
        log.error(f"Thư mục nguồn không tồn tại: {src}")
        sys.exit(1)

    log.info(f"Nguồn (raw)      : {src}")
    log.info(f"Đích (normalized): {dst}")
    log.info(f"Dry run          : {args.dry_run}")
    log.info(f"Overwrite        : {args.overwrite}")
    print()

    # ── Quét toàn bộ cấu trúc src ──────────────────────────────────────────
    labels = sorted([
        d for d in os.listdir(src)
        if os.path.isdir(os.path.join(src, d))
    ])

    if not labels:
        log.error(f"Không tìm thấy nhãn nào trong {src}")
        sys.exit(1)

    log.info(f"Tìm thấy {len(labels)} nhãn trong '{src}'")

    # ── Thống kê ──────────────────────────────────────────────────────────
    stats = defaultdict(int)   # both_hands / left_only / right_only / no_hands
    total_files   = 0
    skipped_exist = 0
    errors        = 0

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata_path = os.path.join(dst, 'metadata.csv')
    metadata_rows = []

    # ── Xử lý từng nhãn ───────────────────────────────────────────────────
    for label in labels:
        label_src_dir = os.path.join(src, label)
        label_dst_dir = os.path.join(dst, label)

        # Liệt kê sequence ID (các thư mục tên số nguyên 0,1,2,...)
        seq_ids = sorted([
            d for d in os.listdir(label_src_dir)
            if os.path.isdir(os.path.join(label_src_dir, d))
        ], key=lambda x: int(x) if x.isdigit() else -1)

        for seq_id in seq_ids:
            src_seq_dir = os.path.join(label_src_dir, seq_id)
            dst_seq_name = f"webcam_{seq_id}"
            dst_seq_dir  = os.path.join(label_dst_dir, dst_seq_name)

            # ── Collision guard ─────────────────────────────────────────
            if os.path.isdir(dst_seq_dir) and os.listdir(dst_seq_dir):
                if not args.overwrite:
                    raise FileExistsError(
                        f"\nThư mục đích đã tồn tại và có dữ liệu:\n  {dst_seq_dir}\n"
                        f"Dùng --overwrite để ghi đè, hoặc xóa thư mục đích trước."
                    )
                else:
                    log.warning(f"  Ghi đè: {dst_seq_dir}")

            # ── Liệt kê frame files ─────────────────────────────────────
            frame_files = sorted(
                [f for f in os.listdir(src_seq_dir) if f.endswith('.npy')],
                key=lambda x: int(x.replace('.npy', ''))
            )

            if not frame_files:
                log.warning(f"  Bỏ qua sequence rỗng: {src_seq_dir}")
                continue

            if not args.dry_run:
                os.makedirs(dst_seq_dir, exist_ok=True)

            for fname in frame_files:
                src_frame = os.path.join(src_seq_dir, fname)
                dst_frame = os.path.join(dst_seq_dir, fname)

                try:
                    raw = np.load(src_frame)

                    if raw.shape != (126,):
                        log.warning(f"  Shape không hợp lệ {raw.shape}: {src_frame}")
                        errors += 1
                        continue

                    norm = normalize_keypoints(raw)

                    # Sanity check — không bao giờ xảy ra nếu normalize đúng
                    if np.any(np.isnan(norm)) or np.any(np.isinf(norm)):
                        log.error(f"  NaN/Inf sau normalize: {src_frame}")
                        errors += 1
                        continue

                    frame_class = classify_frame(raw)
                    stats[frame_class] += 1
                    total_files += 1

                    if not args.dry_run:
                        np.save(dst_frame, norm)

                    # Metadata row
                    rel_dst = os.path.relpath(dst_frame, dst)
                    metadata_rows.append({
                        'file_path'    : rel_dst.replace('\\', '/'),
                        'label'        : label,
                        'source'       : 'webcam',
                        'url'          : '',
                        'original_fps' : '',
                        'valid_ratio'  : '',
                        'timestamp'    : datetime.now().isoformat(timespec='seconds'),
                    })

                except Exception as e:
                    log.error(f"  Lỗi khi xử lý {src_frame}: {e}")
                    errors += 1

    # ── Ghi metadata.csv ──────────────────────────────────────────────────
    if not args.dry_run and metadata_rows:
        os.makedirs(dst, exist_ok=True)
        file_exists = os.path.isfile(metadata_path)
        fieldnames = ['file_path', 'label', 'source', 'url',
                      'original_fps', 'valid_ratio', 'timestamp']
        mode = 'a' if file_exists else 'w'
        with open(metadata_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(metadata_rows)
        log.info(f"Đã ghi metadata: {metadata_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    prefix = "[DRY RUN] " if args.dry_run else ""
    print()
    print(f"{'='*55}")
    print(f"{prefix}KẾT QUẢ NORMALIZE")
    print(f"{'='*55}")
    print(f"  Tổng frame đã xử lý : {total_files:,}")
    print(f"  Cả 2 tay             : {stats['both_hands']:,}")
    print(f"  Chỉ tay trái         : {stats['left_only']:,}")
    print(f"  Chỉ tay phải         : {stats['right_only']:,}")
    print(f"  Không có tay         : {stats['no_hands']:,}")
    if skipped_exist:
        print(f"  Bỏ qua (đã tồn tại) : {skipped_exist:,}")
    if errors:
        print(f"  ⚠  Lỗi              : {errors:,}")
    print(f"{'='*55}")
    if args.dry_run:
        print("  → Dry run hoàn tất. Không có file nào được ghi.")
    else:
        print(f"  → Dữ liệu đã lưu tại: {dst}")
    print()


if __name__ == '__main__':
    main()
