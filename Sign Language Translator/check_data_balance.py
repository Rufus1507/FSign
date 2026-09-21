# -*- coding: utf-8 -*-
"""
check_data_balance.py — Data Balance Checker for FSign
=======================================================
Thống kê số mẫu từng nguồn (webcam / youtube) cho mỗi nhãn trong
Data_normalized/, cảnh báo khi tỷ lệ lệch quá xa mục tiêu.

Ví dụ:
    python check_data_balance.py

    python check_data_balance.py \\
        --data_path Data_normalized \\
        --metadata Data_normalized/metadata.csv \\
        --target_youtube_ratio 0.7 \\
        --warn_threshold 0.15
"""

import os
import sys
import csv
import argparse
import logging
from collections import defaultdict

log = logging.getLogger(__name__)

# Fix Unicode output on Windows terminals (cp1252 -> utf-8)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Thống kê cân bằng nguồn dữ liệu webcam/youtube theo nhãn'
    )
    p.add_argument('--data_path', default='Data_normalized',
                   help='Thư mục dataset (mặc định: Data_normalized)')
    p.add_argument('--metadata', default=None,
                   help='Đường dẫn metadata.csv (mặc định: <data_path>/metadata.csv). '
                        'Nếu không tồn tại sẽ suy từ tên thư mục con.')
    p.add_argument('--target_youtube_ratio', type=float, default=0.7,
                   help='Tỷ lệ YouTube mục tiêu 0.0–1.0 (mặc định: 0.7 = 70%%)')
    p.add_argument('--warn_threshold', type=float, default=0.15,
                   help='Ngưỡng lệch tỷ lệ để cảnh báo (mặc định: 0.15 = ±15%%)')
    return p.parse_args()


# ─── Count from directory structure ──────────────────────────────────────────

def count_from_dirs(data_path: str) -> dict[str, dict[str, int]]:
    """
    Đếm sequences theo nguồn bằng cách đọc tên thư mục con:
      webcam_*  → webcam
      yt_*      → youtube
      khác      → other

    Returns:
        {label: {'webcam': N, 'youtube': M, 'other': K}}
    """
    result = {}
    if not os.path.isdir(data_path):
        return result

    for label in sorted(os.listdir(data_path)):
        label_dir = os.path.join(data_path, label)
        if not os.path.isdir(label_dir):
            continue  # bỏ qua file (metadata.csv, ...)
        counts = defaultdict(int)
        for seq in os.listdir(label_dir):
            if not os.path.isdir(os.path.join(label_dir, seq)):
                continue
            if seq.startswith('webcam_'):
                counts['webcam'] += 1
            elif seq.startswith('yt_'):
                counts['youtube'] += 1
            else:
                counts['other'] += 1
        result[label] = dict(counts)

    return result


def count_from_metadata(metadata_path: str) -> dict[str, dict[str, int]]:
    """
    Đếm samples từ metadata.csv theo cột 'source'.

    Returns:
        {label: {'webcam': N, 'youtube': M, ...}}
    """
    result = defaultdict(lambda: defaultdict(int))
    if not os.path.isfile(metadata_path):
        return {}

    with open(metadata_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label  = row.get('label', '').strip()
            source = row.get('source', 'unknown').strip()
            if label:
                result[label][source] += 1

    return {k: dict(v) for k, v in result.items()}


# ─── Report ───────────────────────────────────────────────────────────────────

def print_balance_report(
    counts: dict[str, dict[str, int]],
    target_yt_ratio: float,
    warn_threshold: float,
) -> None:
    """In bảng thống kê và cảnh báo."""

    col_label   = 42
    col_num     = 8
    col_pct     = 10
    col_status  = 12

    header = (
        f"{'Nhãn':<{col_label}}"
        f"{'Tổng':>{col_num}}"
        f"{'Webcam':>{col_num}}"
        f"{'YouTube':>{col_num}}"
        f"{'% YouTube':>{col_pct}}"
        f"{'Trạng thái':>{col_status}}"
    )
    sep = "─" * (col_label + col_num * 3 + col_pct + col_status)

    print()
    print("=== THỐNG KÊ CÂN BẰNG DỮ LIỆU ===")
    print(f"Mục tiêu YouTube: {target_yt_ratio:.0%}  |  "
          f"Ngưỡng cảnh báo: ±{warn_threshold:.0%}")
    print()
    print(header)
    print(sep)

    grand_total = grand_webcam = grand_yt = 0
    warned_labels = []

    for label in sorted(counts.keys()):
        src  = counts[label]
        n_wc = src.get('webcam', 0)
        n_yt = src.get('youtube', 0)
        n_ot = src.get('other', 0)
        total = n_wc + n_yt + n_ot

        pct_yt = n_yt / total if total > 0 else 0.0
        deviation = abs(pct_yt - target_yt_ratio)

        if total == 0:
            status = "⚠  RỖNG"
        elif deviation > warn_threshold:
            status = "⚠  LỆCH"
            warned_labels.append(label)
        else:
            status = "✓  OK"

        print(
            f"  {label:<{col_label - 2}}"
            f"{total:>{col_num}}"
            f"{n_wc:>{col_num}}"
            f"{n_yt:>{col_num}}"
            f"{pct_yt:>{col_pct - 1}.1%} "
            f"{status:>{col_status}}"
        )

        grand_total   += total
        grand_webcam  += n_wc
        grand_yt      += n_yt

    print(sep)
    grand_pct = grand_yt / grand_total if grand_total > 0 else 0.0
    print(
        f"  {'TỔNG CỘNG':<{col_label - 2}}"
        f"{grand_total:>{col_num}}"
        f"{grand_webcam:>{col_num}}"
        f"{grand_yt:>{col_num}}"
        f"{grand_pct:>{col_pct - 1}.1%} "
        f"{'':>{col_status}}"
    )
    print()

    if warned_labels:
        print(f"⚠  CẢNH BÁO: {len(warned_labels)} nhãn lệch tỷ lệ YouTube > "
              f"±{warn_threshold:.0%} so với mục tiêu {target_yt_ratio:.0%}:")
        for lb in warned_labels:
            src   = counts[lb]
            total = sum(src.values())
            pct   = src.get('youtube', 0) / total if total > 0 else 0
            print(f"   - {lb}: {pct:.1%} YouTube "
                  f"(webcam={src.get('webcam', 0)}, youtube={src.get('youtube', 0)})")
        print()
        print("Gợi ý: thu thập thêm dữ liệu YouTube cho các nhãn thiếu,")
        print("       hoặc điều chỉnh --target_youtube_ratio cho phù hợp.")
    else:
        print(f"✓  Tất cả nhãn đều nằm trong ngưỡng ±{warn_threshold:.0%}.")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    data_path     = os.path.abspath(args.data_path)
    metadata_path = args.metadata or os.path.join(data_path, 'metadata.csv')

    # Ưu tiên metadata.csv nếu tồn tại (nguồn chính xác hơn)
    if os.path.isfile(metadata_path):
        log.info(f"Đọc từ metadata: {metadata_path}")
        counts = count_from_metadata(metadata_path)
        if not counts:
            log.warning("metadata.csv rỗng hoặc không đọc được — dùng tên thư mục")
            counts = count_from_dirs(data_path)
    else:
        log.info(f"Không tìm thấy metadata.csv — suy từ tên thư mục: {data_path}")
        counts = count_from_dirs(data_path)

    if not counts:
        log.error(f"Không có dữ liệu để thống kê trong {data_path}")
        sys.exit(1)

    print_balance_report(
        counts          = counts,
        target_yt_ratio = args.target_youtube_ratio,
        warn_threshold  = args.warn_threshold,
    )


if __name__ == '__main__':
    main()
