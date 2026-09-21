# -*- coding: utf-8 -*-
"""
normalize_fast.py -- Phien ban nhanh cua normalize_existing_data.py
Xu ly batch ca sequence (60 frames) mot lan thay vi tung file
Nhanh hon ~10-15x so voi ban goc.

Chay:
    python normalize_fast.py --src_path Data --dst_path Data_normalized
    python normalize_fast.py --src_path Data --dst_path Data_normalized --overwrite
"""

import os, sys, csv, argparse
from datetime import datetime
from collections import defaultdict

import numpy as np

# Fix Unicode tren Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from preprocessing import normalize_keypoints


def parse_args():
    p = argparse.ArgumentParser(description='Normalize webcam data (fast batch version)')
    p.add_argument('--src_path',  default='Data',            help='Thu muc nguon raw')
    p.add_argument('--dst_path',  default='Data_normalized', help='Thu muc dich normalized')
    p.add_argument('--overwrite', action='store_true',       help='Ghi de neu da ton tai')
    p.add_argument('--dry_run',   action='store_true',       help='Mo phong, khong ghi file')
    return p.parse_args()


def main():
    args = parse_args()
    src = os.path.abspath(args.src_path)
    dst = os.path.abspath(args.dst_path)

    if not os.path.isdir(src):
        print(f'[ERROR] Thu muc nguon khong ton tai: {src}')
        sys.exit(1)

    labels = sorted([d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))])
    print(f'Nguon : {src}')
    print(f'Dich  : {dst}')
    print(f'Nhan  : {len(labels)} nhan')
    print(f'Mode  : {"DRY RUN" if args.dry_run else "REAL"} | overwrite={args.overwrite}')
    print()

    total_files = 0
    errors      = 0
    stats       = defaultdict(int)
    metadata_rows = []
    metadata_path = os.path.join(dst, 'metadata.csv')

    for label_idx, label in enumerate(labels, 1):
        label_src = os.path.join(src, label)
        label_dst = os.path.join(dst, label)

        seq_ids = sorted(
            [d for d in os.listdir(label_src) if os.path.isdir(os.path.join(label_src, d))],
            key=lambda x: int(x) if x.isdigit() else 9999
        )

        label_files = 0
        for seq_id in seq_ids:
            src_seq = os.path.join(label_src, seq_id)
            dst_seq = os.path.join(label_dst, f'webcam_{seq_id}')

            # Collision guard
            if os.path.isdir(dst_seq) and os.listdir(dst_seq) and not args.overwrite:
                print(f'  [SKIP] Da ton tai: {dst_seq}  (them --overwrite de ghi de)')
                continue

            # Load tat ca frames trong 1 lan
            frame_files = sorted(
                [f for f in os.listdir(src_seq) if f.endswith('.npy')],
                key=lambda x: int(x.replace('.npy', ''))
            )
            if not frame_files:
                continue

            try:
                raw_batch  = np.stack([np.load(os.path.join(src_seq, f)) for f in frame_files])
                norm_batch = np.stack([normalize_keypoints(raw_batch[i]) for i in range(len(raw_batch))])
            except Exception as e:
                print(f'  [ERROR] {src_seq}: {e}')
                errors += 1
                continue

            # Phan loai frame
            for i in range(len(raw_batch)):
                has_l = not np.all(raw_batch[i, :63] == 0)
                has_r = not np.all(raw_batch[i, 63:] == 0)
                if has_l and has_r:   stats['both'] += 1
                elif has_l:           stats['left_only'] += 1
                elif has_r:           stats['right_only'] += 1
                else:                 stats['no_hands'] += 1

            label_files += len(frame_files)
            total_files += len(frame_files)

            if not args.dry_run:
                os.makedirs(dst_seq, exist_ok=True)
                for fname, norm_frame in zip(frame_files, norm_batch):
                    np.save(os.path.join(dst_seq, fname), norm_frame)
                metadata_rows.append({
                    'file_path'    : f'{label}/webcam_{seq_id}/0.npy',
                    'label'        : label,
                    'source'       : 'webcam',
                    'url'          : '',
                    'original_fps' : '',
                    'valid_ratio'  : '',
                    'timestamp'    : datetime.now().isoformat(timespec='seconds'),
                })

        print(f'  [{label_idx:2d}/{len(labels)}] {label}: {label_files} frames  OK', flush=True)

    # Ghi metadata
    if not args.dry_run and metadata_rows:
        os.makedirs(dst, exist_ok=True)
        file_exists = os.path.isfile(metadata_path)
        with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['file_path','label','source','url','original_fps','valid_ratio','timestamp']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerows(metadata_rows)

    print()
    print('=' * 55)
    print('KET QUA NORMALIZE')
    print('=' * 55)
    print(f'  Tong frame xu ly : {total_files:,}')
    print(f'  Ca 2 tay         : {stats["both"]:,}')
    print(f'  Chi tay trai     : {stats["left_only"]:,}')
    print(f'  Chi tay phai     : {stats["right_only"]:,}')
    print(f'  Khong co tay     : {stats["no_hands"]:,}')
    if errors:
        print(f'  Loi              : {errors:,}')
    if args.dry_run:
        print('  -> DRY RUN: khong ghi file nao')
    else:
        print(f'  -> Da luu tai: {dst}')
    print('=' * 55)


if __name__ == '__main__':
    main()
