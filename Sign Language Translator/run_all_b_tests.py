# -*- coding: utf-8 -*-
"""
run_all_b_tests.py — Chạy lại toàn bộ 5 test cases Giai đoạn B (B1-B5) trên TFLite Engine
========================================================================================
Mục tiêu:
  Xác nhận 100% logic nghiệp vụ GĐ B (chuẩn hóa per-hand, Idle 0.5s, consensus 10 frames,
  cooldown 1.2s, chống rò rỉ frame zero khi mất dấu tay) vẫn hoạt động CHÍNH XÁC TUYỆT ĐỐI
  sau khi RunModel.py được chuyển sang TFLite Engine (Models/model_normalized_v1.tflite).
"""
import os
import sys
import subprocess
import time

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

tests = [
    ("Mục B1 (Pipeline & 61 sequences)", "test_b1_pipeline.py"),
    ("Mục B2 (Idle Rule-based 0.5s)", "test_b2_idle.py"),
    ("Mục B3 (Consensus Check 10 frames)", "test_b3_consensus.py"),
    ("Mục B4 (Cooldown Realtime 1.2s)", "test_b4_cooldown.py"),
    ("Mục B5 (Chống rò rỉ frame zero khi dropout)", "test_b5_dropout.py"),
]

log_file_path = "test_b_all_results.log"
with open(log_file_path, "w", encoding="utf-8") as f:
    header = (
        "================================================================================\n"
        "=== KIỂM ĐỊNH TOÀN BỘ LOGIC GĐ B (B1 - B5) TRÊN TFLITE ENGINE ===\n"
        f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "================================================================================\n"
    )
    print(header)
    f.write(header)

    all_passed = True
    summary = []

    for name, script in tests:
        sep = f"\n{'=' * 80}\n>>> ĐANG CHẠY: {name} ({script})\n{'=' * 80}\n"
        print(sep)
        f.write(sep)
        f.flush()

        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - t0

        print(proc.stdout)
        f.write(proc.stdout)
        if proc.stderr:
            print("[STDERR]:", proc.stderr)
            f.write("\n[STDERR]:\n" + proc.stderr)

        f.flush()

        if proc.returncode == 0:
            status = f"[PASS] {name} thành công ({elapsed:.2f}s)"
        else:
            status = f"[FAIL] {name} THẤT BẠI (exit code {proc.returncode}, {elapsed:.2f}s)"
            all_passed = False

        print(status)
        f.write(status + "\n")
        summary.append(status)

    footer = (
        f"\n{'=' * 80}\n"
        "=== TỔNG KẾT TOÀN BỘ 5 BỘ TEST B1 - B5 TRÊN TFLITE ENGINE ===\n" +
        "\n".join(summary) + "\n" +
        f"{'=' * 80}\n"
        f"KẾT QUẢ CHUNG CUỘC: {'TẤT CẢ TEST ĐỀU PASS 100%!' if all_passed else 'CÓ TEST BỊ LỖI!'}\n"
        f"Chi tiết log đã ghi tại: {log_file_path}\n"
        f"{'=' * 80}\n"
    )
    print(footer)
    f.write(footer)
