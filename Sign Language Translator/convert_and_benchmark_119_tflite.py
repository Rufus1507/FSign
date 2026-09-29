# -*- coding: utf-8 -*-
"""
convert_and_benchmark_119_tflite.py — Convert Baseline 119 sang TFLite, Test Assert & Benchmark (Task MERGE-6)
================================================================================================================
Mục tiêu:
  1. Convert Models/baseline_119_tanh.h5 -> Models/baseline_119_tanh.tflite (chuẩn TFLite Builtin ops).
  2. Đánh giá độ đồng nhất (Parity Check) giữa H5 và TFLite trên tập test holdout.
  3. Đo latency suy luận thực tế (CPU 4 threads, 100 lần đo) xác định mean/min/max latency và throughput (FPS).
  4. Kiểm thử Negative Asserts: Chặn đứng mọi nguy cơ nạp nhầm giữa model 119 và model/nhãn 60.
"""

import os
import sys
import json
import time
from pathlib import Path

# Fix Unicode UTF-8 console output trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_def_merged119 import build_model


def log_print(msg="", log_file=None):
    print(msg, flush=True)
    if log_file:
        log_file.write(str(msg) + "\n")
        log_file.flush()


def run_benchmark():
    models_dir = SCRIPT_DIR / "Models"
    logs_dir = SCRIPT_DIR / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "benchmark_merge_6_tflite_119.log"
    log_f = open(log_file_path, "w", encoding="utf-8")

    log_print("=" * 80, log_f)
    log_print("  TASK MERGE-6: AUDIT LỆCH NHÃN & CONVERT TFLITE CHO MODEL BASELINE 119", log_f)
    log_print(f"  TensorFlow Version: {tf.__version__}", log_f)
    log_print(f"  Thời gian bắt đầu:  {time.strftime('%Y-%m-%d %H:%M:%S')}", log_f)
    log_print("=" * 80, log_f)

    # ── PHẦN 1: AUDIT TOÀN BỘ MODEL FILE TRONG MODELS/ ────────────────────────
    log_print("\n[PHẦN 1/4] AUDIT TOÀN BỘ MODEL FILE HIỆN CÓ TRONG MODELS/", log_f)
    log_print("-" * 80, log_f)

    models_inventory = [
        {
            "filename": "model_b3_v2_descending.h5",
            "expected_classes": 60,
            "expected_input_dim": 129,
            "label_file": "label_map.json (60 nhãn)",
            "pipeline": "Production 60 Nhãn (Webcam)"
        },
        {
            "filename": "model_b3_v2_descending.tflite",
            "expected_classes": 60,
            "expected_input_dim": 129,
            "label_file": "label_map.json (60 nhãn)",
            "pipeline": "Production 60 Nhãn (Webcam TFLite)"
        },
        {
            "filename": "fsign_159classes.h5",
            "expected_classes": 159,
            "expected_input_dim": 126,
            "label_file": "Models/label_map_159.json (159 nhãn)",
            "pipeline": "Thử nghiệm nhánh Phú (159 Nhãn cũ)"
        },
        {
            "filename": "baseline_119_tanh.h5",
            "expected_classes": 119,
            "expected_input_dim": 129,
            "label_file": "Models/label_map_119.json (119 nhãn)",
            "pipeline": "Baseline Hợp Nhất 119 Nhãn (MERGE-5)"
        }
    ]

    audit_results = []
    for item in models_inventory:
        m_path = models_dir / item["filename"]
        if not m_path.exists():
            log_print(f"  [BỎ QUA] Không tìm thấy file: {item['filename']}", log_f)
            continue

        file_size_mb = m_path.stat().st_size / (1024 * 1024)
        actual_input_dim = None
        actual_classes = None

        if m_path.suffix == ".h5":
            try:
                # Nạp Keras model
                k_model = tf.keras.models.load_model(str(m_path), compile=False)
                actual_input_dim = int(k_model.input_shape[-1])
                actual_classes = int(k_model.output_shape[-1])
            except Exception:
                # Fallback đọc trực tiếp metadata từ file HDF5 bằng h5py
                try:
                    import h5py
                    with h5py.File(str(m_path), 'r') as h5_f:
                        if 'model_config' in h5_f.attrs:
                            cfg = h5_f.attrs['model_config']
                            if isinstance(cfg, bytes):
                                cfg = cfg.decode('utf-8')
                            cfg_dict = json.loads(cfg)
                            layers = cfg_dict.get('config', {}).get('layers', [])
                            if layers:
                                # Input dimension
                                first_cfg = layers[0].get('config', {})
                                in_shape = first_cfg.get('batch_input_shape') or first_cfg.get('batch_shape')
                                if in_shape:
                                    actual_input_dim = int(in_shape[-1])
                                # Output classes
                                last_cfg = layers[-1].get('config', {})
                                if 'units' in last_cfg:
                                    actual_classes = int(last_cfg['units'])
                except Exception as h5_err:
                    log_print(f"  [LỖI NẠP H5] {item['filename']}: {h5_err}", log_f)
        elif m_path.suffix == ".tflite":
            try:
                interp = tf.lite.Interpreter(model_path=str(m_path))
                interp.allocate_tensors()
                actual_input_dim = int(interp.get_input_details()[0]['shape'][-1])
                actual_classes = int(interp.get_output_details()[0]['shape'][-1])
            except Exception as e:
                log_print(f"  [LỖI NẠP TFLITE] {item['filename']}: {e}", log_f)

        is_match = (actual_classes == item["expected_classes"]) and (actual_input_dim == item["expected_input_dim"])
        status_str = "KHỚP 100%" if is_match else "LỆCH NGUY HIỂM!"

        res_record = {
            "filename": item["filename"],
            "size_mb": round(file_size_mb, 2),
            "input_dim": actual_input_dim,
            "output_classes": actual_classes,
            "expected_classes": item["expected_classes"],
            "label_file": item["label_file"],
            "pipeline": item["pipeline"],
            "status": status_str
        }
        audit_results.append(res_record)

        dim_str = f"{actual_input_dim:3d}" if actual_input_dim is not None else "N/A"
        cls_str = f"{actual_classes:3d}" if actual_classes is not None else "N/A"

        log_print(
            f"  * {item['filename']:<30} | Size: {file_size_mb:4.2f}MB | "
            f"Dim: {dim_str} (Exp: {item['expected_input_dim']:3d}) | "
            f"Classes: {cls_str} (Exp: {item['expected_classes']:3d}) | "
            f"-> {status_str}",
            log_f
        )

    # ── PHẦN 2: CONVERT BASELINE_119_TANH.H5 SANG TFLITE ───────────────────────
    log_print("\n[PHẦN 2/4] CONVERT MODEL BASELINE 119 SANG TFLITE CHUẨN TFLITE_BUILTINS", log_f)
    log_print("-" * 80, log_f)

    h5_path = models_dir / "baseline_119_tanh.h5"
    tflite_path = models_dir / "baseline_119_tanh.tflite"

    if not h5_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model H5 tại {h5_path}")

    h5_size_mb = h5_path.stat().st_size / (1024 * 1024)
    log_print(f"  -> File nguồn Keras: {h5_path} ({h5_size_mb:.2f} MB)", log_f)

    # Nạp mô hình Keras
    keras_model = build_model(input_shape=(60, 129), num_classes=119)
    keras_model.load_weights(str(h5_path))

    # Cấu hình Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    t0_conv = time.perf_counter()
    pure_builtin = False
    try:
        log_print("  -> Đang chuyển đổi với target_spec=[TFLITE_BUILTINS]...", log_f)
        tflite_content = converter.convert()
        conv_time = time.perf_counter() - t0_conv
        pure_builtin = True
        log_print(f"  => Chuyển đổi THÀNH CÔNG trong {conv_time:.2f}s bằng 100% TFLITE_BUILTINS thuần túy!", log_f)
    except Exception as e:
        log_print(f"  [CẢNH BÁO] Không thể dùng thuần TFLite Builtins: {e}. Thử Select TF Ops...", log_f)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        t0_conv = time.perf_counter()
        tflite_content = converter.convert()
        conv_time = time.perf_counter() - t0_conv
        pure_builtin = False
        log_print(f"  => Chuyển đổi hoàn thành với Select TF Ops trong {conv_time:.2f}s", log_f)

    with open(tflite_path, "wb") as f:
        f.write(tflite_content)

    tflite_size_mb = tflite_path.stat().st_size / (1024 * 1024)
    comp_ratio = (1.0 - tflite_size_mb / h5_size_mb) * 100.0

    log_print(f"  => Đã lưu file TFLite tại: {tflite_path}", log_f)
    log_print(f"     Dung lượng H5:     {h5_size_mb:.2f} MB", log_f)
    log_print(f"     Dung lượng TFLite: {tflite_size_mb:.2f} MB", log_f)
    log_print(f"     Tỷ lệ nén giảm:    {comp_ratio:.1f}% dung lượng", log_f)

    # ── PHẦN 3: PARITY CHECK (ĐỐI ĐẦU H5 VS TFLITE) & BENCHMARK HIỆU NĂNG ─────
    log_print("\n[PHẦN 3/4] ĐỐI ĐẦU CHÍNH XÁC (PARITY CHECK) & BENCHMARK TFLITE (CPU 4 THREADS)", log_f)
    log_print("-" * 80, log_f)

    # Nạp tập test từ cache
    cache_path = SCRIPT_DIR / "Data_normalized_merged119" / "dataset_cache_119.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cache dữ liệu tại {cache_path}")

    c = np.load(str(cache_path), allow_pickle=True)
    X_all = c['X']
    y_all = c['y']
    y_indices = np.argmax(y_all, axis=1)

    _, X_test, _, y_test = train_test_split(
        X_all, y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_indices
    )
    y_test_idx = np.argmax(y_test, axis=1)
    num_test_samples = len(X_test)
    log_print(f"  -> Nạp tập Holdout Test: {num_test_samples} mẫu (chuẩn Stratified 80/20)", log_f)

    # Khởi tạo TFLite Interpreter với 4 threads
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    # 1. Đo Parity Check (so sánh kết quả H5 vs TFLite trên toàn bộ 1,368 mẫu)
    log_print("  -> Đang kiểm tra độ sai lệch phân bố xác suất giữa H5 và TFLite...", log_f)
    keras_preds = keras_model.predict(X_test, batch_size=32, verbose=0)
    keras_pred_idx = np.argmax(keras_preds, axis=1)
    keras_acc = accuracy_score(y_test_idx, keras_pred_idx)

    tflite_preds = []
    for i in range(num_test_samples):
        sample = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(in_idx, sample)
        interpreter.invoke()
        out = interpreter.get_tensor(out_idx)
        tflite_preds.append(out[0])

    tflite_preds = np.array(tflite_preds)
    tflite_pred_idx = np.argmax(tflite_preds, axis=1)
    tflite_acc = accuracy_score(y_test_idx, tflite_pred_idx)

    max_diff = np.max(np.abs(keras_preds - tflite_preds))
    mean_diff = np.mean(np.abs(keras_preds - tflite_preds))
    pred_match_rate = np.mean(keras_pred_idx == tflite_pred_idx) * 100.0

    log_print(f"     H5 Accuracy:               {keras_acc * 100:.2f}%", log_f)
    log_print(f"     TFLite Accuracy:           {tflite_acc * 100:.2f}%", log_f)
    log_print(f"     Tỷ lệ khớp nhãn dự đoán:  {pred_match_rate:.2f}% ({np.sum(keras_pred_idx == tflite_pred_idx)}/{num_test_samples})", log_f)
    log_print(f"     Độ lệch xác suất lớn nhất (Max Diff):  {max_diff:.8f}", log_f)
    log_print(f"     Độ lệch xác suất trung bình (Mean Diff): {mean_diff:.8f}", log_f)

    # 2. Đo Latency & FPS (100 lần đo trên CPU 4 threads)
    log_print("\n  -> Bắt đầu benchmark thời gian suy luận (100 lần lặp trên CPU 4 Threads)...", log_f)
    warmup_sample = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    # Warmup 20 lần
    for _ in range(20):
        interpreter.set_tensor(in_idx, warmup_sample)
        interpreter.invoke()

    latencies_ms = []
    num_runs = 100
    for r in range(num_runs):
        idx = r % num_test_samples
        sample = np.expand_dims(X_test[idx], axis=0).astype(np.float32)

        t_start = time.perf_counter()
        interpreter.set_tensor(in_idx, sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
        t_end = time.perf_counter()

        latencies_ms.append((t_end - t_start) * 1000.0)

    mean_lat = np.mean(latencies_ms)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    median_lat = np.median(latencies_ms)
    std_lat = np.std(latencies_ms)
    throughput_fps = 1000.0 / mean_lat

    log_print(f"     Số lần đo:                 {num_runs} runs", log_f)
    log_print(f"     Độ trễ trung bình (Mean):  {mean_lat:6.2f} ms", log_f)
    log_print(f"     Độ trễ trung vị (Median):  {median_lat:6.2f} ms", log_f)
    log_print(f"     Độ trễ nhanh nhất (Min):   {min_lat:6.2f} ms", log_f)
    log_print(f"     Độ trễ chậm nhất (Max):    {max_lat:6.2f} ms", log_f)
    log_print(f"     Độ lệch chuẩn (Std Dev):   {std_lat:6.2f} ms", log_f)
    log_print(f"     Tốc độ suy luận trần:      {throughput_fps:6.1f} FPS", log_f)

    # ── PHẦN 4: THỬ NGHIỆM ASSERT CHẶN CỨNG (NEGATIVE & POSITIVE TESTS) ────────
    log_print("\n[PHẦN 4/4] THỬ NGHIỆM CHỐT CHẶN ASSERT KIỂM TRA CHÉO (SAFETY GUARDS)", log_f)
    log_print("-" * 80, log_f)

    # Cấu hình thử nghiệm
    with open(models_dir / "label_map_119.json", "r", encoding="utf-8") as f:
        lmap_119 = json.load(f)["classes"]

    with open(SCRIPT_DIR / "label_map.json", "r", encoding="utf-8") as f:
        lmap_60 = json.load(f)["classes"]

    model_119_interp = interpreter
    model_60_path = models_dir / "model_b3_v2_descending.tflite"
    model_60_interp = tf.lite.Interpreter(model_path=str(model_60_path))
    model_60_interp.allocate_tensors()

    def verify_pipeline_wiring(interp_obj, actions_list, expected_dim=129, expected_num=119):
        in_dim = int(interp_obj.get_input_details()[0]['shape'][-1])
        out_num = int(interp_obj.get_output_details()[0]['shape'][-1])

        assert in_dim == expected_dim, (
            f"Input feature dimension mismatch: expected {expected_dim} features, but model expects {in_dim}"
        )
        assert out_num == expected_num, (
            f"Model output classes mismatch: expected {expected_num} classes, but model has {out_num}"
        )
        assert len(actions_list) == expected_num, (
            f"Label count mismatch: expected {expected_num} labels, but found {len(actions_list)}"
        )
        return True

    # Test 1: Positive Test (Model 119 + Nhãn 119)
    test1_passed = False
    try:
        verify_pipeline_wiring(model_119_interp, lmap_119, expected_dim=129, expected_num=119)
        test1_passed = True
        log_print("  [TEST 1 - POSITIVE] Model 119 + Nhãn 119 (129 dim):   VƯỢT QUA 100% (PASS)", log_f)
    except AssertionError as e:
        log_print(f"  [TEST 1] THẤT BẠI BẤT THƯỜNG: {e}", log_f)

    # Test 2: Negative Test (Model 119 + Nhãn 60 cũ) -> PHẢI CHẶN ĐỨNG
    test2_blocked = False
    try:
        verify_pipeline_wiring(model_119_interp, lmap_60, expected_dim=129, expected_num=119)
        log_print("  [TEST 2 - NEGATIVE] NGUY HIỂM! Không chặn được lệch 60 vs 119!", log_f)
    except AssertionError as e:
        test2_blocked = True
        log_print(f"  [TEST 2 - NEGATIVE] Model 119 + Nhãn 60:              ĐÃ CHẶN ĐỨNG THÀNH CÔNG -> '{e}'", log_f)

    # Test 3: Negative Test (Model 60 cũ + Nhãn 119) -> PHẢI CHẶN ĐỨNG
    test3_blocked = False
    try:
        verify_pipeline_wiring(model_60_interp, lmap_119, expected_dim=129, expected_num=119)
        log_print("  [TEST 3 - NEGATIVE] NGUY HIỂM! Không chặn được Model 60 nạp vào pipeline 119!", log_f)
    except AssertionError as e:
        test3_blocked = True
        log_print(f"  [TEST 3 - NEGATIVE] Model 60 cũ + Nhãn 119:           ĐÃ CHẶN ĐỨNG THÀNH CÔNG -> '{e}'", log_f)

    # Test 4: Negative Test (Sai chiều đặc trưng, ví dụ 126 dim cũ) -> PHẢI CHẶN ĐỨNG
    test4_blocked = False
    try:
        # Giả lập pipeline 126 dim nạp vào model 129
        verify_pipeline_wiring(model_119_interp, lmap_119, expected_dim=126, expected_num=119)
    except AssertionError as e:
        test4_blocked = True
        log_print(f"  [TEST 4 - NEGATIVE] Sai chiều đặc trưng (126 vs 129): ĐÃ CHẶN ĐỨNG THÀNH CÔNG -> '{e}'", log_f)

    # Lưu toàn bộ báo cáo JSON
    benchmark_json_path = logs_dir / "benchmark_merge_6_tflite_119.json"
    report_dict = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "audit_models": audit_results,
        "conversion": {
            "source_h5": str(h5_path),
            "output_tflite": str(tflite_path),
            "h5_size_mb": round(h5_size_mb, 2),
            "tflite_size_mb": round(tflite_size_mb, 2),
            "compression_ratio_pct": round(comp_ratio, 1),
            "pure_tflite_builtin": pure_builtin,
            "convert_duration_s": round(conv_time, 2)
        },
        "parity_check": {
            "test_samples": num_test_samples,
            "h5_accuracy": round(keras_acc, 4),
            "tflite_accuracy": round(tflite_acc, 4),
            "prediction_match_rate_pct": round(pred_match_rate, 2),
            "max_abs_diff": float(max_diff),
            "mean_abs_diff": float(mean_diff)
        },
        "latency_benchmark_cpu_4_threads": {
            "num_runs": num_runs,
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(median_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "std_latency_ms": round(std_lat, 2),
            "throughput_fps": round(throughput_fps, 1)
        },
        "assert_tests": {
            "test1_positive_pass": test1_passed,
            "test2_mismatch_label_blocked": test2_blocked,
            "test3_mismatch_model_blocked": test3_blocked,
            "test4_mismatch_input_dim_blocked": test4_blocked
        }
    }

    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    log_print(f"\n=> Đã lưu toàn bộ số liệu benchmark vào: {benchmark_json_path}", log_f)
    log_print("=" * 80, log_f)
    log_f.close()
    return report_dict


if __name__ == "__main__":
    run_benchmark()
