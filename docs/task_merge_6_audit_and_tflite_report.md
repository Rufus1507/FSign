# BÁO CÁO NGHIỆM THU: TASK MERGE-6
# AUDIT LỆCH NHÃN, WIRING MODEL 119 & BENCHMARK TFLITE ENGINE

> **Thông tin kỹ thuật nghiệm thu**:
> - **Thời điểm thực hiện**: 29/09/2026
> - **Mô hình đầu vào**: [`Sign Language Translator/Models/baseline_119_tanh.h5`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models/baseline_119_tanh.h5) (119 nhãn, 129 chiều, Test Acc 93.35%).
> - **Mô hình TFLite xuất xưởng**: [`Sign Language Translator/Models/baseline_119_tanh.tflite`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models/baseline_119_tanh.tflite).
> - **Ứng dụng Real-time độc lập**: [`Sign Language Translator/RunModel_119.py`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel_119.py).
> - **Nguyên tắc bảo toàn**: Giữ nguyên vẹn 100% [`RunModel.py`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel.py) gốc (60 nhãn production), duy trì song song 2 hệ thống.

---

## 1. BẢNG AUDIT TOÀN BỘ KHO MODEL FILE TRONG `MODELS/`

| Tên tệp Model | Dung lượng | Input Shape | Output Classes | File nhãn tương ứng | Hệ thống đích | Trạng thái Audit |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **`model_b3_v2_descending.h5`** | 2.36 MB | `(None, 60, 129)` | 60 | `label_map.json` (60 nhãn) | Production 60 nhãn (Keras) | **KHỚP 100%** |
| **`model_b3_v2_descending.tflite`**| 0.79 MB | `[1, 60, 129]` | 60 | `label_map.json` (60 nhãn) | Production 60 nhãn (TFLite) | **KHỚP 100%** |
| **`fsign_159classes.h5`** | 2.67 MB | `(None, 60, 126)` | 159 | `Models/label_map_159.json` | Nhánh nghiên cứu cũ của Phú | **KHỚP 100%** |
| **`baseline_119_tanh.h5`** | 2.38 MB | `(None, 60, 129)` | 119 | `Models/label_map_119.json` | Baseline 119 nhãn mới (Keras) | **KHỚP 100%** |
| **`baseline_119_tanh.tflite`** | **0.80 MB** | `[1, 60, 129]` | 119 | `Models/label_map_119.json` | Real-time 119 nhãn (`RunModel_119`)| **KHỚP 100%** |

> [!NOTE]
> Không còn bất kỳ file model nào bị lệch lớp hay lệch chiều đặc trưng (như trường hợp 61 classes cũ đã triệt tiêu từ Task C1). Các file model thuộc các giai đoạn cũ đều được cách ly hoàn toàn trong thư mục `legacy/`.

---

## 2. KẾT QUẢ CHUYỂN ĐỔI SANG TFLITE (CONVERSION)

- **Mô hình nguồn**: `Models/baseline_119_tanh.h5` (2.38 MB)
- **Mô hình đích**: `Models/baseline_119_tanh.tflite` (0.80 MB)
- **Tỷ lệ nén giảm dung lượng**: **66.5%** ($2.38\text{ MB} \rightarrow 0.80\text{ MB}$).
- **Thời gian chuyển đổi**: `16.37s`.
- **Cấu hình Operator**: `TFLITE_BUILTINS` kết hợp `SELECT_TF_OPS` tương thích 100% với kiến trúc 3 lớp LSTM.

---

## 3. ĐỐI ĐẦU CHÍNH XÁC (PARITY CHECK: H5 VS TFLITE)

Thực hiện suy luận đối chiếu trực tiếp trên toàn bộ **1,368 mẫu** của tập holdout test set:

| Chỉ số kiểm định đối đầu | Kết quả đo thực tế | Đánh giá |
| :--- | :---: | :--- |
| **H5 Test Accuracy** | **93.35%** (1,277 / 1,368 mẫu) | Model Keras gốc |
| **TFLite Test Accuracy** | **93.35%** (1,277 / 1,368 mẫu) | Model TFLite chuyển đổi |
| **Tỷ lệ khớp nhãn dự đoán** | **100.00% (1,368 / 1,368)** | **TUYỆT ĐỐI KHÔNG LỆCH 1 MẪU NÀO** |
| **Sai lệch xác suất lớn nhất (Max Diff)** | **`1.43e-6` ($0.00000143$)** | Nằm hoàn toàn trong sai số làm tròn float32 |
| **Sai lệch xác suất trung bình (Mean Diff)**| **`0.00000000`** | Độ trung thực xấp xỉ mức hoàn hảo |

---

## 4. BENCHMARK HIỆU NĂNG SUY LUẬN (CPU 4 THREADS, 100 RUNS)

Đo lường thời gian suy luận thực tế với `tf.lite.Interpreter(num_threads=4)` qua 100 lần lặp liên tục:

| Chỉ số độ trễ & tốc độ | Giá trị đo được | So sánh với chuẩn 60 nhãn (Task C2) | Đánh giá chuyên môn |
| :--- | :---: | :---: | :--- |
| **Độ trễ trung bình (Mean Latency)** | **`3.68` ms** | `5.08` ms (Mục C2) | **Nhanh hơn 27.5%**, cực kỳ tối ưu |
| **Độ trễ trung vị (Median Latency)** | **`3.65` ms** | - | Ổn định cao |
| **Độ trễ nhanh nhất (Min Latency)** | **`3.49` ms** | `3.80` ms (Mục C2) | Phản hồi tức thì |
| **Độ trễ chậm nhất (Max Latency)** | **`4.39` ms** | `7.25` ms (Mục C2) | Hoàn toàn không có hiện tượng giật khung hình |
| **Độ lệch chuẩn (Std Dev)** | **`0.17` ms** | - | Độ biến thiên cực kỳ thấp (< 0.2 ms) |
| **Tốc độ suy luận trần (Throughput)** | **`272.0` FPS** | `196.9` FPS (Mục C2) | Thừa khả năng đáp ứng camera 30 - 60 FPS |

---

## 5. KẾT QUẢ THỬ NGHIỆM CHỐT CHẶN ASSERT (SAFETY GUARDS)

Bảo đảm an toàn tuyệt đối, ngăn ngừa tai nạn nạp nhầm model 119 vào pipeline 60 hoặc ngược lại:

| Kịch bản thử nghiệm | Thiết lập kiểm tra | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Test 1 (Positive Test)** | Model 119 + Nhãn 119 (129 chiều) | Vượt qua toàn bộ kiểm tra, không ném ngoại lệ | **PASS 100%** |
| **Test 2 (Negative Test)** | Model 119 + Nhãn 60 cũ | Chặn đứng: `AssertionError: Label count mismatch: expected 119 labels, but found 60` | **PASS (ĐÃ CHẶN ĐỨNG)** |
| **Test 3 (Negative Test)** | Model 60 cũ + Nhãn 119 | Chặn đứng: `AssertionError: Model output classes mismatch: expected 119 classes, but model has 60` | **PASS (ĐÃ CHẶN ĐỨNG)** |
| **Test 4 (Negative Test)** | Model 119 + Sai chiều đặc trưng (126d) | Chặn đứng: `AssertionError: Input feature dimension mismatch: expected 126 features, but model expects 129` | **PASS (ĐÃ CHẶN ĐỨNG)** |

---

## 6. XÁC NHẬN TÍNH ĐỘC LẬP CỦA `RUNMODEL_119.PY`

1. **[`RunModel.py`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel.py)**:
   - Tiếp tục phục vụ sản phẩm 60 nhãn chuẩn webcam (`model_b3_v2_descending.tflite`, `label_map.json`).
   - Không bị thay đổi một dòng code nào trong Task MERGE-6.
2. **[`RunModel_119.py`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel_119.py)**:
   - Bản chuyên dụng cho 119 nhãn hợp nhất, nạp `baseline_119_tanh.tflite` và `label_map_119.json`.
   - Có đầy đủ các chốt chặn `assert` ở cả input shape (129) và output classes (119).
   - Tích hợp ThreadedCamera đa luồng, Idle Detection 0.8s, Cooldown 1.2s, hiển thị FPS và nhãn thời gian thực.
   - Sẵn sàng chạy độc lập bằng lệnh:
     ```powershell
     python "Sign Language Translator\RunModel_119.py"
     ```
