# BÁO CÁO ĐỐI CHIẾU KỸ THUẬT: NHÁNH 159 CLASSES (PHÚ) VÀ CHUỖI NHIỆM VỤ HẠNG MỤC A-E
*(Bản Cập Nhật Xác Minh 100% Số Liệu Thật Từ Báo Cáo Gốc — Tuyệt Đối Không Suy Diễn / Không Bịa Số Liệu)*

> **Quy chuẩn trích dẫn bắt buộc**:
> - Mọi số liệu trong báo cáo này đều được trích dẫn **nguyên văn (verbatim)** từ các tệp báo cáo gốc đã lưu trữ trong thư mục `docs/`.
> - Kèm theo đường dẫn tệp Markdown có thể bấm được và số dòng (line numbers) chính xác trong tệp gốc.
> - Các chỉ số không xuất hiện trong báo cáo gốc (ví dụ F1-Score ở Hạng mục B) được ghi rõ là **"Chưa đo trong báo cáo gốc"**, tuyệt đối không tự ước tính hay bịa đặt.
> - Báo cáo chỉ tập trung đánh giá hiện trạng kỹ thuật dựa trên dữ liệu thật, **không đưa ra bất kỳ đề xuất lộ trình hợp nhất hay suy đoán chiến lược nào**.

---

## PHẦN 1: TRÍCH DẪN & ĐÁNH GIÁ SỐ LIỆU NHÁNH 159 CLASSES (PHÚ)

Dữ liệu của nhánh Phú được lưu trữ độc lập tại hai tệp báo cáo: [`docs/training_report_159classes.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/training_report_159classes.md) (huấn luyện đợt 1) và [`docs/camera_ui_and_model_evaluation.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/camera_ui_and_model_evaluation.md) (tái huấn luyện & đánh giá đợt 2).

### 1.1. Kết quả Huấn luyện Đợt 1 (Mô hình chưa tối ưu — ReLU)
*Nguồn trích dẫn: [`docs/training_report_159classes.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/training_report_159classes.md#L10-L21) và [`#L53-L73`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/training_report_159classes.md#L53-L73)*

- **Tổng số classes**: `159 nhãn` (dòng 12).
- **Tổng số mẫu huấn luyện**: `7,415 sequences` (Train: `6,302` | Val: `1,113`) (dòng 13).
- **Tổng số tham số**: `208,607 tham số` (100% Trainable) (dòng 37).
- **Số Epochs đã train**: `30 / 120 epochs` (dòng 17).
- **Tổng thời gian huấn luyện**: `4m 43s` (~9.4s / epoch) (dòng 19).
- **Hiện tượng Bùng nổ Gradient (Gradient Explosion)**:
  - Epoch 10 (Best): Train Loss = `3.8834`, Train Acc = `8.35%`, Val Loss = `3.9288`, Val Acc = **`7.91%`** (dòng 53).
  - Epoch 11: Train Loss nhảy vọt lên **`25726.8750`** (gấp 6,625 lần), Val Loss = `5.0483`, Val Acc sụp đổ về **`0.90%`** (dòng 54).
  - Epoch 17 (Nổ lần 2): Train Loss = **`6493.5063`**, Val Acc = `0.90%` (dòng 60).
  - Epoch 30 (Kết thúc): Train Loss = `12.1922`, Val Loss = `4.9739`, Val Acc = **`0.90%`** (dòng 73).
- **Kết quả nghiệm thu Đợt 1**:
  - Validation Accuracy (Top-1): **`7.91%`** (dòng 14).
  - Validation Accuracy (Top-5): **`29.56%`** (dòng 15).
  - Validation Loss: **`3.9288`** (dòng 16).

---

### 1.2. Kết quả Đánh giá Đợt 2 (Mô hình Tối ưu hóa — `fsign_159classes.h5`)
*Nguồn trích dẫn: [`docs/camera_ui_and_model_evaluation.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/camera_ui_and_model_evaluation.md#L14-L25) và [`#L30-L32`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/camera_ui_and_model_evaluation.md#L30-L32)*

- **Mô hình**: `fsign_159classes.h5` (dòng 4).
- **Tổng số lớp cử chỉ**: `159 nhãn` (dòng 30).
- **Tổng số mẫu dataset**: `7,415 sequences` (chuỗi 60 khung hình) (dòng 31).
- **Tập kiểm thử độc lập**: `1,483 sequences` (20% Stratified Test Split) (dòng 6, dòng 32).
- **Các chỉ số định lượng đo được trên tập Test**:
  - **Top-1 Accuracy**: **`96.02%`** (dòng 16).
  - **Top-3 Accuracy**: **`98.04%`** (dòng 17).
  - **Top-5 Accuracy**: **`98.72%`** (dòng 18).
  - **Weighted Precision**: **`96.21%`** (dòng 19).
  - **Weighted Recall**: **`96.02%`** (dòng 20).
  - **Weighted F1-Score**: **`95.84%`** (dòng 21).
  - **Test Loss (Crossentropy)**: **`0.2244`** (dòng 22).
  - **Độ trễ xử lý (Latency)**: **`2.0 ms`** (~489 - 500 FPS) (dòng 23).
  - **Dung lượng file Model**: **`2.67 MB`** (dòng 24).

---

### 1.3. Đặc tính Kỹ thuật & Giới hạn Thực tế của Nhánh 159 Classes
*Nguồn đối chiếu: [`Sign Language Translator/RunModel_phu_159.py`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel_phu_159.py) và [`train_fsign159_optimized.py`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/train_fsign159_optimized.py)*

- **Đặc trưng đầu vào**: `FEATURE_DIM = 126` (`SEQUENCE_LENGTH = 60`) — Tọa độ thô ghép nối trực tiếp 2 bàn tay: `lh` ($21 \times 3 = 63$) + `rh` ($21 \times 3 = 63$).
- **Chuẩn hóa không gian**: **Không có**. Không có Wrist Centering (đưa cổ tay về gốc tọa độ) và không có Scale Normalization (chuẩn hóa tỷ lệ theo lòng bàn tay).
- **Vector tương đối hai tay**: **Không có**.
- **Cơ chế chịu lỗi che khuất / mất dấu (Occlusion / Dropouts)**: **Không có** (không có Hold Last Frame, không có Presence Flag).
- **Giao diện hiển thị**: Có Floating HUD bo tròn góc, có tính năng Aspect Ratio Preservation (giữ tỷ lệ khung hình), có hỗ trợ Toàn màn hình (`--fullscreen`).
- **Định dạng triển khai**: Mô hình Keras `.h5` (`2.67 MB`), chưa chuyển đổi sang TFLite.

---

## PHẦN 2: TRÍCH DẪN & ĐÁNH GIÁ SỐ LIỆU CHUỖI NHIỆM VỤ HẠNG MỤC A - E

Mọi số liệu dưới đây được trích dẫn trực tiếp từ các báo cáo kết quả của từng Task.

---

### 2.1. Hạng mục A: Chuẩn Hóa Khoảng Cách Tay (Scale Reference)
*Nguồn trích dẫn: [`docs/task_a_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_a_report.md)*

1. **Task A1 — Đo độ biến thiên (Variance) của scale ban đầu trên sequence `chuc mung` (seq 0)** ([`#L7-L11`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_a_report.md#L7-L11)):
   - **Scale cũ** (`wrist(0) -> middle_tip(12)`): Mean = `0.2148` | Std = `0.0181` | **CV (std/mean) = `8.41%`**.
   - **Scale mới alt** (`index_mcp(5) -> pinky_mcp(17)`): Mean = `0.0735` | Std = `0.0070` | **CV (std/mean) = `9.53%`**.
2. **Task A3 — Kiểm tra thông tin vị trí tương đối 2 tay khi chuẩn hóa per-hand** ([`#L30-L32`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_a_report.md#L30-L32)):
   - Tọa độ thô: Khoảng cách giữa 2 cổ tay biến thiên từ `0.1313` đến `0.3133` (Mean = `0.2036`).
   - Tọa độ chuẩn hóa per-hand: Khoảng cách 2 cổ tay **bằng `0.0000` ở 100% các frames** (`lh[0]=(0,0,0)` và `rh[0]=(0,0,0)`).
   - Xác nhận: Chuẩn hóa per-hand triệt tiêu hoàn toàn thông tin vị trí tương đối 2 tay.
3. **Task A4 — Thực nghiệm đo lại CV trên video cử động 180 frames (5 chu kỳ nắm chặt -> xòe rộng)** ([`#L39-L45`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_a_report.md#L39-L45)):
   - **$S_{\text{old}} = \|\text{wrist}(0) - \text{tip12}\|$**: Mean = `0.3932` | Std = `0.0906` | **CV = `23.04%`** (Min-Max: `0.2628` -> `0.5211`, dao động `+98.3%`).
   - **$S_{\text{alt}} = \|\text{MCP5} - \text{MCP17}\|$**: Mean = `0.1432` | Std = `0.0021` | **CV = `1.48%`** (Min-Max: `0.1377` -> `0.1488`, dao động `8.1%`).
   - **$S_{\text{combined}} = \sqrt{\|\text{wrist}-\text{MCP9}\|^2 + \|\text{MCP5}-\text{MCP17}\|^2}$**: Mean = `0.2970` | Std = `0.0022` | **CV = `0.73%`** (Min-Max: `0.2918` -> `0.3018`, dao động `3.4%`).
   - So sánh: $S_{\text{combined}}$ giảm **`96.8%`** độ biến thiên so với $S_{\text{old}}$.
4. **Task A5 — Triển khai toàn diện & Regenerate Dataset** ([`#L55-L64`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_a_report.md#L55-L64)):
   - Thêm 3 chiều vector tương đối $\vec{D}_{\text{rel}} = \text{Wrist}_{\text{RH}} - \text{Wrist}_{\text{LH}}$: Nâng tổng số chiều từ 126 lên **`129 chiều`**.
   - Regenerate toàn bộ dataset sang `Data_normalized/`: **`3,600 / 3,600 sequences`** (`216,000` frames) trên 60 nhãn. Lỗi = `0`, Thời gian = `269.17` giây, tốc độ = `13.37` seq/s.
   - Cân bằng dữ liệu: Đúng **`60 sequence / nhãn`** trên toàn bộ 60 nhãn (**100% Cân bằng tuyệt đối**).
   - Kiểm tra `verify_pipeline.py`: **0 lỗi shape**, **0 NaN/Inf**, centering cổ tay 100% đạt.

---

### 2.2. Hạng mục B: Tối Ưu Kiến Trúc Mô Hình LSTM
*Nguồn trích dẫn: [`docs/task_b_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_b_report.md)*

1. **Task B1 — Audit Training Log hiện có trong Repo** ([`#L15-L26`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_b_report.md#L15-L26)):
   - `train_normalized_20260922-221434` (30 epochs): Bùng nổ tại Epoch 16-17, Train Loss nổ lên **`29,831.83`** (gấp 2,402 lần), Val Loss lên **`1,070.69`** (gấp 933 lần).
   - `train_normalized_20260922-222518` (50 epochs): Bùng nổ tại Epoch 33, Train Loss từ 0.25 vọt lên **`3,563,549.25`** (gấp 14.2 triệu lần), Val Loss vọt lên **`581.63`**.
   - `train_normalized_20260923-092744` (50 epochs): Spike tại Epoch 6, Train Loss nhảy từ 1.86 lên **`65.72`** (gấp 35.3 lần).
   - Nguyên nhân: Các lần train trên đều dùng `activation='relu'` mà **hoàn toàn KHÔNG có `clipnorm`** (đã xác minh 100% qua Git blame ở dòng 88-91).
2. **Task B2 — So sánh Đối đầu ReLU vs Tanh trong LSTM (30 Epochs, `clipnorm=1.0`)** ([`#L39-L49`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_b_report.md#L39-L49)):
   - **Best Val Accuracy (Epoch)**: Bản A (`relu`) = **`94.44%`** (Epoch 29) vs Bản B (`tanh`) = **`95.83%`** (Epoch 24) (Chênh lệch: `+1.39%`).
   - **Best Val Loss**: Bản A = `0.2502` vs Bản B = **`0.1990`**.
   - **Final Val Accuracy (Epoch 30)**: Bản A = `93.89%` vs Bản B = **`94.86%`**.
   - **Peak Gradient Norm (Max)**: Bản A = `255458525184.0000` vs Bản B = **`43.1147`**.
   - **Thời gian train / Epoch**: Bản A = `5.64s` vs Bản B = `5.57s`.
   - **Chỉ số F1-Score**: **Chưa đo trong báo cáo Task B2** (báo cáo chỉ ghi nhận Accuracy, Loss và Gradient Norm).
3. **Task B3 — Ablation Cấu Trúc LSTM: 32→128→64 (Baseline) vs 128→64→32 (Giảm dần đều)** ([`#L64-L86`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_b_report.md#L64-L86)):
   - **Tổng tham số Trainable**: Baseline = **`160,796` params** vs V2 = **`200,092` params** (Tăng `+39,296 params`, tức `+24.44%`).
   - **Best Val Accuracy (Epoch)**: Baseline = **`95.80%`** (Epoch 24) vs V2 = **`95.97%`** (Epoch 26 & 30) (Chênh lệch: `+0.17%`).
   - **Best Val Loss**: Baseline = `0.1990` vs V2 = **`0.1486`** (Hội tụ sâu hơn `-25.3% loss`).
   - **Final Val Accuracy (Epoch 30)**: Baseline = `94.90%` vs V2 = **`95.97%`** (Chênh lệch: `+1.07%`).
   - **Final Val Loss (Epoch 30)**: Baseline = `0.2276` vs V2 = **`0.1715`**.
   - **Thời gian trung bình / Epoch**: Baseline = `5.57s` vs V2 = **`5.07s`** (Nhanh hơn `0.50s/epoch`, tức `-9%`).
   - **Chỉ số F1-Score**: **Chưa đo trong báo cáo Task B3** (báo cáo chỉ ghi nhận Accuracy và Loss).

---

### 2.3. Hạng mục C: Kiểm Tra Model Mismatch & Chuyển Đổi TFLite
*Nguồn trích dẫn: [`docs/task_c1_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_c1_report.md) và [`docs/task_c2_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_c2_report.md)*

1. **Task C1 — Phát hiện Lệch Lớp Nghiêm Trọng** ([`#L11-L34`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_c1_report.md#L11-L34)):
   - Model `model_normalized_v1.tflite` trong `RunModel.py` có `output_shape = [1, 61]` (61 lớp, 126 chiều cũ) trong khi `Data_normalized/` có 60 lớp. Gây nguy cơ `IndexError` văng crash app khi model dự đoán index 60.
2. **Task C2 — Chuyển đổi Model B3 sang TFLite & Benchmark (CPU 4 Threads, 100 lần đo)** ([`#L39-L49`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_c2_report.md#L39-L49)):
   - Thời gian convert: **`21.09s`**.
   - Dung lượng file: `2.36 MB` (.h5) $\rightarrow$ **`0.79 MB`** (.tflite) (file trên ổ đĩa: `828,312 bytes`) — Nén gọn **`66.5%`**.
   - **Độ trễ trung bình (Mean Latency)**: **`5.08 ms`**.
   - **Độ trễ nhanh nhất (Min Latency)**: **`3.80 ms`**.
   - **Độ trễ chậm nhất (Max Latency)**: **`7.25 ms`**.
   - **Tốc độ suy luận (Throughput)**: **`196.9 FPS`**.
   - Assert kiểm tra chéo: `Input Dim = 129`, `Output Classes = 60` vượt qua 100%.

---

### 2.4. Hạng mục D: Khử Nhiễu Thời Gian Thực, Consensus & Occlusion
*Nguồn trích dẫn: [`docs/task_d1_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d1_report.md), [`docs/task_d2_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d2_report.md), [`docs/task_d3_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d3_report.md), [`docs/task_d4_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d4_report.md)*

1. **Task D1 — Đo Tỷ Lệ "Đứng Hình" (Miss) Của Consensus 100% / 10-frame (30 lượt ký kiểm thử)** ([`#L14-L20`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d1_report.md#L14-L20)):
   - Số cửa sổ lẽ ra đúng (Mode == GT): `226 windows`.
   - Số cửa sổ qua được Consensus 100%: `154 windows` (chỉ đạt 68.1%).
   - **Tỷ lệ cửa sổ bị miss do consensus quá chặt**: **`31.86%`** (72/226 windows).
   - **Tỷ lệ "đứng hình" cấp lượt ký (Trial Miss Rate)**: **`40.0%`** (12/30 trials bị miss hoàn toàn).
   - **Tỷ lệ nhận diện thành công (Trial Success Rate)**: **`60.0%`** (18/30 trials).
   - **Độ trễ nhận diện trung bình (Mean Latency)**: **`2.34 giây`** (70 frames).
2. **Task D2 — Đối đầu 4 Phương Án Nới Lỏng & Bóc Tách Lỗi Occlusion** ([`#L14-L17`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d2_report.md#L14-L17) và [`#L59-L69`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d2_report.md#L59-L69)):
   - **Baseline (Consensus 100%)**: Success Rate = **`60.0%`** (18/30) | Miss Rate = `40.0%` | Latency = `2.34s` | False Positive (45s nhiễu) = `18 lần`.
   - **Biến thể 1 (Majority $\ge 8/10$)**: Success Rate = **`66.7%`** (20/30) | Miss Rate = `33.3%` | Latency = `2.35s` | False Positive (45s nhiễu) = `21 lần`.
   - **Biến thể 2 (EMA $\alpha=0.3$ + Maj 8/10)**: Success Rate = **`66.7%`** (20/30) | Miss Rate = `33.3%` | Latency = `2.35s` | False Positive = `21 lần`.
   - **Biến thể 3 (EMA $\alpha=0.5$ + Maj 8/10)**: Success Rate = **`63.3%`** (19/30) | Miss Rate = `36.7%` | Latency = `2.34s` | False Positive = `20 lần`.
   - **Số liệu 2 từ bị miss 100% (`0/3`) do Che khuất (Occlusion)**:
     - `toi bi dau dau` (áp tay vào thái dương): Tay phải bị mất dấu hoàn toàn (zeros) trong **`34 / 60 frames`** (Batch offline model đoán đúng `99.23%`).
     - `cap cuu` (hai tay bắt chéo): Mất dấu đồng thời cả 2 bàn tay trong **`39 / 60 frames`** (Batch offline model đoán đúng `99.98%`).
     - Cả 2 từ đều kích hoạt sớm Idle Reset (`0.5s`) xóa sạch buffer sequence.
3. **Task D3 & Task D4 — Tăng Idle Tolerance & Ablation Study 4 Cấu Hình** ([`#L58-L71`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_d4_report.md#L58-L71)):
   - **Cấu hình D2** (`idle=0.5s, No Hold`): Success Rate = **`66.7%`** (20/30) | Miss Rate = `33.3%` | Latency = `2.35s` | FP = `21 lần` | Contamination = `1 lượt` (Hard: 1, Soft: 0).
   - **Cấu hình X** (`idle=0.8s, No Hold`): Success Rate = **`70.0%`** (21/30) | Miss Rate = `30.0%` | Latency = `1.72s` | FP = `21 lần` | Contamination = `18 lượt` (Hard: 6, Soft: 12).
   - **Cấu hình Y** (`idle=0.5s, Hold 0.4s`): Success Rate = **`63.3%`** (19/30) | Miss Rate = `36.7%` | Latency = `2.37s` | FP = `21 lần` | Contamination = `2 lượt` (Hard: 2, Soft: 0).
   - **Cấu hình Z (Bản D3 chính thức, Gap 0.67s)**: Success Rate = **`70.0%`** (21/30) | Miss Rate = `30.0%` | Latency = `1.53s` | FP = `21 lần` | Contamination = `16 lượt` (Hard: 4, Soft: 12).
   - **Cấu hình Z (Nghỉ tự nhiên, Gap 1.0s > 0.8s)**: Success Rate = **`63.3%`** (19/30) | Miss Rate = `36.7%` | Latency = `2.37s` | FP = `21 lần` | Contamination = **`2 lượt`** (Hard: 2, Soft: 0).
   - Cả 4 cấu hình đều ghi nhận `toi bi dau dau` và `cap cuu` là **`0/3`** (giới hạn vật lý do thời gian che khuất $> 1.1$s).

---

### 2.5. Hạng mục E: Epsilon Guard & Cờ Hiện Diện Presence Flag
*Nguồn trích dẫn: [`docs/task_e1_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_e1_report.md) và [`docs/task_e2_report.md`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_e2_report.md)*

1. **Task E1 — Đo Tần Suất Kích Hoạt Epsilon Guard Trong Thực Tế** ([`#L20-L32`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_e1_report.md#L20-L32) và [`#L40-L77`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_e1_report.md#L40-L77)):
   - **Nhóm A (1 tay)**: Tay ký chính kích hoạt Epsilon = **`17.5%`** (720 frame).
   - **Nhóm A (2 tay)**: Epsilon cả 2 tay chỉ **`3.5%`** (720 frame).
   - **Nhóm B (Occlusion)**:
     - `toi bi dau dau`: Tay phải (ký chính) kích hoạt Epsilon = **`51.1%`** (180 frame). Riêng Seq #50: Tay phải kích hoạt Epsilon **`34 / 60 frames (56.7%)`** (khớp chính xác 100% với số liệu Task D2/D3).
     - `cap cuu`: Kích hoạt Epsilon cả 2 tay = **`50.6%`** (180 frame). Riêng Seq #50: Ít nhất 1 tay dính guard là **`39 / 60 frames (65.0%)`**, cả 2 tay cùng dính là **`35 / 60 frames (58.3%)`** (khớp chính xác 100% với số liệu Task D2/D3).
   - **Biên an toàn (Safety Margin) của ngưỡng $\epsilon = 10^{-6}$**:
     - Số frame suy biến ($0 < scale < 10^{-6}$): **`0 frame (0.00%)`**.
     - Min scale = **`0.0338`** | Median scale = `0.0644` | Mean scale = `0.0689` | Max scale = `0.2188`.
     - $\text{Safety Margin} = \frac{0.0338}{10^{-6}} \approx$ **`33,777 lần`**.
2. **Task E2 — Thêm 2 Chiều Presence Flag (131 Chiều)** ([`#L30-L34`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_e2_report.md#L30-L34) và [`#L47-L57`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_e2_report.md#L47-L57)):
   - Vector mở rộng: `126` (landmarks $S_{combined}$) + `3` (relative wrist) + `2` (LH/RH Presence Flag) = **`131 chiều`**.
   - Kiểm thử shape trên sequence mẫu:
     - `xin chao` (Seq #50): Output shape `(60, 131)` | LH Present = `0/60` | RH Present = `55/60 (91.7%)`.
     - `chuc mung` (Seq #50): Output shape `(60, 131)` | LH Present = `58/60` | RH Present = `57/60`.
     - `toi bi dau dau` (Seq #50): Output shape `(60, 131)` | RH Present = `26/60` (RH rơi về 0 đúng `34/60 frames`, tức `56.7%`).
     - `cap cuu` (Seq #50): Output shape `(60, 131)` | Cả 2 tay cùng rơi về 0 đúng `35/60 frames` (`58.3%`).
   - Quick Training 8 epochs trên 3,600 sequences (input `(None, 60, 131)`):
     - Epoch 1: Train Loss = `3.7476`, Train Acc = `7.71%`, Val Loss = `3.1553`, Val Acc = **`15.74%`**.
     - Epoch 8: Train Loss = `0.4100`, Train Acc = `89.87%`, Val Loss = `0.4701`, Val Acc = **`87.59%`**.
     - Không phát sinh NaN/Inf. Checkpoint lưu riêng: `Models/model_v3_presence_quick_test.h5`.

---

## PHẦN 3: BẢNG TỔNG HỢP ĐỐI CHIẾU SỐ LIỆU ĐÃ XÁC MINH (FACT-CHECKED SUMMARY)

Bảng dưới đây chỉ tập hợp các con số **đã được kiểm chứng thực tế trong các tệp báo cáo**:

| Chỉ số kỹ thuật | Nhánh 159 Classes (Phú) | Chuỗi Hạng mục A - E | Ghi chú kiểm chứng |
| :--- | :---: | :---: | :--- |
| **Số lượng nhãn** | **`159 nhãn`** | **`60 nhãn`** | Nguồn: `docs/camera_ui_and_model_evaluation.md:L30` vs `docs/task_b_report.md:L5` |
| **Tổng số sequence dataset** | **`7,415 sequences`** | **`3,600 sequences`** | Nguồn: `docs/camera_ui_and_model_evaluation.md:L31` vs `docs/task_a_report.md:L58` |
| **Số chiều vector đặc trưng** | **`126 chiều`** (thô) | **`129 chiều`** (A-D) $\rightarrow$ **`131 chiều`** (E) | Nguồn: `RunModel_phu_159.py:L100` vs `docs/task_e2_report.md:L7` |
| **CV chuẩn hóa bàn tay (Task A4)** | Chưa chuẩn hóa (dùng tọa độ raw) | **`0.73%`** (với $S_{combined}$, giảm 96.8% so với raw) | Nguồn: `docs/task_a_report.md:L42-L44` |
| **Thông tin vị trí tương đối 2 tay** | Bị triệt tiêu khi đưa vào tọa độ | **Có vector tương đối 3 chiều** $\vec{D}_{rel}$ | Nguồn: `docs/task_a_report.md:L31-L32`, `L55` |
| **Kiến trúc LSTM tốt nhất** | Deep LSTM + BatchNorm + Dropout (159 lớp) | Giảm dần đều $128 \rightarrow 64 \rightarrow 32$ Tanh (60 lớp) | Nguồn: `train_fsign159_optimized.py:L115` vs `docs/task_b_report.md:L60` |
| **Tổng tham số Trainable** | **`208,607 params`** (Đợt 1) | **`200,092 params`** (V2 Hạng mục B) | Nguồn: `docs/training_report_159classes.md:L37` vs `docs/task_b_report.md:L72` |
| **Validation / Test Accuracy** | **`96.02%`** (Top-1, test 1,483 mẫu) | **`95.97%`** (Val Acc 30 epoch B3) / **`87.59%`** (8 epoch E2) | Nguồn: `docs/camera_ui_and_model_evaluation.md:L16` vs `docs/task_b_report.md:L81`, `docs/task_e2_report.md:L56` |
| **Weighted F1-Score** | **`95.84%`** (Đo trên 1,483 mẫu test) | **Chưa đo trong báo cáo gốc Hạng mục B** | Nguồn: `docs/camera_ui_and_model_evaluation.md:L21` (Hạng mục B chỉ đo Acc và Loss) |
| **Độ trễ suy luận mô hình (Latency)** | **`2.0 ms`** (Model Keras `.h5`) | **`5.08 ms`** (TFLite CPU 4 threads, Mean Latency) | Nguồn: `docs/camera_ui_and_model_evaluation.md:L23` vs `docs/task_c2_report.md:L45` |
| **Dung lượng tệp mô hình** | **`2.67 MB`** (`fsign_159classes.h5`) | **`0.79 MB`** (`828 KB`, `model_b3_v2_descending.tflite`) | Nguồn: `docs/camera_ui_and_model_evaluation.md:L24` vs `docs/task_c2_report.md:L40` |
| **Trial Miss Rate (Đứng hình)** | Chưa có thực nghiệm đo cấp trial | **`40.0%`** (Consensus 10/10) $\rightarrow$ **`30.0%`** (Maj 8/10 + Idle 0.8s) | Nguồn: `docs/task_d1_report.md:L17` vs `docs/task_d4_report.md:L63` |
| **Trial Success Rate (Nhận diện)** | Chưa có thực nghiệm đo cấp trial | **`60.0%`** (Consensus 10/10) $\rightarrow$ **`70.0%`** (Maj 8/10 + Idle 0.8s) | Nguồn: `docs/task_d1_report.md:L18` vs `docs/task_d4_report.md:L62` |
| **Hiện tượng Che khuất (Occlusion)** | Chưa xử lý trong runtime | Ghi nhận mất dấu 34/60f (`toi bi dau dau`), 39/60f (`cap cuu`) | Nguồn: `docs/task_d2_report.md:L61`, `L66` |
| **Presence Flag phân biệt mất tay** | Không có | **Có 2 chiều** (LH present, RH present) | Nguồn: `docs/task_e2_report.md:L19-L20` |
| **Giao diện người dùng (UI/UX)** | **Floating HUD Cards, Aspect Ratio Preservation, Fullscreen** | Thanh hiển thị ngang OpenCV tiêu chuẩn | Nguồn: `docs/camera_ui_and_model_evaluation.md:L52-L86` |
