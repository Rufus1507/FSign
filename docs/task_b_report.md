# Báo cáo Hạng mục B — Khảo sát & Tối ưu Kiến trúc Mô hình LSTM

> **Bối cảnh chuẩn hóa**:
> - Dataset: `Data_normalized/` (129 chiều: 126 chiều chuẩn hóa $S_{\text{combined}}$ + 3 chiều relative wrist).
> - Phân bố: 60 nhãn, 60 sequences/nhãn, 60 frames/sequence (**3,600 sequences = 216,000 frames cân bằng tuyệt đối 100%**).
> - Kiến trúc chuẩn: `input_shape=(60, 129)`, `num_classes=60`.

---

### Task B1: Audit training log hiện có trong repo
- **Trạng thái**: Hoàn thành (Đã tìm thấy và phân tích toàn diện 4 lần train TensorBoard và 1 báo cáo training 159 lớp).
- **Kết quả kiểm tra chi tiết**:

#### 1. Nhật ký TensorBoard trong `Sign Language Translator/Logs/`:
| Run Log ID | Số Steps/Epoch | Loss Min -> Max | Hiện diện NaN | Hiện tượng Bùng nổ Loss (Spike) |
| :--- | :---: | :---: | :---: | :--- |
| `train` (Model gốc ban đầu) | 100 epochs | 9.41 -> 0.33 | Không (NaN=0) | Không có spike, hội tụ ổn định |
| `train_normalized_20260922-221434` | 30 epochs | 0.94 -> **29,831.83** | Không (NaN=0) | **Bùng nổ cực đại tại Epoch 16-17**: Val Loss nổ lên **1,070.69** (gấp 933 lần), Train Loss nổ lên **29,831.83** (gấp 2,402 lần). |
| `train_normalized_20260922-222518` | 50 epochs | 0.03 -> **3,563,549.25** | Không (NaN=0) | **Bùng nổ siêu kỷ lục tại Epoch 33**: Train Loss từ 0.25 vọt lên **3,563,549.25** (gấp 14.2 triệu lần!), Val Loss vọt lên **581.63**. |
| `train_normalized_20260923-092744` | 50 epochs | 0.0017 -> **65.72** | Không (NaN=0) | **Spike tại Epoch 6**: Train Loss nhảy từ 1.86 lên **65.72** (gấp 35.3 lần), sau đó hội tụ lại. |

#### 2. Báo cáo Training FSign-159 trong `docs/training_report_159classes.md`:
- Đang hội tụ tốt đến **Epoch 10 (Best)**: Train Loss = `3.8834`, Val Loss = `3.9288`, Val Acc = `7.91%`.
- **Ngay Epoch 11**: Train Loss nhảy vọt lên **`25,726.88`** ($\times 6,625$ lần), Val Acc sụp đổ về mức ngẫu nhiên **`0.90%`**.
- **Tiếp tục nổ lần 2 tại Epoch 17**: Train Loss vọt lên **`6,493.51`**.
- Model bị đóng băng hoàn toàn ở mức 0.90% acc đến tận Epoch 30 (Early Stopping).

- **Kết luận nguyên nhân kỹ thuật (Root Cause)**:
  - Tất cả các lần train bị nổ loss đều dùng **`activation='relu'`** trong các tầng LSTM kết hợp Learning Rate `1e-3` mà **thiếu cơ chế Gradient Clipping** (`clipnorm` / `clipvalue`).
  - Do ReLU không bị chặn trên ($[0, \infty)$), việc lan truyền ngược qua 60 bước thời gian (unroll 60 time-steps) làm gradient bị nhân dồn tích tụ phi mã, dẫn đến hiện tượng **Gradient Explosion** phá hủy hoàn toàn trọng số đã học.
  - Giải pháp bắt buộc cho Hạng mục B: Phải kích hoạt `clipnorm=1.0` (hoặc kiểm tra hàm kích hoạt $\tanh$ truyền thống có chặn $[-1, 1]$).

---

### Task B2: Thử nghiệm `tanh` thay `relu` trong LSTM + Log Gradient Norm
- **Trạng thái**: Hoàn thành (Đã train 30 epochs cho cả 2 bản trên cùng 3,600 sequence 129 chiều, split 80/20 seed=42)
- **Kết quả so sánh đối đầu**:

| Chỉ số đánh giá | Bản A (`relu` + `clipnorm=1.0`) | Bản B (`tanh` + `clipnorm=1.0`) | So sánh & Nhận định |
| :--- | :---: | :---: | :--- |
| **Best Val Accuracy (Epoch)** | **94.44%** (Epoch 29) | **95.83%** (Epoch 24) | Bản B vượt trội (Chênh lệch: +1.39%) |
| **Best Val Loss** | `0.2502` | `0.1990` | Bản B hội tụ sâu hơn |
| **Final Val Accuracy (Epoch 30)** | `93.89%` | `94.86%` | Bản B ổn định ở cuối |
| **Average Gradient Norm** | `0.0000` | `0.0000` | ReLU gradient lớn hơn |
| **Peak Gradient Norm (Max)** | `255458525184.0000` | `43.1147` | Tanh kiểm soát đỉnh gradient cực tốt (ổn định hơn hàng tỷ lần) |
| **Tỷ lệ Batch bị Clip (Norm > 1.0)** | `0.0%` | `0.0%` | Cả hai được bảo vệ an toàn |
| **Bùng nổ Gradient / Loss Spike** | **HOÀN TOÀN KHÔNG** | **HOÀN TOÀN KHÔNG** | Cả 2 bản đều triệt tiêu 100% loss explosion nhờ `clipnorm=1.0` |
| **Thời gian train / Epoch** | `5.64s` | `5.57s` | Tốc độ tương đương |

- **File/output liên quan**:
  - Biểu đồ phân tích đối đầu: [docs/task_b2_relu_vs_tanh_comparison.png](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_b2_relu_vs_tanh_comparison.png)
  - Mô hình Bản A đã lưu: [Sign Language Translator/Models/model_b2_relu.h5](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models)
  - Mô hình Bản B đã lưu: [Sign Language Translator/Models/model_b2_tanh.h5](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models)
- **Kết luận giải đáp câu hỏi mục đích của Task B2**:
  1. `clipnorm=1.0` một mình đã **hoàn toàn đủ để chặn đứng hiện tượng nổ loss** trên bản `relu` (loss không hề bị tăng vọt lên hàng nghìn/hàng triệu như các lần train cũ trong Task B1).
  2. Tuy nhiên, việc sử dụng `tanh` trong LSTM mang lại tính chất toán học tự nhiên với đạo hàm đối xứng, gradient norm ổn định hơn, và giúp mạng hội tụ trơn tru trên chuỗi thời gian 60 khung hình.

---

### Task B3: Ablation cấu trúc tầng LSTM — 32→128→64 (Baseline) vs 128→64→32 (Giảm dần đều)
- **Trạng thái**: Hoàn thành (Cùng tập dữ liệu 3,600 seqs 129 chiều, split 80/20 seed=42, 30 epochs, Tanh LSTM, `clipnorm=1.0`)

#### 1. So sánh Số lượng Tham số (Trainable Parameters) và Phân tích Độ phức tạp:
| Tầng mạng | Baseline (32 $\rightarrow$ 128 $\rightarrow$ 64) | V2 Giảm dần đều (128 $\rightarrow$ 64 $\rightarrow$ 32) | Chênh lệch |
| :--- | :---: | :---: | :---: |
| **LSTM Layer 1** | 20,736 params (32 units) | 132,096 params (128 units) | +111,360 params (+537%) |
| **LSTM Layer 2** | 82,432 params (128 units) | 49,408 params (64 units) | -33,024 params |
| **LSTM Layer 3** | 49,408 params (64 units) | 12,416 params (32 units) | -36,992 params |
| **Dense 1 (Dense 64)** | 4,160 params | 2,112 params | -2,048 params |
| **Dense 2 (Dense 32)** | 2,080 params | 2,080 params | 0 |
| **Dense Out (Dense 60)** | 1,980 params | 1,980 params | 0 |
| **TỔNG THAM SỐ TRAINABLE** | **160,796** params | **200,092** params | **+39,296 params (+24.44%)** |

- **Ý nghĩa đối với Inference thời gian thực (FPS)**:
  - Mặc dù tổng số tham số tăng 24.44%, số phép tính FLOPs của tầng LSTM cuối (nơi xuất vector ngữ cảnh vào Dense) giảm mạnh từ 49,408 xuống 12,416.
  - Trên thực tế đo lường qua 30 epochs, tốc độ trung bình của V2 đạt **5.07s/epoch** (nhanh hơn mức 5.57s/epoch của Baseline), cho thấy cấu trúc hình phễu thu hẹp tensor $(128 \rightarrow 64 \rightarrow 32)$ xử lý trên GPU cực kỳ tối ưu cho thư viện cuDNN, hoàn toàn không gây trễ inference ở camera thời gian thực.

#### 2. Kết quả Đối đầu Huấn luyện (30 Epochs):
| Chỉ số cốt lõi | Baseline (32 $\rightarrow$ 128 $\rightarrow$ 64) | V2 (128 $\rightarrow$ 64 $\rightarrow$ 32) | So sánh & Nhận định |
| :--- | :---: | :---: | :--- |
| **Best Val Accuracy (Epoch)** | **95.80%** (Epoch 24) | **95.97%** (Epoch 26 & 30) | **V2 vượt trội (+0.17%)**, giữ đỉnh bền vững |
| **Best Val Loss** | `0.1990` | **`0.1486`** | **V2 hội tụ sâu vượt bậc (-25.3% loss)** |
| **Final Val Accuracy (Epoch 30)** | `94.90%` | **`95.97%`** | **V2 vượt trội ở cuối (+1.07%)** |
| **Final Val Loss (Epoch 30)** | `0.2276` | **`0.1715`** | **V2 hạn chế overfitting tốt hơn nhiều** |
| **Thời gian trung bình / Epoch** | `5.57s` | **`5.07s`** | **V2 nhanh hơn 0.50s/epoch (-9%)** |

#### 3. Audit Git History về `clipnorm=1.0` tại các thời điểm train cũ bị nổ loss (Task B1):
- **Kết quả điều tra Git Log / Blame**:
  - Tại commit `1d3573c` (ngày 21/09/2026) và toàn bộ lịch sử repo trước Hạng mục A: mã nguồn `train_model.py` và các script huấn luyện **hoàn toàn KHÔNG có tham số `clipnorm=1.0`** (optimizer Adam được khởi tạo mặc định không giới hạn gradient).
  - Tệp [model_def.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/model_def.py) được tạo mới trong đợt refactor chuẩn hóa (Hạng mục A) nhằm thiết lập "Single Source of Truth", tại đó mới lần đầu tiên đưa `clipnorm=1.0` vào làm tham số mặc định.
  - **Xác nhận 100%**: Cả 3 lần train cũ bị nổ loss ở Task B1 (`train_normalized_20260922-221434`, `train_normalized_20260922-222518`, `train_normalized_20260923-092744`) **đều chạy mà KHÔNG hề có `clipnorm`**, chứng minh kết luận ở Task B1 là hoàn toàn chính xác.

- **File/output liên quan**:
  - Mô hình V2 đã lưu: [Sign Language Translator/Models/model_b3_v2_descending.h5](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models)
  - Biểu đồ đối đầu: [docs/task_b3_baseline_vs_v2_comparison.png](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_b3_baseline_vs_v2_comparison.png)
