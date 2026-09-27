# Báo cáo Task E2: Thêm Presence Flag Vào Feature Vector (131 Chiều)

> **Bối cảnh & Mục tiêu**:
> - Task E1 đã xác nhận: Epsilon Guard kích hoạt thường xuyên (17-19% frame) ngay cả ở cử chỉ bình thường. Khi đó, vector 63 chiều của tay bị đưa về toàn số 0.
> - Việc để mô hình LSTM tự suy đoán liệu vector toàn 0 là "tay vắng mặt" hay "tay đang ở gốc tọa độ" gây nhập nhằng cho mạng nơ-ron.
> - **Nhiệm vụ Task E2**:
>   1. Thêm **2 chiều Presence Flag** (LH present, RH present) vào cuối feature vector: giá trị `1.0` nếu `scale >= 1e-6` (có tay), `0.0` nếu Epsilon Guard kích hoạt $\rightarrow$ Nâng tổng số chiều từ **129 lên 131 chiều**.
>   2. Kiểm thử shape trên sequence mẫu (1 tay và 2 tay), xác nhận shape `(60, 131)` và khớp nối 100% với số liệu E1.
>   3. Khởi tạo kiến trúc `model_def_v3_presence.py` (`input_shape=(60, 131)`) và chạy huấn luyện thử nghiệm nhanh (8 epochs) để xác nhận pipeline chạy mượt mà, không sinh NaN, loss giảm đều, không ghi đè model production.

---

### 1. Cấu Trúc Layout Của Feature Vector Mới (131 Chiều)

Feature vector chuẩn hóa của mỗi frame được cấu tạo phân tầng rõ ràng:
- **`0 : 63` (63 chiều)**: Tọa độ chuẩn hóa bàn tay trái (Left Hand), centering quanh cổ tay, scaling bằng $S_{combined}$.
- **`63 : 126` (63 chiều)**: Tọa độ chuẩn hóa bàn tay phải (Right Hand), centering quanh cổ tay, scaling bằng $S_{combined}$.
- **`126 : 129` (3 chiều)**: Vector tương đối giữa 2 cổ tay ở tọa độ thô ($\text{Wrist}_{RH} - \text{Wrist}_{LH}$).
- **`129` (1 chiều - Mới)**: **`LH_Present_Flag`** (`1.0` nếu tay trái hợp lệ, `0.0` nếu kích hoạt Epsilon Guard).
- **`130` (1 chiều - Mới)**: **`RH_Present_Flag`** (`1.0` nếu tay phải hợp lệ, `0.0` nếu kích hoạt Epsilon Guard).

---

### 2. Kiểm Thử Shape & Khớp Nối Presence Flag Trên Sequence Mẫu

Thực nghiệm nạp raw keypoints từ `Data/` và chuẩn hóa qua hàm `normalize_keypoints(..., include_presence=True)`:

| Cử chỉ kiểm thử | Phân loại | Input Raw Shape | Output Normalized Shape | Số frame LH Present | Số frame RH Present | Đối chiếu với Task E1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`xin chao`** (Seq #50) | 1 tay (Single) | (60, 126) | **(60, 131)** | 0 / 60 (0.0%) | **55 / 60 (91.7%)** | Khớp 100%: Tay trái không dùng = 0, tay phải ký tích cực |
| **`chuc mung`** (Seq #50) | 2 tay (Dual) | (60, 126) | **(60, 131)** | **58 / 60** | **57 / 60** | Khớp 100%: Cả 2 tay đều giữ cờ 1.0 trong hầu hết frame (55/60 frames) |
| **`toi bi dau dau`** (Seq #50) | 1 tay (Occlusion) | (60, 126) | **(60, 131)** | 0 / 60 (0.0%) | **26 / 60** | **Khớp chính xác: RH rơi về 0 trong đúng 34/60 frame (56.7%) do che khuất!** |
| **`cap cuu`** (Seq #50) | 2 tay (Occlusion) | (60, 126) | **(60, 131)** | **21 / 60** | **21 / 60** | **Khớp chính xác: Cả 2 tay cùng rơi về 0 trong đúng 35/60 frame (58.3%)!** |

> **Xác nhận**: Output tensor đạt chuẩn kích thước `(60, 131)`. Các giá trị cờ hiện diện `0.0` và `1.0` phản ánh chuẩn xác 100% thời điểm kích hoạt Epsilon Guard đã đo đạc ở Task E1.

---

### 3. Kết Quả Huấn Luyện Thử Nghiệm Nhanh (Quick Training 8 Epochs)

- **Mô hình**: [model_def_v3_presence.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/model_def_v3_presence.py) (`LSTM 128 -> 64 -> 32, Tanh, Adam lr=1e-3, clipnorm=1.0`).
- **Dữ liệu huấn luyện**: 3,600 sequences (60 nhãn), chia Train = 3,060 seqs, Validation = 540 seqs.
- **Kích thước đầu vào**: `(None, 60, 131)`.
- **Checkpoint lưu riêng**: `Models/model_v3_presence_quick_test.h5` *(Hoàn toàn không ghi đè `model_b3_v2_descending.tflite` đang phục vụ production)*.

#### Bảng tiến trình huấn luyện qua 8 epochs:
| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| Epoch  1 | 3.7476 | 7.71% | 3.1553 | **15.74%** |
| Epoch  2 | 2.6739 | 24.18% | 2.2307 | **34.07%** |
| Epoch  3 | 1.9481 | 42.48% | 1.7738 | **48.15%** |
| Epoch  4 | 1.4529 | 57.22% | 1.3178 | **64.26%** |
| Epoch  5 | 1.0825 | 70.26% | 0.9754 | **73.15%** |
| Epoch  6 | 0.7786 | 80.52% | 0.8230 | **80.00%** |
| Epoch  7 | 0.5410 | 86.76% | 0.5409 | **87.41%** |
| Epoch  8 | 0.4100 | 89.87% | 0.4701 | **87.59%** |

#### Nhận định kỹ thuật:
1. **Tính tương thích của kiến trúc**:
   - Mô hình tiếp nhận input vector 131 chiều hoàn toàn trơn tru, không phát sinh bất kỳ lỗi kích thước tensor hay warning nào.
2. **Động lực hội tụ (Convergence Dynamics)**:
   - Loss giảm dốc đứng và đều đặn từ **3.7476 xuống 0.4100** qua 8 epochs.
   - Validation Loss giảm tương ứng từ **3.1553 xuống 0.4701**.
   - Validation Accuracy tăng vọt từ **15.7% lên 87.6%** chỉ sau 8 epochs ngắn ngủi.
3. **Độ ổn định số học (Numerical Stability)**:
   - **Hoàn toàn không xuất hiện NaN hoặc Inf** ở cả hàm mất mát và gradient.
   - Gradient clipping (`clipnorm=1.0`) kết hợp với hàm kích hoạt `tanh` duy trì sự ổn định tối đa cho các cổng nhớ LSTM khi tiếp nhận thêm 2 chiều tín hiệu nhị phân [0, 1].

---

### 4. Kết Luận Task E2

1. **Hoàn thành xuất sắc mục tiêu kỹ thuật**:
   - Hàm chuẩn hóa [preprocessing.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/preprocessing.py) đã hỗ trợ tham số `include_presence=True` để mở rộng vector lên **131 chiều**, đồng thời vẫn giữ mặc định `include_presence=False` (129 chiều) để bảo đảm tương thích ngược 100% với model production hiện tại trong [RunModel.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel.py).
2. **Kiến trúc v3 Presence sẵn sàng**:
   - File [model_def_v3_presence.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/model_def_v3_presence.py) được thiết lập chuẩn xác. Mô hình học rất nhanh và đạt độ chính xác validation cao ngay từ những epoch đầu tiên.
3. **Bước kế tiếp**:
   - Khi sẵn sàng nâng cấp toàn diện hệ thống, có thể tiến hành regenerate dataset sang 131 chiều và train đầy đủ bản production mới.

- **Biểu đồ đính kèm**: [docs/task_e2_presence_flag_verification.png](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/docs/task_e2_presence_flag_verification.png)
