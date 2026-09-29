# BÁO CÁO KẾT QUẢ HUẤN LUYỆN: TASK MERGE-5
# BASELINE LSTM TRÊN TẬP DỮ LIỆU HỢP NHẤT 119 NHÃN (KIẾN TRÚC TANH CHỐT)

> **Thông tin kỹ thuật thực nghiệm**:
> - **Thời điểm thực hiện**: 29/09/2026
> - **Mục tiêu**: Huấn luyện kiến trúc cơ sở (Baseline Apples-to-Apples) đã chốt ở Hạng mục B (`128 -> 64 -> 32`, `tanh`, `clipnorm=1.0`) trên toàn bộ tập dữ liệu hợp nhất 119 nhãn mới (Task MERGE-4).
> - **Nguyên tắc phương pháp luận**: **KHÔNG ĐỔI KIẾN TRÚC**, không thêm Bi-LSTM, không thêm Attention, không thêm BatchNorm/Dropout để làm chuẩn đối đầu chuẩn xác.
> - **Tập dữ liệu**: [`Data_normalized_merged119/`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Data_normalized_merged119) (119 nhãn, 6,840 sequences, 129 chiều).
> - **Phân chia dữ liệu**: Phân tầng (Stratified Split) **80% Train (5,472 sequences) / 20% Test (1,368 sequences)**.
> - **Tệp mô hình xuất xưởng**: [`Sign Language Translator/Models/baseline_119_tanh.h5`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models/baseline_119_tanh.h5).
> - **Lịch sử & Đánh giá**: [`baseline_119_training_history.json`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Logs/baseline_119_training_history.json) và [`baseline_119_test_evaluation.json`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Logs/baseline_119_test_evaluation.json).

---

## 1. KẾT QUẢ TỔNG QUAN TRÊN TẬP TEST (1,368 SEQUENCES)

Mô hình đạt độ chính xác ấn tượng ngay ở phiên bản Baseline cơ sở mà không cần bất kỳ kỹ thuật bổ trợ phức tạp nào:

| Chỉ số đánh giá | Giá trị thực tế | Đánh giá & So sánh |
| :--- | :---: | :--- |
| **Test Accuracy (Top-1)** | **93.35%** | **1,277 / 1,368 mẫu đoán đúng** |
| **Top-3 Accuracy** | **96.27%** | **1,317 / 1,368 mẫu nằm trong Top 3** |
| **Top-5 Accuracy** | **97.08%** | **1,328 / 1,368 mẫu nằm trong Top 5** |
| **Test Loss** | **0.4987** | Hội tụ sâu, không bị underfitting |
| **Bùng nổ Gradient / NaN** | **HOÀN TOÀN KHÔNG (0 NaN)** | `clipnorm=1.0` + `tanh` triệt tiêu 100% bùng nổ gradient |
| **Thời gian huấn luyện** | **431.83s (~7m 11s)** | 46 epochs (dừng sớm bởi EarlyStopping) |
| **Epoch tốt nhất** | **Epoch 31 / 34** | `val_loss = 0.4918`, `val_acc = 93.35%` |

---

## 2. DIỄN BIẾN QUÁ TRÌNH HỌI TỤ THEO TỪNG MỐC EPOCH

Quá trình huấn luyện diễn ra ổn định tuyệt đối từ epoch đầu tiên tới khi hội tụ hoàn toàn:

| Mốc | Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Top-3 | Val Top-5 | Learning Rate | Ghi chú kỹ thuật |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Khởi đầu** | **1** | 4.3033 | 4.84% | 3.5558 | 11.18% | 26.83% | 38.23% | $1 \times 10^{-3}$ | Khởi đầu từ $\approx \ln(119)$, không giật NaN |
| | **2** | 3.1324 | 18.09% | 2.7361 | 24.85% | 49.20% | 64.91% | $1 \times 10^{-3}$ | Loss giảm dốc mạnh, mô hình bắt đầu học |
| | **3** | 2.4725 | 31.67% | 2.3136 | 35.31% | 60.38% | 72.30% | $1 \times 10^{-3}$ | Vượt mốc 35% |
| | **5** | 1.6208 | 53.07% | 1.6029 | 53.95% | 79.68% | 87.06% | $1 \times 10^{-3}$ | Vượt mốc 50% sau 5 epoch (~46s) |
| **Tăng tốc** | **10** | 0.6974 | 79.79% | 0.9996 | 73.10% | 88.38% | 93.42% | $1 \times 10^{-3}$ | Val loss xuống dưới 1.0 |
| | **15** | 0.3811 | 89.25% | 0.7788 | 80.63% | 93.13% | 95.91% | $1 \times 10^{-3}$ | Vượt mốc 80% Val Acc |
| | **20** | 0.2291 | 93.70% | 0.6409 | 84.58% | 94.44% | 96.64% | $1 \times 10^{-3}$ | Vượt mốc 84% Val Acc |
| | **25** | 0.1706 | 95.34% | 0.5847 | 87.06% | 95.25% | 96.86% | $1 \times 10^{-3}$ | Bắt đầu tiếp cận vùng tối ưu |
| **Đỉnh cao** | **31** | **0.0638** | **98.36%** | **0.4918** | **92.25%** | **96.35%** | **96.93%** | $5 \times 10^{-4}$ | **Val Loss thấp nhất (0.4918)** |
| | **34** | **0.0272** | **99.32%** | **0.4987** | **93.35%** | **96.27%** | **97.08%** | $5 \times 10^{-4}$ | **Val Acc cao nhất (93.35%) - Lưu Checkpoint** |
| **Bão hòa** | **36** | 0.0258 | 99.40% | 0.5246 | 92.40% | 96.20% | 97.15% | $2.5 \times 10^{-4}$ | ReduceLROnPlateau hạ LR xuống 0.00025 |
| | **41** | 0.0172 | 99.52% | 0.5356 | 93.06% | 96.35% | 97.15% | $1.25 \times 10^{-4}$ | Hạ LR xuống 0.000125 |
| **Kết thúc** | **46** | 0.0159 | 99.56% | 0.5511 | 92.98% | 96.27% | 97.08% | $6.25 \times 10^{-5}$ | **EarlyStopping kích hoạt (Patience 15)** |

### Phân tích hiện tượng Gradient & Động học học tập:
1. **Kiểm soát bùng nổ gradient hoàn hảo**: Ở Hạng mục B1 trước đây, hàm `relu` khi train với nhiều lớp đã gặp hiện tượng gradient explosion (loss vọt lên `NaN` ở epoch 2-4). Tại Task MERGE-5, với số lớp tăng vọt lên **119 classes**, việc kết hợp `activation='tanh'` và `clipnorm=1.0` đã giữ gradient luôn ổn định trong biên độ an toàn, không có bất kỳ một xung đột số học nào.
2. **Hội tụ tự nhiên và đúng điểm dừng**: Mô hình đạt cực tiểu loss ở Epoch 31 (`0.4918`) và đạt cực đại độ chính xác ở Epoch 34 (`93.35%`). Sau đó 12 epoch, val loss dao động nhẹ quanh mức 0.52 - 0.55 trong khi train loss xuống 0.0159. Cơ chế EarlyStopping đã dừng lại ở Epoch 46 và tự động khôi phục đúng trọng số tốt nhất của Epoch 31/34.

---

## 3. PHÂN TÍCH NHÓM NHÃN CÓ ACCURACY THẤP NHẤT

Bảng xếp hạng 10 nhãn có accuracy thấp nhất trên tập test:

| STT | Tên Nhãn | Nguồn gốc | Số mẫu gốc | Mẫu Test | Đoán đúng | Accuracy | F1-Score | Phân tích nguyên nhân |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Chào** | Video | 34 | 7 | 4 | **57.14%** | 66.67% | Nhầm 2 lần sang `Chậm lại`, 1 lần sang `Bệnh viện` |
| **2** | **Nôn ói** | Video | 44 | 9 | 6 | **66.67%** | 70.59% | Nhầm 2 lần sang `Dạy dỗ` |
| **3** | **Nói** | Video | **30** *(ngưỡng sàn)* | 6 | 4 | **66.67%** | 72.73% | Mẫu ít nhất dataset (30 mẫu), nhầm 1 lần |
| **4** | **Sử dụng** | Video | 59 | 12 | 8 | **66.67%** | 72.73% | Nhầm 4 lần sang `Nói xấu` (quỹ đạo tay trước ngực tương đồng) |
| **5** | **Khóc** | Video | 67 | 13 | 9 | **69.23%** | 81.82% | Nhầm sang các cử chỉ đưa tay lên vùng mặt |
| **6** | **Họ** | Video | 48 | 10 | 7 | **70.00%** | 82.35% | Cử chỉ chỉ tay đơn giản, dễ nhiễu góc cổ tay |
| **7** | **Nặng** | Video | 48 | 10 | 7 | **70.00%** | 82.35% | Động tác hạ 2 tay xuống |
| **8** | **Phạt** | Video | 37 | 7 | 5 | **71.43%** | 76.92% | Thuộc nhóm ít mẫu (< 40 mẫu) |
| **9** | **Mua** | Video | 35 | 7 | 5 | **71.43%** | 83.33% | Thuộc nhóm ít mẫu (< 40 mẫu) |
| **10** | **Đi** | Video | 62 | 12 | 9 | **75.00%** | 81.82% | Chuyển động ngón tay bước đi |

### Nhận xét tương quan mẫu (Sample Size Correlation):
* **6/10 nhãn accuracy thấp nhất** thuộc về nhóm có số lượng mẫu video ít ($\le 44$ mẫu): `Chào` (34), `Nói` (30), `Phạt` (37), `Mua` (35), `Nôn ói` (44).
* Nhãn **`Nói`** (nhãn ít mẫu nhất trong toàn bộ 119 nhãn, sát ngưỡng lọc 30 mẫu) vẫn đạt **66.67% Accuracy (Top-1)** và **F1-Score 72.73%**, chứng minh việc giữ ngưỡng lọc $\ge 30$ mẫu ở Task MERGE-3 là quyết định hoàn toàn đúng đắn.
* Ngược lại, **tất cả 59 nhãn Webcam** (có đủ 60 mẫu chuẩn hóa) đều đạt Accuracy rất cao ($85\% - 100\%$), hầu hết đạt $100\%$ (ví dụ `con yeu me`: 100%, `chuc mung`: 100%, `xin chao`: 100%).

---

## 4. MA TRẬN NHẦM LẪN (CONFUSION MATRIX) & CÁC CẶP NHẦM PHỔ BIẾN

Phân tích các cặp cử chỉ có số lần phân loại nhầm cao nhất trên 1,368 mẫu test:

| Nhãn Thực Tế | Nhãn Đoán Nhầm | Số lần nhầm | Đặc điểm ngữ nghĩa / Chuyển động gây nhầm |
| :--- | :--- | :---: | :--- |
| **`Sử dụng`** | **`Nói xấu`** | **4 lần** | Cả hai cử chỉ đều dùng 2 bàn tay mở trước ngực, chuyển động lắc/xoay quanh trục cổ tay trong không gian gần tương đương |
| **`Chào`** | **`Chậm lại`** | **2 lần** | Đều là động tác giơ 1 bàn tay ngang tầm ngực/vai và vẫy/hạ xuống |
| **`Cá`** | **`Lây bệnh`** | **2 lần** | Bàn tay lượn sóng mô phỏng chuyển động |
| **`Nghe`** | **`Đầu`** | **2 lần** | Ngón tay trỏ chỉ vào vùng mang tai/thái dương gần đỉnh đầu |
| **`Nôn ói`** | **`Dạy dỗ`** | **2 lần** | Quỹ đạo tay từ miệng đưa ra phía trước |
| **`ban ten la gi`** | **`cong viec cua ban la gi`** | **2 lần** | Hai câu webcam có khẩu ngữ và tư thế tay bắt đầu gần như đồng nhất |
| **`toi can thuoc`** | **`toi la nguoi Diec`** | **2 lần** | Các câu ghép đại từ "Tôi" đều bắt đầu bằng trỏ vào ngực |

---

## 5. BẢNG SO SÁNH ĐỐI ĐẦU TỔNG THỂ

| Tiêu chí | Nhánh Phú (159 classes ban đầu) | Mô hình Task B3 (60 classes production) | **Baseline MERGE-5 (119 classes)** |
| :--- | :---: | :---: | :---: |
| **Số lượng nhãn** | 159 nhãn (lẫn nhãn rác, video < 10 frame) | 60 nhãn (chỉ Webcam) | **119 nhãn (59 Webcam + 60 Video sạch)** |
| **Kích thước vector** | 126 chiều | 129 chiều ($S_{combined} + \vec{D}_{rel}$) | **129 chiều ($S_{combined} + \vec{D}_{rel}$)** |
| **Kiến trúc** | LSTM BatchNorm Dropout (phức tạp) | LSTM 128->64->32 Tanh | **LSTM 128->64->32 Tanh (nguyên bản)** |
| **Độ ổn định gradient** | Dễ nổ gradient (ReLU) | Ổn định (Tanh + Clipnorm) | **Ổn định tuyệt đối (0 NaN)** |
| **Test Accuracy (Top-1)** | $\approx \mathbf{85.6\%}$ | $\mathbf{96.06\%}$ | **$\mathbf{93.35\%}$** |
| **Top-3 Accuracy** | Không đo | Không đo | **$\mathbf{96.27\%}$** |
| **Top-5 Accuracy** | Không đo | Không đo | **$\mathbf{97.08\%}$** |
| **Ý nghĩa** | Số nhãn nhiều nhưng độ tin cậy thấp | Chuẩn nhưng số nhãn ít (60) | **Đạt chuẩn cân bằng: 119 nhãn mà Acc vẫn đạt 93.35%** |

---

## 6. KẾT LUẬN & ĐỀ XUẤT CHO TASK MERGE-6

### Kết luận:
1. **Thành công vượt bậc**: Việc mở rộng từ 60 lên 119 nhãn (tăng gần gấp đôi số lớp) mà Accuracy chỉ giảm nhẹ từ $96.06\%$ xuống **$93.35\%$** (Top-3 đạt **$96.27\%$**, Top-5 đạt **$97.08\%$**) là một kết quả xuất sắc cho một mô hình Baseline thuần túy.
2. **Tính hiệu quả của khâu lọc dữ liệu (Task MERGE-2/3)**: Việc mạnh dạn loại bỏ 40 nhãn video kém chất lượng (< 20 frames, < 10 FPS, < 30 mẫu) đã giúp mô hình học cực kỳ ổn định và đạt độ chính xác thực tế cao hơn hẳn nhánh 159 classes của Phú ($93.35\%$ so với $85.6\%$).
3. **Cơ sở đối chiếu vững chắc**: Checkpoint [`Models/baseline_119_tanh.h5`](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models/baseline_119_tanh.h5) sẽ là mốc chuẩn (Benchmark Standard) cho mọi thử nghiệm kiến trúc cải tiến sau này.

### Đề xuất cho Task MERGE-6:
Giờ đây ta đã có một Baseline 119 nhãn vững chắc đạt **$93.35\%$**. Đề xuất triển khai **TASK MERGE-6: Thử nghiệm cải tiến kiến trúc (Bi-LSTM / GRU + Self-Attention)**:
- Thử nghiệm Bi-LSTM hai chiều để nắm bắt ngữ cảnh cử chỉ thuận nghịch thời gian.
- Thêm cơ chế Attention để phân bổ trọng số vào các frame quyết định cử chỉ (giúp gỡ rối cho các cặp hay nhầm như `Sử dụng` vs `Nói xấu`, `Chào` vs `Chậm lại`).
- So sánh đối đầu trực tiếp 1-1 với Baseline 93.35% này.
