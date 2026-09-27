# Báo cáo Task C1: Audit Toàn bộ Model File Hiện có trong Repo

> **Mục tiêu**: Kiểm tra chéo toàn bộ model files (`.h5`, `.tflite`) trong thư mục `Models/` với danh sách nhãn tương ứng nhằm phát hiện triệt để mọi trường hợp lệch số lớp trước khi thiết lập assert chặn cứng ở Task C2.

---

### 1. Bảng Tổng hợp Đối chiếu Toàn bộ Model trong `Models/`

| Tên file model | output_shape (số lớp) | Label file gắn kèm | Số lớp trong label file | Khớp hay không |
|---|---|---|---|---|
| `model_normalized_v1.tflite` | `[1, 61]` (61 lớp) | `Data_normalized/` (qua `RunModel.py`) | 60 | **LỆCH NGUY HIỂM** (Model: 61 vs Label: 60) |
| `model_normalized_v1.h5` | `(None, 61)` (61 lớp) | `label_map.json` (qua `train_model.py`) | 60 | **LỆCH** (Model: 61 vs Label: 60) |
| `model_normalized_v1_backup_run1.h5` | `(None, 61)` (61 lớp) | `label_map.json` / `Data_normalized/` | 60 | **LỆCH** (Model: 61 vs Label: 60) |
| `fsign_159classes.h5` | `(None, 159)` (159 lớp) | `Models/label_map_159.json` | 159 | **KHỚP** (Model nhánh `feature/Phu`, 126 chiều) |
| `model_b2_relu.h5` | `(None, 60)` (60 lớp) | `Data_normalized/` (Task B2 ReLU) | 60 | **KHỚP 100%** |
| `model_b2_tanh.h5` | `(None, 60)` (60 lớp) | `Data_normalized/` (Task B2 Tanh Baseline) | 60 | **KHỚP 100%** |
| `model_b3_v2_descending.h5` | `(None, 60)` (60 lớp) | `Data_normalized/` (Task B3 Tanh V2) | 60 | **KHỚP 100%** |

---

### 2. Phát hiện Trọng yếu: Lệch Lớp Nghiêm trọng ở `model_normalized_v1.tflite` (Model Inference Chính)

1. **Vấn đề cốt tử được phát hiện**:
   - Tệp [RunModel.py](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/RunModel.py) (chương trình chạy camera thời gian thực) hiện đang nạp [Models/model_normalized_v1.tflite](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models).
   - Tuy nhiên, file TFLite này được convert từ thời kỳ trước Hạng mục A, có **`output_shape = [1, 61]`** (61 classes, 126 chiều cũ).
   - Trong khi đó, `RunModel.py` gọi hàm `load_actions()` để đọc trực tiếp thư mục `Data_normalized/` (đã được chốt chuẩn hóa về **60 nhãn**, 129 chiều ở Task A6.1).
   - **Hậu quả**:
     - Mảng nhãn `actions` chỉ có 60 phần tử (chỉ số $0 \rightarrow 59$), nhưng model lại xuất ra vector xác suất 61 phần tử (chỉ số $0 \rightarrow 60$).
     - Khi model dự đoán lớp index `60`, chương trình sẽ ngay lập tức ném ngoại lệ:
       ```
       IndexError: index 60 is out of bounds for axis 0 with size 60
       ```
       dẫn đến **sập camera thời gian thực (crash app)** ngay khi người dùng đưa tay thực hiện động tác!
     - Chưa kể `model_normalized_v1.tflite` vẫn mong đợi đầu vào 126 chiều (thay vì 129 chiều $S_{\text{combined}}$ mới), gây xung đột kích thước đầu vào.

2. **Hai model Keras cũ (`model_normalized_v1.h5` và `model_normalized_v1_backup_run1.h5`)**:
   - Cả 2 đều có `output_shape = (None, 61)` và `input_shape = (None, 60, 126)`.
   - Đây là các checkpoint huấn luyện từ thời điểm cũ trước Task A6.1 (vẫn còn dính 1 sample `xin loi` từ YouTube và chuẩn hóa 126 chiều cũ). Hiện không còn khớp với codebase mới (`model_def.py` và `label_map.json` đã chốt 60 lớp, 129 chiều).

3. **Mô hình 159 lớp (`fsign_159classes.h5`)**:
   - Khớp 159 lớp với [label_map_159.json](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Models/label_map_159.json).
   - Đây là mô hình thử nghiệm riêng của nhánh `feature/Phu` cho bài toán 159 lớp, không thuộc luồng 60 lớp chuẩn webcam hiện tại.

4. **Các mô hình mới từ Hạng mục B (`model_b2_relu.h5`, `model_b2_tanh.h5`, `model_b3_v2_descending.h5`)**:
   - Đều có `input_shape = (None, 60, 129)` và `output_shape = (None, 60)`.
   - **Khớp chuẩn xác 100%** với tập dữ liệu [Data_normalized/](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/Data_normalized) và [label_map.json](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/label_map.json).

---

### 3. Tình trạng các Model Nằm Ngoài `Models/`

Toàn bộ 36 file model `.h5` và `.tflite` từ các giai đoạn nghiên cứu trước đây đã được cách ly hoàn toàn vào thư mục [Sign Language Translator/legacy/](file:///d:/Desktop/5/DPL302m/project/Sign-Language-Translator/Sign%20Language%20Translator/legacy) (`legacy/Models_old/`, `legacy/Structure/`, `legacy/backup/`, `legacy/release/`), không còn bất kỳ script hoạt động nào tham chiếu đến chúng.
