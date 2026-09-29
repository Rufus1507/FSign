# TASK MERGE-1: BÁO CÁO AUDIT TOÀN DIỆN DATASET 159 NHÃN TRƯỚC KHI MERGE
*(Thực hiện theo tiêu chuẩn kiểm định độc lập — Tương tự phương pháp Task A6)*

> **Thông tin kiểm định**:
> - **Ngày thực hiện**: 27/09/2026
> - **Phạm vi kiểm tra**: Toàn bộ 159 nhãn định nghĩa trong `Models/label_map_159.json` và tập huấn luyện 7,415 sequences từ nhánh Phú.
> - **Mục tiêu**: Đánh giá số lượng mẫu thực tế, nguồn gốc thu thập, tính toàn vẹn của shape, mức độ mất cân bằng lớp (class imbalance), và xác định các nhãn có nguy cơ cao để đưa ra khuyến nghị xử lý trước khi hợp nhất pipeline A-E.

---

## 1. TỔNG QUAN PHÂN BỐ & NGUỒN GỐC DỮ LIỆU (EXECUTIVE SUMMARY)

Dữ liệu của 159 nhãn được tạo thành từ 2 nguồn dữ liệu hoàn toàn khác biệt về phương pháp thu thập và đặc tính vật lý:

| Nguồn dữ liệu | Số nhãn | Số sequence | Tỷ lệ sequence | Nguồn gốc thu thập | Cơ chế xử lý frame | Mức độ cân bằng |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Dữ liệu Cũ (Webcam)** | **59 nhãn** | **3,540** | **47.7%** | Quay trực tiếp qua webcam (`CollectData.py`) | Quay chuẩn 60 frame liên tục | **Cân bằng tuyệt đối 100%** (Đúng 60 seq / nhãn) |
| **Dữ liệu Mới (`dataset/train/`)** | **100 nhãn** | **3,875** | **52.3%** | Video clip MP4 trích xuất qua `extract_all_new_dataset.py` | Linear Interpolation từ 16–30 fps về 60 fps | **Mất cân bằng nghiêm trọng** (6 đến 74 seq / nhãn) |
| **Tổng cộng hợp nhất** | **159 nhãn** | **7,415** | **100.0%** | Hỗn hợp Webcam + Video dataset | Cùng format `(60, 126)` | Lệch giữa max/min lên tới **12.33 lần** |

> *Ghi chú về nhãn `cam on`*: Trong 60 nhãn webcam ban đầu có `cam on` (60 mẫu). Khi tích hợp 100 nhãn mới từ video, nhãn `cam on` cũ được thay thế bằng nhãn có dấu `Cảm ơn` (62 video mới), đưa tổng số nhãn webcam giữ lại là 59 nhãn ($59 \times 60 = 3,540$ mẫu) + 100 nhãn mới ($3,875$ mẫu) = đúng **7,415 mẫu**.

---

## 2. KIỂM TRA CHẤT LƯỢNG SHAPE & FRAME TỪNG SEQUENCE

1. **Shape đầu ra quy chuẩn**:
   - Định dạng: `(60, 126)` tương ứng với $60 \text{ frames} \times [21 \times 3 \text{ (tay trái)} + 21 \times 3 \text{ (tay phải)}]$.
   - Cả 7,415 sequences đều đạt shape đầu vào `(60, 126)`.
2. **Sự khác biệt bản chất giữa 2 nguồn dữ liệu**:
   - **Nhóm 59 nhãn Webcam**: 100% frame là tín hiệu vật lý thực tế được đọc từ luồng webcam thời gian thực ở tốc độ ~30 FPS. Độ mượt và động học bàn tay liên tục tự nhiên.
   - **Nhóm 100 nhãn Video mới**:
     - Video gốc có thời lượng trung bình chỉ **`2.4 giây`** (ngắn nhất `0.8s`, dài nhất `6.5s`).
     - Số frame gốc trung bình chỉ đạt **`30.0 frames`** (có video chỉ 16 frames).
     - FPS trung bình của video nguồn chỉ đạt **`13.0 fps`** (nhiều video thấp tới 4.6 – 10 fps).
     - **Hệ quả của Resample Tuyến tính (`resample_sequence`)**: Để đưa chuỗi 16–30 frames về đúng 60 frames, thuật toán phải nội suy nhân đôi/nhân ba các tọa độ. Do đó, các frame sinh ra mang tính chất làm mịn nhân tạo, chuyển động tay giữa các frame có thể bị chậm hoặc thiếu các bước chuyển tiếp vi mô (micro-movements) so với dữ liệu webcam thực.

---

## 3. PHÂN BỐ CHI TIẾT THEO CẤP ĐỘ CÂN BẰNG DỮ LIỆU

Dựa trên số lượng mẫu thực tế, 159 nhãn được phân làm 4 nhóm rõ rệt:

```
[Nhóm 1: Chuẩn Cao (≥50 mẫu)] ────── 102 nhãn (64.2%) ─► 59 nhãn Webcam (60 mẫu) + 43 nhãn Video (50-74 mẫu)
[Nhóm 2: Đạt Chuẩn (30-49 mẫu)] ────  20 nhãn (12.6%) ─► 20 nhãn Video (30-49 mẫu)
[Nhóm 3: Thiếu Vừa (10-29 mẫu)] ────  18 nhãn (11.3%) ─► 18 nhãn Video (11-26 mẫu)
[Nhóm 4: BÁO ĐỘNG ĐỎ (<10 mẫu)] ────  19 nhãn (11.9%) ─► 19 nhãn Video (CHỈ 6-8 MẪU)
```

### 3.1. Danh sách 19 Nhãn Thiếu Dữ Liệu Nghiêm Trọng (Báo Động Đỏ — Dưới 10 Sequences)

Đây là nhóm nhãn có nguy cơ gây hại lớn nhất cho mô hình (tương tự như nhãn `xin loi` trước đây):

| STT | Tên nhãn | Số sequence | Nguồn gốc | Tỷ lệ Test (20%) | Nguy cơ kỹ thuật |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **Bệnh nhân** | **6** | Video | **1 mẫu** | Train chỉ 5 mẫu, Test 1 mẫu $\rightarrow$ Nguy cơ overfit 100% hoặc F1 sụp đổ |
| 2 | **Cách ly** | **6** | Video | **1 mẫu** | Train 5 mẫu, không đủ biến thiên cử động |
| 3 | **Cơ thể** | **6** | Video | **1 mẫu** | Dữ liệu quá nghèo nàn, dễ nhầm lẫn với các nhãn chỉ thân |
| 4 | **Ho** | **6** | Video | **1 mẫu** | Cử động tay che miệng ngắn, dễ dính occlusion |
| 5 | **Phục hồi** | **6** | Video | **1 mẫu** | Thời lượng video gốc dao động bất thường (1.67s - 6.54s) |
| 6 | **Sốt** | **6** | Video | **1 mẫu** | Cử động áp trán, train 5 mẫu |
| 7 | **Xe máy** | **6** | Video | **1 mẫu** | Cử động 2 tay lái xe, train 5 mẫu |
| 8 | **Ủng hộ** | **6** | Video | **1 mẫu** | Train 5 mẫu, cực kỳ dễ học vẹt |
| 9 | **Ban đêm** | **7** | Video | **1 mẫu** | Cử động tối/khép, chỉ 7 mẫu |
| 10 | **Chấp nhận** | **7** | Video | **1 mẫu** | Chỉ 7 mẫu |
| 11 | **Cần** | **7** | Video | **1 mẫu** | Cử động ngắn, chỉ 7 mẫu |
| 12 | **Học sinh** | **7** | Video | **1 mẫu** | Chỉ 7 mẫu |
| 13 | **Lo lắng** | **7** | Video | **1 mẫu** | Chỉ 7 mẫu |
| 14 | **Ngón tay** | **7** | Video | **1 mẫu** | Cử động vi mô ngón tay, chỉ 7 mẫu |
| 15 | **Nhầm** | **7** | Video | **1 mẫu** | Chỉ 7 mẫu |
| 16 | **Virus** | **7** | Video | **1 mẫu** | Chỉ 7 mẫu |
| 17 | **Bàn tay** | **8** | Video | **1 mẫu** | Chỉ 8 mẫu |
| 18 | **Thương** | **8** | Video | **1 mẫu** | Chỉ 8 mẫu |
| 19 | **Vâng lời** | **8** | Video | **1 mẫu** | Chỉ 8 mẫu |

> **Phân tích rủi ro học máy**:
> Với 6 mẫu, khi thực hiện Stratified Split 80/20: Tập Train nhận 5 mẫu, tập Test nhận đúng **1 mẫu duy nhất**.
> Nếu mẫu test này bị nhận diện sai, Accuracy của lớp đó lập tức rơi về **0.0%**. Ngược lại nếu đúng, Accuracy nhảy lên **100%**. Con số này mang tính may rủi thống kê, không có giá trị nghiệm thu khoa học trong luận văn.

---

### 3.2. Danh sách 18 Nhãn Thiếu Dữ Liệu Mức Trung Bình (10 đến 29 Sequences)

| STT | Tên nhãn | Số sequence | Nguồn gốc | Đánh giá |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Khu cách ly** | 11 | Video | Thiếu dữ liệu, train ~9, test ~2 |
| 2 | **Khẩu trang** | 11 | Video | Thiếu dữ liệu, train ~9, test ~2 |
| 3 | **Thích** | 11 | Video | Thiếu dữ liệu, train ~9, test ~2 |
| 4 | **Tập luyện** | 11 | Video | Thiếu dữ liệu, train ~9, test ~2 |
| 5 | **Chúng ta** | 14 | Video | Mức chấp nhận yếu |
| 6 | **Rẽ phải** | 15 | Video | Mức chấp nhận yếu |
| 7 | **Xe đạp** | 16 | Video | Mức chấp nhận yếu |
| 8 | **Bộ y tế** | 17 | Video | Mức chấp nhận yếu |
| 9 | **Bạn thân** | 18 | Video | Mức chấp nhận yếu |
| 10 | **Đẹp** | 18 | Video | Mức chấp nhận yếu |
| 11 | **Thăm** | 19 | Video | Mức chấp nhận yếu |
| 12 | **Xin lỗi** | 19 | Video | Đã có 19 mẫu mới từ video (khắc phục 1 mẫu YouTube cũ) |
| 13 | **Có thể** | 22 | Video | Dưới 25 mẫu |
| 14 | **Ban ngày** | 23 | Video | Dưới 25 mẫu |
| 15 | **Thất lạc** | 24 | Video | Dưới 25 mẫu |
| 16 | **Tối** | 24 | Video | Dưới 25 mẫu |
| 17 | **Uống** | 25 | Video | Ngưỡng 25 mẫu |
| 18 | **Hôm nay** | 26 | Video | Ngưỡng 26 mẫu |

---

## 4. BẢNG PHÂN BỐ TOÀN DIỆN 159 NHÃN (THEO THỨ TỰ BẢNG MÃ LABEL_MAP_159.JSON)

| ID | Tên Nhãn trong Dataset | Tên Hiển Thị Tiếng Việt | Nguồn Gốc | Số Mẫu | Trạng Thái Shape | Nhóm Đánh Giá |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 0 | `An ủi` | An ủi | Video | 32 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 1 | `Ban ngày` | Ban ngày | Video | 23 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 2 | `Ban đêm` | Ban đêm | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 3 | `Biết` | Biết | Video | 67 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 4 | `Biếu tặng` | Biếu tặng | Video | 51 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 5 | `Bàn tay` | Bàn tay | Video | 8 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 6 | `Băn khoăn` | Băn khoăn | Video | 62 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 7 | `Bạn thân` | Bạn thân | Video | 18 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 8 | `Bế mạc` | Bế mạc | Video | 35 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 9 | `Bệnh nhân` | Bệnh nhân | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 10 | `Bệnh viện` | Bệnh viện | Video | 48 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 11 | `Bộ y tế` | Bộ y tế | Video | 17 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 12 | `Chiều` | Chiều | Video | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 13 | `Cho` | Cho | Video | 49 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 14 | `Chào` | Chào | Video | 34 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 15 | `Chân` | Chân | Video | 62 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 16 | `Chúng ta` | Chúng ta | Video | 14 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 17 | `Chạy` | Chạy | Video | 67 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 18 | `Chấp nhận` | Chấp nhận | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 19 | `Chậm lại` | Chậm lại | Video | 69 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 20 | `Con gấu` | Con gấu | Video | 57 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 21 | `Cá` | Cá | Video | 58 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 22 | `Cách ly` | Cách ly | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 23 | `Cám dỗ` | Cám dỗ | Video | 68 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 24 | `Có thể` | Có thể | Video | 22 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 25 | `Cơ thể` | Cơ thể | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 26 | `Cảm ơn` | Cảm ơn | Video | 62 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 27 | `Cần` | Cần | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 28 | `Cứu` | Cứu | Video | 53 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 29 | `Dạy dỗ` | Dạy dỗ | Video | 42 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 30 | `Dễ` | Dễ | Video | 51 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 31 | `Ghét` | Ghét | Video | 32 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 32 | `Giúp` | Giúp | Video | 45 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 33 | `Ho` | Ho | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 34 | `Hy sinh` | Hy sinh | Video | 47 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 35 | `Hâm mộ` | Hâm mộ | Video | 50 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 36 | `Hôm nay` | Hôm nay | Video | 26 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 37 | `Họ` | Họ | Video | 48 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 38 | `Học sinh` | Học sinh | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 39 | `Khai báo` | Khai báo | Video | 56 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 40 | `Khu cách ly` | Khu cách ly | Video | 11 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 41 | `Khóc` | Khóc | Video | 67 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 42 | `Khẩu trang` | Khẩu trang | Video | 11 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 43 | `Kết hôn` | Kết hôn | Video | 31 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 44 | `Lo lắng` | Lo lắng | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 45 | `Lây bệnh` | Lây bệnh | Video | 50 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 46 | `Mua` | Mua | Video | 35 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 47 | `Mời vào` | Mời vào | Video | 56 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 48 | `Nghe` | Nghe | Video | 68 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 49 | `Nghỉ ngơi` | Nghỉ ngơi | Video | 67 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 50 | `Ngón tay` | Ngón tay | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 51 | `Nhà` | Nhà | Video | 71 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 52 | `Nhìn` | Nhìn | Video | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 53 | `Nhầm` | Nhầm | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 54 | `Nhớ` | Nhớ | Video | 57 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 55 | `Nói` | Nói | Video | 30 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 56 | `Nói xấu` | Nói xấu | Video | 68 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 57 | `Nôn ói` | Nôn ói | Video | 44 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 58 | `Nặng` | Nặng | Video | 48 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 59 | `Phía sau` | Phía sau | Video | 67 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 60 | `Phạt` | Phạt | Video | 37 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 61 | `Phỏng vấn` | Phỏng vấn | Video | 45 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 62 | `Phục hồi` | Phục hồi | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 63 | `Rau` | Rau | Video | 72 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 64 | `Rẽ phải` | Rẽ phải | Video | 15 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 65 | `Rẽ trái` | Rẽ trái | Video | 65 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 66 | `San sẻ` | San sẻ | Video | 46 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 67 | `Sốt` | Sốt | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 68 | `Sử dụng` | Sử dụng | Video | 59 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 69 | `Thích` | Thích | Video | 11 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 70 | `Thăm` | Thăm | Video | 19 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 71 | `Thương` | Thương | Video | 8 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 72 | `Thất lạc` | Thất lạc | Video | 24 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 73 | `Thức dậy` | Thức dậy | Video | 67 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 74 | `Thức ăn` | Thức ăn | Video | 56 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 75 | `Trưa` | Trưa | Video | 57 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 76 | `Trường học` | Trường học | Video | 74 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 77 | `Tôi` | Tôi | Video | 52 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 78 | `Tập luyện` | Tập luyện | Video | 11 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 79 | `Tối` | Tối | Video | 24 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 80 | `Uống` | Uống | Video | 25 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 81 | `Virus` | Virus | Video | 7 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 82 | `Vâng lời` | Vâng lời | Video | 8 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 83 | `Xa` | Xa | Video | 62 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 84 | `Xe máy` | Xe máy | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |
| 85 | `Xe đạp` | Xe đạp | Video | 16 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 86 | `Xin lỗi` | Xin lỗi | Video | 19 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 87 | `Xin phép` | Xin phép | Video | 38 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 88 | `Xuất viện` | Xuất viện | Video | 48 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 89 | `Xúc động` | Xúc động | Video | 59 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 90 | `ban dang lam gi` | Bạn đang làm gì? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 91 | `ban di dau the` | Bạn đi đâu thế? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 92 | `ban hieu ngon ngu ky hieu khong` | Bạn hiểu ngôn ngữ ký hiệu không? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 93 | `ban hoc lop may` | Bạn học lớp mấy? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 94 | `ban khoe khong` | Bạn khỏe không? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 95 | `ban muon gio roi` | Bạn muộn giờ rồi | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 96 | `ban phai canh giac` | Bạn phải cảnh giác | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 97 | `ban ten la gi` | Bạn tên là gì? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 98 | `ban tien bo day` | Bạn tiến bộ đấy | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 99 | `ban trong cau co the` | Bạn trông cáu cọ thế | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 100 | `bo me toi cung la nguoi Diec` | Bố mẹ tôi cũng là người Điếc | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 101 | `cai nay bao nhieu tien` | Cái này bao nhiêu tiền? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 102 | `cai nay la cai gi` | Cái này là cái gì? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 103 | `cap cuu` | Cấp cứu | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 104 | `chuc mung` | Chúc mừng | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 105 | `chung toi giao tiep voi nhau bang ngon ngu ky hieu` | Chúng tôi giao tiếp với nhau bằng NNKH | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 106 | `con yeu me` | Con yêu mẹ | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 107 | `cong viec cua ban la gi` | Công việc của bạn là gì? | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 108 | `hen gap lai cac ban` | Hẹn gặp lại các bạn | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 109 | `mon nay khong ngon` | Món này không ngon | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 110 | `toi bi chong mat` | Tôi bị chóng mặt | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 111 | `toi bi cuop` | Tôi bị cướp | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 112 | `toi bi dau dau` | Tôi bị đau đầu | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 113 | `toi bi dau hong` | Tôi bị đau họng | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 114 | `toi bi ket xe` | Tôi bị kẹt xe | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 115 | `toi bi lac` | Tôi bị lạc | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 116 | `toi bi phan biet doi xu` | Tôi bị phân biệt đối xử | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 117 | `toi cam thay rat hoi hop` | Tôi cảm thấy rất hồi hộp | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 118 | `toi cam thay rat vui` | Tôi cảm thấy rất vui | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 119 | `toi can an sang` | Tôi cần ăn sáng | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 120 | `toi can di ve sinh` | Tôi cần đi vệ sinh | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 121 | `toi can gap bac si` | Tôi cần gặp bác sĩ | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 122 | `toi can phien dich` | Tôi cần phiên dịch | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 123 | `toi can thuoc` | Tôi cần thuốc | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 124 | `toi dang an sang` | Tôi đang ăn sáng | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 125 | `toi dang buon` | Tôi đang buồn | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 126 | `toi dang o ben xe` | Tôi đang ở bến xe | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 127 | `toi dang o cong vien` | Tôi đang ở công viên | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 128 | `toi dang phai cach ly` | Tôi đang phải cách ly | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 129 | `toi dang phan van` | Tôi đang phân vân | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 130 | `toi di sieu thi` | Tôi đi siêu thị | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 131 | `toi di toi Ha Noi` | Tôi đi tới Hà Nội | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 132 | `toi doc kem` | Tôi đọc kém | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 133 | `toi khoi benh roi` | Tôi khỏi bệnh rồi | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 134 | `toi khong dem theo tien` | Tôi không đem theo tiền | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 135 | `toi khong hieu` | Tôi không hiểu | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 136 | `toi khong quan tam` | Tôi không quan tâm | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 137 | `toi la hoc sinh` | Tôi là học sinh | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 138 | `toi la nguoi Diec` | Tôi là người Điếc | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 139 | `toi la tho theu` | Tôi là thợ thêu | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 140 | `toi lam viec o cua hang` | Tôi làm việc ở cửa hàng | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 141 | `toi nham dia chi` | Tôi nhầm địa chỉ | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 142 | `toi song o Ha Noi` | Tôi sống ở Hà Nội | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 143 | `toi thay doi bung` | Tôi thấy đói bụng | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 144 | `toi thay nho ban` | Tôi thấy nhớ bạn | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 145 | `toi thich an mi` | Tôi thích ăn mì | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 146 | `toi thich phim truyen` | Tôi thích phim truyện | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 147 | `toi viet kem` | Tôi viết kém | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 148 | `xin chao` | Xin chào | Webcam | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 149 | `Áp dụng` | Áp dụng | Video | 71 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 150 | `Ô tô` | Ô tô | Video | 47 | `(60, 126)` OK | Đạt chuẩn (30-49) |
| 151 | `Ăn` | Ăn | Video | 60 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 152 | `Ăn mừng` | Ăn mừng | Video | 62 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 153 | `Đi` | Đi | Video | 62 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 154 | `Đâu` | Đâu | Video | 53 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 155 | `Đầu` | Đầu | Video | 54 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 156 | `Đẹp` | Đẹp | Video | 18 | `(60, 126)` OK | Thiếu vừa (10-29) |
| 157 | `Đồng ý` | Đồng ý | Video | 53 | `(60, 126)` OK | Chuẩn cao (≥50) |
| 158 | `Ủng hộ` | Ủng hộ | Video | 6 | `(60, 126)` OK | **Báo động đỏ (<10)** |

---

## 5. ĐÁNH GIÁ ĐỐI CHIẾU VỚI BÀI HỌC "XIN LOI" Ở TASK A6 & ĐỀ XUẤT XỬ LÝ

### 5.1. Nhìn lại tiền lệ nhãn `xin loi` ở Task A6
- Ở Task A6, tập dataset cũ từng có nhãn `xin loi` chỉ có **1 sample** từ YouTube, trong khi 60 nhãn còn lại có đúng 60 samples. 
- Lệch phân phối 60:1 dẫn đến tình trạng mô hình dự đoán `xin loi` với độ tin cậy cực thấp, gây sai số hệ thống và làm hỏng tính toàn vẹn của ma trận nhầm lẫn (confusion matrix).
- **Cách xử lý tại Task A6**: Đã quyết định **loại bỏ hoàn toàn mẫu ngoại lai** này để giữ tập 60 nhãn cân bằng tuyệt đối ($60 \times 60 = 3,600$ mẫu). Quyết định này giúp các thí nghiệm Hạng mục B (ReLU vs Tanh, V1 vs V2) đạt độ ổn định và độ tin cậy khoa học cao nhất.

### 5.2. Hai Kịch Bản Đề Xuất Cho Việc Hợp Nhất (Merge Decision)

| Tiêu chí | Kịch bản 1: **Lọc Chuẩn Khoa Học (Khuyến nghị cho Luận văn)** | Kịch bản 2: **Giữ Toàn Bộ 159 Nhãn (Thực dụng / Demo)** |
| :--- | :--- | :--- |
| **Quy tắc lọc** | **Loại bỏ các nhãn dưới ngưỡng $\ge 30$ mẫu** (hoặc loại 19 nhãn $<10$ mẫu) | Giữ nguyên 100% không loại nhãn nào |
| **Số lượng nhãn giữ lại** | **122 nhãn** (nếu lọc $\ge 30$) hoặc **140 nhãn** (nếu loại $<10$) | **Đủ 159 nhãn** |
| **Tổng số mẫu** | ~**6,900+ mẫu** (chỉ loại bỏ ~500 mẫu nghèo nàn) | Đúng **7,415 mẫu** |
| **Ưu điểm** | - Loại bỏ hoàn toàn rủi ro 1 mẫu test may rủi.<br>- F1-Score giữa các nhãn đồng đều, minh bạch số liệu.<br>- Đáp ứng chuẩn mực thẩm định của hội đồng luận văn. | - Tối đa hóa số lượng từ vựng nhận diện (159 từ/câu).<br>- Giữ nguyên vẹn thành quả dataset của nhánh Phú. |
| **Nhược điểm** | Mất đi một số từ thông dụng nhưng ít video (như `Ho`, `Cơ thể`, `Cách ly`). | - 19 nhãn có mẫu test chỉ 1 sample $\rightarrow$ F1-score có độ lệch chuẩn cao.<br>- Bắt buộc phải áp dụng Data Augmentation hoặc Class Weights để tránh mô hình bỏ quên các lớp ít mẫu. |
| **Khuyến nghị áp dụng** | **Ưu tiên hàng đầu khi viết báo cáo Luận văn chính thức**. | Dùng cho phiên bản demo sản phẩm ứng dụng. |

---
*Báo cáo được hoàn thành trong khuôn khổ Task MERGE-1 của dự án FSign.*
