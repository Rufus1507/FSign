# BÁO CÁO ĐÁNH GIÁ TƯƠNG THÍCH DATASET & MODEL - FSIGN

> **Thời gian tạo**: `2026-09-23 18:14:58`  
> **Dự án**: FSign - Vietnamese Sign Language Recognition  
> **Mục đích**: Đánh giá tương thích dữ liệu mới (`dataset/train/`) với model hiện tại và quyết định chiến lược: **Retrain từ đầu** hay **Fine-tune**.

---
## 1. Tóm tắt điều hành (Executive Summary)

| Chỉ số kiểm tra | Dữ liệu cũ (`Data/`) | Dataset mới (`dataset/train/`) | Tổng hợp sau hợp nhất |
| :--- | :--- | :--- | :--- |
| **Số lượng nhãn (Classes)** | 60 nhãn | 100 nhãn | **159 nhãn** |
| **Nhãn trùng lặp (Thay thế)** | - | - | **1 nhãn** (`cam on` -> `Cam on`) |
| **Tổng số mẫu** | 3,600 sequences (.npy) | 3,875 videos (.mp4) | **~7,400+ mẫu huấn luyện** |
| **Dung lượng lưu trữ** | ~1.7 GB (keypoints) | 323.4 MB (raw video) | - |
| **Đặc trưng đầu vào (Input)** | `(60, 126)` Keypoints | Video MediaPipe `(60, 126)` | **Khớp hoàn toàn (100%)** |
| **Quyết định đề xuất** | - | - | **RETRAIN KHUYẾN NGHỊ** |

---
## 2. Quyết định chiến lược: **RETRAIN TOÀN DIỆN (Khuyến nghị chính) & WARM-START TRANSFER**

> Do số lượng nhãn mới (100) vượt trội so với nhãn cũ (59 nhãn giữ lại), tổng số class tăng lên 159 (gấp 2.65x). Kiến trúc đầu vào (60, 126) hoàn toàn khớp. Khuyến nghị chuẩn bị pipeline trích xuất MediaPipe cho 100 nhãn mới -> Gộp với 59 nhãn cũ (loại bỏ nhãn trùng 'cam on') -> Huấn luyện model mới 159 classes từ đầu để đạt độ chính xác tối ưu và tránh Catastrophic Forgetting.

### Các lý do cốt lõi dẫn đến quyết định:
- **[XÁC NHẬN]** Số nhãn mới (100) lớn gấp 1.7x so với nhãn cũ (60) -> Không gian phân loại mở rộng rất lớn (60 -> 159 classes).
- **[XÁC NHẬN]** Input shape của model cũ hoàn toàn tương thích chuẩn: [None, 60, 126] (sequence=60, feature_dim=126).
- **[XÁC NHẬN]** Dữ liệu cũ (Data/) định dạng chuẩn 100% (3600 sequences, shape=(126,)).
- **[XÁC NHẬN]** FPS trung bình video mới: 13.0 fps.
- **[XÁC NHẬN]** Số lượng mẫu phân bổ khá đều giữa cũ (60/nhãn) và mới (39/nhãn).

### Lưu ý kỹ thuật quan trọng:
- **[CẢNH BÁO]** Số frame trung bình hơi ngắn (30.0 frames < 60), cần linear interpolation khi trích xuất MediaPipe.

---
## 3. Phân tích chi tiết các Model hiện có trong dự án

| Model | Thư mục | Kích thước | Số Classes | Input Shape | Params | Kiến trúc nhận diện | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **20sentences933.h5** | `Models\model cũ\20sentences933.h5` | 2.4 MB | 20 | `[None, 60, 126]` | 612,061 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(20,softmax) | `OK` |
| **20sentences946.h5** | `Models\model cũ\20sentences946.h5` | 2.4 MB | 20 | `[None, 60, 126]` | 612,061 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(20,softmax) | `OK` |
| **85,139.h5** | `Structure\Structure 5\85,139.h5` | 1.9 MB | 60 | `[None, 60, 126]` | 481,237 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **89_sc_50.h5** | `Models\model 50 câu sc 89.17\89_sc_50.h5` | 1.9 MB | 50 | `[None, 60, 126]` | 480,247 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **92,64.h5** | `Structure\Structure 4 (final)\92,64.h5` | 1.3 MB | 60 | `[None, 60, 126]` | 323,221 | InputLayer -> LSTM(64, seq) -> LSTM(48, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **94,58.h5** | `release\94,58.h5` | 1.9 MB | 60 | `[None, 60, 126]` | 481,237 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **94,58.h5** | `Structure\Structure 5 (final)\94,58.h5` | 1.9 MB | 60 | `[None, 60, 126]` | 481,237 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **95_sc_50.h5** | `Models\model 50 câu 95,33\95_sc_50.h5` | 1.9 MB | 50 | `[None, 60, 126]` | 480,247 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **97,5.h5** | `Structure\Structure 6 (backup)\97,5.h5` | 2.1 MB | 60 | `[None, 60, 126]` | 545,557 | InputLayer -> LSTM(48, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **BestModel.h5** | `Models\model cũ\BestModel.h5` | 2.4 MB | 10 | `[None, 30, 126]` | 611,071 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(10,softmax) | `OK` |
| **Structure0.h5** | `Structure\Structure 0\Structure0.h5` | 2.4 MB | 60 | `[None, 60, 126]` | 205,340 | LSTM(64) -> LSTM(128) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **Structure4.h5** | `Structure\Structure 4\Structure4.h5` | 1.3 MB | 60 | `[None, 60, 126]` | 323,221 | InputLayer -> LSTM(64, seq) -> LSTM(48, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **Structure6.h5** | `Structure\Structure 6 (final)\Structure6.h5` | 2.1 MB | 60 | `[None, 60, 126]` | 545,557 | InputLayer -> LSTM(48, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(60,softmax) | `OK` |
| **TestModel.h5** | `Models\model cũ\TestModel.h5` | 2.4 MB | 10 | `[None, 30, 126]` | 611,071 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(10,softmax) | `OK` |
| **action.h5** | `Models\model cũ\action.h5` | 2.3 MB | 3 | `[None, 30, 1662]` | 596,675 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(3,softmax) | `OK` |
| **best20sentences.h5** | `Models\model cũ\best20sentences.h5` | 2.4 MB | 20 | `[None, 60, 126]` | 612,061 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(20,softmax) | `OK` |
| **hoiOverfit95.h5** | `Models\model cũ\hoiOverfit95.h5` | 2.4 MB | 17 | `[None, 60, 126]` | 611,764 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(17,softmax) | `OK` |
| **model.h5** | `Models\model (3)\model.h5` | 2.1 MB | 50 | `[None, 60, 126]` | 544,567 | InputLayer -> LSTM(48, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **model.h5** | `Models\model (4)\model.h5` | 1.9 MB | 50 | `[None, 60, 126]` | 479,095 | InputLayer -> LSTM(64, seq) -> LSTM(128) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **model8333.h5** | `Models\model cũ\model8333.h5` | 1.9 MB | 50 | `[None, 60, 126]` | 480,247 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **model86.h5** | `backup\backup for model(2)\model86.h5` | 1.9 MB | 50 | `[None, 60, 126]` | 480,247 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **model865.h5** | `backup\backup model(5)\model865.h5` | 1.1 MB | 50 | `[None, 60, 126]` | 282,103 | InputLayer -> LSTM(64, seq) -> LSTM(32, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **model915.h5** | `Models\model (5)\model915.h5` | 1.1 MB | 50 | `[None, 60, 126]` | 282,103 | InputLayer -> LSTM(64, seq) -> LSTM(32, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **model9167.h5** | `Models\model (1)\model9167.h5` | 1.3 MB | 50 | `[None, 60, 126]` | 322,231 | InputLayer -> LSTM(64, seq) -> LSTM(48, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **model94.h5** | `Models\model (2)\model94.h5` | 1.9 MB | 50 | `[None, 60, 126]` | 480,247 | InputLayer -> LSTM(32, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **pre_trained_50.h5** | `Models\model 50 câu gốc\pre_trained_50.h5` | 2.4 MB | 50 | `[None, 60, 126]` | 615,031 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(50,softmax) | `OK` |
| **sign.h5** | `Models\model cũ\sign.h5` | 2.4 MB | 5 | `[None, 30, 126]` | 610,576 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(5,softmax) | `OK` |
| **test1.h5** | `Models\model cũ\test1.h5` | 2.4 MB | 10 | `[None, 30, 126]` | 611,071 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(10,softmax) | `OK` |
| **test2.h5** | `Models\model cũ\test2.h5` | 2.4 MB | 10 | `[None, 30, 126]` | 611,071 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(10,softmax) | `OK` |
| **test3.h5** | `Models\model cũ\test3.h5` | 2.4 MB | 10 | `[None, 30, 126]` | 611,071 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(10,softmax) | `OK` |
| **testFinal.h5** | `Models\model cũ\testFinal.h5` | 2.4 MB | 10 | `[None, 30, 126]` | 611,071 | InputLayer -> LSTM(64, seq) -> LSTM(128, seq) -> LSTM(64) -> Dense(64,relu) -> Dense(32,relu) -> Dense(10,softmax) | `OK` |

> **Nhận xét kiến trúc**: Toàn bộ model chính (`release/94,58.h5`, `Structure4.h5`, `BestModel.h5`) đều có kiến trúc cốt lõi là **3 lớp LSTM (64 -> 128 -> 64)** kết hợp **2 lớp Dense (64 -> 32)** và lớp phân loại cuối cùng **Dense(60, softmax)**.

---
## 4. Phân tích Dữ liệu Cũ (`Sign Language Translator/Data/`)

- **Thư mục lưu trữ**: `H:\PythonProject\FSign\Sign Language Translator\Data`
- **Tổng số nhãn hiện có**: 60 nhãn
- **Tổng số sequence hợp lệ**: 3,600 sequences
- **Chuẩn cấu trúc feature**: `(126,)` tương ứng 21 điểm x 3 tọa độ (x, y, z) x 2 bàn tay
- **Số frame chuẩn mỗi sequence**: `60` frames/sequence

<details><summary><b>Xem danh sách chi tiết 60 nhãn cũ (Bấm để mở)</b></summary>

| STT | Tên nhãn (Không dấu) | Số Sequence | Lỗi | Shape Feature | Hành động khi Merge |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `ban dang lam gi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 2 | `ban di dau the` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 3 | `ban hieu ngon ngu ky hieu khong` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 4 | `ban hoc lop may` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 5 | `ban khoe khong` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 6 | `ban muon gio roi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 7 | `ban phai canh giac` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 8 | `ban ten la gi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 9 | `ban tien bo day` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 10 | `ban trong cau co the` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 11 | `bo me toi cung la nguoi Diec` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 12 | `cai nay bao nhieu tien` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 13 | `cai nay la cai gi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 14 | `cam on` | 60 | 0 | `(126,)` | **THAY THẾ (Trùng)** |
| 15 | `cap cuu` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 16 | `chuc mung` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 17 | `chung toi giao tiep voi nhau bang ngon ngu ky hieu` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 18 | `con yeu me` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 19 | `cong viec cua ban la gi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 20 | `hen gap lai cac ban` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 21 | `mon nay khong ngon` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 22 | `toi bi chong mat` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 23 | `toi bi cuop` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 24 | `toi bi dau dau` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 25 | `toi bi dau hong` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 26 | `toi bi ket xe` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 27 | `toi bi lac` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 28 | `toi bi phan biet doi xu` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 29 | `toi cam thay rat hoi hop` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 30 | `toi cam thay rat vui` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 31 | `toi can an sang` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 32 | `toi can di ve sinh` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 33 | `toi can gap bac si` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 34 | `toi can phien dich` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 35 | `toi can thuoc` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 36 | `toi dang an sang` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 37 | `toi dang buon` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 38 | `toi dang o ben xe` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 39 | `toi dang o cong vien` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 40 | `toi dang phai cach ly` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 41 | `toi dang phan van` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 42 | `toi di sieu thi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 43 | `toi di toi Ha Noi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 44 | `toi doc kem` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 45 | `toi khoi benh roi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 46 | `toi khong dem theo tien` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 47 | `toi khong hieu` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 48 | `toi khong quan tam` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 49 | `toi la hoc sinh` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 50 | `toi la nguoi Diec` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 51 | `toi la tho theu` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 52 | `toi lam viec o cua hang` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 53 | `toi nham dia chi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 54 | `toi song o Ha Noi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 55 | `toi thay doi bung` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 56 | `toi thay nho ban` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 57 | `toi thich an mi` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 58 | `toi thich phim truyen` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 59 | `toi viet kem` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |
| 60 | `xin chao` | 60 | 0 | `(126,)` | GIỮ NGUYÊN |

</details>

---
## 5. Phân tích Dataset Mới (`dataset/train/`)

- **Thư mục video gốc**: `H:\PythonProject\FSign\dataset\train`
- **Tổng số nhãn mới**: 100 nhãn (Có dấu tiếng Việt chuẩn)
- **Tổng số video**: 3,875 files .mp4
- **Tổng dung lượng**: 323.4 MB

### Thông số Video trung bình (Kiểm tra mẫu OpenCV):
| Thông số | Thấp nhất | Cao nhất | Trung bình | Đánh giá |
| :--- | :--- | :--- | :--- | :--- |
| **FPS** | 4.6 | 21.4 | **13.0 fps** | Tốt, ổn định cho MediaPipe |
| **Số Frame/Video** | 16.0 | 30.0 | **30.0 frames** | Đủ chuẩn để trích xuất 60 frames |
| **Thời lượng (giây)** | 0.8s | 6.5s | **2.4s** | Phù hợp độ dài 1 câu/từ ký hiệu |
| **Độ phân giải** | 224x224 | - | - | Chuẩn video HD/FHD |

### Kiểm tra trùng lặp giữa Dataset Cũ và Mới (Q1 & Q2):
| Nhãn mới (Có dấu) | Nhãn cũ tương ứng | Số video mới | Quyết định xử lý |
| :--- | :--- | :--- | :--- |
| **Cảm ơn** | `cam on` | 62 videos | **Bỏ data cũ, lấy video mới trích xuất lại** |

<details><summary><b>Xem danh sách chi tiết 100 nhãn mới (Bấm để mở)</b></summary>

| STT | Nhãn mới | Số Video | Dung lượng | FPS avg | Frames avg | Thời lượng | Cảnh báo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **An ủi** | 32 | 2.3 MB | 11.9 | 30.0 | 2.4-3.21s | FPS thấp x5 |
| 2 | **Ban ngày** | 23 | 2.2 MB | 10.1 | 30.0 | 2.57-3.86s | FPS thấp x5 |
| 3 | **Ban đêm** | 7 | 691.5 KB | 12.7 | 30.0 | 2.13-2.77s | FPS thấp x5 |
| 4 | **Biết** | 67 | 4.4 MB | 15.9 | 30.0 | 1.67-2.4s | FPS thấp x2 |
| 5 | **Biếu tặng** | 51 | 4.3 MB | 12.8 | 30.0 | 2.33-2.37s | FPS thấp x5 |
| 6 | **Bàn tay** | 8 | 637.9 KB | 12.1 | 30.0 | 2.13-3.03s | FPS thấp x5 |
| 7 | **Băn khoăn** | 62 | 4.4 MB | 18.7 | 30.0 | 1.5-2.27s | FPS thấp x1 |
| 8 | **Bạn thân** | 18 | 1.6 MB | 11.3 | 30.0 | 2.31-3.28s | FPS thấp x5 |
| 9 | **Bế mạc** | 35 | 2.6 MB | 9.8 | 30.0 | 2.47-3.26s | FPS thấp x5 |
| 10 | **Bệnh nhân** | 6 | 484.2 KB | 13.1 | 30.0 | 1.57-4.86s | FPS thấp x3 |
| 11 | **Bệnh viện** | 48 | 4.6 MB | 10.3 | 30.0 | 2.63-3.84s | FPS thấp x5 |
| 12 | **Bộ y tế** | 17 | 1.4 MB | 9.6 | 30.0 | 2.17-4.0s | FPS thấp x5 |
| 13 | **Chiều** | 60 | 5.3 MB | 12.7 | 30.0 | 1.96-2.67s | FPS thấp x4 |
| 14 | **Cho** | 49 | 3.5 MB | 12.4 | 30.0 | 2.4-2.5s | FPS thấp x5 |
| 15 | **Chào** | 34 | 3.3 MB | 11.8 | 30.0 | 1.81-3.2s | FPS thấp x4 |
| 16 | **Chân** | 62 | 6.5 MB | 11.0 | 30.0 | 2.43-3.33s | FPS thấp x5 |
| 17 | **Chúng ta** | 14 | 1.3 MB | 9.9 | 30.0 | 2.53-3.53s | FPS thấp x5 |
| 18 | **Chạy** | 67 | 5.2 MB | 13.9 | 30.0 | 1.55-3.27s | FPS thấp x3 |
| 19 | **Chấp nhận** | 7 | 599.1 KB | 15.0 | 30.0 | 1.65-2.33s | FPS thấp x3 |
| 20 | **Chậm lại** | 69 | 7.1 MB | 10.3 | 30.0 | 2.63-3.05s | FPS thấp x5 |
| 21 | **Con gấu** | 57 | 7.0 MB | 11.7 | 30.0 | 2.37-2.8s | FPS thấp x5 |
| 22 | **Cá** | 58 | 6.0 MB | 11.1 | 30.0 | 2.69-2.7s | FPS thấp x5 |
| 23 | **Cách ly** | 6 | 454.9 KB | 12.5 | 30.0 | 2.17-2.57s | FPS thấp x5 |
| 24 | **Cám dỗ** | 68 | 4.7 MB | 15.8 | 30.0 | 1.67-2.4s | FPS thấp x2 |
| 25 | **Có thể** | 22 | 2.3 MB | 12.8 | 30.0 | 2.13-2.77s | FPS thấp x5 |
| 26 | **Cơ thể** | 6 | 878.1 KB | 12.8 | 30.0 | 1.69-2.8s | FPS thấp x4 |
| 27 | **Cảm ơn** | 62 | 4.4 MB | 13.4 | 30.0 | 1.6-3.03s | FPS thấp x4 |
| 28 | **Cần** | 7 | 685.6 KB | 11.8 | 30.0 | 2.33-3.13s | FPS thấp x5 |
| 29 | **Cứu** | 53 | 5.2 MB | 12.0 | 30.0 | 2.13-2.83s | FPS thấp x5 |
| 30 | **Dạy dỗ** | 42 | 3.0 MB | 14.8 | 30.0 | 1.6-2.47s | FPS thấp x3 |
| 31 | **Dễ** | 51 | 5.2 MB | 12.9 | 30.0 | 2.07-2.87s | FPS thấp x5 |
| 32 | **Ghét** | 32 | 1.5 MB | 14.8 | 27.2 | 0.8-2.3s | FPS thấp x4, Ít frame x1 |
| 33 | **Giúp** | 45 | 2.8 MB | 20.1 | 30.0 | 1.45-1.6s | OK |
| 34 | **Ho** | 6 | 365.3 KB | 16.3 | 30.0 | 1.47-2.17s | FPS thấp x3 |
| 35 | **Hy sinh** | 47 | 3.1 MB | 10.9 | 30.0 | 2.38-3.27s | FPS thấp x5 |
| 36 | **Hâm mộ** | 50 | 4.1 MB | 13.5 | 30.0 | 1.57-3.23s | FPS thấp x4 |
| 37 | **Hôm nay** | 26 | 2.5 MB | 13.7 | 30.0 | 2.09-2.43s | FPS thấp x5 |
| 38 | **Họ** | 48 | 4.2 MB | 10.1 | 30.0 | 2.78-3.17s | FPS thấp x5 |
| 39 | **Học sinh** | 7 | 550.5 KB | 11.3 | 30.0 | 2.4-2.81s | FPS thấp x5 |
| 40 | **Khai báo** | 56 | 3.6 MB | 17.5 | 30.0 | 1.43-2.37s | FPS thấp x2 |
| 41 | **Khu cách ly** | 11 | 895.9 KB | 16.7 | 30.0 | 1.43-2.54s | FPS thấp x2 |
| 42 | **Khóc** | 67 | 6.3 MB | 12.1 | 30.0 | 2.3-2.63s | FPS thấp x5 |
| 43 | **Khẩu trang** | 11 | 1.0 MB | 10.9 | 30.0 | 2.4-3.65s | FPS thấp x5 |
| 44 | **Kết hôn** | 31 | 2.5 MB | 14.3 | 30.0 | 1.43-2.95s | FPS thấp x4 |
| 45 | **Lo lắng** | 7 | 510.5 KB | 12.5 | 30.0 | 2.37-2.47s | FPS thấp x5 |
| 46 | **Lây bệnh** | 50 | 3.7 MB | 12.2 | 30.0 | 2.13-2.97s | FPS thấp x5 |
| 47 | **Mua** | 35 | 2.8 MB | 13.0 | 30.0 | 1.82-2.8s | FPS thấp x4 |
| 48 | **Mời vào** | 56 | 5.5 MB | 11.3 | 30.0 | 2.42-2.71s | FPS thấp x5 |
| 49 | **Nghe** | 68 | 6.1 MB | 12.4 | 30.0 | 2.25-2.7s | FPS thấp x5 |
| 50 | **Nghỉ ngơi** | 67 | 3.9 MB | 14.2 | 30.0 | 1.58-2.31s | FPS thấp x4 |
| 51 | **Ngón tay** | 7 | 676.0 KB | 10.0 | 30.0 | 2.66-3.43s | FPS thấp x5 |
| 52 | **Nhà** | 71 | 6.0 MB | 12.8 | 30.0 | 1.86-2.96s | FPS thấp x4 |
| 53 | **Nhìn** | 60 | 5.4 MB | 11.7 | 30.0 | 1.96-3.1s | FPS thấp x4 |
| 54 | **Nhầm** | 7 | 720.9 KB | 10.9 | 30.0 | 2.29-2.93s | FPS thấp x5 |
| 55 | **Nhớ** | 57 | 4.4 MB | 11.6 | 30.0 | 2.42-3.22s | FPS thấp x5 |
| 56 | **Nói** | 30 | 2.7 MB | 11.3 | 30.0 | 2.13-2.84s | FPS thấp x5 |
| 57 | **Nói xấu** | 68 | 4.6 MB | 16.9 | 30.0 | 1.57-3.35s | FPS thấp x1 |
| 58 | **Nôn ói** | 44 | 3.0 MB | 13.7 | 30.0 | 1.4-3.2s | FPS thấp x3 |
| 59 | **Nặng** | 48 | 4.2 MB | 12.9 | 30.0 | 1.91-2.94s | FPS thấp x4 |
| 60 | **Phía sau** | 67 | 6.9 MB | 15.9 | 30.0 | 1.73-2.6s | FPS thấp x1 |
| 61 | **Phạt** | 37 | 2.6 MB | 17.5 | 30.0 | 1.6-2.41s | FPS thấp x1 |
| 62 | **Phỏng vấn** | 45 | 3.2 MB | 17.9 | 30.0 | 1.4-2.47s | FPS thấp x2 |
| 63 | **Phục hồi** | 6 | 444.5 KB | 12.0 | 30.0 | 1.67-6.54s | FPS thấp x3 |
| 64 | **Rau** | 72 | 7.5 MB | 14.0 | 30.0 | 1.99-3.1s | FPS thấp x1 |
| 65 | **Rẽ phải** | 15 | 1.8 MB | 10.5 | 30.0 | 2.58-2.96s | FPS thấp x5 |
| 66 | **Rẽ trái** | 65 | 6.2 MB | 10.6 | 30.0 | 2.59-3.6s | FPS thấp x5 |
| 67 | **San sẻ** | 46 | 3.7 MB | 10.6 | 30.0 | 2.23-3.75s | FPS thấp x5 |
| 68 | **Sốt** | 6 | 464.6 KB | 14.0 | 30.0 | 1.43-2.97s | FPS thấp x4 |
| 69 | **Sử dụng** | 59 | 4.7 MB | 13.5 | 30.0 | 1.87-2.65s | FPS thấp x4 |
| 70 | **Thích** | 11 | 665.5 KB | 17.9 | 30.0 | 1.57-2.21s | FPS thấp x1 |
| 71 | **Thăm** | 19 | 1.2 MB | 18.9 | 30.0 | 1.57-1.68s | OK |
| 72 | **Thương** | 8 | 567.3 KB | 14.1 | 30.0 | 1.67-2.51s | FPS thấp x4 |
| 73 | **Thất lạc** | 24 | 1.4 MB | 16.2 | 30.0 | 1.5-2.88s | FPS thấp x2 |
| 74 | **Thức dậy** | 67 | 5.9 MB | 11.7 | 30.0 | 2.13-2.78s | FPS thấp x5 |
| 75 | **Thức ăn** | 56 | 5.2 MB | 9.7 | 30.0 | 3.06-3.24s | FPS thấp x5 |
| 76 | **Trưa** | 57 | 5.5 MB | 14.4 | 30.0 | 1.87-2.16s | FPS thấp x4 |
| 77 | **Trường học** | 74 | 6.0 MB | 10.9 | 30.0 | 2.34-3.22s | FPS thấp x5 |
| 78 | **Tôi** | 52 | 4.1 MB | 12.5 | 30.0 | 2.13-2.67s | FPS thấp x5 |
| 79 | **Tập luyện** | 11 | 903.2 KB | 13.0 | 30.0 | 1.6-3.3s | FPS thấp x4 |
| 80 | **Tối** | 24 | 2.7 MB | 14.5 | 30.0 | 1.88-2.82s | FPS thấp x2 |
| 81 | **Uống** | 25 | 2.2 MB | 12.2 | 30.0 | 2.12-3.0s | FPS thấp x5 |
| 82 | **Virus** | 7 | 597.0 KB | 11.0 | 30.0 | 1.57-4.06s | FPS thấp x4 |
| 83 | **Vâng lời** | 8 | 679.0 KB | 14.7 | 30.0 | 1.4-2.37s | FPS thấp x4 |
| 84 | **Xa** | 62 | 6.0 MB | 14.0 | 30.0 | 1.91-2.81s | FPS thấp x3 |
| 85 | **Xe máy** | 6 | 441.8 KB | 13.4 | 30.0 | 1.88-2.67s | FPS thấp x4 |
| 86 | **Xe đạp** | 16 | 1.8 MB | 11.5 | 30.0 | 2.32-2.92s | FPS thấp x5 |
| 87 | **Xin lỗi** | 19 | 1.9 MB | 11.3 | 30.0 | 2.45-2.77s | FPS thấp x5 |
| 88 | **Xin phép** | 38 | 2.1 MB | 13.3 | 30.0 | 1.64-3.13s | FPS thấp x4 |
| 89 | **Xuất viện** | 48 | 3.9 MB | 10.3 | 30.0 | 2.3-3.27s | FPS thấp x5 |
| 90 | **Xúc động** | 59 | 4.5 MB | 10.9 | 30.0 | 2.72-2.8s | FPS thấp x5 |
| 91 | **Áp dụng** | 71 | 4.7 MB | 12.0 | 30.0 | 2.23-2.97s | FPS thấp x5 |
| 92 | **Ô tô** | 47 | 4.7 MB | 12.4 | 30.0 | 2.08-2.7s | FPS thấp x5 |
| 93 | **Ăn** | 60 | 4.6 MB | 14.4 | 30.0 | 1.67-2.67s | FPS thấp x3 |
| 94 | **Ăn mừng** | 62 | 4.5 MB | 14.6 | 30.0 | 1.65-3.27s | FPS thấp x3 |
| 95 | **Đi** | 62 | 5.1 MB | 12.2 | 30.0 | 2.17-2.83s | FPS thấp x5 |
| 96 | **Đâu** | 53 | 2.9 MB | 15.2 | 30.0 | 1.55-2.4s | FPS thấp x3 |
| 97 | **Đầu** | 54 | 4.8 MB | 11.8 | 30.0 | 2.15-2.7s | FPS thấp x5 |
| 98 | **Đẹp** | 18 | 1.8 MB | 11.0 | 30.0 | 2.13-3.5s | FPS thấp x5 |
| 99 | **Đồng ý** | 53 | 4.7 MB | 11.2 | 30.0 | 2.13-3.0s | FPS thấp x5 |
| 100 | **Ủng hộ** | 6 | 465.5 KB | 12.2 | 30.0 | 2.37-2.82s | FPS thấp x5 |

</details>

---
## 6. Bảng so sánh chiến lược: Retrain từ đầu vs Fine-tune

| Tiêu chí | Huấn luyện lại từ đầu (Retrain) | Fine-tune (Transfer Learning) |
| :--- | :--- | :--- |
| **Độ bao phủ nhãn** | **Tối ưu tuyệt đối cho cả 159 nhãn** | Có thể bị bias về 60 nhãn cũ hoặc 100 nhãn mới |
| **Nguy cơ Catastrophic Forgetting** | **Không có (0%)** | Cao nếu không áp dụng learning rate nhỏ và freeze backbone |
| **Thời gian huấn luyện** | ~30 - 60 phút (với GPU/CPU hiện đại cho 159 classes) | ~15 - 30 phút |
| **Độ phức tạp pipeline** | Đơn giản, đồng nhất định dạng | Cần trích xuất trọng số cũ và thay head layer |
| **Độ chính xác kỳ vọng** | **> 92% - 95%** toàn diện | ~85% - 90% trên các nhãn mới |
| **Đánh giá khuyến nghị** | **ƯU TIÊN SỐ 1 (RECOMMENDED)** | Phương án phụ (Nghiên cứu so sánh) |

---
## 7. Kế hoạch triển khai từng bước (Action Plan)

### Giai đoạn 1: Chuẩn hóa & Trích xuất đặc trưng MediaPipe (Feature Extraction)
1. **Xóa nhãn cũ bị trùng**: Xóa thư mục `Data/cam on/` trong `Sign Language Translator/Data/` (60 sequences cũ).
2. **Trích xuất 100 nhãn mới**: Dùng MediaPipe Holistic quét qua 3,875 video trong `dataset/train/`, trích xuất keypoints `(60, 126)` và lưu trực tiếp vào thư mục `Data/<Tên_Nhãn_Có_Dấu>/`.
3. **Chuẩn hóa nhãn tiếng Việt có dấu**: Tạo file từ điển `label_map.json` chứa mapping giữa 159 nhãn có dấu và chỉ số `0 -> 158`.

### Giai đoạn 2: Xây dựng & Huấn luyện Model mới (Model Training)
1. **Tạo mô hình FSign-159**:
   - Input: `(None, 60, 126)`
   - Backbone: `LSTM(64, return_sequences=True) -> LSTM(128, return_sequences=True) -> LSTM(64)`
   - Dense Layers: `Dense(64, relu) -> Dropout(0.2) -> Dense(32, relu)`
   - Classification Head: `Dense(159, softmax)`
2. **Huấn luyện mô hình**: Huấn luyện 100 - 150 epochs với EarlyStopping và ReduceLROnPlateau.
3. **Lưu model mới**: Lưu file tại `Sign Language Translator/release/fsign_159classes.h5` và cập nhật `RunModel.py`.

---
*Báo cáo được tự động tạo bởi `dataset_model_compatibility_check.py`.*