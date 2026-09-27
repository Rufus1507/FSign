# THƯ MỤC LƯU TRỮ LEGACY - YOUTUBE TEST SAMPLES (KHÔNG DÙNG TRAIN)

Thư mục này chứa các sequence thử nghiệm YouTube cũ (126 chiều):
- `xin loi/`: mẫu đơn lẻ từ YouTube, không thuộc 60 nhãn webcam chuẩn.
- `cam on/`: mẫu đơn lẻ yt_0 từ YouTube.

**LƯU Ý QUAN TRỌNG:** Toàn bộ pipeline nạp dữ liệu (dataset_loader.py, train_model.py) chỉ đọc từ Data_normalized/ và Data/. Thư mục này tuyệt đối không được nạp vào training.
