import numpy as np
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

# Chuyển working directory về thư mục chứa file script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

def main():
    print("=== KIỂM TRA DỮ LIỆU DỰ ÁN ===")
    
    # 1. Kiểm tra file test data
    api_test_data_path = os.path.join('Videos', 'api test data.npy')
    if os.path.exists(api_test_data_path):
        data = np.load(api_test_data_path)
        print(f"\n[1] Đã tìm thấy: {api_test_data_path}")
        print(f"    Shape: {data.shape}")
        print(f"    Kiểu dữ liệu: {data.dtype}")
    else:
        print(f"\n[1] Không tìm thấy file: {api_test_data_path}")

    # 2. Kiểm tra các thư mục hành động trong Data/
    data_dir = 'Data'
    if os.path.exists(data_dir):
        actions = next(os.walk(data_dir), (None, [], []))[1]
        print(f"\n[2] Tổng số hành động trong thư mục '{data_dir}': {len(actions)}")
        print("    Danh sách các hành động:")
        for idx, action in enumerate(sorted(actions), 1):
            print(f"    {idx:2d}. {action}")
    else:
        print(f"\n[2] Không tìm thấy thư mục '{data_dir}'")

if __name__ == '__main__':
    main()
