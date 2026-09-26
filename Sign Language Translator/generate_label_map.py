# -*- coding: utf-8 -*-
"""
generate_label_map.py
=====================
Tạo file từ điển label_map.json chuẩn hóa 159 nhãn tiếng Việt có dấu
cho dự án FSign (59 nhãn cũ đã map có dấu + 100 nhãn mới từ dataset/train/).
"""

import os
import sys
import json
from pathlib import Path

# Fix Unicode UTF-8 output cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Bảng mapping chuẩn từ 59 nhãn cũ (không dấu) sang tiếng Việt có dấu chuẩn
OLD_TO_VIETNAMESE_ACCENT = {
    'ban dang lam gi': 'Bạn đang làm gì?',
    'ban di dau the': 'Bạn đi đâu thế?',
    'ban hieu ngon ngu ky hieu khong': 'Bạn hiểu ngôn ngữ ký hiệu không?',
    'ban hoc lop may': 'Bạn học lớp mấy?',
    'ban khoe khong': 'Bạn khỏe không?',
    'ban muon gio roi': 'Bạn muộn giờ rồi',
    'ban phai canh giac': 'Bạn phải cảnh giác',
    'ban ten la gi': 'Bạn tên là gì?',
    'ban tien bo day': 'Bạn tiến bộ đấy',
    'ban trong cau co the': 'Bạn trông cáu cọ thế',
    'bo me toi cung la nguoi Diec': 'Bố mẹ tôi cũng là người Điếc',
    'cai nay bao nhieu tien': 'Cái này bao nhiêu tiền?',
    'cai nay la cai gi': 'Cái này là cái gì?',
    'cap cuu': 'Cấp cứu',
    'chuc mung': 'Chúc mừng',
    'chung toi giao tiep voi nhau bang ngon ngu ky hieu': 'Chúng tôi giao tiếp với nhau bằng ngôn ngữ ký hiệu',
    'con yeu me': 'Con yêu mẹ',
    'cong viec cua ban la gi': 'Công việc của bạn là gì?',
    'hen gap lai cac ban': 'Hẹn gặp lại các bạn',
    'mon nay khong ngon': 'Món này không ngon',
    'toi bi chong mat': 'Tôi bị chóng mặt',
    'toi bi cuop': 'Tôi bị cướp',
    'toi bi dau dau': 'Tôi bị đau đầu',
    'toi bi dau hong': 'Tôi bị đau họng',
    'toi bi ket xe': 'Tôi bị kẹt xe',
    'toi bi lac': 'Tôi bị lạc',
    'toi bi phan biet doi xu': 'Tôi bị phân biệt đối xử',
    'toi cam thay rat hoi hop': 'Tôi cảm thấy rất hồi hộp',
    'toi cam thay rat vui': 'Tôi cảm thấy rất vui',
    'toi can an sang': 'Tôi cần ăn sáng',
    'toi can di ve sinh': 'Tôi cần đi vệ sinh',
    'toi can gap bac si': 'Tôi cần gặp bác sĩ',
    'toi can phien dich': 'Tôi cần phiên dịch',
    'toi can thuoc': 'Tôi cần thuốc',
    'toi dang an sang': 'Tôi đang ăn sáng',
    'toi dang buon': 'Tôi đang buồn',
    'toi dang o ben xe': 'Tôi đang ở bến xe',
    'toi dang o cong vien': 'Tôi đang ở công viên',
    'toi dang phai cach ly': 'Tôi đang phải cách ly',
    'toi dang phan van': 'Tôi đang phân vân',
    'toi di sieu thi': 'Tôi đi siêu thị',
    'toi di toi Ha Noi': 'Tôi đi tới Hà Nội',
    'toi doc kem': 'Tôi đọc kém',
    'toi khoi benh roi': 'Tôi khỏi bệnh rồi',
    'toi khong dem theo tien': 'Tôi không đem theo tiền',
    'toi khong hieu': 'Tôi không hiểu',
    'toi khong quan tam': 'Tôi không quan tâm',
    'toi la hoc sinh': 'Tôi là học sinh',
    'toi la nguoi Diec': 'Tôi là người Điếc',
    'toi la tho theu': 'Tôi là thợ thêu',
    'toi lam viec o cua hang': 'Tôi làm việc ở cửa hàng',
    'toi nham dia chi': 'Tôi nhầm địa chỉ',
    'toi song o Ha Noi': 'Tôi sống ở Hà Nội',
    'toi thay doi bung': 'Tôi thấy đói bụng',
    'toi thay nho ban': 'Tôi thấy nhớ bạn',
    'toi thich an mi': 'Tôi thích ăn mì',
    'toi thich phim truyen': 'Tôi thích phim truyện',
    'toi viet kem': 'Tôi viết kém',
    'xin chao': 'Xin chào'
}


def build_label_map(data_dir):
    data_path = Path(data_dir)
    # Lấy danh sách tất cả các thư mục nhãn trong Data/
    folder_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    # Loại bỏ 'cam on' nếu còn sót lại (đã thay bằng 'Cam on')
    folder_names = [f for f in folder_names if f != 'cam on']

    id_to_label = {}
    label_to_id = {}
    folder_to_id = {}
    id_to_display = {}
    classes_list = []

    for idx, folder_name in enumerate(folder_names):
        display_name = OLD_TO_VIETNAMESE_ACCENT.get(folder_name, folder_name)
        id_to_label[idx] = folder_name
        label_to_id[folder_name] = idx
        folder_to_id[folder_name] = idx
        id_to_display[idx] = display_name
        classes_list.append(folder_name)

    result = {
        'num_classes': len(folder_names),
        'classes': classes_list,
        'id_to_label': id_to_label,
        'label_to_id': label_to_id,
        'id_to_display': id_to_display,
        'old_accent_mapping': OLD_TO_VIETNAMESE_ACCENT
    }
    return result


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'Sign Language Translator' / 'Data'
    output_json_1 = base_dir / 'Sign Language Translator' / 'label_map.json'
    output_json_2 = base_dir / 'Sign Language Translator' / 'release' / 'label_map.json'

    print("=" * 60)
    print("  FSIGN - TẠO FILE TỪ ĐIỂN LABEL MAP 159 CLASSES")
    print("=" * 60)
    print(f"Thư mục Data: {data_dir}")

    label_map_data = build_label_map(data_dir)
    print(f"-> Đã quét {label_map_data['num_classes']} nhãn!")

    output_json_2.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_1, 'w', encoding='utf-8') as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)
    print(f"-> Đã ghi file: {output_json_1}")

    with open(output_json_2, 'w', encoding='utf-8') as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)
    print(f"-> Đã ghi file: {output_json_2}")

    print("=" * 60)
    print("Hoàn tất tạo từ điển nhãn tiếng Việt chuẩn!")


if __name__ == '__main__':
    main()
