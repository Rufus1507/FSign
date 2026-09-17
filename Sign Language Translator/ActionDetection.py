import cv2
import numpy as np
import os
import itertools
from matplotlib import pyplot as plt
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, multilabel_confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

# Chuyển working directory về thư mục chứa file script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Khởi tạo MediaPipe Holistic
mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Danh sách 60 hành động mẫu
DEFAULT_ACTIONS = np.array([
    'ban dang lam gi', 'ban di dau the', 'ban hieu ngon ngu ky hieu khong', 'ban hoc lop may',
    'ban khoe khong', 'ban muon gio roi', 'ban phai canh giac', 'ban ten la gi', 'ban tien bo day',
    'ban trong cau co the', 'bo me toi cung la nguoi Diec', 'cai nay bao nhieu tien', 
    'cai nay la cai gi', 'cam on', 'cap cuu', 'chuc mung', 'chung toi giao tiep voi nhau bang ngon ngu ky hieu', 
    'con yeu me', 'cong viec cua ban la gi', 'hen gap lai cac ban', 'mon nay khong ngon', 'toi bi chong mat', 
    'toi bi cuop', 'toi bi dau dau', 'toi bi dau hong', 'toi bi ket xe', 'toi bi lac', 'toi bi phan biet doi xu', 
    'toi cam thay rat hoi hop', 'toi cam thay rat vui', 'toi can an sang', 'toi can di ve sinh', 
    'toi can gap bac si', 'toi can phien dich', 'toi can thuoc', 'toi dang an sang', 'toi dang buon', 
    'toi dang o ben xe', 'toi dang o cong vien', 'toi dang phai cach ly', 'toi dang phan van', 'toi di sieu thi', 
    'toi di toi Ha Noi', 'toi doc kem', 'toi khoi benh roi', 'toi khong dem theo tien', 'toi khong hieu', 
    'toi khong quan tam', 'toi la hoc sinh', 'toi la nguoi Diec', 'toi la tho theu', 'toi lam viec o cua hang', 
    'toi nham dia chi', 'toi song o Ha Noi', 'toi thay doi bung', 'toi thay nho ban', 'toi thich an mi', 
    'toi thich phim truyen', 'toi viet kem', 'xin chao'
])

def load_dataset(data_path='Data', actions=DEFAULT_ACTIONS, no_sequences=60, sequence_length=60):
    """
    Tải dữ liệu keypoints từ thư mục Data/
    """
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    print(f"=> Đang tải dữ liệu từ '{data_path}' cho {len(actions)} hành động...")
    for action in actions:
        action_dir = os.path.join(data_path, action)
        if not os.path.exists(action_dir):
            continue
        for sequence in range(no_sequences):
            window = []
            valid_seq = True
            for frame_num in range(sequence_length):
                frame_path = os.path.join(action_dir, str(sequence), f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    res = np.load(frame_path)
                    window.append(res)
                else:
                    valid_seq = False
                    break
            if valid_seq and len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    print(f"=> Đã tải xong! Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y

def build_model(input_shape=(60, 126), num_classes=60):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def train(epochs=100, save_name='model_trained.h5'):
    X, y = load_dataset()
    if len(X) == 0:
        print("[Lỗi] Không tìm thấy dữ liệu để train!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    log_dir = os.path.join('Logs')
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = TensorBoard(log_dir=log_dir)

    model = build_model(input_shape=(X.shape[1], X.shape[2]), num_classes=y.shape[1])
    model.summary()

    print(f"\n=> Bắt đầu huấn luyện mô hình ({epochs} epochs)...")
    model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback], validation_data=(X_test, y_test))

    model.save(save_name)
    print(f"=> Đã lưu mô hình tại: {save_name}")

    # Đánh giá trên tập test
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc*100:.2f}%")

def evaluate_model(model_path='release/94,58.h5'):
    X, y = load_dataset()
    if len(X) == 0:
        print("[Lỗi] Không tìm thấy dữ liệu để đánh giá!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if not os.path.exists(model_path):
        print(f"[Lỗi] Không tìm thấy file mô hình tại {model_path}")
        return

    model = build_model(num_classes=y.shape[1])
    model.load_weights(model_path)
    
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1)
    ypred = np.argmax(yhat, axis=1)

    acc = accuracy_score(ytrue, ypred)
    print(f"\n=> Accuracy Score: {acc * 100:.2f}%")

    cnf_matrix = confusion_matrix(ytrue, ypred)
    print("\nConfusion Matrix:")
    print(cnf_matrix)

def export_tflite(keras_model_path='release/94,58.h5', output_tflite='model.tflite'):
    """
    Xuất mô hình Keras sang TensorFlow Lite (.tflite) để nhúng vào Mobile/Web/Edge
    """
    model = build_model(num_classes=len(DEFAULT_ACTIONS))
    model.load_weights(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_tflite, 'wb') as f:
        f.write(tflite_model)
    print(f"=> Đã xuất thành công mô hình TFLite tại: {output_tflite}")

if __name__ == '__main__':
    print("=== ACTION DETECTION & MODEL MANAGEMENT ===")
    print("1. Huấn luyện mô hình mới (Train)")
    print("2. Đánh giá mô hình đã lưu (Evaluate)")
    print("3. Xuất mô hình sang TFLite")
    choice = input("Lựa chọn (1, 2, hoặc 3): ").strip()

    if choice == '1':
        epochs_input = input("Nhập số epochs (mặc định 100): ").strip()
        epochs = int(epochs_input) if epochs_input.isdigit() else 100
        train(epochs=epochs)
    elif choice == '2':
        model_path = input("Nhập đường dẫn file model/weights (mặc định: release/94,58.h5): ").strip()
        if not model_path:
            model_path = 'release/94,58.h5'
        evaluate_model(model_path)
    elif choice == '3':
        export_tflite()
    else:
        print("Lựa chọn không hợp lệ.")
