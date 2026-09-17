import sys

def check_all():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}\n")
    
    packages = ["tensorflow", "mediapipe", "cv2", "sklearn", "matplotlib"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed (no __version__)")
            print(f"  [OK] {pkg} -> {version}")
        except ImportError as e:
            print(f"  [FAILED] {pkg} -> {e}")

    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
            print("  [OK] mediapipe.solutions.holistic -> sẵn sàng sử dụng!")
        else:
            print("  [FAILED] mediapipe.solutions không tồn tại (phiên bản mediapipe quá mới, cần downgrade về <= 0.10.14)")
    except Exception as e:
        print(f"  [FAILED] Kiểm tra mediapipe: {e}")

if __name__ == "__main__":
    check_all()
