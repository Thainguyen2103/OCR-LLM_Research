import cv2
import os
import io
from ultralytics import YOLO

# Khởi tạo model bằng file trọng số vừa train xong
# (Dùng file .pt tốt nhất sau khi train)
MODEL_PATH = r"c:\Users\LAPTOP T&T\Desktop\AI_Science\ai\models\stamp_model\weights\best.pt"

# Load model (chỉ load 1 lần khi server start)
model = None

def init_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            print(f"✅ AI Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

def detect_stamps_on_image(image_bytes):
    """
    Nhận diện con dấu trên ảnh đầu vào, trả về danh sách bounding boxes
    [ {'x':10,'y':10,'w':50,'h':50, 'conf': 0.95}, ... ]
    """
    if model is None:
        init_model()
        
    if model is None:
        return []

    # Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return []

    # Dự đoán bằng YOLO
    results = model.predict(source=img, conf=0.5, save=False)
    
    stamps = []
    # Kết quả trả về cho bản đầu tiên
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Tọa độ xywh
            x, y, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            stamps.append({
                "x_center": x,
                "y_center": y,
                "width": w,
                "height": h,
                "confidence": round(conf * 100, 2)
            })
            
    return stamps
