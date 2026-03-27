import os
from ultralytics import YOLO

# Sử dụng mô hình YOLOv8n (nano) nhẹ nhất, huấn luyện cực nhanh và tối ưu cho CPU
model = YOLO('yolov8n.pt')

# Đường dẫn tới file config data
data_path = r"c:\Users\LAPTOP T&T\Desktop\AI_Science\ai\dataset\data.yaml"

if __name__ == '__main__':
    print("🚀 Bắt đầu huấn luyện mô hình YOLOv8 nhận diện CON DẤU")
    # Huấn luyện mô hình ngay tại đây
    results = model.train(
        data=data_path,
        epochs=30,          # Huấn luyện 30 vòng (để test nhanh)
        imgsz=640,          # Kích thước ảnh chuẩn YOLO
        batch=8,            # Số lượng ảnh 1 lần nạp
        name='stamp_model', # Tên project xuất ra
        project=r"c:\Users\LAPTOP T&T\Desktop\AI_Science\ai\models"
    )
    print("✅ Huấn luyện hoàn tất! Model lưu trong thư mục ai/models/stamp_model/weights/")
