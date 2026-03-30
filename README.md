# VietIDP — Hệ thống Xử lý Tài liệu Hành chính Thông minh

> **Đề tài NCKH Sinh viên 2026**  
> Tích hợp YOLOv8 và LLM (Qwen2.5) để tự động hóa xử lý văn bản hành chính Việt Nam

![Stack](https://img.shields.io/badge/Frontend-React%20%2F%20Vite-61DAFB?style=flat-square&logo=react)
![Stack](https://img.shields.io/badge/Backend-Node.js%20%2F%20Express-339933?style=flat-square&logo=nodedotjs)
![Stack](https://img.shields.io/badge/AI-YOLOv8%20%2B%20Qwen2.5-FF6F00?style=flat-square&logo=python)
![Stack](https://img.shields.io/badge/Privacy-100%25%20On--Premise-0D1117?style=flat-square&logo=lock)

---

## 📌 Giới thiệu

**VietIDP** (Vietnamese Intelligent Document Processor) là hệ thống ứng dụng Trí tuệ Nhân tạo để phân tích, tóm tắt và trích xuất thông tin từ các văn bản hành chính Việt Nam (Quyết định, Thông tư, Công văn, Nghị định...).

Hệ thống được thiết kế theo mô hình **On-Premise 100%** — mọi xử lý diễn ra hoàn toàn trên máy tính nội bộ, không cần kết nối Internet, không gửi dữ liệu cho bên thứ ba, đảm bảo bảo mật tuyệt đối thông tin hành chính.

---

## 🔬 Đề tài nghiên cứu

| Mục | Chi tiết |
|---|---|
| **Vấn đề** | Khối lượng văn bản hành chính ngày càng lớn, việc đọc và phân loại thủ công tốn nhiều thời gian và dễ sai sót |
| **Giải pháp** | Xây dựng pipeline AI 2 giai đoạn: Nhận diện thực thể hình ảnh (Con dấu) + Phân tích ngữ nghĩa (Tóm tắt nội dung) |
| **Điểm mới** | Triển khai LLM hoàn toàn cục bộ (không dùng API ChatGPT/Gemini) để bảo vệ dữ liệu hành chính nhạy cảm |
| **Đối tượng áp dụng** | Các cơ quan nhà nước, trường đại học, đơn vị hành chính cần xử lý khối lượng lớn văn bản |

---

## 🏗️ Kiến trúc hệ thống

```
VietIDP/
│
├── 📁 frontend/          → Giao diện Web (React + Vite)
│   └── src/
│       ├── pages/         → Các trang: Upload, Results, Summarize, History
│       └── components/    → Sidebar, thanh điều hướng
│
├── 📁 backend/           → Máy chủ API (Node.js + Express)
│   └── index.js           → Điều phối các request; gọi Python AI scripts
│
├── 📁 ai/                → Các mô-đun AI (Python)
│   ├── detect_api.py      → Nhận diện con dấu bằng YOLOv8 + trích xuất text PDF
│   ├── summarize.py       → Tóm tắt & phân tích văn bản bằng Qwen2.5 (Ollama)
│   ├── train_yolo.py      → Script huấn luyện model YOLOv8 tùy chỉnh
│   └── models/            → Chứa file trọng số mô hình YOLOv8 (best.pt)
│
├── 📁 raw_word_files/    → Tập dữ liệu thô để huấn luyện
├── 📁 stamps_dataset/    → Dataset ảnh con dấu đã gán nhãn (YOLO format)
│
├── run_vietidp.bat        → Script khởi động toàn bộ hệ thống 1-click (Windows)
├── Dockerfile             → Đóng gói Container cho triển khai Cloud
└── README.md              → Tài liệu này
```

---

## 🧠 Công nghệ sử dụng

### Frontend
| Công nghệ | Vai trò |
|---|---|
| **React 18 + Vite** | Framework giao diện người dùng, hot-reload cho phát triển nhanh |
| **React Router v6** | Điều hướng đa trang (Upload → Results → Summarize) |
| **react-dropzone** | Hỗ trợ kéo-thả file tải lên trực quan |
| **lucide-react** | Bộ icon nhất quán trên toàn giao diện |
| **Vanilla CSS** | Thiết kế giao diện "Warm Paper" không phụ thuộc thư viện nặng |

### Backend
| Công nghệ | Vai trò |
|---|---|
| **Node.js + Express** | Máy chủ API xử lý HTTP Request từ Frontend |
| **Multer** | Nhận và lưu trữ tạm thời file tải lên |
| **child_process.spawn** | Gọi Python AI scripts và nhận kết quả JSON qua stdout |

### AI / Machine Learning
| Công nghệ | Vai trò |
|---|---|
| **YOLOv8 (Ultralytics)** | Phát hiện vị trí con dấu đỏ trên ảnh/PDF với độ chính xác cao |
| **PyMuPDF (fitz)** | Render PDF thành ảnh (cho YOLO) và trích xuất văn bản thô |
| **OpenCV (cv2)** | Vẽ khung bounding box lên ảnh để trực quan hóa kết quả |
| **Ollama** | Runtime chạy mô hình LLM cục bộ trên CPU/GPU |
| **Qwen2.5 (1.5B)** | Mô hình ngôn ngữ lớn phân tích, tóm tắt văn bản tiếng Việt |
| **python-docx** | Đọc và trích xuất text từ file `.docx` (Microsoft Word) |

---

## 🔄 Quy trình xử lý (Pipeline)

### Luồng 1: File PDF / Hình ảnh (PNG, JPG)

```
[Người dùng tải file]
        │
        ▼
[Backend: /api/process]
        │
        ▼
[detect_api.py]
   ├── PyMuPDF: Render PDF → Ảnh (150 DPI) + Trích xuất text thô
   ├── YOLOv8: Phát hiện vị trí con dấu đỏ trên từng trang ảnh
   ├── OpenCV: Vẽ bounding box đỏ xung quanh con dấu
   └── Nếu có text đủ dài → Gọi Qwen2.5 để tóm tắt
        │
        ▼
[Frontend: ResultsPage]
   ├── Hiển thị ảnh có đánh dấu con dấu (trái)
   └── Hiển thị Dashboard tóm tắt AI (phải)
```

### Luồng 2: File văn bản (DOCX, TXT)

```
[Người dùng tải file]
        │
        ▼
[Backend: /api/summarize]
        │
        ▼
[summarize.py]
   ├── python-docx / open(): Trích xuất toàn bộ text
   └── Qwen2.5 (Ollama, local): Phân tích & tóm tắt toàn diện
        │
        ▼
[Frontend: SummarizePage]
   └── Hiển thị Dashboard: Tóm tắt 1 dòng, Chi tiết, Điểm chính, Từ khóa...
```

---

## 📊 Kết quả phân tích AI trả về

Với mỗi văn bản, mô hình Qwen2.5 sẽ phân tích và trả về JSON chuẩn gồm:

| Trường | Nội dung |
|---|---|
| `loai_van_ban` | Loại văn bản (Quyết định / Thông tư / Công văn...) |
| `so_hieu` | Số hiệu văn bản (VD: 123/QĐ-BCA) |
| `co_quan_ban_hanh` | Cơ quan, Bộ, Ngành ban hành |
| `nguoi_ky` | Họ tên và chức vụ người ký |
| `tom_tat_ngan` | Tóm tắt 1 câu về nội dung chính |
| `tom_tat_day_du` | Tóm tắt chi tiết 5-8 câu |
| `diem_chinh` | Danh sách các điểm quan trọng / quy định cốt lõi |
| `tu_khoa` | Từ khóa trích xuất từ nội dung |
| `muc_do_quan_trong` | Đánh giá mức độ: Cao / Trung bình / Thấp |
| `linh_vuc` | Lĩnh vực (Giáo dục / Y tế / Kinh tế / An ninh...) |

---

## 🚀 Hướng dẫn cài đặt & Chạy

### Yêu cầu hệ thống
- **HĐH:** Windows 10/11 (64-bit)
- **RAM:** Tối thiểu 8GB (khuyến nghị 16GB)
- **Ổ cứng:** Tối thiểu 10GB trống (cho mô hình AI)
- **Python:** 3.9+ với pip
- **Node.js:** 18+

### Bước 1: Cài đặt thư viện Python

```bash
pip install ultralytics pymupdf opencv-python requests python-docx
```

### Bước 2: Cài đặt thư viện Node.js

```bash
npm install
cd frontend && npm install
```

### Bước 3: Cài đặt Ollama và mô hình Qwen2.5

1. Tải và cài đặt Ollama từ: https://ollama.com/download
2. Mở Command Prompt và tải mô hình AI:

```bash
ollama run qwen2.5:1.5b
```

> **Lưu ý:** Việc tải về chỉ cần thực hiện **1 lần duy nhất**. Mô hình sẽ được lưu vĩnh viễn trên máy tính.

### Bước 4: Khởi động hệ thống

**Cách 1 — 1-Click (Khuyến nghị):**
Click đúp vào file `run_vietidp.bat` trong thư mục gốc. Hệ thống tự động mở trình duyệt tại `http://localhost:3000`.

**Cách 2 — Thủ công (3 terminal riêng biệt):**

```bash
# Terminal 1: Máy chủ AI
ollama serve

# Terminal 2: Backend API
node backend/index.js

# Terminal 3: Frontend Web
cd frontend && node node_modules/vite/bin/vite.js
```

---

## 🐳 Triển khai lên Cloud (Tùy chọn)

Dự án đã được đóng gói Docker sẵn sàng triển khai lên các nền tảng Cloud:

```bash
docker build -t vietidp .
docker run -p 5000:5000 vietidp
```

> **Lưu ý quan trọng:** Khi triển khai Cloud, chức năng Tóm tắt bằng Qwen2.5 sẽ không hoạt động (do LLM cần chạy Local). Cloud phù hợp để demo giao diện và tính năng nhận diện con dấu YOLOv8.

---

## 📂 Các định dạng file được hỗ trợ

| Định dạng | Xử lý |
|---|---|
| `.pdf` | ✅ Render ảnh (YOLO) + Trích xuất text (Tóm tắt) |
| `.png`, `.jpg`, `.jpeg` | ✅ Nhận diện con dấu bằng YOLOv8 |
| `.docx` | ✅ Trích xuất và Tóm tắt toàn bộ nội dung |
| `.txt` | ✅ Đọc và Tóm tắt nội dung |

---

## 📜 Giấy phép & Tác giả

- **Đề tài:** Nghiên cứu Khoa học Sinh viên 2026
- **Lĩnh vực:** LLM + OCR Engine ứng dụng trong Hành chính số
- **Mô hình AI:** Qwen2.5 (Alibaba Cloud) — Apache 2.0 License
- **Framework YOLO:** Ultralytics YOLOv8 — AGPL-3.0 License
