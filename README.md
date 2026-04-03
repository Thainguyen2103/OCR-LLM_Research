# 🇻🇳 VietIDP — Vietnamese Intelligent Document Processing

> Hệ thống trích xuất thông tin tự động từ văn bản hành chính Việt Nam, sử dụng kết hợp **Computer Vision** (YOLOv8 + Pix2Pix GAN), **OCR** (PaddleOCR PP-OCRv4), và **LLM** (Qwen2.5:7b via Ollama).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NodeJS](https://img.shields.io/badge/Node.js-20-339933?logo=node.js&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-Internal-red)

## 📋 Tổng quan

VietIDP giải quyết bài toán **số hóa văn bản hành chính** end-to-end:

1. **Stamp Detection** (YOLOv8) — Phát hiện vị trí con dấu đỏ trên ảnh scan
2. **Stamp Removal** (Pix2Pix GAN) — Xóa con dấu để cải thiện OCR
3. **OCR** (PaddleOCR PP-OCRv4) — Nhận dạng chữ tiếng Việt (2-tier: text layer → OCR fallback)
4. **Information Extraction** (Qwen2.5:7b) — Trích xuất: loại văn bản, số hiệu, ngày, cơ quan, người ký
5. **QLoRA Fine-tuning** — Tinh chỉnh LLM trên dữ liệu hành chính Việt Nam

**100% xử lý cục bộ (offline)** — Không gửi dữ liệu ra internet.

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React 18 + Vite)                │
│            Upload PDF/Image → Hiển thị kết quả              │
├──────────────────────────┬──────────────────────────────────┤
│   Backend (Express.js)   │      FastAPI (Python)             │
│   Port 5000              │      (Optional, Port 8000)        │
├──────────────────────────┴──────────────────────────────────┤
│                         src/ Pipeline                        │
│  ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐│
│  │Preprocessing│→│   OCR    │→│   LLM    │→│  Validation  ││
│  │Deskew+Denoi-│ │PaddleOCR │ │Qwen2.5:7b│ │ JSON Output  ││
│  │se+GAN       │ │PP-OCRv4  │ │via Ollama│ │              ││
│  └─────────────┘ └──────────┘ └──────────┘ └──────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Models: YOLOv8n │ U-Net GAN │ PaddleOCR │ Qwen2.5:7b     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Cấu trúc thư mục

```
VietIDP/
├── src/                           # ✨ Production Modules
│   ├── config.py                  # Centralized configuration
│   ├── preprocessing/             # Deskew, denoise, stamp removal
│   ├── ocr/                       # PaddleOCR engine + postprocess
│   ├── llm/                       # Ollama client + prompt templates
│   ├── pipeline/                  # End-to-end OCR-LLM pipeline
│   ├── api/                       # FastAPI + authentication
│   ├── evaluation/                # CER/WER/F1 metrics
│   └── data/                      # Data preparation utilities
├── ai/                            # Production scripts
│   ├── detect_api.py              # YOLO stamp detection
│   ├── summarize.py               # LLM summarization
│   ├── inference.py               # Model inference
│   ├── train_yolo.py              # YOLO training
│   └── generate_dataset.py        # Synthetic dataset generation
├── backend/                       # Express.js API server
├── frontend/                      # React 18 + Vite web interface
├── notebooks/                     # Research notebooks (5 phases)
│   ├── Phase1_Data_Preparation.py
│   ├── Phase2_Stamp_Removal_GAN.py
│   ├── Phase3_OCR_Engine.py
│   ├── Phase4_LLM_Finetuning.py
│   └── Phase5_End_to_End_Pipeline.py
├── models/                        # Model weights (YOLO, GAN, QLoRA)
├── data/                          # Datasets
│   ├── raw_word_files/            # Source DOCX files
│   ├── test/                      # 150 test PDFs
│   └── stamps/                    # Extracted/synthetic stamps
├── configs/                       # YAML configurations
├── requirements.txt               # CPU dependencies
├── requirements-gpu.txt           # GPU dependencies (Colab)
├── setup.py                       # Python package setup
├── Dockerfile                     # Docker deployment
├── .env.example                   # Environment config template
├── run_vietidp.bat                # Windows launcher
└── README.md                      # This file
```

## 🚀 Cài đặt nhanh (Windows)

### Yêu cầu tối thiểu
- **RAM**: 16 GB (cho Qwen2.5:7b)
- **Storage**: 30 GB trống
- **Python**: 3.10+
- **Node.js**: 20+
- **Ollama**: Đã cài đặt ([ollama.ai](https://ollama.ai))

### Bước 1: Pull models

```powershell
# Pull Qwen2.5:7b LLM (4.7 GB)
ollama pull qwen2.5:7b

# (Tùy chọn) Pull embedding model cho RAG
ollama pull nomic-embed-text
```

### Bước 2: Cài dependencies

```powershell
# Python dependencies (CPU)
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..

# Backend dependencies
cd backend && npm install && cd ..
```

### Bước 3: Khởi động

```powershell
# Cách 1: Dùng launcher script
run_vietidp.bat

# Cách 2: Chạy thủ công
ollama serve                                # Terminal 1
node backend/index.js                       # Terminal 2
cd frontend && npx vite --port 5173         # Terminal 3
```

Truy cập: **http://localhost:5173**

## 🔧 Configuration

Sao chép `.env.example` → `.env` và sửa:

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model Ollama |
| `OLLAMA_MAX_CHARS` | `32000` | Giới hạn ký tự input |
| `OCR_DPI` | `200` | DPI render PDF→ảnh |
| `VIETIDP_API_KEY` | _(trống)_ | API key (trống = tắt auth) |

## 📊 Evaluation Metrics

| Metric | Mô tả | Target |
|--------|--------|--------|
| CER | Character Error Rate | < 5% |
| WER | Word Error Rate | < 10% |
| F1 | Trích xuất thông tin | > 85% |

```python
from src.evaluation import compute_cer, compute_wer
cer = compute_cer("Cộng hòa xã hội chủ nghĩa Việt Nam", ocr_text)
```

## 🔒 Bảo mật

- ✅ CORS restricted (localhost only)
- ✅ File type/size validation (20MB max)
- ✅ Path traversal protection
- ✅ 100% offline processing
- ✅ Temp files cleanup
- ✅ No external API calls

## 📖 Nghiên cứu

Dự án phục vụ nghiên cứu khoa học, tham khảo:
- `Document/Final_TRI.pdf` — Bài báo nghiên cứu
- `Document/Đề cương nghiên cứu.pdf` — Đề cương chi tiết
- `notebooks/` — 5 Phase nghiên cứu đầy đủ

---

**VietIDP v2.0** | VietIDP Research Team | 2026
