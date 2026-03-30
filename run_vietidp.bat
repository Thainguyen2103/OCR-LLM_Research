@echo off
chcp 65001 >nul
echo ==============================================
echo        HỆ THỐNG TRÍ TUỆ NHÂN TẠO VIETIDP
echo      (Tích hợp YOLOv8 và LLM Qwen2.5 Local)
echo ==============================================
echo.

echo [1/3] Dang danh thuc May chu AI (Ollama)...
start "Ollama Server" /MIN cmd /c "ollama serve"

echo [2/3] Dang khoi dong Backend API...
cd "%~dp0"
start "VietIDP Backend" /MIN cmd /c "node backend/index.js"

echo [3/3] Dang mo Giao dien Web...
cd "%~dp0\frontend"
start "VietIDP Frontend" /MIN cmd /c "node node_modules/vite/bin/vite.js"

echo.
echo He thong dang duoc ket noi... Vui long doi 3 giay!
timeout /t 3 >nul
start http://localhost:3000

echo.
echo ==============================================
echo [!] XONG! Trang web da duoc mo tren trinh duyet cua ban.
echo [!] De tat he thong, chi can dong 3 cua so mau den nho dang thu nho.
echo ==============================================
pause
