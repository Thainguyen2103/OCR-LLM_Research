import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { FileUp, File, CheckCircle } from 'lucide-react';

export default function UploadPage() {
  const navigate = useNavigate();
  const [isUploading, setIsUploading] = useState(false);

  const [progress, setProgress] = useState(0);

  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles.length > 0) {
      handleUpload(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg'],
      'application/pdf': ['.pdf']
    },
    multiple: false
  });

  const handleUpload = async (file) => {
    setIsUploading(true);
    setProgress(0);
    
    // Simulate progress while waiting for backend
    const progressInterval = setInterval(() => {
      setProgress(prev => (prev < 90 ? prev + 2 : prev));
    }, 100);
    
    const formData = new FormData();
    formData.append('document', file);
    
    try {
      const response = await fetch('http://localhost:5000/api/process', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      clearInterval(progressInterval);
      setProgress(100);
      
      if (data.success) {
        localStorage.setItem('last_processed_pages', JSON.stringify(data.pages));
        
        setTimeout(() => {
          const mockId = Math.random().toString(36).substring(7);
          navigate(`/results/${mockId}`);
        }, 500);
      } else {
        alert("Lỗi AI: " + data.error);
        setIsUploading(false);
      }
    } catch (e) {
      clearInterval(progressInterval);
      console.error(e);
      alert("Lỗi kết nối Server Backend!");
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-page">
      <header className="page-header" style={{padding:0, marginBottom: '32px'}}>
        <h2 className="page-title">Tải lên văn bản hành chính</h2>
        <p className="page-subtitle">Hệ thống hỗ trợ PDF, PNG, JPG (ưu tiên bản scan độ phân giải cao)</p>
      </header>

      <div 
        {...getRootProps()} 
        className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
        style={{ pointerEvents: isUploading ? 'none' : 'auto', opacity: isUploading ? 0.9 : 1 }}
      >
        <input {...getInputProps()} />
        
        {isUploading ? (
          <div style={{width:'80%', margin:'0 auto', textAlign:'center'}}>
             <div style={{marginBottom:'16px'}}>
               <FileUp size={36} color="var(--accent-blue)" className="pulse" style={{display:'inline-block'}}/>
             </div>
             <h3 style={{fontSize:'18px', color:'var(--text-main)', marginBottom:'12px'}}>
               Hệ thống AI đang phân tích tài liệu... {progress}%
             </h3>
             <div style={{width:'100%', height:'8px', background:'var(--border)', borderRadius:'4px', overflow:'hidden'}}>
               <div style={{
                 width: `${progress}%`, 
                 height:'100%', 
                 background:'var(--accent-blue)', 
                 transition:'width 0.2s ease-out'
               }}></div>
             </div>
             <p style={{marginTop:'12px', color:'var(--text-muted)', fontSize:'14px'}}>
               Đang chạy mô hình YOLOv8 để tìm vị trí con dấu đỏ
             </p>
          </div>
        ) : (
          <>
            <div className="upload-icon">
              <FileUp size={28} color="var(--accent)" />
            </div>
            <h3 className="upload-title">
              {isDragActive ? 'Thả file vào đây...' : 'Kéo thả file hoặc Click để chọn'}
            </h3>
            <p className="upload-sub">Kích thước tối đa: 20MB</p>
            <div className="upload-formats">
              <span className="badge badge-blue">PDF</span>
              <span className="badge badge-green">PNG</span>
              <span className="badge badge-yellow">JPG</span>
            </div>
          </>
        )}
      </div>

      <div style={{marginTop: '40px'}}>
        <h3 style={{fontSize: '16px', marginBottom: '16px', color: 'var(--text-secondary)'}}>
          Văn bản xử lý gần đây
        </h3>
        <div className="documents-grid">
          {/* Mock Recent Files */}
          {[1, 2, 3].map(i => (
            <div key={i} className="document-card" onClick={() => navigate(`/results/mock-${i}`)}>
              <div className="doc-card-header">
                <div className="doc-icon">
                  <File size={20} color="var(--accent)" />
                </div>
                <div>
                  <div className="doc-name">Quyet_dinh_bo_nhiem_can_bo_2025.pdf</div>
                  <div className="doc-meta" style={{marginTop:'4px', display:'flex', alignItems:'center', gap:'6px'}}>
                    <CheckCircle size={12} color="var(--accent-success)" />
                    <span>Hoàn thành • 2 giờ trước</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
