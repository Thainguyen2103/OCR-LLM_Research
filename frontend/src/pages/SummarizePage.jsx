import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  FileText, Upload, Loader2, AlertCircle, CheckCircle2,
  Tag, Target, Users, Globe, Clock, Link2, BookOpen,
  BarChart2, FileSearch, ChevronDown, ChevronUp
} from 'lucide-react';

// ─── Mini Components ───────────────────────────────────────────────────────

function InfoCard({ icon: Icon, label, value, accent }) {
  if (!value) return null;
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      padding: '14px 16px',
      display: 'flex',
      gap: '12px',
      alignItems: 'flex-start'
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: 8,
        background: accent ? `${accent}18` : 'var(--bg-hover)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0, marginTop: 2
      }}>
        <Icon size={16} color={accent || 'var(--text-secondary)'} />
      </div>
      <div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 3 }}>{label}</div>
        <div style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 500, lineHeight: 1.5 }}>{value}</div>
      </div>
    </div>
  );
}

function TagList({ icon: Icon, label, items, color }) {
  if (!items || items.length === 0) return null;
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
        <Icon size={14} color={color || 'var(--text-secondary)'} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {items.map((item, i) => (
          <span key={i} style={{
            padding: '5px 12px', borderRadius: 20, fontSize: 13,
            background: color ? `${color}14` : 'var(--bg-hover)',
            color: color || 'var(--text-primary)',
            border: `1px solid ${color ? `${color}35` : 'var(--border)'}`,
            fontWeight: 500
          }}>{item}</span>
        ))}
      </div>
    </div>
  );
}

function ImportanceBadge({ level }) {
  const map = {
    'Cao': { bg: '#fee2e2', color: '#dc2626', border: '#fca5a5' },
    'Trung bình': { bg: '#fef9c3', color: '#ca8a04', border: '#fde047' },
    'Thấp': { bg: '#dcfce7', color: '#16a34a', border: '#86efac' }
  };
  const s = map[level] || map['Trung bình'];
  return (
    <span style={{
      padding: '4px 14px', borderRadius: 20, fontSize: 12, fontWeight: 700,
      background: s.bg, color: s.color, border: `1px solid ${s.border}`
    }}>{level}</span>
  );
}

function StepDot({ label, done }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
      <div style={{
        width: 20, height: 20, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: done ? 'var(--accent-success)' : 'var(--border)',
        flexShrink: 0
      }}>
        {done
          ? <CheckCircle2 size={12} color="white" />
          : <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--text-muted)' }} />}
      </div>
      <span style={{ color: done ? 'var(--text-primary)' : 'var(--text-muted)' }}>{label}</span>
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────────────────

export default function SummarizePage() {
  const [status, setStatus] = useState('idle'); // idle | uploading | done | error
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState(0);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [fileName, setFileName] = useState('');

  const steps = [
    'Đọc nội dung tài liệu',
    'Gửi tới Qwen2.5 (Ollama)',
    'Phân tích ngữ nghĩa',
    'Trích xuất thực thể',
    'Hoàn tất'
  ];

  const onDrop = useCallback(files => {
    if (files.length > 0) handleUpload(files[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    multiple: false
  });

  const handleUpload = async (file) => {
    setFileName(file.name);
    setStatus('uploading');
    setProgress(0);
    setStep(0);
    setResult(null);
    setErrorMsg('');

    // Animate step progress while waiting
    const stepInterval = setInterval(() => {
      setStep(prev => {
        if (prev < steps.length - 1) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, 4000);

    const bar = setInterval(() => setProgress(p => p < 88 ? p + 1 : p), 500);

    const formData = new FormData();
    formData.append('document', file);

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

    try {
      const res = await fetch(`${API_URL}/api/summarize`, { method: 'POST', body: formData });
      clearInterval(bar); clearInterval(stepInterval);
      setProgress(100); setStep(steps.length - 1);

      const data = await res.json();
      if (data.error) { setStatus('error'); setErrorMsg(data.error); return; }

      setResult(data);
      setStatus('done');
    } catch {
      clearInterval(bar); clearInterval(stepInterval);
      setStatus('error');
      setErrorMsg(`Không kết nối được server backend (${API_URL})`);
    }
  };

  const s = result?.summary || {};
  const stats = result?.stats || {};

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', animation: 'fadeIn 0.3s ease' }}>
      {/* Header */}
      <header style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)' }}>
          Phân tích & Tóm tắt Văn bản
        </h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: 6 }}>
          Tải lên tài liệu .docx — AI Qwen2.5 sẽ phân tích toàn diện nội dung, trích xuất thực thể và tạo tóm tắt chuyên sâu.
        </p>
      </header>

      {/* Upload Zone */}
      {status !== 'done' && (
        <div
          {...getRootProps()}
          style={{
            border: `2px dashed ${isDragActive ? 'var(--accent)' : 'var(--border-bright)'}`,
            borderRadius: 'var(--radius)',
            padding: '40px 32px',
            background: isDragActive ? 'var(--accent-glow)' : 'var(--bg-card)',
            textAlign: 'center',
            cursor: status === 'uploading' ? 'wait' : 'pointer',
            pointerEvents: status === 'uploading' ? 'none' : 'auto',
            transition: 'all 0.2s ease',
            marginBottom: 28
          }}
        >
          <input {...getInputProps()} />

          {status === 'uploading' ? (
            <div style={{ maxWidth: 480, margin: '0 auto' }}>
              <Loader2 size={36} color="var(--accent)" style={{ animation: 'spin 1s linear infinite', marginBottom: 16 }} />
              <h3 style={{ fontSize: 17, fontWeight: 600, marginBottom: 20 }}>AI đang phân tích tài liệu…</h3>

              {/* Steps */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, textAlign: 'left', marginBottom: 20 }}>
                {steps.map((s, i) => <StepDot key={i} label={s} done={i <= step} />)}
              </div>

              {/* Progress bar */}
              <div style={{ width: '100%', height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden', marginBottom: 8 }}>
                <div style={{
                  height: '100%', width: `${progress}%`,
                  background: 'linear-gradient(90deg, var(--accent), var(--accent-2))',
                  transition: 'width 0.5s ease', borderRadius: 3
                }} />
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>Qwen2.5 đang đọc văn bản, có thể mất 30–60 giây…</p>
            </div>
          ) : (
            <>
              <Upload size={32} color="var(--accent)" style={{ marginBottom: 12 }} />
              <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 6 }}>
                {isDragActive ? 'Thả file vào đây…' : 'Kéo thả hoặc Click để chọn'}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Hỗ trợ .docx, .txt</p>
            </>
          )}

          {status === 'error' && (
            <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--accent-error)', justifyContent: 'center', fontSize: 14 }}>
              <AlertCircle size={16} />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {status === 'done' && result && (
        <div style={{ animation: 'fadeIn 0.4s ease' }}>
          {/* Doc Identity Bar */}
          <div style={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: '20px 24px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            flexWrap: 'wrap', gap: 16, marginBottom: 24
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div style={{ width: 44, height: 44, borderRadius: 10, background: 'var(--accent-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <FileText size={22} color="var(--accent)" />
              </div>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>{fileName}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
                  {s.loai_van_ban || 'Văn bản hành chính'} {s.so_hieu ? `• ${s.so_hieu}` : ''}
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>
                  {s.co_quan_ban_hanh} {s.ngay_ban_hanh ? `— ${s.ngay_ban_hanh}` : ''}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              {s.muc_do_quan_trong && <ImportanceBadge level={s.muc_do_quan_trong} />}
              {s.linh_vuc && <span style={{ background: 'var(--accent-glow)', color: 'var(--accent)', border: '1px solid rgba(30,103,146,0.2)', padding: '4px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600 }}>{s.linh_vuc}</span>}
              <button
                onClick={() => { setStatus('idle'); setResult(null); }}
                style={{ background: 'var(--bg-hover)', border: '1px solid var(--border-bright)', borderRadius: 8, padding: '6px 14px', cursor: 'pointer', fontSize: 13, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}
              >
                <Upload size={13} /> Tải lên lại
              </button>
            </div>
          </div>

          {/* Stats row */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
            {[
              { label: 'Số từ', value: stats.word_count?.toLocaleString(), color: 'var(--accent)' },
              { label: 'Ký tự', value: stats.char_count?.toLocaleString(), color: 'var(--accent-2)' },
              { label: 'Đối tượng', value: s.doi_tuong_ap_dung?.substring(0, 40) || '—', color: 'var(--accent-success)' },
              { label: 'Hiệu lực', value: s.thoi_han_hieu_luc || '—', color: 'var(--accent-warning)' }
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                flex: '1 1 180px', background: 'var(--bg-card)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', padding: '14px 18px',
                borderLeft: `3px solid ${color}`
              }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{value}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 20, alignItems: 'start' }}>
            {/* Left Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

              {/* Abstract short */}
              {s.tom_tat_ngan && (
                <div style={{ background: 'linear-gradient(135deg, #eff6ff, #f0fdf4)', border: '1px solid #bfdbfe', borderRadius: 'var(--radius)', padding: '20px 24px' }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: '#2563eb', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>Tóm tắt nhanh</div>
                  <p style={{ fontSize: 15, color: 'var(--text-primary)', lineHeight: 1.7, fontWeight: 500 }}>{s.tom_tat_ngan}</p>
                </div>
              )}

              {/* Full abstract */}
              {s.tom_tat_day_du && (
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px 24px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                    <BookOpen size={16} color="var(--accent)" />
                    <span style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)' }}>Tóm tắt chi tiết</span>
                  </div>
                  <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.85 }}>{s.tom_tat_day_du}</p>
                </div>
              )}

              {/* Bullet points */}
              {s.diem_chinh && s.diem_chinh.length > 0 && (
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px 24px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                    <BarChart2 size={16} color="var(--accent)" />
                    <span style={{ fontWeight: 700, fontSize: 14 }}>Các điểm nội dung chính</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {s.diem_chinh.map((p, i) => (
                      <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                        <div style={{
                          width: 24, height: 24, borderRadius: 6,
                          background: 'var(--accent-glow)', color: 'var(--accent)',
                          fontSize: 12, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                        }}>{i + 1}</div>
                        <p style={{ fontSize: 14, color: 'var(--text-primary)', lineHeight: 1.7, paddingTop: 3 }}>{p}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Obligations */}
              {s.nghia_vu_va_quyen_han && s.nghia_vu_va_quyen_han.length > 0 && (
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px 24px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                    <CheckCircle2 size={16} color="var(--accent-success)" />
                    <span style={{ fontWeight: 700, fontSize: 14 }}>Nghĩa vụ & Quyền hạn</span>
                  </div>
                  <ul style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {s.nghia_vu_va_quyen_han.map((o, i) => (
                      <li key={i} style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.7 }}>{o}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Preview text toggle */}
              {stats.preview_text && (
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
                  <button
                    onClick={() => setShowPreview(!showPreview)}
                    style={{ width: '100%', padding: '14px 20px', background: 'transparent', border: 'none', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: 'inherit', textAlign: 'left' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>
                      <FileSearch size={15} /> Xem nội dung gốc (800 ký tự đầu)
                    </div>
                    {showPreview ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                  {showPreview && (
                    <div style={{ padding: '0 20px 20px', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.8, borderTop: '1px solid var(--border)', paddingTop: 16, maxHeight: 300, overflowY: 'auto' }}>
                      {stats.preview_text}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <InfoCard icon={Target} label="Mục đích chính" value={s.muc_dich_chinh} accent="#2563eb" />
              <InfoCard icon={Users} label="Đối tượng áp dụng" value={s.doi_tuong_ap_dung} accent="#16a34a" />
              <InfoCard icon={Globe} label="Phạm vi áp dụng" value={s.pham_vi_ap_dung} accent="#7c3aed" />
              <InfoCard icon={Clock} label="Thời hạn hiệu lực" value={s.thoi_han_hieu_luc} accent="#d97706" />

              <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '18px 20px' }}>
                <TagList icon={Tag} label="Từ khóa" items={s.tu_khoa} color="#2563eb" />
                <TagList icon={Link2} label="Văn bản liên quan" items={s.van_ban_lien_quan} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
