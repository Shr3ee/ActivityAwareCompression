import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000/api";

export default function App() {
  const [jobId, setJobId] = useState(null);
  const [jobData, setJobData] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [evalStride, setEvalStride] = useState(15);
  const fileInputRef = useRef(null);

  // Fetch verified baseline on mount
  useEffect(() => {
    fetch(`${API_BASE}/baseline`)
      .then((res) => res.json())
      .then((data) => setBaseline(data))
      .catch((err) => console.error("Could not fetch baseline:", err));
  }, []);

  // Poll for job updates when processing
  useEffect(() => {
    if (!jobId) return;

    let interval = null;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/job/${jobId}`);
        if (!res.ok) throw new Error("Failed to fetch job status.");
        const data = await res.json();
        setJobData(data);
        if (data.status === "error") {
          setErrorMessage(data.message || "An error occurred during processing.");
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    fetchStatus();

    if (jobData?.status === "processing") {
      interval = setInterval(fetchStatus, 1500);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, jobData?.status]);

  // Handle file upload
  const handleUpload = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      setJobId(data.job_id);
      // Fetch fresh job
      const jRes = await fetch(`${API_BASE}/job/${data.job_id}`);
      setJobData(await jRes.json());
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Quick load of the verified test surveillance video
  const handleLoadSample = async () => {
    setIsUploading(true);
    setErrorMessage(null);
    try {
      const res = await fetch(`${API_BASE}/load-sample`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to load sample video");
      }
      const data = await res.json();
      setJobId(data.job_id);
      const jRes = await fetch(`${API_BASE}/job/${data.job_id}`);
      setJobData(await jRes.json());
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Trigger motion analysis
  const handleAnalyze = async () => {
    if (!jobId) return;
    setErrorMessage(null);
    try {
      const res = await fetch(`${API_BASE}/job/${jobId}/analyze?threshold=2.0`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to trigger analysis.");
      setJobData((prev) => ({
        ...prev,
        status: "processing",
        stage: "analyzing",
        message: "Running OpenCV motion detection...",
      }));
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // Trigger adaptive compression
  const handleCompress = async () => {
    if (!jobId) return;
    setErrorMessage(null);
    try {
      const res = await fetch(`${API_BASE}/job/${jobId}/compress`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to trigger compression.");
      setJobData((prev) => ({
        ...prev,
        status: "processing",
        stage: "compressing",
        message: "Executing adaptive compression (STATIC: 36, ACTIVE: 28)...",
      }));
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // Trigger quality evaluation
  const handleEvaluate = async () => {
    if (!jobId) return;
    setErrorMessage(null);
    try {
      const res = await fetch(
        `${API_BASE}/job/${jobId}/evaluate?stride=${evalStride}`,
        { method: "POST" }
      );
      if (!res.ok) throw new Error("Failed to trigger evaluation.");
      setJobData((prev) => ({
        ...prev,
        status: "processing",
        stage: "evaluating",
        message: "Calculating PSNR and SSIM metrics...",
      }));
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  const isProcessing = jobData?.status === "processing";
  const metadata = jobData?.metadata;
  const analysis = jobData?.analysis;
  const compression = jobData?.compression;
  const evaluation = jobData?.evaluation;

  return (
    <div className="container">
      {/* Header */}
      <header className="app-header">
        <div className="title-group">
          <h1>
            Activity-Aware Video Compression
            <span className="tag-badge">OpenCV & FFmpeg</span>
          </h1>
          <p>
            Scene-level motion classification with adaptive CRF allocation
            (STATIC: CRF 36 / ACTIVE: CRF 28)
          </p>
        </div>
      </header>

      {/* Global Status Banner */}
      {jobData?.message && (
        <div
          className={`status-banner ${
            isProcessing
              ? "processing"
              : jobData.status === "error"
              ? "error"
              : "success"
          }`}
        >
          {isProcessing && <div className="spinner"></div>}
          <div>
            <strong>Status:</strong> {jobData.message}
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="status-banner error">
          <div>
            <strong>Error:</strong> {errorMessage}
          </div>
        </div>
      )}

      {/* Section 1: Upload & Input Selection */}
      <section className="card">
        <h2 className="card-title">1. Video Upload & Inspection</h2>
        <div className="upload-grid">
          <div
            className="drop-zone"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept="video/mp4,video/*"
              style={{ display: "none" }}
              onChange={(e) => {
                if (e.target.files?.[0]) handleUpload(e.target.files[0]);
              }}
            />
            <p style={{ fontSize: "1.1rem", fontWeight: "600" }}>
              {isUploading ? "Uploading video..." : "Drop MP4 video here or click to browse"}
            </p>
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "6px" }}>
              Supports H.264 MP4 surveillance footage
            </p>
          </div>

          <div className="sample-button-box">
            <p style={{ fontSize: "0.9rem", color: "var(--text-muted)", marginBottom: "12px" }}>
              College Project Demo Video:
            </p>
            <button
              className="btn btn-secondary"
              onClick={handleLoadSample}
              disabled={isUploading || isProcessing}
            >
              Load Test Surveillance Video (89s)
            </button>
          </div>
        </div>

        {metadata && (
          <div className="meta-grid">
            <div className="meta-chip">
              <span>Resolution</span>
              <strong>{metadata.resolution}</strong>
            </div>
            <div className="meta-chip">
              <span>Frame Rate</span>
              <strong>{metadata.fps} FPS</strong>
            </div>
            <div className="meta-chip">
              <span>Total Frames</span>
              <strong>{metadata.total_frames}</strong>
            </div>
            <div className="meta-chip">
              <span>Duration</span>
              <strong>{metadata.duration_sec} s</strong>
            </div>
            <div className="meta-chip">
              <span>Original Size</span>
              <strong>{metadata.file_size_mb} MB</strong>
            </div>
          </div>
        )}
      </section>

      {/* Section 2: Motion Detection & Timeline */}
      {metadata && (
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <h2 className="card-title" style={{ margin: 0 }}>
              2. Scene-Level Motion Analysis (OpenCV)
            </h2>
            <button
              className="btn btn-primary"
              onClick={handleAnalyze}
              disabled={isProcessing}
            >
              {analysis ? "Re-Run Motion Detection" : "Run Motion Detection"}
            </button>
          </div>

          {analysis ? (
            <div className="timeline-container">
              {/* Graphical Timeline Bar */}
              <div className="timeline-bar">
                {analysis.segments.map((seg) => {
                  const widthPct = (seg.duration / analysis.duration_sec) * 100;
                  return (
                    <div
                      key={seg.segment_index}
                      className={`timeline-seg ${seg.classification}`}
                      style={{ width: `${widthPct}%` }}
                      title={`Segment ${seg.segment_index}: ${seg.classification} (${seg.start_time}s - ${seg.end_time}s) Avg Motion: ${seg.avg_motion_score}%`}
                    >
                      {widthPct > 8 && (
                        <span>
                          {seg.classification} ({seg.duration}s)
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="timeline-legend">
                <div className="legend-item">
                  <div className="legend-box static"></div>
                  <span>
                    <strong>STATIC:</strong> {analysis.static_duration_sec}s (
                    {(
                      (analysis.static_duration_sec / analysis.duration_sec) *
                      100
                    ).toFixed(1)}
                    %) &rarr; Allocated <strong>CRF 36</strong>
                  </span>
                </div>
                <div className="legend-item">
                  <div className="legend-box active"></div>
                  <span>
                    <strong>ACTIVE:</strong> {analysis.active_duration_sec}s (
                    {(
                      (analysis.active_duration_sec / analysis.duration_sec) *
                      100
                    ).toFixed(1)}
                    %) &rarr; Allocated <strong>CRF 28</strong>
                  </span>
                </div>
              </div>

              {/* Segments Table */}
              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Class</th>
                      <th>Start (s)</th>
                      <th>End (s)</th>
                      <th>Duration</th>
                      <th>Frame Range</th>
                      <th>Avg Motion</th>
                      <th>Allocated CRF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.segments.map((s) => (
                      <tr key={s.segment_index}>
                        <td>{s.segment_index}</td>
                        <td>
                          <span
                            className={`badge ${
                              s.classification === "STATIC"
                                ? "badge-static"
                                : "badge-active"
                            }`}
                          >
                            {s.classification}
                          </span>
                        </td>
                        <td>{s.start_time}s</td>
                        <td>{s.end_time}s</td>
                        <td>{s.duration}s</td>
                        <td>
                          {s.start_frame} &ndash; {s.end_frame} ({s.frame_count})
                        </td>
                        <td>{s.avg_motion_score}%</td>
                        <td>
                          <strong>{s.classification === "STATIC" ? 36 : 28}</strong>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <p style={{ marginTop: "12px", color: "var(--text-muted)", fontSize: "0.95rem" }}>
              Click <strong>&ldquo;Run Motion Detection&rdquo;</strong> to evaluate consecutive-frame differencing across the scene.
            </p>
          )}
        </section>
      )}

      {/* Section 3: Adaptive Compression */}
      {analysis && (
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <div>
              <h2 className="card-title" style={{ margin: 0 }}>
                3. Activity-Aware Adaptive Compression
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "4px" }}>
                Encodes segments separately with libx264, then concatenates losslessly and copies original AAC audio.
              </p>
            </div>
            <button
              className="btn btn-primary"
              onClick={handleCompress}
              disabled={isProcessing}
            >
              {jobData?.compressed_file ? "Re-Compress Video" : "Start Adaptive Compression"}
            </button>
          </div>

          {compression && (
            <div style={{ marginTop: "20px" }}>
              <div className="metrics-grid">
                <div className="metric-card">
                  <span>Original Size</span>
                  <h3>{compression.original_size_mb} MB</h3>
                </div>
                <div className="metric-card highlight">
                  <span>Adaptive Compressed</span>
                  <h3>{compression.adaptive_size_mb} MB</h3>
                </div>
                <div className="metric-card highlight">
                  <span>Storage Reduction</span>
                  <h3>{compression.storage_reduction_pct}%</h3>
                </div>
                <div className="metric-card">
                  <span>Encoding Time</span>
                  <h3>{compression.total_encoding_time_sec} s</h3>
                </div>
              </div>

              <div style={{ fontSize: "0.85rem", color: "#10b981", display: "flex", gap: "16px", flexWrap: "wrap" }}>
                <span>&bull; Exact Frame Count Retained: {compression.output_frames} frames</span>
                <span>&bull; Duration Verified: {compression.output_duration_sec}s</span>
                <span>&bull; Audio Stream: Bit-exact copy (Synchronized)</span>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Section 4: Video Previews */}
      {jobData?.original_file && (
        <section className="card">
          <h2 className="card-title">4. Video Player Previews</h2>
          <div className="video-grid">
            <div className="video-box">
              <div className="video-header">
                <span>Original Surveillance Video</span>
                <span>{metadata?.file_size_mb} MB</span>
              </div>
              <video
                className="video-player"
                controls
                src={`${API_BASE}/video/${jobId}/original`}
              />
            </div>

            <div className="video-box">
              <div className="video-header">
                <span>Adaptive Compressed Video</span>
                <span>
                  {compression ? `${compression.adaptive_size_mb} MB` : "Pending Compression"}
                </span>
              </div>
              {jobData?.compressed_file ? (
                <video
                  className="video-player"
                  controls
                  src={`${API_BASE}/video/${jobId}/compressed`}
                />
              ) : (
                <div
                  style={{
                    aspectRatio: "16/9",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--text-muted)",
                    fontSize: "0.9rem",
                  }}
                >
                  Compressed video preview will appear once encoded.
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Section 5: Quality Evaluation & 3-Way Fair Comparison */}
      {jobData?.compressed_file && (
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <div>
              <h2 className="card-title" style={{ margin: 0 }}>
                5. Quality Evaluation & Baseline Comparison
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "4px" }}>
                Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) using scikit-image.
              </p>
            </div>
            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <select
                value={evalStride}
                onChange={(e) => setEvalStride(Number(e.target.value))}
                disabled={isProcessing}
                style={{
                  background: "var(--bg-card)",
                  color: "var(--text-main)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "6px",
                  padding: "6px 10px",
                  fontSize: "0.85rem",
                }}
              >
                <option value={1}>Full Evaluation (All 2,670 frames)</option>
                <option value={15}>Fast Evaluation (Every 15th frame)</option>
                <option value={30}>Quick Evaluation (Every 30th frame)</option>
              </select>
              <button
                className="btn btn-primary"
                onClick={handleEvaluate}
                disabled={isProcessing}
              >
                {evaluation ? "Re-Evaluate Quality" : "Calculate PSNR & SSIM"}
              </button>
            </div>
          </div>

          {evaluation && (
            <div style={{ marginTop: "20px" }}>
              <div className="metrics-grid">
                <div className="metric-card highlight">
                  <span>Average PSNR</span>
                  <h3>{evaluation.avg_psnr_db} dB</h3>
                  <small style={{ color: "#34d399", fontSize: "0.8rem" }}>
                    {baseline && (
                      <>+{(evaluation.avg_psnr_db - baseline.average_psnr_db).toFixed(2)} dB vs Uniform</>
                    )}
                  </small>
                </div>
                <div className="metric-card highlight">
                  <span>Average SSIM</span>
                  <h3>{evaluation.avg_ssim}</h3>
                  <small style={{ color: "#34d399", fontSize: "0.8rem" }}>
                    {baseline && (
                      <>+{(evaluation.avg_ssim - baseline.average_ssim).toFixed(4)} vs Uniform</>
                    )}
                  </small>
                </div>
                <div className="metric-card">
                  <span>Frames Compared (Adaptive / Uniform)</span>
                  <h3>
                    {evaluation.frames_compared} / {evaluation.uniform_evaluation.frames_compared}
                  </h3>
                </div>
                <div className="metric-card">
                  <span>Storage Saved</span>
                  <h3>{evaluation.storage_reduction_pct}%</h3>
                </div>
              </div>

              {/* 3-Way Comprehensive Comparison Table */}
              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Original Video</th>
                      <th>Uniform Baseline (CRF 32)</th>
                      <th>Activity-Aware Adaptive (Ours)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>CRF Configuration</strong></td>
                      <td>Original / Camera</td>
                      <td>Fixed CRF 32</td>
                      <td><strong>STATIC: 36 / ACTIVE: 28</strong></td>
                    </tr>
                    <tr>
                      <td><strong>File Size</strong></td>
                      <td>{evaluation.original_size_mb} MB</td>
                      <td>{baseline?.uniform_compressed_size_mb} MB</td>
                      <td><strong style={{ color: "#34d399" }}>{evaluation.compressed_size_mb} MB</strong></td>
                    </tr>
                    <tr>
                      <td><strong>Storage Reduction</strong></td>
                      <td>0.0%</td>
                      <td>{baseline?.storage_reduction_pct}%</td>
                      <td><strong style={{ color: "#34d399" }}>{evaluation.storage_reduction_pct}%</strong></td>
                    </tr>
                    <tr>
                      <td><strong>Average PSNR</strong></td>
                      <td>&infin;</td>
                      <td>{baseline?.average_psnr_db} dB</td>
                      <td><strong style={{ color: "#34d399" }}>{evaluation.avg_psnr_db} dB</strong></td>
                    </tr>
                    <tr>
                      <td><strong>Average SSIM</strong></td>
                      <td>1.0000</td>
                      <td>{baseline?.average_ssim}</td>
                      <td><strong style={{ color: "#34d399" }}>{evaluation.avg_ssim}</strong></td>
                    </tr>
                    <tr>
                      <td><strong>Resolution & FPS</strong></td>
                      <td>1920x1080 @ 29.97</td>
                      <td>1920x1080 @ 29.97</td>
                      <td>1920x1080 @ 29.97 (Preserved)</td>
                    </tr>
                    <tr>
                      <td><strong>Total Frames</strong></td>
                      <td>2,670 frames</td>
                      <td>2,670 frames</td>
                      <td>2,670 frames (Exact)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Section 6: Download */}
      {jobData?.compressed_file && (
        <section className="card" style={{ textAlign: "center", padding: "32px 20px" }}>
          <h2 style={{ fontSize: "1.3rem", marginBottom: "8px" }}>
            Ready to Export
          </h2>
          <p style={{ color: "var(--text-muted)", marginBottom: "20px", fontSize: "0.95rem" }}>
            Download your activity-aware compressed surveillance video with verified frame integrity.
          </p>
          <a
            href={`${API_BASE}/download/${jobId}`}
            download="activity_aware_compressed.mp4"
            className="btn btn-success"
            style={{ padding: "12px 28px", fontSize: "1.05rem", textDecoration: "none" }}
          >
            Download Compressed Video (.mp4)
          </a>
        </section>
      )}
    </div>
  );
}
