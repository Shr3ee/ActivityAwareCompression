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
  const [activeVideoTab, setActiveVideoTab] = useState("compare"); // "compare" | "annotated"
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
      const jRes = await fetch(`${API_BASE}/job/${data.job_id}`);
      setJobData(await jRes.json());
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Quick load of the verified surveillance test video
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

  // Trigger Human Activity Detection & Critical Event Assessment
  const handleDetectActivities = async () => {
    if (!jobId) return;
    setErrorMessage(null);
    try {
      const res = await fetch(`${API_BASE}/job/${jobId}/detect-activities?sample_step=5`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to trigger activity detection.");
      setJobData((prev) => ({
        ...prev,
        status: "processing",
        stage: "analyzing",
        message: "Detecting human pose keypoints, activity patterns, and critical events...",
      }));
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // Trigger Adaptive Compression
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
        message: "Executing adaptive compression (EMERGENCY: 30 FPS / CRF 24, NORMAL: 10 FPS / CRF 32, IDLE: 5 FPS / CRF 36)...",
      }));
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // Trigger Objective Quality Evaluation (PSNR / SSIM)
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
        message: "Calculating PSNR & SSIM metrics against source...",
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
            Human Activity Detection for Surveillance Video Compression
          </h1>
          <p>
            Activity importance guided adaptive compression with YOLOv8 Pose Estimation,
            Critical Alert Assessment, and FFmpeg Encoding.
          </p>
        </div>
        <div className="header-badges">
          <span className="college-badge">ABES Engineering College</span>
          <span className="tag-badge">Mini Project BCS554</span>
          <span className="tag-badge">YOLOv8 + OpenCV + FFmpeg</span>
        </div>
      </header>

      {/* Methodology Pipeline Flow */}
      <div className="pipeline-flow">
        <div className="pipeline-step">
          <span className="num">1</span>
          <span>Detect Activity (Pose &amp; Objects)</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step">
          <span className="num">2</span>
          <span>Assess Importance (Emergency / Normal / Idle)</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step">
          <span className="num">3</span>
          <span>Adaptive Compression (Variable FPS &amp; CRF)</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step">
          <span className="num">4</span>
          <span>Storage Optimization &amp; PSNR/SSIM Eval</span>
        </div>
      </div>

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

      {/* Section 1: Video Input & Inspection */}
      <section className="card">
        <h2 className="card-title">1. Surveillance Video Input &amp; Inspection</h2>
        <div className="upload-grid">
          <div
            className="drop-zone"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept="video/mp4,video/avi,video/mkv,video/*"
              style={{ display: "none" }}
              onChange={(e) => {
                if (e.target.files?.[0]) handleUpload(e.target.files[0]);
              }}
            />
            <p style={{ fontSize: "1.1rem", fontWeight: "600" }}>
              {isUploading ? "Uploading surveillance video..." : "Drop Surveillance Video Here or Click to Browse"}
            </p>
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "6px" }}>
              Supports MP4, AVI, MKV surveillance video recordings
            </p>
          </div>

          <div className="sample-button-box">
            <p style={{ fontSize: "0.9rem", color: "var(--text-muted)", marginBottom: "12px" }}>
              Mini Project Verified Surveillance Video:
            </p>
            <button
              className="btn btn-secondary"
              onClick={handleLoadSample}
              disabled={isUploading || isProcessing}
            >
              Load Demo Surveillance Video (89s)
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
              <span>Native FPS</span>
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

      {/* Section 2: Human Activity Detection & Importance Assessment */}
      {metadata && (
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <div>
              <h2 className="card-title" style={{ margin: 0 }}>
                2. Human Activity Detection &amp; Importance Assessment
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "4px" }}>
                Extracts 17 body pose landmarks, classifies activity (Sitting, Standing, Walking),
                detects critical objects/hazards, and assigns operational modes.
              </p>
            </div>
            <button
              className="btn btn-primary"
              onClick={handleDetectActivities}
              disabled={isProcessing}
            >
              {analysis ? "Re-Run Activity Detection" : "Run Activity & Importance Assessment"}
            </button>
          </div>

          {analysis ? (
            <div className="timeline-container">
              {/* Graphical Timeline Bar */}
              <div className="timeline-bar">
                {analysis.segments.map((seg) => {
                  const widthPct = (seg.duration / analysis.duration_sec) * 100;
                  const segClass = seg.mode === "EMERGENCY" ? "EMERGENCY" : seg.mode === "NORMAL HUMAN" ? "NORMAL-HUMAN" : "IDLE";
                  return (
                    <div
                      key={seg.segment_index}
                      className={`timeline-seg ${segClass}`}
                      style={{ width: `${widthPct}%` }}
                      title={`Segment ${seg.segment_index}: ${seg.mode} (${seg.activity}) | Duration: ${seg.duration}s | Target: ${seg.target_fps} FPS, CRF ${seg.target_crf}`}
                    >
                      {widthPct > 9 && (
                        <span>
                          {seg.mode} ({seg.duration}s)
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Legend with exact parameters matching Page 6 */}
              <div className="timeline-legend">
                <div className="legend-item">
                  <div className="legend-box emergency"></div>
                  <span>
                    <strong>EMERGENCY MODE:</strong> {analysis.emergency_duration_sec}s &rarr;{" "}
                    <strong>30 FPS (Full)</strong>, Allocated <strong>CRF 24</strong> (High Detail)
                  </span>
                </div>
                <div className="legend-item">
                  <div className="legend-box normal"></div>
                  <span>
                    <strong>NORMAL HUMAN MODE:</strong> {analysis.normal_duration_sec}s &rarr;{" "}
                    <strong>10 FPS</strong>, Allocated <strong>CRF 32</strong> (Sitting / Standing / Walking)
                  </span>
                </div>
                <div className="legend-item">
                  <div className="legend-box idle"></div>
                  <span>
                    <strong>IDLE MODE:</strong> {analysis.idle_duration_sec}s &rarr;{" "}
                    <strong>5 FPS</strong>, Allocated <strong>CRF 36</strong> (Static scene, high compression)
                  </span>
                </div>
              </div>

              {/* Segments Table */}
              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Mode</th>
                      <th>Detected Activity</th>
                      <th>Importance</th>
                      <th>Start (s)</th>
                      <th>End (s)</th>
                      <th>Duration</th>
                      <th>Target FPS</th>
                      <th>Allocated CRF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.segments.map((s) => (
                      <tr key={s.segment_index}>
                        <td>{s.segment_index + 1}</td>
                        <td>
                          <span
                            className={`badge ${
                              s.mode === "EMERGENCY"
                                ? "badge-emergency"
                                : s.mode === "NORMAL HUMAN"
                                ? "badge-normal"
                                : "badge-idle"
                            }`}
                          >
                            {s.mode}
                          </span>
                        </td>
                        <td>
                          <strong>{s.activity}</strong>
                        </td>
                        <td>{s.importance}</td>
                        <td>{s.start_time}s</td>
                        <td>{s.end_time}s</td>
                        <td>{s.duration}s</td>
                        <td>{s.target_fps} FPS</td>
                        <td>
                          <strong>CRF {s.target_crf}</strong>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <p style={{ marginTop: "14px", color: "var(--text-muted)", fontSize: "0.95rem" }}>
              Click <strong>&ldquo;Run Activity &amp; Importance Assessment&rdquo;</strong> to run YOLOv8 Pose Estimation,
              body landmark tracking, and critical alert detection.
            </p>
          )}
        </section>
      )}

      {/* Section 3: Adaptive Video Compression */}
      {analysis && (
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <div>
              <h2 className="card-title" style={{ margin: 0 }}>
                3. Activity-Aware Adaptive Compression
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "4px" }}>
                Applies variable framerates and adaptive CRF encoding per activity tier, then losslessly concatenates into an optimized MP4.
              </p>
            </div>
            <button
              className="btn btn-primary"
              onClick={handleCompress}
              disabled={isProcessing}
            >
              {jobData?.compressed_file ? "Re-Run Compression" : "Start Adaptive Compression"}
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

              <div style={{ fontSize: "0.85rem", color: "#10b981", display: "flex", gap: "18px", flexWrap: "wrap" }}>
                <span>&bull; Exact Frame Integrity Retained: {compression.output_frames} frames</span>
                <span>&bull; Verified Duration: {compression.output_duration_sec}s</span>
                <span>&bull; Video Codec: H.264 / libx264 (Universally Playable)</span>
                <span>&bull; Audio: Preserved and Synchronized</span>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Section 4: Video Players & Surveillance Monitoring (Page 6 Reference) */}
      {jobData?.original_file && (
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px", marginBottom: "16px" }}>
            <div>
              <h2 className="card-title" style={{ margin: 0 }}>
                4. Surveillance Video Previews &amp; Real-Time Monitoring
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "4px" }}>
                Inspect original footage, activity-aware compressed stream, and live AI surveillance annotations.
              </p>
            </div>
            {jobData?.annotated_file && (
              <div className="video-tabs">
                <button
                  className={`video-tab-btn ${activeVideoTab === "compare" ? "active" : ""}`}
                  onClick={() => setActiveVideoTab("compare")}
                >
                  Side-by-Side Comparison
                </button>
                <button
                  className={`video-tab-btn ${activeVideoTab === "annotated" ? "active" : ""}`}
                  onClick={() => setActiveVideoTab("annotated")}
                >
                  AI Surveillance Live Monitor (Page 6 View)
                </button>
              </div>
            )}
          </div>

          {activeVideoTab === "annotated" && jobData?.annotated_file ? (
            <div className="video-box" style={{ maxWidth: "800px", margin: "0 auto" }}>
              <div className="video-header">
                <span>AI Surveillance Live Monitor (Pose Skeleton &amp; Critical Alert Overlay)</span>
                <span style={{ color: "#34d399" }}>Implementation Reference: Page 6</span>
              </div>
              <video
                className="video-player"
                controls
                autoPlay
                loop
                src={`${API_BASE}/video/${jobId}/annotated`}
              />
            </div>
          ) : (
            <div className="video-grid">
              <div className="video-box">
                <div className="video-header">
                  <span>Original Surveillance Footage</span>
                  <span>{metadata?.file_size_mb} MB ({metadata?.fps} FPS)</span>
                </div>
                <video
                  className="video-player"
                  controls
                  src={`${API_BASE}/video/${jobId}/original`}
                />
              </div>

              <div className="video-box">
                <div className="video-header">
                  <span>Activity-Aware Compressed Surveillance</span>
                  <span>
                    {compression ? `${compression.adaptive_size_mb} MB (${compression.storage_reduction_pct}% Saved)` : "Pending Compression"}
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
                    Compressed surveillance preview will appear once encoded.
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      )}

      {/* Section 5: Objective Quality Evaluation & 3-Way Comparative Analysis */}
      {jobData?.compressed_file && (
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <div>
              <h2 className="card-title" style={{ margin: 0 }}>
                5. Objective Quality Evaluation &amp; Baseline Comparison
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
                <option value={15}>Fast Evaluation (Every 15th frame)</option>
                <option value={30}>Quick Evaluation (Every 30th frame)</option>
                <option value={1}>Full Evaluation (Every frame)</option>
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
                  <span>Frames Evaluated</span>
                  <h3>{evaluation.frames_compared} frames</h3>
                </div>
                <div className="metric-card">
                  <span>Storage Saved</span>
                  <h3>{evaluation.storage_reduction_pct}%</h3>
                </div>
              </div>

              {/* 3-Way Comparative Benchmark Table */}
              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Original Surveillance Source</th>
                      <th>Uniform Baseline (CRF 32)</th>
                      <th>Activity-Aware Adaptive (Proposed)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>Compression Methodology</strong></td>
                      <td>Uncompressed / Direct Camera</td>
                      <td>Fixed Uniform CRF 32</td>
                      <td><strong>Adaptive (Emergency: 24, Normal: 32, Idle: 36)</strong></td>
                    </tr>
                    <tr>
                      <td><strong>Framerate Allocation</strong></td>
                      <td>Fixed {metadata?.fps} FPS</td>
                      <td>Fixed {metadata?.fps} FPS</td>
                      <td><strong>Dynamic: 30 FPS / 10 FPS / 5 FPS</strong></td>
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
                      <td><strong>Video Resolution</strong></td>
                      <td>{metadata?.resolution}</td>
                      <td>{metadata?.resolution}</td>
                      <td>{metadata?.resolution} (Preserved)</td>
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
            Ready to Export Compressed Surveillance Footage
          </h2>
          <p style={{ color: "var(--text-muted)", marginBottom: "20px", fontSize: "0.95rem" }}>
            Download the activity-aware compressed surveillance video with verified frame integrity.
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
