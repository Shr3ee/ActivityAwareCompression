"""
backend/main.py
FastAPI Server for Activity-Aware Video Compression Web Application.

Provides REST endpoints for:
- Video upload & validation
- Human activity detection, pose estimation & critical event assessment
- Adaptive compression based on activity importance (EMERGENCY / NORMAL HUMAN / IDLE)
- Objective video quality evaluation (PSNR, SSIM, Storage Reduction)
- Surveillance video preview streaming (Original, Adaptive Compressed, and Annotated Live Monitor)
- File download
"""

import os
import sys
import uuid
import json
import shutil
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

# Ensure project root is in sys.path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import cv2

from src.activity_detector import analyze_video_activities
from src.adaptive_compressor import compress_adaptive, get_stream_info
from src.evaluator import evaluate_videos

app = FastAPI(
    title="Human Activity Detection for Surveillance Video Compression API",
    description="Backend API for Human Activity-Aware Adaptive Surveillance Video Compression",
    version="2.0.0"
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# Thread pool executor for running CPU-heavy CV/FFmpeg tasks without blocking event loop
executor = ThreadPoolExecutor(max_workers=2)

# In-memory job state store
jobs: Dict[str, Dict[str, Any]] = {}

# Verified baseline experiment measurements
BASELINE_EXPERIMENT = {
    "original_size_mb": 60.45,
    "uniform_compressed_size_mb": 30.07,
    "storage_reduction_pct": 50.25,
    "average_psnr_db": 33.41,
    "average_ssim": 0.9247,
    "total_frames": 2670,
    "resolution": "1920x1080",
    "fps": 29.97,
    "duration_sec": 89.09,
    "codec": "H.264 / libx264",
    "preset": "medium",
    "crf": 32,
    "audio": "copied"
}


def extract_metadata(video_path: str) -> Dict[str, Any]:
    """Extracts resolution, fps, frame count, duration, and file size via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file with OpenCV.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    return {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "file_size_mb": round(size_mb, 2)
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "app": "Human Activity Detection for Surveillance Video Compression",
        "version": "2.0.0"
    }


@app.get("/api/baseline")
def get_baseline():
    """Returns the verified uniform compression baseline results."""
    return BASELINE_EXPERIMENT


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Handles video upload, validates format, streams to disk incrementally
    to avoid memory exhaustion, and extracts metadata.
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload an MP4 or compatible video."
        )

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    dest_path = os.path.join(job_dir, "original.mp4")

    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    try:
        metadata = extract_metadata(dest_path)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")

    jobs[job_id] = {
        "job_id": job_id,
        "status": "ready",
        "stage": "uploaded",
        "message": "Video uploaded and verified successfully.",
        "original_file": dest_path,
        "compressed_file": None,
        "annotated_file": None,
        "metadata": metadata,
        "analysis": None,
        "compression": None,
        "evaluation": None
    }

    return {"job_id": job_id, "metadata": metadata, "status": "ready"}


@app.post("/api/load-sample")
def load_sample_video():
    """
    Loads verified surveillance demo video for instant demonstration.
    """
    sample_path = os.path.join(PROJECT_ROOT, "input", "test_video.mp4")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(PROJECT_ROOT, "input", "test_video_89s.mp4")

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample video not found in input/ directory.")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    dest_path = os.path.join(job_dir, "original.mp4")
    shutil.copyfile(sample_path, dest_path)

    metadata = extract_metadata(dest_path)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "ready",
        "stage": "uploaded",
        "message": "Surveillance test video loaded successfully.",
        "original_file": dest_path,
        "compressed_file": None,
        "annotated_file": None,
        "metadata": metadata,
        "analysis": None,
        "compression": None,
        "evaluation": None
    }

    return {"job_id": job_id, "metadata": metadata, "status": "ready"}


def run_activity_analysis_task(job_id: str, sample_step: int = 5):
    job = jobs[job_id]
    job["status"] = "processing"
    job["stage"] = "analyzing"
    job["message"] = "Detecting human pose keypoints, activity patterns, and critical events..."

    try:
        job_dir = os.path.join(JOBS_DIR, job_id)
        raw_preview = os.path.join(job_dir, "raw_annotated.mp4")
        final_annotated = os.path.join(job_dir, "annotated_surveillance.mp4")

        # Run full human activity detection, pose estimation, and critical object analysis
        result = analyze_video_activities(
            video_path=job["original_file"],
            sample_step=sample_step,
            output_annotated_preview=raw_preview
        )

        # Transcode annotated preview with libx264 for universal HTML5 browser playback
        if os.path.exists(raw_preview):
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", raw_preview,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "26",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                final_annotated
            ], check=True)
            job["annotated_file"] = final_annotated
            try:
                os.remove(raw_preview)
            except Exception:
                pass

        job["analysis"] = result
        job["status"] = "ready"
        job["stage"] = "analyzed"
        job["message"] = (
            f"Activity assessment complete! Classified {result['segment_count']} segments: "
            f"Emergency ({result['emergency_duration_sec']}s), "
            f"Normal Human ({result['normal_duration_sec']}s), "
            f"Idle ({result['idle_duration_sec']}s)."
        )
    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Activity detection failed: {str(e)}"


@app.post("/api/job/{job_id}/detect-activities")
@app.post("/api/job/{job_id}/analyze")
async def start_activity_detection(job_id: str, sample_step: int = Query(5, ge=1, le=30)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, run_activity_analysis_task, job_id, sample_step)
    return {"status": "started", "stage": "analyzing"}


def run_compression_task(job_id: str):
    job = jobs[job_id]
    job["status"] = "processing"
    job["stage"] = "compressing"
    job["message"] = "Executing adaptive compression (EMERGENCY: 30 FPS / CRF 24, NORMAL: 10 FPS / CRF 32, IDLE: 5 FPS / CRF 36)..."

    try:
        job_dir = os.path.join(JOBS_DIR, job_id)
        output_path = os.path.join(job_dir, "compressed.mp4")
        temp_dir = os.path.join(job_dir, "temp_segments")

        result = compress_adaptive(
            input_video=job["original_file"],
            output_video=output_path,
            analysis_result=job.get("analysis"),
            emergency_crf=24,
            normal_crf=32,
            idle_crf=36,
            preset="medium",
            temp_dir=temp_dir
        )

        job["compressed_file"] = output_path
        job["compression"] = result
        job["status"] = "ready"
        job["stage"] = "compressed"
        job["message"] = f"Adaptive compression complete! Storage reduced by {result['storage_reduction_pct']}%."
    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Compression failed: {str(e)}"


@app.post("/api/job/{job_id}/compress")
async def start_compression(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, run_compression_task, job_id)
    return {"status": "started", "stage": "compressing"}


def run_evaluation_task(job_id: str, stride: int):
    job = jobs[job_id]
    job["status"] = "processing"
    job["stage"] = "evaluating"
    job["message"] = f"Calculating objective PSNR & SSIM metrics (sampling stride: {stride})..."

    try:
        if not job.get("compressed_file") or not os.path.exists(job["compressed_file"]):
            raise RuntimeError("Compressed video file does not exist. Please compress first.")

        original_path = job["original_file"]
        job_dir = os.path.join(JOBS_DIR, job_id)
        uniform_compressed_path = os.path.join(job_dir, "uniform_compressed.mp4")

        # Encode uniform baseline (Fixed CRF 32) if not already generated
        if not os.path.exists(uniform_compressed_path):
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", original_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "32",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    "-y",
                    uniform_compressed_path
                ],
                check=True
            )

        result = evaluate_videos(
            original_path=original_path,
            compressed_path=job["compressed_file"],
            sample_stride=stride
        )
        result["uniform_evaluation"] = evaluate_videos(
            original_path=original_path,
            compressed_path=uniform_compressed_path,
            sample_stride=stride
        )

        job["evaluation"] = result
        job["status"] = "ready"
        job["stage"] = "evaluated"
        job["message"] = f"Evaluation complete. PSNR: {result['avg_psnr_db']} dB, SSIM: {result['avg_ssim']}."
    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Evaluation failed: {str(e)}"


@app.post("/api/job/{job_id}/evaluate")
async def start_evaluation(job_id: str, stride: int = Query(15, ge=1, le=100)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, run_evaluation_task, job_id, stride)
    return {"status": "started", "stage": "evaluating"}


@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs[job_id]


@app.api_route("/api/video/{job_id}/{kind}", methods=["GET", "HEAD"])
def stream_video(job_id: str, kind: str):
    """Streams original, compressed, annotated surveillance preview, or baseline video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    if kind == "original":
        path = job["original_file"]
    elif kind == "compressed":
        path = job.get("compressed_file")
    elif kind == "annotated":
        path = job.get("annotated_file")
    elif kind == "baseline":
        path = os.path.join(PROJECT_ROOT, "output", "uniform_compressed.mp4")
    else:
        raise HTTPException(status_code=400, detail="Invalid video kind.")

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video file not found.")

    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))


@app.api_route("/api/download/{job_id}", methods=["GET", "HEAD"])
def download_compressed_video(job_id: str):
    """Download endpoint for the final compressed video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    comp_file = job.get("compressed_file")
    if not comp_file or not os.path.exists(comp_file):
        raise HTTPException(status_code=404, detail="Compressed video not ready for download.")

    return FileResponse(
        comp_file,
        media_type="video/mp4",
        filename="activity_aware_compressed.mp4",
        headers={"Content-Disposition": "attachment; filename=activity_aware_compressed.mp4"}
    )
