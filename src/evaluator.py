"""
src/evaluator.py
Objective Video Quality Evaluator (PSNR & SSIM) and Storage Analysis.

Evaluates compressed surveillance footage against the original source using
scikit-image peak_signal_noise_ratio and structural_similarity.
Optimized for responsive web evaluation.
"""

import os
import sys
import time
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict, Any


def evaluate_videos(
    original_path: str,
    compressed_path: str,
    sample_stride: int = 15,
    eval_resolution: tuple = (640, 360)
) -> Dict[str, Any]:
    """
    Evaluates PSNR, SSIM, file size, and storage reduction between original and compressed videos.
    """
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original video not found: {original_path}")
    if not os.path.exists(compressed_path):
        raise FileNotFoundError(f"Compressed video not found: {compressed_path}")

    orig_cap = cv2.VideoCapture(original_path)
    comp_cap = cv2.VideoCapture(compressed_path)

    if not orig_cap.isOpened():
        raise RuntimeError(f"Could not open original video: {original_path}")
    if not comp_cap.isOpened():
        raise RuntimeError(f"Could not open compressed video: {compressed_path}")

    orig_size_mb = os.path.getsize(original_path) / (1024 * 1024)
    comp_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
    storage_reduction = max(0.0, ((orig_size_mb - comp_size_mb) / orig_size_mb) * 100.0)

    psnr_values = []
    ssim_values = []

    print(f"\n[Evaluator] Comparing:")
    print(f"  Original:   {original_path} ({orig_size_mb:.2f} MB)")
    print(f"  Compressed: {compressed_path} ({comp_size_mb:.2f} MB)")
    print(f"  Storage Reduction: {storage_reduction:.2f}%")
    print(f"  Sample stride: {sample_stride}")

    t_start = time.time()
    frame_counter = 0

    while True:
        ret1, frame1 = orig_cap.read()
        ret2, frame2 = comp_cap.read()

        if not ret1 or not ret2:
            break

        if frame_counter % sample_stride == 0:
            # Resize both to standardized evaluation resolution for fast and fair perceptual metrics
            frame1_std = cv2.resize(frame1, eval_resolution)
            frame2_std = cv2.resize(frame2, eval_resolution)

            # Convert BGR to RGB for accurate perceptual SSIM / PSNR
            frame1_rgb = cv2.cvtColor(frame1_std, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2_std, cv2.COLOR_BGR2RGB)

            psnr = peak_signal_noise_ratio(frame1_rgb, frame2_rgb, data_range=255)
            ssim = structural_similarity(frame1_rgb, frame2_rgb, channel_axis=2, data_range=255)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

        frame_counter += 1

    orig_cap.release()
    comp_cap.release()

    eval_duration = time.time() - t_start
    avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0

    print(f"\n========== QUALITY EVALUATION RESULTS ==========")
    print(f"Frames compared: {len(psnr_values)}")
    print(f"Evaluation time: {eval_duration:.2f}s")
    print(f"Average PSNR:    {avg_psnr:.2f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")
    print(f"Storage Saved:   {storage_reduction:.2f}%")
    print("================================================\n")

    return {
        "original_path": original_path,
        "compressed_path": compressed_path,
        "original_size_mb": round(orig_size_mb, 2),
        "compressed_size_mb": round(comp_size_mb, 2),
        "storage_reduction_pct": round(storage_reduction, 2),
        "frames_compared": len(psnr_values),
        "avg_psnr_db": round(avg_psnr, 2),
        "avg_ssim": round(avg_ssim, 4),
        "evaluation_time_sec": round(eval_duration, 2)
    }


if __name__ == "__main__":
    orig = sys.argv[1] if len(sys.argv) > 1 else "input/test_video.mp4"
    comp = sys.argv[2] if len(sys.argv) > 2 else "output/adaptive_compressed.mp4"
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    evaluate_videos(orig, comp, sample_stride=stride)
