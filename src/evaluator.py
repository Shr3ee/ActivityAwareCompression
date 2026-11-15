"""
src/evaluator.py
Evaluation Module for Video Quality (PSNR & SSIM) and Storage Reduction.

Fairly evaluates compressed videos against the original input video using
the exact same skimage peak_signal_noise_ratio and structural_similarity
methods established in src/quality_metrics.py.
"""

import cv2
import numpy as np
import os
import sys
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate_videos(
    original_path: str = "input/test_video.mp4",
    compressed_path: str = "output/adaptive_compressed.mp4",
    sample_stride: int = 1  # 1 = evaluate every frame; >1 = uniform frame subsample
):
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
    storage_reduction = ((orig_size_mb - comp_size_mb) / orig_size_mb) * 100.0

    psnr_values = []
    ssim_values = []

    print(f"\n[Evaluator] Comparing:")
    print(f"  Original:   {original_path} ({orig_size_mb:.2f} MB)")
    print(f"  Compressed: {compressed_path} ({comp_size_mb:.2f} MB)")
    print(f"  Storage Reduction: {storage_reduction:.2f}%")
    print(f"  Sample stride: {sample_stride} (every {sample_stride} frame(s))")

    t_start = time.time()
    frame_counter = 0

    while True:
        ret1, frame1 = orig_cap.read()
        ret2, frame2 = comp_cap.read()

        if not ret1 or not ret2:
            break

        if frame_counter % sample_stride == 0:
            # Resize if shapes differ
            if frame1.shape != frame2.shape:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

            # Convert BGR to RGB
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            psnr = peak_signal_noise_ratio(frame1_rgb, frame2_rgb, data_range=255)
            ssim = structural_similarity(frame1_rgb, frame2_rgb, channel_axis=2, data_range=255)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if len(psnr_values) % 100 == 0:
                print(f"  Evaluated {len(psnr_values)} frames...", flush=True)

        frame_counter += 1

    orig_cap.release()
    comp_cap.release()

    eval_duration = time.time() - t_start

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0

    print(f"\n========== QUALITY EVALUATION RESULTS ==========")
    print(f"Frames compared: {len(psnr_values)}")
    print(f"Evaluation time: {eval_duration:.2f} seconds")
    print(f"Original Size:   {orig_size_mb:.2f} MB")
    print(f"Compressed Size: {comp_size_mb:.2f} MB")
    print(f"Storage Reduction: {storage_reduction:.2f}%")
    print(f"Average PSNR:    {avg_psnr:.2f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")
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
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    evaluate_videos(orig, comp, sample_stride=stride)
