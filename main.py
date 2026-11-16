"""
main.py
CLI Entrypoint for Human Activity Detection for Surveillance Video Compression.

Usage:
    python main.py [input_video] [output_video]
"""

import os
import sys
from src.activity_detector import analyze_video_activities
from src.adaptive_compressor import compress_adaptive
from src.evaluator import evaluate_videos


def main():
    input_video = sys.argv[1] if len(sys.argv) > 1 else "input/test_video.mp4"
    output_video = sys.argv[2] if len(sys.argv) > 2 else "output/adaptive_compressed.mp4"
    annotated_video = "output/annotated_surveillance.mp4"

    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found.")
        sys.exit(1)

    print("==================================================================")
    print(" HUMAN ACTIVITY DETECTION FOR SURVEILLANCE VIDEO COMPRESSION")
    print(" ABES Engineering College | Mini Project BCS554")
    print("==================================================================")
    print(f"Target Video: {input_video}\n")

    # Phase 1: Activity Detection & Importance Assessment
    print("[Step 1/3] Running Human Activity Detection & Importance Assessment...")
    analysis = analyze_video_activities(
        video_path=input_video,
        sample_step=10,
        output_annotated_preview=annotated_video
    )
    print(f"Identified {analysis['segment_count']} operational segments:")
    print(f" - Emergency Mode:    {analysis['emergency_duration_sec']}s (30 FPS, CRF 24)")
    print(f" - Normal Human Mode: {analysis['normal_duration_sec']}s (10 FPS, CRF 32)")
    print(f" - Idle Mode:         {analysis['idle_duration_sec']}s (5 FPS, CRF 36)")

    # Phase 2: Adaptive Compression
    print("\n[Step 2/3] Executing Activity-Aware Adaptive Compression...")
    comp_result = compress_adaptive(
        input_video=input_video,
        output_video=output_video,
        analysis_result=analysis,
        preset="faster"
    )
    print(f"Compressed Output: {output_video}")
    print(f"Original Size:     {comp_result['original_size_mb']} MB")
    print(f"Compressed Size:   {comp_result['adaptive_size_mb']} MB")
    print(f"Storage Reduction: {comp_result['storage_reduction_pct']}%")

    # Phase 3: Objective Evaluation
    print("\n[Step 3/3] Evaluating Video Quality (PSNR & SSIM)...")
    eval_result = evaluate_videos(
        original_path=input_video,
        compressed_path=output_video,
        sample_stride=15
    )
    print(f"Average PSNR:      {eval_result['avg_psnr_db']} dB")
    print(f"Average SSIM:      {eval_result['avg_ssim']}")
    print(f"Storage Saved:     {eval_result['storage_reduction_pct']}%")

    print("\n[SUCCESS] Pipeline executed successfully!")
    print(f"Annotated Preview: {annotated_video}")
    print(f"Compressed Video:  {output_video}")
    print("==================================================================")


if __name__ == "__main__":
    main()
