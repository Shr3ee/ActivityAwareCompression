"""
src/motion_detector.py
Scene-Level Motion Detection Module for Activity-Aware Video Compression.

Analyzes scene activity using OpenCV frame differencing on downscaled,
blurred grayscale frames. Classifies time intervals into STATIC or ACTIVE
segments with frame-accurate boundaries.
"""

import cv2
import numpy as np
import os
import sys
import time


def detect_motion_segments(
    video_path: str,
    sample_step: int = 5,                 # Sample every 5 frames (~6 fps analysis)
    motion_threshold_pct: float = 2.0,    # Percent of changed pixels to mark as ACTIVE
    min_segment_duration_sec: float = 3.0, # Minimum segment length to prevent micro-switching
    analysis_width: int = 640,
    analysis_height: int = 360,
    blur_kernel_size: int = 21,
    diff_threshold: int = 25
):
    """
    Analyzes video motion and segments it into continuous STATIC and ACTIVE intervals.

    Uses consecutive-frame differencing at sampled checkpoints to capture true
    instantaneous scene movement while eliminating slow camera drift or lighting accumulation.
    Frame sampling is ONLY used to evaluate scene motion over time.
    All segment boundaries map back to exact, contiguous original frame indices.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    print(f"\n[MotionDetector] Analyzing video: {video_path}")
    print(f"[MotionDetector] Resolution: {orig_width}x{orig_height}, FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration_sec:.2f}s")
    print(f"[MotionDetector] Sampling probe every {sample_step} frames (~{fps/sample_step:.1f} probes/sec)")
    print(f"[MotionDetector] Motion Threshold: {motion_threshold_pct:.2f}% | Min Segment Duration: {min_segment_duration_sec:.1f}s")

    total_pixels = analysis_width * analysis_height
    sampled_records = []
    start_time = time.time()

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame from video.")

    # Downscale and blur the first frame
    prev_gray = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(first_frame, (analysis_width, analysis_height)), cv2.COLOR_BGR2GRAY),
        (blur_kernel_size, blur_kernel_size),
        0
    )

    frame_idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Process motion checkpoint every sample_step frames
        if frame_idx % sample_step == 0:
            curr_gray = cv2.GaussianBlur(
                cv2.cvtColor(cv2.resize(curr_frame, (analysis_width, analysis_height)), cv2.COLOR_BGR2GRAY),
                (blur_kernel_size, blur_kernel_size),
                0
            )

            # Instantaneous difference between consecutive frames
            delta = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(delta, diff_threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            changed_pixels = cv2.countNonZero(dilated)
            motion_score = (changed_pixels / total_pixels) * 100.0

            timestamp = frame_idx / fps
            is_active = motion_score >= motion_threshold_pct

            sampled_records.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 3),
                "motion_score": round(motion_score, 3),
                "is_active": is_active
            })

            prev_gray = curr_gray
        else:
            # Keep prev_gray updated to immediate previous frame for consecutive diff
            prev_gray = cv2.GaussianBlur(
                cv2.cvtColor(cv2.resize(curr_frame, (analysis_width, analysis_height)), cv2.COLOR_BGR2GRAY),
                (blur_kernel_size, blur_kernel_size),
                0
            )

        frame_idx += 1

    cap.release()
    elapsed_analysis = time.time() - start_time
    print(f"[MotionDetector] Processed {len(sampled_records)} motion probe points across {total_frames} frames in {elapsed_analysis:.2f}s.")

    if not sampled_records:
        raise RuntimeError("No frames could be analyzed.")

    # Distribution statistics of raw motion scores
    raw_scores = [r["motion_score"] for r in sampled_records]
    stats = {
        "min": float(np.min(raw_scores)),
        "max": float(np.max(raw_scores)),
        "mean": float(np.mean(raw_scores)),
        "median": float(np.median(raw_scores)),
        "p25": float(np.percentile(raw_scores, 25)),
        "p75": float(np.percentile(raw_scores, 75)),
        "p90": float(np.percentile(raw_scores, 90)),
    }

    # Temporal smoothing: Rolling window of ~1.5 seconds to smooth out single-frame anomalies
    probes_per_sec = fps / sample_step
    window_pts = max(3, int(round(1.5 * probes_per_sec)))
    if window_pts % 2 == 0:
        window_pts += 1

    smoothed_classifications = []
    num_pts = len(sampled_records)

    for i in range(num_pts):
        start_w = max(0, i - window_pts // 2)
        end_w = min(num_pts, i + window_pts // 2 + 1)
        active_count = sum(1 for j in range(start_w, end_w) if sampled_records[j]["is_active"])
        ratio = active_count / (end_w - start_w)
        # Marked active if motion is present in at least 40% of the local window
        smoothed_classifications.append(ratio >= 0.40)

    # Initial grouping into continuous candidate blocks
    raw_blocks = []
    curr_label = smoothed_classifications[0]
    curr_start_idx = 0

    for i in range(1, num_pts):
        if smoothed_classifications[i] != curr_label:
            raw_blocks.append({
                "label": "ACTIVE" if curr_label else "STATIC",
                "start_sample_idx": curr_start_idx,
                "end_sample_idx": i - 1
            })
            curr_label = smoothed_classifications[i]
            curr_start_idx = i

    raw_blocks.append({
        "label": "ACTIVE" if curr_label else "STATIC",
        "start_sample_idx": curr_start_idx,
        "end_sample_idx": num_pts - 1
    })

    # Enforce minimum segment duration by merging short fluctuations
    min_samples = max(2, int(round(min_segment_duration_sec * probes_per_sec)))
    merged_blocks = []

    for block in raw_blocks:
        length = block["end_sample_idx"] - block["start_sample_idx"] + 1
        if not merged_blocks:
            merged_blocks.append(block)
        else:
            prev = merged_blocks[-1]
            if length < min_samples:
                # Merge short segment into preceding segment
                prev["end_sample_idx"] = block["end_sample_idx"]
            elif block["label"] == prev["label"]:
                # Merge adjacent segments with identical classification
                prev["end_sample_idx"] = block["end_sample_idx"]
            else:
                merged_blocks.append(block)

    # Final pass to merge any newly adjacent identical blocks
    final_blocks = []
    for block in merged_blocks:
        if not final_blocks:
            final_blocks.append(block)
        elif final_blocks[-1]["label"] == block["label"]:
            final_blocks[-1]["end_sample_idx"] = block["end_sample_idx"]
        else:
            final_blocks.append(block)

    # Map sample points back to exact, contiguous original frames [0 ... total_frames - 1]
    segments = []
    num_blocks = len(final_blocks)

    for b_idx, block in enumerate(final_blocks):
        # Start frame index
        if b_idx == 0:
            start_frame = 0
        else:
            prev_sample_frame = sampled_records[block["start_sample_idx"]]["frame_idx"]
            prev_block_last_sample = sampled_records[final_blocks[b_idx - 1]["end_sample_idx"]]["frame_idx"]
            start_frame = (prev_block_last_sample + prev_sample_frame) // 2 + 1

        # End frame index
        if b_idx == num_blocks - 1:
            end_frame = total_frames - 1
        else:
            curr_last_sample = sampled_records[block["end_sample_idx"]]["frame_idx"]
            next_first_sample = sampled_records[final_blocks[b_idx + 1]["start_sample_idx"]]["frame_idx"]
            end_frame = (curr_last_sample + next_first_sample) // 2

        # Motion scores inside this segment
        seg_scores = [
            r["motion_score"] for r in sampled_records
            if start_frame <= r["frame_idx"] <= end_frame
        ]
        avg_motion = float(np.mean(seg_scores)) if seg_scores else 0.0
        max_motion = float(np.max(seg_scores)) if seg_scores else 0.0

        start_time_sec = start_frame / fps
        end_time_sec = (end_frame + 1) / fps
        seg_duration = end_time_sec - start_time_sec
        seg_frame_count = end_frame - start_frame + 1

        segments.append({
            "segment_index": b_idx,
            "classification": block["label"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": seg_frame_count,
            "start_time": round(start_time_sec, 3),
            "end_time": round(end_time_sec, 3),
            "duration": round(seg_duration, 3),
            "avg_motion_score": round(avg_motion, 3),
            "max_motion_score": round(max_motion, 3)
        })

    # Validate that every single frame [0, total_frames - 1] is mapped with no gaps/overlaps
    total_segmented_frames = sum(s["frame_count"] for s in segments)
    assert total_segmented_frames == total_frames, (
        f"Frame count mismatch! Segmented: {total_segmented_frames}, Total: {total_frames}"
    )
    for i in range(len(segments) - 1):
        assert segments[i]["end_frame"] + 1 == segments[i + 1]["start_frame"], (
            f"Boundary gap/overlap between segment {i} and {i+1}!"
        )

    summary = {
        "video_path": video_path,
        "resolution": f"{orig_width}x{orig_height}",
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 3),
        "analysis_time_sec": round(elapsed_analysis, 2),
        "motion_threshold_pct": motion_threshold_pct,
        "sample_step": sample_step,
        "min_segment_duration_sec": min_segment_duration_sec,
        "motion_score_stats": stats,
        "segment_count": len(segments),
        "static_duration_sec": round(sum(s["duration"] for s in segments if s["classification"] == "STATIC"), 3),
        "active_duration_sec": round(sum(s["duration"] for s in segments if s["classification"] == "ACTIVE"), 3),
        "segments": segments
    }

    return summary


if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else "input/test_video.mp4"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    print(f"Running motion detection on: {target_video} (Threshold: {threshold}%)")
    result = detect_motion_segments(
        video_path=target_video,
        sample_step=5,
        motion_threshold_pct=threshold,
        min_segment_duration_sec=3.0
    )

    print("\n========== MOTION SCORE DISTRIBUTION ==========")
    stats = result["motion_score_stats"]
    print(f"Min: {stats['min']:.3f}%, Max: {stats['max']:.3f}%")
    print(f"Mean: {stats['mean']:.3f}%, Median: {stats['median']:.3f}%")
    print(f"25th percentile: {stats['p25']:.3f}%, 75th percentile: {stats['p75']:.3f}%, 90th percentile: {stats['p90']:.3f}%")

    print("\n========== DETECTED ACTIVITY SEGMENTS ==========")
    print(f"{'#':<3} {'Classification':<10} {'Start (s)':<10} {'End (s)':<10} {'Duration (s)':<13} {'Frames':<16} {'Avg Motion %':<12} {'Max Motion %':<12}")
    print("-" * 88)
    for s in result["segments"]:
        frames_str = f"{s['start_frame']}-{s['end_frame']} ({s['frame_count']})"
        print(f"{s['segment_index']:<3} {s['classification']:<10} {s['start_time']:<10.2f} {s['end_time']:<10.2f} {s['duration']:<13.2f} {frames_str:<16} {s['avg_motion_score']:<12.3f} {s['max_motion_score']:<12.3f}")

    print("\n========== SUMMARY ==========")
    print(f"Total video duration: {result['duration_sec']:.2f} s ({result['total_frames']} frames)")
    print(f"STATIC duration: {result['static_duration_sec']:.2f} s ({(result['static_duration_sec'] / result['duration_sec']) * 100:.1f}%)")
    print(f"ACTIVE duration: {result['active_duration_sec']:.2f} s ({(result['active_duration_sec'] / result['duration_sec']) * 100:.1f}%)")
    print(f"Total segments: {result['segment_count']}")
    print(f"Analysis completed in: {result['analysis_time_sec']:.2f} s")
    print("================================================\n")
