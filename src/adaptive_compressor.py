"""
src/adaptive_compressor.py
Adaptive Activity-Aware Video Compression Module.

Encodes STATIC segments with CRF 36 and ACTIVE segments with CRF 28 using FFmpeg libx264.
Concatenates segments frame-accurately, preserves the original audio track,
and validates frame count, duration, resolution, and audio integrity.
"""

import os
import sys
import time
import shutil
import subprocess
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.motion_detector import detect_motion_segments


def get_stream_info(video_path: str):
    """Inspects video and audio stream properties using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json",
        video_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    probe_data = json.loads(res.stdout)
    
    video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)
    
    return {
        "video": video_stream,
        "audio": audio_stream,
        "format": probe_data.get("format", {})
    }


def compress_adaptive(
    input_video: str,
    output_video: str = "output/adaptive_compressed.mp4",
    static_crf: int = 36,
    active_crf: int = 28,
    preset: str = "medium",
    threshold: float = 2.0,
    temp_dir: str = "output/temp_segments"
):
    """
    Executes activity-aware adaptive compression.
    STATIC segments: static_crf (default 36)
    ACTIVE segments: active_crf (default 28)
    """
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    print("\n================ ADAPTIVE VIDEO COMPRESSION ================")
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"CRF Settings: STATIC = {static_crf}, ACTIVE = {active_crf}")
    print(f"Preset: {preset}")

    # Step 1: Inspect original stream properties
    info_before = get_stream_info(input_video)
    v_info = info_before["video"]
    a_info = info_before["audio"]
    
    orig_frames = int(v_info.get("nb_frames", 0))
    orig_duration = float(v_info.get("duration", 0.0))
    orig_fps = v_info.get("r_frame_rate", "30000/1001")
    orig_width = int(v_info.get("width", 1920))
    orig_height = int(v_info.get("height", 1080))
    has_audio = a_info is not None

    print(f"Original Video: {orig_width}x{orig_height}, {orig_fps} fps, {orig_frames} frames, {orig_duration:.2f}s")
    if has_audio:
        print(f"Original Audio: {a_info.get('codec_name')} ({a_info.get('sample_rate')} Hz, {a_info.get('channels')} channels)")
    else:
        print("Original Audio: None")

    # Step 2: Detect motion segments
    print("\n--- Running Scene-Level Motion Detection ---")
    motion_result = detect_motion_segments(
        video_path=input_video,
        sample_step=5,
        motion_threshold_pct=threshold,
        min_segment_duration_sec=3.0
    )
    segments = motion_result["segments"]
    print(f"Generated {len(segments)} segment(s) for compression.")

    # Step 3: Encode each segment
    segment_files = []
    t_start_encoding = time.time()

    for idx, seg in enumerate(segments):
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        classification = seg["classification"]
        crf = static_crf if classification == "STATIC" else active_crf
        
        seg_out = os.path.join(temp_dir, f"segment_{idx:03d}_{classification.lower()}.mp4")
        segment_files.append(seg_out)

        print(f"\nEncoding Segment {idx+1}/{len(segments)}: [{classification}] Frames {start_frame}-{end_frame} ({seg['frame_count']} frames) @ CRF {crf}...")

        # Frame-accurate extraction and encoding via select filter
        select_filter = f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=N/FRAME_RATE/TB"

        cmd_seg = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_video,
            "-vf", select_filter,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-r", str(orig_fps),
            "-pix_fmt", "yuv420p",
            "-an",
            seg_out
        ]

        t0_seg = time.time()
        subprocess.run(cmd_seg, check=True)
        seg_duration = time.time() - t0_seg
        seg_size_mb = os.path.getsize(seg_out) / (1024 * 1024)
        print(f"Segment {idx+1} encoded in {seg_duration:.2f}s (Size: {seg_size_mb:.2f} MB)")

    # Step 4: Write concatenation list
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for seg_file in segment_files:
            # Concat demuxer works best with relative or escaped paths
            abs_path = os.path.abspath(seg_file).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    # Step 5: Concatenate video streams and mux original audio
    print("\n--- Concatenating Segments & Muxing Audio ---")
    if has_audio:
        cmd_merge = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-i", input_video,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_video
        ]
    else:
        cmd_merge = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c:v", "copy",
            "-movflags", "+faststart",
            output_video
        ]

    subprocess.run(cmd_merge, check=True)
    total_encoding_time = time.time() - t_start_encoding
    print(f"Final output assembled in {total_encoding_time:.2f}s")

    # Step 6: Clean up temporary segment files
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary segment files cleaned up.")
    except Exception as e:
        print(f"Warning: Could not remove temp folder: {e}")

    # Step 7: Post-compression validation
    print("\n--- Validating Final Output Integrity ---")
    info_after = get_stream_info(output_video)
    v_out = info_after["video"]
    a_out = info_after["audio"]

    out_frames_raw = v_out.get("nb_frames")
    # For accurate frame count in MP4 containers
    cmd_count = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", output_video
    ]
    out_frames_exact = int(subprocess.check_output(cmd_count, text=True).strip())
    out_duration = float(v_out.get("duration", 0.0))
    out_width = int(v_out.get("width", 0))
    out_height = int(v_out.get("height", 0))
    orig_size_mb = os.path.getsize(input_video) / (1024 * 1024)
    out_size_mb = os.path.getsize(output_video) / (1024 * 1024)
    storage_reduction = ((orig_size_mb - out_size_mb) / orig_size_mb) * 100

    print(f"Output Video: {out_width}x{out_height}, {v_out.get('r_frame_rate')} fps, {out_frames_exact} frames, {out_duration:.2f}s")
    print(f"Original size: {orig_size_mb:.2f} MB -> Adaptive size: {out_size_mb:.2f} MB (Reduction: {storage_reduction:.2f}%)")

    # Rigorous validation checks
    assert out_width == orig_width and out_height == orig_height, (
        f"Resolution mismatch! Expected {orig_width}x{orig_height}, got {out_width}x{out_height}"
    )
    assert out_frames_exact == orig_frames, (
        f"Frame count mismatch! Expected {orig_frames}, got {out_frames_exact}"
    )
    assert abs(out_duration - orig_duration) < 0.2, (
        f"Duration mismatch! Expected {orig_duration:.2f}s, got {out_duration:.2f}s"
    )
    if has_audio:
        assert a_out is not None, "Audio stream missing in output!"
        assert a_out.get("codec_name") == a_info.get("codec_name"), "Audio codec changed!"
        print("Audio integrity: Verified (Audio stream retained with original codec and sample rate).")

    print("\n[SUCCESS] Adaptive video compression completed and validated successfully!")
    print(f"Output saved to: {output_video}")
    print("============================================================\n")

    return {
        "output_path": output_video,
        "original_size_mb": round(orig_size_mb, 2),
        "adaptive_size_mb": round(out_size_mb, 2),
        "storage_reduction_pct": round(storage_reduction, 2),
        "total_encoding_time_sec": round(total_encoding_time, 2),
        "original_frames": orig_frames,
        "output_frames": out_frames_exact,
        "original_duration_sec": round(orig_duration, 2),
        "output_duration_sec": round(out_duration, 2),
        "segments": segments
    }


if __name__ == "__main__":
    in_video = sys.argv[1] if len(sys.argv) > 1 else "input/test_video.mp4"
    out_video = sys.argv[2] if len(sys.argv) > 2 else "output/adaptive_compressed.mp4"

    compress_adaptive(
        input_video=in_video,
        output_video=out_video,
        static_crf=36,
        active_crf=28,
        preset="medium",
        threshold=2.0
    )
