
import cv2
import os
import sys


def inspect_video(video_path):

    # Check whether the file exists
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return

    # Open the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video!")
        return

    # Extract video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if fps > 0 else 0
    file_size = os.path.getsize(video_path) / (1024 * 1024)

    # Display information
    print("\n========== VIDEO INFORMATION ==========")
    print(f"File: {video_path}")
    print(f"Resolution: {width} x {height}")
    print(f"Frame Rate: {fps:.2f} FPS")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"File Size: {file_size:.2f} MB")
    print("=======================================\n")

    video.release()


if __name__ == "__main__":

    path = sys.argv[1] if len(sys.argv) > 1 else "input/test_video.mp4"

    inspect_video(path)
