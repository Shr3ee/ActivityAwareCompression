
import subprocess
import os

input_video = "input/test_video.mp4"
output_video = "output/uniform_compressed.mp4"

os.makedirs("output", exist_ok=True)

command = [
    "ffmpeg",
    "-hide_banner",
    "-i", input_video,
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "32",
    "-c:a", "copy",
    "-movflags", "+faststart",
    "-y",
    output_video
]

print("Starting uniform compression at CRF 32...")

try:
    subprocess.run(command, check=True)
    print("\nCompression completed successfully!")

except subprocess.CalledProcessError as error:
    print("\nCompression failed:", error)
