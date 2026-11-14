
import cv2
from ultralytics import YOLO

# Load the lightweight pre-trained pose model
print("Loading YOLOv8 Pose model...")
model = YOLO("yolov8n-pose.pt")

# Open the input video
video = cv2.VideoCapture("input/test_video.mp4")

# Read just one frame
success, frame = video.read()
video.release()

if success:
    print("Running pose detection...")

    # Detect human poses
    results = model(frame, imgsz=640, device="cpu")

    # Draw detected keypoints and skeleton
    output_frame = results[0].plot()

    # Save the result
    cv2.imwrite("output/pose_test.jpg", output_frame)

    print("Pose detection completed!")
    print("Result saved to output/pose_test.jpg")
else:
    print("Could not read the video.")
