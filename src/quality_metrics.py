
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

original_path = "input/test_video.mp4"
compressed_path = "output/uniform_compressed.mp4"

original = cv2.VideoCapture(original_path)
compressed = cv2.VideoCapture(compressed_path)

psnr_values = []
ssim_values = []

while True:
    ret1, frame1 = original.read()
    ret2, frame2 = compressed.read()

    if not ret1 or not ret2:
        break
    if len(psnr_values) % 100 == 0:
        print(f"Processed {len(psnr_values)} frames...", flush=True)
    # Ensure both frames have the same dimensions
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(
            frame2, (frame1.shape[1], frame1.shape[0])
        )

    # Convert BGR to RGB
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    psnr = peak_signal_noise_ratio(frame1, frame2, data_range=255)
    ssim = structural_similarity(
        frame1, frame2, channel_axis=2, data_range=255
    )

    psnr_values.append(psnr)
    ssim_values.append(ssim)

original.release()
compressed.release()

print("\n========== QUALITY METRICS ==========")
print(f"Frames compared: {len(psnr_values)}")

if psnr_values:
    print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")
else:
    print("No frames could be compared.")

print("=====================================")
