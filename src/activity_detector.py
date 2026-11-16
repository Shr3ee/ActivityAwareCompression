"""
src/activity_detector.py
Human Activity Detection, Pose Estimation, and Critical Event Assessment Module.

Implements the core detection and importance assessment requirements from MINI_PROJECT.pdf:
1. Human Detection & Pose Estimation (YOLOv8-Pose with 17 body keypoints)
2. Activity Classification (Sitting, Standing, Walking, Fall/Distress)
3. Object & Critical Event Detection (YOLOv8-Detect for handheld/suspicious objects)
4. Importance Assessment (EMERGENCY vs NORMAL HUMAN vs IDLE)
5. Visual Frame Annotation matching Page 6 of MINI_PROJECT.pdf
"""

import os
import cv2
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional
from ultralytics import YOLO

# Keypoint indices in YOLOv8-pose (COCO format)
# 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear,
# 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow,
# 9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip,
# 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # Facial
    (5, 6),                                     # Shoulders
    (5, 7), (7, 9),                             # Left arm
    (6, 8), (8, 10),                            # Right arm
    (5, 11), (6, 12),                           # Torso
    (11, 12),                                   # Hips
    (11, 13), (13, 15),                         # Left leg
    (12, 14), (14, 16)                          # Right leg
]

# Critical object classes from COCO (classes that could indicate security risks / critical situations)
# In standard COCO: 43: knife, 76: scissors, 67: cell phone, 24: backpack, 26: handbag, 39: bottle
CRITICAL_OBJECT_CLASSES = {
    "knife": "Critical Hazard",
    "scissors": "Sharp Object",
    "cell phone": "Handheld Device",
    "bottle": "Handheld Item",
    "backpack": "Unattended Item",
    "handbag": "Personal Item"
}


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates angle (in degrees) at p2 given points p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return float(angle)


class ActivityDetector:
    def __init__(self, pose_model_path: str = "yolov8n-pose.pt", det_model_path: str = "yolov8n.pt"):
        print(f"[ActivityDetector] Initializing YOLOv8 Pose model ({pose_model_path})...")
        self.pose_model = YOLO(pose_model_path)
        print(f"[ActivityDetector] Initializing YOLOv8 Object model ({det_model_path})...")
        self.det_model = YOLO(det_model_path)

    def classify_activity(
        self,
        keypoints: np.ndarray,
        bbox: Tuple[int, int, int, int],
        prev_center: Optional[Tuple[float, float]] = None,
        fps: float = 30.0
    ) -> Tuple[str, float]:
        """
        Classifies human activity based on body keypoint geometry and spatial dynamics:
        Returns: (activity_label, confidence)
        Activities: 'SITTING', 'STANDING', 'WALKING', 'FALL / DISTRESS'
        """
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        aspect_ratio = box_h / box_w  # Standing humans typically have aspect ratio > 1.8; sitting < 1.6

        # Keypoints array: shape (17, 2) or (17, 3)
        kp_xy = keypoints[:, :2]
        kp_conf = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(17)

        # Extract landmark points if confidence is acceptable (> 0.3)
        l_sh, r_sh = kp_xy[5], kp_xy[6]
        l_hip, r_hip = kp_xy[11], kp_xy[12]
        l_knee, r_knee = kp_xy[13], kp_xy[14]
        l_ank, r_ank = kp_xy[15], kp_xy[16]

        # Calculate torso verticality
        sh_center = (l_sh + r_sh) / 2.0
        hip_center = (l_hip + r_hip) / 2.0
        torso_dy = hip_center[1] - sh_center[1]
        torso_dx = abs(hip_center[0] - sh_center[0])
        torso_angle = np.degrees(np.arctan2(torso_dy, torso_dx + 1e-6))

        # Check for Fall / Horizontal distress (torso nearly horizontal)
        if torso_angle < 35.0 and box_w > box_h:
            return "FALL / DISTRESS", 0.90

        # Knee angle (hip - knee - ankle)
        knee_angles = []
        if kp_conf[11] > 0.3 and kp_conf[13] > 0.3 and kp_conf[15] > 0.3:
            knee_angles.append(calculate_angle(l_hip, l_knee, l_ank))
        if kp_conf[12] > 0.3 and kp_conf[14] > 0.3 and kp_conf[16] > 0.3:
            knee_angles.append(calculate_angle(r_hip, r_knee, r_ank))

        avg_knee_angle = np.mean(knee_angles) if knee_angles else 150.0

        # Movement tracking
        speed = 0.0
        curr_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if prev_center is not None:
            dist = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
            speed = dist * (fps / 30.0)

        # Upper body only visible (e.g. webcam or seated at desk)
        legs_visible = (kp_conf[13] > 0.3 or kp_conf[14] > 0.3)

        if not legs_visible:
            # If aspect ratio is compact and speed is low -> SITTING
            if aspect_ratio < 1.7:
                return "SITTING", 0.85
            return "STANDING", 0.75

        # Both legs visible:
        if avg_knee_angle < 130.0 or aspect_ratio < 1.55:
            return "SITTING", 0.92
        elif speed > 15.0:
            return "WALKING", 0.88
        else:
            return "STANDING", 0.90

    def detect_critical_objects(self, frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        Detects relevant objects and assesses if any pose an alert or critical situation.
        """
        results = self.det_model(frame, imgsz=640, verbose=False, conf=0.25)
        critical_detections = []

        if not results or len(results) == 0:
            return critical_detections

        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, "").lower()
            conf = float(box.conf[0])

            # Check if object is in critical catalog
            matched_hazard = None
            for key, desc in CRITICAL_OBJECT_CLASSES.items():
                if key in cls_name:
                    matched_hazard = desc
                    break

            if matched_hazard:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                bx1, by1, bx2, by2 = xyxy

                # Verify proximity or association with detected persons
                is_held = False
                for px1, py1, px2, py2 in person_boxes:
                    # Expanded bounding box overlap check
                    if not (bx2 < px1 - 50 or bx1 > px2 + 50 or by2 < py1 - 50 or by1 > py2 + 50):
                        is_held = True
                        break

                critical_detections.append({
                    "class_name": cls_name,
                    "description": matched_hazard,
                    "confidence": round(conf, 2),
                    "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                    "is_held": is_held,
                    "is_emergency": (matched_hazard in ["Critical Hazard", "Sharp Object"] or (is_held and "Handheld" in matched_hazard))
                })

        return critical_detections

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        storage_saved_pct: float = 0.0,
        total_frames: int = 0,
        prev_centers: Optional[Dict[int, Tuple[float, float]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single surveillance frame:
        - Runs pose estimation
        - Classifies activity
        - Detects critical objects
        - Renders Page 6 presentation overlay
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        pose_res = self.pose_model(frame, imgsz=640, verbose=False, conf=0.35)
        persons = []
        person_boxes = []

        current_centers = {}

        if pose_res and len(pose_res[0].boxes) > 0:
            boxes = pose_res[0].boxes.xyxy.cpu().numpy()
            keypoints = pose_res[0].keypoints.data.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                person_boxes.append((x1, y1, x2, y2))
                kpts = keypoints[i]

                prev_c = prev_centers.get(i) if prev_centers else None
                act_label, act_conf = self.classify_activity(kpts, (x1, y1, x2, y2), prev_c, fps)
                current_centers[i] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                persons.append({
                    "id": i,
                    "bbox": [x1, y1, x2, y2],
                    "activity": act_label,
                    "confidence": round(act_conf, 2),
                    "keypoints": kpts.tolist()
                })

        # Object / Critical situation detection
        critical_objs = self.detect_critical_objects(frame, person_boxes)
        has_emergency = any(obj["is_emergency"] for obj in critical_objs) or any(p["activity"] == "FALL / DISTRESS" for p in persons)

        # Assess Importance & Operating Mode
        if has_emergency:
            mode = "EMERGENCY"
            mode_fps = 30
            target_crf = 24
            active_label = "CRITICAL ALERT: HAZARD / ITEM DETECTED"
            if any(p["activity"] == "FALL / DISTRESS" for p in persons):
                active_label = "CRITICAL ALERT: ABNORMAL ACTIVITY DETECTED"
            mode_color = (0, 0, 255)  # Red (BGR)
            badge_icon = "●"
        elif len(persons) > 0:
            mode = "NORMAL HUMAN"
            mode_fps = 10
            target_crf = 32
            active_label = persons[0]["activity"]
            mode_color = (0, 215, 255)  # Amber / Yellow (BGR)
            badge_icon = "●"
        else:
            mode = "IDLE"
            mode_fps = 5
            target_crf = 36
            active_label = "NO HUMAN DETECTED (STATIC SCENE)"
            mode_color = (160, 160, 160)  # Gray
            badge_icon = "○"

        # ----------------- Visual Annotation (Page 6 Style) -----------------
        # 1. Top dark status banner
        banner_h = 48
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (25, 25, 28), -1)
        cv2.line(annotated, (0, banner_h), (w, banner_h), mode_color, 2)

        # Top line text: MODE & ACTIVITY
        top_title = f"MODE: {mode} ({mode_fps} FPS) | ACTIVITY: {active_label}"
        cv2.circle(annotated, (20, 20), 7, mode_color, -1)
        cv2.putText(annotated, top_title, (38, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Subtitle metrics: FPS, Encoded frames, Storage saved
        curr_encoded = frame_idx + 1
        sub_metrics = f"FPS: {fps:.1f} | Encoded: {curr_encoded}/{total_frames if total_frames > 0 else curr_encoded} frames | Storage Saved: {storage_saved_pct:.1f}%"
        cv2.putText(annotated, sub_metrics, (38, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # 2. Draw Human Bounding Boxes and Skeleton Landmarks
        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            # Yellow bounding box around human as shown in Page 6
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 230, 255), 2)
            cv2.putText(annotated, f"PERSON: {p['activity']}", (px1, max(20, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2, cv2.LINE_AA)

            kpts = np.array(p["keypoints"])
            # Draw green skeleton lines
            for pair in SKELETON_PAIRS:
                idx1, idx2 = pair
                if kpts[idx1, 2] > 0.3 and kpts[idx2, 2] > 0.3:
                    pt1 = (int(kpts[idx1, 0]), int(kpts[idx1, 1]))
                    pt2 = (int(kpts[idx2, 0]), int(kpts[idx2, 1]))
                    cv2.line(annotated, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw red keypoint dots
            for k_idx in range(len(kpts)):
                if kpts[k_idx, 2] > 0.3:
                    pt = (int(kpts[k_idx, 0]), int(kpts[k_idx, 1]))
                    cv2.circle(annotated, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)

        # 3. Draw Critical Object Alerts (Red bounding box with alert label)
        for obj in critical_objs:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
            alert_text = f"[!] {obj['description'].upper()}"
            cv2.putText(annotated, alert_text, (ox1, max(20, oy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": round(frame_idx / fps, 3),
            "mode": mode,
            "mode_fps": mode_fps,
            "target_crf": target_crf,
            "activity_label": active_label,
            "person_count": len(persons),
            "persons": persons,
            "critical_objects": critical_objs,
            "has_emergency": has_emergency,
            "centers": current_centers
        }

        return annotated, frame_data


def analyze_video_activities(
    video_path: str,
    sample_step: int = 5,
    output_annotated_preview: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs full human activity detection and importance assessment across video.
    Segments video into continuous operational modes (EMERGENCY, NORMAL HUMAN, IDLE).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    detector = ActivityDetector()

    sampled_records = []
    prev_centers = {}
    frame_idx = 0
    start_time = time.time()

    preview_writer = None
    if output_annotated_preview:
        os.makedirs(os.path.dirname(output_annotated_preview), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(output_annotated_preview, fourcc, max(1, int(fps / sample_step)), (w, h))

    print(f"[ActivityDetector] Analyzing video: {video_path}")
    print(f"[ActivityDetector] Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            ann_frame, frame_data = detector.process_frame(
                frame=frame,
                frame_idx=frame_idx,
                fps=fps,
                storage_saved_pct=0.0,
                total_frames=total_frames,
                prev_centers=prev_centers
            )
            prev_centers = frame_data["centers"]
            sampled_records.append(frame_data)

            if preview_writer:
                preview_writer.write(ann_frame)

        frame_idx += 1

    cap.release()
    if preview_writer:
        preview_writer.release()

    analysis_duration = time.time() - start_time
    print(f"[ActivityDetector] Processed {len(sampled_records)} checkpoints in {analysis_duration:.2f}s.")

    # Group sample checkpoints into continuous operational segments
    num_pts = len(sampled_records)
    if num_pts == 0:
        raise RuntimeError("No frames could be processed.")

    raw_blocks = []
    curr_mode = sampled_records[0]["mode"]
    curr_act = sampled_records[0]["activity_label"]
    curr_start = 0

    for i in range(1, num_pts):
        if sampled_records[i]["mode"] != curr_mode:
            raw_blocks.append({
                "mode": curr_mode,
                "activity": curr_act,
                "start_idx": curr_start,
                "end_idx": i - 1
            })
            curr_mode = sampled_records[i]["mode"]
            curr_act = sampled_records[i]["activity_label"]
            curr_start = i

    raw_blocks.append({
        "mode": curr_mode,
        "activity": curr_act,
        "start_idx": curr_start,
        "end_idx": num_pts - 1
    })

    # Map blocks to exact frame ranges [0, total_frames - 1]
    segments = []
    for b_idx, block in enumerate(raw_blocks):
        if b_idx == 0:
            start_frame = 0
        else:
            prev_sample_frame = sampled_records[block["start_idx"]]["frame_idx"]
            prev_last_sample = sampled_records[raw_blocks[b_idx - 1]["end_idx"]]["frame_idx"]
            start_frame = (prev_last_sample + prev_sample_frame) // 2 + 1

        if b_idx == len(raw_blocks) - 1:
            end_frame = total_frames - 1
        else:
            curr_last_sample = sampled_records[block["end_idx"]]["frame_idx"]
            next_first_sample = sampled_records[raw_blocks[b_idx + 1]["start_idx"]]["frame_idx"]
            end_frame = (curr_last_sample + next_first_sample) // 2

        # Parameters based on Mode
        mode = block["mode"]
        if mode == "EMERGENCY":
            target_fps = 30
            target_crf = 24
            importance = "CRITICAL"
        elif mode == "NORMAL HUMAN":
            target_fps = 10
            target_crf = 32
            importance = "NORMAL"
        else:
            target_fps = 5
            target_crf = 36
            importance = "LOW (IDLE)"

        seg_duration = (end_frame - start_frame + 1) / fps

        segments.append({
            "segment_index": b_idx,
            "mode": mode,
            "importance": importance,
            "activity": block["activity"],
            "target_fps": target_fps,
            "target_crf": target_crf,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": end_frame - start_frame + 1,
            "start_time": round(start_frame / fps, 3),
            "end_time": round((end_frame + 1) / fps, 3),
            "duration": round(seg_duration, 3)
        })

    # Duration aggregations
    emergency_dur = sum(s["duration"] for s in segments if s["mode"] == "EMERGENCY")
    normal_dur = sum(s["duration"] for s in segments if s["mode"] == "NORMAL HUMAN")
    idle_dur = sum(s["duration"] for s in segments if s["mode"] == "IDLE")

    summary = {
        "video_path": video_path,
        "resolution": f"{w}x{h}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "analysis_time_sec": round(analysis_duration, 2),
        "segment_count": len(segments),
        "emergency_duration_sec": round(emergency_dur, 2),
        "normal_duration_sec": round(normal_dur, 2),
        "idle_duration_sec": round(idle_dur, 2),
        "segments": segments,
        "sample_records": sampled_records[:60]  # Store first sample records for timeline inspect
    }

    return summary
