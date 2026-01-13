#!/usr/bin/env python3
"""Fast video processing with WiLoR - outputs hand coordinates only (no display)."""

import cv2
import numpy as np
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# Landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8


def calculate_gripper_angle(joints_3d: np.ndarray) -> float:
    """Calculate angle between thumb and index finger vectors from wrist."""
    wrist = joints_3d[WRIST]
    thumb = joints_3d[THUMB_TIP]
    index = joints_3d[INDEX_TIP]

    v_thumb = thumb - wrist
    v_index = index - wrist

    cos_angle = np.dot(v_thumb, v_index) / (np.linalg.norm(v_thumb) * np.linalg.norm(v_index) + 1e-6)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    return angle_deg


def process_video(video_path: str):
    """Process video and print hand coordinates for each frame."""
    # Initialize pipeline
    print("Initializing WiLoR pipeline...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print("Pipeline ready!\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"Frames: {total_frames}, FPS: {fps:.1f}")
    print("-" * 50)

    frame_num = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nVideo finished.")
                break
            frame_num += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs = pipe.predict(rgb_frame)

            if outputs:
                for hand_data in outputs:
                    wilor_preds = hand_data.get('wilor_preds', {})
                    if 'pred_keypoints_3d' not in wilor_preds:
                        continue

                    joints_3d = np.squeeze(wilor_preds['pred_keypoints_3d'], axis=0)
                    is_right = hand_data.get('is_right', 1.0)
                    hand_side = 'R' if is_right > 0.5 else 'L'

                    wrist = joints_3d[WRIST]
                    thumb = joints_3d[THUMB_TIP]
                    index = joints_3d[INDEX_TIP]
                    angle = calculate_gripper_angle(joints_3d)

                    print(f"[{frame_num:4d}] {hand_side} | "
                          f"wrist: ({wrist[0]:6.3f}, {wrist[1]:6.3f}, {wrist[2]:6.3f}) | "
                          f"thumb: ({thumb[0]:6.3f}, {thumb[1]:6.3f}, {thumb[2]:6.3f}) | "
                          f"index: ({index[0]:6.3f}, {index[1]:6.3f}, {index[2]:6.3f}) | "
                          f"angle: {angle:5.1f}°")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        print(f"Processed {frame_num} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video with WiLoR hand detection")
    parser.add_argument("video", nargs="?", default="video/video1.mov",
                        help="Path to video file (default: video/video1.mov)")
    args = parser.parse_args()

    process_video(args.video)
