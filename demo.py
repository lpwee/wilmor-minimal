#!/usr/bin/env python3
"""Minimal WiLoR demo - 3D hand pose estimation."""

import cv2
import numpy as np
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


# Landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8


def draw_hand_landmarks(image: np.ndarray, outputs: dict) -> np.ndarray:
    """Draw wrist, thumb tip, and index tip landmarks on the image."""
    img = image.copy()

    if not outputs or len(outputs) == 0:
        return img

    # Landmarks to draw: (index, name, color)
    landmarks = [
        (WRIST, "Wrist", (0, 255, 0)),        # green
        (THUMB_TIP, "Thumb", (255, 0, 0)),    # blue (BGR)
        (INDEX_TIP, "Index", (0, 0, 255)),    # red (BGR)
    ]

    for hand_idx, hand_data in enumerate(outputs):
        if 'joints_2d' not in hand_data or 'joints_3d' not in hand_data:
            continue

        joints_2d = hand_data['joints_2d']
        joints_3d = hand_data['joints_3d']
        hand_side = hand_data.get('hand_side', f'hand_{hand_idx}')

        # Print 3D coordinates
        print(f"\n{hand_side.upper()} hand:")
        for idx, name, _ in landmarks:
            x, y, z = joints_3d[idx]
            print(f"  {name:6s}: x={x:7.3f}, y={y:7.3f}, z={z:7.3f}")

        # Draw line between thumb and index tip
        thumb_pt = tuple(map(int, joints_2d[THUMB_TIP][:2]))
        index_pt = tuple(map(int, joints_2d[INDEX_TIP][:2]))
        cv2.line(img, thumb_pt, index_pt, (255, 0, 255), 2)  # magenta line

        # Draw landmarks on image
        for idx, name, color in landmarks:
            pt = tuple(map(int, joints_2d[idx][:2]))
            cv2.circle(img, pt, 8, color, -1)
            cv2.circle(img, pt, 8, (0, 0, 0), 2)
            cv2.putText(img, name, (pt[0] + 10, pt[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def run_webcam_demo():
    """Run WiLoR on webcam feed."""
    print("Initializing WiLoR pipeline...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print("Pipeline ready!")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("\nWebcam demo running. Press 'q' to quit.")
    print("Show your hands to the camera!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        outputs = pipe.predict(rgb_frame)

        # Draw results
        result_frame = draw_hand_landmarks(frame, outputs)

        # Add info text
        num_hands = len(outputs) if outputs else 0
        cv2.putText(result_frame, f"Hands detected: {num_hands}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("WiLoR Demo", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image_demo(image_path: str):
    """Run WiLoR on a single image."""
    print("Initializing WiLoR pipeline...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print("Pipeline ready!")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    print("Running inference...")
    outputs = pipe.predict(rgb_image)

    # Print results
    print(f"\nDetected {len(outputs) if outputs else 0} hand(s)")
    if outputs:
        for i, hand in enumerate(outputs):
            print(f"\nHand {i + 1}:")
            for key in hand.keys():
                val = hand[key]
                if isinstance(val, np.ndarray):
                    print(f"  {key}: shape={val.shape}")
                else:
                    print(f"  {key}: {val}")

    # Draw and show results
    result_image = draw_hand_landmarks(image, outputs)
    cv2.imshow("WiLoR Result", result_image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal WiLoR Demo")
    parser.add_argument("--image", "-i", type=str, help="Path to image file (if not provided, uses webcam)")
    args = parser.parse_args()

    if args.image:
        run_image_demo(args.image)
    else:
        run_webcam_demo()
