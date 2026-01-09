#!/usr/bin/env python3
"""Minimal WiLoR demo - 3D hand pose estimation."""

import cv2
import numpy as np
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


def draw_hand_landmarks(image: np.ndarray, outputs: dict) -> np.ndarray:
    """Draw 2D hand landmarks on the image."""
    img = image.copy()

    if not outputs or len(outputs) == 0:
        return img

    # Hand connections for visualization (standard 21-point hand model)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17),  # palm
    ]

    colors = [
        (255, 0, 0),    # red - thumb
        (0, 255, 0),    # green - index
        (0, 0, 255),    # blue - middle
        (255, 255, 0),  # yellow - ring
        (255, 0, 255),  # magenta - pinky
    ]

    for hand_idx, hand_data in enumerate(outputs):
        # Get 2D keypoints if available
        if 'joints_2d' in hand_data:
            joints_2d = hand_data['joints_2d']
        elif 'bbox' in hand_data and 'joints_3d' in hand_data:
            # Project 3D to 2D using bbox center as reference
            continue
        else:
            continue

        # Draw connections
        for i, (start, end) in enumerate(connections):
            if start < len(joints_2d) and end < len(joints_2d):
                pt1 = tuple(map(int, joints_2d[start][:2]))
                pt2 = tuple(map(int, joints_2d[end][:2]))
                color = colors[i // 4 % len(colors)]
                cv2.line(img, pt1, pt2, color, 2)

        # Draw keypoints
        for j, pt in enumerate(joints_2d):
            pt_int = tuple(map(int, pt[:2]))
            cv2.circle(img, pt_int, 4, (0, 255, 255), -1)
            cv2.circle(img, pt_int, 4, (0, 0, 0), 1)

        # Draw bounding box if available
        if 'bbox' in hand_data:
            bbox = hand_data['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            label = f"Hand {hand_idx + 1}"
            if 'hand_side' in hand_data:
                label = hand_data['hand_side'].capitalize()
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img


def run_webcam_demo():
    """Run WiLoR on webcam feed."""
    print("Initializing WiLoR pipeline...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print("Pipeline ready!")

    cap = cv2.VideoCapture(0)
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
