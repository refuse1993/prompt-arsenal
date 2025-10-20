"""
Video Adversarial Attack Generator
"""

import cv2
import numpy as np
from typing import Optional


class VideoAdversarial:
    """Video adversarial example generator"""

    def temporal_attack(self, video_path: str, output_path: str, frame_skip: int = 5):
        """
        Temporal Attack
        Add perturbations to specific frames

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_skip: Apply noise every N frames
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add noise to every Nth frame
            if frame_count % frame_skip == 0:
                noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    def subliminal_frame_injection(self, video_path: str, output_path: str,
                                   inject_image_path: str, inject_at: int = 30):
        """
        Subliminal Frame Injection
        Insert single-frame images

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            inject_image_path: Path to image to inject
            inject_at: Frame number to inject at
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        inject_img = cv2.imread(inject_image_path)
        inject_img = cv2.resize(inject_img, (width, height))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count == inject_at:
                out.write(inject_img)  # Subliminal frame
            else:
                out.write(frame)

            frame_count += 1

        cap.release()
        out.release()

    def frame_drop_attack(self, video_path: str, output_path: str, drop_ratio: float = 0.1):
        """
        Frame Drop Attack
        Randomly drop frames

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            drop_ratio: Ratio of frames to drop (0.0-0.5)
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Randomly drop frames
            if np.random.rand() > drop_ratio:
                out.write(frame)

        cap.release()
        out.release()

    def color_shift_video(self, video_path: str, output_path: str, shift_amount: int = 5):
        """
        Color Shift Video
        Shift colors across entire video

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            shift_amount: Amount to shift colors
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Shift colors
            frame = frame.astype(int)
            frame[:,:,0] = np.clip(frame[:,:,0] + shift_amount, 0, 255)  # B
            frame[:,:,1] = np.clip(frame[:,:,1] - shift_amount//2, 0, 255)  # G
            frame[:,:,2] = np.clip(frame[:,:,2] + shift_amount//3, 0, 255)  # R
            frame = frame.astype(np.uint8)

            out.write(frame)

        cap.release()
        out.release()

    def brightness_flicker(self, video_path: str, output_path: str, flicker_freq: int = 10):
        """
        Brightness Flicker
        Add subtle brightness changes

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            flicker_freq: Flicker every N frames
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add flicker
            if frame_count % flicker_freq == 0:
                factor = 1.2 if (frame_count // flicker_freq) % 2 == 0 else 0.8
                frame = np.clip(frame.astype(float) * factor, 0, 255).astype(np.uint8)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    def get_attack_types(self) -> list:
        """Get list of available attack types"""
        return [
            'temporal',
            'subliminal',
            'frame_drop',
            'color_shift',
            'brightness_flicker'
        ]
