# Copyright 2026 The xTask Authors.
"""Process-local VBench decoder compatibility, leaving legacy I2V untouched."""
from __future__ import annotations


def prepare_decoder(path):
    import vbench.utils as utils
    try:
        import decord
        reader = decord.VideoReader(str(path), num_threads=1)
        reader.get_batch([0])
        return "decord"
    except Exception:
        # Some bundled decord wheels cannot configure FFmpeg 9 pixel_formats.
        # Use the same RGB frames and official sampling/transform functions.
        utils.load_video = load_video_cv2
        return "opencv"


def load_video_cv2(video_path, data_transform=None, num_frames=None,
                    return_tensor=True, width=None, height=None):
    import cv2
    import numpy as np
    import torch
    from vbench.utils import get_frame_indices
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {video_path}")
    frames = []
    try:
        while True:
            success, image = capture.read()
            if not success:
                break
            if width:
                if not height:
                    raise ValueError("Decoder resizing needs height with width")
                image = cv2.resize(image, (width, height))
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decoded frames: {video_path}")
    buffer = np.stack(frames).astype(np.uint8)
    if num_frames:
        indices = get_frame_indices(num_frames, len(buffer), sample="middle")
        buffer = buffer[indices]
    if data_transform:
        return data_transform(buffer)
    if return_tensor:
        return torch.Tensor(buffer).permute(0, 3, 1, 2)
    return buffer
