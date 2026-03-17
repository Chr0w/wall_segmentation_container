"""
Live wall segmentation from Intel RealSense D457.
Displays RGB on the left and wall mask on the right. Press 'q' or Escape to quit.
"""
import argparse
import time
import numpy as np
import cv2
from PIL import Image

from models.models import SegmentationModule, build_encoder, build_decoder
from src.eval import segment_image
from utils.constants import DEVICE
from utils.utils import get_wall_mask_overlay


# RealSense color stream settings (modest resolution for latency and GPU)
RS_WIDTH = 640
RS_HEIGHT = 480
RS_FPS = 30

# Process every Nth frame to keep display smooth on limited GPU
PROCESS_EVERY_N = 1


def main():
    parser = argparse.ArgumentParser(description="Live wall segmentation with RealSense")
    parser.add_argument(
        "--encoder",
        default="model_weights/transfer_encoder.pth",
        help="Path to encoder weights",
    )
    parser.add_argument(
        "--decoder",
        default="model_weights/transfer_decoder.pth",
        help="Path to decoder weights",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Max side length for inference (lower = less VRAM)",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=PROCESS_EVERY_N,
        help="Run segmentation every N frames (1 = every frame)",
    )
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        raise SystemExit("Install pyrealsense2: pip install pyrealsense2")

    # Load model once
    print("Loading segmentation model...")
    net_encoder = build_encoder(args.encoder)
    net_decoder = build_decoder(args.decoder)
    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    segmentation_module = segmentation_module.to(DEVICE).eval()

    # RealSense pipeline: color only
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
    pipeline = rs.pipeline()

    print("Starting RealSense pipeline (color stream)...")
    pipeline.start(config)

    last_pred = None
    frame_count = 0
    fps_last_time = time.monotonic()
    fps_last_count = 0

    try:
        print("Display: RGB (left) | Wall mask (right). Press 'q' or Escape to quit.")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run segmentation every N frames; reuse last mask otherwise
            frame_count += 1
            if last_pred is None or (frame_count % args.process_every == 0):
                img_pil = Image.fromarray(frame_rgb)
                last_pred = segment_image(
                    segmentation_module,
                    img_pil,
                    disp_image=False,
                    max_size=args.max_size,
                )

            # Right panel: original image with green mask on walls
            mask_view = get_wall_mask_overlay(frame_rgb, last_pred, walls_on_black=False)

            # Side-by-side: left = RGB, right = mask (both RGB)
            combined_rgb = np.concatenate([frame_rgb, mask_view], axis=1)
            combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("RGB | Wall mask", combined_bgr)
            # Print framerate every 5 seconds
            now = time.monotonic()
            if now - fps_last_time >= 5.0:
                elapsed = now - fps_last_time
                frames_in_interval = frame_count - fps_last_count
                fps = frames_in_interval / elapsed if elapsed > 0 else 0.0
                print("Framerate: %.1f Hz (%d frames in %.1f s)" % (fps, frames_in_interval, elapsed))
                fps_last_time = now
                fps_last_count = frame_count
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Escape
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
