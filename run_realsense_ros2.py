"""
ROS2 node: capture RealSense RGB, run wall segmentation, publish /rgb and /rgb_wall_mask.

3D display in RViz2: The built-in Camera display shows the image in a 2D panel. To show
the feed as a textured quad in the 3D view, use the rviz_textured_quads plugin:
  https://github.com/lucasw/rviz_textured_quads
Subscribe its "Image" topic to /rgb or /rgb_wall_mask.
"""
import argparse
import threading
import time
import numpy as np
import cv2
from PIL import Image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from tf2_ros import StaticTransformBroadcaster

from models.models import SegmentationModule, build_encoder, build_decoder
from src.eval import segment_image
from utils.constants import DEVICE
from utils.utils import get_wall_mask_overlay


RS_WIDTH = 640
RS_HEIGHT = 480
RS_FPS = 30
PROCESS_EVERY_N = 1
FRAME_ID = "camera_optical_frame"
# Default parent frame for TF (must exist in your TF tree; override with --tf-parent)
TF_PARENT_FRAME_DEFAULT = "map"


class WallSegmentationNode(Node):
    def __init__(self, args):
        super().__init__("wall_segmentation_node")
        self.args = args
        self.bridge = CvBridge()
        self.pub_rgb = self.create_publisher(ImageMsg, "/rgb", 10)
        self.pub_rgb_wall_mask = self.create_publisher(ImageMsg, "/rgb_wall_mask", 10)
        self.pub_camera_info = self.create_publisher(CameraInfo, "/camera_info", 10)

        try:
            import pyrealsense2 as rs
        except ImportError:
            self.get_logger().error("pyrealsense2 not installed")
            raise

        # Publish static TF so RViz2 Camera display can place the image (frame_id must be in TF tree)
        self._tf_static = StaticTransformBroadcaster(self)
        parent_frame = getattr(args, "tf_parent_frame", TF_PARENT_FRAME_DEFAULT)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = FRAME_ID
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self._tf_static.sendTransform(t)
        self.get_logger().info("Published static TF %s -> %s" % (parent_frame, FRAME_ID))

        self.get_logger().info("Loading segmentation model...")
        net_encoder = build_encoder(args.encoder)
        net_decoder = build_decoder(args.decoder)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder)
        self.segmentation_module = self.segmentation_module.to(DEVICE).eval()

        config = rs.config()
        config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
        self.pipeline = rs.pipeline()
        self.get_logger().info("Starting RealSense pipeline (color stream)...")
        self.pipeline.start(config)

        self.last_pred = None
        self.frame_count = 0
        self._stop = False
        self._fps_last_time = time.monotonic()
        self._fps_last_count = 0
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def _camera_loop(self):
        """Tight loop like run_realsense_live.py: no timer/executor overhead."""
        while not self._stop:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
            except Exception:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            self.frame_count += 1
            if self.last_pred is None or (self.frame_count % self.args.process_every == 0):
                img_pil = Image.fromarray(frame_rgb)
                self.last_pred = segment_image(
                    self.segmentation_module,
                    img_pil,
                    disp_image=False,
                    max_size=self.args.max_size,
                )

            mask_view = get_wall_mask_overlay(frame_rgb, self.last_pred, walls_on_black=False)

            stamp = self.get_clock().now().to_msg()

            # RViz2 Camera display requires CameraInfo on /camera_info (same stamp/frame_id as image)
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = FRAME_ID
            info.height = RS_HEIGHT
            info.width = RS_WIDTH
            info.distortion_model = "plumb_bob"
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            info.k = [500.0, 0.0, RS_WIDTH / 2.0, 0.0, 500.0, RS_HEIGHT / 2.0, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [500.0, 0.0, RS_WIDTH / 2.0, 0.0, 0.0, 500.0, RS_HEIGHT / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.pub_camera_info.publish(info)

            msg_rgb = self.bridge.cv2_to_imgmsg(
                np.ascontiguousarray(frame_rgb), encoding="rgb8"
            )
            msg_rgb.header.stamp = stamp
            msg_rgb.header.frame_id = FRAME_ID
            self.pub_rgb.publish(msg_rgb)

            msg_mask = self.bridge.cv2_to_imgmsg(
                np.ascontiguousarray(mask_view), encoding="rgb8"
            )
            msg_mask.header.stamp = stamp
            msg_mask.header.frame_id = FRAME_ID
            self.pub_rgb_wall_mask.publish(msg_mask)

            # Print framerate every 5 seconds
            now = time.monotonic()
            if now - self._fps_last_time >= 5.0:
                elapsed = now - self._fps_last_time
                frames_in_interval = self.frame_count - self._fps_last_count
                fps = frames_in_interval / elapsed if elapsed > 0 else 0.0
                self.get_logger().info("Framerate: %.1f Hz (%d frames in %.1f s)" % (fps, frames_in_interval, elapsed))
                self._fps_last_time = now
                self._fps_last_count = self.frame_count

    def shutdown(self):
        self.get_logger().info("Stopping RealSense pipeline...")
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description="Wall segmentation ROS2 publisher")
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
    parser.add_argument(
        "--tf-parent",
        default=TF_PARENT_FRAME_DEFAULT,
        dest="tf_parent_frame",
        help="Parent frame for camera_optical_frame in TF (must exist in tree; default: map)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = WallSegmentationNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
