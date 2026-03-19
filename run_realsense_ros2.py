"""
ROS2 node: capture RealSense RGB, run wall segmentation, publish /rgb and /rgb_wall_mask.

3D display in RViz2: The built-in Camera display shows the image in a 2D panel. To show
the feed as a textured quad in the 3D view, use the rviz_textured_quads plugin:
  https://github.com/lucasw/rviz_textured_quads
Subscribe its "Image" topic to /rgb or /rgb_wall_mask.
"""
import argparse
import math
import threading
import time
import queue
import numpy as np
import cv2
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rosgraph_msgs.msg import Clock
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
TF_PARENT_FRAME_DEFAULT = "base_link"
# If no /clock message received for this long, treat clock as unavailable and stop publishing
CLOCK_TIMEOUT_SEC = 1.0

# Downscale percentage for published images (0.1 = 10% of width/height)
DOWNSCALE_PERCENTAGE = 1.0
# Max publish rate (Hz); skip publishing if last publish was sooner
PUBLISH_MAX_HZ = 10.0

# Test variable to set all walls to true
ALL_TRUE = False

# Source detection timeout
SOURCE_DETECTION_TIMEOUT_SEC = 10.0

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
        # Camera is rotated so it looks forward.
        self._tf_static = StaticTransformBroadcaster(self)
        parent_frame = getattr(args, "tf_parent_frame", TF_PARENT_FRAME_DEFAULT)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = FRAME_ID
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 3.0

        # RPY (deg) -> quaternion for TF (ROS order: Rz(yaw)*Ry(pitch)*Rx(roll))
        # Rotation is around parent frame (base_link)
        roll = -90.0
        pitch = 0.0
        yaw = -90.0
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        cr, sr = math.cos(roll_rad / 2), math.sin(roll_rad / 2)
        cp, sp = math.cos(pitch_rad / 2), math.sin(pitch_rad / 2)
        cy, sy = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        self._tf_static.sendTransform(t)
        self.get_logger().info("Published static TF %s -> %s" % (parent_frame, FRAME_ID))

        self.get_logger().info("Loading segmentation model...")
        net_encoder = build_encoder(args.encoder)
        net_decoder = build_decoder(args.decoder)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder)
        self.segmentation_module = self.segmentation_module.to(DEVICE).eval()

        # Detect source (simulator or RealSense camera)
        self.source_type = None  # "simulator" or "realsense"
        self.pipeline = None
        self._color_intrinsics = None
        self._sim_frame_queue = queue.Queue(maxsize=10)
        self._sim_sub = None
        
        # Always initialize RealSense pipeline to get camera_info intrinsics
        # (needed for both sources)
        try:
            config = rs.config()
            config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
            self.pipeline = rs.pipeline()
            self.get_logger().info("Initializing RealSense pipeline for camera_info...")
            self.pipeline.start(config)
            
            # Cache color stream intrinsics for CameraInfo
            profile = self.pipeline.get_active_profile()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._color_intrinsics = color_profile.get_intrinsics()
            self.get_logger().info(
                "RealSense color intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
                % (
                    self._color_intrinsics.fx,
                    self._color_intrinsics.fy,
                    self._color_intrinsics.ppx,
                    self._color_intrinsics.ppy,
                )
            )
        except Exception as e:
            self.get_logger().warn(f"Could not initialize RealSense for camera_info: {e}")
            # Use default intrinsics if RealSense not available
            self._color_intrinsics = type('obj', (object,), {
                'fx': 525.0, 'fy': 525.0, 'ppx': 320.0, 'ppy': 240.0,
                'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
            })()
        
        # Detect which source to use
        self._detect_source()

        # Use /clock for message stamps (e.g. simulation, bag playback); require fresh messages
        self._clock_stamp = None
        self._clock_recv_time = 0.0
        self._clock_lock = threading.Lock()
        self._clock_sub = self.create_subscription(
            Clock,
            "/clock",
            self._clock_callback,
            10,
        )

        self.last_pred = None
        self.frame_count = 0
        self._stop = False
        self._fps_last_time = time.monotonic()
        self._fps_last_count = 0
        self._clock_error_last_log = 0.0
        self._last_publish_time = 0.0
        
        # Start camera loop only after source is detected
        if self.source_type:
            self._thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._thread.start()
        else:
            self.get_logger().error("No source detected, cannot start camera loop")

    def _clock_callback(self, msg):
        with self._clock_lock:
            self._clock_stamp = msg.clock
            self._clock_recv_time = time.monotonic()

    def _sim_rgb_callback(self, msg):
        """Callback for /rgb_sim subscription."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            # Put frame in queue (non-blocking, drop if queue full)
            try:
                self._sim_frame_queue.put_nowait(cv_image)
            except queue.Full:
                pass  # Drop frame if queue is full
        except Exception as e:
            self.get_logger().warn(f"Error processing sim frame: {e}")

    def _detect_source(self):
        """Simultaneously check for simulator feed (/rgb_sim) and RealSense camera.
        Whichever is found first becomes the source."""
        self.get_logger().info("Detecting video source (simulator or RealSense camera)...")
        
        sim_detected = threading.Event()
        realsense_detected = threading.Event()
        source_lock = threading.Lock()
        detected_source = None
        
        # Create subscription for simulator detection
        sim_msg_received = threading.Event()
        
        def sim_callback(msg):
            sim_msg_received.set()
        
        temp_sim_sub = self.create_subscription(ImageMsg, "/rgb_sim", sim_callback, 10)
        
        def check_realsense():
            """Check for RealSense camera."""
            if self.pipeline is None:
                return
            
            start_time = time.monotonic()
            while time.monotonic() - start_time < SOURCE_DETECTION_TIMEOUT_SEC:
                if sim_detected.is_set() or realsense_detected.is_set():
                    break
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=100)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        with source_lock:
                            if detected_source is None:
                                detected_source = "realsense"
                                realsense_detected.set()
                                self.get_logger().info("RealSense camera source detected")
                        break
                except Exception:
                    pass
                time.sleep(0.1)
        
        # Start RealSense check in separate thread
        realsense_thread = threading.Thread(target=check_realsense, daemon=True)
        realsense_thread.start()
        
        # Check for simulator feed in main thread (so executor can process callbacks)
        # Use executor to process callbacks
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        
        start_time = time.monotonic()
        while time.monotonic() - start_time < SOURCE_DETECTION_TIMEOUT_SEC:
            # Process callbacks to receive simulator messages
            executor.spin_once(timeout_sec=0.1)
            
            if sim_msg_received.is_set():
                with source_lock:
                    if detected_source is None:
                        detected_source = "simulator"
                        sim_detected.set()
                        self.get_logger().info("Simulator source detected")
                break
            
            if realsense_detected.is_set():
                break
        
        # Clean up
        executor.remove_node(self)
        self.destroy_subscription(temp_sim_sub)
        
        # Wait for RealSense thread to finish
        realsense_thread.join(timeout=0.5)
        
        with source_lock:
            if detected_source is None:
                self.get_logger().error(
                    f"No video source found after {SOURCE_DETECTION_TIMEOUT_SEC} seconds. "
                    "Retrying..."
                )
                # Retry detection
                time.sleep(1.0)
                self._detect_source()
                return
            else:
                self.source_type = detected_source
        
        # Set up source-specific resources
        if self.source_type == "simulator":
            # Create subscription for simulator feed
            self._sim_sub = self.create_subscription(
                ImageMsg, "/rgb_sim", self._sim_rgb_callback, 10
            )
            self.get_logger().info("Subscribed to /rgb_sim for simulator feed")
        elif self.source_type == "realsense":
            # Pipeline already initialized, ready to use
            self.get_logger().info("Using RealSense camera pipeline")

    def _get_stamp(self):
        """Stamp from /clock only. Returns None if /clock not received or stale (e.g. bag stopped)."""
        with self._clock_lock:
            if self._clock_stamp is None:
                return None
            if time.monotonic() - self._clock_recv_time > CLOCK_TIMEOUT_SEC:
                return None
            return self._clock_stamp

    def _camera_loop(self):
        """Tight loop: process frames from either simulator or RealSense source."""
        while not self._stop:
            frame_rgb = None
            
            if self.source_type == "simulator":
                # Get frame from simulator queue
                try:
                    frame_rgb = self._sim_frame_queue.get(timeout=0.5)
                    # Resize simulator frame to match RealSense dimensions if needed
                    if frame_rgb.shape[1] != RS_WIDTH or frame_rgb.shape[0] != RS_HEIGHT:
                        frame_rgb = cv2.resize(frame_rgb, (RS_WIDTH, RS_HEIGHT), interpolation=cv2.INTER_LINEAR)
                except queue.Empty:
                    continue
            elif self.source_type == "realsense":
                # Get frame from RealSense pipeline
                if self.pipeline is None:
                    time.sleep(0.1)
                    continue
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=500)
                except Exception:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_bgr = np.asanyarray(color_frame.get_data())
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                # Unknown source type, wait and retry
                time.sleep(0.1)
                continue
            
            if frame_rgb is None:
                continue

            self.frame_count += 1
            if self.last_pred is None or (self.frame_count % self.args.process_every == 0):
                img_pil = Image.fromarray(frame_rgb)
                self.last_pred = segment_image(
                    self.segmentation_module,
                    img_pil,
                    disp_image=False,
                    max_size=self.args.max_size,
                )

            pred_for_display = (
                np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)
                if ALL_TRUE
                else self.last_pred
            )
            mask_view = get_wall_mask_overlay(frame_rgb, pred_for_display, walls_on_black=True)

            stamp = self._get_stamp()
            if stamp is None:
                now = time.monotonic()
                if now - self._clock_error_last_log >= 5.0:
                    self.get_logger().error(
                        "/clock not available: cannot timestamp images. "
                        "Publish /clock (e.g. use_sim_time or rosbag) or check topic."
                    )
                    self._clock_error_last_log = now
                continue

            now = time.monotonic()
            if now - self._last_publish_time < 1.0 / PUBLISH_MAX_HZ:
                continue
            self._last_publish_time = now

            # Downscale for publish (reduces bandwidth and CPU)
            scale = DOWNSCALE_PERCENTAGE
            pub_w = max(1, int(RS_WIDTH * scale))
            pub_h = max(1, int(RS_HEIGHT * scale))
            frame_pub = cv2.resize(frame_rgb, (pub_w, pub_h), interpolation=cv2.INTER_LINEAR)
            mask_pub = cv2.resize(mask_view, (pub_w, pub_h), interpolation=cv2.INTER_NEAREST)

            # RViz2 Camera display requires CameraInfo on /camera_info (same stamp/frame_id as image)
            # Scale intrinsics to match downscaled image size
            scale_x = pub_w / RS_WIDTH
            scale_y = pub_h / RS_HEIGHT
            intr = self._color_intrinsics
            fx = intr.fx * scale_x
            fy = intr.fy * scale_y
            cx = intr.ppx * scale_x
            cy = intr.ppy * scale_y
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = FRAME_ID
            info.height = pub_h
            info.width = pub_w
            info.distortion_model = "plumb_bob"
            info.d = list(intr.coeffs) if len(intr.coeffs) >= 5 else [0.0, 0.0, 0.0, 0.0, 0.0]
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.pub_camera_info.publish(info)

            msg_rgb = self.bridge.cv2_to_imgmsg(
                np.ascontiguousarray(frame_pub), encoding="rgb8"
            )
            msg_rgb.header.stamp = stamp
            msg_rgb.header.frame_id = FRAME_ID
            self.pub_rgb.publish(msg_rgb)

            msg_mask = self.bridge.cv2_to_imgmsg(
                np.ascontiguousarray(mask_pub), encoding="rgb8"
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
        self.get_logger().info("Stopping...")
        self._stop = True
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.pipeline is not None:
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
