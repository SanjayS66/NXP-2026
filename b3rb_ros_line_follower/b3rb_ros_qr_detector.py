# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

import time
import re

try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None

class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload and distance proxy on the `/qr_detection` topic.
    """
    def __init__(self):
        super().__init__('qr_detector')

        # Best Effort QoS Profile to ensure compatibility with standard camera drivers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscription for camera images with QoS profile applied
        self.subscription_camera = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_image_callback,
            qos_profile)
            
        self.bridge = CvBridge()

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        # ── Debounce and Cooldown State ──
        self.last_qr_payload = None
        self.qr_consecutive_count = 0
        self.qr_cooldown_until = 0.0
        self.QR_CONFIRM_THRESHOLD = 3
        self.QR_COOLDOWN_SECS = 5.0
        self.frame_counter = 0

        if pyzbar is None:
            self.get_logger().warn("Pyzbar is NOT installed or failed to import! Falling back to OpenCV QRCodeDetector.")
        else:
            self.get_logger().info("Pyzbar loaded successfully.")

        self.get_logger().info("QR Detector Node started. Waiting for images on topic '/camera/image_raw'...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        self.frame_counter += 1
        # Rate limit: Process every 4th frame (~7.5 Hz)
        if self.frame_counter % 4 != 0:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image via CvBridge: {e}")
            return

        # 2. Image verification log
        h, w = image.shape[:2]
        self.get_logger().debug(f"Image successfully converted. Resolution: {w}x{h}", throttle_duration_sec=5.0)

        result = self.detect_qr_code(image)
        current_time = time.time()

        if result is not None:
            raw_payload, bbox_area = result
            clean_payload = self.clean_qr_payload(raw_payload)
            
            # Log only occasionally to avoid spam
            self.get_logger().debug(f"QR Detected! Raw: '{raw_payload}' | Clean: '{clean_payload}' | BBox Area: {bbox_area}px", throttle_duration_sec=2.0)

            # ── Debounce Logic ──
            if clean_payload == self.last_qr_payload:
                self.qr_consecutive_count += 1
            else:
                self.get_logger().info(f"New candidate payload detected: '{clean_payload}'. Resetting count.")
                self.last_qr_payload = clean_payload
                self.qr_consecutive_count = 1
                
            self.get_logger().debug(f"Debounce progress: {self.qr_consecutive_count}/{self.QR_CONFIRM_THRESHOLD}")

            if self.qr_consecutive_count >= self.QR_CONFIRM_THRESHOLD:
                # Publish continuously so the runner can track when it disappears
                msg = String()
                msg.data = f"{clean_payload}|bbox={bbox_area}"
                self.publisher_qr.publish(msg)

                if current_time > self.qr_cooldown_until:
                    self.get_logger().info(f"PUBLISHING CONFIRMED QR: {msg.data}")
                    # Activate log cooldown
                    self.qr_cooldown_until = current_time + self.QR_COOLDOWN_SECS
                else:
                    remaining = round(self.qr_cooldown_until - current_time, 1)
                    self.get_logger().debug(f"QR in view (logging cooldown: {remaining}s).", throttle_duration_sec=2.0)
        else:
            if self.qr_consecutive_count > 0:
                # self.get_logger().warn(f"Lost QR code visibility. Resetting consecutive count (was {self.qr_consecutive_count}).")
                self.get_logger().debug(f"Lost QR code visibility. Resetting consecutive count (was {self.qr_consecutive_count}).")
            self.qr_consecutive_count = 0

    def clean_qr_payload(self, raw_data):
        """Extracts the identifier from strings like '{LOC: PATIENT_1}'."""
        match = re.search(r'\{LOC:\s*(.*?)\}', raw_data)
        if match:
            return match.group(1).strip()
        return raw_data.strip()

    def detect_qr_code(self, image):
        """
        Detect and decode QR code in the image using Pyzbar / OpenCV.
        Returns a tuple: (raw_string_data, bbox_area_pixels)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Try Pyzbar Detection
        if pyzbar is not None:
            # Try raw grayscale first (often works better than adaptive thresholding)
            decoded_objects = pyzbar.decode(gray)
            
            if not decoded_objects:
                # Fallback to adaptive thresholding if grayscale failed
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                decoded_objects = pyzbar.decode(thresh)
                
            if decoded_objects:
                obj = decoded_objects[0]
                try:
                    data = obj.data.decode('utf-8')
                except UnicodeDecodeError:
                    data = str(obj.data)
                
                rect = obj.rect
                bbox_area = rect.width * rect.height
                return data, bbox_area

        # 2. Fallback to OpenCV QRCodeDetector
        try:
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(gray)
            if bbox is not None and data != "":
                pts = np.array(bbox[0], np.int32)
                area = cv2.contourArea(pts)
                return data, int(area)
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR Detector error: {e}")

        return None

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()