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

    Cooldown Strategy (FIXED):
    - Old approach: a single 5-second global timer blocked ALL QR reads after any publish.
      This caused the hospital QR to be silently ignored when the buggy arrived at a hospital
      within 5 seconds of leaving a patient building.
    - New approach: cooldown is PAYLOAD-AWARE. After publishing a QR code, only that exact
      same payload is suppressed from being re-published. A completely different QR payload
      (e.g. HOSPITAL_2 after PATIENT_1) bypasses the cooldown entirely and goes straight
      through the 3-frame debounce — no time-gate blocking it.
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

        # ── Debounce State ──
        # Tracks the current candidate payload being confirmed via consecutive frames.
        self.last_qr_payload = None
        self.qr_consecutive_count = 0
        self.QR_CONFIRM_THRESHOLD = 3   # Number of hits (with possible gaps) needed to confirm

        # ── Miss-Buffer: tolerate brief decode failures without resetting ──
        # pyzbar intermittently fails to decode even when the QR is in clear view
        # (motion blur, angle, lighting). Resetting on every single missed frame
        # makes the debounce yo-yo and prevents confirmation while moving.
        # We allow up to QR_MISS_TOLERANCE consecutive decode failures before
        # treating visibility as truly lost.
        self.qr_miss_count = 0
        self.QR_MISS_TOLERANCE = 3      # Consecutive misses before resetting count

        # ── Payload-Aware Cooldown State ──
        # After publishing a payload, that exact same payload is suppressed for
        # SAME_PAYLOAD_COOLDOWN_SECS. A DIFFERENT payload bypasses this entirely.
        self.last_published_payload = None   # The most recently published QR payload
        self.same_payload_cooldown_until = 0.0  # Time until same-payload re-publish is allowed
        self.SAME_PAYLOAD_COOLDOWN_SECS = 8.0   # How long to block re-reading the identical QR

        if pyzbar is None:
            self.get_logger().warn("Pyzbar is NOT installed or failed to import! Falling back to OpenCV QRCodeDetector.")
        else:
            self.get_logger().info("Pyzbar loaded successfully.")

        self.get_logger().info("QR Detector Node started. Waiting for images on topic '/camera/image_raw'...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        # 1. Heartbeat check: Confirms callback is actually executing (prints every 2 seconds)
        self.get_logger().info("Callback triggered: Received image frame.", throttle_duration_sec=2.0)

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

            self.get_logger().info(f"QR Detected! Raw: '{raw_payload}' | Clean: '{clean_payload}' | BBox Area: {bbox_area}px")

            # ── Debounce Logic ──
            # On a hit: reset miss counter, increment hit counter.
            # On a hit for a different payload: reset everything and start fresh.
            self.qr_miss_count = 0   # Clear any pending misses — we have a valid frame

            if clean_payload == self.last_qr_payload:
                self.qr_consecutive_count += 1
            else:
                self.get_logger().info(
                    f"New candidate payload detected: '{clean_payload}'. "
                    f"Resetting debounce count.")
                self.last_qr_payload = clean_payload
                self.qr_consecutive_count = 1

            self.get_logger().info(
                f"Debounce progress: {self.qr_consecutive_count}/{self.QR_CONFIRM_THRESHOLD}")

            if self.qr_consecutive_count >= self.QR_CONFIRM_THRESHOLD:
                # ── Payload-Aware Cooldown Gate ──
                # CASE 1: Different from last published → publish immediately.
                # CASE 2: Same as last published → apply cooldown (avoid spamming server).
                is_same_as_last_published = (clean_payload == self.last_published_payload)
                still_on_cooldown = (current_time < self.same_payload_cooldown_until)

                if is_same_as_last_published and still_on_cooldown:
                    remaining = round(self.same_payload_cooldown_until - current_time, 1)
                    self.get_logger().info(
                        f"QR '{clean_payload}' confirmed but same-payload cooldown active "
                        f"for {remaining}s more. Skipping publish."
                    )
                else:
                    msg = String()
                    msg.data = f"{clean_payload}|bbox={bbox_area}"
                    self.publisher_qr.publish(msg)
                    self.get_logger().info(f"PUBLISHING CONFIRMED QR: {msg.data}")
                    self.last_published_payload = clean_payload
                    self.same_payload_cooldown_until = current_time + self.SAME_PAYLOAD_COOLDOWN_SECS
                    self.qr_consecutive_count = 0  # Reset after publish
        else:
            # ── Miss handling: count consecutive misses, only reset after tolerance ──
            self.qr_miss_count += 1
            if self.qr_miss_count >= self.QR_MISS_TOLERANCE:
                if self.qr_consecutive_count > 0:
                    self.get_logger().warn(
                        f"QR visibility lost ({self.qr_miss_count} consecutive misses). "
                        f"Resetting debounce count (was {self.qr_consecutive_count}).")
                self.qr_consecutive_count = 0
                self.qr_miss_count = 0

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