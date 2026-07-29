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
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os
import time

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Confidence threshold for sign detection
SIGN_CONFIDENCE_THRESHOLD = 0.5

# Cooldown between publishing the same sign (seconds)
SIGN_COOLDOWN = 1.0

# Direction mapping from model class names
# The YOLO model detects sign boards. Each sign has a letter (A/B/C/X/Y/Z)
# and a direction (Left/Straight/Right).
# The label format from the model is expected to be like "A_left", "X_straight", etc.
# We map these to the format the runner node expects.

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign boards
    using a YOLOv8 model. It publishes the detected sign type/labels on the
    `/sign_board_detection` topic.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for sign board detection results.
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        # Load YOLOv8 model
        self.model = None
        if YOLO is not None:
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(dir_path, 'best.pt')
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    # Log the class names the model was trained on
                    self.class_names = self.model.names
                    self.get_logger().info(f"Loaded YOLO model from {model_path}")
                    self.get_logger().info(f"Model classes: {self.class_names}")
                else:
                    self.get_logger().warn(f"Model file not found at {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
        else:
            self.get_logger().warn("ultralytics is not installed. Running in placeholder mode.")

        # Cooldown tracking
        self.last_published_sign = None
        self.last_publish_time = 0.0

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        sign_detected = self.classify_sign(image)

        if sign_detected is not None:
            # Apply cooldown — don't publish the same sign repeatedly
            now = time.time()
            if (sign_detected != self.last_published_sign or
                    (now - self.last_publish_time) > SIGN_COOLDOWN):
                msg = String()
                msg.data = sign_detected
                self.publisher_sign.publish(msg)
                self.last_published_sign = sign_detected
                self.last_publish_time = now
                self.get_logger().info(f"Detected Sign Board: {sign_detected}")

    def classify_sign(self, image):
        """
        Detect and classify traffic sign boards using YOLOv8.

        Returns a string like "A_left", "X_straight", "B_right" or None.
        """
        if self.model is None:
            return None

        try:
            results = self.model.predict(
                image,
                conf=SIGN_CONFIDENCE_THRESHOLD,
                verbose=False
            )

            if len(results) == 0 or len(results[0].boxes) == 0:
                return None

            # Get the detection with highest confidence
            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            class_id = int(boxes.cls[best_idx])
            confidence = float(boxes.conf[best_idx])

            # Get class name from model
            sign_label = self.class_names.get(class_id, f"class_{class_id}")

            return sign_label

        except Exception as e:
            self.get_logger().debug(f"YOLO inference failed: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
