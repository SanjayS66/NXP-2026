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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign boards.
    It publishes the detected sign type/labels on the `/sign_board_detection` topic.
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

        self.model = None
        self.model_type = None
        self.frame_counter = 0

        # Attempt to load the pre-trained PyTorch/YOLO model (best.pt)
        if HAS_TORCH:
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(dir_path, 'best.pt')
                
                if os.path.exists(model_path):
                    if HAS_ULTRALYTICS:
                        self.get_logger().info(f"Loading YOLOv8 model from {model_path}")
                        self.model = YOLO(model_path)
                        self.model_type = 'yolov8'
                    else:
                        self.get_logger().info(f"Loading YOLOv5 model via torch.hub from {model_path}")
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, _verbose=False)
                        self.model_type = 'yolov5'
                else:
                    self.get_logger().warn(f"Model file not found at {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load PyTorch model: {e}")
        else:
            self.get_logger().warn("PyTorch is not installed. Running in CV/Placeholder mode.")

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""
        self.frame_counter += 1
        # Rate limit: Process every 5th frame (~6 Hz)
        if self.frame_counter % 5 != 0:
            return

        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        sign_detected = self.classify_sign(image)

        if sign_detected:
            msg = String()
            msg.data = sign_detected
            self.publisher_sign.publish(msg)
            
            # Only log if it's a new detection to reduce spam
            if getattr(self, 'last_published_sign', None) != sign_detected:
                self.get_logger().info(f"Detected Sign Board: {sign_detected}")
                self.last_published_sign = sign_detected

    def classify_sign(self, image):
        """
        Classify traffic sign boards using YOLO model.
        Returns a comma-separated string of detections e.g., "A:LEFT,X:RIGHT"
        """
        if self.model is None:
            return None

        letters = []
        arrows = []
        
        try:
            if self.model_type == 'yolov8':
                results = self.model(image, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf > 0.5:
                            cls_id = int(box.cls[0])
                            class_name = self.model.names[cls_id]
                            
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            
                            if class_name in ['A', 'B', 'C', 'X', 'Y', 'Z']:
                                letters.append({'name': class_name, 'cx': cx, 'cy': cy})
                            elif class_name in ['Left', 'Right', 'Straight']:
                                arrows.append({'name': class_name, 'cx': cx, 'cy': cy})
                                
            elif self.model_type == 'yolov5':
                results = self.model(image)
                preds = results.xyxy[0].cpu().numpy()
                for pred in preds:
                    xmin, ymin, xmax, ymax, conf, cls_id = pred
                    if conf > 0.7:
                        class_name = self.model.names[int(cls_id)]
                        
                        cx = (xmin + xmax) / 2.0
                        cy = (ymin + ymax) / 2.0
                        
                        if class_name in ['A', 'B', 'C', 'X', 'Y', 'Z']:
                            letters.append({'name': class_name, 'cx': cx, 'cy': cy})
                        elif class_name in ['Left', 'Right', 'Straight']:
                            arrows.append({'name': class_name, 'cx': cx, 'cy': cy})
                            
            if len(letters) > 0 and len(arrows) > 0:
                formatted_signs = []
                for letter in letters:
                    best_arrow = None
                    min_dist = float('inf')
                    
                    for arrow in arrows:
                        # Find the arrow with the closest horizontal center
                        dist = abs(letter['cx'] - arrow['cx'])
                        if dist < min_dist:
                            min_dist = dist
                            best_arrow = arrow
                            
                    if best_arrow is not None:
                        # Convert arrow class to uppercase (e.g. 'Left' -> 'LEFT')
                        direction = best_arrow['name'].upper()
                        formatted_signs.append(f"{letter['name']}:{direction}")
                        
                # Remove duplicates in a single frame
                unique_signs = list(set(formatted_signs))
                # Sort to ensure consistent order (prevents spam from order changes)
                unique_signs.sort()
                
                # Only consider the detection valid if all 6 signs are visible
                if len(unique_signs) == 6:
                    return ",".join(unique_signs)
                
        except Exception as e:
            self.get_logger().debug(f"Inference failed: {e}")

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
