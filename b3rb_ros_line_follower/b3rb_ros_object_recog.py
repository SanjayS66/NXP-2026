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
            self.get_logger().info(f"Detected Sign Board: {sign_detected}")

    def classify_sign(self, image):
        """
        Classify traffic sign boards using YOLO model.

        The model detects letters (A,B,C,X,Y,Z) and direction arrows
        (Left,Right,Straight) as SEPARATE bounding boxes. This function
        pairs each letter with its spatially nearest direction arrow to
        produce paired "CODE:DIRECTION" tokens (e.g. "A:LEFT,X:RIGHT").

        Returns a comma-separated string of paired detections, or None.
        """
        if self.model is None:
            return None

        # ── Class-set constants ───────────────────────────────────────────
        LETTER_CLASSES    = {'A','B','C','X','Y','Z'}
        DIRECTION_CLASSES = {'LEFT','RIGHT','STRAIGHT'}
        # Max pixel distance between a letter-center and arrow-center to
        # count as the same sign board. Prevents cross-sign pairings.
        MAX_PAIRING_DIST  = 200  # pixels

        letters    = []   # [(letter_str, cx, cy)]
        directions = []   # [(direction_str, cx, cy)]

        try:
            if self.model_type == 'yolov8':
                results = self.model(image, verbose=False)
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < 0.5:
                            continue
                        cls_id     = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        xyxy       = box.xyxy[0].cpu().numpy()
                        cx = float((xyxy[0] + xyxy[2]) / 2)
                        cy = float((xyxy[1] + xyxy[3]) / 2)
                        name_up = class_name.strip().upper()
                        if name_up in LETTER_CLASSES:
                            letters.append((name_up, cx, cy))
                        elif name_up in DIRECTION_CLASSES:
                            directions.append((name_up, cx, cy))
                        elif '_' in class_name or ':' in class_name:
                            # Pre-paired class name from model (e.g. "A_LEFT" or "A:LEFT")
                            letters.append((class_name.replace('_', ':').upper(), cx, cy))

            elif self.model_type == 'yolov5':
                results = self.model(image)
                preds   = results.xyxy[0].cpu().numpy()
                for pred in preds:
                    xmin, ymin, xmax, ymax, conf, cls_id = pred
                    if conf < 0.5:
                        continue
                    class_name = self.model.names[int(cls_id)]
                    cx = float((xmin + xmax) / 2)
                    cy = float((ymin + ymax) / 2)
                    name_up = class_name.strip().upper()
                    if name_up in LETTER_CLASSES:
                        letters.append((name_up, cx, cy))
                    elif name_up in DIRECTION_CLASSES:
                        directions.append((name_up, cx, cy))
                    elif '_' in class_name or ':' in class_name:
                        letters.append((class_name.replace('_', ':').upper(), cx, cy))

        except Exception as e:
            self.get_logger().debug(f"Inference failed: {e}")
            return None

        # ── Spatial pairing: match each letter to its nearest arrow ───────
        # The YOLO model emits separate bboxes for the letter label and the
        # direction arrow on a sign board. We pair each letter with the
        # closest arrow whose centre is within MAX_PAIRING_DIST pixels.
        paired    = []
        used_dirs = set()

        for (letter, lx, ly) in letters:
            if ':' in letter:          # already paired ("A:LEFT" style class)
                paired.append(letter)
                continue

            best_dist = float('inf')
            best_dir  = None
            best_idx  = -1
            for i, (dir_name, dx, dy) in enumerate(directions):
                if i in used_dirs:
                    continue
                dist = ((lx - dx) ** 2 + (ly - dy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_dir  = dir_name
                    best_idx  = i

            if best_dir is not None and best_dist <= MAX_PAIRING_DIST:
                paired.append(f"{letter}:{best_dir}")
                used_dirs.add(best_idx)
                self.get_logger().info(
                    f"[SIGN_PAIR] {letter}:{best_dir} (dist={best_dist:.0f}px)")
            else:
                self.get_logger().debug(
                    f"[SIGN_PAIR] '{letter}' no nearby arrow "
                    f"(best={best_dist:.0f}px > {MAX_PAIRING_DIST}px thr)")

        if not paired:
            return None

        # Deduplicate then publish
        unique = list(dict.fromkeys(paired))
        result = ",".join(unique)
        self.get_logger().info(f"[SIGN_DETECT] Publishing: '{result}'")
        return result


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
