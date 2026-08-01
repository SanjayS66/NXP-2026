# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# ── TUNABLE PARAMETERS ──────────────────────────────────────────────
# Percentage of image from the bottom to analyze (higher = more lookahead)
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.45
VECTOR_MAGNITUDE_MINIMUM = 2.25

# HSV thresholds for black lane boundary detection
# Black in HSV: low Value, any Hue, low-to-moderate Saturation
HSV_H_MIN, HSV_H_MAX = 0, 179
HSV_S_MIN, HSV_S_MAX = 0, 255
HSV_V_MIN, HSV_V_MAX = 0, 30       # ← PRIMARY TUNING KNOB for black sensitivity

# Minimum angle (radians) for valid lane vectors — rejects near-horizontal noise
VECTOR_MIN_ANGLE_DEG = 2.0
VECTOR_MIN_ANGLE = VECTOR_MIN_ANGLE_DEG * (PI / 180.0)

# EMA smoothing factor for vector coordinates (lower = smoother, higher = more responsive)
VECTOR_SMOOTHING_ALPHA = 0.5

# For curve fitting: how many pixels from the bottom of a contour to use for fitLine
FIT_LINE_SPAN_PX = 60
MIN_CONTOUR_AREA = 5

class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    It publishes the detected boundaries as synapse_msgs/EdgeVectors.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)
            
        self.bridge = CvBridge()

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            Image,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            Image,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # EMA smoothing state: previous smoothed vector endpoints
        self.smoothed_v1 = None  # [[x0,y0], [x1,y1]]
        self.smoothed_v2 = None  # [[x0,y0], [x1,y1]]

        print("Edge Vectors Publisher initialized")
    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        try:
            message = self.bridge.cv2_to_imgmsg(image, encoding="passthrough")
            publisher.publish(message)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def process_image_for_edge_vectors(self, image):
        """
        Analyzes the binary threshold image and extracts robust left and right lane edge vectors.
        Uses cv2.fitLine on the contour segments to generate mathematically straight local tangents.
        """
        if image is None:
            return []

        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        # 1. Convert to HSV for robust black line detection under varying lighting
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. Threshold for black lane boundaries in HSV space
        lower_black = np.array([HSV_H_MIN, HSV_S_MIN, HSV_V_MIN])
        upper_black = np.array([HSV_H_MAX, HSV_S_MAX, HSV_V_MAX])
        thresh = cv2.inRange(hsv, lower_black, upper_black)

        # 3. Crop the image to focus on the lower section close to the buggy
        thresh_cropped = thresh[self.image_height - self.lower_image_height:]
        debug_img = image[self.image_height - self.lower_image_height:].copy()

        # 3b. Apply morphological closing to bridge gaps in pixelated/thin lines without thickening them
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        thresh_cropped = cv2.morphologyEx(thresh_cropped, cv2.MORPH_CLOSE, kernel)

        # 4. Find contours
        contours = cv2.findContours(thresh_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        rover_point = np.array([self.image_width / 2.0, self.lower_image_height])
        half_width = self.image_width / 2.0

        # Group valid points into left and right sides
        left_pts = []
        right_pts = []

        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue

            pts = c[:, 0, :]
            # Use the bottom-most point to determine if the line belongs to the left or right side
            bottom_pt_x = pts[np.argmax(pts[:, 1])][0]
            
            if bottom_pt_x < half_width:
                left_pts.extend(pts)
            else:
                right_pts.extend(pts)

        vectors = []

        # Process each side to extract a single local tangent vector
        for pts_list in [left_pts, right_pts]:
            if len(pts_list) == 0:
                continue

            pts = np.array(pts_list)
            
            # Find the bottom-most point of this grouped line
            max_y = np.max(pts[:, 1])
            
            # Extract local points near the buggy to find the local tangent on curves.
            # This smoothly groups dashed lines that fall within this local region!
            local_mask = pts[:, 1] >= (max_y - FIT_LINE_SPAN_PX)
            local_pts = pts[local_mask]

            if len(local_pts) < 10:
                continue

            # Fit a 2D line to these local grouped points
            [vx, vy, x, y] = cv2.fitLine(local_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(vx[0])
            vy = float(vy[0])
            x = float(x[0])
            y = float(y[0])
            
            # Force the direction vector to point "up" the screen
            if vy > 0:
                vx = -vx
                vy = -vy
                
            # If the line is perfectly horizontal, we handle it safely.
            if abs(vy) > 1e-4:
                t_bottom = (max_y - y) / vy
                x2 = x + t_bottom * vx
                y2 = float(max_y)
            else:
                x2 = x
                y2 = y
            
            # Create the top point by extending the vector by a fixed length
            VECTOR_LENGTH = 100.0
            x1 = x2 + VECTOR_LENGTH * vx
            y1 = y2 + VECTOR_LENGTH * vy
            
            # Reject near-horizontal vectors
            angle = self.get_vector_angle_in_radians([[x1, y1], [x2, y2]])
            if abs(angle) < VECTOR_MIN_ANGLE:
                continue
                
            middle_point = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            distance = np.linalg.norm(middle_point - rover_point)

            vectors.append([[x1, y1], [x2, y2], distance])

        # 5. Sort vectors based on distance from buggy
        vectors = sorted(vectors, key=lambda x: x[2])

        # 6. Split vectors based on left/right halves of the image
        half_width = self.image_width / 2
        vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) < half_width]
        vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) >= half_width]

        final_vectors = []
        # Select the closest vector from each side (left/right)
        for side_vectors in [vectors_left, vectors_right]:
            if len(side_vectors) > 0:
                best_vector = side_vectors[0]
                
                # Draw the selected key lane vector in green on the debug image
                cv2.line(debug_img, (int(best_vector[0][0]), int(best_vector[0][1])), 
                                    (int(best_vector[1][0]), int(best_vector[1][1])), GREEN_COLOR, 2)
                
                # Transform coordinates back to the original uncropped image space
                best_vector[0][1] += self.upper_image_height
                best_vector[1][1] += self.upper_image_height
                final_vectors.append(best_vector[:2])

        # Publish visual debugging images
        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        return final_vectors

    def smooth_vectors(self, vectors):
        """Apply EMA smoothing to reduce frame-to-frame jitter in vector positions."""
        alpha = VECTOR_SMOOTHING_ALPHA

        if len(vectors) > 0:
            new_v1 = vectors[0]
            if self.smoothed_v1 is not None:
                self.smoothed_v1 = [
                    [alpha * new_v1[0][0] + (1 - alpha) * self.smoothed_v1[0][0],
                     alpha * new_v1[0][1] + (1 - alpha) * self.smoothed_v1[0][1]],
                    [alpha * new_v1[1][0] + (1 - alpha) * self.smoothed_v1[1][0],
                     alpha * new_v1[1][1] + (1 - alpha) * self.smoothed_v1[1][1]]
                ]
            else:
                self.smoothed_v1 = [list(new_v1[0]), list(new_v1[1])]
            vectors[0] = [list(self.smoothed_v1[0]), list(self.smoothed_v1[1])]
        else:
            self.smoothed_v1 = None

        if len(vectors) > 1:
            new_v2 = vectors[1]
            if self.smoothed_v2 is not None:
                self.smoothed_v2 = [
                    [alpha * new_v2[0][0] + (1 - alpha) * self.smoothed_v2[0][0],
                     alpha * new_v2[0][1] + (1 - alpha) * self.smoothed_v2[0][1]],
                    [alpha * new_v2[1][0] + (1 - alpha) * self.smoothed_v2[1][0],
                     alpha * new_v2[1][1] + (1 - alpha) * self.smoothed_v2[1][1]]
                ]
            else:
                self.smoothed_v2 = [list(new_v2[0]), list(new_v2[1])]
            vectors[1] = [list(self.smoothed_v2[0]), list(self.smoothed_v2[1])]
        else:
            self.smoothed_v2 = None

        return vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        vectors = self.process_image_for_edge_vectors(image)

        # Apply EMA smoothing to reduce frame-to-frame jitter
        vectors = self.smooth_vectors(vectors)

        # Construct and publish the ROS 2 EdgeVectors message
        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Vector 1 (usually representing Left boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Vector 2 (usually representing Right boundary)
        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)

def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
