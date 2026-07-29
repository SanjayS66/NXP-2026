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
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# Percentage of image (from the bottom) to process. Kept at 0.45 so we
# also reach the farther-away, narrower-looking sections of track
# where both edges actually fall inside the frame (see prior fix).
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.45

# ------------------------------------------------------------------
# Black-line detection (direct HSV thresholding).
#
# NOTE: an earlier version of this file isolated the white ROAD via a
# flood-fill/connected-components pass, and derived the edges as the
# boundary of that isolated blob. That depends on the black boundary
# line being an unbroken "wall" for the flood-fill to respect -- and
# at a distance the line gets thin enough that morphological opening
# (or just image noise) breaks it, letting the fill leak past it. That
# leak is exactly the inconsistent-mask behavior seen frame-to-frame.
#
# This version goes back to directly thresholding for the black LINE
# itself, per frame, per band -- no dependency on wall continuity
# across the whole crop. It uses HSV instead of plain brightness so it
# doesn't fall into the original problem either: requiring BOTH low
# Value (dark) AND low Saturation (desaturated/true black, not a
# colorful sign or obstacle that merely looks locally darker) keeps it
# from firing on colored objects the way a brightness-only adaptive
# threshold would.
# ------------------------------------------------------------------
BLACK_LINE_VALUE_MAX = 90         # V (brightness, after CLAHE) below this = candidate line pixel
BLACK_LINE_SATURATION_MAX = 80    # also require low saturation -> rejects colorful obstacles/signs

# Minimum number of black-line pixels in a single column (within one
# band) before that column counts as "this is the boundary".
EDGE_PIXEL_COUNT_THRESHOLD = 3

# Sliding window parameters
SLIDING_WINDOW_N_BANDS = 10        # number of horizontal bands to scan from bottom to top
SLIDING_WINDOW_TRACKING_MARGIN_BASE = 30   # base search margin in pixels (bottom bands)
SLIDING_WINDOW_TRACKING_MARGIN_TOP = 60    # search margin at top bands
SLIDING_WINDOW_MIN_DETECTIONS = 2  # minimum edge detections needed to form a vector
LANE_TRACKING_SMOOTHING = 0.3     # EMA alpha for tracking position smoothing


class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    It publishes the detected boundaries as synapse_msgs/EdgeVectors.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # CLAHE instance reused every frame for local brightness
        # normalization (handles shadows before the HSV threshold).
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Temporal tracking: remember previous lane positions for continuity
        self.prev_left_x = None     # previous left lane x-position (bottom band)
        self.prev_right_x = None    # previous right lane x-position (bottom band)
        self.prev_vectors = []      # previous frame's vectors for fallback

    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def weighted_line_fit(self, points, crop_h):
        """
        Fits a straight line through points with bottom-heavy weighting.
        Bottom points (close to buggy) get higher weight.

        Returns (top_point, bottom_point) or None if fit fails.
        """
        pts = np.array(points, dtype=np.float64)
        ys = pts[:, 1]
        xs = pts[:, 0]

        if len(ys) < 2:
            return None

        # Weight: bottom points (high y) get more influence
        raw_weights = (ys / crop_h) ** 2
        weights = raw_weights / np.sum(raw_weights)

        # Fit straight line: x = m*y + b
        coeffs = np.polyfit(ys, xs, 1, w=weights)
        m, b = coeffs

        y_top = float(np.min(ys))
        y_bottom = float(np.max(ys))

        top_x = m * y_top + b
        bottom_x = m * y_bottom + b

        return ([top_x, y_top], [bottom_x, y_bottom])

    def median_filter_peaks(self, points):
        """
        Filters outlier peaks by checking x-position against the median.
        Points whose x is more than 40% away from the median x are rejected.
        """
        if len(points) < 3:
            return points

        xs = [p[0] for p in points]
        median_x = np.median(xs)

        # Accept points within 40% of the median x position
        threshold = max(abs(median_x) * 0.4, 25.0)  # at least 25px tolerance
        filtered = [p for p in points if abs(p[0] - median_x) < threshold]
        return filtered

    # ------------------ Black-line segmentation ------------------

    def build_black_line_mask(self, image_cropped):
        """
        Builds a binary mask of pixels that plausibly belong to a black
        boundary line: dark (low Value) AND desaturated (low Saturation).

        Requiring both conditions -- not brightness alone -- is what
        keeps this from firing on colorful obstacles/sign boards that
        happen to be dark-toned but are clearly not black, and (via the
        CLAHE-normalized Value channel) reduces false positives from
        soft shadows on the white road.
        """
        hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Local contrast normalization so shadows don't uniformly
        # depress brightness across a whole region of otherwise-valid
        # white road.
        v_eq = self.clahe.apply(v_ch)

        value_dark = cv2.inRange(v_eq, 0, BLACK_LINE_VALUE_MAX)
        saturation_low = cv2.inRange(s_ch, 0, BLACK_LINE_SATURATION_MAX)
        line_mask = cv2.bitwise_and(value_dark, saturation_low)

        # Remove speckle noise first (open), then close small gaps so
        # a thin/distant line segment reads as one continuous run
        # instead of a handful of isolated pixels the column-count
        # threshold below might miss.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        return line_mask

    def find_lane_edge(self, histogram, mid, search_min, search_max, side):
        """
        Scans OUTWARD from the lane center (rather than inward from the
        image border) to find the nearest column with enough black-line
        pixels.

        Scanning outward means the very first cluster of black pixels
        we hit -- closest to where the buggy actually is -- is treated
        as the boundary. A stray dark speck or unrelated dark object
        sitting further out past the real line is simply never reached,
        because we stop as soon as we find the first genuine line.
        """
        if side == 'left':
            for x in range(mid - 1, search_min - 1, -1):
                if histogram[x] >= EDGE_PIXEL_COUNT_THRESHOLD:
                    return x
        else:
            for x in range(mid, search_max):
                if histogram[x] >= EDGE_PIXEL_COUNT_THRESHOLD:
                    return x
        return None

    def compute_vectors_from_image(self, image, line_mask):
        """
        Sliding-window vector construction over the black-line mask.
        Divides the mask into horizontal bands from bottom to top and,
        in each band, finds the nearest black-line column left and
        right of center by summing pixel counts per column
        (histogram) and scanning outward from the middle.

        Returns vectors in the format: [[top_point, bottom_point], ...]
        """
        crop_h, crop_w = line_mask.shape[:2]
        band_h = crop_h // SLIDING_WINDOW_N_BANDS
        mid = crop_w // 2

        left_points = []
        right_points = []

        for i in range(SLIDING_WINDOW_N_BANDS):
            y_start = crop_h - (i + 1) * band_h
            y_end = crop_h - i * band_h
            band = line_mask[y_start:y_end, :]
            band_center_y = (y_start + y_end) / 2.0

            # Histogram: sum black-line pixels along each column of this band.
            histogram = np.sum(band, axis=0) / 255.0

            # Band-dependent tracking margin: gate each edge against
            # the previous frame's tracked position so a one-off false
            # detection can't yank the fitted line off course.
            margin_ratio = i / max(SLIDING_WINDOW_N_BANDS - 1, 1)
            tracking_margin = int(SLIDING_WINDOW_TRACKING_MARGIN_BASE +
                                  margin_ratio * (SLIDING_WINDOW_TRACKING_MARGIN_TOP - SLIDING_WINDOW_TRACKING_MARGIN_BASE))

            left_edge = self.find_lane_edge(histogram, mid, 0, mid, 'left')
            if left_edge is not None:
                if self.prev_left_x is None or abs(left_edge - self.prev_left_x) <= tracking_margin:
                    left_points.append((left_edge, band_center_y))

            right_edge = self.find_lane_edge(histogram, mid, mid, crop_w, 'right')
            if right_edge is not None:
                if self.prev_right_x is None or abs(right_edge - self.prev_right_x) <= tracking_margin:
                    right_points.append((right_edge, band_center_y))

            # Draw search bands on debug image
            cv2.line(image, (0, y_start), (crop_w, y_start), (80, 80, 80), 1)

        # Draw detected edge points on debug image (red)
        for pt in left_points:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 3, RED_COLOR, -1)
        for pt in right_points:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 3, RED_COLOR, -1)

        # Median filter: reject outlier points
        left_points = self.median_filter_peaks(left_points)
        right_points = self.median_filter_peaks(right_points)

        vectors = []
        left_detected = False
        right_detected = False

        # Fit straight line through each set of points
        for points, side in [(left_points, 'left'), (right_points, 'right')]:
            if len(points) < SLIDING_WINDOW_MIN_DETECTIONS:
                continue

            result = self.weighted_line_fit(points, crop_h)
            if result is None:
                continue

            top_point, bottom_point = result
            vectors.append([top_point, bottom_point])

            # Debug: draw line in blue
            cv2.line(image,
                     (int(top_point[0]), int(top_point[1])),
                     (int(bottom_point[0]), int(bottom_point[1])),
                     BLUE_COLOR, 2)

            # Update tracking with EMA smoothing
            xs = [p[0] for p in points]
            ys_arr = [p[1] for p in points]
            bottom_weights = [(y / crop_h) ** 2 for y in ys_arr]
            raw_x = float(np.average(xs, weights=bottom_weights))

            if side == 'left':
                if self.prev_left_x is not None:
                    self.prev_left_x = LANE_TRACKING_SMOOTHING * raw_x + (1 - LANE_TRACKING_SMOOTHING) * self.prev_left_x
                else:
                    self.prev_left_x = raw_x
                left_detected = True
            else:
                if self.prev_right_x is not None:
                    self.prev_right_x = LANE_TRACKING_SMOOTHING * raw_x + (1 - LANE_TRACKING_SMOOTHING) * self.prev_right_x
                else:
                    self.prev_right_x = raw_x
                right_detected = True

        # Reset tracking if detection lost
        if not left_detected:
            self.prev_left_x = None
        if not right_detected:
            self.prev_right_x = None

        return vectors, image

    def process_image_for_edge_vectors(self, image):
        """
        Applies preprocessing and extracts lane edge vectors from the bottom
        portion of the image.

        Steps:
          1. Crop to bottom portion -- focus on nearest/relevant lane edges.
          2. Build an HSV-based black-line mask (low Value + low
             Saturation) -- targets the boundary line directly, every
             frame, independent of any other frame or of the line
             forming an unbroken "wall".
          3. Per horizontal band, sum the mask into a column histogram
             and scan outward from the lane center to find the nearest
             left/right boundary column.
          4. Gate each band against the previous frame's tracked
             position, median-filter outliers, fit lines, publish.
          5. If fewer vectors were found than last frame, fall back to
             the previous frame's vectors rather than steering on a
             partial/noisy result.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = self.image_height - self.lower_image_height

        # 1. Crop to bottom portion ONLY -- reduces noise from sky/buildings/horizon
        y_start = self.upper_image_height
        image_cropped = image[y_start:].copy()

        # 2. Segment the black boundary lines directly (HSV).
        line_mask = self.build_black_line_mask(image_cropped)

        # 3-4. Compute vectors from the line mask.
        vectors, debug_img = self.compute_vectors_from_image(image_cropped, line_mask)

        # Vectors are already left/right split. Take the first two
        # (left, right) if available.
        final_vectors = []
        for vec in vectors[:2]:
            # Transform y-coordinates back to original (uncropped) image space
            vec[0][1] += self.upper_image_height
            vec[1][1] += self.upper_image_height
            final_vectors.append(vec)

        # Fallback: if we got fewer vectors than last frame, reuse
        # previous vectors rather than steering on a partial result.
        if len(final_vectors) < len(self.prev_vectors) and len(self.prev_vectors) >= 2:
            final_vectors = self.prev_vectors

        # Save current vectors for next frame's fallback
        self.prev_vectors = final_vectors

        # Publish debug images (viewable in Foxglove)
        self.publish_debug_image(self.publisher_thresh_image, line_mask)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        return final_vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        vectors = self.process_image_for_edge_vectors(image)

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