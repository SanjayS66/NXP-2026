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
import time
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

# ---- Lane centering tuning ----
STEER_GAIN = 0.005          # proportional gain, pixel error -> turn command
STEER_CLAMP = 0.5           # max turn magnitude during normal lane-following
HALF_WIDTH_DEFAULT = 100.0  # px, fallback half-lane-width before we've ever seen 2 vectors
HALF_WIDTH_ALPHA = 0.2      # EMA smoothing for the learned half-width (0=frozen, 1=no smoothing)
HALF_WIDTH_MIN_VALID = 10.0 # ignore degenerate 2-vector reads narrower than this (px)

# ---- Turn-commit tuning ----
# The sign board is recognized well before the physical turn/intersection, so we only
# *queue* the turn on sign detection and *commit* to it once geometry says we're actually
# at the junction mouth (lane opens up / edges vanish), gated by a minimum dwell time so a
# single noisy frame right after the sign can't fire it immediately.
MIN_TIME_BEFORE_TURN = 1.0        # s, ignore intersection cues this soon after seeing the sign
INTERSECTION_WIDTH_RATIO = 1.6    # lane width vs. learned normal width that means "opening up"
TURN_MANEUVER_DURATION = 0.6      # s, how long we drive the committed hard turn
TURN_STRENGTH = 0.8               # turn magnitude while committing
TURN_SPEED = 0.12                 # speed while committing (slower than cruise)
NORMAL_SPEED = 0.15                # cruise speed while lane-following

# CONFIGURATION:
# The buggy is driven in manual mode by publishing standard controller Joy messages to /cerebri/in/joy.
# The layout is: msg.axes = [0.0, speed, 0.0, turn]
# - speed: positive for forward, negative for reverse. Range: [-1.0, 1.0]
# - turn: positive for left steer, negative for right steer. Range: [-1.0, 1.0]
# msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] (Keep buttons set to this pattern for manual override mode)

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    By default, it publishes a safe drive-straight command on a timer loop.
    Implement logic inside the callbacks to steer, dodge obstacles, detect destinations,
    communicate with the server, and park.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        
        # 1. Lane Edge Vectors (from edge_vectors_publisher)
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # 2. LIDAR Obstacle Scanner
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # 3. Server Communication Feedback Loop
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # 4. QR Code Detections (from qr_detector)
        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        # 5. Sign Board Detections (from object_recognizer)
        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        
        # Publisher to drive/steer the buggy
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher to send messages to the Server
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables & Timer ------------------
        
        # Default controls: drive straight slowly
        self.target_speed = NORMAL_SPEED
        self.target_turn = 0.0

        # State variables (You can add your own state flags / state machines here)
        self.obstacle_in_front = False
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = None  # letter (A/B/C/X/Y/Z) the state machine is currently seeking
        self.mission_completed = False

        # --- Lane centering memory ---
        # half_width = learned distance (px) from lane center to a single edge, updated only
        # when both edges are visible. Used to reconstruct the center when only one edge shows.
        self.half_width = None

        # --- Turn-commit state ---
        self.pending_turn = None          # 'left' / 'right' queued from a sign board, or None
        self.pending_turn_target = None   # which letter the queued turn is for
        self.last_sign_time = 0.0
        self.turn_maneuver_active = False
        self.turn_end_timer = None

        # Timer to publish drive commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info("Line Follower controller initialized. Safe Drive-Straight Mode active.")

    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        """
        Receives lane boundaries from the camera vector extractor.
        Computes centerline deviation and adjusts steering.

        Centering strategy:
        - 2 vectors: true center = midpoint of both edges. We also update our running
          estimate of the lane's half-width (center-to-edge distance) from this reading.
        - 1 vector: the other edge is occluded/gone (curve, shadow, or an intersection
          opening up). We reconstruct the center by offsetting the visible edge by the
          last known half-width, instead of just nudging away from it with a fixed turn.
        - 0 vectors: hold the last command briefly rather than snapping to straight.

        This callback also watches for the lane suddenly widening well past the learned
        half-width, which is the geometric signature of arriving at an intersection mouth,
        and uses that (plus a minimum dwell time since the sign was seen) to commit a
        queued turn at the right physical moment instead of at sign-detection time.
        """
        # While we're mid-turn-commit, ignore lane centering entirely — the committed
        # turn command is driving, and the old edge geometry (pre-turn lane) is stale.
        if self.turn_maneuver_active:
            return

        width = message.image_width
        image_center = width / 2.0

        if message.vector_count == 2:
            left_x = message.vector_1[1].x
            right_x = message.vector_2[1].x
            lane_center = (left_x + right_x) / 2.0
            current_half_width = (right_x - left_x) / 2.0

            # Update learned half-width (only on sane, non-degenerate readings)
            if current_half_width > HALF_WIDTH_MIN_VALID:
                if self.half_width is None:
                    self.half_width = current_half_width
                else:
                    self.half_width = ((1 - HALF_WIDTH_ALPHA) * self.half_width
                                        + HALF_WIDTH_ALPHA * current_half_width)

            # Intersection-opening check: lane suddenly much wider than the learned norm
            if self._should_commit_turn(current_half_width):
                self.trigger_turn_maneuver()
                return

            error = lane_center - image_center
            self.target_turn = self._steer_from_error(error)

        elif message.vector_count == 1:
            vec_x = message.vector_1[1].x
            half_w = self.half_width if self.half_width is not None else HALF_WIDTH_DEFAULT

            if vec_x < image_center:
                # Left edge visible -> center is half_w to its right
                estimated_center = vec_x + half_w
            else:
                # Right edge visible -> center is half_w to its left
                estimated_center = vec_x - half_w

            # A single vanished edge with no obvious curve is also consistent with sitting
            # at a junction mouth, so it can commit a queued turn too.
            if self._should_commit_turn(None):
                self.trigger_turn_maneuver()
                return

            error = estimated_center - image_center
            self.target_turn = self._steer_from_error(error)

        else:
            # No lanes visible - most likely deep inside an open intersection.
            if self._should_commit_turn(None):
                self.trigger_turn_maneuver()
                return
            # Otherwise hold the last steering command rather than snapping to straight;
            # this keeps us curving the way we were already headed until lanes reappear.
            pass

    def _steer_from_error(self, error):
        """Proportional steering command from a pixel centerline error."""
        turn = error * STEER_GAIN
        return max(-STEER_CLAMP, min(STEER_CLAMP, turn))

    def _should_commit_turn(self, current_half_width):
        """
        Decide whether it's time to act on a queued sign-board turn.

        Gated by:
        1. There actually is a pending turn.
        2. Enough time has passed since the sign was seen (the sign fires far from the
           actual junction, so an immediate trigger would turn in the wrong spot).
        3. Either the lane has visibly widened past the learned half-width (a real
           2-vector opening reading), or we're being called from the 1-vector/0-vector
           branches where an edge has vanished outright — both are consistent with
           having reached the junction mouth.
        """
        if self.pending_turn is None:
            return False
        if (time.time() - self.last_sign_time) < MIN_TIME_BEFORE_TURN:
            return False
        if current_half_width is not None and self.half_width:
            return current_half_width > self.half_width * INTERSECTION_WIDTH_RATIO
        # Called from the 1-vector / 0-vector branch: an edge already vanished, and the
        # time gate above already protects against reacting to noise right after the sign.
        return current_half_width is None

    def lidar_callback(self, message):
        """
        Receives LIDAR range measurements.
        
        GUIDELINE (Obstacle Avoidance & Building Range):
        - `message.ranges` is an array of distances in meters around the buggy.
        - The laser scans cover 360 degrees. Find which indices correspond to the front of the buggy.
        - If a range value in the front sector is below a threshold (e.g. 0.8m), flag an obstacle.
        - Write obstacle avoidance maneuvers (e.g. stop, steer left/right around the block, and merge back).
        - Use LIDAR side-ranges to verify distance to building/QR signs before patient pickup/hospital drop actions.
        """
        # HINTS:
        # num_readings = len(message.ranges)
        # front_sector = message.ranges[int(num_readings * 7/18): int(num_readings * 11/18)]
        # min_front_dist = min(front_sector)
        pass

    def server_communication_callback(self, message):
        """
        Receives coordination commands from the server.
        
        GUIDELINE (Server Communication):
        - Check if the message is destined for the Buggy (`message.dest == 1`).
		- Do not forget to check for ACK messages from server
        - The server communicates mission info in the `message.msg` payload string.
        - Parse server instructions (e.g., patient pickup, target hospitals).
        - Call `self.send_server_update` to report your status when you reach a checkpoint.
        """
        if message.dest == 1:
            self.get_logger().info(f"Received Server Message: {message.msg}")
            # Parse payload and update state machine destination/objectives here
            pass

    def send_server_update(self, text_msg):
        """Sends status messages to the server. (Do not forget to send ACK messages to server)"""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = 100     # Replace with a rolling message ID/counter
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.
        
        GUIDELINE (Patient/Hospital Identification):
        - Parse the decoded string payload in `message.data` (e.g. "PATIENT_A", "HOSPITAL_B").
        - If it matches your target destination, stop the vehicle close to the building (verify range using LIDAR),
          perform the action (pick patient / drop patient), and communicate the arrival to the server.
        """
        self.get_logger().info(f"Heard QR code: {message.data}")
        pass

    def sign_board_callback(self, message):
        """
        Receives traffic sign boards (e.g. "A_left", "X_straight", "B_right").

        The model sees these well before the actual intersection, so we don't steer here.
        We just queue the turn; edge_vectors_callback commits it once geometry (and the
        minimum dwell timer) says we're actually at the junction.
        """
        label = message.data
        self.get_logger().info(f"Heard Sign Board: {label}")

        try:
            letter, direction = label.split('_')
        except ValueError:
            self.get_logger().warn(f"Unrecognized sign format: {label}")
            return
        direction = direction.lower()

        # Ignore signs for a target we're not currently seeking (avoids queuing a turn
        # meant for a different patient/hospital leg of the mission).
        if self.current_destination is not None and letter != self.current_destination:
            return

        # Don't clobber a turn we're already about to commit / are executing.
        if self.pending_turn is not None or self.turn_maneuver_active:
            return

        if direction == 'straight':
            # Nothing to queue, but note the dwell timer anyway so a stray sign right
            # before a real turn sign doesn't confuse the timing.
            self.last_sign_time = time.time()
            return

        self.pending_turn = direction        # 'left' or 'right'
        self.pending_turn_target = letter
        self.last_sign_time = time.time()

    def trigger_turn_maneuver(self):
        """Commit to the queued turn: drive a fixed hard-turn burst, then resume lane-following."""
        if self.turn_maneuver_active or self.pending_turn is None:
            return

        self.get_logger().info(
            f"Committing turn '{self.pending_turn}' for target {self.pending_turn_target}")

        self.turn_maneuver_active = True
        self.target_turn = TURN_STRENGTH if self.pending_turn == 'left' else -TURN_STRENGTH
        self.target_speed = TURN_SPEED

        # One-shot timer to end the maneuver and hand control back to lane-following.
        self.turn_end_timer = self.create_timer(TURN_MANEUVER_DURATION, self._end_turn_maneuver)

    def _end_turn_maneuver(self):
        """Ends the committed turn burst and resets state for the new lane segment."""
        self.turn_maneuver_active = False
        self.pending_turn = None
        self.pending_turn_target = None
        # The lane beyond the turn may have a different width than the one we just left,
        # so drop the learned half-width and let it relearn from fresh 2-vector reads.
        self.half_width = None
        self.target_speed = NORMAL_SPEED

        if self.turn_end_timer is not None:
            self.turn_end_timer.cancel()
            self.turn_end_timer = None

        self.get_logger().info("Turn maneuver complete, resuming lane following")

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
