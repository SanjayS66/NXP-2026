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
import numpy as np
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 1.5
TURN_MIN = -1.5
TURN_MAX = 1.5

# ── LANE FOLLOWING PD CONTROLLER ────────────────────────────────────────
KP = 0.6                      # Proportional gain (steer response to lateral offset)
KD = 0.15                     # Derivative gain (dampens oscillation)
BASE_SPEED = 0.30            # Forward speed on straight segments
CURVE_SPEED = 0.15            # Reduced speed when steering hard (curves)
CURVE_STEER_THRESHOLD = 0.4   # |steer| above this triggers curve-speed
NO_VECTOR_SLOW_SPEED = 0.20   # Crawl speed when lane vectors are lost
MAX_FRAMES_NO_VECTORS = 15    # Frames of dead-reckoning before slowing down
ASSUMED_LANE_WIDTH_PX = 220   # Approx lane width in pixels (single-vector fallback)

# ── TUNABLE CONSTANTS ─────────────────────────────────────────────
APPROACH_SPEED = 0.12
OBSTACLE_REACT_DIST = 0.8
OBSTACLE_EMERGENCY_DIST = 0.4
NUDGE_STEER_MAGNITUDE = 0.4
WALL_DETECT_RANGE = 2.0
WALL_DETECT_STDDEV = 0.3
QR_CLOSE_THRESHOLD = 3000
SIGN_TO_BUILDING = {
    'A': 'PATIENT_1', 'B': 'PATIENT_2', 'C': 'PATIENT_3',
    'X': 'HOSPITAL_1', 'Y': 'HOSPITAL_2', 'Z': 'HOSPITAL_3'
}

# ── LIDAR TURN CONSTANTS ────────────────────────────────────────────
TURN_TIMEOUT         = 6.0     # Hard safety timeout (seconds) for any turn
TURN_APPROACH_SPEED  = 0.14    # Speed during the turn phase
TURN_KP              = 1.1     # Proportional gain on bearing error
TURN_KD              = 0.18    # Derivative gain on bearing error
TURN_EXIT_COAST_TIME = 0.6     # Time to coast straight before handing back to PD

# ── LIDAR GATE-TRACKING ROBUSTNESS ───────────────────────────────────
# These address the failure modes in the "why this is hard" analysis:
# stray objects masquerading as a pillar, and one pillar of the gate
# being occluded behind the other.
GATE_WIDTH_DEFAULT   = 1.2     # Nominal distance (m) between a gate's two pillars
GATE_WIDTH_MIN       = 0.6     # Reject candidate pillar pairs narrower than this
GATE_WIDTH_MAX       = 2.4     # Reject candidate pairs wider than this (rules out
                                # picking up an unrelated far-away object as the
                                # "other" pillar)
GATE_WIDTH_EMA_ALPHA = 0.3     # Smoothing for the running gate-width estimate
MIN_CLUSTER_POINTS   = 2       # A single stray LiDAR return is more likely noise
                                # (or a thin unrelated object) than a real pillar
GATE_SIDE_MARGIN     = 0.05    # rad; small tolerance when checking a pair's mean
                                # bearing is on the expected side for the turn

# Commit bias: for a short window right as the turn begins, add a small
# steer push in the commanded direction on top of the raw bearing PD term.
# Early in a turn the visible gate is nearly straight ahead, so the raw
# bearing error is small and the buggy tends to track it almost linearly
# into the gate at a steep angle. The bias nudges the trajectory to start
# curving immediately, producing a closer approximation of the arced path
# an Ackermann-steered vehicle actually needs, and making it easier for
# the camera to re-acquire parallel lane lines on exit.
TURN_COMMIT_BIAS      = 0.30
TURN_COMMIT_WINDOW    = 0.5     # seconds from turn_start_time

class MissionState:
    LANE_FOLLOWING = "LANE_FOLLOWING"
    APPROACHING_BUILDING = "APPROACHING_BUILDING"
    STOPPED_AT_PATIENT = "STOPPED_AT_PATIENT"
    WAITING_HOSPITAL_ASSIGNMENT = "WAITING_HOSPITAL_ASSIGNMENT"
    NAVIGATING_TO_HOSPITAL = "NAVIGATING_TO_HOSPITAL"
    STOPPED_AT_HOSPITAL = "STOPPED_AT_HOSPITAL"
    WAITING_NEXT_PATIENT = "WAITING_NEXT_PATIENT"
    MISSION_COMPLETE = "MISSION_COMPLETE"

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        self.subscription_vectors = self.create_subscription(EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.subscription_server = self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_communication_callback, QOS_PROFILE_DEFAULT)
        self.subscription_qr = self.create_subscription(String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)
        self.subscription_signs = self.create_subscription(String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        # ------------------ Core State Variables ------------------
        self.target_speed = BASE_SPEED
        self.target_turn = 0.0

        # Phase 1: Lane following PD state
        self.prev_error = 0.0
        self.last_steer = 0.0
        self.frames_without_vectors = 0
        self.image_width = 320
        
        # Phase 2 & 6: Obstacle Avoidance & Zone state
        self.nudge_steer = 0.0
        self.speed_factor = 1.0
        self.building_nearby = False
        self.building_distance = float('inf')

        # Phase 3: QR state
        self.last_qr_payload = None
        self.last_qr_bbox_area = 0

        # Phase 5: Server state
        self.buggy_uid_counter = 0
        self.last_sent_uid = None
        self.server_ack_received = False
        self.server_retry_time = 0.0
        self.last_sent_payload = None

        # Phase 7: Sign Turning State — LiDAR-Guided Pillar Tracking
        #
        # sign_vote_dir   : current candidate direction being confirmed
        # sign_vote_count : number of positive detections accumulated
        # sign_miss_count : consecutive frames since last positive detection
        # _SIGN_CONFIRM   : hits needed to commit (gaps of <=SIGN_MISS_TOLERANCE allowed)
        # _SIGN_MISS_TOL  : consecutive misses allowed before decaying the vote
        self.sign_vote_dir    = None
        self.sign_vote_count  = 0
        self.sign_miss_count  = 0          # Consecutive missed frames
        SIGN_CONFIRM_FRAMES   = 3          # Reduced from 4 — 3 positive hits required
        SIGN_MISS_TOLERANCE   = 2          # Allow 2 consecutive misses before decay
        self._SIGN_CONFIRM    = SIGN_CONFIRM_FRAMES
        self._SIGN_MISS_TOL   = SIGN_MISS_TOLERANCE

        self.pending_turn = None          # 'LEFT' | 'RIGHT' | 'STRAIGHT' | None (committed)
        self.in_turn = False              # True while pillar-tracking turn is active
        self.turn_start_time = 0.0        # Wall-clock time when turn began
        self.turn_bearing_error = 0.0     # Previous bearing error (for KD term)
        self.turn_exit_start = None       # Wall-clock time when exit coast began

        # Latest LiDAR scan (stored for use in turn logic)
        self.latest_scan = None           # sensor_msgs/LaserScan

        # Running estimate of the physical distance between a gate's two
        # pillars, updated (EMA) whenever both pillars are clearly seen.
        # Used to (a) reject cluster pairs that are implausible gate
        # candidates, and (b) reconstruct an occluded pillar's position
        # when only one is visible.
        self.gate_width_estimate = GATE_WIDTH_DEFAULT

        # Phase 8: Mission State Machine
        self.mission_state = MissionState.LANE_FOLLOWING
        self.patients_delivered = 0
        self.current_target_code = 'A'  # Start by seeking patient 1

        # Timer to publish drive commands and handle state ticks at 10Hz
        self.control_timer = self.create_timer(0.1, self.tick)
        self.get_logger().info("Line Follower controller initialized. Full Mission Loop active.")

    def tick(self):
        """10Hz Timer callback. Acts as the state machine tick."""
        self.update_state_machine()
        self.publish_drive_commands()

    def update_state_machine(self):
        """Handles transitions for Phase 8."""
        current_time = time.time()

        if self.mission_state in [MissionState.LANE_FOLLOWING, MissionState.NAVIGATING_TO_HOSPITAL]:
            # Look out for building QR
            target_building = SIGN_TO_BUILDING.get(self.current_target_code)
            if self.last_qr_payload == target_building:
                self.get_logger().info(f"Target building {target_building} spotted. Approaching...")
                self.mission_state = MissionState.APPROACHING_BUILDING

        elif self.mission_state == MissionState.APPROACHING_BUILDING:
            target_building = SIGN_TO_BUILDING.get(self.current_target_code)
            
            # Phase 6: Zone Fusion (LIDAR + QR)
            in_zone = self.building_nearby and (self.last_qr_bbox_area > QR_CLOSE_THRESHOLD or self.building_distance < 1.0)
            
            # Hard override if QR is lost but we are very close
            if self.building_distance < 0.6: 
                in_zone = True

            if in_zone:
                self.get_logger().info(f"In zone for {target_building}. Stopping.")
                if target_building.startswith("PATIENT"):
                    self.mission_state = MissionState.STOPPED_AT_PATIENT
                else:
                    self.mission_state = MissionState.STOPPED_AT_HOSPITAL
            else:
                self.target_speed = APPROACH_SPEED # Slow approach

        elif self.mission_state == MissionState.STOPPED_AT_PATIENT:
            self.target_speed = 0.0
            self.target_turn = 0.0
            target_building = SIGN_TO_BUILDING.get(self.current_target_code)
            self.send_to_server(target_building)
            self.server_ack_received = False
            self.mission_state = MissionState.WAITING_HOSPITAL_ASSIGNMENT
            self.server_retry_time = current_time + 3.0

        elif self.mission_state == MissionState.WAITING_HOSPITAL_ASSIGNMENT:
            self.target_speed = 0.0
            self.target_turn = 0.0
            if not self.server_ack_received and current_time > self.server_retry_time:
                self.get_logger().warn("Server ACK timeout. Retrying...")
                self.send_to_server(self.last_sent_payload)
                self.server_retry_time = current_time + 3.0
            # Transition handled in server_communication_callback when hospital is assigned

        elif self.mission_state == MissionState.STOPPED_AT_HOSPITAL:
            self.target_speed = 0.0
            self.target_turn = 0.0
            target_building = SIGN_TO_BUILDING.get(self.current_target_code)
            self.send_to_server(target_building)
            self.server_ack_received = False
            self.mission_state = MissionState.WAITING_NEXT_PATIENT
            self.server_retry_time = current_time + 3.0

        elif self.mission_state == MissionState.WAITING_NEXT_PATIENT:
            self.target_speed = 0.0
            self.target_turn = 0.0
            if not self.server_ack_received and current_time > self.server_retry_time:
                self.get_logger().warn("Server ACK timeout. Retrying...")
                self.send_to_server(self.last_sent_payload)
                self.server_retry_time = current_time + 3.0
            # Transition handled in server_communication_callback when next patient assigned

        elif self.mission_state == MissionState.MISSION_COMPLETE:
            self.target_speed = 0.0
            self.target_turn = 0.0

    def publish_drive_commands(self):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] 
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    # ── Phase 5: Server Communication Protocol ──
    def next_uid(self):
        uid = self.buggy_uid_counter
        self.buggy_uid_counter = (self.buggy_uid_counter + 1) % 256
        return uid

    def send_to_server(self, payload, is_ack=False, ack_uid=None):
        msg = ServerCommunication()
        msg.src = 1
        msg.dest = 2
        if is_ack:
            msg.uid = ack_uid
            msg.ack = 1
            msg.msg = ""
        else:
            msg.uid = self.next_uid()
            msg.ack = 0
            msg.msg = payload
            self.last_sent_uid = msg.uid
            self.last_sent_payload = payload
            self.get_logger().info(f"Sending to Server: {payload} (uid={msg.uid})")
        self.publisher_server.publish(msg)

    def server_communication_callback(self, message):
        if message.dest != 1:
            return
            
        if message.ack == 1:
            if message.uid == self.last_sent_uid:
                self.get_logger().info(f"Server ACK received for uid={message.uid}")
                self.server_ack_received = True
        else:
            payload = message.msg.strip()
            self.get_logger().info(f"Received Server Data: {payload} (uid={message.uid})")
            
            # Immediately ACK data
            self.send_to_server("", is_ack=True, ack_uid=message.uid)
            
            if payload == "INVALID":
                self.get_logger().error("Server reported INVALID! Penalty incurred.")
                return

            if self.mission_state == MissionState.WAITING_HOSPITAL_ASSIGNMENT:
                if payload in ['X', 'Y', 'Z']:
                    self.current_target_code = payload
                    self.mission_state = MissionState.NAVIGATING_TO_HOSPITAL
                    self.last_qr_payload = None  # Reset QR memory
                    self._reset_sign_state()     # Flush any stale sign/turn state
                    self.get_logger().info(f"Assigned Hospital: {payload}. Navigating...")

            elif self.mission_state == MissionState.WAITING_NEXT_PATIENT:
                self.patients_delivered += 1
                self.get_logger().info(f"Delivered {self.patients_delivered}/3 patients.")

                if self.patients_delivered >= 3:
                    self.mission_state = MissionState.MISSION_COMPLETE
                    self.get_logger().info("MISSION COMPLETE!")
                elif payload in ['A', 'B', 'C']:
                    self.current_target_code = payload
                    self.mission_state = MissionState.LANE_FOLLOWING
                    self.last_qr_payload = None  # Reset QR memory
                    self._reset_sign_state()     # Flush any stale sign/turn state
                    self.get_logger().info(f"Assigned Next Patient: {payload}. Seeking...")

    # ── Phase 3 & 6: QR Detection ──
    def qr_detection_callback(self, message):
        parts = message.data.split("|bbox=")
        if len(parts) == 2:
            self.last_qr_payload = parts[0]
            self.last_qr_bbox_area = int(parts[1])

    # ── Phase 7: Sign state reset helper ──
    def _reset_sign_state(self):
        """Fully flush sign/turn state. Call whenever current_target_code changes."""
        self.sign_vote_dir    = None
        self.sign_vote_count  = 0
        self.sign_miss_count  = 0
        self.pending_turn     = None
        self.in_turn          = False
        self.turn_exit_start  = None
        self.turn_bearing_error = 0.0
        self.frames_without_vectors = 0
        self.get_logger().info("Sign state reset (new target assigned).")

    # ── Phase 7: Sign Board Detection ──
    def sign_board_callback(self, message):
        # Do not process new signs while a turn is actively executing
        if self.in_turn or self.pending_turn is not None:
            return

        raw_msg = message.data.strip()
        target  = self.current_target_code

        # Throttled summary — one line per second is enough to confirm the topic is live
        self.get_logger().info(
            f"[SIGN] Raw: '{raw_msg}' | seeking='{target}' | "
            f"vote={self.sign_vote_count}/{self._SIGN_CONFIRM} "
            f"candidate='{self.sign_vote_dir}'",
            throttle_duration_sec=1.0)

        # ── Parse: only track the sign matching current target ────────────
        # YOLO publishes paired "CODE:DIRECTION" tokens (after object_recog fix).
        # CODE  = destination letter (A/B/C for patients, X/Y/Z for hospitals).
        # DIRECTION = which way to turn at the intersection to reach that destination.
        detected_dir = None
        signs = raw_msg.split(",")
        for sign in signs:
            sign = sign.strip()
            if ":" not in sign:
                self.get_logger().debug(f"[SIGN] Malformed token: '{sign}'")
                continue
            code, _, direction = sign.partition(":")
            code      = code.strip().upper()
            direction = direction.strip().upper()
            if code == target:
                if direction in ('LEFT', 'RIGHT', 'STRAIGHT'):
                    detected_dir = direction
                    self.get_logger().info(
                        f"[SIGN] MATCH '{code}:{direction}' -> will vote")
                    break
                else:
                    self.get_logger().debug(
                        f"[SIGN] MATCH '{code}' bad direction='{direction}'")
            else:
                self.get_logger().debug(
                    f"[SIGN] skip '{code}:{direction}' (want '{target}')")

        # ── Confidence voting with miss-buffer ────────────────────────────
        if detected_dir is not None:
            self.sign_miss_count = 0
            if detected_dir == self.sign_vote_dir:
                self.sign_vote_count += 1
                self.get_logger().info(
                    f"[SIGN] VOTE {self.sign_vote_count}/{self._SIGN_CONFIRM} "
                    f"for '{target}:{detected_dir}'")
            else:
                old = self.sign_vote_dir
                self.sign_vote_dir   = detected_dir
                self.sign_vote_count = 1
                self.get_logger().info(
                    f"[SIGN] NEW CANDIDATE '{target}:{detected_dir}' "
                    f"(replaced '{old}'). Vote=1/{self._SIGN_CONFIRM}")
        else:
            self.sign_miss_count += 1
            if self.sign_miss_count > self._SIGN_MISS_TOL:
                if self.sign_vote_count > 0:
                    self.sign_vote_count -= 1
                    self.get_logger().info(
                        f"[SIGN] MISS #{self.sign_miss_count} decay "
                        f"'{target}:{self.sign_vote_dir}' "
                        f"-> {self.sign_vote_count}/{self._SIGN_CONFIRM}")
                if self.sign_vote_count == 0 and self.sign_vote_dir is not None:
                    self.get_logger().info(
                        f"[SIGN] Vote EXPIRED for '{target}'. Candidate cleared.")
                    self.sign_vote_dir   = None
                    self.sign_miss_count = 0
            else:
                self.get_logger().debug(
                    f"[SIGN] Miss #{self.sign_miss_count}/{self._SIGN_MISS_TOL} "
                    f"buffered (vote={self.sign_vote_count})")

        # ── DECISION: commit pending turn once vote threshold reached ─────
        if self.sign_vote_count >= self._SIGN_CONFIRM and self.pending_turn is None:
            self.pending_turn = self.sign_vote_dir
            self.get_logger().info(
                f"[SIGN] *** DECISION: '{target}' => TURN {self.pending_turn} *** "
                f"({self.sign_vote_count}/{self._SIGN_CONFIRM} hits, "
                f"tol={self._SIGN_MISS_TOL}). Waiting for intersection.")


    # ── Phase 7: LiDAR Signboard Midpoint Detection Helper ────────────────
    def _find_target_midpoint(self, scan, direction):
        if scan is None or len(scan.ranges) == 0:
            return 0.0, 1.0

        angle_min = scan.angle_min
        angle_inc = scan.angle_increment

        points = []
        for i, r in enumerate(scan.ranges):
            # Focus on 1.5m to 5.0m range to ignore current signboard, and only front hemisphere
            if 1.5 < r < 5.0 and not math.isinf(r) and not math.isnan(r):
                angle = angle_min + i * angle_inc
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                if -math.pi/2 <= angle <= math.pi/2:
                    points.append((angle, r))

        if not points:
            return 0.0, 1.0

        # Sort left to right (positive angle to negative angle)
        points.sort(key=lambda x: x[0], reverse=True)

        clusters = []
        current = [points[0]]

        for i in range(1, len(points)):
            prev_a, prev_r = points[i-1]
            curr_a, curr_r = points[i]
            x1, y1 = prev_r * math.cos(prev_a), prev_r * math.sin(prev_a)
            x2, y2 = curr_r * math.cos(curr_a), curr_r * math.sin(curr_a)
            if math.hypot(x2 - x1, y2 - y1) < 0.8:
                current.append(points[i])
            else:
                clusters.append(current)
                current = [points[i]]
        clusters.append(current)

        # Drop single-return clusters where possible — a lone stray point is
        # more likely sensor noise or a thin unrelated object than a pillar.
        # If filtering would remove everything, keep the unfiltered set
        # rather than going blind.
        filtered = [c for c in clusters if len(c) >= MIN_CLUSTER_POINTS]
        if filtered:
            clusters = filtered

        centroids = []
        for c in clusters:
            c_a = float(np.mean([p[0] for p in c]))
            c_r = float(np.mean([p[1] for p in c]))
            centroids.append((c_a, c_r))

        def to_xy(c):
            a, r = c
            return r * math.cos(a), r * math.sin(a)

        if len(centroids) >= 2:
            # Score every candidate pair instead of blindly grabbing the two
            # extreme clusters — a stray object (tree, other buggy) sitting
            # further out than the true pillar would otherwise get picked.
            # A real gate pair has a known-ish separation and sits on the
            # side that matches the commanded turn.
            best = None
            best_score = None
            n = len(centroids)
            for i in range(n):
                for j in range(i + 1, n):
                    c1, c2 = centroids[i], centroids[j]
                    x1, y1 = to_xy(c1)
                    x2, y2 = to_xy(c2)
                    sep = math.hypot(x2 - x1, y2 - y1)
                    mean_angle = (c1[0] + c2[0]) / 2.0

                    if direction == 'LEFT' and mean_angle < -GATE_SIDE_MARGIN:
                        continue
                    if direction == 'RIGHT' and mean_angle > GATE_SIDE_MARGIN:
                        continue

                    width_ok = GATE_WIDTH_MIN <= sep <= GATE_WIDTH_MAX
                    width_err = abs(sep - self.gate_width_estimate)
                    # Heavily penalize implausible widths rather than
                    # excluding them outright, so we still have a fallback
                    # if nothing satisfies the tolerance band.
                    score = width_err + (0.0 if width_ok else 5.0)

                    # Prefer pairs further toward the commanded side as a
                    # tiebreaker (closer to how the original logic behaved).
                    if direction == 'LEFT':
                        score -= 0.05 * max(c1[0], c2[0])
                    elif direction == 'RIGHT':
                        score -= 0.05 * abs(min(c1[0], c2[0]))

                    if best_score is None or score < best_score:
                        best_score = score
                        best = (c1, c2, sep, width_ok)

            if best is None:
                # Degenerate case (shouldn't happen with n>=2), fall back
                # to the original leftmost/rightmost heuristic.
                if direction == 'LEFT':
                    c1, c2 = centroids[0], centroids[1]
                elif direction == 'RIGHT':
                    c1, c2 = centroids[-1], centroids[-2]
                else:
                    mid_idx = n // 2
                    c1, c2 = centroids[mid_idx - 1], centroids[mid_idx]
            else:
                c1, c2, sep, width_ok = best
                if width_ok:
                    self.gate_width_estimate = (
                        (1 - GATE_WIDTH_EMA_ALPHA) * self.gate_width_estimate
                        + GATE_WIDTH_EMA_ALPHA * sep
                    )

            x1, y1 = to_xy(c1)
            x2, y2 = to_xy(c2)
            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return math.atan2(mid_y, mid_x), math.hypot(mid_x, mid_y)

        elif len(centroids) == 1:
            # Occlusion case: only one pillar visible, the other is hidden
            # behind it or out of range. Reconstruct its approximate
            # position using the running gate-width estimate rather than a
            # fixed angular offset, so the correction scales with distance
            # (closer pillar -> bigger angular offset for the same physical
            # gate width) instead of over- or under-shooting at range.
            c_a, c_r = centroids[0]
            if direction not in ('LEFT', 'RIGHT'):
                # No reliable side to reconstruct toward for STRAIGHT.
                return 0.0, 1.0

            angle_offset = min(0.6, self.gate_width_estimate / max(c_r, 0.3))
            synthetic_angle = c_a + angle_offset if direction == 'LEFT' else c_a - angle_offset

            x1, y1 = to_xy((c_a, c_r))
            x2, y2 = to_xy((synthetic_angle, c_r))
            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return math.atan2(mid_y, mid_x), math.hypot(mid_x, mid_y)

        return 0.0, 1.0

    # ── Phase 7: Execute One Tick of the LiDAR-Guided Turn ────────────────
    def _execute_lidar_turn(self):
        """
        Called from edge_vectors_callback every tick while self.in_turn is True.
        Steers toward the midpoint of the target path boundary clusters.
        Returns True when the turn is complete (hand control back to lane follower).
        """
        now = time.time()
        elapsed = now - self.turn_start_time

        # Safety timeout
        if elapsed > TURN_TIMEOUT:
            self.get_logger().warn(
                f"[TURN] TIMEOUT after {elapsed:.1f}s "
                f"(dir={self.pending_turn}). Forcing exit to lane follow.")
            return True

        # Exit coast phase
        if getattr(self, 'turn_exit_start', None) is not None:
            coast_elapsed = now - self.turn_exit_start
            self.target_speed = TURN_APPROACH_SPEED
            self.target_turn = 0.0
            self.get_logger().info(
                f"[TURN] EXIT COAST {coast_elapsed:.2f}/{TURN_EXIT_COAST_TIME:.2f}s",
                throttle_duration_sec=0.2)
            if coast_elapsed > TURN_EXIT_COAST_TIME:
                self.get_logger().info(f"[TURN] Coast complete. Handing control back.")
                self.turn_exit_start = None
                return True
            return False

        if self.frames_without_vectors == 0 and elapsed > 1.0:
            self.turn_exit_start = now
            self.target_speed = TURN_APPROACH_SPEED
            self.target_turn = 0.0
            self.get_logger().info(
                f"[TURN] Vectors restored after {elapsed:.1f}s. Starting exit coast.")
            return False

        bearing, dist = self._find_target_midpoint(self.latest_scan, self.pending_turn)

        # PD controller
        error      = bearing
        deriv      = error - self.turn_bearing_error
        raw_steer  = (TURN_KP * error) + (TURN_KD * deriv)

        # Commit bias: early in the turn the gate midpoint is nearly
        # straight ahead, so raw_steer alone tracks it almost linearly and
        # the buggy arrives at the gate at a steep angle (see arced-vs-
        # linear trajectory issue). Add a short push in the commanded
        # direction so the path starts curving immediately, then let it
        # decay out so the PD term takes over for fine tracking.
        if elapsed < TURN_COMMIT_WINDOW and self.pending_turn in ('LEFT', 'RIGHT'):
            decay = 1.0 - (elapsed / TURN_COMMIT_WINDOW)
            bias = TURN_COMMIT_BIAS * decay
            raw_steer += bias if self.pending_turn == 'LEFT' else -bias

        steer = max(TURN_MIN, min(TURN_MAX, raw_steer))

        self.turn_bearing_error = error
        self.target_turn        = steer
        self.target_speed       = TURN_APPROACH_SPEED

        self.get_logger().info(
            f"[TURN] Lidar Tracking — dir={self.pending_turn} "
            f"bearing={math.degrees(bearing):+.1f}° dist={dist:.2f}m "
            f"gate_est={self.gate_width_estimate:.2f}m "
            f"steer={steer:+.2f} elapsed={elapsed:.1f}s",
            throttle_duration_sec=0.2)
        return False

    # ── Phase 1 & 7: Steering Controller ──
    def project_x(self, vector, target_y):
        dx = vector[0].x - vector[1].x
        dy = vector[0].y - vector[1].y
        if abs(dy) < 1e-4: return vector[1].x
        return vector[1].x + (target_y - vector[1].y) * (dx / dy)

    def edge_vectors_callback(self, message):
        # Only steer if moving
        if self.mission_state in [MissionState.STOPPED_AT_PATIENT, MissionState.STOPPED_AT_HOSPITAL,
                                  MissionState.WAITING_HOSPITAL_ASSIGNMENT, MissionState.WAITING_NEXT_PATIENT,
                                  MissionState.MISSION_COMPLETE]:
            self.target_turn = 0.0
            return

        self.image_width = message.image_width if message.image_width > 0 else 320
        half_width = self.image_width / 2.0

        # ── Phase 7: Track lane-vector loss (intersection detection) ───────
        prev_frames = self.frames_without_vectors
        if message.vector_count == 0:
            self.frames_without_vectors += 1
            if self.frames_without_vectors == 1:
                self.get_logger().info(
                    f"[INTERSECTION] Lane vectors LOST (vector_count=0). "
                    f"pending_turn={self.pending_turn} "
                    f"target={self.current_target_code} "
                    f"state={self.mission_state}")
            elif self.frames_without_vectors % 5 == 0:
                self.get_logger().info(
                    f"[INTERSECTION] Still no vectors: frame {self.frames_without_vectors} "
                    f"pending_turn={self.pending_turn} in_turn={self.in_turn}",
                    throttle_duration_sec=0.5)
        else:
            if prev_frames > 0:
                self.get_logger().info(
                    f"[INTERSECTION] Lane vectors RESTORED (count={message.vector_count}) "
                    f"after {prev_frames} frames without. in_turn={self.in_turn}")
            self.frames_without_vectors = 0

        # ── Trigger: pending turn + vectors lost for several frames ─────────
        if (
            self.pending_turn is not None
            and not self.in_turn
            and self.frames_without_vectors > 4
        ):
            self.in_turn            = True
            self.turn_start_time    = time.time()
            self.turn_bearing_error = 0.0
            self.turn_exit_start    = None
            self.get_logger().info(
                f"[INTERSECTION] *** EXECUTING TURN: {self.pending_turn} *** "
                f"target='{self.current_target_code}' "
                f"frames_no_vec={self.frames_without_vectors} "
                f"state={self.mission_state}. Scanning LiDAR for pillars...")
        elif (
            self.pending_turn is None
            and not self.in_turn
            and self.frames_without_vectors > 4
        ):
            self.get_logger().info(
                f"[INTERSECTION] *** GOING STRAIGHT (no confirmed sign) *** "
                f"target='{self.current_target_code}' "
                f"vote={self.sign_vote_count}/{self._SIGN_CONFIRM} "
                f"candidate='{self.sign_vote_dir}' "
                f"frames_no_vec={self.frames_without_vectors}",
                throttle_duration_sec=1.0)

        # ── Execute LiDAR-guided turn ──────────────────────────────
        if self.in_turn:
            done = self._execute_lidar_turn()
            if done:
                self._reset_sign_state()
                self.get_logger().info(
                    f"[INTERSECTION] TURN COMPLETE -> "
                    f"target='{self.current_target_code}' "
                    f"state={self.mission_state}. Resuming lane follow.")
            return   # Skip lane-follow PD while turning

        # ── Normal Phase 1 Lane Following ──
        left_vector, right_vector = None, None
        
        if message.vector_count == 2:
            v1_bot_x = message.vector_1[1].x
            v2_bot_x = message.vector_2[1].x
            if v1_bot_x < v2_bot_x:
                left_vector, right_vector = message.vector_1, message.vector_2
            else:
                left_vector, right_vector = message.vector_2, message.vector_1
        elif message.vector_count == 1:
            if message.vector_1[1].x < half_width: left_vector = message.vector_1
            else: right_vector = message.vector_1
        
        target_x = half_width
        OFFSET_BOT, OFFSET_TOP = 240.0 / 2.0, 160.0 / 2.0
        
        if left_vector and right_vector:
            mid_bot_x = (self.project_x(left_vector, 240.0) + self.project_x(right_vector, 240.0)) / 2.0
            mid_top_x = (self.project_x(left_vector, 140.0) + self.project_x(right_vector, 140.0)) / 2.0
            target_x = 0.7 * mid_top_x + 0.3 * mid_bot_x
        elif left_vector:
            mid_bot_x = self.project_x(left_vector, 240.0) + OFFSET_BOT
            mid_top_x = self.project_x(left_vector, 140.0) + OFFSET_TOP
            target_x = 0.7 * mid_top_x + 0.3 * mid_bot_x
        elif right_vector:
            mid_bot_x = self.project_x(right_vector, 240.0) - OFFSET_BOT
            mid_top_x = self.project_x(right_vector, 140.0) - OFFSET_TOP
            target_x = 0.7 * mid_top_x + 0.3 * mid_bot_x
        else:
            if self.frames_without_vectors > MAX_FRAMES_NO_VECTORS:
                self.target_speed = NO_VECTOR_SLOW_SPEED
            return

        error_norm = (target_x - half_width) / half_width
        steer = -(KP * error_norm + KD * (error_norm - self.prev_error))
        steer = max(TURN_MIN, min(TURN_MAX, steer + self.nudge_steer))

        self.prev_error = error_norm
        self.last_steer = steer
        self.target_turn = steer

        if self.mission_state != MissionState.APPROACHING_BUILDING:
            if abs(steer) > CURVE_STEER_THRESHOLD:
                self.target_speed = CURVE_SPEED
            else:
                self.target_speed = BASE_SPEED
        self.target_speed *= self.speed_factor

    # ── Phase 2 & 6 & 7: LiDAR Callback ────────────────────────────────────
    def lidar_callback(self, message):
        num = len(message.ranges)
        if num == 0:
            return

        # Always cache the latest scan for the turn pillar-tracker
        self.latest_scan = message

        def get_valid(slice_):
            return [r for r in slice_ if r > 0.05 and not math.isinf(r) and not math.isnan(r)]

        # ── Phase 2: Obstacle Avoidance (skip nudging during active turn) ──
        if not self.in_turn:
            front_center = get_valid(message.ranges[int(num * 8/18) : int(num * 10/18)])
            front_left   = get_valid(message.ranges[int(num * 10/18) : int(num * 12/18)])
            front_right  = get_valid(message.ranges[int(num * 6/18)  : int(num * 8/18)])

            min_front = min(
                [min(s) for s in [front_center, front_left, front_right] if s]
                or [float('inf')])

            if min_front < OBSTACLE_REACT_DIST:
                ml = min(front_left)  if front_left  else float('inf')
                mr = min(front_right) if front_right else float('inf')
                self.nudge_steer  = -NUDGE_STEER_MAGNITUDE if ml < mr else NUDGE_STEER_MAGNITUDE
                self.speed_factor = max(0.3, min_front / OBSTACLE_REACT_DIST)
                if min_front < OBSTACLE_EMERGENCY_DIST:
                    self.speed_factor = 0.0
            else:
                self.nudge_steer  = 0.0
                self.speed_factor = 1.0
        else:
            # While turning, disable obstacle nudging so it doesn't fight the
            # turn controller. Speed is managed by _execute_lidar_turn.
            self.nudge_steer  = 0.0
            self.speed_factor = 1.0

        # ── Phase 6: Zone Wall Detection (always active) ──────────────────
        left_sec  = get_valid(message.ranges[int(num * 12/18) : int(num * 15/18)])
        right_sec = get_valid(message.ranges[int(num * 3/18)  : int(num * 6/18)])

        wl = (min(left_sec)  < WALL_DETECT_RANGE and np.std(left_sec)  < WALL_DETECT_STDDEV) if left_sec  else False
        wr = (min(right_sec) < WALL_DETECT_RANGE and np.std(right_sec) < WALL_DETECT_STDDEV) if right_sec else False

        self.building_nearby   = wl or wr
        self.building_distance = min(
            min(left_sec)  if left_sec  else float('inf'),
            min(right_sec) if right_sec else float('inf'))

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
