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
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.2
SPEED_MAX = 1.5
TURN_MIN = -1.5
TURN_MAX = 1.5

# ── LANE FOLLOWING PD CONTROLLER ────────────────────────────────────────
KP = 0.6                      # Proportional gain (steer response to lateral offset)
KD = 0.15                     # Derivative gain (dampens oscillation)
BASE_SPEED = 0.8             # Forward speed on straight segments
CURVE_SPEED = 0.6            # Reduced speed when steering hard (curves)
CURVE_STEER_THRESHOLD = 0.3   # |steer| above this triggers curve-speed
NO_VECTOR_SLOW_SPEED = 0.20   # Crawl speed when lane vectors are lost
MAX_FRAMES_NO_VECTORS = 15    # Frames of dead-reckoning before slowing down
ASSUMED_LANE_WIDTH_PX = 230   # Approx lane width in pixels (single-vector fallback)

# ── TUNABLE CONSTANTS ─────────────────────────────────────────────
APPROACH_SPEED = 0.3
OBSTACLE_REACT_DIST = 1.1
OBSTACLE_EMERGENCY_DIST = 0.5
NUDGE_STEER_MAGNITUDE = 0.85
WALL_DETECT_RANGE = 2.0
WALL_DETECT_STDDEV = 0.3
QR_CLOSE_THRESHOLD = 3000
SIGN_TO_BUILDING = {
    'A': 'PATIENT_1', 'B': 'PATIENT_2', 'C': 'PATIENT_3',
    'X': 'HOSPITAL_1', 'Y': 'HOSPITAL_2', 'Z': 'HOSPITAL_3'
}

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
        self.subscription_server = self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_communication_callback, qos_profile_sensor_data)
        self.subscription_qr = self.create_subscription(String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)
        self.subscription_signs = self.create_subscription(String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', qos_profile_system_default)

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

        # Phase 7: Sign Turning State
        self.sign_memory = {}
        self.pending_turn = None
        self.in_intersection = False
        self.turn_start_time = 0.0

        # Phase 8: Mission State Machine
        self.mission_state = MissionState.LANE_FOLLOWING
        self.patient_sequence = ['A', 'B', 'C']
        self.patients_delivered = 0
        self.current_target_code = self.patient_sequence[self.patients_delivered]  # Start by seeking patient 1
        
        self.last_target_qr_time = 0.0
        self.building_detect_start_time = 0.0

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
            
            if self.last_qr_payload and getattr(self, 'debug_last_checked_qr', None) != self.last_qr_payload:
                self.get_logger().info(f"[DEBUG] State: {self.mission_state} | Seeking: '{target_building}' | Just Saw: '{self.last_qr_payload}'")
                self.debug_last_checked_qr = self.last_qr_payload
                
            if self.last_qr_payload == target_building:
                self.get_logger().info(f"Target building {target_building} spotted. Approaching...")
                self.mission_state = MissionState.APPROACHING_BUILDING

        elif self.mission_state == MissionState.APPROACHING_BUILDING:
            target_building = SIGN_TO_BUILDING.get(self.current_target_code)
            
            # Keep driving using line follower, but at approach speed
            self.target_speed = APPROACH_SPEED
            
            # Check if camera has driven past the SPECIFIC target QR code
            qr_disappeared = (current_time - getattr(self, 'last_target_qr_time', 0.0)) > 0.5
            
            self.get_logger().info(f"[DEBUG APPROACH] QR lost: {qr_disappeared} | Lidar near: {self.building_nearby}", throttle_duration_sec=1.5)
            
            if qr_disappeared:
                # Once QR is passed, look for the wall on LIDAR
                if self.building_nearby:
                    if self.building_detect_start_time == 0.0:
                        self.building_detect_start_time = current_time
                        self.get_logger().info("LIDAR wall detected alongside buggy. Starting 2.5s stop timer...")
                    elif current_time - self.building_detect_start_time >= 2.5:
                        self.get_logger().info(f"In zone for {target_building}. Stopping.")
                        self.building_detect_start_time = 0.0  # Reset
                        if target_building.startswith("PATIENT"):
                            self.mission_state = MissionState.STOPPED_AT_PATIENT
                        else:
                            self.mission_state = MissionState.STOPPED_AT_HOSPITAL
                else:
                    if self.building_detect_start_time != 0.0:
                        self.get_logger().info("LIDAR wall lost, resetting stop timer.")
                    # Reset timer if building is lost
                    self.building_detect_start_time = 0.0

        elif self.mission_state == MissionState.STOPPED_AT_PATIENT:
            self.target_speed = 0.0
            self.target_turn = 0.0
            target_building = SIGN_TO_BUILDING.get(self.current_target_code)
            
            # Send exactly what was on the QR code
            exact_qr_payload = f"{{LOC: {target_building}}}"
            self.send_to_server(exact_qr_payload)
            
            self.server_ack_received = False
            self.mission_state = MissionState.WAITING_HOSPITAL_ASSIGNMENT
            self.server_retry_time = current_time + 3333333.0

        elif self.mission_state == MissionState.WAITING_HOSPITAL_ASSIGNMENT:
            self.target_speed = 0.0
            self.target_turn = 0.0
            if not self.server_ack_received and current_time > self.server_retry_time:
                self.get_logger().warn("Server ACK timeout. Retrying...")
                self.send_to_server(self.last_sent_payload)
                self.server_retry_time = current_time + 3333333.0
            # Transition handled in server_communication_callback when hospital is assigned

        elif self.mission_state == MissionState.STOPPED_AT_HOSPITAL:
            self.target_speed = 0.0
            self.target_turn = 0.0
            target_building = SIGN_TO_BUILDING.get(self.current_target_code)
            
            # Send exactly what was on the QR code
            exact_qr_payload = f"{{LOC: {target_building}}}"
            self.send_to_server(exact_qr_payload)
            
            self.server_ack_received = False
            self.mission_state = MissionState.WAITING_NEXT_PATIENT
            self.server_retry_time = current_time + 3333333.0

        elif self.mission_state == MissionState.WAITING_NEXT_PATIENT:
            self.target_speed = 0.0
            self.target_turn = 0.0
            if not self.server_ack_received and current_time > self.server_retry_time:
                self.get_logger().warn("Server ACK timeout. Retrying...")
                self.send_to_server(self.last_sent_payload)
                self.server_retry_time = current_time + 3333333.0
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
                
            if payload == "OK":
                self.get_logger().info("Server confirmed parking! RUN COMPLETE!")
                return

            if self.mission_state == MissionState.WAITING_HOSPITAL_ASSIGNMENT:
                if payload in ['X', 'Y', 'Z']:
                    self.current_target_code = payload
                    self.mission_state = MissionState.NAVIGATING_TO_HOSPITAL
                    self.last_qr_payload = None  # Reset QR memory
                    self.get_logger().info(f"Assigned Hospital: {payload}. Navigating...")

            elif self.mission_state == MissionState.WAITING_NEXT_PATIENT:
                self.patients_delivered += 1
                self.get_logger().info(f"Delivered {self.patients_delivered}/3 patients.")
                
                if self.patients_delivered >= 3:
                    self.mission_state = MissionState.MISSION_COMPLETE
                    self.get_logger().info("Sending final PARKED message to wrap up.")
                    self.send_to_server("PARKED")
                else:
                    self.current_target_code = self.patient_sequence[self.patients_delivered]
                    self.mission_state = MissionState.LANE_FOLLOWING
                    self.last_qr_payload = None # Reset QR memory
                    self.get_logger().info(f"Strict Sequence: Next target is {self.current_target_code}. Seeking...")

    def qr_detection_callback(self, message):
        parts = message.data.split("|bbox=")
        if len(parts) == 2:
            payload = parts[0]
            if "FAKE" in payload:
                self.get_logger().warn(f"Fake building '{payload}' detected! Ignoring.")
                return
            if getattr(self, 'last_qr_payload', None) != payload:
                self.get_logger().info(f"[DEBUG QR] Received new valid QR: '{payload}' (bbox area: {parts[1]})")
            self.last_qr_payload = payload
            self.last_qr_bbox_area = int(parts[1])
            
            # Only update the disappearance timer if it's our actual target
            if payload == SIGN_TO_BUILDING.get(self.current_target_code):
                self.last_target_qr_time = time.time()

    # ── Phase 7: Sign Board Detection ──
    def sign_board_callback(self, message):
        signs = message.data.split(",")
        for sign in signs:
            if ":" in sign:
                code, direction = sign.split(":")
                self.sign_memory[code] = direction
                
        # If we see our target on a sign, prepare to turn
        if self.current_target_code in self.sign_memory:
            direction = self.sign_memory[self.current_target_code]
            if direction in ['LEFT', 'RIGHT', 'STRAIGHT']:
                self.pending_turn = direction

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
        
        # ── Phase 7 Intersection Turn Execution ──
        # When vectors are lost (intersection) and we have a pending turn
        if message.vector_count == 0:
            self.frames_without_vectors += 1
            if self.pending_turn and self.frames_without_vectors > 5:
                if not self.in_intersection:
                    self.in_intersection = True
                    self.turn_start_time = time.time()
                    self.get_logger().info(f"Executing Intersection Turn: {self.pending_turn}")
                
                # Execute hardcode turn for minimum duration
                if self.pending_turn == 'LEFT':
                    self.target_turn = 1.0
                elif self.pending_turn == 'RIGHT':
                    self.target_turn = -1.0
                else:
                    self.target_turn = 0.0
                    
                self.target_speed = CURVE_SPEED
                return
        else:
            # Vectors are present. If we were turning and enough time passed, complete the turn
            if self.in_intersection and (time.time() - self.turn_start_time > 1.5):
                self.in_intersection = False
                self.pending_turn = None
                self.sign_memory.clear()
                self.get_logger().info("Intersection Cleared. Resuming Lane Follow.")
                
            self.frames_without_vectors = 0
            
        if self.in_intersection:
            return # Let the turn complete

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
                self.get_logger().warn("Lost lane vectors! Crawling blindly.", throttle_duration_sec=2.0)
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
                self.get_logger().info(f"Taking a curve (steer: {steer:.2f})", throttle_duration_sec=1.5)
            else:
                self.target_speed = BASE_SPEED
                self.get_logger().info("Following lane lines straight", throttle_duration_sec=3.0)
        self.target_speed *= self.speed_factor

    # ── Phase 2 & 6: Obstacle and Building Lidar Detection ──
    def lidar_callback(self, message):
        num = len(message.ranges)
        if num == 0: return
            
        def get_valid(slice):
            return [r for r in slice if r > 0.05 and r != float('inf')]
            
        front_center = get_valid(message.ranges[int(num * 17/36) : int(num * 19/36)])
        front_left = get_valid(message.ranges[int(num * 19/36) : int(num * 22/36)])
        front_right = get_valid(message.ranges[int(num * 14/36) : int(num * 17/36)])
        
        min_front = min([min(s) for s in [front_center, front_left, front_right] if s] or [float('inf')])
        
        if min_front < OBSTACLE_REACT_DIST:
            ml = min(front_left) if front_left else float('inf')
            mr = min(front_right) if front_right else float('inf')
            self.nudge_steer = -NUDGE_STEER_MAGNITUDE if ml < mr else NUDGE_STEER_MAGNITUDE
            self.speed_factor = max(0.3, min_front / OBSTACLE_REACT_DIST)
            if min_front < OBSTACLE_EMERGENCY_DIST:
                self.speed_factor = 0.0
        else:
            self.nudge_steer = 0.0
            self.speed_factor = 1.0
            
        # Phase 6: Zone Wall Detection (Directly 90 degrees Left and Right +/- 1 degree)
        left_sec = get_valid(message.ranges[int(num * 269/360) : int(num * 271/360)])
        right_sec = get_valid(message.ranges[int(num * 89/360) : int(num * 91/360)])
        
        wl = min(left_sec) < WALL_DETECT_RANGE and np.std(left_sec) < WALL_DETECT_STDDEV if left_sec else False
        wr = min(right_sec) < WALL_DETECT_RANGE and np.std(right_sec) < WALL_DETECT_STDDEV if right_sec else False
        
        self.building_nearby = wl or wr
        self.building_distance = min([min(left_sec) if left_sec else float('inf'), min(right_sec) if right_sec else float('inf')])

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
