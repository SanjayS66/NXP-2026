import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from synapse_msgs.msg import EdgeVectors, ServerCommunication
import time
import math
import numpy as np

class LidarParkingDetector(Node):

    # --- Cone-fence geometry ---
    # Each parking barrier is made of exactly 3 real cones. Calibrated
    # from two independent, map-agnostic sources that agree closely:
    #   1) Actual cone <pose> placements in Raceway_1.sdf (Hospital 1 &
    #      2 fences, both consistent): first-to-last-cone span of a
    #      3-cone barrier = 0.733 - 0.744m; neighboring cones within a
    #      barrier are 0.24m - 0.41m apart.
    #   2) The reusable Parking_Lot.obj base-plate prefab that the cones
    #      are placed on: each individual spot's plate is 0.72m deep
    #      along the axis the barrier runs - i.e. the same ~0.72-0.74m
    #      figure, coming straight from the reusable model geometry
    #      rather than one hand-placed instance. This should hold on
    #      any map built from the same spot/cone prefabs.
    # A support pillar is a single solid object. Even if it happens to
    # sit CLOSER to the last real cone than cones sit to each other (as
    # observed in sim), it can only ever contribute ONE object to the
    # scan - it can't fake "three objects spanning ~0.73m" on its own.
    # So instead of asking "is something nearby" (which the pillar
    # passes trivially), we look for that specific 3-object/~0.73m
    # fingerprint.
    CONE_NEAR_THRESH = 2.5          # max range to consider an object at all
    OBJECT_POINT_GAP = 0.12         # max cartesian gap between LIDAR points on the SAME object
    BARRIER_ADJ_MIN = 0.15          # min real gap between two neighboring cones in a barrier
    BARRIER_ADJ_MAX = 0.50          # max real gap between two neighboring cones in a barrier (measured max 0.41 + margin)
    BARRIER_SPAN_MIN = 0.65         # min real span of a 3-cone barrier (two sources agree at 0.72-0.744, minus margin)
    BARRIER_SPAN_MAX = 0.85         # max real span of a 3-cone barrier (two sources agree at 0.72-0.744, plus margin)

    def __init__(self):
        super().__init__('parking_driver')
        
        self.sub_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.sub_vectors = self.create_subscription(EdgeVectors, '/edge_vectors', self.vectors_cb, 10)
        self.pub_joy = self.create_publisher(Joy, '/cerebri/in/joy', 10)
        self.pub_server = self.create_publisher(ServerCommunication, '/ServerCommunication', 10)
        
        self.state = "SEARCHING"
        self.start_time = time.time()
        self.state_time = time.time()
        
        self.timer = self.create_timer(0.1, self.tick)
        self.uid = 100
        self.target_speed = 0.5
        self.target_turn = 0.0 

        # +1.0 => turn left (wall on right, cones on left)
        # -1.0 => turn right (wall on left, cones on right)
        self.turn_dir = -1.0

        # Lidar state
        self.wall_on_left = False
        self.wall_on_right = False
        self.cones_on_right = False
        self.cones_on_left = False
        self.front_dist = 10.0
        self.front_left_dist = 10.0
        self.front_right_dist = 10.0
        self.left_side_dist = 10.0
        self.right_side_dist = 10.0

        # Lane keeping state
        self.prev_error = 0.0

    def _project_x(self, vector, target_y):
        dx = vector[0].x - vector[1].x
        dy = vector[0].y - vector[1].y
        if abs(dy) < 1e-4:
            return vector[1].x
        return vector[1].x + (target_y - vector[1].y) * (dx / dy)

    def vectors_cb(self, msg):
        if self.state != "SEARCHING":
            return
            
        half_width = 160.0
        target_x = half_width
        
        if msg.vector_count == 2:
            mid = (self._project_x(msg.vector_1, 200.0) + self._project_x(msg.vector_2, 200.0)) / 2.0
            target_x = mid
        elif msg.vector_count == 1:
            if msg.vector_1[1].x < half_width:
                target_x = self._project_x(msg.vector_1, 200.0) + 120.0
            else:
                target_x = self._project_x(msg.vector_1, 200.0) - 120.0
                
        error_norm = (target_x - half_width) / half_width
        steer = -(0.6 * error_norm + 0.15 * (error_norm - self.prev_error))
        self.prev_error = error_norm
        
        self.target_turn = max(-1.5, min(1.5, steer))

    def _points_to_objects(self, points):
        """Group consecutive (x, y) LIDAR hits into discrete physical
        objects. Multiple beams landing on the same cone are only ~cm
        apart -> same object. A real gap between two separate objects
        (two cones, or a cone and the pillar) is bigger. Returns a list
        of object centroids in angular order."""
        objects = []
        current = []
        for p in points:
            if current and math.hypot(p[0] - current[-1][0], p[1] - current[-1][1]) > self.OBJECT_POINT_GAP:
                cx = sum(q[0] for q in current) / len(current)
                cy = sum(q[1] for q in current) / len(current)
                objects.append((cx, cy))
                current = []
            current.append(p)
        if current:
            cx = sum(q[0] for q in current) / len(current)
            cy = sum(q[1] for q in current) / len(current)
            objects.append((cx, cy))
        return objects

    def _has_barrier_signature(self, objects):
        """Slide a 3-object window across the detected objects and check
        whether any window matches the real, measured 3-cone barrier
        fingerprint (tight, consistent inter-cone gaps AND a total span
        of ~0.74m). A lone pillar - no matter how close it sits to the
        nearest real cone - can only ever be ONE object in the window,
        so it can't produce this signature by itself."""
        for i in range(len(objects) - 2):
            a, b, c = objects[i], objects[i + 1], objects[i + 2]
            gap_ab = math.hypot(a[0] - b[0], a[1] - b[1])
            gap_bc = math.hypot(b[0] - c[0], b[1] - c[1])
            span_ac = math.hypot(a[0] - c[0], a[1] - c[1])
            if (self.BARRIER_ADJ_MIN <= gap_ab <= self.BARRIER_ADJ_MAX and
                    self.BARRIER_ADJ_MIN <= gap_bc <= self.BARRIER_ADJ_MAX and
                    self.BARRIER_SPAN_MIN <= span_ac <= self.BARRIER_SPAN_MAX):
                return True
        return False

    def _detect_cone_barrier(self, msg, num, start_deg, end_deg):
        """Convert a slice of the scan to cartesian points and check for
        the real 3-cone barrier fingerprint within it."""
        start_idx = int(num * start_deg / 360.0)
        end_idx = int(num * end_deg / 360.0)
        points = []
        for i in range(start_idx, end_idx):
            r = msg.ranges[i]
            if 0.05 < r < self.CONE_NEAR_THRESH:
                angle = 2.0 * math.pi * i / num
                points.append((r * math.cos(angle), r * math.sin(angle)))
        objects = self._points_to_objects(points)
        return self._has_barrier_signature(objects)

    def lidar_cb(self, msg):
        num = len(msg.ranges)
        if num == 0: return
        
        def get_slice(start_deg, end_deg):
            start_idx = int(num * start_deg / 360.0)
            end_idx = int(num * end_deg / 360.0)
            return [r for r in msg.ranges[start_idx:end_idx] if 0.05 < r < 30.0]

        # 180 is FRONT, 90 is RIGHT, 270 is LEFT

        # --- Wall detection (flat, solid, close => low std dev) ---
        # Left side (approaching with wall on our left, cones on our right)
        left_slice = get_slice(250, 290)
        self.wall_on_left = False
        if len(left_slice) > 5:
            if min(left_slice) < 2.5 and np.std(left_slice) < 0.5:
                self.wall_on_left = True

        # Right side (mirror, for approaching from the opposite direction)
        right_wall_slice = get_slice(70, 110)
        self.wall_on_right = False
        if len(right_wall_slice) > 5:
            if min(right_wall_slice) < 2.5 and np.std(right_wall_slice) < 0.5:
                self.wall_on_right = True

        # --- Cone-row detection: real 3-cone barrier fingerprint ---
        # Right side (original approach direction)
        self.cones_on_right = self._detect_cone_barrier(msg, num, 45, 135)

        # Left side (mirror, for the opposite approach direction)
        self.cones_on_left = self._detect_cone_barrier(msg, num, 225, 315)

        self.front_dist = min(get_slice(170, 190) + [10.0])
        self.front_left_dist = min(get_slice(185, 225) + [10.0])
        self.front_right_dist = min(get_slice(135, 175) + [10.0])
        self.left_side_dist = min(get_slice(250, 290) + [10.0])
        self.right_side_dist = min(get_slice(70, 110) + [10.0])

    def set_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_time = time.time()
            self.get_logger().info(f"--- Switched to state: {new_state} ---")

    def tick(self):
        now = time.time()
        
        if self.state == "SEARCHING":
            self.target_speed = 0.5
            # target_turn is set by vectors_cb (lane following)

            # Original layout: wall on left, cone fence on right => turn right.
            if self.wall_on_left and self.cones_on_right:
                self.get_logger().info("Detected Hospital Wall (Left) and Cones (Right)!")
                self.turn_dir = -1.0
                self.set_state("TURNING")
            # Mirrored layout (opposite approach direction): wall on
            # right, cone fence on left => turn left.
            elif self.wall_on_right and self.cones_on_left:
                self.get_logger().info("Detected Hospital Wall (Right) and Cones (Left)!")
                self.turn_dir = 1.0
                self.set_state("TURNING")

        elif self.state == "TURNING":
            # Very slow speed and max steering to achieve the tightest possible turning radius.
            # Same magnitudes as before, just signed by self.turn_dir.
            self.target_speed = 0.25
            self.target_turn = 1.5 * self.turn_dir
            
            # Give the buggy enough time to physically complete the 90-degree sweep 
            if (now - self.state_time) > 3.5:
                if self.front_left_dist < 3.0 and self.front_right_dist < 3.0:
                    self.set_state("CENTERING")
                    
        elif self.state == "CENTERING":
            # Creep forward very slowly while balancing left/right distances
            self.target_speed = 0.15
            error = self.front_left_dist - self.front_right_dist
            
            # More aggressive correction to keep it centered
            self.target_turn = max(-1.0, min(1.0, error * 2.0))
            
            # Stop if there are obstacles on three sides: left, right, and front.
            if self.left_side_dist < 1.2 and self.right_side_dist < 1.2 and self.front_dist < 1.5:
                self.get_logger().info("Obstacles detected on 3 sides! Fully parked.")
                self.set_state("STOPPED")
            # Fallback timeout
            elif (now - self.state_time) > 8.0:
                self.set_state("STOPPED")
                
        elif self.state == "STOPPED":
            self.target_speed = 0.0
            self.target_turn = 0.0
            
            # Spam PARKED message every 2 seconds
            if int(now - self.start_time) % 2 == 0:
                self.send_parked()

        # Publish drive commands
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.pub_joy.publish(msg)

    def send_parked(self):
        msg = ServerCommunication()
        msg.src = 1
        msg.dest = 2
        msg.uid = self.uid
        self.uid = (self.uid + 1) % 256
        msg.ack = 0
        msg.msg = 'PARKED'
        self.pub_server.publish(msg)
        self.get_logger().info('Sent PARKED')

def main(args=None):
    rclpy.init(args=args)
    node = LidarParkingDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
