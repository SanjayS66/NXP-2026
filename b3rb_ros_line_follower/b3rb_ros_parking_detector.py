import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from synapse_msgs.msg import EdgeVectors, ServerCommunication
import time
import numpy as np

class LidarParkingDetector(Node):
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
        
        # Lidar state
        self.wall_on_left = False
        self.cones_on_right = False
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

    def lidar_cb(self, msg):
        num = len(msg.ranges)
        if num == 0: return
        
        def get_slice(start_deg, end_deg):
            start_idx = int(num * start_deg / 360.0)
            end_idx = int(num * end_deg / 360.0)
            return [r for r in msg.ranges[start_idx:end_idx] if 0.05 < r < 30.0]
            
        # 180 is FRONT, 90 is RIGHT, 270 is LEFT
        
        # Wall detection on left (250 to 290 deg)
        left_slice = get_slice(250, 290)
        self.wall_on_left = False
        if len(left_slice) > 5:
            # A solid flat wall will have a low standard deviation and be close
            if min(left_slice) < 2.5 and np.std(left_slice) < 0.5:
                self.wall_on_left = True
                
        # Cone detection on right (45 to 135 deg to catch them as we approach)
        right_slice = get_slice(45, 135)
        self.cones_on_right = False
        if len(right_slice) > 0:
            if min(right_slice) < 2.5:
                self.cones_on_right = True

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
            
            # Check the strict conditions: Wall on left AND cones on right
            if self.wall_on_left and self.cones_on_right:
                self.get_logger().info("Detected Hospital Wall (Left) and Cones (Right)!")
                self.set_state("TURN_RIGHT")
                
        elif self.state == "TURN_RIGHT":
            # Very slow speed and max steering to achieve the tightest possible turning radius.
            self.target_speed = 0.25
            self.target_turn = -1.5
            
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