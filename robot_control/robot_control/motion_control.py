import rclpy
from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from hello_helpers.hello_misc import HelloNode

from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import Point, Twist
from robot_interfaces.srv import Camera
import time

class MoveJoints(HelloNode):

    def __init__(self):
        HelloNode.__init__(self)
        HelloNode.main(
            self,
            'move_joints',
            'move_joints',
            wait_for_first_pointcloud=False
        )

        # Declare Constants
        self.mode = None
        self.gripper_open = True
        # Create services
        self.create_service(
            Camera,
            'adjust_camera',
            self.move_camera
        )

        self.set_nav_mode_srv = self.create_service(
            Trigger, 'set_navigation_mode', self.handle_set_navigation_mode)

        self.set_pos_mode_srv = self.create_service(
            Trigger, 'set_position_mode', self.handle_set_position_mode)

        # Create Clients

        self.nav_mode_client = self.create_client(
            Trigger,
            '/switch_to_navigation_mode'
        )

        self.pos_mode_client = self.create_client(
            Trigger,
            '/switch_to_position_mode'
        )
        # Create Publishers

        self.vel_pub = self.create_publisher(
            Twist,
            '/stretch/cmd_vel',
            10
        )

        # Create Subscribers

        self.create_subscription(
            String,
            '/stretch/mode',
            self.mode_cb,
            10
        )

    # --- Client callbacks ---
    def switch_mode(self, client, mode):
        """Calls the drivers mode change service."""

        # Check current mode
        if self.mode == mode:
            return True
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'Mode switch service not available')
            return False

        future = client.call_async(Trigger.Request())
        self.get_logger().info(f'Switching to {mode} mode')
        return True

    # --- Service callbacks ---
    def handle_set_navigation_mode(self, request, response):
        success = self.switch_mode(self.nav_mode_client, 'navigation')
        response.success = success
        response.message = 'Switched to navigation mode' if success else 'Failed'
        return response

    def handle_set_position_mode(self, request, response):
        success = self.switch_mode(self.pos_mode_client, 'position')
        response.success = success
        response.message = 'Switched to position mode' if success else 'Failed'
        return response

    # --- Subscriber callbacks ---
    def mode_cb(self, msg):
        """Checks the mode and updates internal mode."""

        self.mode = msg.data

    # --- Control functions ---
    def move_camera(self, request, response):
        """"Adjust the camera position."""

        # before moving joints, switch to position mode
        self.switch_mode(self.pos_mode_client, 'position')

        joint_positions = [float(x) for x in request.angles]
        pan = joint_positions[0]
        tilt = joint_positions[1]
        self.move_to_pose(
            {'joint_head_pan': pan, 'joint_head_tilt': tilt},
            blocking=True
        )
        self.get_logger().info(f'Moving camera to {pan}, {tilt}')
        return response

    def drive_robot(self, speed=0.1, duration=2.0):
        """Drive the robot using cmd_vel commands."""

        self.switch_mode(self.nav_mode_client, 'navigation')
        command = Twist()
        command.linear.x = speed
        command.angular.z = 0.0

        self.get_logger().info(f'Driving forward at {speed} m/s')
        end_time = self.get_clock().now().nanoseconds + int(duration * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            self.vel_pub.publish(command)
            time.sleep(0.1)  # publish at ~10Hz

        # stop
        command.linear.x = 0.0
        self.vel_pub.publish(command)

    def send_command(self):
        while not self.joint_state.position:
            self.get_logger().info('Waiting for joint state msg to arrive')
            continue

        self.get_logger().info('Moving lift')

        point = JointTrajectoryPoint()
        point.time_from_start = Duration(seconds=4.0).to_msg()
        point.positions = [-0.1] # lift height in meters

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.trajectory.joint_names = ['joint_lift']
        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.frame_id = 'base_link'

        self.trajectory_client.send_goal_async(trajectory_goal)
        self.get_logger().info('Goal Sent')

def main(args=None):
    node = MoveJoints()
    node.drive_robot()
    node.new_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()