"""Node to follow waypoints and take video feed of various poses at each waypoint."""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from hello_helpers.hello_misc import HelloNode
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult
from stretch_body.robot import Robot

from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from robot_interfaces.srv import Camera
from sensor_msgs.msg import Image, CameraInfo
from math import atan2, sqrt, pi
from copy import deepcopy
import time, os, threading
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np

class MoveJoints(HelloNode):

    def __init__(self):
        HelloNode.__init__(self)
        HelloNode.main(
            self,
            'move_joints',
            'move_joints',
            wait_for_first_pointcloud=False
        )

        # Declare Parameters
        if not self.has_parameter('stretch_waypoints'):
            self.declare_parameter('stretch_waypoints', [0.0, 0.0])
        if not self.has_parameter('d435_poses_pan'):
            self.declare_parameter('d435_poses_pan', [0.0])
        if not self.has_parameter('d435_poses_tilt'):
            self.declare_parameter('d435_poses_tilt', [0.0])
        if not self.has_parameter('d405_poses_tilt'):
            self.declare_parameter('d405_poses_tilt', [0.0])
        if not self.has_parameter('d405_poses_yaw'):
            self.declare_parameter('d405_poses_yaw', [0.0])
        if not self.has_parameter('d405_poses_lift'):
            self.declare_parameter('d405_poses_lift', [0.4])

        pans = self.get_parameter('d435_poses_pan').value
        self.get_logger().info(f'd435 pans are {pans}')
        tilts = self.get_parameter('d435_poses_tilt').value
        self.d435_poses = list(zip(pans, tilts))  # -> [(pan, tilt), ...]

        self.d405_tilts = self.get_parameter('d405_poses_tilt').value
        self.d405_yaws = self.get_parameter('d405_poses_yaw').value
        self.d405_lifts = self.get_parameter('d405_poses_lift').value

        self.waypoints = [self.get_parameter('stretch_waypoints').value]
        self.get_logger().info(f'waypoints are {self.waypoints}')

        # Declare Constants
        self.stretch_mode = None
        self.gripper_open = True

        # Variables for frame captures
        self.capture_dir = os.path.expanduser('~/stretch_user/captures')
        os.makedirs(self.capture_dir, exist_ok=True)

        self.bridge = CvBridge()
        self._capture_lock = threading.Lock()
        self._d435_capture_requested = False
        self._d435_captured_frame = None
        self._d405_capture_requested = False
        self._d405_captured_frame = None

        # Declare global vars
        self.navigator = BasicNavigator()

        # Set initial pose for the bot
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = 0.0
        initial_pose.pose.position.y = 0.0
        initial_pose.pose.orientation.z = 0.0
        initial_pose.pose.orientation.w = 1.0
        self.navigator.setInitialPose(initial_pose)

        self.navigator.waitUntilNav2Active()

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
            '/mode',
            self.mode_cb,
            10
        )

        # RGB-D subscriptions for mounted camera
        rgb_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/color/image_raw'
        )

        depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/aligned_depth_to_color/image_raw'
        )

        info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            '/camera/aligned_depth_to_color/camera_info'
        )

        # RGB-D subscriptions for gripper camera

        gripper_rgb_sub = message_filters.Subscriber(
            self,
            Image,
            '/gripper_camera/color/image_rect_raw'
        )

        gripper_depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/gripper_camera/aligned_depth_to_color/image_raw'
        )

        gripper_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            '/gripper_camera/aligned_depth_to_color/camera_info'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.5 # maybe adjust slop
        )

        self.gripper_ts = message_filters.ApproximateTimeSynchronizer(
            [gripper_rgb_sub, gripper_depth_sub, gripper_info_sub], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.synced_frame_cb)
        self.gripper_ts.registerCallback(self.gripper_synced_frame_cb)


    # --- Client callbacks ---
    def switch_mode(self, client, mode, timeout_sec=2.0):
        """Calls the drivers mode change service."""

        # Check current mode
        if self.stretch_mode == mode:
            return True
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'Mode switch service not available')
            return False

        future = client.call_async(Trigger.Request())

        start = self.get_clock().now()
        while not future.done():
            time.sleep(0.05)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn(f'Mode switch service call timed out for {mode}')
                return False
            
        result = future.result()
        if result is None or not result.success:
            self.get_logger().warn(f'Mode switch service reported failure for {mode}')
            return False
        
        # The service call succeeding means the driver accepted the request,
        # but self.mode only updates once /stretch/mode publishes the change.
        # Wait for that so callers can trust self.mode immediately after this returns.
        start = self.get_clock().now()
        while self.stretch_mode != mode:
            time.sleep(0.05)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn(
                    f'Mode switch service succeeded but /stretch/mode never reported {mode}'
                )
                return False
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

        self.stretch_mode = msg.data

    def synced_frame_cb(self, rgb_msg, depth_msg, info_msg):
        """Stores a camera frame from the mounted d435 when requested."""

        with self._capture_lock:
            if not self._d435_capture_requested:
                return
            self._d435_captured_frame = (rgb_msg, depth_msg, info_msg)
            self._d435_capture_requested = False

    def gripper_synced_frame_cb(self, rgb_msg, depth_msg, info_msg):
        """Stores a camera frame from the gripper d405 camera."""

        with self._capture_lock:
            if not self._d405_capture_requested:
                return
            
            self._d405_captured_frame = (rgb_msg, depth_msg, info_msg)
            self._d405_capture_requested = False
    # --- Helper functions ---
    def wait_until_settled(self, joint_names, vel_thresh=0.01, timeout_sec=2.0):
        """Block frame capture until joint vels fall below threshold."""

        start = self.get_clock().now()
        while True:
            time.sleep(0.05)
            if self.joint_state is not None and self.joint_state.velocity:
                idxs = [self.joint_state.name.index(j)
                        for j in joint_names if j in self.joint_state.name]
                vels = [abs(self.joint_state.velocity[i]) for i in idxs]
                if vels and max(vels) < vel_thresh:
                    return True
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn('Settle timeout, capturing anyway')
                return False
            
    def capture_frame(self, pose_name, timeout_sec=3.0, camera='d435'):
        """Request and block until one frame is captured."""

        assert hasattr(self, f'_{camera}_capture_requested')
        assert hasattr(self, f'_{camera}_captured_frame')
        flag_attr = f'_{camera}_capture_requested'
        frame_attr = f'_{camera}_captured_frame'

        with self._capture_lock:
            setattr(self, frame_attr, None)
            setattr(self, flag_attr, True)
        start = self.get_clock().now()
        while True:
            with self._capture_lock:
                frame = getattr(self, frame_attr)
            if frame is not None:
                break
            time.sleep(0.05)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn(f'Capture timed out at {pose_name} ({camera})')
                with self._capture_lock:
                    setattr(self, flag_attr, False)
                return None
 
        rgb_msg, depth_msg, info_msg = frame
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
 
        rgb_path = os.path.join(self.capture_dir, f'{pose_name}_{camera}_rgb.png')
        depth_path = os.path.join(self.capture_dir, f'{pose_name}_{camera}_depth.npy')
        cv2.imwrite(rgb_path, rgb_img)
        np.save(depth_path, depth_img)
        self.get_logger().info(f'Saved capture: {rgb_path}, {depth_path}')
        return rgb_img, depth_img


    # --- Control functions ---
    def follow_waypoints(self):
        """Sends the hello robot to a given list of points."""

        # switch to navigation mode 
        self.switch_mode(self.nav_mode_client, 'navigation')

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1.0

        # move through all poses in the waypoints array
        for wp_idx, pt in enumerate(self.waypoints):
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.header.stamp = self.navigator.get_clock().now().to_msg()

            # move to the first pose
            self.navigator.goToPose(pose)
            while not self.navigator.isTaskComplete():
                time.sleep(0.1)
            result = self.navigator.getResult()
            if result != TaskResult.SUCCEEDED:
                self.get_logger().warn('Failed to reach waypoint, skipping frame capture')
                continue

            time.sleep(0.5)


            # Now we need to move the camera around to get different views
            # first capture larger images with the d435
            for idx, cp in enumerate(self.d435_poses):
                self.camera_capture_pose(cp)
                self.wait_until_settled(['joint_head_pan', 'joint_head_tilt'])
                pose_name = f'wp{wp_idx}_d435_pose{idx}'
                self.capture_frame(pose_name, camera='d435')

            self.get_logger().info('Completed d435 camera poses')
            # capture smaller images with the d405

            # Send lift to various positions
            for idx, cp in enumerate(self.d405_lifts):
                self.camera_capture_pose(cp, camera='d405', joint='joint_lift')
                self.wait_until_settled(['joint_lift'])
                pose_name = f'wp{wp_idx}_d405_pose{idx}'
                self.capture_frame(pose_name, camera='d405')

                # Within each lift position, yaw and pitch the gripper
                for yaw_idx, yaw_pose in enumerate(self.d405_yaws):
                    self.camera_capture_pose(yaw_pose, camera='d405', joint='joint_wrist_yaw')
                    self.wait_until_settled((['joint_wrist_yaw']))
                    pose_name = f'wp{wp_idx}_d405_pose{idx}_yaw_{yaw_idx}'
                    self.capture_frame(pose_name, camera='d405')

                    # Within each yaw, pitch the gripper 
                    for p_idx, p_pose, in enumerate(self.d405_tilts):
                        self.camera_capture_pose(p_pose, camera='d405', joint='joint_wrist_pitch')
                        self.wait_until_settled((['joint_wrist_pitch']))
                        pose_name = f'wp{wp_idx}_d405_pose{idx}_yaw_{yaw_idx}_pitch_{p_idx}'
                        self.capture_frame(pose_name, camera='d405')

            # self.send_command(0.6, 'joint_lift')
            # self.wait_until_settled(['joint_lift'])
            # pose_name = f'wp_d405_pose1'
            # self.capture_frame(pose_name, camera='d405')
            # self.send_command(0.3, 'joint_lift')
            # pose_name = f'wp_d405_pose2'
            # self.capture_frame(pose_name, camera='d405')
            # self.send_command(0.3, 'joint_lift')

            # Send arm out and wrist tilt to 0.0
            # self.send_command(0.2, 'joint_arm_l1') # broken :(
            # self.send_command(0.0, 'joint_wrist_pitch')

    def camera_capture_pose(self, pose, camera='d435', joint='joint_lift'):
        """Move the camera on the bot to a given pose."""

        # grab tilt and pan from the list
        if camera=='d435':
            pan = pose[0]
            tilt = pose[1]

            request = Camera.Request()
            response = Camera.Response()
            request.angles = [pan, tilt]
        elif camera=='d405':
            yaw = pose
            extra = pose # TODO Adding this because the service requires a 2D array, should not affect anything
            request = Camera.Request()
            response = Camera.Response()
            request.angles = [yaw, extra]
        self.move_camera(request=request, response=response, camera=camera, joint=joint)

    def move_camera(self, request, response, camera='d435', joint='joint_lift'):
        """"Adjust the camera position."""

        # before moving joints, switch to position mode
        if self.stretch_mode != 'position':
            self.switch_mode(self.pos_mode_client, 'position')

        joint_positions = [float(x) for x in request.angles]
        pan = joint_positions[0]

        if camera=='d435':
            tilt = joint_positions[1]
            self.move_to_pose(
                {'joint_head_pan': pan, 'joint_head_tilt': tilt},
                blocking=True
            )
            self.get_logger().info(f'Moving camera to {pan}, {tilt}')

        elif camera=='d405':
            self.move_to_pose(
                {joint: pan},
                blocking=True
            )
            self.get_logger().info(f'Moving gripper camera joint: {joint} to {pan}')
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

    def send_command(self, position, link_name: String, timeout_sec=2.0):
        while not self.joint_state.position:
            self.get_logger().info('Waiting for joint state msg to arrive')
            time.sleep(0.05)

        self.get_logger().info(f'Moving {link_name}')

        point = JointTrajectoryPoint()
        point.time_from_start = Duration(seconds=4.0).to_msg()
        point.positions = [position] # lift height in meters

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.trajectory.joint_names = [link_name]
        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.frame_id = 'base_link'

        goal_future = self.trajectory_client.send_goal_async(trajectory_goal)
        self.get_logger().info('Goal Sent')

        start = self.get_clock().now()
        while not goal_future.done():
            time.sleep(0.05)
            if (self.get_clock().now() - start).nanoseconds / 1e9 > timeout_sec:
                self.get_logger().warn(f'Goal send timed out for {link_name}')
                return False
            
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f'Goal for {link_name} was rejected')
            return False
        
        result_future = goal_handle.get_result_async()
        start = self.get_clock().now()
        while not result_future.done():
            time.sleep(0.05)
            if (self.get_clock().now() - start).nanoseconds / 1e9 > timeout_sec:
                self.get_logger().warn(f'{link_name} move timed out')
                return False

        self.get_logger().info(f'{link_name} move complete')
        return True

def main(args=None):
    node = MoveJoints()
    node.follow_waypoints()
    node.new_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()