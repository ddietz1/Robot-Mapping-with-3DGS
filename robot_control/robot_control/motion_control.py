"""Node to follow waypoints and take video feed of various poses at each waypoint."""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from hello_helpers.hello_misc import HelloNode
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult
from stretch_body.robot import Robot

from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from robot_interfaces.srv import Camera, TestPoses
from sensor_msgs.msg import Image, CameraInfo
from math import atan2, sqrt, pi, sin, cos
from copy import deepcopy
import json
import time, os, threading
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

from nav2_msgs.action import ComputeAndTrackRoute, ComputeRoute, FollowPath
from rclpy.action import ActionClient

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

from visualization_msgs.msg import Marker

class MoveJoints(HelloNode):

    def __init__(self):
        HelloNode.__init__(self)
        HelloNode.main(
            self,
            'move_joints',
            'move_joints',
            wait_for_first_pointcloud=False
        )

        # Callback groups
        self.mode_cb_group = ReentrantCallbackGroup()
        self.default_cb_group = MutuallyExclusiveCallbackGroup()

        # Declare Parameters
        if not self.has_parameter('waypoints_x'):
            self.declare_parameter('waypoints_x', [0.0])
        if not self.has_parameter('waypoints_y'):
            self.declare_parameter('waypoints_y', [0.0])
        if not self.has_parameter('waypoints_yaw'):
            self.declare_parameter('waypoints_yaw', [0.0])
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
        if not self.has_parameter('run_num'):
            self.declare_parameter('run_num', 0)
        if not self.has_parameter('initial_pose'):
            self.declare_parameter('initial_pose', [0.0, 0.0, 0.0])

        # -- Robot Joints ---
        if not self.has_parameter('wrist_yaw'):
            self.declare_parameter('wrist_yaw', [-1.17, 4.27])
        if not self.has_parameter('wrist_pitch'):
            self.declare_parameter('wrist_pitch', [-1.57, 0.56])
        if not self.has_parameter('wrist_roll'):
            self.declare_parameter('wrist_roll', [-1.17, 4.27])
        if not self.has_parameter('head_pan'):
            self.declare_parameter('head_pan', [-4.0, 1.7])
        if not self.has_parameter('head_tilt'):
            self.declare_parameter('head_tilt', [-1.57, 0.56])

        self.joint_limits = {
            'wrist_yaw': self.get_parameter('wrist_yaw').value,
            'wrist_pitch': self.get_parameter('wrist_pitch').value,
            'wrist_roll': self.get_parameter('wrist_roll').value,
            'head_pan': self.get_parameter('head_pan').value,
            'head_tilt': self.get_parameter('head_tilt').value
        }

        pans = self.get_parameter('d435_poses_pan').value
        self.get_logger().info(f'd435 pans are {pans}')
        tilts = self.get_parameter('d435_poses_tilt').value
        self.d435_poses = list(zip(pans, tilts))  # -> [(pan, tilt), ...]
        self.get_logger().info(f'd435 poses are {self.d435_poses}')

        self.d405_tilts = self.get_parameter('d405_poses_tilt').value
        self.d405_yaws = self.get_parameter('d405_poses_yaw').value
        self.d405_lifts = self.get_parameter('d405_poses_lift').value

        xs = self.get_parameter('waypoints_x').value
        ys = self.get_parameter('waypoints_y').value
        yaws = self.get_parameter('waypoints_yaw').value
        self.waypoints = list(zip(xs, ys, yaws))  # -> [(x, y, yaw), ...]
        self.get_logger().info(f'waypoints are {self.waypoints}')

        self.run = self.get_parameter('run_num').value

        # Declare Constants
        self.stretch_mode = None
        self.gripper_open = True

        # Rotation constants defined by the URDF of the robot
        self._R1 = R.from_quat([-0.002, 1.000, 0.006, -0.003])   # base_link -> link_wrist_yaw
        self._R2 = R.from_quat([0.507, 0.497, 0.506, 0.491])     # base_link -> link_wrist_pitch
        self._R3 = R.from_quat([-0.012, -0.006, -0.709, 0.705])  # link_wrist_pitch -> gripper_camera_color_optical_frame
        self._R_mid = self._R1.inv() * self._R2

        # Variables for frame captures
        self.capture_dir = os.path.expanduser(f'~/stretch_user/captures/{self.run}')
        os.makedirs(self.capture_dir, exist_ok=True)

        self.bridge = CvBridge()
        self._capture_lock = threading.Lock()
        self._d435_capture_requested = False
        self._d435_captured_frame = None
        self._d405_capture_requested = False
        self._d405_captured_frame = None

        # Declare global vars
        self.navigator = BasicNavigator()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.route_client = ActionClient(
            self,
            ComputeRoute,
            'compute_route'
        )

        self.path_client = ActionClient(
            self,
            FollowPath,
            'follow_path'
        )

        # Set initial pose for the bot
        initial_pose_params = self.get_parameter('initial_pose').value
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = initial_pose_params[0]
        initial_pose.pose.position.y = initial_pose_params[1]
        initial_pose.pose.orientation.z = initial_pose_params[2]
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
        
        self.route_srv = self.create_service(
            Trigger,
            'follow_route',
            self.follow_route_cb
        )

        self.test_IK = self.create_service(
            TestPoses,
            'test_ik',
            self.test_pan_tilt_cb
        )

        # Create Clients

        self.nav_mode_client = self.create_client(
            Trigger,
            '/switch_to_navigation_mode',
            callback_group=self.mode_cb_group
        )

        self.pos_mode_client = self.create_client(
            Trigger,
            '/switch_to_position_mode',
            callback_group=self.mode_cb_group
        )
        # Create Publishers

        self.vel_pub = self.create_publisher(
            Twist,
            '/stretch/cmd_vel',
            10
        )

        self.test_marker_pub = self.create_publisher(
            Marker,
            '/test/marker',
            10
        )

        # Create Subscribers
        self.create_subscription(
            String,
            '/mode',
            self.mode_cb,
            10,
            callback_group=self.mode_cb_group
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
            '/gripper_camera/color/image_rect_raw',
            callback_group=self.mode_cb_group
        )

        gripper_depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/gripper_camera/aligned_depth_to_color/image_raw',
            callback_group=self.mode_cb_group
        )

        gripper_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            '/gripper_camera/aligned_depth_to_color/camera_info',
            callback_group=self.mode_cb_group
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.5 # maybe adjust slop
        )

        self.gripper_ts = message_filters.ApproximateTimeSynchronizer(
            [gripper_rgb_sub, gripper_depth_sub, gripper_info_sub], queue_size=10, slop=0.5
        )

        self.timer = self.create_timer(
            0.1,
            self.timer_cb,
            callback_group=self.mode_cb_group
        )

        self.ts.registerCallback(self.synced_frame_cb)
        self.gripper_ts.registerCallback(self.gripper_synced_frame_cb)

        self.test_marker = Pose()
        self.test_marker.position.x = 0.0
        self.test_marker.position.y = 0.0
        self.test_marker.position.z = 0.0

        self.move_to_pose({'joint_wrist_yaw': 0.0, 'joint_wrist_pitch': 0.0}, blocking=True)

    def timer_cb(self):
        """Timer callback."""

        self.marker = Marker()
        self.marker.header.frame_id = 'map'
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.id = 1
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD
        self.marker.pose = self.test_marker
        self.marker.scale.x = 0.2
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.2
        self.marker.color.a = 1.0
        self.marker.color.b = 1.0
        self.marker.color.r = 1.0
        self.test_marker_pub.publish(self.marker)

    # --- Testing Functions ---
    def test_pan_tilt_cb(self, request, response):
        """Temporary test service, hardcoded target pose for validation."""

        self.move_to_pose({'joint_wrist_yaw': 0.0, 'joint_wrist_pitch': 0.0}, blocking=True)
        test_pose = PoseStamped()
        test_pose.header.frame_id = 'map'
        test_pose.header.stamp = self.get_clock().now().to_msg()
        test_pose.pose.position.x = request.pose.position.x  # adjust based on robot's actual current map position
        test_pose.pose.position.y = request.pose.position.y
        test_pose.pose.position.z = request.pose.position.z  # roughly camera height

        test_pose.pose.orientation.x = request.pose.orientation.x
        test_pose.pose.orientation.y = request.pose.orientation.y
        test_pose.pose.orientation.z = request.pose.orientation.z
        test_pose.pose.orientation.w = request.pose.orientation.w
        self.get_logger().info(f'Test pose is x: {test_pose.pose.position.x}')
        self.get_logger().info(f'Test pose is y: {test_pose.pose.position.y}')
        self.get_logger().info(f'Test pose is z: {test_pose.pose.position.z}')

        self.test_marker = test_pose.pose
        self.get_logger().info('Test marker has been changed!')

        base_yaw_delta, wrist_yaw, wrist_pitch, residual = self.solve_base_and_wrist(test_pose)
        self.get_logger().info(
            f'DEBUG (no-nav): base_yaw_delta={base_yaw_delta:.3f}, wrist_yaw={wrist_yaw:.3f}, '
            f'wrist_pitch={wrist_pitch:.3f}, residual={residual:.4f}'
        )

        success = self.command_pose_d405(test_pose)
        response.success = bool(success)

        self.capture_frame('TestPose', camera='d405')
        return response

    # --- Client callbacks ---
    def switch_mode(self, client, mode, timeout_sec=2.0):
        """Calls the drivers mode change service."""

        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('Mode switch service not available')
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

        self.stretch_mode = mode
        return True
    # --- Service callbacks ---
    def follow_route_cb(self, request, response):
        """Service call to trigger the robot to follow a defined route."""
        threading.Thread(target=self.follow_route, args=([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],), kwargs={'loops': 1}, daemon=True).start()
        self.get_logger().info('Route following started in background')
        response.success = True
        return response

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

    # ----- Camera helpers -----

    def _camera_orientation_in_base(self, yaw, pitch):
        """Forward model: given wrist_yaw and wrist_pitch, compute the camera
        optical frame's orientation in base_link, using the real measured
        transforms (no re-derivation of axis conventions needed).

        Both joints rotate about their local -Z axis per the URDF.
        """
        Rz_yaw = R.from_euler('z', -yaw)
        Rz_pitch = R.from_euler('z', -pitch)
        return self._R1 * Rz_yaw * self._R_mid * Rz_pitch * self._R3

    def solve_wrist_for_direction(self, desired_forward_in_base, yaw_limits, pitch_limits):
        """Find (yaw, pitch) so the camera's forward axis matches
        desired_forward_in_base (a unit vector, in base_link frame).

        This runs entirely on the analytical model above — no physical robot
        movement required, so it's fast (milliseconds) even though it's a
        numerical search.
        """
        desired_forward_in_base = np.array(desired_forward_in_base) / np.linalg.norm(desired_forward_in_base)

        def cost(params):
            yaw, pitch = params
            cam_rot = self._camera_orientation_in_base(yaw, pitch)
            # Camera's own forward axis is +Z in its optical frame (per your
            # earlier RViz finding: Z-forward, X-left, Y-up)
            current_forward = cam_rot.apply([0.0, 0.0, 1.0])
            return 1.0 - np.dot(current_forward, desired_forward_in_base)  # 0 = perfect alignment

        yaw0 = np.clip(0.0, *yaw_limits)
        pitch0 = np.clip(0.0, *pitch_limits)

        result = minimize(
            cost, x0=[yaw0, pitch0], method='Nelder-Mead',
            bounds=[yaw_limits, pitch_limits],
            options={'xatol': 1e-4, 'fatol': 1e-6}
        )
        return result.x[0], result.x[1], result.fun  # yaw, pitch, residual error

    def solve_base_and_wrist(self, target_pose: PoseStamped, base_rotation_weight=0.05, proximity_weight=0.02, safety_margin=0.05):
        """Jointly solve for base yaw rotation + wrist yaw/pitch to face
        target_pose's orientation, minimizing base rotation while respecting
        wrist joint limits.

        Returns (base_yaw_delta, wrist_yaw, wrist_pitch, residual).
        """
        q = target_pose.pose.orientation
        target_rot = R.from_quat([q.x, q.y, q.z, q.w])
        desired_forward_map = target_rot.apply([0.0, 0.0, 1.0])

        try:
            base_tf = self.tf_buffer.lookup_transform(
                'base_link', 'map', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().warn(f'Could not get map->base_link transform: {e}')
            return None, None, None, None

        q_base = base_tf.transform.rotation
        map_to_base_rot = R.from_quat([q_base.x, q_base.y, q_base.z, q_base.w])
        desired_forward_base = map_to_base_rot.apply(desired_forward_map)

        yaw_min, yaw_max = self.joint_limits['wrist_yaw']
        pitch_min, pitch_max = self.joint_limits['wrist_pitch']

        # Shrink the usable range to stay clear of hard limits / collision zones
        yaw_min_safe, yaw_max_safe = yaw_min + safety_margin, yaw_max - safety_margin
        pitch_min_safe, pitch_max_safe = pitch_min + safety_margin, pitch_max - safety_margin

        # Read current wrist position so we can prefer solutions close to it
        current_yaw = self.joint_state.position[self.joint_state.name.index('joint_wrist_yaw')] \
            if self.joint_state and 'joint_wrist_yaw' in self.joint_state.name else 0.0
        current_pitch = self.joint_state.position[self.joint_state.name.index('joint_wrist_pitch')] \
            if self.joint_state and 'joint_wrist_pitch' in self.joint_state.name else 0.0

        def cost(params):
            base_yaw_delta, wrist_yaw, wrist_pitch = params
            rotated_target = R.from_euler('z', -base_yaw_delta).apply(desired_forward_base)
            cam_rot = self._camera_orientation_in_base(wrist_yaw, wrist_pitch)
            current_forward = cam_rot.apply([0.0, 0.0, 1.0])
            alignment_error = 1.0 - np.dot(current_forward, rotated_target)
            base_penalty = base_rotation_weight * abs(base_yaw_delta)
            proximity_penalty = proximity_weight * (
                (wrist_yaw - current_yaw) ** 2 + (wrist_pitch - current_pitch) ** 2
            )
            return alignment_error + base_penalty + proximity_penalty

        x0 = np.array([0.0, 0.0, 0.0])

        # Build an explicit initial simplex so base_yaw_delta is actually explored --
        # Nelder-Mead's default simplex step for a zero-valued x0 component is
        # tiny (~0.00025), which effectively never perturbs base_yaw_delta.
        best_result = None
        starting_points = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
        ]
        for start in starting_points:
            x0 = np.array(start)
            simplex = np.array([
                x0,
                x0 + [0.5, 0.0, 0.0],
                x0 + [0.0, 0.3, 0.0],
                x0 + [0.0, 0.0, 0.3],
            ])
            r = minimize(
                cost, x0=x0, method='Nelder-Mead',
                bounds=[(-np.pi, np.pi), (yaw_min_safe, yaw_max_safe), (pitch_min_safe, pitch_max_safe)],
                options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 300, 'initial_simplex': simplex}
            )
            if best_result is None or r.fun < best_result.fun:
                best_result = r
        base_yaw_delta, wrist_yaw, wrist_pitch = best_result.x
        return base_yaw_delta, wrist_yaw, wrist_pitch, best_result.fun

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

        request = self.get_clock().now()
        with self._capture_lock:
            setattr(self, frame_attr, None)
            setattr(self, flag_attr, True)
        start = self.get_clock().now()
        while True:
            with self._capture_lock:
                frame = getattr(self, frame_attr)
            if frame is not None:
                rgb_msg, depth_msg, info_msg = frame
                stamp = rgb_msg.header.stamp
                frame_time = rclpy.time.Time.from_msg(stamp)
                if frame_time > request:
                    break  # genuinely fresh, accept it
                else:
                    # stale match slipped through; discard and keep waiting
                    with self._capture_lock:
                        setattr(self, frame_attr, None)
                        setattr(self, flag_attr, True)
            time.sleep(0.05)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn(f'Capture timed out at {pose_name} ({camera})')
                with self._capture_lock:
                    setattr(self, flag_attr, False)
                return None
 
        rgb_msg, depth_msg, info_msg = frame
        try:
            cam_tf = self.tf_buffer.lookup_transform(
                'map', 'camera_color_optical_frame', rgb_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            pose_record = {
                'position': [cam_tf.transform.translation.x,
                            cam_tf.transform.translation.y,
                            cam_tf.transform.translation.z],
                'orientation': [cam_tf.transform.rotation.x, cam_tf.transform.rotation.y,
                                cam_tf.transform.rotation.z, cam_tf.transform.rotation.w],
            }
        except Exception as e:
            self.get_logger().warn(f'Could not log map-frame pose for {pose_name}: {e}')
            pose_record = None

        if pose_record is not None:
            pose_path = os.path.join(self.capture_dir, f'{pose_name}_{camera}_map_pose.json')
            with open(pose_path, 'w') as f:
                json.dump(pose_record, f)
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
 
        rgb_path = os.path.join(self.capture_dir, f'{pose_name}_{camera}_rgb.png')
        depth_path = os.path.join(self.capture_dir, f'{pose_name}_{camera}_depth.npy')
        cv2.imwrite(rgb_path, rgb_img)
        np.save(depth_path, depth_img)
        self.get_logger().info(f'Saved capture: {rgb_path}, {depth_path}')
        return rgb_img, depth_img


    def compute_standoff_dist(self, pose: PoseStamped, standoff_dist=0.5):
        """Computes where the robot should go given a pose.
        
        Computes the x, y position to send the base of the robot to
        given the map frame pose."""

        # Grab the robots current pose in the map frame
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().warn(f'Could not get robot pose in map frame: {e}')
            return None


        rx =  robot_tf.transform.translation.x
        ry =  robot_tf.transform.translation.y
        tx =  pose.pose.position.x
        ty =  pose.pose.position.y
        
        # Get dist from robot to target
        dx = tx-rx
        dy = ty-ry
        dist = sqrt((dx)**2 + (dy)**2)
        if dist < 1e-6:
            self.get_logger().warn('Robot is already at target position')
            return None

        ux, uy = dx/dist, dy/dist # Unit vectors in x and y
        yaw = atan2(dy, dx)
        standoff_pose = PoseStamped()
        standoff_pose.header.frame_id = 'map'
        standoff_pose.header.stamp = self.get_clock().now().to_msg()
        standoff_pose.pose.position.x = tx - ux * standoff_dist # TODO maybe just use standoff dist
        standoff_pose.pose.position.y = ty - uy * standoff_dist
        standoff_pose.pose.orientation.z = sin(yaw / 2.0)
        standoff_pose.pose.orientation.w = cos(yaw / 2.0)

        return standoff_pose

    def command_pose_d405(self, target_pose: PoseStamped, standoff_dist=0.5):
        """Command the bot to go to a certain pose and take a photo with the d405."""

        # compute standoff distance
        standoff_pose = self.compute_standoff_dist(target_pose, standoff_dist)
        if not standoff_pose:
            self.get_logger().warn('Could not compute standoff pose, returning...')
            return False

        # Switch to navigation mode
        if self.stretch_mode != 'navigation':
            self.switch_mode(self.nav_mode_client, 'navigation')

        self.navigator.goToPose(standoff_pose)
        while not self.navigator.isTaskComplete():
            time.sleep(0.2)
        result = self.navigator.getResult()
        if result != TaskResult.SUCCEEDED:
            self.get_logger().warn('Could not reach given standoff pose')
            return False

        time.sleep(0.5)
        camera_response = Camera.Response()
        camera_request = Camera.Request()

        self.move_camera(camera_request, camera_response, 'd405', target_pose)

        return camera_response.success

    def compute_d435_pan_tilt(self, target_pose):
        """Transform target pose from map frame into the d435 cameras
        
        mount frame, then compute pan/tilt."""

        self.move_to_pose({'joint_head_pan': 0.0, 'joint_head_tilt': 0.0}, blocking=True)
        self.wait_until_settled(['joint_head_pan', 'joint_head_tilt'])

        target_in_cam_frame = self.transform_pose(target_pose, 'camera_link')
        x, y, z = target_in_cam_frame.position.x, target_in_cam_frame.position.y, target_in_cam_frame.position.z
        pan = atan2(y, x)
        tilt = atan2(z, sqrt(x**2 + y**2))
        return pan, tilt
    
    def compute_d405_rpy(self, target_pose: PoseStamped):
        """Same as previous function but for the gripper camera."""

        self.move_to_pose({'joint_head_pan': 0.0, 'joint_head_tilt': 0.0}, blocking=True)
        self.wait_until_settled(['joint_head_pan', 'joint_head_tilt'])

        q = target_pose.pose.orientation
        target_rot = R.from_quat([q.x, q.y, q.z, q.w])
        desired_forward_map = target_rot.apply([0.0, 0.0, 1.0])  # camera's own +Z convention

        # Rotate that direction from map frame into base_link, since the wrist
        # joints operate relative to the base, not the world
        try:
            base_tf = self.tf_buffer.lookup_transform(
                'base_link', 'map', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().warn(f'Could not get map->base_link transform: {e}')
            return None, None, None

        q_base = base_tf.transform.rotation
        map_to_base_rot = R.from_quat([q_base.x, q_base.y, q_base.z, q_base.w])
        desired_forward_base = map_to_base_rot.apply(desired_forward_map)

        yaw, pitch, residual = self.solve_wrist_for_direction(
            desired_forward_base,
            self.joint_limits['wrist_yaw'],
            self.joint_limits['wrist_pitch']
        )


        if residual > 0.05:  # threshold — tune based on testing; large residual means unreachable
            self.get_logger().warn(
                f'Could not closely match desired camera orientation (residual={residual:.4f}); '
                f'target direction may require base rotation or be otherwise unreachable'
            )

        return pitch, yaw, residual
    
    def transform_pose(self, pose_stamped, target_frame, timeout_sec = 2.0):
        """Transform a PoseStamped into a given frame."""

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )
            transformed = do_transform_pose(pose_stamped.pose, transform)
            return transformed
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')
            return None

    # --- Control functions ---

    def compute_path(self, start_id, goal_id, use_start=True, timeout_sec=60.0):
        """Command the robot to move between two nodes defined in the route geojson file."""

        if not self.route_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('compute_and_track_route action server not available')
            return False
        
        goal_msg = ComputeRoute.Goal()
        goal_msg.use_poses = False
        goal_msg.use_start = use_start
        goal_msg.start_id = start_id
        goal_msg.goal_id = goal_id

        self.get_logger().info(f'Requesting route between {start_id} and {goal_id}')
        goal_future = self.route_client.send_goal_async(
            goal=goal_msg
        )

        start = self.get_clock().now()
        while not goal_future.done():
            time.sleep(0.05)
            if (self.get_clock().now() - start).nanoseconds / 1e9 > timeout_sec:
                self.get_logger().warn(f'Route goal send timed out ({start_id}->{goal_id})')
                return False
            
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f'Route goal rejected ({start_id}->{goal_id})')
            return False
        
        result_future = goal_handle.get_result_async()
        start = self.get_clock().now()
        while not result_future.done():
            time.sleep(0.05)
            if (self.get_clock().now() - start).nanoseconds / 1e9 > timeout_sec:
                self.get_logger().warn(f'Route execution timed out ({start_id}->{goal_id})')
                return False

        result_wrapper = result_future.result()
        status = result_wrapper.status
        result = result_wrapper.result

        self.get_logger().info(f'compute_route finished with status: {status}')
        self.get_logger().info(f'Path has {len(result.path.poses)} poses')

        if len(result.path.poses) == 0:
            self.get_logger().error('Computed path is empty!')
            return None

        return result.path
    
    def route_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'Route progress: last_node={fb.last_node_id}, next_node={fb.next_node_id}'
        )

    def path_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'FollowPath feedback: distance_to_goal={fb.distance_to_goal:.3f}, '
            f'speed={fb.speed:.3f}'
        )

    def follow_route(self, nodes, loops=1):
        """Follow a route sequence. """

        self.switch_mode(self.nav_mode_client, 'navigation')

        first_leg = True
        for _ in range(loops):
            for start_id, goal_id in zip(nodes[:-1], nodes[1:]):
                use_start = not first_leg
                path = self.compute_path(start_id, goal_id, use_start=use_start)
                first_leg = False
                if path is None:
                    self.get_logger().warn(f'Failed to find a valid path')
                    return
                success = self.follow_path(path)
                if success:
                    self.get_logger().info('SUCCESS: robot followed the path!')
                else:
                    self.get_logger().error('FollowPath did not succeed')
                
    def follow_path(self, path, timeout_sec=60.0):
        """Commands the robot to follow a Path object. """

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = ''
        goal_msg.goal_checker_id = ''

        self.get_logger().info(f'Sending goal with {len(path.poses)} poses')
        goal_future = self.path_client.send_goal_async(
            goal_msg, feedback_callback=self.path_feedback_cb
        )
        start = self.get_clock().now()
        while not goal_future.done():
            time.sleep(0.05)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().error('Timed out sending follow_path goal')
                return False
            
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('follow_path goal was rejected')
            return False
        
        result_future = goal_handle.get_result_async()
        start = self.get_clock().now()
        while not result_future.done():
            time.sleep(0.05)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().error('Timed out waiting for follow_path result')
                return False

        result_wrapper = result_future.result()
        self.get_logger().info(f'follow_path finished with status: {result_wrapper.status}')
        return result_wrapper.status == 4  # GoalStatus.STATUS_SUCCEEDED
        
    def follow_waypoints(self):
        """Sends the hello robot to a given list of points."""

        # # switch to navigation mode 
        # self.switch_mode(self.nav_mode_client, 'navigation')

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.orientation.w = 1.0

        # move through all poses in the waypoints array
        for wp_idx, (x, y, yaw) in enumerate(self.waypoints):
            self.switch_mode(self.nav_mode_client, 'navigation')
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = sin(yaw / 2.0)
            pose.pose.orientation.w = cos(yaw / 2.0)
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
                self.wait_until_settled(['joint_head_pan', 'joint_head_tilt'], vel_thresh=.005)
                time.sleep(0.5)
                pose_name = f'wp{wp_idx}_d435_pose{idx}'
                self.capture_frame(pose_name, camera='d435')

            self.get_logger().info('Completed d435 camera poses')
            # capture smaller images with the d405

            # Send lift to various positions
            # make an if statement here to determine if d405 is used
            # for idx, cp in enumerate(self.d405_lifts):
            #     self.camera_capture_pose(cp, camera='d405', joint='joint_lift')
            #     self.wait_until_settled(['joint_lift'])
            #     pose_name = f'wp{wp_idx}_d405_pose{idx}'
            #     self.capture_frame(pose_name, camera='d405')

            #     # Within each lift position, yaw and pitch the gripper
            #     for yaw_idx, yaw_pose in enumerate(self.d405_yaws):
            #         self.camera_capture_pose(yaw_pose, camera='d405', joint='joint_wrist_yaw')
            #         self.wait_until_settled((['joint_wrist_yaw']))
            #         pose_name = f'wp{wp_idx}_d405_pose{idx}_yaw_{yaw_idx}'
            #         self.capture_frame(pose_name, camera='d405')

            #         # Within each yaw, pitch the gripper 
            #         for p_idx, p_pose, in enumerate(self.d405_tilts):
            #             self.camera_capture_pose(p_pose, camera='d405', joint='joint_wrist_pitch')
            #             self.wait_until_settled((['joint_wrist_pitch']))
            #             pose_name = f'wp{wp_idx}_d405_pose{idx}_yaw_{yaw_idx}_pitch_{p_idx}'
            #             self.capture_frame(pose_name, camera='d405')

        self.get_logger().info('Completed image captures!')

    def camera_capture_pose(self, pose, camera='d435', joint='joint_lift'):
        """Move the camera on the bot to a given pose."""

        # grab tilt and pan from the list
        if camera=='d435':
            pan = pose[0]
            tilt = pose[1]

            request = Camera.Request()
            response = Camera.Response()
            request.angles = [pan, tilt]
            request.joints = ['joint_head_pan', 'joint_head_tilt']
        elif camera=='d405':
            yaw = pose
            request = Camera.Request()
            response = Camera.Response()
            request.angles = [yaw]
            request.joints = ['joint_wrist_yaw']
        self.move_camera(request=request, response=response, camera=camera)

    def move_camera(self, request, response, camera='d435', target_pose=None):
        """"Adjust the camera position."""

        # before moving joints, switch to position mode
        if self.stretch_mode != 'position':
            self.switch_mode(self.pos_mode_client, 'position')

        if camera == 'd405' and target_pose:
            target_in_base = self.transform_pose(target_pose, 'base_link')
            if not target_in_base:
                self.get_logger().warn('Could not transform target pose to base_link; aborting move')
                response.success = False
                return response

            LIFT_ZERO_HEIGHT = 0.312
            lift = target_in_base.position.z - LIFT_ZERO_HEIGHT
            lift = max(0.0, min(lift, 1.1))
            self.move_to_pose({'joint_lift': lift}, blocking=True)
            self.wait_until_settled(['joint_lift'])
            self.get_logger().info(f'Moved lift to {lift} before solving orientation')

            # Solve jointly for base rotation + wrist yaw/pitch
            base_yaw_delta, wrist_yaw, wrist_pitch, residual = self.solve_base_and_wrist(target_pose)
            if base_yaw_delta is None:
                response.success = False
                return response

            self.get_logger().info(
                f'Solved: base_yaw_delta={base_yaw_delta:.3f}, wrist_yaw={wrist_yaw:.3f}, '
                f'wrist_pitch={wrist_pitch:.3f}, residual={residual:.4f}'
            )

            if residual > 0.05:
                self.get_logger().warn(
                    f'High residual ({residual:.4f}) -- target may not be fully reachable '
                    f'even with base rotation'
                )

            # Execute: one base rotation, then one wrist move
            if abs(base_yaw_delta) > 0.02:  # skip trivial rotations
                self.rotate_base(base_yaw_delta)

            self.move_to_pose(
                {'joint_wrist_yaw': wrist_yaw, 'joint_wrist_pitch': wrist_pitch},
                blocking=True
            )
            self.get_logger().info(f'Moved wrist camera to yaw={wrist_yaw:.3f}, pitch={wrist_pitch:.3f}')
            response.success = bool(residual <= 0.05)
            return response

        joint_positions = [float(x) for x in request.angles]
        joint_names = [string for string in request.joints]
        print(f'Joint names are : {joint_names} and joint positions are {joint_positions}')

        if camera == 'd435':
            for idx, joint in enumerate(joint_names):
                if joint == 'joint_head_pan':
                    pan = joint_positions[idx]
                if joint == 'joint_head_tilt':
                    tilt = joint_positions[idx]
            if 'joint_head_pan' not in joint_names:
                pan = 0.0
            if 'joint_head_tilt' not in joint_names:
                tilt = 0.0
            self.move_to_pose(
                {'joint_head_pan': pan, 'joint_head_tilt': tilt},
                blocking=True
            )
            self.get_logger().info(f'Moving camera to {pan}, {tilt}')

        if camera == 'd405':
            for idx, joint in enumerate(joint_names):
                if joint == 'joint_lift':
                    lift = joint_positions[idx]
                if joint == 'joint_wrist_yaw':
                    yaw = joint_positions[idx]
                if joint == 'joint_wrist_pitch':
                    pitch = joint_positions[idx]
            if 'joint_wrist_yaw' not in joint_names:
                yaw = 0.0
            if 'joint_wrist_pitch' not in joint_names:
                pitch = 0.0

            if 'joint_wrist_yaw' in joint_names or 'joint_wrist_pitch' in joint_names:
                self.move_to_pose(
                    {'joint_wrist_yaw': yaw, 'joint_wrist_pitch': pitch},
                    blocking=True
                )
                self.get_logger().info(f'Moving wrist camera to {yaw}, {pitch}')

            if 'joint_lift' in joint_names:
                self.move_to_pose(
                    {'joint_lift': lift},
                    blocking=True
                )
                self.get_logger().info(f'Moving lift to {lift}')

        return response
        
    
    def move_d405_lift(self, request, response):
        """Adjust the lift height for the gripper mounted camera."""

        # before moving joints, switch to position mode
        if self.stretch_mode != 'position':
            self.switch_mode(self.pos_mode_client, 'position')

        joint_positions = [float(x) for x in request.angles]
        lift = joint_positions[0]
        self.move_to_pose(
            {'jount_lift': lift},
            blocking=True
        )
        self.get_logger().info(f'Moving lift to {lift}')
        return response

    def move_d405_pitch_yaw(self, request, response):
        """Adjust the pitch and yaw of the gripper mount."""

        # before moving joints, switch to position mode
        if self.stretch_mode != 'position':
            self.switch_mode(self.pos_mode_client, 'position')

        joint_positions = [float(x) for x in request.angles]
        pitch = joint_positions[0]
        yaw = joint_positions[1]
        self.move_to_pose(
            {'joint_wrist_yaw': yaw, 'joint_wrist_pitch': pitch},
            blocking=True
        )
        self.get_logger().info(f'Moving pitch to {pitch}, moving yaw to {yaw}')
        return response

    def rotate_base(self, angle, angular_speed=0.3, tolerance=0.05, timeout_sec=15.0):
        """Rotate the base of the robot."""

        if not self.switch_mode(self.nav_mode_client, 'navigation'):
            self.get_logger().error('Could not switch to navigation mode; aborting rotation')
            return False

        try:
            start_tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().warn(f'Could not get starting yaw: {e}')
            return False

        q = start_tf.transform.rotation
        start_yaw = 2.0 * atan2(q.z, q.w)
        target_yaw = start_yaw + angle
        command = Twist()

        self.get_logger().info(f'DEBUG: entering rotate loop, start_yaw={start_yaw:.3f}, target_yaw={target_yaw:.3f}, tolerance={tolerance}')

        start = self.get_clock().now()
        loop_count = 0
        while True:
            loop_count += 1
            try:
                current_tf = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time()
                )
            except Exception as e:
                self.get_logger().warn(f'DEBUG: tf lookup failed in loop iter {loop_count}: {e}')
                time.sleep(0.05)
                continue

            q = current_tf.transform.rotation
            current_yaw = 2.0 * atan2(q.z, q.w)
            error = atan2(sin(target_yaw - current_yaw), cos(target_yaw - current_yaw))

            self.get_logger().info(f'DEBUG: iter={loop_count}, current_yaw={current_yaw:.3f}, error={error:.3f}')

            if abs(error) < tolerance:
                self.get_logger().info(f'DEBUG: breaking, error {error:.3f} < tolerance {tolerance}')
                break

            command.angular.z = max(-angular_speed, min(angular_speed, error * 1.5))
            self.get_logger().info(f'DEBUG: publishing angular.z={command.angular.z:.3f}')
            self.vel_pub.publish(command)
            time.sleep(0.05)

            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn('Base rotation timed out')
                break

        command.angular.z = 0.0
        self.vel_pub.publish(command)
        self.switch_mode(self.pos_mode_client, 'position')

        end_tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        q = end_tf.transform.rotation
        end_yaw = 2.0 * atan2(q.z, q.w)
        self.get_logger().info(f'DEBUG: end_yaw = {end_yaw:.3f} (started at {start_yaw:.3f}), total iterations={loop_count}')
        return True


def main(args=None):
    node = MoveJoints()
    node.follow_waypoints()
    node.new_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()