"""
Warning: TF_OLD_DATA ignoring data from the past for frame tool0_target at time 0.000000 according to authority default_authority
Possible reasons are listed at http://wiki.ros.org/tf/Errors%20explained
         at line 294 in ./src/buffer_core.cpp
"""
import os
import json
import time
import datetime
import random
from collections import deque
from queue import Queue, Empty
from typing import Tuple, List
import threading
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import cv2

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, JointState
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from geometry_msgs.msg import PoseStamped, TransformStamped
from moveit_msgs.action import MoveGroup

from cv_bridge import CvBridge

from workcell_interfaces.action import (
    BoolAction,
    MoveArmAction,
    PoseDetectAction,
    PickPlaceAction,
    InspectionAction,
)
from workcell_interfaces.srv import (
    PoseDetect,
    MeshGen,
    GraspFilter,
    PickPlace,
    StereoDepth,
)

from ..configs.variety_info import VarietyInfo
from arm_ur_lib import utils as ArmUtils
from arm_ur_ros2.arm_wrapper_ros2 import ArmWrapper
from as_utils import utils_geo, utils_ros, utils_pc
from as_utils.timer import RTimer

from ..configs.pickup_config import PickupConfig
from ..models.sam_data import SamData
from ..bundles.comms import (
    ArmComms,
    PerceptionComms,
    InspectionComms,
    CameraParamComms,
    TfComms,
)
from ..bundles.state import ImageSyncState, RuntimeState
from ..bundles.utils import NodeUtils
from ..workflows.pickup_workflow import run_pickup_workflow

USE_DEBUG = False

class TaskManagerNode(Node):
    """
    旧 PickupClientNode をベースに、ROS通信はここに残し、
    巨大ロジックだけを workflow / phases に外した Node。
    """

    def __init__(self, variety='mdc25'):
        super().__init__('task_manager_node')

        self.cfg = PickupConfig(variety=variety)

        # =====================================================
        # Arm
        # =====================================================
        self.arm = ArmWrapper(self)

        # =====================================================
        # Action server
        # =====================================================
        self.cbg_action = ReentrantCallbackGroup()
        self._as_user = ActionServer(
            self,
            BoolAction,
            '/user_trigger',
            execute_callback=self.user_execute_cb,
            goal_callback=self.user_goal_cb,
            cancel_callback=self.user_cancel_cb,
            callback_group=self.cbg_action
        )
        self.get_logger().info('[SUCCESS] Action server ready')

        # =====================================================
        # Action clients
        # =====================================================
        self._ac_move = ActionClient(
            self,
            MoveArmAction,
            '/ur_command/move_arm_action',
            callback_group=self.cbg_action
        )
        while not self._ac_move.wait_for_server(timeout_sec=1.0):
            print('wait for move server')
            break

        self._ac_pick = ActionClient(
            self,
            PickPlaceAction,
            '/ur_command/pickup_object_action',
            callback_group=self.cbg_action
        )
        while not self._ac_pick.wait_for_server(timeout_sec=1.0):
            print('wait for pick server')
            break

        self._ac_place = ActionClient(
            self,
            PickPlaceAction,
            '/ur_command/place_object_action',
            callback_group=self.cbg_action
        )
        while not self._ac_place.wait_for_server(timeout_sec=1.0):
            print('wait for place server')
            break

        self._ac_pose = ActionClient(
            self,
            PoseDetectAction,
            '/pose_detect_action',
            callback_group=self.cbg_action
        )
        while not self._ac_pose.wait_for_server(timeout_sec=1.0):
            print('wait for sam6d  server')

        self._ac_inspect = ActionClient(
            self,
            InspectionAction,
            '/inspection_action',
            callback_group=self.cbg_action
        )
        while not self._ac_inspect.wait_for_server(timeout_sec=1.0):
            print('wait for inspect server / abort!!')
            break

        # =====================================================
        # Subscribers（旧コードと同じ）
        # =====================================================
        self.cgb_rs = MutuallyExclusiveCallbackGroup()
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=5)

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/d405/aligned_depth_to_color/image_raw',
            self.depth_cb,
            callback_group=self.cgb_rs,
            qos_profile=qos_profile
        )
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/d405/color/image_rect_raw',
            self.rgb_cb,
            callback_group=self.cgb_rs,
            qos_profile=qos_profile
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/d405/aligned_depth_to_color/camera_info',
            self.caminfo_cb,
            callback_group=self.cgb_rs,
            qos_profile=qos_profile
        )

        # =====================================================
        # Service clients（旧コードと同じ）
        # =====================================================
        self.cbg_srv = ReentrantCallbackGroup()
        self.cli_pose_pred = self.create_client(
            PoseDetect, '/pose_detect', callback_group=self.cbg_srv
        )
        self.cli_mesh = self.create_client(
            MeshGen, '/gen_mesh', callback_group=self.cbg_srv
        )
        while not self.cli_mesh.wait_for_service(timeout_sec=1.0):
            print('wait for mesh server')

        self.cli_filter = self.create_client(
            GraspFilter, '/filter_grasps', callback_group=self.cbg_srv
        )
        self.cli_ur_pickup = self.create_client(
            PickPlace, '/ur_command/pickup_object', callback_group=self.cbg_srv
        )

        target_node = '/camera/d405'
        self.cli_camparam_get = self.create_client(
            GetParameters, f'{target_node}/get_parameters'
        )
        self.cli_camparam_set = self.create_client(
            SetParameters, f'{target_node}/set_parameters'
        )

        self.resizer = utils_ros.ImageResizerUtil()

        # =====================================================
        # TF
        # =====================================================
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30)) # default: 10s
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # =====================================================
        # variety
        # =====================================================
        self.variety = variety
        self.vi = VarietyInfo(variety)

        # =====================================================
        # state (用途別に分割)
        # =====================================================
        self.image_state = ImageSyncState()
        self.runtime_state = RuntimeState()

        self.get_logger().info('[SUCCESS] all ready')

        # =====================================================
        # (debug) save inspection results
        # =====================================================
        self.inspection_results = Queue(maxsize=10)
        self.inspection_save_flag = threading.Event()
        self.thread = threading.Thread(
            target=self.auto_save,
            daemon=True,
        )
        self.thread.start()

    # =====================================================
    # (debug) save inspection results
    # =====================================================
    def auto_save(self):
        def stamp_to_yymmdd_hhmmss(stamp, use_utc=True):
            dt = datetime.datetime.fromtimestamp(
                stamp.sec,
                tz=datetime.timezone(datetime.timedelta(hours=9)) if use_utc else None
            )
            return dt.strftime("%y%m%d_%H%M%S")

        save_dir = Path("~/data/inspection_capture").expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        bridge = CvBridge()

        while not self.inspection_save_flag.is_set() or not self.inspection_results.empty():
            try:
                results = self.inspection_results.get(timeout=1.0)
            except Empty:
                continue

            date_str = None
            for ang, ret in results.result_images.items():
                if date_str is None:
                    date_str = stamp_to_yymmdd_hhmmss(ret["rgb"].header.stamp)

                rgb_cv = bridge.imgmsg_to_cv2(ret["rgb"], "bgr8")
                cv2.imwrite(str(save_dir / f"{date_str}_{int(ang)}_rgb.png"), rgb_cv)

            self.inspection_results.task_done()
            self.get_logger().info('Captured images saved!')

    def end_auto_save(self):
        self.inspection_save_flag.set()
        self.thread.join()
        print("SAVE THREAD JOINED")


    # =====================================================
    # build bundles
    # =====================================================
    def build_arm_comms(self):
        return ArmComms(
            ac_move=self._ac_move,
            ac_pick=self._ac_pick,
            ac_place=self._ac_place,
        )

    def build_perception_comms(self):
        return PerceptionComms(
            ac_pose=self._ac_pose,
            cli_pose_pred=self.cli_pose_pred,
            cli_mesh=self.cli_mesh,
            cli_filter=self.cli_filter,
            cli_ur_pickup=self.cli_ur_pickup,
        )

    def build_inspection_comms(self):
        return InspectionComms(
            ac_inspect=self._ac_inspect
        )

    def build_camera_param_comms(self):
        return CameraParamComms(
            cli_camparam_get=self.cli_camparam_get,
            cli_camparam_set=self.cli_camparam_set,
        )

    def build_tf_comms(self):
        return TfComms(
            tf_buffer=self.tf_buffer,
            tf_broadcaster=self.tf_broadcaster,
        )

    def build_utils(self):
        return NodeUtils(
            move_arm=self.move_arm,
            wait_images_and_tf=self.wait_images_and_tf,
            run_sam_request=self.run_sam_request,
            run_sam_result=self.run_sam_result,
            get_obj_pose=self.get_obj_pose,
            try_pickup=self.try_pickup,
            try_pick_candidates=self.try_pick_candidates,
            try_place=self.try_place,
            capture_images=self.capture_images,
            run_inspection=self.run_inspection, # TOD: 未定義
            result_inspection=self.result_inspection,
            lookup_tf=self.lookup_tf,
            get_camparam=self.get_camparam,
            set_camparam=self.set_camparam,
            wait_action_server=self.wait_action_server,
            yield_once=self._yield_once,
            remove_close_obj=self.remove_close_obj,
            create_grasp_poses=self.create_grasp_poses,
            filter_by_angle=self.filter_by_angle,
            trans_cam_to_world=self._trans_cam_to_world,
            select_target_obj=self.select_target_obj,
            estimate_obj_pose=self.estimate_obj_pose,
            gen_grasp_pose=self.gen_grasp_pose,
        )

    # =====================================================
    # Action hooks
    # =====================================================
    def user_goal_cb(self, goal_request):
        self.get_logger().info('Got user request')
        return GoalResponse.ACCEPT

    def user_cancel_cb(self, goal_handle):
        self.get_logger().info('Got cancel request')
        return CancelResponse.ACCEPT

    # =====================================================
    # Execute
    # =====================================================
    async def user_execute_cb(self, goal_handle):
        try:
            self.get_cam_tf() # カメラとtool0の相対位置を取得してcfgに保存
            result_wf = await run_pickup_workflow(
                arm_comms=self.build_arm_comms(),
                perception_comms=self.build_perception_comms(),
                inspection_comms=self.build_inspection_comms(),
                tf_comms=self.build_tf_comms(),
                image_state=self.image_state,
                runtime_state=self.runtime_state,
                utils=self.build_utils(),
                cfg=self.cfg,
                vi=self.vi,
                inspection_results=self.inspection_results,
                logger=self.get_logger()
            )

            result = BoolAction.Result()
            result.success = result_wf.success
            result.message = result_wf.message

            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()

            return result

        except Exception as e:
            return self._abort(goal_handle, f'[exception] {e}')
        finally:
            self.get_logger().info('-------------------------------------')

    # =====================================================
    # callbacks
    # =====================================================
    def depth_cb(self, msg):
        st = self.image_state
        if (st.action_sub_flag or st.collision_sub_flag) \
                and st.started_time is not None \
                and utils_ros.stamp_to_sec(msg.header.stamp) >= st.started_time:
            st.depth_buffer.append(msg)

    def rgb_cb(self, msg):
        st = self.image_state
        if (st.action_sub_flag or st.collision_sub_flag) \
                and st.started_time is not None \
                and utils_ros.stamp_to_sec(msg.header.stamp) >= st.started_time:
            st.rgb_buffer.append(msg)

    def caminfo_cb(self, msg):
        st = self.image_state
        if (st.action_sub_flag or st.collision_sub_flag) \
                and st.started_time is not None \
                and utils_ros.stamp_to_sec(msg.header.stamp) >= st.started_time:
            st.caminfo_buffer.append(msg)

    # =====================================================
    # utilities / images
    # =====================================================
    async def wait_images_and_tf(self, timeout_sec=6.0, use_tf=True, use_resize=True):
        st = self.image_state
        st.depth_buffer.clear()
        st.rgb_buffer.clear()
        st.caminfo_buffer.clear()

        st.started_time = utils_ros.stamp_to_sec(self.get_clock().now().to_msg())
        t0 = time.time()

        while time.time() - t0 < timeout_sec:
            ok, depth_msg, rgb_msg, caminfo_msg, stamp_ns = self.try_sync_once()
            if not ok:
                await self._yield_once()
                continue

            if use_tf:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        'world',
                        depth_msg.header.frame_id,
                        rclpy.time.Time(nanoseconds=stamp_ns),
                        timeout=Duration(seconds=5.0)
                    )

                    st.depth_buffer.clear()
                    st.rgb_buffer.clear()
                    st.caminfo_buffer.clear()

                    if use_resize:
                        rgb_msg_, depth_msg_, caminfo_msg_ = self.resizer.resize_all(
                            rgb_msg,
                            depth_msg,
                            caminfo_msg
                        )
                        return True, depth_msg_, rgb_msg_, caminfo_msg_, stamp_ns, trans
                    else:
                        return True, depth_msg, rgb_msg, caminfo_msg, stamp_ns, trans

                except Exception as e:
                    self.get_logger().warn(f'faile to get tf: {e}')
                    await self._yield_once()
                    continue
            else:
                if use_resize:
                    rgb_msg_, depth_msg_, caminfo_msg_ = self.resizer.resize_all(
                        rgb_msg,
                        depth_msg,
                        caminfo_msg
                    )
                    return True, depth_msg_, rgb_msg_, caminfo_msg_, stamp_ns, None
                else:
                    return True, depth_msg, rgb_msg, caminfo_msg, stamp_ns, None

        return False, None, None, None, 0, None

    def try_sync_once(self) -> Tuple[bool, Image, Image, CameraInfo, int]:
        st = self.image_state

        def s2n(stamp):
            return stamp.sec * 10 ** 9 + stamp.nanosec

        if not (st.depth_buffer and st.rgb_buffer and st.caminfo_buffer):
            return False, None, None, None, 0

        depth_stamps = {s2n(msg.header.stamp): msg for msg in st.depth_buffer}
        rgb_stamps = {s2n(msg.header.stamp): msg for msg in st.rgb_buffer}
        caminfo_stamps = {s2n(msg.header.stamp): msg for msg in st.caminfo_buffer}
        common = set(depth_stamps) & set(rgb_stamps) & set(caminfo_stamps)

        if not common:
            return False, None, None, None, 0

        latest = max(common)
        return True, depth_stamps[latest], rgb_stamps[latest], caminfo_stamps[latest], latest

    # =====================================================
    # utilities / arm
    # =====================================================
    async def move_arm(self, pose=None, va=None, start_js=None, js=None, n_try=3, rough_mode=False):
        # set move condition
        goal = MoveGroup.Goal()
        va = va if va is not None else [0.1, 0.1]
        goal.request.max_velocity_scaling_factor = va[0]
        goal.request.max_acceleration_scaling_factor = va[1]

        tolerance = 0.020 if rough_mode else 0.001
        if not isinstance(start_js, JointState):
            _start_js = JointState()
            _start_js.name = self.arm.joint_names
            _start_js.position = start_js
        else:
            _start_js = start_js

        # pose -> joint_position
        if pose is not None:
            if isinstance(pose, PoseStamped):
                pose = pose.pose
            success, joint_position = await self.arm.pose_to_js(pose, _start_js)
            if not success:
                return False, '[FAIL]pose to ik'
        elif js is not None:
            joint_position = js
        else:
            return False, '[FAIL]pose and js are invalid'

        # joint_position -> goal
        success = self.arm.build_joint_goal_constraints(
            goal=goal,
            joint_positions=joint_position,
            joint_names=None,
            tolerance=tolerance,
        )
        if not success:
            return False, '[FAIL]add constraint'

        # execute
        success, msg = await self.arm.move_to_joint_positions_async(goal)
        if not success:
            return False, f'[FAIL]execute]{msg}'
        else:
            return True, '[SUCCESS]'


#    async def _move_arm(self, pose=None, va=None, start_js=None, js=None, n_try=3, rough_mode=False):
#        if not self.wait_action_server(self._ac_move, '/ur_command/move_arm_action', total_timeout=10.0):
#            return (False, '[fail] /ur_command/move_arm_action server not ready')
#
#        g = MoveArmAction.Goal()
#        if pose:
#            g.pose = pose
#        elif js:
#            g.joint_positions = js if isinstance(js, List) else js.tolist()
#        else:
#            return (False, '[fail] pose and js is empty')
#
#        g.va = va if va is not None else [0.1, 0.1]
#        g.rough_mode = rough_mode
#        if start_js is not None:
#            g.start_js = start_js if isinstance(start_js, List) else start_js.tolist()
#
#        for idx in range(n_try):
#            gh_future = self._ac_move.send_goal_async(g)
#            gh = await gh_future
#            if not gh.accepted:
#                self.get_logger().warn(f'[fail] move goal rejected at {idx+1}/{n_try}')
#                continue
#            res = await gh.get_result_async()
#            if res is None or not getattr(res.result, 'success', False):
#                self.get_logger().warn(f'[FAIL] move execute at {idx+1}/{n_try}')
#                continue
#            else:
#                return (True,)
#
#        return (False, '[fail] move arm')

    # =====================================================
    # utilities / SAM
    # =====================================================
    def run_sam_request(self, depth_msg, rgb_msg, caminfo_dict, use_crop=None):
        if use_crop is not None:
            depth_header = depth_msg.header
            rgb_header = rgb_msg.header
            depth_cv = CvBridge().imgmsg_to_cv2(depth_msg, 'passthrough')
            rgb_cv = CvBridge().imgmsg_to_cv2(rgb_msg, 'bgr8')
            h, w = rgb_cv.shape[:2]
            mask = np.zeros([h, w, 3])

            if isinstance(use_crop, float):
                h_win = int(h * use_crop)
                w_win = int(w * use_crop)
                mask[h_win:-h_win, w_win:-w_win, :] = 1
            elif isinstance(use_crop, str) and use_crop == 'ul':
                mask[:h // 2, :w // 2] = 1
            elif isinstance(use_crop, str) and use_crop == 'ur':
                mask[:h // 2, :w // 2] = 1

            depth_cv = (depth_cv * mask[:, :, 0]).astype('uint16')
            depth_msg = CvBridge().cv2_to_imgmsg(depth_cv, 'passthrough')
            depth_msg.header = depth_header

            rgb_cv = (rgb_cv * mask).astype('uint8')
            rgb_msg = CvBridge().cv2_to_imgmsg(rgb_cv, 'bgr8')
            rgb_msg.header = rgb_header

        if self._ac_pose is not None and self._ac_pose.wait_for_server(timeout_sec=0.5):
            goal = PoseDetectAction.Goal()
            goal.depth_image = depth_msg
            goal.rgb_image = rgb_msg
            goal.camera_params_json = json.dumps(caminfo_dict)
            gh_future = self._ac_pose.send_goal_async(goal)
            return gh_future

    async def run_sam_result(self, gh_future) -> List[PoseStamped]:
        poses = []
        gh = await gh_future
        if gh.accepted:
            res_future = gh.get_result_async()
            res = await res_future
            if res and res.result:
                for p in res.result.poses:
                    poses.append(p)
                return poses
        else:
            self.get_logger().warn('[WARN] pose detect goal rejected, fallback to service')
        return poses

    async def get_obj_pose(self, cam_pose, use_grasp_pose=True, target_frame_id=None, use_cc=True):
        return
        va = list(self.cfg.va_get_obj_pose)

        with RTimer(self, 'MOVE TO WATCH', False):
            success, *msg = await self.move_arm(cam_pose, va, n_try=3, rough_mode=True)
            if not success:
                return (False, '[FAIL] move arm')

        with RTimer(self, '[SAM]CAP', False):
            self.image_state.action_sub_flag = True
            ok, depth_msg, rgb_msg, caminfo_msg, stamp_ns, trans = \
                await self.wait_images_and_tf(timeout_sec=6.0)
            self.image_state.action_sub_flag = False

            if not ok:
                return (False, '[fail] no synced images + tf')

            caminfo_dict = {"cam_K": list(caminfo_msg.k), "depth_scale": 1.0}
            sd = SamData(
                rgb_msg=rgb_msg,
                depth_msg=depth_msg,
                caminfo_msg=caminfo_msg,
                caminfo_dict=caminfo_dict,
                trans=trans
            )

        if use_grasp_pose and use_cc:
            with RTimer(self, 'GEN MESH REQUEST', False):
                mesh_req = MeshGen.Request()
                mesh_req.depth_image = sd.depth_msg
                mesh_req.camera_params_json = json.dumps(sd.caminfo_dict)
                if not self.cli_mesh.wait_for_service(timeout_sec=5.0):
                    return (False, '[fail] gen_mesh service not ready')
                mesh_future = self.cli_mesh.call_async(mesh_req)

        with RTimer(self, 'SAM', False):
            gh_fut = self.run_sam_request(sd.depth_msg, sd.rgb_msg, sd.caminfo_dict)
            obj_poses = await self.run_sam_result(gh_fut)

            if obj_poses is None or len(obj_poses) == 0:
                self.get_logger().warn('[WARN] no obj')
                return (False, 'no obj poses')

            self.get_logger().info(f'[SUCCESS] receive {len(obj_poses)} obj poses')
            obj_poses = self.remove_close_obj(obj_poses, 0.15)
            obj_poses_world = [self._trans_cam_to_world(pose, sd.trans) for pose in obj_poses]
            sd.obj_poses_w = obj_poses_world

            if target_frame_id:
                trans_target = self.lookup_tf(frame_id='world', child=target_frame_id, time=None, timeout_sec=10)
                T_w_target = utils_geo.transform_to_hmat(trans_target.transform)
            else:
                T_w_target = utils_geo.pose_to_hmat(cam_pose.pose)

            obj_pose_world = self.select_target_obj(obj_poses_world, target=T_w_target[:3, 3], threshold=0.15)

            if not obj_pose_world:
                self.get_logger().warn('[FAIL] select obj pose')
                return (False, '[FAIL] select obj pose')

            sd.obj_pose_w = obj_pose_world

        if not use_grasp_pose:
            return (True, obj_pose_world, sd)

        if use_cc:
            with RTimer(self, 'WAIT GEN-MESH', False):
                mesh_res = await mesh_future
                if mesh_res is None or not mesh_res.success:
                    self.get_logger().warn('[FAIL] mesh gen may not be ready')
                    return (False, '[FAIL] mesh gen may not be ready', obj_pose_world)

        with RTimer(self, 'Gen grasp pose candidates', False):
            poses, widths, ids = self.create_grasp_poses([
                self._trans_cam_to_world(obj_pose_world, sd.trans, inv=True),
            ])
            poses_f, widths_f, ids_f = self.filter_by_angle(
                poses, widths, ids, sd.trans, thresh_ang=120
            )
            if not use_cc:
                remains = []
                for p, w, i in zip(poses_f, widths_f, ids_f):
                    remains.append((self._trans_cam_to_world(p, sd.trans), w, i))
                sd.remains = remains

        if use_cc:
            with RTimer(self, 'Collision Check', False):
                cc_req = GraspFilter.Request()
                cc_req.poses = poses_f
                cc_req.gripper_widths = widths_f
                if not self.cli_filter.wait_for_service(timeout_sec=5.0):
                    return (False, '[fail] filter_grasps service not ready', obj_pose_world)

                cc_future = self.cli_filter.call_async(cc_req)
                cc_res = await cc_future
                if cc_res is None:
                    return (False, '[fail] cc srv', obj_pose_world)

                remains = []
                for p, w, i, jd in zip(poses_f, widths_f, ids_f, cc_res.judges):
                    if not jd:
                        continue
                    remains.append((self._trans_cam_to_world(p, sd.trans), w, i))

        if len(remains) == 0:
            return (False, '[fail]no grs poses', obj_pose_world)
        else:
            self.get_logger().info(f'[SUCCESS] cc return {len(remains)} poses')
            sd.remains = remains
            return (True, obj_pose_world, remains)




    # get_obj_poseを改修 / move_armは外してcapture&sam専用に / meshのrequest初音に実行 / CC不要なときは実行しないように改修
    async def estimate_obj_pose(self, target_frame_id):
        with RTimer(self, '[SAM]CAP', False):
            self.image_state.action_sub_flag = True
            ok, depth_msg, rgb_msg, caminfo_msg, stamp_ns, trans = \
                await self.wait_images_and_tf(timeout_sec=6.0)
            self.image_state.action_sub_flag = False

            if not ok:
                return (False, '[fail] no synced images + tf')

            caminfo_dict = {"cam_K": list(caminfo_msg.k), "depth_scale": 1.0}
            sd = SamData(
                rgb_msg=rgb_msg,
                depth_msg=depth_msg,
                caminfo_msg=caminfo_msg,
                caminfo_dict=caminfo_dict,
                trans=trans
            )

        with RTimer(self, 'GEN MESH REQUEST', False): # 不要なケースも想定されるが負荷が小さいので実行
            mesh_req = MeshGen.Request()
            mesh_req.depth_image = sd.depth_msg
            mesh_req.camera_params_json = json.dumps(sd.caminfo_dict)
            if not self.cli_mesh.wait_for_service(timeout_sec=5.0):
                return (False, '[fail] gen_mesh service not ready', sd)
            mesh_future = self.cli_mesh.call_async(mesh_req)

        with RTimer(self, 'SAM', False):
            gh_fut = self.run_sam_request(sd.depth_msg, sd.rgb_msg, sd.caminfo_dict)
            obj_poses = await self.run_sam_result(gh_fut)

            if obj_poses is None or len(obj_poses) == 0:
                self.get_logger().warn('[WARN] no obj')
                return (False, '[FAIL] SAM not detected no obj', sd)

            self.get_logger().info(f'[SUCCESS] receive {len(obj_poses)} obj poses')
            obj_poses = self.remove_close_obj(obj_poses, 0.15)
            obj_poses_world = [self._trans_cam_to_world(pose, sd.trans) for pose in obj_poses]
            sd.obj_poses_w = obj_poses_world

            # samで検出された物体分から、target_grame_idに最も違いものを選定
        with RTimer(self, 'SELECT OBJ POSE', False):
            trans_target = self.lookup_tf(frame_id='world', child=target_frame_id, time=None, timeout_sec=10)
            T_w_target = utils_geo.transform_to_hmat(trans_target.transform)

            obj_pose_world = self.select_target_obj(obj_poses_world, target=T_w_target[:3, 3], threshold=0.15)

            if not obj_pose_world:
                self.get_logger().warn('[FAIL] select obj pose')
                return (False, '[FAIL] select obj pose', sd)
            else:
                sd.obj_pose_w = obj_pose_world

        with RTimer(self, 'WAIT GEN-MESH', False):
            mesh_res = await mesh_future
            if mesh_res is None or not mesh_res.success:
                self.get_logger().warn('[FAIL] mesh gen may not be ready')
                return (False, '[FAIL] mesh gen may not be ready', sd)
            else:
                return (True, '[SUCCESS] sam and gen mesh', sd)


    async def gen_grasp_pose(self, sd, skip_cc=False):
        # カメラ座標に戻す
        poses, widths, ids = self.create_grasp_poses([
            self._trans_cam_to_world(sd.obj_pose_w, sd.trans, inv=True),
        ])

        # 把持姿勢を角度でフィルタリング
        poses_f, widths_f, ids_f = self.filter_by_angle(
            poses, widths, ids, sd.trans, thresh_ang=120
        )

        if not skip_cc: # 干渉チェック
            cc_req = GraspFilter.Request()
            cc_req.poses = poses_f
            cc_req.gripper_widths = widths_f
            if not self.cli_filter.wait_for_service(timeout_sec=5.0):
                return (False, '[fail] filter_grasps service not ready', sd)

            # 非同期実行
            cc_future = self.cli_filter.call_async(cc_req)
            cc_res = await cc_future # ここが遅い
            if cc_res is None:
                return (False, '[fail] cc srv', sd)

            # okの姿勢を選別
            remains = []
            for p, w, i, jd in zip(poses_f, widths_f, ids_f, cc_res.judges):
                if not jd:
                    continue
                remains.append((self._trans_cam_to_world(p, sd.trans), w, i))

        else: # すべての姿勢を返す
            remains = []
            for p, w, i  in zip(poses_f, widths_f, ids_f):
                remains.append((self._trans_cam_to_world(p, sd.trans), w, i))

        if len(remains) == 0:
            return (False, '[fail]no grs poses', sd)
        else:
            self.get_logger().info(f'[SUCCESS] gen_grasp_pose: {len(remains)} poses')
            sd.remains = remains
            return (True, '[SUCCESS] gen_grasp_pose', sd)


    # =====================================================
    # utilities / grasp / pick / place
    # =====================================================
    def remove_close_obj(self, poses, thresh=0.2):
        poses_result = []
        for pose in poses:
            T = utils_geo.pose_to_hmat(pose.pose)
            xyz = T[:3, 3]
            if np.linalg.norm(xyz) >= thresh:
                poses_result.append(pose)
        return poses_result

    def create_grasp_poses(self, obj_poses):
        pose_no = 0
        result_poses = []
        result_widths = []
        result_ids = []

        pose_dict = self.vi.get_grasp_pose_as_hmat_dict(stable_first=True, stable_only=False)
        for idx, T_obj_grs in pose_dict.items():
            pose_no = int(idx.replace('pose', ''))
            gripper_width = self.vi.get_gripper_width(f'pose{pose_no}')

            for obj_no, posestamped_world_obj in enumerate(obj_poses):
                T_world_obj = utils_geo.pose_to_hmat(posestamped_world_obj.pose)
                T_world_grs = T_world_obj @ T_obj_grs
                pose_world_grs = utils_geo.hmat_to_pose(T_world_grs)

                posestamped_world_grs = PoseStamped()
                posestamped_world_grs.pose = pose_world_grs
                posestamped_world_grs.header = posestamped_world_obj.header

                result_poses.append(posestamped_world_grs)
                result_widths.append(float(gripper_width))
                result_ids.append((obj_no, pose_no))


#        while True:
#            T_obj_grs = self.vi.to_matrix(f'pose{pose_no}')
#            gripper_width = self.vi.get_gripper_width(f'pose{pose_no}')
#            if any((T_obj_grs is None, gripper_width is None)):
#                break
#
#            for obj_no, posestamped_world_obj in enumerate(obj_poses):
#                T_world_obj = utils_geo.pose_to_hmat(posestamped_world_obj.pose)
#                T_world_grs = T_world_obj @ T_obj_grs
#                pose_world_grs = utils_geo.hmat_to_pose(T_world_grs)
#
#                posestamped_world_grs = PoseStamped()
#                posestamped_world_grs.pose = pose_world_grs
#                posestamped_world_grs.header = posestamped_world_obj.header
#
#                result_poses.append(posestamped_world_grs)
#                result_widths.append(float(gripper_width))
#                result_ids.append((obj_no, pose_no))
#
#            pose_no += 1

        self.result_grs_poses = result_poses
        self.result_grs_widths = result_widths
        self.grs_ids = result_ids
        return (result_poses, result_widths, result_ids)

    def filter_by_angle(self, poses, widths, ids, trans, thresh_ang=45):
        poses_f, widths_f, ids_f = [], [], []
        for p, w, i in zip(poses, widths, ids):
            p_world = self._trans_cam_to_world(p, trans)
            if ArmUtils.cal_pose_angle_to_z(p_world) < thresh_ang:
                poses_f.append(p)
                widths_f.append(w)
                ids_f.append(i)
        return poses_f, widths_f, ids_f

    def _trans_cam_to_world(self, pose, trans, inv=False):
        if not isinstance(pose, PoseStamped):
            self.get_logger().warn('[FAIL] invalid type as trans_to_world')
            return None

        if not inv:
            T_cam_grs = utils_geo.pose_to_hmat(pose.pose)
            T_world_cam = utils_geo.transform_to_hmat(trans.transform)
            T_world_grs = T_world_cam @ T_cam_grs

            posestamped = PoseStamped()
            posestamped.pose = utils_geo.hmat_to_pose(T_world_grs)
            posestamped.header.stamp = pose.header.stamp
            posestamped.header.frame_id = trans.header.frame_id
            return posestamped
        else:
            T_world_grs = utils_geo.pose_to_hmat(pose.pose)
            T_world_cam = utils_geo.transform_to_hmat(trans.transform)
            T_cam_grs = np.linalg.inv(T_world_cam) @ T_world_grs

            posestamped = PoseStamped()
            posestamped.pose = utils_geo.hmat_to_pose(T_cam_grs)
            posestamped.header.stamp = pose.header.stamp
            posestamped.header.frame_id = trans.child_frame_id
            return posestamped

    # TODO: 書き換え予予定 (python関数でgoal生成してmove_armを使う)
    async def try_pickup(
      self, pose, width, setback=0.1, use_reverse=True, retreat_axis=None, va=None, force_reverse=False
    ):
        if use_reverse:
            pose = ArmUtils.reverse_yaxis(pose, force_reverse)
        va = [0.1, 0.1] if va is None else list(va)

        success, msg, goal_dict = await self.arm.pickup_planner(
            target_pose=pose.pose if isinstance(pose, PoseStamped) else pose,
            gripper_width=width, setback=setback, start_js=None, retreat_axis=retreat_axis
        )
        if not success:
            return False, msg

        # execute
        for idx, name, goal in enumerate(goal_dict.items()):
            if name in  ['approach', 'retreet']: # 速度上書き
                goal.request.max_velocity_scaling_factor = va[0]
                goal.request.max_acceleration_scaling_factor = va[1]

            # arm
            success, msg = await self.arm.move_to_joint_position_async(goal)
            if not success:
                return False, msg

            # open gripper
            if name == 'approach':
                width_ready = np.clip(width_mm + 20, 0, 85)
                self.arm.gripper.move_gripper(
                  width=width_ready, speed=100.0, force=0.0, wait_complete=True
                )

            # grasp
            if name == 'grasp':
                self.arm.gripper.grasp()

            # grasp
            if name == 'retreat_micro':
                pass # protect tip
                self.arm.gripper.grasp_full()

        return True, '[SUCCESS]'

    async def _try_pickup(self, pose, width, setback=0.1, use_reverse=True, retreat_axis=None, va=None, force_reverse=False):
        if use_reverse:
            pose = ArmUtils.reverse_yaxis(pose, force_reverse)

        if pose.header.frame_id == '':
            pose.header.frame_id = 'world'
        va = [0.1, 0.1] if va is None else list(va)
        retreat_axis = [] if retreat_axis is None else retreat_axis

        if self._ac_pick is not None and self._ac_pick.wait_for_server(timeout_sec=0.5):
            goal = PickPlaceAction.Goal()
            goal.pose = pose
            goal.width_mm = float(width)
            goal.setback = setback
            goal.retreat_axis = retreat_axis
            goal.va = va

            gh = await self._ac_pick.send_goal_async(goal)
            if not gh.accepted:
                self.get_logger().warn('[WARN] pickup goal rejected, try next')
                return (False, '')

            res = await gh.get_result_async()
            if res and res.result and res.result.success:
                self.get_logger().info('[SUCCESS] pickup (action)')
                msg = getattr(getattr(res, 'result', None), 'message', '')
                return (True, msg)
            else:
                msg = getattr(getattr(res, 'result', None), 'message', '')
                self.get_logger().warn(f'pickup fail: {msg}')
                return (False, msg)
        else:
            self.get_logger().warn('failed to use _ac_pick')
            return (False, '')

    async def try_pick_candidates(self, pose_list, setback=0.1, use_reverse=True, va=None) -> bool:
        # use_recverse:
        # True: 強制的に両方向を検証する
        # False: 既存の姿勢のみを検証する
        cand_buffer = deque(pose_list)
        while len(cand_buffer) > 0:
            pose, width, id_ = cand_buffer.popleft()
            _continue = False
            for force in [False, True]: # use_reverseがTrueのときは両方検証する
                ret, msg = await self.try_pickup(pose, width, setback, use_reverse=use_reverse, retreat_axis=None, va=va, force_reverse=force)
                if ret:
                    return id_
                elif msg in ['ik_fail', 'not_exec_pick'] and use_reverse:
                    _continue = True
                    continue
                else:
                    _continue = False
                    break

            if not _continue:
                break
        return False

    async def try_place(self, pose_place_tip, setback=0.1, va=None, use_reverse=True, approach_axis=None, start_js=None) -> bool:
        pose = ArmUtils.reverse_yaxis(pose_place_tip) if use_reverse else pose_place_tip
        if pose.header.frame_id == '':
            pose.header.frame_id = 'world'
        va = [0.1,0.1] if va is None else va

        if self._ac_place is not None and self._ac_place.wait_for_server(timeout_sec=0.5):
            goal = PickPlaceAction.Goal()
            goal.pose = pose
            goal.setback = setback
            goal.va = va
            goal.approach_axis = [] if approach_axis is None else [float(i) for i in approach_axis]
            if start_js is not None:
                goal.start_js = list(start_js)

            gh = await self._ac_place.send_goal_async(goal)
            if not gh.accepted:
                self.get_logger().warn('[WARN] place goal rejected, try next')
                return False

            res = await gh.get_result_async()
            if res and res.result and res.result.success:
                if USE_DEBUG:
                    self.get_logger().info('[SUCCESS] place (action)')
                return True
            else:
                self.get_logger().warn(f'[FAIL] place (action){res}{res.result}{res.result.success}')
                return False
        else:
            return False

    def select_target_obj(self, obj_poses_world, target=None, threshold=0.2):
        if target is None:
            target = self.cfg.xyz_inspect

        candidates = []
        for idx, ps in enumerate(obj_poses_world):
            dx = ps.pose.position.x - target[0]
            dy = ps.pose.position.y - target[1]
            dz = ps.pose.position.z - target[2]
            dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            if dist <= threshold:
                candidates.append((dist, ps))
                if idx == 0:
                    break
            if idx == 0:
                self.get_logger().warn('[INFO] max score pose is eliminated')

        if not candidates:
            return False

        return min(candidates, key=lambda x: x[0])[1]

    # =====================================================
    # utilities / inspection
    # =====================================================
    async def capture_images(self, obj_pose, yaw_start, start_js=None, va=None):
        if start_js is None:
            start_js = self.cfg.startjs_cap

        with RTimer(self, 'GEN CAPTURE POSES'):
            T_w_obj = utils_geo.pose_to_hmat(obj_pose.pose)
            z_angle = utils_geo.calc_angle_deg([0, 0, 1], T_w_obj)

            if z_angle < 30:
                surface = 'front'
                cam_pose_dict_ = self.vi.gen_capture_poses_default(surface=surface)
#                cam_pose_dict_ = self.vi.gen_capture_poses(
#                    height=self.cfg.capture_front_height,
#                    radius=self.cfg.capture_radius,
#                    z_offset=self.cfg.capture_front_z_offset,
#                    pitch_deg=self.cfg.capture_pitch_deg,
#                    vec2=[0, 0, -1]
#                )
            elif z_angle > 150:
                surface = 'back'
                cam_pose_dict_ = self.vi.gen_capture_poses_default(surface=surface)
#                cam_pose_dict_ = self.vi.gen_capture_poses(
#                    height=self.cfg.capture_back_height,
#                    radius=self.cfg.capture_radius,
#                    pitch_deg=self.cfg.capture_pitch_deg,
#                    z_offset=self.cfg.capture_back_z_offset,
#                    vec2=[0, 0, 1]
#                )
            else:
                return (False, f'[FAIL] invalid obj pose, z_angle:{z_angle}')

            if not cam_pose_dict_:
                return (False, f'[FAIL] no cam pose/{cam_pose_dict}')

            while True:
                try:
                    trans_cam_t0 = self.tf_buffer.lookup_transform(
                        self.cfg.cam_color_link,
                        'tool0',
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.2)
                    )
                    break
                except Exception:
                    await self._yield_once()
                    continue

            T_cam_t0 = utils_geo.transform_to_hmat(trans_cam_t0.transform)

            T_w_t0_dict = {}
            for id_, pose in cam_pose_dict_.items():
                T_w_c = T_w_obj @ utils_geo.pose_to_hmat(pose)
                z_angle = utils_geo.calc_angle_deg([0, 0, -1], T_w_c)
#                if z_angle > 90:
#                    continue
                T_w_c_revy = T_w_c
                T_w_t0_dict[id_] = {
                    'T': T_w_c_revy @ T_cam_t0,
                    'rev': False if np.array_equal(T_w_c, T_w_c_revy) else True
                }

            if not T_w_t0_dict:
                return (False, f'[FAIL] contain invalid cam pose / {cam_pose_dict_}')

        with RTimer(self, 'TOTAL Move and CAP'):
            result = {}
            sort_list = []

            for id_, v in T_w_t0_dict.items():
                T = v['T']
                yaw = np.arctan2(T[1, 0], T[0, 0])
                relative_yaw = (yaw - yaw_start + np.pi) % (2 * np.pi) - np.pi
                sort_list.append((relative_yaw, id_))

            sort_list.sort()

            for no, (yaw_val, id_) in enumerate(sort_list[::-1]):
                v = T_w_t0_dict[id_]
                T = v['T']

                with RTimer(self, f'{id_:4.0f}: Move and CAP, yaw={np.rad2deg(yaw_val):4.0f}', USE_DEBUG):
                    ps = PoseStamped()
                    ps.header.frame_id = 'world'
                    ps.pose = utils_geo.hmat_to_pose(T)

                    success, *msg = await self.move_arm(
                        ps,
                        va,
                        start_js=start_js if no == 0 else None,
                        rough_mode=False,
                    )
                    if not success:
                        self.get_logger().warn(f'failed move for cap: {msg[0]}')
                        continue

                    self.image_state.action_sub_flag = True
                    ok, depth_msg, rgb_msg, caminfo_msg, stamp_ns, trans = \
                        await self.wait_images_and_tf(
                            timeout_sec=6.0,
                            use_tf=True,
                            use_resize=False
                        )
                    self.image_state.action_sub_flag = False

                    if not ok:
                        self.get_logger().warn(f'failed to cap: {id_:4.0f}')
                        continue

                    result[id_] = {
                        'rgb': rgb_msg,
                        'depth': depth_msg,
                        'caminfo': caminfo_msg,
                        'rev': v['rev'],
                        'surface': surface
                    }

            if result:
                return (True, result)
            else:
                return (False, 'no result')

    def run_inspection(self, obj_name, img_msg_list, angles, surface, dist_err):
        if self._ac_inspect is not None and self._ac_inspect.wait_for_server(timeout_sec=0.5):
            goal = InspectionAction.Goal()
            goal.obj_name = obj_name
            goal.rgb_images = img_msg_list
            goal.angles = [str(int(ang)) for ang in angles]
            goal.surfaces = [surface for _ in range(len(img_msg_list))]
            goal.dist_errs = [dist_err for _ in range(len(img_msg_list))]
            gh_future = self._ac_inspect.send_goal_async(goal)
            return gh_future

    async def result_inspection(self, gh_future) -> List:
        gh = await gh_future
        if gh.accepted:
            res_future = gh.get_result_async()
            res = await res_future
            if res and res.result:
                return res.result.scores
        else:
            self.get_logger().warn('[WARN] inspection goal rejected, fallback to service')
            return []

    # =====================================================
    # optional utilities preserved from old code
    # =====================================================
    async def capture_stereo_and_publish_pointcloud(self, base_length: float = 0.1):
        with RTimer(self, 'GET IMAGES'):
            self.image_state.action_sub_flag = True
            ok, left_depth_msg, left_rgb_msg, left_caminfo_msg, stamp_ns, trans_w_color = \
                await self.wait_images_and_tf(timeout_sec=4.0, use_tf=True)
            stamp_str = utils_ros.current_stamp_str()
            utils_geo.ts_to_yaml(trans_w_color, fname=f'trans_left{stamp_str}.yaml')
            self.image_state.action_sub_flag = False

            if not ok:
                self.get_logger().warn('left画像取得失敗')
                return False

            trans_left = self.lookup_tf(
                'world', 'd405_color_optical_frame',
                rclpy.time.Time(nanoseconds=stamp_ns), 2.0
            )
            if not trans_left:
                return False

            trans_cam_t0 = self.lookup_tf(
                'd405_color_optical_frame', 'tool0',
                rclpy.time.Time(nanoseconds=stamp_ns), 2.0
            )
            if not trans_cam_t0:
                return False

            T_wc_left = utils_geo.transform_to_hmat(trans_left.transform)
            T_c_t0 = utils_geo.transform_to_hmat(trans_cam_t0.transform)
            T_offset_cam = np.eye(4)
            T_offset_cam[0, 3] = float(base_length)
            T_wc_right = T_wc_left @ T_offset_cam @ T_c_t0

            right_pose = PoseStamped()
            right_pose.header.frame_id = 'world'
            right_pose.pose = utils_geo.hmat_to_pose(T_wc_right)

            success, *msg = await self.move_arm(right_pose, [0.2, 0.2], self.cfg.startjs_pickhold)
            if not success:
                self.get_logger().warn(f"right位置移動失敗: {msg}")
                return False

            for _ in range(5):
                await self._yield_once()

            self.image_state.action_sub_flag = True
            ok, right_depth_msg, right_rgb_msg, right_caminfo_msg, stamp_ns_r, trans_right = \
                await self.wait_images_and_tf(timeout_sec=4.0, use_tf=True)
            self.image_state.action_sub_flag = False
            if not ok:
                self.get_logger().warn('right画像取得失敗')
                return False

            T_wc_right = utils_geo.transform_to_hmat(trans_right.transform)
            T_left_right = np.linalg.inv(T_wc_left) @ T_wc_right
            base_length = T_left_right[0, 3]
            self.get_logger().info(f'[INFO]base length: {base_length*1e3:.3f}')

        with RTimer(self, 'STEREO SRV'):
            cli = self.create_client(StereoDepth, '/stereo_depth_service')
            if not cli.wait_for_service(timeout_sec=3.0):
                self.get_logger().error('stereo_depth_service not ready')
                return False

            req = StereoDepth.Request()
            req.left_image = left_rgb_msg
            req.right_image = right_rgb_msg
            req.depth_image = left_depth_msg
            req.caminfo = left_caminfo_msg
            req.base_length = float(base_length)
            future = cli.call_async(req)
            res = await future

            if not (res and res.success):
                self.get_logger().warn(f"stereoサービス失敗: {res.message if res else ''}")
                return False

        with RTimer(self, 'TO PC'):
            depth_cv_ = CvBridge().imgmsg_to_cv2(res.depth_image, desired_encoding='passthrough')
            depth_cv = np.where((0 < depth_cv_) & (depth_cv_ < 1.0), depth_cv_, -1.0)
            rgb_cv = CvBridge().imgmsg_to_cv2(left_rgb_msg, desired_encoding='bgr8')
            cloud_msg, points_rgb = utils_pc.depth_to_pointcloud_color(
                depth_cv, left_caminfo_msg, rgb_cv
            )

            if not hasattr(self, 'pc_pub') or self.pc_pub is None:
                self.pc_pub = self.create_publisher(PointCloud2, '/stereo/pointcloud', 1)
            self.pc_pub.publish(cloud_msg)
            self.get_logger().info('[SUCCESS] stereo点群 publish')

            save_dir = os.path.expanduser('~/data/stereo_pc')
            os.makedirs(save_dir, exist_ok=True)
            stamp_sec = left_rgb_msg.header.stamp.sec + left_rgb_msg.header.stamp.nanosec * 1e-9
            stamp_str = datetime.datetime.fromtimestamp(stamp_sec).strftime('%Y%m%d_%H%M%S_%f')
            ply_path = os.path.join(save_dir, f'{stamp_str}_stereo_pc.ply')

            try:
                utils_pc.write_ply_xyzrgb(ply_path, points_rgb)
                self.get_logger().info(f'[SUCCESS] 点群PLY保存: {ply_path}')
            except Exception as e:
                self.get_logger().warn(f'[WARN] 点群PLY保存失敗: {e}')
            return True

    async def get_camparam(self, param_list=None):
        if param_list is None:
            param_list = ['depth_module.exposure', 'depth_module.enable_auto_exposure']

        request = GetParameters.Request()
        request.names = param_list
        fut = self.cli_camparam_get.call_async(request)

        result_dict = {}
        res = await fut
        if res is None:
            self.get_logger().warn('[WARN] get cam-param')
            return False

        for name, value in zip(param_list, res.values):
            if value.type == ParameterType.PARAMETER_BOOL:
                val = value.bool_value
            elif value.type == ParameterType.PARAMETER_INTEGER:
                val = value.integer_value
            elif value.type == ParameterType.PARAMETER_DOUBLE:
                val = value.double_value
            elif value.type == ParameterType.PARAMETER_STRING:
                val = value.string_value
            else:
                val = '未対応の型です'
            result_dict[name] = val

        return result_dict

    async def set_camparam(self, param_dict=None):
        if param_dict is None:
            param_dict = {'depth_module.exposure': 5000, 'depth_module.enable_auto_exposure': False}

        param_msgs = []
        for name, val in param_dict.items():
            _param = Parameter()
            _param.name = name
            _param.value = ParameterValue()

            if isinstance(val, bool):
                _param.value.type = ParameterType.PARAMETER_BOOL
                _param.value.bool_value = val
            elif isinstance(val, int):
                _param.value.type = ParameterType.PARAMETER_INTEGER
                _param.value.integer_value = val
            elif isinstance(val, float):
                _param.value.type = ParameterType.PARAMETER_DOUBLE
                _param.value.double_value = val
            elif isinstance(val, str):
                _param.value.type = ParameterType.PARAMETER_STRING
                _param.value.string_value = val

            param_msgs.append(_param)

        request = SetParameters.Request()
        request.parameters = param_msgs
        fut = self.cli_camparam_set.call_async(request)

        res = await fut
        if res is not None:
            return True
        else:
            return False

    # =====================================================
    # misc
    # =====================================================
    def wait_action_server(self, ac: ActionClient, name: str, total_timeout: float = 10.0, poll: float = 0.2) -> bool:
        t0 = time.time()
        while time.time() - t0 < total_timeout:
            if ac.wait_for_server(timeout_sec=poll):
                return True
        self.get_logger().warn(f'[WARN] action server not ready: {name}')
        return False

    async def _yield_once(self):
        time.sleep(0.05)

    def lookup_tf(self, frame_id='world', child='tool0', time=None, timeout_sec=2):
        try:
            trans = self.tf_buffer.lookup_transform(
                frame_id,
                child,
                time if time is not None else rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec)
            )
            return trans
        except Exception as e:
            self.get_logger().warn(f"TF取得失敗(left時刻): {e}")
            return False

    def get_cam_tf(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.cfg.cam_color_link,
                'tool0',
                rclpy.time.Time(),
                timeout=Duration(seconds=5.0)
            )
            self.cfg.cam_t0_xyz = [getattr(trans.transform.translation, attr) for attr in 'xyz']
            self.cfg.cam_t0_quat = [getattr(trans.transform.rotation, attr) for attr in 'xyzw']
            self.get_logger().info(f"[SUCCESS] GET cam_t0_tf")
        except Exception as e:
            self.get_logger().warn(f"[FAIL] GET cam_t0_tf: {e}")
            return False

    def _abort(self, goal_handle, message: str):
        self.get_logger().warn(f'abort:{message}')
        result = BoolAction.Result()
        result.success = False
        result.message = message
        goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt as e:
        print(f'KeyboardInterrupt: {e}')
    finally:
        node.arm.gripper.closing()
        executor.shutdown()
        node.end_auto_save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
