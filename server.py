# generated with azure
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from tf2_ros import TransformBroadcaster

import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
# from workcell_interfaces.srv import PickPlace, MoveArm, ComputeIK
from workcell_interfaces.action import MoveArmAction, PickPlaceAction  # ← 追加（PickPlaceAction）

from pymoveit2 import MoveIt2
from gripper_2f85_lib.utils import GripperUtils
from gripper_2f85_lib.controller import GripperControl
from arm_ur_ros2.simple_node import URControlSimple
# from arm_ur_ros2.complex_node import URControl
from arm_ur_lib import utils as ArmUtils
from as_utils import utils_geo
from vgn.utils.ur_control import URCommander

USE_DEBUG = False

class URServerNode(Node):
    def __init__(self, driver):
        """
        初回動作時にdriverのthreadが落ちる問題は二重spinが原因と考えられるため
        collision objectなどコールバックで処理したいものが終わった段階で
        executor.shutdown()することで暫定的な対処は行った
        """
        super().__init__('ur_server_node')
        self.driver = driver
        self.pub_js_gripper = self.create_publisher(JointState, '/gripper/joint_states', 10)
        self.gripper = GripperControl()
        self.gripper_command('close')

        # 排他制御（ドライバ・グリッパ操作の混在防止）
        self._op_lock = threading.Lock()

        # サービス
        #self.move_srv = self.create_service(MoveArm, '/ur_command/move_arm', self.move_cb)
        #self.pickup_srv = self.create_service(PickPlace, '/ur_command/pickup_object', self.pickup_cb)  # 後方互換のため残す
        #self.compute_ik_srv = self.create_service(ComputeIK, '/ur_command/compute_ik', self.compute_ik_cb)

        # MoveArmアクションサーバ（推奨インタフェース）
        self._move_action_cbg = MutuallyExclusiveCallbackGroup()
        self._move_active = False
        self._as_move = ActionServer(
            self,
            MoveArmAction,
            '/ur_command/move_arm_action',
            execute_callback=self.execute_move_action,
            goal_callback=self.move_goal_cb,
            cancel_callback=self.move_cancel_cb,
            callback_group=self._move_action_cbg
        )

        # Pickアクションサーバ
        self._pick_action_cbg = MutuallyExclusiveCallbackGroup()
        self._pick_active = False
        self._as_pick = ActionServer(
            self,
            PickPlaceAction,
            '/ur_command/pickup_object_action',
            execute_callback=self.execute_pick_action,
            goal_callback=self.pick_goal_cb,
            cancel_callback=self.pick_cancel_cb,
            callback_group=self._pick_action_cbg
        )

        # Placeアクションサーバ
        self._place_action_cbg = MutuallyExclusiveCallbackGroup()
        self._place_active = False
        self._as_place = ActionServer(
            self,
            PickPlaceAction,
            '/ur_command/place_object_action',
            execute_callback=self.execute_place_action,
            goal_callback=self.place_goal_cb,
            cancel_callback=self.place_cancel_cb,
            callback_group=self._place_action_cbg
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info('all ready')


    # ========== アクション: MoveArm ==========
    def move_goal_cb(self, goal_request):
        if self._move_active:
            self.get_logger().warn('move_arm_action: 忙しいためゴールを拒否します')
            return GoalResponse.REJECT
        self._move_active = True
        self.get_logger().info('move_arm_action: ゴール受理')
        return GoalResponse.ACCEPT

    def move_cancel_cb(self, goal_handle):
        self.get_logger().info('move_arm_action: キャンセル要求を受信')
        return CancelResponse.ACCEPT

    async def execute_move_action(self, goal_handle):
        feedback = MoveArmAction.Feedback()
        result = MoveArmAction.Result()
        try:
            joints = list(goal_handle.request.joint_positions) \
                if len(goal_handle.request.joint_positions) > 0 else None
            pose = goal_handle.request.pose \
                if hasattr(goal_handle.request, 'pose') else None
            va = list(goal_handle.request.va) \
                if hasattr(goal_handle.request, 'va') and len(goal_handle.request.va) > 0 else [0.1, 0.1]
            start_js = list(goal_handle.request.start_js) \
                if hasattr(goal_handle.request, 'start_js') and len(goal_handle.request.start_js) > 0 else None
            rough_mode = goal_handle.request.rough_mode if hasattr(goal_handle.request, 'rough_mode') else False
            self.get_logger().info('start moving soon...')
            feedback.message = 'moving...'
            goal_handle.publish_feedback(feedback)

            with self._op_lock:
                if joints is not None and len(joints) == 6:
                    # 可視化TF
                    pose_pub = self.driver.compute_fk(joints)
                    pose_pub.header.stamp = self.get_clock().now().to_msg()
                    self.publish_tf(pose_pub, child_frame_id='tool0_target')
                    tolerance = 0.02 if rough_mode else 0.001
                    ok = self.driver.move_joint_position(joints, va, tolerance)
                    exec_info = f'joint_positions={joints}'
                elif pose is not None and pose.header.frame_id == 'world':
                    # 可視化TF
                    pose_pub = PoseStamped()
                    pose_pub.header.frame_id = 'world'
                    pose_pub.header.stamp = self.get_clock().now().to_msg()
                    pose_pub.pose = pose.pose
                    self.publish_tf(pose_pub, child_frame_id='tool0_target')
                    ok = self.driver.move_arm(pose.pose, start_js, va, False, rough_mode)
                    exec_info = f'pose={pose.pose.position}'
                else:
                    ok = False
                    exec_info = f'no valid joint_positions or pose: pose:{pose}, joints:{joints}'

            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'canceled'
                goal_handle.canceled()
                return result

            result.success = bool(ok)
            result.message = f'success ({exec_info})' if ok else f'failed ({exec_info})'
            if not result.success:
                self.get_logger().warn(f'message:{result.message}')

            if ok:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        except Exception as e:
            self.get_logger().error(f'move_arm_action: 例外 {e}')
            result.success = False
            result.message = f'exception: {e}'
            goal_handle.abort()
            return result
        finally:
            self._move_active = False

    # ========== アクション: Pick ==========
    def pick_goal_cb(self, goal_request):
        if self._pick_active:
            self.get_logger().warn('pickup_object_action: 忙しいためゴールを拒否します')
            return GoalResponse.REJECT
        self._pick_active = True
        self.get_logger().info('pickup_object_action: ゴール受理')
        return GoalResponse.ACCEPT

    def pick_cancel_cb(self, goal_handle):
        self.get_logger().info('pickup_object_action: キャンセル要求を受信')
        return CancelResponse.ACCEPT

    async def execute_pick_action(self, goal_handle):
        feedback = PickPlaceAction.Feedback()
        result = PickPlaceAction.Result()
        try:
            goal = goal_handle.request
            self.get_logger().info('receive pickup (action) goal')
            self.publish_tf(goal.pose, child_frame_id='tip_target')

            feedback.message = 'executing pickup...'
            goal_handle.publish_feedback(feedback)

            va = None if len(goal.va) == 0 else list(goal.va)
            with self._op_lock:
                ret = self.pickup(
                    pose=goal.pose.pose,
                    width_mm=goal.width_mm,
                    setback=goal.setback,
                    start_js=None,
                    va=va,
                    retreat_axis=goal.retreat_axis,
                )

            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'canceled'
                goal_handle.canceled()
                return result

            result.success = ret['success']
            result.message = ret['message']

            if result.success:
                self.get_logger().info('[SUCCESS] pickup (action)')
                goal_handle.succeed()
            else:
                self.get_logger().warn(f'[FAIL] pickup (action): {result.message}')
                goal_handle.abort()
            self.get_logger().info('-----------------------------')
            return result
        except Exception as e:
            self.get_logger().error(f'pickup_object_action: 例外 {e}')
            result.success = False
            result.message = f'exception: {e}'
            goal_handle.abort()
            return result
        finally:
            self._pick_active = False

    # ========== アクション: Place ==========
    def place_goal_cb(self, goal_request):
        if self._place_active:
            self.get_logger().warn('place_object_action: 忙しいためゴールを拒否します')
            return GoalResponse.REJECT
        self._place_active = True
        if USE_DEBUG:
            self.get_logger().info('place_object_action: ゴール受理')
        return GoalResponse.ACCEPT

    def place_cancel_cb(self, goal_handle):
        self.get_logger().info('place_object_action: キャンセル要求を受信')
        return CancelResponse.ACCEPT

    async def execute_place_action(self, goal_handle):
        feedback = PickPlaceAction.Feedback()
        result = PickPlaceAction.Result()
        try:
            goal = goal_handle.request
            self.get_logger().info('receive place (action) goal')
            self.publish_tf(goal.pose, child_frame_id='tip_target')

            feedback.message = 'executing place...'
            goal_handle.publish_feedback(feedback)

            # 受け取ったものが型どおりにlistになっていない場合があるarray.arrayとか
            start_js = list(goal.start_js) if len(goal.start_js) > 0 else None
            va = [0.1,0.1] if len(goal.va) == 0 else list(goal.va)
            with self._op_lock:
                ret = self.place(
                    pose=goal.pose.pose,
                    setback=goal.setback,
                    start_js=start_js,
                    va=va,
                    approach_axis=goal.approach_axis,
                )

            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'canceled'
                goal_handle.canceled()
                return result

            result.success = ret['success']
            result.message = ret['message']

            if result.success:
                self.get_logger().info('[SUCCESS] place (action)')
                goal_handle.succeed()
            else:
                self.get_logger().warn(f'[FAIL] place (action): {result.message}')
                goal_handle.abort()
            self.get_logger().info('-----------------------------')
            return result
        except Exception as e:
            self.get_logger().error(f'place_object_action: 例外 {e}')
            result.success = False
            result.message = f'exception: {e}'
            goal_handle.abort()
            return result
        finally:
            self._place_active = False

    # ========== ComputeIK サービス ==========
    def compute_ik_cb(self, request, response):
        self.get_logger().info('receive compute_ik request')
        with self._op_lock:
            try:
                pose = request.pose.pose if isinstance(request.pose, PoseStamped) else request.pose
                start_js = list(request.start_joint_state) if hasattr(request, 'start_joint_state') else None
                js = self.driver.compute_ik(pose, start_js=start_js) if start_js else self.driver.compute_ik(pose)
                if js is False or js is None:
                    response.success = False
                    response.joint_state = []
                    response.message = 'ik_fail'
                else:
                    response.success = True
                    response.joint_state = list(js)
                    response.message = 'success'
            except Exception as e:
                self.get_logger().error(f'compute_ik: 例外 {e}')
                response.success = False
                response.joint_state = []
                response.message = f'exception: {e}'
        self.get_logger().info('response\n----------------------------')
        return response

    # ========== サービスコールバック ==========
    def _move_cb(self, request, response): # TODO: モウツカッテイナイノデハ？
        self.get_logger().warn('NOTICE: MoveArmはアクション通信(/ur_command/move_arm_action)への切り替えを推奨します。')
        self.get_logger().info('receive move_arm request')
        with self._op_lock:
            if request.message == 'home':
                ret = self.home()
            elif request.pose.header.frame_id != 'world':
                self.get_logger().warn(f'[FAIL] <move_arm> invalid frame_id: {request.pose.header.frame_id}')
                response.success = False
                return response
            else:
                va = list(request.va) if hasattr(request, 'va') and len(request.va) > 0 else [0.1, 0.1]
                ret = self.driver.move_arm(request.pose.pose, None, va, False)
        response.success = bool(ret)
        return response

    def pickup_cb(self, request, response):
        # 後方互換のためサービスは残すが、アクションへの移行を促す
        self.get_logger().warn('NOTICE: ピックアップはアクション通信(/ur_command/pickup_object_action)への切り替えを推奨します。')
        self.get_logger().info('receive pickup (service) request')
        self.publish_tf(request.pose, child_frame_id='grasp_target')
        with self._op_lock:
            ret = self.pickup(
                pose=request.pose.pose, width_mm=request.width_mm,
                setback=request.setback, start_js=None, va=request.va
            )
        response.success = ret['success']
        response.message = ret['message']
        self.get_logger().info('-----------------------------')
        return response

    def place_cb(self, request, response):
        self.get_logger().warn('NOTICE: プレースはアクション通信への切り替えを推奨します。')
        xyz = utils_geo.pose_to_hmat(request.pose.pose)[:3, 3]
        self.get_logger().info(f'[START]receive request: xyz={xyz}')
        self.publish_tf(request.pose)

        with self._op_lock:
            response.success = self.place(
                pose=request.pose.pose, setback=request.setback, va=request.va)
        return response

    # ========== ユーティリティ ==========
    def gripper_command(self, option=None, **kwargs):
        # コマンドを実行した後グリッパーの状態をパブ
        # TODO: hasatterに変更
        if option == 'open':
            self.gripper.open()
        elif option == 'close':
            self.gripper.close()
        elif option == 'move':
            self.gripper.move_gripper(**kwargs)
        elif option == 'grasp':
            self.gripper.grasp()
        elif option == 'grasp_full':
            self.gripper.grasp_full()
        
        # [0,0.8] / [0, 255]
        try:
            gpo = self.gripper.gripper_status['gPO']
            js = np.clip(gpo / 255 , 0, 1) * 0.8 

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ['robotiq_85_left_knuckle_joint',]
            msg.position = [js,]
            self.pub_js_gripper.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'[FAIL] pub gripper js, {e}')
            
    def home(self, va=None):
        if va is None:
            va = [0.1, 0.1]
        xyz = np.array([0.165, 0.325, 0.350])
        rpy = np.deg2rad([-175, 0, 0])
        pose = utils_geo.hmat_to_pose(
            utils_geo.xyzrpy_to_hmat(xyz, rpy))
        return self.driver.move_arm(pose, None, va, False)

    def place_demo(self):
        xyz = np.array([-0.575, 0.125, 0.550])
        rpy = np.deg2rad([-175, 0, 0])
        pose = utils_geo.hmat_to_pose(
            utils_geo.xyzrpy_to_hmat(xyz, rpy))
        if not self.driver.move_arm(pose, None, [0.2, 1.0], False):
            return False
        self.gripper_command('open')
        return True

    def inspect_demo(self):
        xyz = np.array([-0.500, 0.500, 0.550])
        rpy = np.deg2rad([-175, 0, 0])
        pose = utils_geo.hmat_to_pose(
            utils_geo.xyzrpy_to_hmat(xyz, rpy))
        if not self.driver.move_arm(pose, None, [0.2, 1.0], False):
            return False
        return True

    def hold_demo(self, va=None):
        if va is None:
            va = [0.1, 0.1]
        xyz = np.array([0.165, 0.325, 0.450])
        rpy = np.deg2rad([-175, 0, 0])
        pose = utils_geo.hmat_to_pose(
            utils_geo.xyzrpy_to_hmat(xyz, rpy))
        return self.driver.move_arm(pose, None, va, False)

    def publish_tf(self, pose_stamped, child_frame_id='grasp_target'):
        t = TransformStamped()
        t.transform = utils_geo.pose_to_transform(pose_stamped.pose)
        t.header = pose_stamped.header
        if t.header.frame_id not in ['world', 'base_link']:
            self.get_logger().warn(f'\npub tf: {t.header.frame_id}@{child_frame_id}\n')
            t.header.frame_id = 'world'
        t.header.stamp = self.get_clock().now().to_msg()
        t.child_frame_id = child_frame_id
        self.tf_broadcaster.sendTransform(t)
        if USE_DEBUG:
            self.get_logger().info(f'[SUCCESS] publish tf as {child_frame_id}, t={t.header.stamp}')

    def pickup(self, pose, width_mm=10, setback=0.1, start_js=None, va=None, retreat_axis=None):
#        _va = va
#        va = [0.1,0.1] if va is None or len(va) == 0 else list(va)
#        self.get_logger().info(f'**va: {_va} -> {va}**')
#        self.get_logger().info(f'**va={[f"{_va:.1f}" for _va in va]}**')
        if len(retreat_axis) == 0: retreat_axis = None
        # 退避->把持->退避まで実行
        pose_grs = GripperUtils.cal_pick_tool0_pose(pose, width_mm, 0.0)
        pose_approach = ArmUtils.cal_retreat_pose(pose_grs, setback=setback, retreat_axis=None)
        pose_retreat_micro = ArmUtils.cal_retreat_pose(pose_grs, setback=0.010, retreat_axis=retreat_axis)
        pose_retreat = ArmUtils.cal_retreat_pose(pose_grs, setback, retreat_axis)

        # 可視化TF
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_retreat
        self.publish_tf(pose_pub, child_frame_id='pickup_retreat_tool0')

        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_grs
        self.publish_tf(pose_pub, child_frame_id='pickup_tool0')

        # ckeck ik
        n_try = 3
        for i_try in range(n_try):
            js = self.driver.compute_ik(pose_grs)
            if not js:
                self.get_logger().info('[FAIL] ik_err @ grs')
                return {'success': False, 'message': 'ik_fail'}
            for sb in np.linspace(0,setback,10)[1:]:
                pose_ = ArmUtils.cal_retreat_pose(pose_grs, setback=sb, retreat_axis=None)
                js = self.driver.compute_ik(pose_, start_js=js)
                if not js:
                    self.get_logger().info(f'[FAIL] ik_err sb={sb}')
                    if i_try < n_try - 1:
                        continue
                    return {'success': False, 'message': 'ik_fail'}
                if retreat_axis is not None:
                    pose_ = ArmUtils.cal_retreat_pose(pose_grs, setback=sb, retreat_axis=retreat_axis)
                    js = self.driver.compute_ik(pose_, start_js=js)
                    if not js:
                        self.get_logger().info(f'[FAIL] ik_err setback at {sb}')
                        if i_try < n_try - 1:
                            continue
                        return {'success': False, 'message': 'ik_fail'}
            self.get_logger().info('[SUCCESS] pre-ik check')

        # open gripper
        width_ready = np.clip(width_mm + 20, 0, 85)
        #self.gripper.move_gripper(width=width_ready, speed=100.0, force=0.0, wait_complete=True)
        self.gripper_command('move', width=width_ready, speed=100.0, force=0.0, wait_complete=True)

        # ready
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_approach
        self.publish_tf(pose_pub, child_frame_id='tool0_target')
        if self.driver.move_arm(pose_approach, start_js, va, False, rough_mode=True) is False:
            self.get_logger().warn('[FAIL] not_exec_pick_at_ready')
            return {'success': False, 'message': 'not_exec_pick'}
        # pick
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_grs
        self.publish_tf(pose_pub, child_frame_id='tool0_target')
        if self.driver.move_arm(pose_grs, None, [0.1,0.1], False, rough_mode=False) is False:
            #self.home()
            self.get_logger().warn('[FAIL] not_exec_pick_at_pick')
            return {'success': False, 'message': 'not_exec_pick'}
        #self.gripper.grasp()
        self.gripper_command('grasp')
        # retreat_micro
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_retreat_micro
        self.publish_tf(pose_pub, child_frame_id='tool0_target')
        if self.driver.move_arm(pose_retreat_micro, None, [0.05, 0.1], False, rough_mode=False) is False:
            return {'success': False, 'message': 'exec_fail'}
        # retreat
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_retreat
        self.publish_tf(pose_pub, child_frame_id='tool0_target')
        if self.driver.move_arm(pose_retreat, None, va, False, rough_mode=True) is False:
            return {'success': False, 'message': 'exec_fail'}
        self.gripper_command('grasp_full')
        # return result
        if self.gripper.is_grasp():
            return {'success': True, 'message': 'success'}
        else:
            return {'success': False, 'message': 'grasp_fail'}

    def place(self, pose, setback=0.1, start_js=None, va=None, approach_axis=None):
        if va is None: va = [0.1, 0.1]
        if len(approach_axis) == 0: approach_axis = None
        # 退避->把持->退避まで実行
        pose_place = GripperUtils.cal_pick_tool0_pose(pose, 0, 0.0)
        pose_retreat = ArmUtils.cal_retreat_pose(pose_place, setback)
        pose_approach = ArmUtils.cal_retreat_pose(
            pose_place, setback=setback, retreat_axis=approach_axis)

        # 可視化TF
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_approach
        self.publish_tf(pose_pub, child_frame_id='place_approach_tool0')

        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_retreat
        self.publish_tf(pose_pub, child_frame_id='place_retreat_tool0')

        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_place
        self.publish_tf(pose_pub, child_frame_id='place_tool0')

        # ckeck ik
        js_rt = self.driver.compute_ik(pose_approach, start_js=start_js)
        if not js_rt:
            self.get_logger().info('[FAIL] ik_err approach pose')
            return {'success': False, 'message': 'ik_fail_approach'}

        js_rt = self.driver.compute_ik(pose_retreat, start_js=start_js)
        if not js_rt:
            self.get_logger().info('[FAIL] ik_err retreat pose')
            return {'success': False, 'message': 'ik_fail_retreat'}

        if not self.driver.compute_ik(pose_place, start_js=js_rt):
            self.get_logger().info('[FAIL] ik_err place pose')
            return {'success': False, 'message': 'ik_fail_place'}
        self.get_logger().info('[SUCCESS] pre-ik check')


        # ready
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_approach
        self.publish_tf(pose_pub, child_frame_id='tool0_target')
        if self.driver.move_arm(pose_approach, start_js, va, False, rough_mode=True) is False:
            return {'success': False, 'message': 'not_exec_place'}
        # place
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_place
        self.publish_tf(pose_pub, child_frame_id='tool0_target')
        if self.driver.move_arm(pose_place, None, [0.1,0.1], False, rough_mode=False) is False:
            return {'success': False, 'message': 'exec_fail'}
        width_mm = self.gripper.get_width()
        self.gripper_command('move', width=width_mm + 15, speed=100.0, force=0.0, wait_complete=True)

        # retreat
        pose_pub = PoseStamped()
        pose_pub.header.frame_id = 'world'
        pose_pub.header.stamp = self.get_clock().now().to_msg()
        pose_pub.pose = pose_retreat
        self.publish_tf(pose_pub, child_frame_id='tool0_target')
        if self.driver.move_arm(pose_retreat, None, va, False, rough_mode=True) is False:
            return {'success': False, 'message': 'exec_fail'}
        #self.gripper_command('close')
        self.gripper_command('move', width=0.0, speed=100.0, force=0.0, wait_complete=False)
        return {'success': True, 'message': 'success'}


def main(args=None):
    rclpy.init(args=args)
    use_simplecontroller = True

    if use_simplecontroller:
        # 新しい「純粋な制御ノード」(pymoveit2、spinしない)
        driver = URControlSimple()
        executor_driver = rclpy.executors.SingleThreadedExecutor()
        executor_driver.add_node(driver)
        #driver.executor = executor_driver
        # spinもthreadも不要
        if False: # 260410からおかしくなった強制的ににspinしないと動かない
            thread_driver = threading.Thread(target=executor_driver.spin, daemon=True)
            thread_driver.start()
        if True: # 260410からおかしくなった強制的ににspinしないと動かない/launchのNodeでnameを上書きして問題になっていた/解消した
            thread_driver = threading.Thread(target=executor_driver.spin_once, daemon=True)
            thread_driver.start()
    else:
        # 以前の「環境構築/管理も含む複雑なノード」（spin必須）
        driver = URControl()
        executor_driver = rclpy.executors.SingleThreadedExecutor()
        executor_driver.add_node(driver)
        driver.executor = executor_driver
        thread_driver = threading.Thread(target=executor_driver.spin, daemon=True)
        thread_driver.start()

    node = URServerNode(driver)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.gripper.closing()
        node.destroy_node()
        driver.destroy_node()
        rclpy.shutdown()
