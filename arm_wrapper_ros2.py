import threading
import time
from typing import List, Optional, Tuple, Union

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor,  MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_srvs.srv import Trigger

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from action_msgs.msg import GoalStatus

from gripper_2f85_lib.utils import GripperUtils
from gripper_2f85_lib.controller import GripperControl
from arm_ur_lib import utils as ArmUtils
from as_helper.error_handler import error_to_str

def set_default(obj, attr, default):
    if getattr(obj, attr) in (None, 0, "", False):
        setattr(obj, attr, default)

USE_GRIPPER = True

# wrapper
class ArmWrapper:
    def __init__(self, node, joint_names=None):
        self.node = node
        self.cbg_arm = ReentrantCallbackGroup()

        if joint_names is not None:
            self.joint_names = joint_names
        else:
            self.node.get_logger().warn('Recommend to set joint_name')
            self.joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]

        self._joint_state: Optional[JointState] = None
        self._joint_state_event = threading.Event()
        self._joint_state_lock = threading.Lock()

        self._ik_client = self.node.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cbg_arm
        )
        self._move_group_client = ActionClient(
            self.node, MoveGroup, "move_action", callback_group=self.cbg_arm
        )

        self.node.create_subscription(
            JointState,
            "joint_states",
            self._joint_state_callback,
            10,
            callback_group=self.cbg_arm
        )

        # demo
        self.node.create_service(
            Trigger,
            "move_pose",
            self._on_trigger_move_pose,
            callback_group=self.cbg_arm
        )

        # === GRIPPER ===
        if USE_GRIPPER:
            self.gripper = GripperControl()
            self.gripper.close()
            self._gripper_js = None
            self.pub_js_gripper = self.node.create_publisher(JointState, '/gripper/joint_states', 10)
            self.cbg_gripper = MutuallyExclusiveCallbackGroup()
            self.node.create_timer(1.0, self.publish_gripper_status, self.cbg_gripper)

    def publish_gripper_status(self):
        gpo = self.gripper.gripper_status['gPO']
        js = np.clip(gpo / 255 , 0, 1) * 0.8 

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ['robotiq_85_left_knuckle_joint',]
        msg.position = [js,]
        self.pub_js_gripper.publish(msg)
        with self._joint_state_lock:
            self._gripper_js = msg

    def _joint_state_callback(self, msg: JointState):
        with self._joint_state_lock:
            self._joint_state = msg
        self._joint_state_event.set()

    def get_current_joint_state(self) -> Optional[JointState]:
        with self._joint_state_lock:
            return self._joint_state

    def build_joint_goal_constraints(
        self,
        goal,
        joint_positions: JointState,
        tolerance=0.001,
    ) -> bool:

        joint_names = joint_positions.name
        joint_position = joint_positions.position

        if isinstance(tolerance, float):
            tolerance = [tolerance for _ in range(len(joint_positions.name))]

        constraints = []
        for name, pos, tol in zip(joint_names, joint_position, tolerance):
            c = JointConstraint()
            c.joint_name = name
            c.position = float(pos)
            c.tolerance_above = tol
            c.tolerance_below = tol
            c.weight = 1.0
            constraints.append(c)

        constraint = Constraints()
        constraint.joint_constraints.extend(constraints)
        goal.request.goal_constraints = [constraint]
        return True

    def _build_joint_path_constraints(
        self,
    ) -> Constraints:
        # 未定義
        return Constraints()

    def _check_goal(self, goal):
        # goalの必要事項が抜けていないか確認してgoalを返す(未設定項目はデフォルト値で埋める)

        set_default(goal.request.workspace_parameters.header, 'frame_id', "base_link")
        set_default(goal.request, 'group_name', "ur_manipulator")
        set_default(goal.request, 'max_velocity_scaling_factor', 0.1)
        set_default(goal.request, 'max_acceleration_scaling_factor', 0.1)

        # planner
        set_default(goal.planning_options, 'plan_only', False)
        set_default(goal.request, 'pipeline_id', "")
        set_default(goal.request, 'planner_id', "") # default: RRTConnetction/defined in controllers.yaml?
        set_default(goal.request, 'num_planning_attempts', 5)
        set_default(goal.request, 'allowed_planning_time', 2.0)

#        # cartesian
#        if hasattr(goal.request, "cartesian_speed_limited_link"):
#            goal.request.cartesian_speed_limited_link = end_effector
#        else:
#            goal.request.cartesian_speed_end_effector_link = end_effector
#        goal.request.max_cartesian_speed = 0.0

        # path constraint (経路中の関節毎の可動範囲制限)
        set_default(goal.request, 'path_constraints', self._build_joint_path_constraints())

        # start js
        current_js = self.get_current_joint_state()
        if current_js is not None:
            set_default(goal.request.start_state, 'joint_state', current_js)

        # goal constraint
        if goal.request.goal_constraints is None:
            return False
        return True

    async def compute_ik_async(self, target_pose, start_js=None) -> Optional[JointState]:
        if not self._ik_client.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().error("compute_ik service is not available")
            return None
        if not isinstance(start_js, JointState) and start_js is not None:
            self.node.get_logger().warn(f'start_js is invalid type / {type(start_js)}')

        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose = target_pose

        current_js = self.get_current_joint_state() if start_js is None else start_js
        if current_js is not None:
            req.ik_request.robot_state.joint_state = current_js

        fut = self._ik_client.call_async(req)
        res = await fut

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().error(f"IK failed: {res.error_code.val}")
            return None

        return res.solution.joint_state

    async def pose_to_js(self, target_pose, start_js=None):
        ik = await self.compute_ik_async(target_pose, start_js=start_js)
        if ik is None:
            return False, None

        joint_position = []
        for target_name in self.joint_names:
            if target_name not in ik.name:
                self.node.get_logger().error(f"Joint {target_name} not found in IK result")
                return False, None
            joint_position.append(ik.position[ik.name.index(target_name)])
        js = JointState()
        js.name = self.joint_names
        js.position = joint_position
        return True, js

    async def compute_trajectory(self, poses_list, start_js=None):
        joint_position = start_js
        for pose_info in poses_list:
            pose = pose_info['pose']
            success, joint_position = await self.pose_to_js(pose, start_js=joint_position)
            if not success:
                return False
        return True

    async def move_to_joint_positions_async(self, goal) -> bool:
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("MoveGroup action server is not available")
            return False, 'sever not working'

        success = self._check_goal(goal)
        if not success:
            return False, 'goal is invalid'

        goal_future = self._move_group_client.send_goal_async(goal)
        goal_handle = await goal_future
        if not goal_handle.accepted:
            self.node.get_logger().error("MoveGroup goal was rejected")
            return False, 'goal rejected'

        result = await goal_handle.get_result_async()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().error(f"MoveGroup failed: status={result.status}")
            return False, 'execut error'

        self.node.get_logger().info("MoveGroup succeeded")
        return True, ''

    async def pickup_planner(self, target_pose, gripper_width, setback, start_js=None, retreat_axis=None):
        # 姿勢計算 (退避->把持->退避)
        if isinstance(target_pose, PoseStamped):
            target_pose = target_pose.pose

        pose_grs = GripperUtils.cal_pick_tool0_pose(target_pose, gripper_width, 0.0) # tipの姿勢からtool0の姿勢を計算
        poses = [
            {
                'name': 'approach',
                'pose': ArmUtils.cal_retreat_pose(pose_grs, setback=setback, retreat_axis=None)
            }),
            {
                'name': 'grasp',
                'pose': pose_grs
            },
            {
                'name': 'retreat_micro',
                'pose': ArmUtils.cal_retreat_pose(pose_grs, setback=0.010, retreat_axis=retreat_axis)
            },
            {
                'name': 'retreat',
                'pose': ArmUtils.cal_retreat_pose(pose_grs, setback=setback, retreat_axis=retreat_axis)
            },
        ]

        # # 可視化情報 (ここでは発信せずに返す)
        # tf_dict = {}
        # for pose_info in poses:
        #     if pose_info['name'] in ['grasp', 'retreet']:
        #         pose_pub = PoseStamped()
        #         pose_pub.header.frame_id = 'world'
        #         pose_pub.header.stamp = self.node.get_clock().now().to_msg()
        #         pose_pub.pose = pose_info['pose']
        #         tf_dict[f'pickup_{pose_info["name"]}_tool0'] = pose_pub

        # ikチェック (正式にはtrajectoryのikを実装予定)
        success = await self.compute_trajectory(poses, start_js)
        if not success:
            return False, 'ik_fail', {}
        else:
            self.node.get_logger().info('[SUCCESS] pre-ik check')

        # goal
        goal_dict = {}
        joint_position = start_js
        for i, pose_info in enumerate(poses):
            pose = pose_info['pose']
            name = pose_info['name']

            goal = MoveGroup.Goal()
            goal.request.max_velocity_scaling_factor = 0.1 # 実行時の上書きを想定
            goal.request.max_acceleration_scaling_factor = 0.1 # 実行時の上書きを想定

            succecss, joint_position = await self.pose_to_js(pose, start_js=joint_position)
            if not success:
                return False, '[FAIL]build goal', {}

            success = self.build_joint_goal_constraints(goal, joint_position, tolerance=0.001)
            if not success:
                return False, '[FAIL]build goal2', {}
            goal_dict[name] = goal
        return True, '[SUCCESS]', goal_dict

    async def place_planner(self, tip_pose, setback=0.1, start_js=None, approach_axis=None):
        # gripper widthはpublisherから取得できる
        if USE_GRIPPER and self._gripper_js is not None:
            gripper_value = self._gripper_js.position[0]
            gripper_width = GripperUtils.width_bin2mm(gripper_value * 255)
        else:
            gripper_width = 0.0

        # 姿勢計算 (退避->把持->退避)
        if isinstance(tip_pose, PoseStamped):
            tip_pose = target_pose.pose
        
        pose_place = GripperUtils.cal_pick_tool0_pose(pose, gripper_width, 0.0) # tipの姿勢からtool0の姿勢を計算
        poses = [
            {
                'name': 'approach',
                'pose': ArmUtils.cal_retreat_pose(
                    pose_place, setback=setback, retreat_axis=approach_axis
                )
            }),
            {
                'name': 'place',
                'pose': pose_place
            },
            {
                'name': 'retreat',
                'pose': ArmUtils.cal_retreat_pose(pose_place, setback)
            },
        ]

        # # 可視化情報 (ここでは発信せずに返す)
        # tf_dict = {}
        # for pose_info in poses:
        #     if pose_info['name'] in ['place', 'retreet']:
        #         pose_pub = PoseStamped()
        #         pose_pub.header.frame_id = 'world'
        #         pose_pub.header.stamp = self.node.get_clock().now().to_msg()
        #         pose_pub.pose = pose_info['pose']
        #         tf_dict[f'pickup_{pose_info["name"]}_tool0'] = pose_pub

        # ikチェック (正式にはtrajectoryのikを実装予定)
        success = await self.compute_trajectory(poses, start_js)
        if not success:
            return False, 'ik_fail', {}
        else:
            self.node.get_logger().info('[SUCCESS] pre-ik check')

        # goal
        goal_dict = {}
        joint_position = start_js
        for i, pose_info in enumerate(poses):
            pose = pose_info['pose']
            name = pose_info['name']

            goal = MoveGroup.Goal()
            goal.request.max_velocity_scaling_factor = 0.1 # 実行時の上書きを想定
            goal.request.max_acceleration_scaling_factor = 0.1 # 実行時の上書きを想定

            succecss, joint_position = await self.pose_to_js(pose, start_js=joint_position)
            if not success:
                return False, '[FAIL]build goal', {}

            success = self.build_joint_goal_constraints(goal, joint_position, tolerance=0.001)
            if not success:
                return False, '[FAIL]build goal2', {}
            goal_dict[name] = goal
        return True, '[SUCCESS]', goal_dict
    
    # === demo ===
    async def move_demo(self) -> bool:
        target_pose = Pose()
        target_pose.position.x = 0.3
        target_pose.position.y = 0.4
        target_pose.position.z = 0.4
        target_pose.orientation.x = 1.0
        target_pose.orientation.y = 0.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        if False:
            goal = MoveGroup.Goal()
            success, joint_position = await self.pose_to_js(target_pose)
            if not success:
                return False

            success = self.build_joint_goal_constraints(goal, joint_position, None, tolerance=0.001)
            if not success:
                return False

            return await self.move_to_joint_positions_async(goal)

        if True:
            success, msg, goals = await self.pickup_planner(
                target_pose, gripper_width=10, setback=0.1, start_js=None, retreat_axis=None
            )
            print(goals)
            for name, goal in goals.items():
                success = await self.move_to_joint_positions_async(goal)
                if not success:
                    return False
            return True

    async def _on_trigger_move_pose(self, request, response):
        self.node.get_logger().info("Trigger received: move_pose")
        try:
            ok = await self.move_demo()
            response.success = bool(ok)
            response.message = "move_pose done" if ok else "move_pose failed"
        except Exception as e:
            response.success = False
            response.message = error_to_str(e)
        return response


class ArmClientNode(Node):
    def __init__(self):
        super().__init__("arm_client_node")
        self._client = self.create_client(Trigger, "move_pose")
        self.end_event = threading.Event()

    def send_request(self):
        if not self._client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service move_pose is not available")
            return

        req = Trigger.Request()
        future = self._client.call_async(req)

        def _on_done(fut):
            try:
                res = fut.result()
                self.get_logger().info(
                    f"response: success={res.success}, message={res.message}"
                )
            except Exception as e:
                self.get_logger().error(f"client callback exception: {e}")
            self.end_event.set()

        future.add_done_callback(_on_done)


def main():
    rclpy.init()

    server_node = Node('arm_sever')
    arm_wrapper = ArmWrapper(server_node)
    client_node = ArmClientNode()

    executor_server = MultiThreadedExecutor()
    executor_server.add_node(server_node)

    executor_client = SingleThreadedExecutor()
    executor_client.add_node(client_node)

    service_thread = threading.Thread(target=executor_server.spin, daemon=True)
    service_thread.start()

    client_thread = threading.Thread(target=executor_client.spin, daemon=True)
    client_thread.start()

    try:
        while not arm_wrapper._joint_state_event.is_set():
            time.sleep(0.1)

        client_node.send_request()

        while not client_node.end_event.is_set():
            time.sleep(0.1)

    finally:
        USE_GRIPPER = True
        arm_wrapper.gripper.closing()
        executor_server.shutdown()
        executor_client.shutdown()
        service_thread.join(timeout=2.0)
        client_thread.join(timeout=2.0)
        server_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
