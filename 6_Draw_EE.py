#! 하트,별,세모 다 그릴수 있음. 별은 완벽하지는않다.
# 이제 reset잘됨

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension
from pxr import UsdGeom
import os
import json

from controllers.rmpflow_controller import RMPFlowController
from tasks.follow_target import FollowTarget
from franka import FR3  # Make sure franka.py exports the FR3 class

import isaacsim.core.api.tasks as tasks
import numpy as np
import carb
import random
from typing import List, Optional, Dict

# Debug drawing extension
enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw

script_path = os.path.abspath(__file__)
json_path = os.path.dirname(script_path)
json_path += '/korean.json'
with open(json_path) as f:
    data = json.load(f)
    character_path = data['characters']

def find_paths_by_name(character_name):
    for character in data['characters']:
        if character['name'] == character_name:
            return character['path']
    return None

def get_coordinate(i):
    return [(i['start'][0], i['start'][1], i['start'][2]), 
            (i['end'][0], i['end'][1], i['end'][2])]

# Task 클래스 수정
class FrankaRobotTask(tasks.FollowTarget):
    """여러 개의 FR3 로봇을 생성하여 각자 자신만의 타겟을 따라가도록 제어하는 Task"""

    def __init__(
        self,
        name: str = "fr3_task",
        robot_num: int = 1,  # 여러 개의 로봇을 생성할 수 있도록 개수 추가
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ):
        # 기본 타겟을 생성하기 위한 초기 값 (첫 번째 로봇용)
        if target_position is None:
            target_position = np.array([0.5, 0.0, 0.5])
            
        super().__init__(
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        
        self._robots = []  # 여러 개의 로봇을 저장할 리스트
        self._targets = []  # 여러 개의 타겟을 저장할 리스트
        self.robot_num = robot_num
        self.robot_names = []  # 로봇 이름을 저장할 리스트
        self.target_names = []  # 타겟 이름을 저장할 리스트
        return

    def set_robot(self) -> "List[FR3]":
        """여러 개의 FR3 로봇을 생성하며, 1m 간격으로 배치"""
        robots = []
        for i in range(self.robot_num):
            robot_prim_path = find_unique_string_name(
                initial_name=f"/World/FR3_{i}",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            robot_name = find_unique_string_name(
                initial_name=f"my_fr3_{i}",
                is_unique_fn=lambda x: not self.scene.object_exists(x),
            )
            
            # 로봇을 y축으로 1m씩 간격을 두고 배치
            robot = FR3(prim_path=robot_prim_path, name=robot_name, position=np.array([0, 1 * i, 0]))
            robots.append(robot)
            self.robot_names.append(robot_name)
        return robots

    def set_up_scene(self, scene: Scene) -> None:
        """씬 설정 및 여러 개의 로봇, 타겟 추가"""
        self._scene = scene
        assets_root_path = get_assets_root_path()
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Room/simple_room.usd",
            prim_path="/World/SimpleRoom",
        )

        if self._target_orientation is None:
            self._target_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))

        # 여러 개의 로봇 생성
        self._robots = self.set_robot()
        for robot in self._robots:
            scene.add(robot)
            self._task_objects[robot.name] = robot
            
        # 각 로봇마다 타겟 생성
        for i, robot in enumerate(self._robots):
            target_prim_path = find_unique_string_name(
                initial_name=f"/World/TargetCube_{i}",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            target_name = find_unique_string_name(
                initial_name=f"target_{i}",
                is_unique_fn=lambda x: not self.scene.object_exists(x),
            )
            
            # 각 로봇 앞에 타겟 위치시키기 (x축 방향으로 0.5m 앞에)
            robot_position = robot.get_world_pose()[0]  # 로봇의 현재 위치 가져오기
            target_position = np.array([robot_position[0] + 0.5, robot_position[1], 0.5])
            
            # 타겟 생성 및 추가
            try:
                # 큐브 타겟 추가
                target_obj = self._scene.add_cube(
                    prim_path=target_prim_path,
                    name=target_name,
                    position=target_position,
                    orientation=self._target_orientation,
                    size=0.05,
                    color=np.array([0.9, 0.4, 0.3, 1.0]) if i % 2 == 0 else np.array([0.3, 0.6, 0.9, 1.0])
                )
                
                # 타겟을 task_objects에 추가
                self._task_objects[target_name] = target_obj
                
                # 첫 번째 로봇의 타겟인 경우, 기본 타겟으로 설정 (FollowTarget 호환성)
                # main 함수 내 엔드 이펙터 궤적 그리기 부분 교체

                # 엔드 이펙터 궤적 그리기 추가
                if i == 0:
                    drawing_obj = shape_drawings[target_name]
                    if isinstance(drawing_obj, KoreanCharacterDrawing):
                        # 로봇 위치 가져오기
                        robot_pos = franka.get_world_pose()[0]
                        
                        # 현재 그리고 있는 획과 관련된 정보 확인
                        current_stroke_idx = drawing_obj.stroke_index
                        progress = drawing_obj.stroke_progress
                        is_drawing = drawing_obj.is_drawing
                        strokes = drawing_obj.strokes
                        
                        # 기본 오프셋 - 로봇 팔 앞쪽
                        offset_x = 0.3  # 로봇 앞쪽으로
                        draw_pos = robot_pos + np.array([offset_x, 0, 0])
                        
                        if is_drawing and current_stroke_idx < len(strokes):
                            # 현재 그리는 획 가져오기
                            current_stroke = strokes[current_stroke_idx]
                            
                            if len(current_stroke) >= 2:
                                # 획의 시작점과 끝점
                                start_point = np.array(current_stroke[0])
                                end_point = np.array(current_stroke[1])
                                
                                # 현재 진행 위치 계산
                                current_point = start_point + (end_point - start_point) * progress
                                
                                # 스케일링 및 로봇 위치에 적용
                                scale = 1.0  # 더 큰 스케일로 조정
                                # Y와 Z 좌표만 변경 (X는 고정)
                                draw_pos[1] = robot_pos[1] + current_point[1] * scale
                                draw_pos[2] = robot_pos[2] + current_point[2] * scale
                        
                        # 색상 설정 - 파란색 계열
                        if getattr(ee_drawer1, 'colors', None) is not None:
                            ee_drawer1.colors = (0.0, 0.7, 1.0, 1.0)  # 고정된 밝은 파란색
                        
                        # 엔드 이펙터 궤적 업데이트
                        ee_drawer1.update_drawing(draw_pos, draw_offset=[0, 0, 0])  # 오프셋 없이 직접 그리기








                
                print(f"Successfully created target: {target_name} at position {target_position}")
            except Exception as e:
                print(f"Error creating target {target_name}: {e}")
            
            self.target_names.append(target_name)
            
            self._targets.append(
                {
                    "prim_path": target_prim_path,
                    "name": target_name,
                    "position": target_position,
                    "orientation": self._target_orientation,
                }
            )


        # 첫 번째 로봇을 self._robot으로 설정 (기존 인터페이스와의 호환성 유지)
        self._robot = self._robots[0] if self._robots else None

        self._move_task_objects_to_their_frame()
        return

    def get_robots(self) -> "List[FR3]":
        """여러 개의 로봇 객체 반환"""
        return self._robots
        
    def get_target_names(self) -> List[str]:
        """모든 타겟 이름 반환"""
        return self.target_names

    def get_observations(self) -> Dict:
        """모든 타겟의 관찰 정보 반환"""
        observations = {}
        
        # 상위 클래스의 get_observations 호출 대신 직접 구현
        if self._target is not None:
            target_position, target_orientation = self._target.get_world_pose()
            observations[self._target_name] = {
                "position": target_position,
                "orientation": target_orientation,
            }
        
        # 추가 로봇 정보 설정
        for i, robot_name in enumerate(self.robot_names):
            # 기본 로봇은 이미 처리함
            if self._robot is not None and robot_name == self.robot_names:
                continue
                
            if robot_name in self._task_objects:
                target_obj = self._task_objects[robot_name]
                joints_state = target_obj.get_joints_state()
                
                observations[robot_name] = {
                    "joint_positions": joints_state.positions,
                    "joint_velocities": joints_state.velocities,
                }
        
        # 추가 타겟 정보 설정
        for i, target_name in enumerate(self.target_names):
            # 기본 타겟은 이미 처리함
            if self._target is not None and target_name == self._target_name:
                continue
                
            if target_name in self._task_objects:
                target_obj = self._task_objects[target_name]
                target_position, target_orientation = target_obj.get_world_pose()
                
                observations[target_name] = {
                    "position": target_position,
                    "orientation": target_orientation,
                }
            else:
                # 타겟이 없는 경우 기본 값을 반환
                robot_position = self._robots[i].get_world_pose()[0] if i < len(self._robots) else np.array([0, i, 0])
                default_position = robot_position + np.array([0.5, 0, 0.5])
                
                observations[target_name] = {
                    "position": default_position,
                    "orientation": euler_angles_to_quat(np.array([-np.pi, 0, np.pi])) if self._target_orientation is None else self._target_orientation,
                }
                
        return observations
# 기존 TrajectoryDrawer 클래스를 개선한 새 버전
class EnhancedTrajectoryDrawer:
    """향상된 궤적 그리기 클래스"""

    def __init__(self, name="enhanced_drawer", color=(0.0, 0.7, 1.0, 1.0)):
        self.name = name
        self.draw = None
        self.point_list = []
        self.color = color  # 고정된 색상 사용
        self.line_thickness = 4
        self.is_active = True
        self.max_points = 10000  # 더 많은 포인트를 저장해 더 부드러운 라인 그리기

    def initialize(self):
        """디버그 드로우 인터페이스 초기화"""
        self.draw = _debug_draw.acquire_debug_draw_interface()
        if not self.draw:
            print(f"Warning: Failed to acquire debug draw interface for {self.name}")
        else:
            print(f"Successfully initialized {self.name} drawer")

    def update_drawing(self, position, color=None):
        """
        궤적 그리기 업데이트
        
        Args:
            position: 현재 그릴 위치 (numpy array or tuple)
            color: 선택적 색상 오버라이드
        """
        if not self.is_active or self.draw is None:
            return
            
        # 포인트 리스트에 현재 위치 추가
        pos_tuple = tuple(position)
        self.point_list.append(pos_tuple)
        
        # 사용할 색상 결정
        draw_color = color if color is not None else self.color
        
        # 라인 그리기 - 항상 전체 경로 다시 그림
        if len(self.point_list) > 1:
            self.draw.clear_lines()  # 이전 라인 지우기
            self.draw.draw_lines_spline(
                self.point_list, 
                draw_color, 
                self.line_thickness, 
                False
            )
        
        # 버퍼 크기 관리
        if len(self.point_list) > self.max_points:
            self.point_list = self.point_list[-self.max_points:]

    def reset_drawing(self):
        """궤적 초기화"""
        self.point_list = []
        if self.draw:
            self.draw.clear_lines()



# 엔드 이펙터 위치 계산을 위한 개선된 함수
def calculate_ee_position(robot, drawing_obj):
    """
    로봇의 현재 위치와 드로잉 정보를 기반으로 엔드 이펙터 위치 계산
    
    Args:
        robot: FR3 로봇 객체
        drawing_obj: KoreanCharacterDrawing 객체
    
    Returns:
        numpy array: 엔드 이펙터 위치
    """
    # 로봇 위치 가져오기
    robot_pos = robot.get_world_pose()[0]
    
    # 기본 위치 (로봇 앞쪽)
    base_pos = robot_pos + np.array([0.25, 0, 0])
    
    # 그리기 중이 아니면 기본 위치 반환
    if not drawing_obj.is_drawing:
        return base_pos
    
    # 현재 그리고 있는 획과 관련된 정보 확인
    current_stroke_idx = drawing_obj.stroke_index
    progress = drawing_obj.stroke_progress
    strokes = drawing_obj.strokes
    
    # 모든 획을 그렸으면 기본 위치 반환
    if current_stroke_idx >= len(strokes):
        return base_pos
    
    # 현재 그리는 획 가져오기
    current_stroke = strokes[current_stroke_idx]
    
    if len(current_stroke) < 2:
        return base_pos
    
    # 획의 시작점과 끝점
    start_point = np.array(current_stroke[0])
    end_point = np.array(current_stroke[1])
    
    # 현재 진행 위치 계산
    current_point = start_point + (end_point - start_point) * progress
    
    # 스케일링 및 로봇 위치에 적용
    scale = 1.2  # 더 큰 스케일로 조정
    # 로봇 위치에 현재 점의 Y, Z 좌표 적용 (X는 고정)
    ee_pos = base_pos.copy()
    ee_pos[1] += current_point[1] * scale
    ee_pos[2] += current_point[2] * scale
    
    return ee_pos



# Drawing 베이스 클래스 (모든 도형에 공통적인 기능)
class ShapeDrawing:
    """도형 그리기를 담당하는 기본 클래스"""

    def __init__(self, name="shape_drawing", color=(1.0, 0.5, 0.2, 1.0)):
        # Drawing variables
        self.name = name
        self.custom_timer = 0
        self._frame_counter = 0
        self.timer_speed = 1.0
        self.point_list = []
        self.draw = None
        # Default drawing settings
        self.draw_scale = 1.0
        self.draw_color = color  # Default color (RGBA)
        self.line_thickness = 5
        self.is_active = True
        
    def setup_post_load(self):
        """디버그 드로잉 인터페이스 초기화"""
        self.draw = _debug_draw.acquire_debug_draw_interface()
        return

    def reset_drawing(self):
        """그리기 데이터 초기화"""
        self.custom_timer = 0
        self._frame_counter = 0
        self.point_list = []
        if self.draw:
            self.draw.clear_lines()

    def draw_shape(self, observations, target_name):
        """현재 설정을 기반으로 도형 계산 및 그리기"""
        # 이 메서드는 자식 클래스에서 재정의해야 함
        raise NotImplementedError("Subclasses must implement draw_shape")

  

class KoreanCharacterDrawing(ShapeDrawing):
    """한글 글자를 정면으로 그리기를 담당하는 클래스 (x축 고정, y와 z 좌표 이동)"""
    
    def __init__(self, name="korean_character_drawing", color=(1.0, 0.5, 0.2, 1.0), character = '', robot = "", controller = None):
        super().__init__(name=name, color=color)
        self.stroke_index = 0  # 현재 그리는 획의 인덱스
        self.stroke_progress = 0.0  # 현재 획의 진행 상태 (0.0 ~ 1.0)
        self.stroke_speed = 0.01  # 획 그리기 속도
        self.completed_strokes = []  # 완성된 획들의 포인트 저장
        self.start_time = None  # 그리기 시작 시간 저장
        self.delay_seconds = 2.0  # 그리기 시작 전 대기 시간 (2초)
        self.is_drawing = False  # 그리기 진행 중인지 여부
        self.initial_position = None  # 초기 위치 저장
        self.strokes = []          # 글자의 각 획 정의
        self.character = character    # 그리는 글자
        self.robot = robot  # 그리는 로봇
        self.controller = controller
        
        # 펜 상태 변수 추가
        self.pen_up = True  # 처음에는 펜이に들어올려진 상태
        self.pen_x_offset = 0.03  # 펜이 들어올려질 때 z축 방향 오프셋 (3cm)
        self.transition_state = "none"  # 'none', 'lifting', 'moving', 'lowering' 상태
        self.transition_progress = 0.0  # 전환 동작의 진행 상태
        self.transition_speed = 0.05  # 전환 동작 속도
        self.previous_position = None  # 이전 위치 저장
        self.next_position = None  # 다음 위치 저장
        
        # 더 먼 거리에서 그리기 위한 x축 오프셋 증가
        self.x_offset = 0.4  # 기존 0.2에서 0.4로 증가

        for i in find_paths_by_name(self.character):
            self.strokes.append(get_coordinate(i))
            print(get_coordinate(i))

    def draw_shape(self, observations, target_name):
        """현재 설정을 기반으로 글자를 정면으로 그리기"""
        try:
            if not self.is_active:
                if target_name in observations and "position" in observations[target_name]:
                    return observations[target_name]["position"]
                else:
                    # 타겟 정보가 없는 경우 기본 위치 반환
                    print(f"Target {target_name} position not found, using default")
                    return np.array([0.5, 0.5, 0.5])  # 기본 위치
                
            # 관찰에서 타겟 정보 확인 및 오류 처리
            if target_name not in observations:
                print(f"Target {target_name} not in observations: {list(observations.keys())}")
                return np.array([0.5, 0.5, 0.5])  # 기본 위치
                
            if (target_name in observations) & ("position" not in observations[target_name]):
                print(f"Position not found for {target_name}. Available: {list(observations[target_name].keys())}")
                return np.array([0.5, 0.5, 0.5])  # 기본 위치
                
            # 원본 좌표 저장
            original_position = observations[target_name]["position"]
            
            # 시작 시간이 없으면 현재 시간으로 초기화
            if self.start_time is None:
                import time
                self.start_time = time.time()
                self.initial_position = original_position.copy()
                
                # 그리퍼 방향 설정은 외부에서 처리해야 함 (draw_shape에서는 위치만 반환)
                return original_position  # 첫 프레임에서는 이동하지 않음
            
            # 현재 시간과 시작 시간의 차이 계산
            import time
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 지연 시간이 지나지 않았으면 정면 그리기 시작 위치로 이동
            if elapsed_time < self.delay_seconds:
                # 지연 시간 동안 그리기 위치로 서서히 이동
                # 정면으로 글자를 그리기 위해 x 좌표는 초기값보다 앞으로 이동
                target_position = np.array([self.initial_position[0] + self.x_offset, self.initial_position[1], self.initial_position[2]])
                progress = elapsed_time / self.delay_seconds
                new_pos = self.initial_position + (target_position - self.initial_position) * progress
                return new_pos
            
            # 지연 시간이 지났고 아직 그리기를 시작하지 않았으면 그리기 시작
            if not self.is_drawing:
                self.is_drawing = True
                self.pen_up = True  # 시작 시 펜 들기
                self.transition_state = "lowering"  # 첫 획을 위해 펜 내리기 시작
                self.transition_progress = 0.0
                print("Drawing started after delay")
            
            # 모든 획을 다 그렸는지 확인
            if self.stroke_index >= len(self.strokes):
                # 다 그렸으면 완성된 글자 유지
                self._draw_completed_strokes(original_position)
                # 완성 후에는 펜을 살짝 들어올림
                if not self.pen_up:
                    self.pen_up = True
                    self.transition_state = "lifting"
                    self.transition_progress = 0.0
                
                # 펜이 들어올려진 상태에서 기본 위치로 유지
                new_pos = np.array([self.initial_position[0] + self.x_offset, self.initial_position[1], self.initial_position[2]])
                if self.pen_up:
                    new_pos[0] += self.pen_x_offset  # z축 방향으로 들어올림
                return new_pos
            
            # 기본 위치 계산
            base_position = np.array([self.initial_position[0] + self.x_offset, self.initial_position[1], self.initial_position[2]])
            
            # 전환 상태에 따른 처리 (펜 들기/내리기/이동)
            if self.transition_state != "none":
                # 현재 그리는 획 가져오기
                current_stroke = self.strokes[self.stroke_index]
                
                # 전환 상태에 따른 처리
                if self.transition_state == "lifting":
                    # 획이 끝나고 펜 들어올리기
                    if self.previous_position is None:
                        # 현재 위치가 없으면 마지막 획의 끝점 사용
                        if len(current_stroke) >= 2:
                            end_point = np.array(current_stroke[1]) if len(current_stroke) == 2 else np.array(current_stroke[-1])
                            scale_factor = 0.8 * self.draw_scale
                            end_point = end_point * scale_factor
                            self.previous_position = base_position + np.array([0, end_point[1], end_point[2]])
                        else:
                            self.previous_position = base_position
                    
                    # 펜을 들어올리는 위치 계산
                    lift_position = self.previous_position.copy()
                    lift_position[0] += self.pen_x_offset
                    
                    # 진행 상태에 따라 현재 위치 계산
                    new_pos = self.previous_position + (lift_position - self.previous_position) * self.transition_progress
                    
                    # 전환 상태 업데이트
                    self.transition_progress += self.transition_speed
                    if self.transition_progress >= 1.0:
                        self.transition_state = "moving"  # 다음 상태: 이동
                        self.transition_progress = 0.0
                        # 다음 획의 시작점 계산
                        if self.stroke_index + 1 < len(self.strokes):
                            next_stroke = self.strokes[self.stroke_index + 1]
                            start_point = np.array(next_stroke[0])
                            scale_factor = 0.8 * self.draw_scale
                            start_point = start_point * scale_factor
                            self.next_position = base_position + np.array([start_point[0], start_point[1], 0])
                            self.next_position[0] += self.pen_x_offset  # 펜을 든 상태로 이동
                    
                    return new_pos
                
                elif self.transition_state == "moving":
                    # 펜을 들고 다음 획의 시작점으로 이동
                    if self.next_position is None or self.previous_position is None:
                        # 다음 획의 시작점 계산
                        if self.stroke_index + 1 < len(self.strokes):
                            next_stroke = self.strokes[self.stroke_index + 1]
                            start_point = np.array(next_stroke[0])
                            scale_factor = 0.8 * self.draw_scale
                            start_point = start_point * scale_factor
                            self.next_position = base_position + np.array([start_point[0], start_point[1], 0])
                            self.next_position[0] += self.pen_x_offset  # 펜을 든 상태로 이동
                        else:
                            # 모든 획을 그린 경우
                            self.next_position = base_position.copy()
                            self.next_position[0] += self.pen_x_offset
                            
                        # 이전 위치가 없으면 현재 위치 사용
                        if self.previous_position is None:
                            self.previous_position = base_position.copy()
                            self.previous_position[0] += self.pen_x_offset
                    
                    # 진행 상태에 따라 현재 위치 계산
                    new_pos = self.previous_position + (self.next_position - self.previous_position) * self.transition_progress
                    
                    # 전환 상태 업데이트
                    self.transition_progress += self.transition_speed
                    if self.transition_progress >= 1.0:
                        # 다음 획으로 이동
                        self.transition_state = "lowering" if self.stroke_index + 1 < len(self.strokes) else "none"
                        self.transition_progress = 0.0
                        
                        # 다음 획으로 이동 또는 완료
                        if self.stroke_index + 1 < len(self.strokes):
                            self.stroke_index += 1
                            self.stroke_progress = 0.0
                            self.point_list = []
                            self.previous_position = self.next_position.copy()
                    
                    return new_pos
                
                elif self.transition_state == "lowering":
                    # 펜을 내려서 종이에 닿게 하기
                    if self.next_position is None:
                        # 현재 획의 시작점 계산
                        current_stroke = self.strokes[self.stroke_index]
                        start_point = np.array(current_stroke[0])
                        scale_factor = 0.8 * self.draw_scale
                        start_point = start_point * scale_factor
                        
                        # 기본 위치에 현재 획의 시작점 적용
                        self.next_position = base_position + np.array([start_point[0], start_point[1], 0])
                        
                        # 이전 위치가 없으면 펜을 들고 있는 위치 사용
                        if self.previous_position is None:
                            self.previous_position = self.next_position.copy()
                            self.previous_position[0] += self.pen_x_offset
                    
                    # 진행 상태에 따라 현재 위치 계산
                    new_pos = self.previous_position + (self.next_position - self.previous_position) * self.transition_progress
                    
                    # 전환 상태 업데이트
                    self.transition_progress += self.transition_speed
                    if self.transition_progress >= 1.0:
                        self.transition_state = "none"  # 다음 상태: 그리기
                        self.transition_progress = 0.0
                        self.pen_up = False  # 펜을 내려서 그리기 시작
                        
                        # 그리기 시작 위치 설정
                        self.previous_position = self.next_position.copy()
                        self.next_position = None
                    
                    return new_pos
            
            # 현재 그리는 획 가져오기
            current_stroke = self.strokes[self.stroke_index]
            print(f"!!Stroke Starting Point: {current_stroke[0]}")
            
            # 획 타입에 따라 처리 (직선, 원, 또는 곡선)
            if len(current_stroke) == 2:  # 직선
                start_point = np.array(current_stroke[0])
                end_point = np.array(current_stroke[1])
                
                # 획 진행에 따른 현재 위치 계산
                current_point = start_point + (end_point - start_point) * self.stroke_progress
            elif len(current_stroke) > 4:  # 원 또는 복잡한 곡선
                # 원을 위한 점 사이 보간
                segment_count = len(current_stroke) - 1
                segment_index = int(self.stroke_progress * segment_count)
                segment_progress = (self.stroke_progress * segment_count) - segment_index
                
                if segment_index >= segment_count:
                    segment_index = segment_count - 1
                    segment_progress = 1.0
                
                start_point = np.array(current_stroke[segment_index])
                end_point = np.array(current_stroke[segment_index + 1])
                
                # 선형 보간으로 현재 위치 계산
                current_point = start_point + (end_point - start_point) * segment_progress
            else:  # 곡선 (베지어 곡선으로 처리)
                t = self.stroke_progress
                if len(current_stroke) == 4:  # 3차 베지어 곡선
                    p0 = np.array(current_stroke[0])
                    p1 = np.array(current_stroke[1])
                    p2 = np.array(current_stroke[2])
                    p3 = np.array(current_stroke[3])
                    
                    # 3차 베지어 곡선 계산
                    current_point = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                else:
                    # 지원하지 않는 획 타입
                    print(f"Unsupported stroke type with {len(current_stroke)} points")
                    current_point = np.array(current_stroke[0])
            
            # 크기 조정
            scale_factor = 0.8 * self.draw_scale
            current_point = current_point * scale_factor
            
            # 정면으로 그리기 위한 좌표 변환
            # 기본 위치에서 x축 방향으로 offset만큼 이동한 위치를 기준으로 y, z 좌표 변경
            base_position = np.array([self.initial_position[0] + self.x_offset, self.initial_position[1], self.initial_position[2]])
            
            # 정면에서 그리기 위한 현재 위치 계산
            new_pos = base_position + np.array([0, current_point[1], current_point[2]])
            
            # 획 진행 상태 업데이트
            self.stroke_progress += self.stroke_speed * self.timer_speed
            
            # 포인트 기록 및 그리기
            self._frame_counter += 1
            if self._frame_counter % 3 == 0:  # 포인트 샘플링 빈도
                draw_pos = new_pos.copy()
                
                # 현재 획의 포인트 리스트 관리
                if len(self.point_list) > 100:  # 포인트 제한
                    del self.point_list[0]
                self.point_list.append(tuple(draw_pos))
                
                # 라인 그리기
                if len(self.point_list) > 1 and self.draw:
                    self.draw.draw_lines_spline(
                        self.point_list, self.draw_color, self.line_thickness, False
                    )
            
            # 획이 완성되었는지 확인
            if self.stroke_progress >= 1.0:
                # 완성된 획 저장
                self.completed_strokes.append(self.point_list.copy())
                # 다음 획으로 넘어가기
                self.stroke_index += 1
                self.stroke_progress = 0.0
                self.point_list = []
                
                # 이전 획들을 모두 다시 그리기 (누적 효과)
                self._draw_completed_strokes(original_position)
            if self.stroke_progress == 0.0:
                actions = self.controller.forward(
                    target_end_effector_position=current_stroke[0],
                )
                self.robot.get_articulation_controller().apply_action(actions)
            
            return new_pos
            
        except Exception as e:
            print(f"Error in draw_korean_character: {e}")
            # 오류 발생 시 기본 위치 반환
            if target_name in observations and "position" in observations[target_name]:
                return observations[target_name]["position"]
            else:
                return np.array([0.5, 0.5, 0.5])  # 기본 위치
    
    def _draw_completed_strokes(self, original_position):
        """완성된 획들을 모두 그리는 함수"""
        if not self.draw:
            return
            
        # 각 완성된 획들 그리기
        for stroke_points in self.completed_strokes:
            if len(stroke_points) > 1:
                self.draw.draw_lines_spline(
                    stroke_points, self.draw_color, self.line_thickness, False
                )
    
    def get_front_facing_orientation(self):
        """정면을 향하는 그리퍼의 방향값 반환 (쿼터니언)"""
        # 그리퍼가 정면을 향하도록 하는 쿼터니언 값
        import numpy as np
        
        # 예시: 정면을 바라보는 쿼터니언 (w, x, y, z)
        # x축 기준 90도 회전하여 정면을 바라보게 함

        # return np.array([0.7071, 0.7071, 0.0, 0.0])

        return np.array([0.7071, 0.0, 0.7071, 0.0])

    def reset(self):
        """그리기 상태 초기화"""
        # 화면에서 라인을 명시적으로 지우기
        if self.draw:
            self.draw.clear_lines()
        
        # Call parent's reset_drawing method correctly
        super().reset_drawing()
        
        # Reset all state variables
        self.stroke_index = 0
        self.stroke_progress = 0.0
        self.completed_strokes = []
        self.point_list = []
        self.start_time = None
        self.is_drawing = False
        self.initial_position = None
        
        # Reset any controller state if needed
        if self.controller:
            self.controller.reset()





# 하트 모양 그리기 클래스
  # 기본 위치

# 별 모양 그리기 클래스


# 삼각형 모양 그리기 클래스


def main():
    # 월드 생성
    my_world = World(stage_units_in_meters=1.0)

    # 카메라 뷰 설정
    eye_position = [3.0, 1.5, 1.5]
    target_position = [0.5, 1.0, 0.5]
    camera_prim_path = "/OmniverseKit_Persp"
    set_camera_view(
        eye=eye_position, target=target_position, camera_prim_path=camera_prim_path
    )

    # Task 생성 (로봇 개수를 설정)
    robot_num = 1  # 생성할 로봇 개수
    my_task = FrankaRobotTask(
        name="follow_target_task", 
        robot_num=robot_num
    )
    my_world.add_task(my_task)
    
    # 월드 리셋 및 1 프레임 스텝
    my_world.reset()
    my_world.step(render=True)
    
    # 타겟 이름 가져오기
    target_names = my_task.get_target_names()
    print(f"Target names: {target_names}")
    
    # 관찰 정보 디버깅
    try:
        observations = my_world.get_observations()
        print(f"Available keys in observations: {list(observations.keys())}")
        
        # 각 타겟의 위치 출력
        for target_name in target_names:
            if target_name in observations:
                print(f"Target {target_name} position: {observations[target_name]['position']}")
            else:
                print(f"Target {target_name} not found in observations")
    except Exception as e:
        print(f"Error getting observations: {e}")
    
    # 각 로봇마다 다른 모양의 드로잉 객체 생성
    shape_drawings = {}
    colors = [
        (1.0, 0.5, 0.2, 1.0),  # 오렌지 (획 그리기)
        (0.0, 0.7, 1.0, 1.0),  # 하늘색 (엔드 이펙터 첫번째)
        (1.0, 0.3, 0.5, 1.0),  # 핑크 (엔드 이펙터 두번째) 
    ]

    # 엔드 이펙터 궤적을 위한 드로어 초기화 (향상된 버전 사용)
    ee_drawer = EnhancedTrajectoryDrawer(name="ee_drawer", color=colors[1])
    ee_drawer.initialize()
    
    # 여러 개의 로봇 가져오기
    my_frankas = my_task.get_robots()

    # 각 로봇에 대해 컨트롤러 설정
    controllers = []
    for franka in my_frankas:
        controllers.append(RMPFlowController(name=f"controller_{franka.name}", robot_articulation=franka))

    # 로봇 별로 다른 모양 할당
    for i, target_name in enumerate(target_names):
        if i >= len(my_frankas):
            print(f"Warning: Not enough robots for target {target_name}")
            continue
            
        # 첫 번째 로봇 (한글 ㄹ)
        drawing = KoreanCharacterDrawing(
            name=f"Korean_{target_name}", 
            color=colors[0],  # 오렌지색 (획)
            character='ㄹ',
            robot=my_frankas[i],
            controller=controllers[i]
        )
        
        # 설정
        drawing.setup_post_load()
        drawing.stroke_speed = 0.005  # 느리게 그리기
        shape_drawings[target_name] = drawing
    
    # 메인 시뮬레이션 루프
    reset_needed = False
    frame_counter = 0
    while simulation_app.is_running():
        try:
            my_world.step(render=True)
            frame_counter += 1

            if my_world.is_stopped() and not reset_needed:
                reset_needed = True

            if my_world.is_playing():
                if reset_needed:
                    # 모든 드로잉 명시적 초기화
                    for drawing in shape_drawings.values():
                        if drawing.draw:
                            drawing.draw.clear_lines()
                    
                    # 엔드 이펙터 드로어 초기화
                    ee_drawer.reset_drawing()
                    
                    # 월드 초기화
                    my_world.reset()
                    
                    # 컨트롤러 초기화
                    for controller in controllers:
                        controller.reset()
                    
                    # 드로잉 객체 초기화
                    for drawing in shape_drawings.values():
                        drawing.reset_drawing()
                        if isinstance(drawing, KoreanCharacterDrawing):
                            drawing.reset()
                    
                    # 드로잉 인터페이스 재초기화
                    for drawing in shape_drawings.values():
                        drawing.setup_post_load()
                    
                    # 엔드 이펙터 드로어 재초기화
                    ee_drawer.initialize()
                    
                    reset_needed = False
                    frame_counter = 0
                    
                    # 초기화를 위한 여유 시간
                    for _ in range(5):
                        my_world.step(render=True)

                # 관찰 정보 가져오기
                observations = my_world.get_observations()
                
                # 주기적 디버깅 정보
                if frame_counter % 100 == 0:
                    print(f"Simulation time: {my_world.current_time:.2f}, Frame: {frame_counter}")
                
                # 각 로봇마다 처리
                # In the main loop, replace the relevant part with:
                for i, (franka, target_name) in enumerate(zip(my_frankas, target_names)):
                    try:
                        if target_name not in shape_drawings:
                            continue
            
                        drawing_obj = shape_drawings[target_name]
        
                        # Calculate new target position based on drawing
                        target_pos = drawing_obj.draw_shape(observations, target_name)
        
                        # Get orientation
                        orientation = observations[target_name]["orientation"]
                        if isinstance(drawing_obj, KoreanCharacterDrawing):
                            orientation = drawing_obj.get_front_facing_orientation()

                        # Apply action to robot - using the target position directly
                        actions = controllers[i].forward(
                            target_end_effector_position=target_pos,
                            target_end_effector_orientation=orientation,
                        )
                        franka.get_articulation_controller().apply_action(actions)
        
                        # Step the simulation a bit to let controller take effect
                        # This is optional and may require adjustment based on your simulation setup

                        # Now get the ACTUAL end effector position
                        ee_pos = franka.end_effector.get_world_pose()[0]  # This gets the actual position
        
                        # Use the actual end effector position for drawing
                        if ee_pos[0] >= 0.87 & ee_pos[0] <= 0.85:
                            print(f"ee_pos[0]: {ee_pos[0]}")
                            ee_drawer.update_drawing(ee_pos)
        
                        # Debug info
                        if frame_counter % 100 == 0:
                            print(f"Robot {i}: Stroke {drawing_obj.stroke_index}/{len(drawing_obj.strokes)}, "
                                  f"Progress: {drawing_obj.stroke_progress:.2f}, EE Pos: {ee_pos}")
                    except Exception as e:
                        print(f"Error for robot {i}, target {target_name}: {e}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            print(f"Error in simulation loop: {e}")
            import traceback
            traceback.print_exc()

    simulation_app.close()

if __name__ == "__main__":
    main()
