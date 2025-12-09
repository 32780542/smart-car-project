# Description: This is a python script for the IDC driver model.
# The script is used to control the vehicle in the simulation environment.

from typing import List, Tuple, Dict, Optional, Any
import math
from shapely.geometry import LineString
from math import cos, sin
import numpy as np

from gops.rule_model import mpc_controller
from gops.rule_model import path_plan_frenet
from dataclasses import dataclass, asdict


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

@dataclass
class VehicleState:
    time_step: int
    id: str
    x: float
    y: float
    phi: float
    lon_spd: float
    lat_spd: float
    yaw_spd: float
    dis_to_lane_end: float
    width: float
    length: float

    best_ref_lane: str
    between_ref_lane: str
    ego_index: int
    ref_index: int
    same_lane_flag: bool
    is_adjacent_flag: bool
    is_one_between_flag: bool

    lane_id: str
    link_id: str
    junction_id: str
    position_type: int
    target_path_name: str
    forced_lane_change_flag: bool
    acc: float
    steer: float

    plan_lon_spd:float
    plan_lat_spd: float
    plan_lat_acc: float
    plan_yaw: float
    target_path_point: str
    is_plan: int

    x_sol: None
    u_sol: None
    slack_sol: None

    def to_dict(self):
        return asdict(self)


class Agent(object):
    def __init__(self, id, base_info, veh_config, cosim_config):
        self.id = id
        self.length = base_info.get('length')
        self.width = base_info.get('width')

        self.lon_spd = 0
        self.lat_spd = 0
        self.yaw_spd = 0

        self.x = 0
        self.y = 0
        self.z = 0
        self.phi = 0
        self.lane_id = None
        self.link_id = None
        self.junction_id = None
        self.dis_to_lane_end = 0
        self.position_type = 0  # 0 Unknown, 1 lane, 2 intersection, 3 outside the road

        self.front_veh_dis_list = []
        self.rear_veh_dis_list = []

        self.reference_path = None

        self.forced_lane_change_dis = veh_config["forced_lane_change_dis"]  # The mandatory lane change distance before the intersection
        self.decision_interval = veh_config["decision_interval"]  # Lane change interval
        self.ref_speed = veh_config["ref_spd"]  # vehicle reference speed
        self.ref_speed_intersection = veh_config["ref_spd_inter"]  # vehicle reference speed near intersection
        self.weight_idm = veh_config["weight_idm"]  # idm weight
        self.weight_rear = veh_config["weight_rear"]  # safety weight

        self.decision_step_index = 100
        self.decision_step_index_junction = 100

        # controller
        self.acc = 0
        self.steer = 0

        # planner
        self.plan_pos = None
        self.plan_yaw = 0
        self.plan_lon_spd = 0
        self.plan_lat_spd = 0
        self.plan_lat_acc = 0
        self.R = 0
        self.yaw_rate = 0
        self.lon_acc = 0

        # Some flags for judgment
        self.best_ref_lane = None
        self.between_ref_lane = None
        self.ego_index = 0
        self.ref_index = 0
        self.same_lane_flag = None
        self.is_adjacent_flag = None
        self.is_one_between_flag = None
        self.lane_position = None

        self.front_veh_info_lane = None
        self.global_lane_id_list = None
        self.global_junction_id_list = None
        self.forced_lane_change_flag = None
        self.lane_id_types = None
        self.ref_path_valid = None

        self.last_delta_acc = 0
        self.last_delta_steer = 0
        self.final_tac_lane_index = 0

        self.delta_acc = []
        self.delta_steer = []

        # Calculate the shortest path at the intersection
        self.lane_idxes = []
        self.min_length_index = 0 # Shortest lane index
        # Initialize the stored vehicle information
        self.history_info = {}

        # The controller solves and optimizes the relevant variables
        self.x_sol = None
        self.u_sol = None
        self.slack_sol = None

    def update(self, all_vehicle_info, all_veh_moving_info, all_veh_base_info, step, position_info, moving_info, reference_path, dis_to_link_boundary, action, movement_signal, signal_countdown):
        """
        Args:
        all_vehicle_info (dict):
            Global positional information of all vehicles.
            Expected structure (from simulator):
                {
                    veh_id: {
                        "point": {"x": float, "y": float, "z": float},
                        "phi": float,              # heading [rad]
                        "lane_id": str,
                        "link_id": str,
                        "junction_id": str,
                        "dis_to_lane_end": float,  # distance to lane end [m]
                        "position_type": int,      # 0 unknown, 1 lane, 2 junction, 3 off-road
                        ...
                    },
                    ...
                }

        all_veh_moving_info (dict):
            Global kinematic information of all vehicles.
            Expected structure:
                {
                    "moving_info_dict": {
                        veh_id: {
                            "u": float,  # longitudinal speed [m/s]
                            "v": float,  # lateral speed [m/s]
                            "w": float,  # yaw rate [rad/s]
                            ...
                        },
                        ...
                    },
                    ...
                }

        all_veh_base_info (dict):
            Global geometry / base info of all vehicles.
            Expected structure:
                {
                    "info_dict": {
                        veh_id: {
                            "base_info": {
                                "length": float,  # vehicle length [m]
                                "width": float,   # vehicle width [m]
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    ...
                }

        step (int):
            Current discrete simulation step index. Used as key in history_info.

        position_info (dict):
            Ego vehicle positional info at this step. Typically:
                {
                    "point": {"x": float, "y": float, "z": float},
                    "phi": float,              # heading [rad]
                    "lane_id": str,
                    "link_id": str,
                    "junction_id": str,
                    "dis_to_lane_end": float,  # distance to lane end [m]
                    "position_type": int,      # 0 unknown, 1 lane, 2 junction, 3 off-road
                    ...
                }

        moving_info (dict):
            Ego vehicle kinematic info at this step. Typically:
                {
                    "u": float,  # longitudinal speed [m/s]
                    "v": float,  # lateral speed [m/s]
                    "w": float,  # yaw rate [rad/s]
                    ...
                }

        reference_path (list):
            Ego reference lane/path set from the simulator, used for decision
            and planning. Each element is usually a dict like:
                {
                    "points": [
                        {"x": float, "y": float, "z": float},
                        ...
                    ],
                    "lane_ids":   list[str],
                    "lane_idxes": list[int],
                    "lane_types": list[str],
                    ...
                }

        dis_to_link_boundary (dict):
            Ego vehicle distances to the left and right boundaries of the current link.
            Expected structure:
                {
                    "left": float,   # distance to left boundary [m]
                    "right": float,  # distance to right boundary [m]
                }

        action (sequence[float]):
            Previous ego control increments (from last step), length 2:
                - action[0]: last longitudinal acceleration increment
                - action[1]: last steering increment
            Used as a warm-start / smoothness reference for the controller.

        movement_signal (int):
            Movement / traffic signal state associated with the next movement
            of the ego path (e.g. red(1)/green(2)/yellow(3) , or 0 when unavailable).
        
        signal_countdown (float):
            Countdown time (in seconds) for the current traffic signal phase.
            If unavailable, it can be set to 0.0.

        Returns:
            tuple[float, float]:
                (acc, steer)
                - acc:   longitudinal acceleration command for this step.
                - steer: steering command (steering angle or steering rate,
                        depending on controller convention).
        """
        # 1. updates ego vehicle state
        self.update_state(position_info, moving_info, reference_path, action, movement_signal)
        # 2. The decision updates the target path
        path_id, nearest_vehicles = self.decision(all_vehicle_info, all_veh_moving_info, all_veh_base_info, step)
        # 3. The planner updates the trajectory
        plan_x,plan_y,plan_yaw,plan_spd = self.planner(step, path_id, nearest_vehicles, all_vehicle_info, all_veh_moving_info, all_veh_base_info)
        # 4. The controller updates the next control quantity
        self.Controller(step, plan_y, plan_x, plan_yaw, plan_spd)

        self.history_info[step] = VehicleState(
            time_step=step,
            id=self.id,
            x=self.x,
            y=self.y,
            phi=self.phi,
            lon_spd=self.lon_spd,
            lat_spd=self.lat_spd,
            yaw_spd=self.yaw_spd,
            lane_id=self.lane_id,
            link_id=self.link_id,
            junction_id=self.junction_id,
            position_type=self.position_type,
            dis_to_lane_end=self.dis_to_lane_end,
            length=self.length,
            width=self.width,
            best_ref_lane=self.best_ref_lane,
            between_ref_lane=self.between_ref_lane,
            ego_index=self.ego_index,
            ref_index=self.ref_index,
            same_lane_flag=self.same_lane_flag,
            is_adjacent_flag=self.is_adjacent_flag,
            is_one_between_flag=self.is_one_between_flag,
            forced_lane_change_flag=self.forced_lane_change_flag,

            acc=self.acc,
            steer=self.steer,
            x_sol=self.x_sol,
            u_sol=self.u_sol,
            slack_sol=self.slack_sol,

            plan_lon_spd=self.plan_lon_spd,
            plan_lat_spd=self.plan_lat_spd,
            plan_lat_acc=self.plan_lat_acc,
            plan_yaw=self.plan_yaw,
            is_plan=self.is_plan,
            target_path_name='2',
            target_path_point='2'
        )

        planning_path = [Point(x, y) for x, y in zip(plan_x, plan_y)]
        planning_path = path_to_traj(planning_path)
        return self.acc,self.steer,planning_path

    def update_state(self, position_info, moving_info, reference_path, action, movement_signal):
        """
        Update vehicle state including position, speed, heading, and lane context.
        """
        # Longitudinal, lateral, and yaw speed
        self.lon_spd = moving_info.get('u')
        self.lat_spd = moving_info.get('v')
        self.yaw_spd = moving_info.get('w')

        # Position and orientation
        point = position_info.get('point', {})
        self.x = point.get('x')
        self.y = point.get('y')
        self.z = point.get('z')
        self.phi = position_info.get('phi')

        # Topology and distance to lane end
        self.lane_id = position_info.get('lane_id')
        self.link_id = position_info.get('link_id')
        self.junction_id = position_info.get('junction_id')
        self.dis_to_lane_end = position_info.get('dis_to_lane_end')

        # Position type: 0 unknown, 1 on lane, 2 in intersection, 3 off road
        self.position_type = position_info.get('position_type')

        # Last control increments
        self.last_delta_acc = action[0]
        self.last_delta_steer = action[1]

        # Current reference path and signal state
        self.reference_path = reference_path
        self.movement_signal = movement_signal


    def decision(self, all_vehicle_info, all_veh_moving_info, all_veh_base_info, step):
        # 1. Process the reference trajectory of the controlled vehicle (trajectory interpolation)
        gen_ego_path_y, gen_ego_path_x, gen_ego_path_id, gen_ego_path_rad = self.interpolate_reference_lines_local(
            interval=0.1)
        # 2. global reference path
        global_lane_ids = self.find_global_reference_lane_ids()
        # 3. Determine the location and type of the accused vehicle (to provide judgment conditions for subsequent decision-making)
        self.get_best_ref_lane_relation(gen_ego_path_id, global_lane_ids)
        # 4. Screen cars
        front_vehicles, rear_vehicles, nearest_vehicles = self.select_surr_veh(all_vehicle_info, gen_ego_path_id)
        # 5. compute path cost
        cost, best_index = self.cal_cost(front_vehicles, rear_vehicles, nearest_vehicles, all_vehicle_info,
                                         all_veh_moving_info, all_veh_base_info, gen_ego_path_y, gen_ego_path_x,
                                         gen_ego_path_id, step)
        # 6. Decide whether to force lane changing (and output static path information)
        path_id = self.determine_path(step, best_index, gen_ego_path_id)
        return path_id, nearest_vehicles

    def interpolate_reference_lines_local(
        self,
        interval=0.5,
        max_points_per_line=300
    ) -> Tuple[List[List[float]], List[List[float]], List[str], List[List[float]]]:
        """
        Interpolate reference lines and transform them into the ego-vehicle local frame.

        Args:
            interval (float): Spacing between interpolated points in meters.
            max_points_per_line (int): Maximum number of interpolated points per reference line
                to limit computation.

        Returns:
            Tuple[
                List[List[float]],  # ego_path_y: lateral coordinates in ego frame for each lane
                List[List[float]],  # ego_path_x: longitudinal coordinates in ego frame for each lane
                List[str],          # ego_path_id: lane IDs for each reference path
                List[List[float]]   # ego_path_rad: headings in ego frame for each lane
            ]
        """
        ego_x, ego_y, ego_phi = self.x, self.y, self.phi
        ego_path_x, ego_path_y, ego_path_id, ego_path_rad = [], [], [], []
        lane_types, lane_index = [], []

        # Build rotation matrix for world-to-ego coordinate transform
        rot_mat = np.array([
            [cos(ego_phi), sin(ego_phi)],
            [-sin(ego_phi), cos(ego_phi)]
        ])

        for ref_line in self.reference_path:
            # Extract reference line coordinates
            coords = np.array([(p['x'], p['y']) for p in ref_line.get('points', [])])
            if len(coords) < 2:
                continue

            # Interpolate the line geometry
            line = LineString(coords)
            length = line.length
            num_points = min(max(int(length / interval), 1), max_points_per_line)
            samples = np.linspace(0, length, num_points + 1)
            points = np.array([line.interpolate(d).coords[0] for d in samples])

            # Transform to ego local coordinates
            dx_dy = points - np.array([ego_x, ego_y])
            local_coords = dx_dy @ rot_mat.T
            x_l, y_l = local_coords[:, 0], local_coords[:, 1]

            # Approximate heading along the line
            N = 5
            dx = np.roll(x_l, -N) - x_l
            dy = np.roll(y_l, -N) - y_l
            headings = np.arctan2(dy, dx)
            if len(headings) > N:
                headings[-N:] = headings[-N - 1]
            else:
                headings[:] = 0.0

            # Extract lane metadata
            lane_id = ref_line.get('lane_ids', ["unknown"])[0]
            ego_path_id.append(lane_id)
            lane_index.append(sum(ref_line.get('lane_idxes', [])))
            lane_types.append(ref_line.get('lane_types', ["unknown"]))

            # Store results
            ego_path_x.append(x_l.tolist())
            ego_path_y.append(y_l.tolist())
            ego_path_rad.append(headings.tolist())

        self.lane_id_types = lane_types
        self.lane_idxes = lane_index
        return ego_path_y, ego_path_x, ego_path_id, ego_path_rad


    def find_global_reference_lane_ids(self) -> List[str]:
        """
        Identify and return the list of global reference lane IDs.
        The method scans all reference paths and determines which path contains the largest
        number of lane IDs. It then collects the first lane ID from each reference path
        that has this maximum count.

        Returns:
            List[str]: A list containing the first lane ID from reference paths
                    with the maximum number of lane IDs.
                    If multiple paths have the same count, all their first IDs are included.
        """
        max_len = 0
        result = []

        for ref_line in self.reference_path:
            lane_ids = ref_line.get('lane_ids', [])
            if not lane_ids:
                continue
            if len(lane_ids) > max_len:
                max_len = len(lane_ids)
                result = [lane_ids[0]]
            elif len(lane_ids) == max_len:
                result.append(lane_ids[0])

        self.global_lane_id_list = result
        return result


    def get_best_ref_lane_relation(self, lane_ids: List[str], ref_lane_ids: List[str]):
        """
        Compute the relationship between the ego lane and the set of reference lanes.

        Sets:
            - best_ref_lane: Chosen reference lane ID.
            - between_ref_lane: Lane ID between ego and reference lane if exactly one lies in between.
            - ego_index: Index of ego lane in lane_ids.
            - ref_index: Index of best_ref_lane in lane_ids.
            - same_lane_flag: True if ego lane is the reference lane.
            - is_adjacent_flag: True if ego and reference lanes are adjacent.
            - is_one_between_flag: True if there is exactly one lane between them.
            - lane_position: Encoded lane position type:
                11  -> single lane
                0  -> leftmost lane
                -1  -> rightmost lane
                1  -> middle lane
            101  -> in or near junction
            - lane_idxes_before_junction: Ego lane index before entering junction.
        """
        if self.lane_id not in lane_ids:
            return None

        ego_index = lane_ids.index(self.lane_id)
        best_ref_lane = None
        best_ref_index = None

        # Step 1: try to match the same lane
        for ref_id in ref_lane_ids:
            if ref_id == self.lane_id:
                best_ref_lane = ref_id
                best_ref_index = ego_index
                break

        # Step 2: otherwise search for an adjacent lane
        if not best_ref_lane:
            for ref_id in ref_lane_ids:
                if ref_id in lane_ids:
                    ref_index = lane_ids.index(ref_id)
                    if abs(ref_index - ego_index) == 1:
                        best_ref_lane = ref_id
                        best_ref_index = ref_index
                        break

        if best_ref_lane is None:
            # No identical or adjacent reference lane found
            return None

        # Step 3: compute relation flags
        index_diff = abs(best_ref_index - ego_index)
        same_lane = best_ref_index == ego_index
        is_adjacent = index_diff == 1
        is_one_between = index_diff == 2
        between_lane = None
        if is_one_between:
            between_index = (ego_index + best_ref_index) // 2
            between_lane = lane_ids[between_index]

        if self.position_type != 2 and self.dis_to_lane_end >= 5:
            # On normal lane segment, away from junction
            self.lane_idxes_before_junction = ego_index

            n = len(lane_ids)  # total number of lanes
            if n == 1:
                # Single lane
                self.lane_position = 11
            elif n == 2:
                if ego_index == 0:
                    # Left lane
                    self.lane_position = 0
                else:
                    # Right lane
                    self.lane_position = -1
            else:
                # n >= 3
                if ego_index == 0:
                    # Leftmost lane
                    self.lane_position = 0
                elif ego_index == n - 1:
                    # Rightmost lane
                    self.lane_position = -1
                else:
                    # Middle lane
                    self.lane_position = 1
        else:
            # In junction area or too close to lane end
            self.lane_position = 101
            self.lane_idxes_before_junction = 100

        self.best_ref_lane = best_ref_lane
        self.between_ref_lane = between_lane
        self.ego_index = ego_index
        self.ref_index = best_ref_index
        self.same_lane_flag = same_lane
        self.is_adjacent_flag = is_adjacent
        self.is_one_between_flag = is_one_between


    def select_surr_veh(self, all_veh_info, gen_ego_path_id):
        """
        Select surrounding vehicles:
        - Per lane: closest front and rear vehicles (by dis_to_lane_end).
        - Globally: up to 20 nearest vehicles by Euclidean distance.
        """
        # 1. Remove ego vehicle from the set
        all_veh_info = {k: v for k, v in all_veh_info.items() if k != self.id}

        if self.position_type != 2:
            # 2. Group vehicles by lane order according to gen_ego_path_id
            lanes_split: List[List[Tuple[str, float]]] = [[] for _ in gen_ego_path_id]
            for vid, info in all_veh_info.items():
                lane_id = info.get("lane_id", None)
                if lane_id in gen_ego_path_id:
                    index = gen_ego_path_id.index(lane_id)
                    dis_to_end = info.get("dis_to_lane_end", 0.0)
                    lanes_split[index].append((vid, dis_to_end))

            # 3. For each lane, find closest front/rear vehicles based on dis_to_lane_end
            front_vehicles = []
            rear_vehicles = []
            for lane_vehicles in lanes_split:
                front, rear = None, None
                min_front, min_rear = float('inf'), float('inf')
                for vid, dis in lane_vehicles:
                    delta = self.dis_to_lane_end - dis
                    if delta > 0 and delta < min_front:
                        front = vid
                        min_front = delta
                    elif delta < 0 and abs(delta) < min_rear:
                        rear = vid
                        min_rear = abs(delta)
                front_vehicles.append(front)
                rear_vehicles.append(rear)
        else:
            # In intersection: do not assign lane-based front/rear vehicles
            front_vehicles = [None for _ in gen_ego_path_id]
            rear_vehicles = [None for _ in gen_ego_path_id]

        # 4. Find nearest vehicles in global position space (up to 20)
        ego_x, ego_y = self.x, self.y
        dists = []
        for vid, info in all_veh_info.items():
            point = info.get('point', {})
            x = point.get('x')
            y = point.get('y')
            if x is not None and y is not None:
                dx = x - ego_x
                dy = y - ego_y
                dist = np.hypot(dx, dy)
                dists.append((vid, dist))

        dists.sort(key=lambda x: x[1])
        nearest_vehicles = [vid for vid, _ in dists[:20]]

        return front_vehicles, rear_vehicles, nearest_vehicles


    def extract_vehicle_info(
        self,
        vehicle_ids: List[Optional[str]],
        all_veh_info: Dict[str, Any],
        all_veh_moving_info: Dict[str, Any],
        all_veh_base_info: Dict[str, Any],
        use_world_coord: bool
    ):
        """
        Extract normalized vehicle information for a list of vehicle IDs.

        Args:
            vehicle_ids: List of vehicle IDs to query. May contain None.
            all_veh_info: Static and positional info for all vehicles.
            all_veh_moving_info: Dynamic (velocity, yaw rate, etc.) info for all vehicles.
            all_veh_base_info: Base info such as length and width for all vehicles.
            use_world_coord: If True, output coordinates in world frame;
                            if False, transform to ego-vehicle local frame.

        Returns:
            List of dicts or None:
                Each entry contains:
                    x, y, phi, u, v, w, dis_to_lane_end, length, width
                or None if information is incomplete or invalid.
        """
        result = []

        ego_x, ego_y, ego_phi = self.x, self.y, self.phi

        for vid in vehicle_ids:
            if vid is None:
                result.append(None)
                continue

            info = all_veh_info.get(vid)
            move = all_veh_moving_info.get(vid)
            base = all_veh_base_info.get(vid)

            if not info or not move or not base:
                result.append(None)
                continue

            try:
                # Original world coordinates
                point = info.get('point', {})
                x_w = point.get('x')
                y_w = point.get('y')
                phi_w = info.get('phi')

                if x_w is None or y_w is None or phi_w is None:
                    result.append(None)
                    continue

                # Common kinematic and geometric fields
                common_info = {
                    "u": move.get("u", None),
                    "v": move.get("v", None),
                    "w": move.get("w", None),
                    "dis_to_lane_end": info.get("dis_to_lane_end", None),
                    "length": base.get("base_info", {}).get("length", None),
                    "width": base.get("base_info", {}).get("width", None),
                }

                if use_world_coord:
                    item = {
                        "x": x_w,
                        "y": y_w,
                        "phi": phi_w,
                        **common_info
                    }
                else:
                    # Transform coordinates into ego-vehicle local frame
                    dx = x_w - ego_x
                    dy = y_w - ego_y
                    x_l = dx * math.cos(ego_phi) + dy * math.sin(ego_phi)
                    y_l = -dx * math.sin(ego_phi) + dy * math.cos(ego_phi)

                    # Normalize heading to [-pi, pi)
                    phi_l = (phi_w - ego_phi + math.pi) % (2 * math.pi) - math.pi

                    item = {
                        "x": x_l,
                        "y": y_l,
                        "phi": phi_l,
                        **common_info
                    }

                result.append(item)

            except Exception as e:
                print(f"[Warning] Failed to extract info for {vid}: {e}")
                result.append(None)

        return result

    def cal_cost(
        self,
        front_vehicles,
        rear_vehicles,
        nearest_vehicles,
        all_vehicle_info,
        all_veh_moving_info,
        all_veh_base_info,
        gen_ego_path_y,
        gen_ego_path_x,
        gen_ego_path_id,
        step
    ):
        """
        Compute lane cost for current step.

        For multi-lane road segments:
            - Use IDM-based efficiency (front vehicles).
            - Use rear safety evaluation (TTC-like).
            - Fuse into composite lane scores.

        For intersections:
            - Use road occupancy based evaluation.

        Safety post-processing:
            - Avoid lanes with very close front/rear vehicles.
            - Suppress risky lane changes that skip over one or more lanes.
        """
        cost = []
        index = 0

        nearest_info = self.extract_vehicle_info(
            nearest_vehicles,
            all_vehicle_info,
            all_veh_moving_info.get('moving_info_dict'),
            all_veh_base_info.get('info_dict'),
            False
        )

        # Multi-lane scenario
        if self.position_type != 2:
            self.decision_step_index_junction = 100

            rear_info = self.extract_vehicle_info(
                rear_vehicles,
                all_vehicle_info,
                all_veh_moving_info.get('moving_info_dict'),
                all_veh_base_info.get('info_dict'),
                False
            )
            front_info = self.extract_vehicle_info(
                front_vehicles,
                all_vehicle_info,
                all_veh_moving_info.get('moving_info_dict'),
                all_veh_base_info.get('info_dict'),
                False
            )

            # Efficiency: IDM-based evaluation
            J_idm = self.IDM_traffic_evaluation(front_info)
            # Safety: rear-vehicle evaluation (TTC-based)
            J_rear = self.Rear_safe_evaluation(rear_info)
            # Lane composite scores
            scores, best_index = self.lane_composite_scores(J_idm, J_rear)
            cost = scores
            index = best_index

        # Intersection scenario
        else:
            self.decision_step_index = 100
            # Road occupancy based evaluation at junction
            J_junction, best_index = self.road_occupancy_junction(
                nearest_info,
                gen_ego_path_y,
                gen_ego_path_x,
                gen_ego_path_id,
                step
            )
            # Rear safety at junction is currently not considered separately
            cost = J_junction
            index = best_index

        # Additional safety checks for multi-lane roads
        if self.position_type != 2:
            # If front or rear vehicle in the best lane is too close, stay in current lane
            if (
                self.rear_veh_dis_list[best_index] <= 5
                or self.front_veh_dis_list[best_index] <= 5
            ):
                index = self.ego_index

            # Reduce risky lane changes that cross more than one lane at once
            if abs(best_index - self.ego_index) >= 2:
                if best_index > self.ego_index:
                    unsafe = False
                    for i1 in range(self.ego_index + 1, best_index):
                        if (
                            self.front_veh_dis_list[i1] <= 5 + self.lon_spd
                            or self.rear_veh_dis_list[i1] <= 5 + self.lon_spd
                        ):
                            unsafe = True
                            break
                    index = best_index if not unsafe else self.ego_index
                else:
                    unsafe = False
                    for i2 in range(best_index, self.ego_index - 1):
                        if (
                            self.front_veh_dis_list[i2] <= 5 + self.lon_spd
                            or self.rear_veh_dis_list[i2] <= 5 + self.lon_spd
                        ):
                            unsafe = True
                            break
                    index = best_index if not unsafe else self.ego_index

        return cost, index


    def IDM_traffic_evaluation(self, front_info):
        """
        Evaluate lane efficiency using IDM (Intelligent Driver Model) for each lane
        based on the ego vehicle and its front vehicles.

        Returns:
            J_idm: List[float] of normalized efficiency scores per lane.
        """
        J_idm = []
        self.front_veh_dis_list = []

        # IDM parameters
        a_max: float = 1.5   # maximum acceleration (m/s^2)
        b: float = 2.5       # comfortable deceleration (m/s^2)
        T: float = 1.5       # desired time headway (s)
        s0: float = 8.0      # minimum gap (m)
        delta: int = 3       # acceleration exponent
        gamma: float = 0.02  # small bias favoring current lane

        if front_info:
            self.front_veh_info_lane = front_info[self.ego_index]
        else:
            self.front_veh_info_lane = None

        for i in range(len(front_info)):
            if front_info[i] is not None:
                # Gap to front vehicle
                clips = (
                    self.dis_to_lane_end
                    - front_info[i].get('dis_to_lane_end')
                    - (front_info[i].get('length') + self.length) / 2
                )
                s = np.clip(clips, 0.01, 500)

                # Relative speed
                delta_v = self.lon_spd - front_info[i].get('u')

                # Desired dynamic gap
                s_star = s0 + self.lon_spd * T + (
                    self.lon_spd * delta_v
                ) / (2 * math.sqrt(a_max * b))
                s_star = max(s0, s_star)  # avoid negative values

                # IDM acceleration
                acc = a_max * (
                    1
                    - (self.lon_spd / self.ref_speed) ** delta
                    - (s_star / max(s, 0.1)) ** 2
                )

                # Normalize acceleration into [0, 1]
                acc_clipped = max(-b, min(acc, a_max))
                acc_norm = (acc_clipped + b) / (a_max + b)

                J_idm.append(acc_norm)
                self.front_veh_dis_list.append(s)
            else:
                # No front vehicle on this lane
                J_idm.append(1.0)
                self.front_veh_dis_list.append(500.0)

        # Slightly bias towards staying in the current lane
        J_idm[self.ego_index] = J_idm[self.ego_index] + gamma

        return J_idm


    def Rear_safe_evaluation(self, rear_info):
        """
        Evaluate rear-vehicle safety score for each lane.

        Higher score means safer to change into or stay in that lane considering
        following vehicles.

        Returns:
            J_rear: List[float] safety scores per lane.
        """
        self.rear_veh_dis_list = []
        J_rear = []

        factor: float = 0.8   # factor for ego speed when estimating relative speed loss in lane change
        t_change: float = 3.0 # reference time for lane change
        gamma: float = 0.3    # bias term favoring current lane

        for i in range(len(rear_info)):
            if rear_info[i] is not None:
                # Distance to rear vehicle along lane
                clips = (
                    rear_info[i].get('dis_to_lane_end')
                    - self.dis_to_lane_end
                    - (rear_info[i].get('length') + self.length) / 2
                )
                s = np.clip(clips, 0.01, 80)
                self.rear_veh_dis_list.append(s)

                # Relative speed between rear vehicle and ego (scaled)
                delta_v = rear_info[i].get('u') - factor * self.lon_spd
                if delta_v <= 0:
                    # Rear vehicle is not closing in
                    J_rear.append(1.0)
                else:
                    # Time-to-collision-like metric
                    t_co = s / delta_v
                    safe = 1 - t_change / t_co
                    safe_clipped = max(0.0, min(safe, 1.0))
                    J_rear.append(safe_clipped)
            else:
                # No rear vehicle on this lane
                J_rear.append(1.0)
                self.rear_veh_dis_list.append(80.0)

        # Slightly favor current lane
        J_rear[self.ego_index] = J_rear[self.ego_index] + gamma
        return J_rear

    def lane_composite_scores(self, J_idm, J_rear):
        """
        Compute composite lane scores by combining:
        - IDM efficiency score (J_idm)
        - Rear-vehicle safety score (J_rear)

        Args:
            J_idm:  List[float] - IDM efficiency scores.
            J_rear: List[float] - rear-vehicle safety scores.

        Returns:
            Tuple[List[float], int]:
                - composite_scores: fused lane scores.
                - best_index: index of the lane with the highest score.
        """
        # Weights for efficiency and safety (tunable)
        weight_idm: float = 0.6
        weight_rear: float = 0.4

        assert len(J_idm) == len(J_rear), "Score length mismatch."

        # Fuse scores
        composite_scores = [
            round(weight_idm * j1 + weight_rear * j2, 4)
            for j1, j2 in zip(J_idm, J_rear)
        ]

        # Index of lane with maximum score
        best_index = composite_scores.index(max(composite_scores))
        return composite_scores, best_index

    def road_occupancy_junction(self, nearest_info, gen_ego_path_y, gen_ego_path_x, gen_ego_path_id, step):
        """
        Evaluate lane scores inside an intersection based on:
        - Preference for the shortest path.
        - Penalty for lanes occupied by nearby vehicles.

        Args:
            nearest_info:     List of nearest vehicles in ego frame.
            gen_ego_path_y:   List of lane centerline y-coordinates (ego frame).
            gen_ego_path_x:   List of lane centerline x-coordinates (ego frame).
            gen_ego_path_id:  List of lane IDs.
            step:             Current time step.

        Returns:
            Tuple[List[float], int]:
                - J_junction: lane scores at the junction.
                - best_index: index of best lane.
        """
        lane_scores = [1.0] * len(gen_ego_path_id)

        # Initial shortest-path lane index based on precomputed lane lengths
        min_length_index = self.lane_idxes.index(min(self.lane_idxes))

        # Adjust shortest lane index based on historical lane position before entering junction
        for i_s in range(1, step - 1):
            state_last = self.history_info.get(step - i_s)
            if state_last is None:
                print("Previous state is None when reading junction history.")
                break  # Avoid NoneType access

            lane_pos = getattr(state_last, 'lane_position', 100)
            ego_idx = getattr(state_last, 'lane_idxes_before_junction', 100)
            if ego_idx == 100:
                continue
            else:
                if lane_pos != 100 and lane_pos != 101:
                    print(f"lane_pos: {lane_pos}")
                    if lane_pos == 11 or lane_pos == 0:
                        # Only lane or left lane → choose leftmost
                        min_length_index = 0
                        break
                    elif lane_pos == -1:
                        # Right lane → choose rightmost
                        min_length_index = len(self.lane_idxes) - 1
                        break
                    elif lane_pos == 1:
                        # Middle lane → try to keep same index, fallback to 0 if invalid
                        if len(self.lane_idxes) - 1 >= ego_idx:
                            min_length_index = ego_idx
                        else:
                            min_length_index = 0
                            print("No valid lane index found for junction history.")
                        break

        self.min_length_index = min_length_index
        print(f"Lane length sum list: {self.lane_idxes}")
        print(
            f"Shortest-lane index: {min_length_index}, "
            f"lane ID: {gen_ego_path_id[min_length_index]}"
        )

        # Slightly prefer the (adjusted) shortest lane
        lane_scores[min_length_index] += 0.2

        # Penalize lanes whose centerline passes too close (<1m) to any nearby vehicle
        for i in range(len(gen_ego_path_id)):
            # Traverse all center points on this lane's centerline
            for x, y in zip(gen_ego_path_x[i], gen_ego_path_y[i]):
                # Compute distance to each nearby vehicle
                for veh in nearest_info:
                    dx = x - veh["x"]
                    dy = y - veh["y"]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1.0:
                        # Each occurrence reduces lane score slightly, but not below 0
                        lane_scores[i] = max(0.0, lane_scores[i] - 0.001)

        # Find lane(s) with maximum score
        max_score = max(lane_scores)
        best_indices = [i for i, score in enumerate(lane_scores) if score == max_score]

        # If shortest-path lane is among best, choose it, otherwise choose the first best lane
        if min_length_index in best_indices:
            best_lane_index = min_length_index
        else:
            best_lane_index = best_indices[0]

        print(
            f"Intersection lane scores: {lane_scores}, "
            f"best lane index: {best_lane_index}, "
            f"lane ID: {gen_ego_path_id[best_lane_index]}"
        )

        J_junction = lane_scores
        best_index = best_lane_index
        self.global_junction_id_list = [gen_ego_path_id[min_length_index]]
        return J_junction, best_index

    def determine_path(self, step, best_index, gen_ego_path_id):
        """
        Determine the final target lane (path ID) considering:
        - Normal lane selection.
        - Intersection selection.
        - Forced lane changes near intersections.

        Args:
            step:           Current simulation step.
            best_index:     Index of best lane from cost evaluation.
            gen_ego_path_id: List of lane IDs available to ego.

        Returns:
            path_id: Selected lane ID for tactical planning.
        """
        # 1. Check if forced lane change is required (before intersection)
        self.forced_lane_change_flag = (
            self.position_type != 2
            and self.dis_to_lane_end < self.forced_lane_change_dis
        )
        ego_path_id = gen_ego_path_id[self.ego_index]

        # 2. Case 1: No forced lane change
        if not self.forced_lane_change_flag:
            if self.position_type != 2:
                # Multi-lane segment decision
                if step % self.decision_interval == 0:
                    # Normal decision step: use best_index
                    path_id = gen_ego_path_id[best_index]
                    self.decision_step_index = best_index
                else:
                    # Keep previously decided lane if still safe, otherwise stay in ego lane
                    if self.decision_step_index != 100:
                        rear_safe = self.rear_veh_dis_list[self.decision_step_index] > 5
                        front_safe = self.front_veh_dis_list[self.decision_step_index] > 5
                        if rear_safe and front_safe:
                            path_id = gen_ego_path_id[self.decision_step_index]
                        else:
                            path_id = gen_ego_path_id[self.ego_index]
                    else:
                        path_id = gen_ego_path_id[self.ego_index]
            else:
                # Intersection decision
                if step % self.decision_interval == 0:
                    path_id = gen_ego_path_id[best_index]
                    self.decision_step_index_junction = best_index
                else:
                    if self.decision_step_index_junction != 100:
                        path_id = gen_ego_path_id[self.decision_step_index_junction]
                    else:
                        path_id = gen_ego_path_id[self.min_length_index]

        # 3. Case 2: Forced lane change active
        else:
            if self.same_lane_flag:
                # Already on the reference lane
                path_id = gen_ego_path_id[self.ego_index]
            else:
                # Move toward the reference lane
                path_id = gen_ego_path_id[self.ref_index]

        self.final_tac_lane_index = gen_ego_path_id.index(path_id)
        return path_id


    def planner(self, step, path_id, nearest_vehicles, all_vehicle_info, all_veh_moving_info, all_veh_base_info):

        # =================== Reference trajectory processing ===================
        ref_x, ref_y = self.get_ref_path(path_id=path_id)

        ref_x_j, ref_y_j, x_local, y_local = self.shift_ref_to_local_x0(ref_x, ref_y)

        # Generate Frenet reference
        csp, CENTER_TBL = path_plan_frenet.generate_target_course(ref_x_j, ref_y_j)

        # =================== Obstacle processing ===================
        # Surrounding vehicles in world frame
        nearest_info = self.extract_vehicle_info(
            nearest_vehicles,
            all_vehicle_info,
            all_veh_moving_info.get('moving_info_dict'),
            all_veh_base_info.get('info_dict'),
            True
        )
        # Process obstacles by two-circle method in world frame
        obs = self.process_surrounding_vehicles(nearest_info)

        # Surrounding vehicles in ego frame
        nearest_info_self = self.extract_vehicle_info(
            nearest_vehicles,
            all_vehicle_info,
            all_veh_moving_info.get('moving_info_dict'),
            all_veh_base_info.get('info_dict'),
            False
        )
        # Process obstacles by two-circle method in ego frame
        obs_self = self.process_surrounding_vehicles(nearest_info_self)

        Previous_car, Previous_left_car, Previous_right_car, Safe_car = self.Obstacle_Car_Searching(obs_self)
        ob = obs[:5, :].T
        # Keep only obstacle entries whose coordinates are not the placeholder 1000
        ob = ob[np.all(ob != 1e3, axis=1)]

        # Previous planned state
        state_last_1 = self.history_info.get(step - 1)
        plan_lon_spd = getattr(state_last_1, 'plan_lon_spd', 0.5)
        plan_lat_spd = getattr(state_last_1, 'plan_lat_spd', 0.0)
        plan_lat_acc = getattr(state_last_1, 'plan_lat_acc', 0.0)
        plan_phi = getattr(state_last_1, 'plan_yaw', 0.0)
        is_plan = getattr(state_last_1, 'is_plan', 0)

        # Current lateral position in Frenet frame
        c_d = 0 - y_local
        # Current lateral speed
        c_d_d = plan_lat_spd
        # Current lateral acceleration
        c_d_dd = plan_lat_acc
        # Current longitudinal position (reference)
        s0 = 0

        # === Base speed adjustment from IDM model ===
        acc_cmd = self.compute_idm_acc(nearest_info, self.lon_spd)
        plan_judge_spd = acc_cmd * 0.1 + 1

        # === Adjust speed based on time headway to front vehicle ===
        closest_front = self.find_front_vehicle(nearest_info_self, search_width=5.0, max_forward=20)
        if closest_front and self.lon_spd > 0:
            dx = closest_front["x"] - (self.length + closest_front.get("length", 0)) / 2
            dx = max(dx, 0.1)
            print(f"Front vehicle distance dx: {dx:.2f} m")
            headway = dx / self.lon_spd  # time headway (s)

            # Shorter headway or very small distance → stronger deceleration
            if headway < 1.5 or dx < 5.0:
                # Very close, strong deceleration
                plan_judge_spd -= 5.0
            elif headway < 3.0:
                # Moderate distance, mild deceleration
                plan_judge_spd -= 2.0

        # === Speed adjustment logic during lane change ===
        if self.position_type != 2:  # On normal road segments
            if self.final_tac_lane_index != self.ego_index:
                # If target lane has vehicles, reduce speed further to yield
                if self.final_tac_lane_index > self.ego_index and Previous_right_car:
                    plan_judge_spd -= 4.0
                elif self.final_tac_lane_index < self.ego_index and Previous_left_car:
                    plan_judge_spd -= 4.0
            else:
                # No lane change, adjust speed only based on front obstacles
                if Previous_car:
                    plan_judge_spd -= 12.0 if Safe_car else 5.0
        else:
            # In intersection
            if Previous_car:
                plan_judge_spd -= 12.0 if Safe_car else 5.0

        # === Compute target longitudinal speed ===
        base_target = self.lon_spd + plan_judge_spd

        if (
            (self.dis_to_lane_end <= self.forced_lane_change_dis and 'connection' in self.lane_id_types[self.final_tac_lane_index])
            or self.position_type == 2
        ):
            # Near intersection or forced lane-change zone: lower speed cap
            target_speed = min(self.ref_speed_intersection, np.clip(base_target, 0.000001,  self.ref_speed))
        else:
            # Normal road: clamp within [0.01, 15]
            target_speed = np.clip(base_target, 0.000001, self.ref_speed)

        # === Current speed for planning ===
        c_speed = np.clip(self.lon_spd + 1.5, 0.01, target_speed)

        # Road lateral boundary determination
        if self.position_type != 2 and self.dis_to_lane_end >= 2:
            if self.lane_position == 11:
                # Single lane
                left_boundary = 1.0
                right_boundary = -1.0
            elif self.lane_position == 0:
                # Left lane
                left_boundary = 1.0
                right_boundary = -3.0
            elif self.lane_position == -1:
                # Right lane
                left_boundary = 3.0
                right_boundary = -1.0
            elif self.lane_position == 1:
                # Middle lane
                left_boundary = 3.0
                right_boundary = -3.0
            elif 20 >= self.dis_to_lane_end >= 2:
                # Short distance to lane end, tighten lateral bounds
                left_boundary = 0.5
                right_boundary = -0.5
            else:
                left_boundary = 1.0
                right_boundary = -1.0
        else:
            # In or very near intersection: allow wider lateral range
            left_boundary = 5.0
            right_boundary = -5.0

        bestpath, fplist, fps_1 = path_plan_frenet.frenet_optimal_planning(
            csp,
            s0,
            c_speed,
            c_d,
            c_d_d,
            c_d_dd,
            ob,
            left_boundary,
            right_boundary,
            target_speed,
            self.ref_path_valid,
            CENTER_TBL,
        )

        # Apply the next planned position
        update_x = bestpath.x[1]
        update_y = bestpath.y[1]
        Position = Point(
            x=update_x,
            y=update_y,
            z=0.0
        )
        yaw = bestpath.yaw[1]
        self.is_plan = 1

        self.plan_lon_spd = bestpath.s_d[10] / 1
        self.plan_lat_spd = bestpath.d_d[1] / 1
        self.plan_lat_acc = bestpath.d_dd[1] / 1
        self.plan_pos = Position
        self.plan_yaw = yaw
        self.yaw_rate = (yaw - plan_phi) * 0.1
        self.lon_acc = plan_judge_spd * 10

        return bestpath.x, bestpath.y, bestpath.yaw, bestpath.s_d

    def find_front_vehicle(
            self,
            vehicles,
            search_width,
            max_forward,
            ego_phi: float = 0.0,
            dir_tol: float = math.pi / 4
    ) -> Optional[Dict]:
        """
        Find the closest front vehicle in the ego frame (if any).

        Search region in ego coordinates:
            0 < x <= max_forward
            |y| <= search_width / 2
        """
        half_w = search_width / 2

        def same_direction(phi_other: float) -> bool:
            """
            Check if another vehicle's heading is roughly aligned with ego heading,
            within ±dir_tol.
            """
            diff = (phi_other - ego_phi + math.pi) % (2 * math.pi) - math.pi
            return abs(diff) < dir_tol

        front = None
        min_x = math.inf

        for v in vehicles:
            x = v.get("x", 0.0)
            y = v.get("y", 0.0)

            # Longitudinal search constraint: 0 < x <= max_forward and |y| <= half_w
            if not (0.0 < x <= max_forward) or abs(y) > half_w:
                continue

            if self.position_type != 2:
                # On normal roads require similar heading
                if not same_direction(v.get("phi", 0.0)):
                    continue

            # Vehicle is in the search region, update nearest front vehicle
            if x < min_x:
                min_x = x
                front = v

        return front


    def compute_idm_acc(
            self,
            vehicles: List[Dict],
            ego_v: float,
            ego_length: float = 5,
            search_width: float = 5.0,
            default_acc: float = 0.5,
            idm_params: Dict[str, float] = None
    ) -> float:
        """
        Compute longitudinal acceleration using IDM.
        If no front vehicle is found, return default_acc.
        """
        p = {
            "a_max": 1.5,
            "b_comf": 7,
            "v0": 10.0,
            "T": 3,
            "s0": 8.0,
            "delta": 3
        }
        ego_phi = self.phi
        if idm_params:
            p.update(idm_params)

        front = self.find_front_vehicle(vehicles, search_width, max_forward=100)
        if front is None:
            return default_acc

        s = front["x"] - (ego_length + front.get("length", 0)) / 2
        if s <= 0:
            return -p["b_comf"]

        if self.position_type != 2:
            # Normal road IDM
            dv = ego_v - front["u"]
            s_star = p["s0"] + max(
                0.0,
                ego_v * p["T"] + ego_v * dv / (2 * math.sqrt(p["a_max"] * p["b_comf"]))
            )
            return p["a_max"] * (1 - (ego_v / p["v0"]) ** p["delta"] - (s_star / s) ** 2)
        else:
            # Intersection handling
            distance = math.sqrt(front["x"] ** 2 + front["y"] ** 2) - 3
            if distance <= 10:
                # Required braking to stop within distance
                return -(ego_v ** 2) / (2 * distance)
            else:
                dv = ego_v - 0
                s_star = p["s0"] + max(
                    0.0,
                    ego_v * p["T"] + ego_v * dv / (2 * math.sqrt(p["a_max"] * p["b_comf"]))
                )
                return p["a_max"] * (1 - (ego_v / p["v0"]) ** p["delta"] - (s_star / s) ** 2)


    def get_ref_path(self, path_id):

        world_path_x, world_path_y, world_path_id = [], [], []
        lane_end_point, found_end_point = None, False

        # -------- 1. Parse reference_path and locate each lane trajectory and lane end
        #            using first point + dis_to_lane_end rule. --------
        for ref_line in self.reference_path:
            points = ref_line.get('points')  # list of points with fields x, y, z
            if len(points) < 2:
                continue
            world_path_x.append([p.get('x') for p in points])
            world_path_y.append([p.get('y') for p in points])
            lane_id = ref_line.get('lane_ids')[0]
            world_path_id.append(lane_id)

            if not found_end_point:
                p0 = points[0]
                p1 = points[1]
                dis = float(self.dis_to_lane_end)
                tol = 0.5  # tolerance in meters
                match = None
                for p in points:
                    if abs(math.hypot(p.get('x') - p0.get('x'), p.get('y') - p0.get('y')) - dis) <= tol:
                        match = (p.get('x'), p.get('y'))
                        break
                if match is not None:
                    lane_end_point = [match[0], match[1]]
                else:
                    dx, dy = p1.get('x') - p0.get('x'), p1.get('y') - p0.get('y')
                    L = math.hypot(dx, dy)
                    if L <= 1e-9:
                        lane_end_point = [p0.get('x'), p0.get('y')]
                    else:
                        ux, uy = dx / L, dy / L
                        lane_end_point = [p0.get('x') + ux * dis, p0.get('y') + uy * dis]
                found_end_point = True

        # -------- 2. If red light is present, truncate all paths at the projected stop point
        #            (keep straight line segment from start to truncation point). --------
        if (
            self.movement_signal == 1
            and lane_end_point is not None
            and 'connection' in self.lane_id_types[self.final_tac_lane_index]
            and self.position_type != 2
        ):
            def project_point_to_polyline(px, py, line_x, line_y, offset=8.0):
                """
                Project a point (px, py) onto a polyline defined by (line_x, line_y),
                then shift backward along the tangent by 'offset' meters.
                """
                min_dist = float('inf')
                offset_pt = None
                for i in range(len(line_x) - 1):
                    x1, y1 = line_x[i], line_y[i]
                    x2, y2 = line_x[i + 1], line_y[i + 1]
                    dx, dy = x2 - x1, y2 - y1
                    seg_len_sq = dx * dx + dy * dy
                    if seg_len_sq == 0:
                        continue
                    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
                    t = max(0.0, min(1.0, t))
                    proj_x = x1 + t * dx
                    proj_y = y1 + t * dy
                    dist = math.hypot(proj_x - px, proj_y - py)
                    if dist < min_dist:
                        min_dist = dist
                        seg_len = math.sqrt(seg_len_sq)
                        unit_tx, unit_ty = dx / seg_len, dy / seg_len
                        offset_pt = (
                            proj_x - offset * unit_tx,
                            proj_y - offset * unit_ty
                        )
                return offset_pt

            for i in range(len(world_path_x)):
                rx, ry = world_path_x[i], world_path_y[i]
                if len(rx) < 2:
                    continue
                end_proj_pt = project_point_to_polyline(
                    lane_end_point[0], lane_end_point[1], rx, ry, offset=8.0
                )
                # Build a short path from the start point to the new truncated point
                world_path_x[i] = [rx[0], end_proj_pt[0]]
                world_path_y[i] = [ry[0], end_proj_pt[1]]

        # -------- 3. Extract reference path for current vehicle (then densify and clean). --------
        if path_id not in world_path_id:
            raise ValueError(f"path_id {path_id} not found in reference path.{world_path_id}")
        ref_x = world_path_x[world_path_id.index(path_id)]
        ref_y = world_path_y[world_path_id.index(path_id)]

        self.ref_path_valid = None
        if not ref_x or not ref_y:
            # Empty path is considered invalid
            self.ref_path_valid = False
        else:
            sx, sy = ref_x[0], ref_y[0]
            ex, ey = ref_x[-1], ref_y[-1]
            dist_se = math.hypot(ex - sx, ey - sy)
            # Path is considered valid if start-end distance is at least 2 meters
            self.ref_path_valid = dist_se >= 5

        new_x = [ref_x[0]]
        new_y = [ref_y[0]]
        for i in range(1, len(ref_x)):
            x0, y0 = ref_x[i - 1], ref_y[i - 1]
            x1, y1 = ref_x[i], ref_y[i]
            dx, dy = x1 - x0, y1 - y0
            d = math.hypot(dx, dy)
            if d == 0:
                continue
            if d > 10:
                # If segment is too long, linearly interpolate intermediate points.
                # Use int(d) instead of ceil to avoid overshooting the end point.
                n = int(d)
                for k in range(1, n):
                    t = k / d
                    new_x.append(x0 + dx * t)
                    new_y.append(y0 + dy * t)
            new_x.append(x1)
            new_y.append(y1)

        ref_x = new_x
        ref_y = new_y

        ref_x, ref_y = self.sanitize_reference_xy(ref_x, ref_y)

        return ref_x, ref_y


    def sanitize_reference_xy(self, x, y):
        """
        Clean and smooth the reference path in xy-space.

        Logic:
        - Compute unit direction vectors for each segment: d[i] = P[i+1] - P[i].
        - For internal point i (1..n-2):
            If the angle between (i-1)->i and i->(i+1) is > 90 degrees,
            and the angle between i->(i+1) and (i+1)->(i+2) is also > 90 degrees,
            then remove point i as a likely outlier.
        - Always keep endpoints and preserve ordering.
        - Also remove consecutive points that are extremely close to each other.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        if n < 3:
            return x.tolist(), y.tolist()

        pts = np.column_stack([x, y])
        eps = 1e-12

        # Segment direction vectors
        dirs = []
        for i in range(n - 1):
            v = pts[i + 1] - pts[i]
            nv = np.linalg.norm(v)
            if nv <= eps:
                # Degenerate segment
                dirs.append(None)
            else:
                dirs.append(v / nv)

        keep = [True] * n
        keep[0] = True
        keep[-1] = True

        for i in range(1, n - 1):
            d_prev = dirs[i - 1]                 # segment (i-1) -> i
            d_curr = dirs[i]                     # segment i -> (i+1)
            d_next = dirs[i + 1] if i + 1 < n - 1 else None  # segment (i+1) -> (i+2)

            # Need all three directions to be valid
            if d_prev is None or d_curr is None or d_next is None:
                continue

            # Angle > 90 degrees ⇔ dot product < 0
            bad_prev = np.dot(d_prev, d_curr) < 0.0
            bad_next = np.dot(d_curr, d_next) < 0.0

            if bad_prev and bad_next:
                keep[i] = False

        # Preserve order and remove points that are almost identical
        kept_idx = [i for i, k in enumerate(keep) if k]
        if not kept_idx:
            kept_idx = [0, n - 1]

        filt = [kept_idx[0]]
        for idx in kept_idx[1:]:
            if np.hypot(*(pts[idx] - pts[filt[-1]])) > 1e-6:
                filt.append(idx)

        pts_out = pts[filt]
        return pts_out[:, 0].tolist(), pts_out[:, 1].tolist()


    def shift_ref_to_local_x0(self, ref_x, ref_y):
        """
        Shift the reference path so that the ego vehicle is at x=0 in the
        local path-aligned coordinate frame.

        Steps:
        - Compute heading along the reference path.
        - Compute ego position in this path frame.
        - Translate the entire reference path such that the ego longitudinal
            coordinate becomes zero.
        """
        # Compute heading list along the reference path
        ref_yaw = []
        for i in range(len(ref_x) - 1):
            dx = ref_x[i + 1] - ref_x[i]
            dy = ref_y[i + 1] - ref_y[i]
            yaw = math.atan2(dy, dx)
            ref_yaw.append(yaw)
        # Append the last heading to match length of ref_x / ref_y
        ref_yaw.append(ref_yaw[-1])

        x_ref = ref_x[0]
        y_ref = ref_y[0]
        theta = ref_yaw[0]

        # Difference between ego position and reference path start
        dx = self.x - x_ref
        dy = self.y - y_ref

        # Coordinate transform: global -> reference-path local frame
        x_local = dx * math.cos(theta) + dy * math.sin(theta)
        y_local = -dx * math.sin(theta) + dy * math.cos(theta)

        # To make ego x coordinate zero in local frame, shift reference path
        # in global frame along the first heading direction by -x_local.
        dx_shift = (-x_local) * math.cos(ref_yaw[0])
        dy_shift = (-x_local) * math.sin(ref_yaw[0])

        new_ref_x = []
        new_ref_y = []
        for x, y in zip(ref_x, ref_y):
            # Translate each point in global coordinates
            new_ref_x.append(x - dx_shift)
            new_ref_y.append(y - dy_shift)

        return new_ref_x, new_ref_y, -x_local, -y_local


    def Controller(self, step, tac_path_y, tac_path_x, tac_path_rad, tac_path_v):
        if len(tac_path_x) > 2:
            y_cod = []
            x_cod = []
            for i in range(len(tac_path_x)):
                relative_x = tac_path_x[i] - self.x
                relative_y = tac_path_y[i] - self.y
                rotated_x = relative_x * math.cos(self.phi) + relative_y * math.sin(self.phi)
                rotated_y = -relative_x * math.sin(self.phi) + relative_y * math.cos(self.phi)
                y_cod.append(rotated_y)
                x_cod.append(rotated_x)

            dt = 0.1
            N = len(tac_path_x) - 1
            state_last_1 = self.history_info.get(step - 1)
            acc_last_1 = getattr(state_last_1, 'acc', 0.0)
            steer_last_1 = getattr(state_last_1, 'steer', 0.0)
            state_last_2 = self.history_info.get(step - 2)
            acc_last_2 = getattr(state_last_2, 'acc', 0.0)
            steer_last_2 = getattr(state_last_2, 'steer', 0.0)

            # MPC-based controller
            xt_val = np.array([0.0, 0.0, 0.0, self.lon_spd, steer_last_1])
            u_prev = np.array([acc_last_1, (steer_last_1 - steer_last_2) / 0.1])
            # u_prev = np.array([self.lon_acc + self.last_delta_acc, self.last_delta_steer])
            car = {"length_car": 3.22}
            ref_val = np.array(
                [x_cod[:N + 1], y_cod[:N + 1], tac_path_rad[:N + 1],
                tac_path_v[:N + 1], np.array(np.repeat(0, N + 1))]
            )

            u_k, next_state, delta_acc, delta_steer = mpc_controller.mpc_optimizer(
                xt_val, ref_val, car, N, u_prev, dt, 0.1
            )
            self.acc = u_k[0]
            self.steer = next_state[4]

            # === Always output arrays of length 20 ===
            pad_len = 20 - len(delta_acc)
            if pad_len > 0:
                self.delta_acc = np.concatenate([delta_acc, np.zeros(pad_len)])
                self.delta_steer = np.concatenate([delta_steer, np.zeros(pad_len)])
            else:
                self.delta_acc = delta_acc[:20]
                self.delta_steer = delta_steer[:20]

        else:
            u_k = [0, 0]
            next_state = [0, 0, 0, 0, 0]
            self.acc = u_k[0]
            self.steer = next_state[4]
            self.delta_acc = [0] * 20
            self.delta_steer = [0] * 20

    def process_surrounding_vehicles(self, nearest_info, max_car_num=20):
        """
        Convert raw surrounding vehicle information into a compact obstacle representation
        using the two-circle model (front and rear circles).
        """
        processed_Surr_car = []
        for veh in nearest_info:
            x = veh['x']
            y = veh['y']
            phi = veh['phi']  # Heading in radians
            u = veh['u']
            v = veh['v']
            length = veh['length']

            # Front circle center using two-circle vehicle model
            offset = length / 4
            x_circ = x + math.cos(phi) * offset
            y_circ = y + math.sin(phi) * offset
            processed_Surr_car.append([x_circ, y_circ, u, v, phi])

            # Rear circle center
            rear_offset = length / 4
            x_rear = x - math.cos(phi) * rear_offset
            y_rear = y - math.sin(phi) * rear_offset
            processed_Surr_car.append([x_rear, y_rear, u, v, phi])

        # If less than max_car_num entries, pad with dummy obstacles
        while len(processed_Surr_car) < max_car_num:
            processed_Surr_car.append([1000.0, 1000.0, 0.0, 0.0, 0.0])

        # If more than max_car_num entries, keep only the first max_car_num
        processed_Surr_car = processed_Surr_car[:max_car_num]

        # Build obstacle array
        obs = np.array([
            [row[0] for row in processed_Surr_car],  # x
            [row[1] for row in processed_Surr_car],  # y
            [row[4] for row in processed_Surr_car],  # phi
            [row[2] for row in processed_Surr_car],  # u
            [row[3] for row in processed_Surr_car]   # v
        ])
        return obs

    def get_rectangle_points(self, cx, cy, w, h, heading):
        """
        Compute the four corner points of a rectangle given:
        - center point (cx, cy)
        - width w (across vehicle lateral direction)
        - height h (along vehicle longitudinal direction)
        - heading (in radians)

        Conventions:
        - Width w is lateral to the heading direction.
        - Height h is along the heading direction (front-back).
        - Local frame: x_local along forward, y_local along left-right.
        """
        half_w = w / 2.0
        half_h = h / 2.0

        # Define rectangle corners in the local frame (center at origin)
        # Order: [front-left, front-right, rear-right, rear-left]
        pts_local = np.array([
            [half_h, half_w],   # front, left
            [half_h, -half_w],  # front, right
            [-half_h, -half_w], # rear, right
            [-half_h, half_w]   # rear, left
        ])

        # Rotation matrix for heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        R = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])

        # Rotate and translate local points to global frame
        pts_global = []
        for p in pts_local:
            p_rot = R @ p
            p_glob = [cx + p_rot[0], cy + p_rot[1]]
            pts_global.append(p_glob)

        return pts_global

    def is_point_in_rectangle(self, points, rect_points):
        """
        Check whether any point in `points` lies inside the rectangle defined by `rect_points`.

        Args:
            points:      iterable of (px, py)
            rect_points: 4 points [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] defining a convex quad

        Returns:
            True if at least one point is inside the rectangle, otherwise False.
        """

        def cross_product(p1, p2, p3):
            return ((p2[0] - p1[0]) * (p3[1] - p1[1])
                    - (p2[1] - p1[1]) * (p3[0] - p1[0]))

        for px, py in points:
            signs = []
            for i in range(4):
                p1 = rect_points[i]
                p2 = rect_points[(i + 1) % 4]
                cp = cross_product(p1, p2, (px, py))
                signs.append(cp)

            # If all cross products share the same sign (all >= 0 or all <= 0),
            # the point lies inside the convex quadrilateral.
            all_positive = all(s >= -1e-8 for s in signs)
            all_negative = all(s <= 1e-8 for s in signs)

            if all_positive or all_negative:
                # As soon as one point is inside, return True
                return True

        # No point lies inside the rectangle
        return False

    def calculate_offset_points(self, x, y, phi):
        """
        Compute several reference points of an obstacle (or vehicle)
        in the local frame:
        - center point
        - point shifted 2.5 m backward
        - point shifted 2.5 m forward

        The offsets can be tuned according to the actual vehicle size.
        """
        offset1 = (x, y)
        offset2 = (x - 2.5 * math.cos(phi), y - 2.5 * math.sin(phi))
        offset3 = (x + 2.5 * math.cos(phi), y + 2.5 * math.sin(phi))
        return offset1, offset2, offset3

    def Obstacle_Car_Searching(self, obs_array):
        """
        Classify obstacles around the ego vehicle into directional regions.

        Args:
            obs_array: array of shape (5, N)
                - obs_array[0, i]: obstacle i x-coordinate in ego frame
                - obs_array[1, i]: obstacle i y-coordinate in ego frame
                - obs_array[2, i]: obstacle i heading phi (relative to ego frame)
                - obs_array[3, i]: obstacle i longitudinal velocity (unused here)
                - obs_array[4, i]: obstacle i lateral velocity (unused here)

            Ego vehicle is assumed to be at (0, 0, 0) in the local frame.

        Returns:
            - Previous_car:      list of (x, y) for obstacles in the front region
            - Previous_left_car: list of (x, y) for obstacles in the left region
            - Previous_right_car:list of (x, y) for obstacles in the right region
            - Safe_car:          list of (x, y) for obstacles in the "safety" region
        """

        # -------------------------------
        # 1. Define four rectangular regions in the ego local frame
        #    (x forward-backward, y left-right).
        # -------------------------------
        left_width, left_height = 2.5, 8.0    # left region
        right_width, right_height = 2.5, 8.0  # right region
        front_width, front_height = 3.5, 12.0 # front region
        safe_width, safe_height = 3.5, 8.0    # safety region

        # -------------------------------
        # 2. Compute centers and rectangles for each region.
        #    Ego is at (0, 0), heading = 0.
        # -------------------------------
        # 2.1 Left region center (y > 0 means left, shift by (width/2 + 1))
        left_center_x = 0.0
        left_center_y = (left_width / 2) + 1.0
        left_rect = self.get_rectangle_points(
            left_center_x, left_center_y,
            left_width, left_height,
            heading=0.0  # no rotation
        )

        # 2.2 Right region center (y < 0 means right, shift by (width/2 + 1))
        right_center_x = 0.0
        right_center_y = -((right_width / 2) + 1.0)
        right_rect = self.get_rectangle_points(
            right_center_x, right_center_y,
            right_width, right_height,
            heading=0.0
        )

        # 2.3 Front region center (x > 0 means front, shift by half height)
        front_center_x = (front_height / 2)
        front_center_y = 0.0
        front_rect = self.get_rectangle_points(
            front_center_x, front_center_y,
            front_width, front_height,
            heading=0.0
        )

        # 2.4 Safety region center (a bit in front: half height)
        safe_center_x = (safe_height / 2.0)
        safe_center_y = 0.0
        safe_rect = self.get_rectangle_points(
            safe_center_x, safe_center_y,
            safe_width, safe_height,
            heading=0.0
        )

        # -------------------------------
        # 3. Initialize result containers
        # -------------------------------
        Previous_car = []
        Previous_left_car = []
        Previous_right_car = []
        Safe_car = []

        # -------------------------------
        # 4. Iterate over all obstacles and classify them into regions
        # -------------------------------
        N = obs_array.shape[1]
        for i in range(N):
            veh_x = obs_array[0, i]
            veh_y = obs_array[1, i]
            veh_phi = obs_array[2, i]
            # vx, vy = obs_array[3, i], obs_array[4, i]  # available if needed

            # Get 3 offset points for this obstacle
            offset_pts = self.calculate_offset_points(veh_x, veh_y, veh_phi)

            # Left region
            if self.is_point_in_rectangle(offset_pts, left_rect):
                Previous_left_car.append((veh_x, veh_y))

            # Right region
            if self.is_point_in_rectangle(offset_pts, right_rect):
                Previous_right_car.append((veh_x, veh_y))

            # Front region
            if self.is_point_in_rectangle(offset_pts, front_rect):
                Previous_car.append((veh_x, veh_y))

            # Safety region
            if self.is_point_in_rectangle(offset_pts, safe_rect):
                Safe_car.append((veh_x, veh_y))

        return Previous_car, Previous_left_car, Previous_right_car, Safe_car
    
def path_to_traj(planning_path):
        xs = []
        ys = []
        yaws = []

        for i in range(len(planning_path)):
            p = planning_path[i]
            x = p.x
            y = p.y
            xs.append(x)
            ys.append(y)

        # 计算 yaw（由相邻点计算）
        for i in range(len(planning_path)-1):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            yaw = math.atan2(dy, dx)
            yaws.append(yaw)

        # 最后一个点的航向用倒数第二个点的 yaw
        yaws.append(yaws[-1])

        traj = np.vstack([xs, ys, yaws]).T
        return traj
