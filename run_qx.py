from contextlib import contextmanager
import numpy as np
import json
import os
from datetime import datetime
from dataclasses import asdict
import time
from gops.rule_model.planner import Agent
from gops.create_pkg.create_env import create_env
from gops.utils.common_utils import set_seed

from lasvsim_openapi.simulator_model import Point, LineString, LaneBoundary as QxLine, LocalMap, Polygon as QxPolygon, LocalPath, Obstacle
from lasvsim_env.utils.math_utils import ego_predict_model, convert_ego_coord_to_ground_coord
from lasvsim_env.dataclass import EgoVehicle

from tqdm import trange

np.set_printoptions(precision=4, suppress=True)


class IDCTimer:
    def __init__(self):
        self.times = []

    @contextmanager
    def timer(self):
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.times.append(end_time - start_time)

    def get_stats(self):
        if not self.times:
            return {'count': 0, 'mean': 0.0, 'total': 0.0}
        return {
            'count': len(self.times),
            'mean': sum(self.times) / len(self.times),
            'total': sum(self.times),
            'min': min(self.times),
            'max': max(self.times)
        }


def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np(v) for v in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def get_color(type_array):
    ROAD_EDGE   = [1, 0, 0, 0, 0, 0]
    LINE_EDGE   = [0, 1, 0, 0, 0, 0]
    CENTER_LINE = [0, 0, 1, 0, 0, 0]
    STOP_LINE   = [0, 0, 0, 1, 0, 0]
    ZEBRA       = [0, 0, 0, 0, 1, 0]
    VIRTUAL     = [0, 0, 0, 0, 0, 1]

    # Basic color mapping
    if np.array_equal(type_array[:6], ROAD_EDGE):
        return "red"
    elif np.array_equal(type_array[:6], LINE_EDGE):
        return "yellow"
    elif np.array_equal(type_array[:6], CENTER_LINE):
        # When handling CENTER_LINE, check the color information
        color_info = type_array[6:9]
        if np.array_equal(color_info, [1, 0, 0]):  # green
            return "green"
        elif np.array_equal(color_info, [0, 1, 0]):  # red
            return "red"
        elif np.array_equal(color_info, [0, 0, 1]):  # none (Keep the original blue color)
            return "blue"
        else:
            raise ValueError(f"Unknown color encoding: {color_info}")
    elif np.array_equal(type_array[:6], STOP_LINE):
        return "green"
    elif np.array_equal(type_array[:6], ZEBRA):
        return "brown"
    elif np.array_equal(type_array[:6], VIRTUAL):
        return "purple"
    else:
        raise ValueError(f"Unknown type array: {type_array}")

class QxTester:
    def __init__(self, result_dir, task_id, port, qx_act_seq_len, max_steps, print_interval):
        # read token from file
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        token_path = os.path.join(current_file_dir, "..", "..", "VCE-LasVSim", "lasvsim_env", "lasvsim.token")
        with open(token_path, "r") as f:
            token = f.read().strip()

        # create save folder
        self.save_folder = os.path.join(result_dir, datetime.now().strftime(rf"test-%m%d-%H%M"))
        os.makedirs(self.save_folder, exist_ok=True)

        # read data from json
        with open(os.path.join(result_dir, "config.json"), "r") as f:
            self.config = json.load(f)

        self.config["vector_env_num"] = None
        self.config["gym2gymnasium"] = True

        self.config["env_config"]["max_steps"] = max_steps if max_steps is not None else self.config["env_config"]["eval_max_steps"]
        if self.config["env_config"]["qx_act_seq_len"] < qx_act_seq_len:
            qx_act_seq_len = self.config["env_config"]["qx_act_seq_len"]
            print(f"trainning qx_act_seq_len is less than evaluating qx_act_seq_len, set qx_act_seq_len to {qx_act_seq_len}.")
        self.config["env_config"]["qx_act_seq_len"] = 1

        self.config["env_config"]["reset_traffic_flow"] = False

        self.config["lasvsim_config"]["token"] = token
        self.config["lasvsim_config"]["is_testing"] = True
        self.config["lasvsim_config"]["task_id"] = task_id
        self.config["lasvsim_config"]["server_host"] = 'http://localhost:' + str(port)

        with open(os.path.join(self.save_folder, "config.json"), "w") as f:
            json.dump(self.config, f)

        env_config = self.config["env_config"]

        self.action_lower_bound = np.array(env_config["action_lower_bound"])
        self.action_upper_bound = np.array(env_config["action_upper_bound"])
        self.action_center = (self.action_upper_bound + self.action_lower_bound) / 2
        self.action_half_range = (self.action_upper_bound - self.action_lower_bound) / 2

        self.real_action_upper = np.array(env_config["real_action_upper_bound"])
        self.real_action_lower = np.array(env_config["real_action_lower_bound"])

        self.qx_act_seq_len = qx_act_seq_len
        self.print_interval = print_interval

        self.IDC_timer = IDCTimer()

    def rollout_traj(
            self,
            ego: EgoVehicle,
            action_all: np.array,
            Ts: float = 0.1,
            # m, Iz, lf, lr, Cf, Cr, vx_max, vx_min = vehicle_spec
            vehicle_spec: tuple = (1497.0, 3058, 1.60, 1.62, -206369, -206369, 60, 0.0)) -> np.ndarray:

        trajectory = []
        state = np.array([ego.x, ego.y, ego.u, ego.v, ego.phi, ego.w])

        real_action = ego.action
        for i in range(action_all.shape[0]):
            action = action_all[i]
            real_action = self.env.lasvsim_env.get_clip_real_action(action)
            state = ego_predict_model(state, real_action, Ts, vehicle_spec)
            trajectory.append([state[0], state[1], state[4]])

        return np.array(trajectory)

    def set_ego_planning(self, traj):
        path = LocalPath(points=[Point(x=traj[i, 0], y=traj[i, 1]) for i in range(traj.shape[0])], prob=1.0)
        path_list = [asdict(path)]
        self.env.lasvsim_env.simulator.set_vehicle_local_paths(self.env.lasvsim_env.ego_id, path_list)

    def set_map_render(self, obs, ego):
        obs_dict = self.env.lasvsim_env.config["obs_dict"]
        start_index = obs_dict["ego"] + obs_dict["sur_dim"] * obs_dict["sur_history_length"] * obs_dict["sur_num"]
        end_index = self.env.lasvsim_env.obs_dim - obs_dict["navi"]
        map_obs = obs[start_index:end_index].reshape(obs_dict["map_vec_num"], obs_dict["map_vec_dim"]) # (N, 16)

        rel_x, rel_y = map_obs[:, 0], map_obs[:, 1] # (N, )
        rel_phi = np.arctan2(map_obs[:, 5], map_obs[:, 4]) # (N, )

        ground_x, ground_y, ground_phi = convert_ego_coord_to_ground_coord(rel_x, rel_y, rel_phi, ego.x, ego.y, ego.phi) # (N, )

        result_map = LocalMap([], []) # lane_boundaries=Line, junction=Polygon
        for i in range(map_obs.shape[0]):
            color = get_color(map_obs[i, 7:16])
            half_length = map_obs[i, 2]
            cos_phi = float(np.cos(ground_phi[i]))
            sin_phi = float(np.sin(ground_phi[i]))
            result_map.lane_boundaries.append(QxLine(
                LineString([
                    Point(x=ground_x[i] - half_length * cos_phi, y=ground_y[i] - half_length * sin_phi),
                    Point(x=ground_x[i] + half_length * cos_phi, y=ground_y[i] + half_length * sin_phi)
                ]),
                color=color, style="marked" # solid, broken, marked
            ))

        # draw the reference trajectory
        ref_vocabulary = self.env.ref_vocabulary # [R, 2N+1, 4]
        ref_num = ref_vocabulary.shape[0] if ref_vocabulary is not None else 0
        for i in range(ref_num):
            ref = ref_vocabulary[i]
            ref_path = QxLine(
                LineString([Point(x=ref[j, 0], y=ref[j, 1]) for j in range(ref.shape[0])]),
                color="purple", style="broken"
            )
            result_map.lane_boundaries.append(ref_path)

        self.env.lasvsim_env.simulator.set_vehicle_road_perception_info(self.env.lasvsim_env.ego_id,convert_np(asdict(result_map))# rqf
)

    def set_sur_render(self, history_sur_veh, cur_step):
        result_sur = []
        veh_id_list = [sur_veh.veh_id for sur_veh in history_sur_veh[-1]]
        for sur_veh in history_sur_veh[-1]:
            if sur_veh.veh_id not in veh_id_list:
                continue
            result_sur.append(asdict(Obstacle.from_dict({
                "id": sur_veh.veh_id + str(100 + cur_step),
                "base_info": {
                    "length": float(sur_veh.length),
                    "width": float(sur_veh.width),
                    "height": 0.0,
                },
                "moving_info": {
                    "u": float(sur_veh.u),
                },
                "position": {
                    "point": {
                        "x": float(sur_veh.x),
                        "y": float(sur_veh.y),
                    },
                    "heading": float(sur_veh.phi),
                }
            })))

        self.env.lasvsim_env.simulator.set_vehicle_obstacle_perception_info(vehicle_id=self.env.lasvsim_env.ego_id,obstacles=convert_np(result_sur))# rqf

    def get_action(self, env, cur_step, agent: Agent):
        # 1. get info from simulator
        all_vehicle_list = env.simulator.get_vehicle_id_list().get('list')
        all_vehicle_info = env.simulator.get_vehicle_position(all_vehicle_list).get('position_dict')
        all_veh_moving_info = env.simulator.get_vehicle_moving_info(all_vehicle_list)
        all_veh_base_info = env.simulator.get_vehicle_base_info(all_vehicle_list)
        ego_reference_lines = env.simulator.get_vehicle_reference_lines(env.ego_id).get('reference_lines')
        ego_dis_to_link_boundary = env.simulator.get_vehicle_dis_to_link_boundary(env.ego_id) # get left an right distance to current link boundary
        nav = env.simulator.get_idc_vehicle_nav(env.ego_id)
        movement_id = nav.get('next_movement_id') if nav.get('next_movement_id') else None

        try:
            signal = env.simulator.get_movement_signal(movement_id).get('current_signal')
            signal_countdown = env.simulator.get_movement_signal(movement_id).get('countdown')
        except Exception as e:
            print(
                f"[Warning] Failed to get movement signal for movement_id={movement_id}, using default 0. Error: {e}")
            signal = 0
            signal_countdown = 0.0
        signal_ = signal if signal else 0
        signal_countdown_ = signal_countdown if signal_countdown else 0.0

        # 2. get planning result
        lon_acc, ste_wheel ,planning_path = agent.update(
            all_vehicle_info = all_vehicle_info,
            all_veh_moving_info = all_veh_moving_info,
            all_veh_base_info = all_veh_base_info,
            step = cur_step,
            position_info = all_vehicle_info.get(env.ego_id),
            moving_info = all_veh_moving_info.get('moving_info_dict')[env.ego_id],
            reference_path = ego_reference_lines,
            dis_to_link_boundary = ego_dis_to_link_boundary,
            action = self.env.lasvsim_env.lasvsim_context.ego.action,
            movement_signal = signal_,
            signal_countdown = signal_countdown_
        )

        return np.array([lon_acc, ste_wheel]), planning_path

    def run_an_episode(self):
        """
        Run a complete simulation round
        """
        self.IDC_timer = IDCTimer()
        obs, info = self.env.reset()
        reward = 0.0
        terminated = False
        truncated = False

        cur_step = 0
        # Instantiate the simulation object
        veh_config = {
            "decision_interval": 5,
            "ref_spd": 11,
            "ref_spd_inter": 7,
            "forced_lane_change_dis": 80,
            "weight_idm": 0.6,
            "weight_rear": 0.4
        }
        cosim_config = []
        agent = Agent(self.env.lasvsim_env.ego_id, self.env.lasvsim_env.simulator.get_vehicle_base_info(self.env.lasvsim_env.simulator.get_vehicle_id_list().get('list')).get('info_dict')[self.env.lasvsim_env.ego_id].get('base_info'), veh_config, cosim_config)
        while not (terminated or truncated):
            with self.IDC_timer.timer():
                action_all, planning_path = self.get_action(self.env.lasvsim_env, cur_step, agent)
            action = action_all[0:2]
            ego = self.env.lasvsim_env.lasvsim_context.ego
            planning_traj = self.rollout_traj(ego, action_all.reshape(-1, 2)) # shape: (seq_len, 3)

            self.set_map_render(obs, ego)
            self.set_sur_render(self.env.lasvsim_env.history_sur_veh, cur_step)
            self.set_ego_planning(planning_path)

            # sent mertics to qx
            self.env.lasvsim_env.simulator.set_vehicle_extra_metrics(self.env.lasvsim_env.ego_id, {
                "Ego/ xy": f"x: {ego.x:.2f}, y: {ego.y:.2f}",
                "Ego/ phi": f"{ego.phi:.3f} rad, {ego.phi * 180 / np.pi:.1f} deg",
                "Ego/ Yaw rate": f"{ego.w:.3f} rad/s, {ego.w * 180 / np.pi:.1f} deg/s",
                "Ego/ vx": f"{ego.u:.1f} (m/s)",
                "Ego/ rule_act": f"acc: {action_all[0]:.2f}, steer: {action_all[1]:.3f}",
                "Ego/ real_act": f"acc: {ego.action[0]:.2f}, steer: {ego.action[1]:.3f}",
                "Ego/ Bound": f"left: {ego.left_boundary_distance:4.1f}, right: {ego.right_boundary_distance:4.1f}",
                "Navi/ traffic_light": f"{ego.traffic_light} [{ego.light_countdown:.1f}s countdown]",
                "Navi/ dis_to_next_junction": str(ego.dis_to_next_junction),
                "Navi/ flow_direction": str(ego.flow_direction),
                "Navi/ lane_id": ego.lane_id,
                "Navi/ next_movement_id": ego.movement_id,
                "Step": f"{cur_step:d}",
                **{"Sum/ " + k: f"{v:.3f}" for k, v in info.get("qx_sum_info", {}).items()},
                **{"Avg/ " + k: f"{v:.3f}" for k, v in info.get("qx_avg_info", {}).items()},
                **{"Max/ " + k: f"{v:.3f}" for k, v in info.get("qx_max_info", {}).items()},
            })

            obs, _, terminated, truncated, info = self.env.step(action)
            cur_step += 1

        print(f"Ended at:\nstep: {cur_step:4d}, vx: {obs[0]:6.3f}, terminated: {terminated}, truncated: {truncated}.")
        done_info = {k: round(float(v), 3) for k, v in info.get("qx_max_info", {}).items() if float(v) != 0.0}
        print(f"Info: {done_info}")
        self.env.lasvsim_env.simulator.stop()

        # print the metrics of env
        metrics_summary = self.env.lasvsim_env.episode_metrics.get_summary()
        print(f"Episode Metrics Summary: {metrics_summary}")
        stats = self.IDC_timer.get_stats()
        print(f"IDC Time Stats: {stats}")
        print(f"Average IDC time: {stats['mean']:.4f}s")

        return done_info

    def run_n_episodes(self, n=1):
        eval_result_dict = {}
        seed = 0
        for i in trange(n):
            self.env = create_env(**{
                **self.config,
                "env_idx": i
            })
            self.env.eval()
            set_seed(seed, 0, self.env)

            web_url = f"https://qianxing.risenlighten.com/#/sampleRoad/cartest/?id={task_id}&record_id={self.env.lasvsim_env.record_id}&sim_record_id={self.env.lasvsim_env.sim_record_id}"
            with open(os.path.join(self.save_folder, "result_url.csv"), "a") as f:
                f.write(f"{i},"
                        f"{web_url}\n")
            done_info = self.run_an_episode()

            for key in self.env.lasvsim_env.max_keys:
                if key not in eval_result_dict.keys():
                    eval_result_dict[key] = []
                if key in done_info.keys():
                    eval_result_dict[key].append(done_info[key])
                else:
                    eval_result_dict[key].append(0.0)
            print(f"Website of episode {i+1}: {web_url}")

        print("Evaluation is finished!")


if __name__ == "__main__":
    result_dir = os.path.dirname(os.path.abspath(__file__))
    task_id = 14009
    port = 8290
    qx_act_seq_len = 1
    max_steps = 5000
    print_interval = 1

    tester = QxTester(result_dir, task_id, port, qx_act_seq_len, max_steps, print_interval)
    tester.run_n_episodes()
