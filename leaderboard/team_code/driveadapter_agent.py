import os
import json
import datetime
import pathlib
from select import select
import time
from unittest import result
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import pickle
import copy
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent
from leaderboard.utils.route_manipulation import downsample_route
import sys
from team_code.planner import RoutePlanner
from importlib import import_module
SAVE_PATH = os.environ.get('SAVE_PATH', None)
import cv2
import mmcv
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.points import get_points_type
import open_loop_training.code.datasets.carla_dataset as ds_module
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)
def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)



# Taken from World on Rails
class EgoModel():
    def __init__(self, dt=1./4):
        self.dt = dt
        
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb    = -0.090769015
        self.rear_wb     = 1.4178275

        self.steer_gain  = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs, yaws, spds, acts):
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds


def get_entry_point():
    return 'DriveAdapterAgent'

class GlobalConfig:
    def __init__(self, init_dic):
        for k, v in init_dic.items():
            setattr(self, k, v)

def obtain_transform_matrix(x, y, yaw):
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = 1
    sr = 0
    cp = 1
    sp = 0
    mat = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x,],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, 0],
        [0.0, 0.0, 0.0, 1.0],
        ])
    return mat

def InverseRotateVector(x, y, yaw):
    z = 0
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = 1
    sr = 0
    cp = 1
    sp = 0
    out_x = x * (cp * cy) + y * (cp * sy) + z * (sp)
    out_y = x * (cy * sp * sr - sy * cr) + y * (sy * sp * sr + cy * cr) + z * (-cp * sr)
    return out_x, out_y

def obtain_inv_transform_matrix(x, y, yaw):
    x = -x
    y = -y
    x, y = InverseRotateVector(x, y, yaw)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = 1
    sr = 0
    cp = 1
    sp = 0
    inv_mat = np.array([
        [cp * cy, cp * sy, sp, x,],
        [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr, y,],
        [-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr, 0,],
        [0., 0., 0., 1.0],
        ])
    return inv_mat

class DriveAdapterAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        ## Creep - similar to Transfuser
        self.stuck_detector = 0
        self.stuck_threshold = 100.0
        self.is_stuck = False

        ### Generate Route Mask & Visualization
        self._pixels_per_meter = 5.0
        self._width = 192
        self._pixels_ev_to_bottom = 40
        self.ev_mask = np.load("open_loop_training/ev_mask.npy")
        
        self.data_queue = deque()
        self.data_queue_len = 31 ### 20 Hz!!!
        self.pred_len = 4
        self.points_class = get_points_type('LIDAR')
       
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        
        path_to_conf_file = path_to_conf_file.split("+")
        ckpt_path = path_to_conf_file[0]
        config_path = path_to_conf_file[1]
        cfg = Config.fromfile(config_path)
        cfg["model"]["head"]["config"]["rl_ckpt"] = "roach/log/ckpt_11833344.pth"
        cfg["model"]["train_cfg"]["rl_ckpt"] = "roach/log/ckpt_11833344.pth"
        cfg["model"]["test_cfg"]["rl_ckpt"] = "roach/log/ckpt_11833344.pth"
        

        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    plugin_dir = os.path.join("open_loop_training", plugin_dir)
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        self.device = "cuda:0"
        print(ckpt_path)
        checkpoint = load_checkpoint(self.model, ckpt_path, map_location='cpu', strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        ## For mmcv to preprocess single frames
        self.test_pipeline = []
        cfg.val_full_queue_pipeline[0].cfg.use_depth = False ##No gt for testing
        cfg.val_full_queue_pipeline[0].cfg.use_seg = False ##No gt for testing
        for test_pipeline in cfg.test_pipeline:
            if test_pipeline["type"] not in ["LoadMultiImages", "LoadPoints", 'LoadDepth', 'LoadSeg']:
                self.test_pipeline.append(test_pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        ## For mmcv to preprocess temporal information
        self.seq_test_pipeline = Compose(cfg.val_full_queue_pipeline)

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
            if len(path_to_conf_file) > 2:
                string += "_index" + str(path_to_conf_file[2]) + "_" + str(path_to_conf_file[3])
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'meta').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_front').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_left').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_right').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_back').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'topdown').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'lidar').mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.folder_name = string

        self.ego_model = EgoModel(dt=1.0 / 20.0)
        self.gps_buffer = deque(maxlen=100)
        print("Finish Setup Up!!!", flush=True)


    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.prev_lidar = None
        self.prev_matrix = None
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        self._ego_vehicle = CarlaDataProvider.get_ego() ## Only to turn on vehicle light
        self._ego_vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.All))
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.5, 'y': 0.0, 'z':2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'rgb_front'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0, 'y': -0.3, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0, 'y': 0.3, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -1.6, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'rgb_back'
                    },
                {   'type': 'sensor.lidar.ray_cast',
                    'x': 0.0, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'id': 'lidar'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    },
                ### Debug sensor, not used by the model
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 192, 'height': 192, 'fov': 5 * 10.0,
                    'id': 'topdown'
                    },	
                ]
    def tick(self, input_data):
        self.step += 1
        
        topdown = cv2.cvtColor(input_data['topdown'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_back = cv2.cvtColor(input_data['rgb_back'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        acceleration = input_data['imu'][1][:3]
        angular_velocity = input_data['imu'][1][3:6]
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        
        result = {
                'rgb_front': rgb_front,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'rgb_back': rgb_back,
                'gps': gps,
                'speed': speed,
                'theta': compass,
                "acceleration":acceleration,
                "angular_velocity":angular_velocity,
                "topdown": topdown,
                "acceleration":acceleration.tolist(),
                "angular_velocity":angular_velocity.tolist()
                }
        
        pos = self._get_position(result)
        self.gps_buffer.append(pos)
        pos = np.average(self.gps_buffer, axis=0)
        self._truncate_global_route_till_local_target(carla.Location(x=pos[1], y=-pos[0]))


        result["x"] = pos[0]
        result["y"] = pos[1]
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value
        result['x_target'] = next_wp[0]
        result['y_target'] = next_wp[1]

        now_lidar = input_data['lidar'][1]
        if self.prev_lidar is not None:
            now_inv_mat = obtain_inv_transform_matrix(pos[1], -pos[0], compass-np.pi/2)
            relative_transform_mat = np.dot(now_inv_mat , np.array(self.prev_matrix)) #4 * 4
            transformed_prev_lidar_xyz = np.concatenate([self.prev_lidar[:, :3], np.ones((self.prev_lidar.shape[0], 1))], axis=1) # N * 4
            transformed_prev_lidar_xyz = np.einsum("ij,kj->ki", relative_transform_mat, transformed_prev_lidar_xyz)
            transformed_prev_lidar_xyz = np.concatenate([transformed_prev_lidar_xyz[:, :3], self.prev_lidar[:, 3][:, np.newaxis]], axis=1)
            saved_lidar = np.concatenate([transformed_prev_lidar_xyz, now_lidar], axis=0).copy()
            saved_lidar[:, 2] += 2.5
            self.now_inv_mat = now_inv_mat
        else:
            saved_lidar = now_lidar.copy()
            saved_lidar[:, 2] += 2.5
            
            
        result["lidar"] = saved_lidar.astype(np.float32)
        self.prev_lidar = now_lidar
        self.prev_matrix = obtain_transform_matrix(pos[1], -pos[0], compass-np.pi/2)
        return result
    
    def offset_then_rotate(self, target_2d_world_coor, ref_2d_wolrd_coor, ref_yaw):
        final_coor = target_2d_world_coor - ref_2d_wolrd_coor
        R = np.array([
            [np.cos(ref_yaw), -np.sin(ref_yaw)],
            [np.sin(ref_yaw), np.cos(ref_yaw)]
        ])
        return np.einsum("ij,kj->ki", R.T, final_coor)

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        ################# Preprocess ################
        results = {}
        ego_theta = tick_data["theta"]   if not np.isnan(tick_data["theta"]) else 0
        ego_theta = ego_theta - np.pi/2
        results["input_theta"] = ego_theta

        results["input_x"] = tick_data["y"]
        results["input_y"] = -tick_data["x"]
        ego_xy = np.stack([results["input_x"], results["input_y"]], axis=-1)
        results["speed"] = tick_data["speed"]
        results["can_bus"] = np.zeros(18)
        results["can_bus"][0] = results["input_x"] #Gloabal
        results["can_bus"][1] = results["input_y"] #Global
        accel = np.array(tick_data["acceleration"])
        accel[:2] = self.offset_then_rotate(np.array(accel[:2])[np.newaxis, :], np.array([0, 0]), ego_theta).squeeze(0)
        results["can_bus"][7:10] = accel
        results["can_bus"][10:13] = tick_data["angular_velocity"]
        results["can_bus"][13] = tick_data["speed"]
        results["can_bus"][-2] = ego_theta
        results["can_bus"][-1] = ego_theta / np.pi * 180
        
        results['target_point'] = self.offset_then_rotate(np.array([[tick_data["y_target"], -tick_data["x_target"]]]), ego_xy, ego_theta).squeeze(0)

        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        results['target_command_raw'] = torch.tensor(command).long()
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        results['target_command'] = torch.tensor(cmd_one_hot)

        ## Inference mode, all empty
        results['waypoints'] = np.zeros(4)
        results["action"] = np.zeros(3)
        results["action_mu"] = np.zeros(2)
        results["action_sigma"] = np.zeros(2)
        results['future_action_mu'] = np.zeros((self.pred_len, 2))
        results['future_action_sigma'] = np.zeros((self.pred_len, 2))        
        results['future_action'] = np.zeros((self.pred_len, 3)) 
        results["value"] = 0
        results["feature"] = np.zeros(1)
        results["future_feature"] = np.zeros((self.pred_len, 1))
        results["bev_seg_label"] = np.zeros(1)
        results["img"] = [tick_data[camera_name] for camera_name in self.cfg["camera_list"]]

        points = tick_data["lidar"]
        points = self.points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results["points"] = points
        results = self.test_pipeline(results)

        if len(self.data_queue) >= self.data_queue_len:
            self.data_queue.popleft()
        self.data_queue.append(results)
        

        if self.step < self.data_queue_len:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            self.update_gps_buffer(control, tick_data['theta'], tick_data['speed'])
            self.prev_control = np.array([control.throttle, control.steer, control.brake, 1.0])
            return control
        
        ## Preprocess
        selected_index_lis = []
        for selected_index in self.cfg["cfg"]["history_query_index_lis"][:-1]:
            selected_index = selected_index * 10 ## train in 2Hz Simulator is in 20 Hz
            selected_index_lis.append(selected_index-1)
        selected_index_lis.append(-1)
        input_data_queue = [copy.deepcopy(self.data_queue[selected_index]) for selected_index in selected_index_lis]
        input_data_queue[-2]["previous_one_action"] = self.prev_control.copy()
        input_data_union = ds_module.union2one(self.seq_test_pipeline, input_data_queue)
        input_data_batch = mm_collate_to_batch_form([input_data_union], samples_per_gpu=1)
        input_data_batch["img"] = input_data_batch["img"].data[0]
        if len(selected_index_lis) == 1:
            input_data_batch["img"] = input_data_batch["img"].squeeze(0)
        input_data_batch["points"] = input_data_batch["points"].data[0]
        input_data_batch["img_metas"] = input_data_batch["img_metas"].data[0]
        for _ in input_data_batch:
            if torch.is_tensor(input_data_batch[_]):
                input_data_batch[_] = input_data_batch[_].to(self.device)

        with torch.no_grad():
            route_mask = self.generate_route_mask()
            route_mask = torch.from_numpy(route_mask).to(input_data_batch["img"].device).unsqueeze(0)
            pred = self.model.forward_inference(input_data_batch, route_mask)
            roach_bev_input_numpy = pred["roach_bev_input"].detach().cpu().numpy()[0]
            gt_velocity = torch.FloatTensor([tick_data['speed']]).to(self.device, dtype=torch.float32)
           
            steer_ctrl, throttle_ctrl, brake_ctrl = self.model.process_action(pred)
            steer_traj, throttle_traj, brake_traj, metadata_traj = self.model.control_pid(pred['pred_wp'], gt_velocity,  results['target_point'])
            if brake_traj < 0.05: brake_traj = 0.0
            if throttle_traj > brake_traj: brake_traj = 0.0

            overall_pred_is_accel = ((throttle_traj>0) or (throttle_ctrl>0) or (brake_traj<0.5) or (brake_ctrl<0.5))
            overall_pred_is_brake = ((brake_ctrl>0.) or (brake_traj>0.))
            control = carla.VehicleControl()
            control.steer = np.clip(steer_ctrl, -1, 1)
            is_turn = False
            if abs(control.steer) > 0.07: ## In turning
                is_turn = True
                speed_threshold = 2.5 ## Avoid stuck during turning
            else:
                speed_threshold = 3.0 ## Avoid pass stop/red light/collision

            if overall_pred_is_brake:
                control.brake=1.0
                control.throttle=0.0
            else:
                control.brake=0.0
                control.throttle=0.75
            is_stuck = False
            # By transfuser
            if(self.stuck_detector > self.stuck_threshold):
                is_stuck = True
                if overall_pred_is_accel:
                    control.brake = 0.0
                    control.throttle = max(max(throttle_traj, throttle_ctrl), 0.6)
                    control.steer = control.steer * 0.25
                else:
                    control.brake = 1.0
                    control.throttle = 0.0
            if(float(gt_velocity) < 0.5): # just an arbitrary low number to threshhold when the car is stopped
                self.stuck_detector += 1
            elif(float(gt_velocity) > 0.5):
                self.stuck_detector = 0
      
            if float(tick_data['speed']) > speed_threshold:
                max_throttle = 0.05
            else:
                max_throttle = 0.75
            control.throttle = np.clip(control.throttle, a_min=0.0, a_max=max_throttle)
            if control.brake > 0:
                control.brake = 1.0

        if SAVE_PATH is not None and self.step % 10 == 0:
            metadata = metadata_traj
            metadata['speed'] = float(tick_data["speed"].astype(np.float64))
            metadata["steer"] = float(control.steer)
            metadata["throttle"] = float(control.throttle)
            metadata["brake"] = float(control.brake)
            metadata['steer_ctrl'] = float(steer_ctrl)
            metadata['steer_traj'] = float(steer_traj)
            metadata['throttle_ctrl'] = float(throttle_ctrl)
            metadata['throttle_traj'] = float(throttle_traj)
            metadata['brake_ctrl'] = float(brake_ctrl)
            metadata['brake_traj'] = float(brake_traj)
            metadata['is_brake'] = float(overall_pred_is_brake)
            metadata['is_accel'] = float(overall_pred_is_accel)
            metadata["is_stuck"] = float(is_stuck)
            metadata["stuck_detector"] = self.stuck_detector
            
            visualized_roach_input = self.visualize_roach_input(road_mask=roach_bev_input_numpy[0].astype(bool), route_mask=roach_bev_input_numpy[1].astype(bool), lane_mask=(roach_bev_input_numpy[2]==1.0).astype(bool),lane_mask_broken=(roach_bev_input_numpy[2]==0.5).astype(bool), tl_green_masks=(roach_bev_input_numpy[11:15]==0.3137).astype(bool), tl_yellow_masks=(roach_bev_input_numpy[11:15]==0.6667).astype(bool), tl_red_masks=(roach_bev_input_numpy[11:15]==1.0).astype(bool), vehicle_masks=roach_bev_input_numpy[3:7].astype(bool), walker_masks=roach_bev_input_numpy[7:11].astype(bool))
            self.save(tick_data, metadata,  visualized_roach_input = visualized_roach_input)
        self.update_gps_buffer(control, tick_data['theta'], tick_data['speed'])
        self.prev_control = np.array([control.throttle, control.steer, control.brake, 1.0])
        self.is_stuck = is_stuck
        return control

    def save(self, tick_data, metadata, visualized_roach_input):
        frame = self.step // 10

        Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        #Image.fromarray(tick_data['rgb_left']).save(self.save_path / 'rgb_left' / ('%04d.png' % frame))
        #Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / ('%04d.png' % frame))
        #Image.fromarray(tick_data['rgb_back']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        Image.fromarray(np.concatenate([tick_data['topdown'], visualized_roach_input], axis=1)).save(self.save_path / 'topdown' / ('%04d.png' % frame))
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(metadata, outfile, indent=4)
        outfile.close()
        #np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), tick_data["lidar"].astype(np.float32), allow_pickle=True)

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()
    def update_gps_buffer(self, control, theta, speed):
        yaw = np.array([(theta - np.pi/2.0)])
        speed = np.array([speed])
        action = np.array(np.stack([control.steer, control.throttle, control.brake], axis=-1))
        #Update gps locations
        for i in range(len(self.gps_buffer)):
            loc =self.gps_buffer[i]
            loc_temp = np.array([loc[1], -loc[0]]) #Bicycle model uses a different coordinate system
            next_loc_tmp, _, _ = self.ego_model.forward(loc_temp, yaw, speed, action)
            next_loc = np.array([-next_loc_tmp[1], next_loc_tmp[0]])
            self.gps_buffer[i] = next_loc
        return None
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp_route):
        """
        Set the plan (route) for the agent
        """
        self._global_route = wp_route
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
    
    def _truncate_global_route_till_local_target(self, ev_location, windows_size=5):
        #ev_location = self._ego_vehicle.get_location()
        closest_idx = 0
        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break

            loc0 = self._global_route[i][0].transform.location
            loc1 = self._global_route[i+1][0].transform.location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i+1
        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].transform.location)
        self._global_route = self._global_route[closest_idx:]

    def generate_route_mask(self):
        route_mask = np.zeros([192, 192], dtype=np.uint8)
        forward_vec = np.array([np.cos(0), np.sin(0)])
        right_vec = np.array([np.cos(0 + 0.5*np.pi), np.sin(0 + 0.5*np.pi)])
        bottom_left = - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
        top_left =  (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
        top_right = (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec
        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width-1],
                        [0, 0],
                        [self._width-1, 0]], dtype=np.float32)
        M_warp = cv2.getAffineTransform(src_pts, dst_pts)
        route_in_world = np.array([[wp.transform.location.x, wp.transform.location.y]
                                for wp, _ in self._global_route[0:min(80, len(self._global_route))]])
        route_in_ego = np.einsum("ij,kj->ki", self.now_inv_mat, np.stack([route_in_world[:, 0], route_in_world[:, 1], np.zeros(route_in_world.shape[0]), np.ones(route_in_world.shape[0])], axis=1))[:, :2]
        route_warped = cv2.transform(route_in_ego[:, np.newaxis, :] * self._pixels_per_meter, M_warp)
        cv2.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        return route_mask.astype(np.float32)

    def visualize_roach_input(self, road_mask, route_mask, lane_mask, lane_mask_broken, tl_green_masks, tl_yellow_masks, tl_red_masks, vehicle_masks, walker_masks):
        image = np.zeros([192, 192, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[route_mask] = COLOR_ALUMINIUM_3
        image[lane_mask] = COLOR_MAGENTA
        image[lane_mask_broken] = COLOR_MAGENTA_2
        h_len = 3
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len-i)*0.2)
        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len-i)*0.2)
        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len-i)*0.2)
        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)
        image[self.ev_mask] = COLOR_WHITE
        return image