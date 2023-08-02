import os
import carla

import json
import datetime
import pathlib
import time
import cv2
from collections import deque
import random

import torch
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
import numpy as np
from omegaconf import OmegaConf
import copy

from roach.criteria import run_stop_sign
from roach.obs_manager.birdview.chauffeurnet import ObsManager
from roach.utils.config_utils import load_entry_point
import roach.utils.transforms as trans_utils
from roach.utils.expert_noiser import ExpertNoiser
from roach.utils.traffic_light import TrafficLightHandler
import io
from io import BytesIO

def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()

def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import downsample_route
from agents.navigation.local_planner import RoadOption

from team_code.planner import RoutePlanner
def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

SAVE_PATH = os.environ.get('SAVE_PATH', None)
conf_path = '~/petreloss.conf'



def get_entry_point():
    return 'ROACHAgent'

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])
    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)
    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def get_xyz(_):
    return np.array([_.x, _.y, _.z])


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1
    if abs(np.linalg.det(A)) < 1e-3:
        return False, None
    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

    return collides, p1 + x[0] * v1


def get_nearby_object(vehicle_position, actor_list, radius):
    nearby_objects = []
    for actor in actor_list:
        trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
        trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
        if (trigger_box_global_pos.distance(vehicle_position) < radius):
            nearby_objects.append(actor)
    return nearby_objects

class ROACHAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, ckpt="roach/log/ckpt_11833344.pth"):
        self.is_local = (os.environ['IS_LOCAL'] == "True") ## Ignore non-local situation since we collect data in a cluster with ceph
        if not self.is_local:
            from petrel_client.client import Client
            self.client = Client(conf_path)
        
        self._render_dict = None
        self.supervision_dict = None
        self._ckpt = ckpt
        cfg = OmegaConf.load(path_to_conf_file)
        cfg = OmegaConf.to_container(cfg)
        self.cfg = cfg
        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']
        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        self._policy_kwargs = cfg['policy']['kwargs']
        if self._ckpt is None:
            self._policy = None
        else:
            self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
            self._policy = self._policy.eval()
        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.prev_lidar = None ## The frequency of lidar is 10 Hz while the frequency of simulation is 20Hz -> In each frame, the sensor only returns half lidar points.
        self._active_traffic_light = None
        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            time_string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += time_string
            if self.is_local:
                self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
                self.save_path.mkdir(parents=True, exist_ok=False)
                (self.save_path / '3d_bbs').mkdir(parents=True, exist_ok=True)

                (self.save_path / 'rgb_front').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'rgb_left').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'rgb_right').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'rgb_back').mkdir(parents=True, exist_ok=True)

                (self.save_path / 'seg_front').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'seg_left').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'seg_right').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'seg_back').mkdir(parents=True, exist_ok=True)

                (self.save_path / 'depth_front').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'depth_left').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'depth_right').mkdir(parents=True, exist_ok=True)
                (self.save_path / 'depth_back').mkdir(parents=True, exist_ok=True)

                (self.save_path / 'lidar').mkdir(parents=True, exist_ok=True)

                (self.save_path / 'measurements').mkdir()
                (self.save_path / 'supervision').mkdir()
                (self.save_path / 'bev').mkdir() ## Visualization of Scene for debug
                (self.save_path / 'bev_seg_label').mkdir()
            else:
                 ## Ignore non-local situation since we collect data in a cluster with ceph
                self.save_path = "s3://CarlaData/Multiple/" + os.environ['ROUTE_FILE'] + "/" + string
            self.folder_name = string

    def _init(self):
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)

        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self._world = CarlaDataProvider.get_world()
        self._map = self._world.get_map()

        self._ego_vehicle = CarlaDataProvider.get_ego()

        self._last_route_location = self._ego_vehicle.get_location()
        self._criteria_stop = run_stop_sign.RunStopSign(self._world)
        self.birdview_obs_manager = ObsManager(self.cfg['obs_configs']['birdview'], self._criteria_stop)
        self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)

        self.navigation_idx = -1
        # for stop signs
        self._target_stop_sign = None # the stop sign affecting the ego vehicle
        self._stop_completed = False # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign
        TrafficLightHandler.reset(self._world)
        print("initialized")
        self.initialized = True

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle
        return angle


    def _truncate_global_route_till_local_target(self, windows_size=5):
        ev_location = self._ego_vehicle.get_location()
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

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps

    def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp_route):
        """
        Set the plan (route) for the agent
        """
        self._global_route = wp_route
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._plan_gps_HACK = global_plan_gps
        self._plan_HACK = global_plan_world_coord

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
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z':2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'seg_front'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0, 'y': -0.3, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'seg_left'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0, 'y': 0.3, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'seg_right'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -1.6, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'seg_back'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z':2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'depth_front'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 0, 'y': -0.3, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'depth_left'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 0, 'y': 0.3, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'depth_right'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -1.6, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 150,
                    'id': 'depth_back'
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
                    }
                ]

    def tick(self, input_data, timestamp):
        self._truncate_global_route_till_local_target()

        birdview_obs = self.birdview_obs_manager.get_observation(self._global_route)
        control = self._ego_vehicle.get_control()
        throttle = np.array([control.throttle], dtype=np.float32)
        steer = np.array([control.steer], dtype=np.float32)
        brake = np.array([control.brake], dtype=np.float32)
        gear = np.array([control.gear], dtype=np.float32)

        ev_transform = self._ego_vehicle.get_transform()
        vel_w = self._ego_vehicle.get_velocity()
        vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)
        vel_xy = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)
        self._criteria_stop.tick(self._ego_vehicle, timestamp)

        state_list = []
        state_list.append(throttle)
        state_list.append(steer)
        state_list.append(brake)
        state_list.append(gear)
        state_list.append(vel_xy)
        state = np.concatenate(state_list)
        obs_dict = {
            'state': state.astype(np.float32),
            'birdview': birdview_obs['masks'],
        }
        # Roach Input: Road Mask - 0, Route Mask - 1, Lane_Mask (broken lane - 0.5) -2, 4 * vehicle - 3456, 4 * walker, 4 * traffic_light [-16, -11, -6, -1] 10hz
        bev_seg_label = birdview_obs['masks'].copy().astype(np.float32)/255.0

        rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_back = cv2.cvtColor(input_data['rgb_back'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        acceleration = input_data['imu'][1][:3]
        angular_velocity = input_data['imu'][1][3:6]
        target_gps, target_command = self.get_target_gps(input_data['gps'][1], compass)
        weather = self._weather_to_dict(self._world.get_weather())

        seg_front = np.copy(input_data["seg_front"][1][:, :, 2])
        seg_left = np.copy(input_data["seg_left"][1][:, :, 2])
        seg_right = np.copy(input_data["seg_right"][1][:, :, 2])
        seg_back = np.copy(input_data["seg_back"][1][:, :, 2])

        depth_front = cv2.cvtColor(input_data['depth_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_left = cv2.cvtColor(input_data['depth_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_right = cv2.cvtColor(input_data['depth_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_back = cv2.cvtColor(input_data['depth_back'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        result = {
                'rgb_front': rgb_front,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'rgb_back': rgb_back,
                'seg_front': seg_front,
                'seg_left': seg_left,
                'seg_right': seg_right,
                'seg_back': seg_back,
                'depth_front': depth_front,
                'depth_left': depth_left,
                'depth_right': depth_right,
                'depth_back': depth_back,
                'lidar' : input_data['lidar'][1],
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'weather': weather,
                "acceleration":acceleration,
                "angular_velocity":angular_velocity,
                "bev_seg_label": bev_seg_label
                }
        next_wp, next_cmd = self._route_planner.run_step(self._get_position(result))
        result['next_command'] = next_cmd.value
        result['x_target'] = next_wp[0]
        result['y_target'] = next_wp[1]
        return result, obs_dict, birdview_obs['rendered'], target_gps, target_command, None

    def im_render(self, render_dict):
        im_birdview = render_dict['rendered']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview
        action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
        txt_1 = f'a{action_str}'
        im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        debug_texts = [
            'should_brake: ' + render_dict['should_brake'],
        ]
        for i, txt in enumerate(debug_texts):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1

        if self.step < 20:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            self.last_control = control
            self.prev_lidar = input_data['lidar'][1]
            self.prev_matrix = self._ego_vehicle.get_transform().get_matrix()
            return control

        if self.step % 2 != 0:
            self.prev_lidar = input_data['lidar'][1]
            self.prev_matrix = self._ego_vehicle.get_transform().get_matrix()
            return self.last_control
        tick_data, policy_input, rendered, target_gps, target_command, _ = self.tick(input_data, timestamp)

        gps = self._get_position(tick_data)
        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)
        ## Roach forward
        actions, values, log_probs, mu, sigma, features, cnn_feature = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        control = self.process_act(actions)
        render_dict = {"rendered": rendered, "action": actions}
        ## Rules for emergency brake 
        should_brake = self.collision_detect()
        only_ap_brake = True if (control.brake <= 0 and should_brake) else False
        if should_brake:
            control.steer = control.steer * 0.5
            control.throttle = 0.0
            control.brake = 1.0
        render_dict = {"rendered": rendered, "action": actions, "should_brake":str(should_brake),}

        render_img = self.im_render(render_dict)
        supervision_dict = {
            'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
            'value': values[0],
            'action_mu': mu[0],
            'action_sigma': sigma[0],
            'features': features[0],
            'cnn_features':cnn_feature,
            'speed': tick_data['speed'],
            'target_gps': target_gps,
            'target_command': target_command,
            'should_brake': should_brake,
            'only_ap_brake': only_ap_brake,
        }


        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(near_node, far_node, near_command, far_command, tick_data, supervision_dict, render_img, should_brake)

        steer = control.steer
        control.steer = steer + 1e-2 * np.random.randn() ## Random noise for robustness
        self.last_control = control
        self.prev_lidar = input_data['lidar'][1]
        self.prev_matrix = self._ego_vehicle.get_transform().get_matrix()
        return control

    def collision_detect(self):
        actors = self._world.get_actors()
        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))
        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        return any(x is not None for x in [vehicle, walker])

    def _is_walker_hazard(self, walkers_list):
        z = self._ego_vehicle.get_location().z
        p1 = _numpy(self._ego_vehicle.get_location())
        v1 = 10.0 * _orientation(self._ego_vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))
            if s2 < 0.05:
                v2_hat *= s2
            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat
            collides, collision_point = get_collision(p1, v1, p2, v2)
            if collides:
                return walker
        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._ego_vehicle.get_location().z
        o1 = _orientation(self._ego_vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._ego_vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._ego_vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._ego_vehicle.id:
                continue
            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat
            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue
            return target_vehicle
        return None

    def save(self, near_node, far_node, near_command, far_command, tick_data, supervision_dict, render_img, should_brake):
        frame = self.step // 10 - 2
        if self.is_local:
            Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
            Image.fromarray(tick_data['rgb_left']).save(self.save_path / 'rgb_left' / ('%04d.png' % frame))
            Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / ('%04d.png' % frame))
            Image.fromarray(tick_data['rgb_back']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
            Image.fromarray(tick_data['seg_front']).save(self.save_path / 'seg_front' / ('%04d.png' % frame))
            Image.fromarray(tick_data['seg_left']).save(self.save_path / 'seg_left' / ('%04d.png' % frame))
            Image.fromarray(tick_data['seg_right']).save(self.save_path / 'seg_right' / ('%04d.png' % frame))
            Image.fromarray(tick_data['seg_back']).save(self.save_path / 'seg_back' / ('%04d.png' % frame))
            Image.fromarray(tick_data['depth_front']).save(self.save_path / 'depth_front' / ('%04d.png' % frame))
            Image.fromarray(tick_data['depth_left']).save(self.save_path / 'depth_left' / ('%04d.png' % frame))
            Image.fromarray(tick_data['depth_right']).save(self.save_path / 'depth_right' / ('%04d.png' % frame))
            Image.fromarray(tick_data['depth_back']).save(self.save_path / 'depth_back' / ('%04d.png' % frame))
            Image.fromarray(render_img).save(self.save_path / 'bev' / ('%04d.png' % frame))
            np.save(self.save_path / 'bev_seg_label' / ('%04d.npy' % frame), tick_data['bev_seg_label'], allow_pickle=True)
            with open(self.save_path / 'supervision' / ('%04d.npy' % frame), 'wb') as f:
                np.save(f, supervision_dict)
        else:
            for key_name in ["rgb_front", "rgb_left", "rgb_right", "rgb_back", "seg_front", "seg_left", "seg_right", "seg_back", "depth_front", "depth_left", "depth_right", "depth_back", "topdown"]:
                self.client.put(os.path.join(self.save_path, key_name, '%04d.png' % frame), tick_data[key_name].tostring())
            self.client.put(os.path.join(self.save_path, "bev_seg_label", '%04d.npy' % frame), tick_data['bev_seg_label'].tostring())
            self.client.put(os.path.join(self.save_path, "supervision", '%04d.npy' % frame), array_to_bytes(supervision_dict))

        
        current_inv_mat = np.array(self._ego_vehicle.get_transform().get_inverse_matrix())
        relative_transform_mat = np.dot(current_inv_mat , np.array(self.prev_matrix)) #4 * 4
        transformed_prev_lidar_xyz = np.concatenate([self.prev_lidar[:, :3], np.ones((self.prev_lidar.shape[0], 1))], axis=1) # N * 4
        transformed_prev_lidar_xyz = np.einsum("ij,kj->ki", relative_transform_mat, transformed_prev_lidar_xyz)
        transformed_prev_lidar_xyz = np.concatenate([transformed_prev_lidar_xyz[:, :3], self.prev_lidar[:, 3][:, np.newaxis]], axis=1)
        saved_lidar = np.concatenate([transformed_prev_lidar_xyz, tick_data['lidar']], axis=0)
        saved_lidar[:, 2] += 2.5 #offset lidar z
     
        actor_lis = self._get_3d_bbs(lidar=saved_lidar, max_distance=50)
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']
        weather = tick_data['weather']
        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'x_command_far': far_node[0],
                'y_command_far': far_node[1],
                'command_far': far_command.value,
                'x_command_near': near_node[0],
                'y_command_near': near_node[1],
                'command_near': near_command.value,
                'should_brake': should_brake,
                'x_target': tick_data['x_target'],
                'y_target': tick_data['y_target'],'target_command': tick_data['next_command'],
                'weather': weather,
                "acceleration":tick_data["acceleration"].tolist(),
                "angular_velocity":tick_data["angular_velocity"].tolist()
                }

        if self.is_local:
            outfile = open(self.save_path / '3d_bbs' / ('%04d.json' % frame), 'w')
            json.dump(actor_lis, outfile, indent=4, default=np_encoder)
            outfile.close()

            outfile = open(self.save_path / 'measurements' / ('%04d.json' % frame), 'w')
            json.dump(data, outfile, indent=4)
            outfile.close()

            np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), saved_lidar.astype(np.float32), allow_pickle=True)
        else:
            self.client.put(os.path.join(self.save_path, '3d_bbs', '%04d.json' % frame), json.dumps(actor_lis, default=np_encoder).encode('utf-8'))
            self.client.put(os.path.join(self.save_path, 'measurements', '%04d.json' % frame), json.dumps(data, default=np_encoder).encode('utf-8'))
            self.client.put(os.path.join(self.save_path, 'lidar', '%04d.npy' % frame), array_to_bytes(saved_lidar))


    def get_target_gps(self, gps, compass):
        # target gps
        def gps_to_location(gps):
            lat, lon, z = gps
            lat = float(lat)
            lon = float(lon)
            z = float(z)

            location = carla.Location(z=z)
            xy =  (gps[:2] - self._command_planner.mean) * self._command_planner.scale
            location.x = xy[0]
            location.y = -xy[1] ## Dose not matter, because no explicit judge left or right
            return location
        global_plan_gps = self._global_plan
        next_gps, _ = global_plan_gps[self.navigation_idx+1]
        next_gps = np.array([next_gps['lat'], next_gps['lon'], next_gps['z']])
        next_vec_in_global = gps_to_location(next_gps) - gps_to_location(gps)
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
        loc_in_ev = trans_utils.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        if np.sqrt(loc_in_ev.x**2+loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
            self.navigation_idx += 1

        self.navigation_idx = min(self.navigation_idx, len(global_plan_gps)-2)

        _, road_option_0 = global_plan_gps[max(0, self.navigation_idx)]
        gps_point, road_option_1 = global_plan_gps[self.navigation_idx+1]
        gps_point = np.array([gps_point['lat'], gps_point['lon'], gps_point['z']])

        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0

        return np.array(gps_point, dtype=np.float32), np.array([road_option.value], dtype=np.int8)


    def process_act(self, action):
        acc = action[0][0]
        steer = action[0][1]
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control

    def _weather_to_dict(self, carla_weather):
        weather = {
            'cloudiness': carla_weather.cloudiness,
            'precipitation': carla_weather.precipitation,
            'precipitation_deposits': carla_weather.precipitation_deposits,
            'wind_intensity': carla_weather.wind_intensity,
            'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
            'sun_altitude_angle': carla_weather.sun_altitude_angle,
            'fog_density': carla_weather.fog_density,
            'fog_distance': carla_weather.fog_distance,
            'wetness': carla_weather.wetness,
            'fog_falloff': carla_weather.fog_falloff,
        }
        return weather

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._ego_vehicle.get_velocity()
        if not transform:
            transform = self._ego_vehicle.get_transform()
        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def get_points_in_bbox(self, ego_mat, vec_inv_mat, dx, lidar):
        transform_mat = np.dot(vec_inv_mat, ego_mat)
        lidar =  np.concatenate([lidar[:, :3], np.ones((lidar.shape[0], 1))], axis=1)
        lidar = np.einsum("ij,kj->ki", transform_mat, lidar)
        # check points in bbox
        x, y, z = dx / 2.
        num_points = ((lidar[:, 0] < x) & (lidar[:, 0] > -x) &
                    (lidar[:, 1] < y) & (lidar[:, 1] > -y) &
                    (lidar[:, 2] < z) & (lidar[:, 2] > -z)).sum()
        return num_points

    def _get_3d_bbs(self, lidar=None, max_distance=50):
        results = []
        ego_rotation = self._ego_vehicle.get_transform().rotation
        ego_matrix = np.array(self._ego_vehicle.get_transform().get_matrix()).astype(np.float32)
        ego_extent = self._ego_vehicle.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]).astype(np.float32) * 2.
        ego_yaw =  ego_rotation.yaw/180.0*np.pi
        # also add ego box for visulization
        relative_rotation = [0, 0, 0]
        relative_pos = [0, 0, 0]
        # add vehicle velocity and brake flag
        ego_transform = self._ego_vehicle.get_transform()
        ego_control   = self._ego_vehicle.get_control()
        ego_velocity  = self._ego_vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity) # In m/s
        ego_brake = ego_control.brake
        ego_inv_mat = np.array(self._ego_vehicle.get_transform().get_inverse_matrix()).astype(np.float32)
        ego_loc = self._ego_vehicle.get_location()
        # the position is in lidar coordinates
        result = {"class": "vehicle",
                  "extent": [ego_dx[0], ego_dx[1], ego_dx[2],],
                  "relative_position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  #roll pitch yaw
                  "relative_rotation": relative_rotation,
                  "yaw":ego_yaw,
                  "num_points": -1,
                  "distance": -1,
                  "speed": ego_speed,
                  "brake": ego_brake,
                  "id": int(self._ego_vehicle.id),
                  'ego_matrix': ego_matrix.tolist(),
                  "ego_inv_matrix": ego_inv_mat.tolist(),
                }
        results = {}
        results["ego"] = result
        results["vehicle"+str(self._ego_vehicle.id)] = result

        self._actors = self._world.get_actors()
        for actor_type in ["vehicle", "walker", "traffic_light", "stop"]:
            selected_actors = self._actors.filter('*' + actor_type + '*')
            for selected_actor in selected_actors:
                if (selected_actor.get_location().distance(ego_loc) < max_distance):
                    if (actor_type != "vehicle") or (selected_actor.id != self._ego_vehicle.id):
                        selected_actor_rotation = selected_actor.get_transform().rotation
                        selected_actor_matrix = np.array(selected_actor.get_transform().get_matrix())
                        selected_actor_inv_matrix = np.array(selected_actor.get_transform().get_inverse_matrix())
                        selected_actor_id = selected_actor.id
                        if hasattr(selected_actor, 'bounding_box'):
                            selected_actor_extent = selected_actor.bounding_box.extent
                            dx = np.array([selected_actor_extent.x, selected_actor_extent.y, selected_actor_extent.z]).astype(np.float32) * 2.
                        else:
                            dx = np.array([0.5, 0.5, 2]) * 2.
                        yaw =  selected_actor_rotation.yaw/180*np.pi
                        relative_yaw = yaw - ego_yaw
                        relative_rotation = [selected_actor_rotation.roll-ego_rotation.roll, selected_actor_rotation.pitch-ego_rotation.pitch, selected_actor_rotation.yaw-ego_rotation.yaw]
                        relative_rotation = [_/180*np.pi for _ in relative_rotation]
                        relative_pos = np.dot(ego_inv_mat, selected_actor_matrix)[:3, 3].astype(np.float32)
                        selected_actor_transform = selected_actor.get_transform()
                        selected_actor_control = selected_actor_velocity = selected_actor_speed = selected_actor_brake = None
                        if actor_type in ["vehicle", "walker"]:
                            selected_actor_control   = selected_actor.get_control()
                            selected_actor_velocity  = selected_actor.get_velocity()
                            selected_actor_speed = self._get_forward_speed(transform=selected_actor_transform, velocity=selected_actor_velocity) # In m/s
                            if actor_type == "vehicle":
                                selected_actor_brake = selected_actor_control.brake
                            #vehicle_brake = selected_actor_control.brake
                        # filter bbox that didn't contains points of contains less points
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, selected_actor_inv_matrix, dx, lidar)
                        distance = np.linalg.norm(relative_pos).astype(np.float32)
                        result = {
                            "class": actor_type,
                            "extent": [dx[0], dx[1], dx[2],],
                            "relative_position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                            "relative_rotation": relative_rotation,
                            "num_points": int(num_in_bbox_points),
                            "distance": distance,
                            "speed": selected_actor_speed,
                            "brake": selected_actor_brake,
                            "id": int(selected_actor_id),
                            "ego_matrix": selected_actor_matrix.tolist(),
                            "ego_inv_matrix":selected_actor_inv_matrix.tolist(),
                        }
                        results[actor_type+str(selected_actor_id)] = result
        return results



    def _find_obstacle_3dbb(self, obstacle_type, max_distance=50):
        """Returns a list of 3d bounding boxes of type obstacle_type.
        If the object does have a bounding box, this is returned. Otherwise a bb
        of size 0.5,0.5,2 is returned at the origin of the object.

        Args:
            obstacle_type (String): Regular expression
            max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.

        Returns:
            List: List of Boundingboxes
        """
        obst = list()

        _actors = self._world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            distance_to_car = _obstacle.get_transform().location.distance(self._ego_vehicle.get_location())

            if 0 < distance_to_car <= max_distance:

                if hasattr(_obstacle, 'bounding_box'):
                    loc = _obstacle.bounding_box.location
                    _obstacle.get_transform().transform(loc)

                    extent = _obstacle.bounding_box.extent
                    _rotation_matrix = self.get_matrix(carla.Transform(carla.Location(0,0,0), _obstacle.get_transform().rotation))

                    rotated_extent = np.squeeze(np.array((np.array([[extent.x, extent.y, extent.z, 1]]) @ _rotation_matrix)[:3]))

                    bb = np.array([
                        [loc.x, loc.y, loc.z],
                        [rotated_extent[0], rotated_extent[1], rotated_extent[2]]
                    ])

                else:
                    loc = _obstacle.get_transform().location
                    bb = np.array([
                        [loc.x, loc.y, loc.z],
                        [0.5, 0.5, 2]
                    ])

                obst.append(bb)

        return obst

    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

