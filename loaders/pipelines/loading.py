import os
import mmcv
import torch
import numpy as np
import random
import os.path as osp
from PIL import Image
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info
from mmdet3d.core.points import BasePoints
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')
import mmengine
# pred_trajs = mmengine.load('data/nuscenes/pred_trajs.json')['trajs']


def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img


def T_fut2cur(current_ego_pose, future_ego_pose):
    
    q_cur = Quaternion(current_ego_pose['rotation'])
    R_cur = q_cur.rotation_matrix
    t_cur = np.array(current_ego_pose['translation'])

    q_fut = Quaternion(future_ego_pose['rotation'])
    R_fut = q_fut.rotation_matrix
    t_fut = np.array(future_ego_pose['translation'])

    #  R_fut2cur  t_fut2cur
    R_cur_inv = R_cur.T  
    R_fut2cur = R_cur_inv @ R_fut
    t_fut2cur = R_cur_inv @ (t_fut - t_cur)

    # 4x4 T_fut2cur
    T_fut2cur = np.eye(4)
    T_fut2cur[:3, :3] = R_fut2cur
    T_fut2cur[:3, 3] = t_fut2cur
    T_fut2cur = T_fut2cur.astype(np.float32)

    return T_fut2cur


def generate_random_occ_index(n, a, b):
    random_occ_list = [0]
    tmp = sorted(random.sample(range(a, b), n))
    for i in tmp:
        random_occ_list.append(i)
    
    return random_occ_list


@PIPELINES.register_module()
class LoadOccFromFile:

    def __init__(self, occ_root, future_frames=[0], pred_traj=False, ignore_class_names=[]):
        self.occ_root = occ_root
        self.future_frames = future_frames
        self.pred_traj = pred_traj
        self.ignore_class_names = ignore_class_names
        self.occ_class_names = [
            'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation', 'free'
        ]

    def __call__(self, results):
        
        if self.future_frames == None:
            # fut_nums = random.randint(3, 6)
            fut_nums = 3  # it depends on your GPU memory
            fut_list = generate_random_occ_index(fut_nums, 1, 11)  # 21
        else:
            fut_list = self.future_frames

        occ_file_list = []
        fut2cur_list = []
        semantics_list = []
        mask_lidar_list = []
        mask_camera_list = []
        fut2cur_list.append(np.eye(4).astype(np.float32))

        scene_name, sample_idx = results['scene_name'], results['sample_idx']
        occ_file_curr = osp.join(self.occ_root, scene_name, sample_idx, 'labels.npz')
        occ_file_list.append(occ_file_curr)

        current_sample = nusc.get('sample', sample_idx)
        cur_cam_token = current_sample['data']['CAM_FRONT']
        cur_cam_data = nusc.get('sample_data', cur_cam_token)
        cur_ego_pose = nusc.get('ego_pose', cur_cam_data['ego_pose_token'])

        # if use pred trajs
        # if self.pred_traj:
        #     pred_poses = pred_trajs[sample_idx]['trajectory']

        for i in range(fut_list[-1]):  # self.future_frames[-1]
            curr_sample = nusc.get('sample', sample_idx)
            next_sample_idx = curr_sample['next']
            if next_sample_idx != '':
                sample_idx = next_sample_idx

            if i+1 in fut_list:  # self.future_frames
                occ_file = osp.join(self.occ_root, scene_name, sample_idx, 'labels.npz')
                occ_file_list.append(occ_file)

                fut_sample = nusc.get('sample', sample_idx)
                fut_cam_token = fut_sample['data']['CAM_FRONT']
                fut_cam_data = nusc.get('sample_data', fut_cam_token)
                fut_ego_pose = nusc.get('ego_pose', fut_cam_data['ego_pose_token'])
                
                fut2cur = T_fut2cur(cur_ego_pose, fut_ego_pose)
                fut2cur_list.append(fut2cur)
        
        # load lidar and camera visible label
        for occ_file in occ_file_list:
            occ_labels = np.load(occ_file)
            mask_lidar = occ_labels['mask_lidar'].astype(np.bool_)  # [200, 200, 16]
            mask_camera = occ_labels['mask_camera'].astype(np.bool_)  # [200, 200, 16]
            mask_lidar_list.append(mask_lidar)
            mask_camera_list.append(mask_camera)

            semantics = occ_labels['semantics']  # [200, 200, 16]
            for class_id in range(len(self.occ_class_names) - 1):
                mask = semantics == class_id
                if mask.sum() == 0:
                    continue
                if self.occ_class_names[class_id] in self.ignore_class_names:
                    semantics[mask] = self.num_classes - 1
            semantics_list.append(semantics)

        results['fut_list'] = fut_list
        results['fut2cur'] = fut2cur_list
        results['mask_lidar'] = mask_lidar_list
        results['mask_camera'] = mask_camera_list
        results['voxel_semantics'] = semantics_list
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps:
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 test_interval=6,
                 force_offline=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.force_offline = force_offline

        self.train_interval = train_interval
        self.test_interval = test_interval

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['cam_sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['cam_sweeps']['prev'])
                choices = list(range(len(results['cam_sweeps']['prev']))) + \
                    [len(results['cam_sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['cam_sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    extrinsics = np.eye(4)
                    extrinsics[:3, :3] = sweep[sensor]["sensor2ego_rotation"]
                    extrinsics[:3, 3] = np.array(sweep[sensor]["sensor2ego_translation"])
                    results['extrinsics'].append(extrinsics)
                    intrinsics = np.eye(4)
                    intrinsics[:sweep[sensor]['cam_intrinsic'].shape[0], :sweep[sensor]['cam_intrinsic'].shape[1]] = sweep[sensor]['cam_intrinsic']
                    results['intrinsics'].append(intrinsics)

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    # results['lidar2img'].append(np.copy(results['cam_intrinsic'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    extrinsics = np.eye(4)
                    extrinsics[:3, :3] = sweep[sensor]["sensor2ego_rotation"]
                    extrinsics[:3, 3] = np.array(sweep[sensor]["sensor2ego_translation"])
                    intrinsics = np.eye(4)
                    intrinsics[:sweep[sensor]['cam_intrinsic'].shape[0], :sweep[sensor]['cam_intrinsic'].shape[1]] = sweep[sensor]['cam_intrinsic']
                    results['intrinsics'].append(intrinsics)

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode and (not self.force_offline):
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFuture:
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        # previous sweeps
        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['cam_intrinsic'].append(np.copy(results['cam_intrinsic'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        # future sweeps
        if len(results['cam_sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    # results['lidar2img'].append(np.copy(results['cam_intrinsic'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['next']) - 1)
                sweep = results['cam_sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results


'''
This func loads previous and future frames in interleaved order, 
e.g. curr, prev1, next1, prev2, next2, prev3, next3...
'''
@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFutureInterleave:
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        results_prev = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            lidar2img=[]
        )
        results_next = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            lidar2img=[]
        )

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results_prev['img'].append(results['img'][j])
                    results_prev['img_timestamp'].append(results['img_timestamp'][j])
                    results_prev['filename'].append(results['filename'][j])
                    results_prev['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results_prev['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_prev['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_prev['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_prev['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        if len(results['sweeps']['next']) == 0:
            print(1, len(results_next['img']) )
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results_next['img'].append(results['img'][j])
                    results_next['img_timestamp'].append(results['img_timestamp'][j])
                    results_next['filename'].append(results['filename'][j])
                    results_next['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results_next['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_next['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_next['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_next['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        assert len(results_prev['img']) % 6 == 0
        assert len(results_next['img']) % 6 == 0

        for i in range(len(results_prev['img']) // 6):
            for j in range(6):
                results['img'].append(results_prev['img'][i * 6 + j])
                results['img_timestamp'].append(results_prev['img_timestamp'][i * 6 + j])
                results['filename'].append(results_prev['filename'][i * 6 + j])
                results['lidar2img'].append(results_prev['lidar2img'][i * 6 + j])

            for j in range(6):
                results['img'].append(results_next['img'][i * 6 + j])
                results['img_timestamp'].append(results_next['img_timestamp'][i * 6 + j])
                results['filename'].append(results_next['filename'][i * 6 + j])
                results['lidar2img'].append(results_next['lidar2img'][i * 6 + j])

        return results


# Repalce LoadPointsFromMultiSweeps in mmdet3d to adapt sparsebev data
@PIPELINES.register_module(force=True)
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        sweeps = results['lidar_sweeps']['prev']
        if self.pad_empty_sweeps and len(sweeps) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(sweeps) <= self.sweeps_num:
                choices = np.arange(len(sweeps))
            else:
                choices = np.arange(self.sweeps_num)

            for idx in choices:
                sweep = sweeps[idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointsFromLiDARToEgo:

    def __init__(self, base='ego'):
        self.base = base
    
    def __call__(self, results):
        points, ego2lidar = results['points'], results['ego2lidar']

        lidar2ego = torch.tensor(np.linalg.inv(ego2lidar)).float()
        ones = torch.ones_like(points.tensor[..., :1])
        pts = torch.cat([points.tensor[..., :3], ones], dim=1).transpose(0, 1)
        pts = torch.matmul(lidar2ego, pts).transpose(0, 1)

        points.tensor = torch.cat([pts, points.tensor[..., 3:]], dim=1)
        results['points'] = points
        return results


@PIPELINES.register_module()
class MyLoadMultiViewImageFromFiles(object):
    """
    This image file loader is for Waymo dataset. 
    Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged'):
        self.to_float32 = to_float32  # True
        self.img_scale = img_scale  # (1280, 1920)
        self.color_type = color_type  # 'unchanged'

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            padded = np.zeros((self.img_scale[0],self.img_scale[1],3))
            padded[0:img.shape[0], 0:img.shape[1], :] = img  # zero pad
            img = padded
        return img

    def __call__(self, results):
        """
        Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data.
                  Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        
        # Step 1: load image according to the filename
        filename = results['img_filename']
        filename[2] = filename[2].replace('image_3', 'image_3/image_3')
        img = [np.asarray(Image.open(name.replace('.png', '.jpg')))[...,::-1] for name in filename]  # to_rgb

        # Step 2: record the original shape of the image
        results['ori_shape'] = [img_i.shape for img_i in img]

        # Step 3: pad the image
        if self.img_scale is not None:
            img = [self.pad(img_i) for img_i in img]

        # Step 4: stack the image
        img = np.stack(img, axis=-1)

        # Step 5: convert the image to float32
        if self.to_float32:
            img = img.astype(np.float32)

        # Step 6: record the filename, image, image shape, and image normalization configuration
        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels, dtype=np.float32), 
                                       std=np.ones(num_channels, dtype=np.float32), 
                                       to_rgb=False) # This will be replaced in `NormalizeMultiviewImage`
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module()
class LoadOccGTFromFileWaymo(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
            use_larger=True,
            crop_x=False,
            use_infov_mask=True, 
            use_lidar_mask=False, 
            use_camera_mask=True,
            FREE_LABEL=None, 
            num_classes=None,
            future_frames=[0]
        ):
        self.use_larger=use_larger
        self.data_root = data_root # this is occ_gt_data_root in config file
        self.crop_x = crop_x
        self.use_infov_mask = use_infov_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_camera_mask = use_camera_mask
        self.FREE_LABEL = FREE_LABEL
        self.num_classes = num_classes
        self.future_frames=future_frames

    def __call__(self, results):
        ## Step 1: get the occupancy ground truth file path
        pts_filename = results['pts_filename']
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]
        if self.use_larger:
            file_path = os.path.join(self.data_root, seq_name,  '{}_04.npz'.format(frame_name))
        else:
            file_path = os.path.join(self.data_root, seq_name, '{}.npz'.format(frame_name))
        
        # Step 2: load the file
        occ_labels = np.load(file_path)
        semantics = occ_labels['voxel_label']
        # mask_infov = occ_labels['infov'].astype(bool)  # need check
        # mask_lidar = occ_labels['origin_voxel_state'].astype(bool)
        mask_camera = occ_labels['final_voxel_state'].astype(bool)

        # Step 3: crop the x axis
        # if self.crop_x: # default is False
        #     w, h, d = semantics.shape
        #     semantics = semantics[w//2:, :, :]
        #     mask_infov = mask_infov[w//2:, :, :]
        #     mask_lidar = mask_lidar[w//2:, :, :]
        #     mask_camera = mask_camera[w//2:, :, :]

        # Step 4: unify the mask
        # mask = np.ones_like(semantics).astype(bool) # 200, 200, 16
        # if self.use_infov_mask:  # True
        #     mask = mask & mask_infov
        # if self.use_lidar_mask:  # False
        #     mask = mask & mask_lidar
        # if self.use_camera_mask:  # True
        #     mask = mask & mask_camera
        # mask = mask.astype(bool)
        # results['valid_mask'] = mask
        results['mask_camera'] = mask_camera

        # Step 5: change the FREE_LABEL to num_classes-1
        if self.FREE_LABEL is not None:
            semantics[semantics == self.FREE_LABEL] = self.num_classes - 1  # 23 --> 15
        results['voxel_semantics'] = semantics

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)