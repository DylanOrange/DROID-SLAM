"""
@brief:
@author: Binbin Xu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
import os
import random
import pickle
import functools

import numpy as np
import torch.utils.data as data
import os.path as osp

from imageio import imread
from tqdm import tqdm
import glob
from cv2 import resize, INTER_NEAREST
from cv2 import imread as cv2_imread
from src.conv_onet.config import get_data_fields
from src.utils.visualize import pointcloud_from_depth
import trimesh
import torch
from utils.fusion import TSDFVolumeTorch
from src.utils.geometry import warp_pointcloud


def get_movingobjects3d_dataset(mode, cfg, recache=False):
    if cfg['training']['train_completion']:
        onet_path = cfg['data']['path']
    else:
        onet_path = None
    dataset = MovingObjects3D(root=cfg['data']['simo_path'],
                              category=cfg['data']['classes'],
                              require_pcd=cfg['data']['require_pcd'],
                              load_type=mode,
                              keyframes=[1],
                              keyframe_win=cfg['data']['multi_view'],
                              sampling_freq=cfg['data']['sampling_freq'],
                              load_onet_path=onet_path,
                              onet_cfg=cfg,
                            #   image_resize=1.0,
                              image_resize=0.5,
                              recache=recache,
                              )
    return dataset


category_ids = {
    "chair": "03001627",
}


class MovingObjects3D(data.Dataset):

    # every sequence has 200 frames.
    # change it to percent
    categories = {
        'train': {'aeroplane':  [0, 190],
                  'chair':      [0, 0.8],
                  'mug':      [0, 190],
                  'sofa':          [0, 190],
                  'table':          [0, 190],
                  'car':          [0, 190]},

        'val': {'aeroplane': [190, 200],
                'bicycle':          [190, 200],
                'bus':              [190, 200],
                'car':              [190, 200],
                'chair': [0.8, 0.9],
                },

        'test': {'boat':           [0, 200],
                 'motorbike':        [0, 200],
                 'chair': [0.9, 1.0],
                 }
    }

    def __init__(self, root, load_type='train', keyframes=[1], data_transform=None,
                 load_onet_path=None, onet_cfg=None, keyframe_win=1, sampling_freq=1, require_pcd=False,
                 category=None, image_resize=0.5, recache=False):
        super(MovingObjects3D, self).__init__()

        self.base_folder = osp.join(root)

        self.transforms = data_transform
        self.require_pcd = require_pcd

        if load_type in ['validation', 'test']:
            # should not mix different keyframes in test
            assert(len(keyframes) == 1)
            self.keyframes = [1]
            self.sample_freq = keyframes[0]
        else:
            self.keyframes = keyframes
            self.sample_freq = sampling_freq
        self.keyframe_win = keyframe_win

        self.ids = 0
        # self.images_size = [240, 320]
        # self.images_size = [480, 640]
        self.obj_vis_sampling_frames = []

        # building dataset is expensive, cache so only needs to be performed once
        # cur_path = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(root, 'cache')):
            os.mkdir(osp.join(root, 'cache'))

        cache_path = osp.join(
            root, 'cache', f'simo_{load_type}_{category}.pickle')

        if osp.isfile(cache_path) and not recache:
            scene_info = pickle.load(open(cache_path, 'rb'))[0]
        else:
            scene_info = self._build_dataset(
                load_type, category, load_onet_path)
            with open(cache_path, 'wb') as cachefile:
                pickle.dump((scene_info,), cachefile)
        self.scene_info = scene_info
        self._build_dataset_index()

        # load occupancy and sdf fields
        if load_onet_path is not None:
            self.onet_dataset_path = load_onet_path
            self.fields = get_data_fields(load_type, onet_cfg)
        else:
            self.fields = None

        # downscale the input image to half
        self.fx_s = image_resize
        self.fy_s = image_resize
        self.load_type = load_type

    def _build_dataset_index(self):
        self.seq_acc_ids = [0]
        for obj_visible_frames in self.scene_info["obj_vis_idx"]:
            # sampling based on temporal gap, not the co-visiblity
            obj_vis_sampling_frames = obj_visible_frames[::self.sample_freq]
            # further samping based on the keyframe gap
            # total_valid_frames = max(0, len(obj_visible_frames) - max(self.keyframes))
            total_valid_frames = max(
                0, len(obj_vis_sampling_frames) - self.keyframe_win + 1)
            self.obj_vis_sampling_frames.append(obj_vis_sampling_frames)

            self.ids += total_valid_frames
            self.seq_acc_ids.append(self.ids)
        logging.info('There are a total of {:} valid frames'.format(self.ids))

    def _build_dataset(self, load_type, category, load_onet_path=None):
        # get the accumulated image sequences on the fly
        from tqdm import tqdm
        scene_info = {
            "image_seq": [],
            "depth_seq": [],
            "invalid_seq": [],
            "object_mask_seq": [],
            "cam_pose_seq": [],
            "obj_pose_seq": [],
            "calib": [],
            "obj_vis_idx": [],
            "obj_names": [],
        }

        data_all = self.categories[load_type]
        for data_obj, frame_interval in data_all.items():

            if category is not None and data_obj != category:
                continue

            # assert load_onet_path is not None, "require onet shapent split to build dataset"
            if load_onet_path is not None:
                category_path = os.path.join(
                    load_onet_path, category_ids[category])
                split_file = os.path.join(category_path, load_type + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                seq_list = [os.path.join(self.base_folder, data_obj, model_id) for model_id in models_c if
                            model_id != '']
            else:
                seq_list = glob.glob(osp.join(self.base_folder, data_obj, "*"))
            #     num_seq = len(seq_list)
            #     start_idx, end_idx = frame_interval
            #     start_idx = int(start_idx * num_seq)
            #     end_idx = int(end_idx * num_seq)
            #     print('Load {:} data from frame {:d} to {:d}'.format(
            #         data_obj, start_idx, end_idx))
            #     seq_list = seq_list[start_idx:end_idx]

            # cache the visibility sequences of the whole dataset for the first time
            for seq_id, seq_dir in enumerate(tqdm(seq_list, desc=f"Building {category} dataset")):
                obj_name = seq_dir.split('/')[-1]
                info_pkl = osp.join(seq_dir, 'info.pkl')

                color_seq, depth_seq, invalid_seq, mask_seq, camera_poses_seq, object_poses_seq, \
                    obj_visible_frames, calib_seq = extract_info_pickle(
                        info_pkl)

                scene_info['image_seq'].append(
                    [osp.join(self.base_folder, x) for x in color_seq])
                scene_info['depth_seq'].append(
                    [osp.join(self.base_folder, x) for x in depth_seq])
                # self.invalid_seq.append(invalid_seq)
                scene_info['object_mask_seq'].append(
                    [osp.join(self.base_folder, x) for x in mask_seq])
                scene_info['cam_pose_seq'].append(camera_poses_seq)
                scene_info['obj_pose_seq'].append(object_poses_seq)
                scene_info["calib"].append(calib_seq)
                # store the visiblity map
                scene_info["obj_vis_idx"].append(obj_visible_frames)

                scene_info["obj_names"].append(
                    [data_obj, obj_name, seq_id, category_ids[category]])

        return scene_info

    def __len__(self):
        return self.ids

    def set_dataload_speedup(self, get_color=False, get_depth=False, get_mask=False):
        # to speed up dataloading
        self.get_color = get_color
        self.get_depth = get_depth
        self.get_mask = get_mask

    # def __getitem__(self, index):
    def get_one_frame(self, index, align_to_onet=True, get_color=False, get_depth=False, get_mask=False):
        # the index we want from search sorted is shifted for one
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index + 1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = self.obj_vis_sampling_frames[seq_idx][frame_idx]
        # if in train mode: add random perturbation of index
        if self.load_type == 'train':
            this_idx += random.randint(0, self.sample_freq-1)

        if get_color:
            color0 = self.__load_rgb_tensor(
                self.scene_info["image_seq"][seq_idx][this_idx])
            if self.transforms:
                color0 = self.transforms([color0])
        else:
            color0 = []

        if get_depth:
            depth0 = self.__load_depth_tensor(
                self.scene_info["depth_seq"][seq_idx][this_idx]).squeeze()
        else:
            depth0 = []

        cam_pose0 = self.scene_info["cam_pose_seq"][seq_idx][this_idx].astype(
            np.float32)
        obj_pose0 = self.scene_info["obj_pose_seq"][seq_idx][this_idx].astype(
            np.float32)

        # the validity of the object is up the object mask
        obj_index = 1  # object index is in default to be 1
        if get_mask:
            obj_mask0 = self.__load_binary_mask_tensor(
                self.scene_info["object_mask_seq"][seq_idx][this_idx], obj_index).squeeze()
        else:
            obj_mask0 = []

        calib = np.asarray(self.scene_info["calib"][seq_idx], dtype=np.float32)
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        # camera_info = {
        #     "fx": calib[0],
        #     "fy": calib[1],
        #     "cx": calib[2],
        #     "cy": calib[3],
        # }
        camera_info = np.eye(3)  # , dtype=np.float32)
        camera_info[0, 0] = calib[0]
        camera_info[1, 1] = calib[1]
        camera_info[0, 2] = calib[2]
        camera_info[1, 2] = calib[3]

        obj_name = self.scene_info["obj_names"][seq_idx]

        obj_geom = {}
        if self.fields is not None:
            for field_name, field in self.fields.items():
                onet_model_path = os.path.join(
                    self.onet_dataset_path, category_ids[obj_name[0]], obj_name[1])
                field_data = field.load(onet_model_path, None, None)

                if isinstance(field_data, dict):
                    for k, v in field_data.items():
                        if k is None:
                            obj_geom[field_name] = v
                        else:
                            obj_geom['%s.%s' % (field_name, k)] = v
                else:
                    obj_geom[field_name] = field_data
        else:
            onet_dir = '/media/binbin/code/ssd_data/can_partial_shapenet'
            onet_model_path = os.path.join(
                onet_dir, category_ids[obj_name[0]], obj_name[1])
            scale_trans_file = os.path.join(onet_model_path, 'scale_trans.pkl')

            if os.path.exists(scale_trans_file):
                with open(scale_trans_file, 'rb') as f:
                    scale_trans = pickle.load(f)
            else:
                file_path = os.path.join(onet_model_path, 'points.npz')
                points_dict = np.load(file_path)
                scale_trans = {
                    'loc': points_dict['loc'].astype(np.float32),
                    'scale': points_dict['scale'].astype(np.float32)
                }
                with open(scale_trans_file, 'wb') as handle:
                    pickle.dump(scale_trans, handle)

            obj_geom['points.loc'] = scale_trans['loc'].astype(np.float32)
            obj_geom['points.scale'] = scale_trans['scale'].astype(np.float32)

        if self.require_pcd:
            # cache object pointcloud
            c_obj_pcd_file = self.scene_info["depth_seq"][seq_idx][this_idx].replace(
                ".png", "_c_obj_pcd.npy")
            if osp.isfile(c_obj_pcd_file):
                c_obj_pcd = np.load(c_obj_pcd_file)
            else:
                # only load depth and mask when it is necessary
                if len(depth0) == 0 and not get_depth:
                    depth0 = self.__load_depth_tensor(
                        self.scene_info["depth_seq"][seq_idx][this_idx]).squeeze()
                if len(obj_mask0) == 0 and not get_mask:
                    obj_mask0 = self.__load_binary_mask_tensor(
                        self.scene_info["object_mask_seq"][seq_idx][this_idx], obj_index).squeeze()

                masked_obj_depth = obj_mask0 * depth0
                c_obj_pcd = pointcloud_from_depth(
                    masked_obj_depth, fx=calib[0], fy=calib[1], cx=calib[2], cy=calib[3],
                    apply_mask=True,
                )
                obj_depth_nonnan = (c_obj_pcd != 0).all(axis=2).reshape(-1)
                c_obj_pcd = c_obj_pcd.reshape(-1, 3)[obj_depth_nonnan]
                np.save(c_obj_pcd_file, c_obj_pcd)
                
                # remove the depth & mask, if not need
                if not get_depth:
                    depth0 = []
                if not get_mask:
                    obj_mask0 = []
            # import trimesh
            # trimesh.PointCloud(vertices=c_obj_pcd).show()
            sample_num = 2048  # 1024 is safe for now
            idx = np.random.randint(c_obj_pcd.shape[0],
                                    size=np.min([sample_num, c_obj_pcd.shape[0]]))
            if len(idx) < sample_num:
                # logging.warning(f"check {c_obj_pcd_file} has {len(idx)} points")
                return None
            c_obj_pcd = c_obj_pcd[idx, :]
            obj_geom['view_obj_pointcloud'] = c_obj_pcd.astype(np.float32)

        if align_to_onet:
            # align the camera pose to the onet
            obj_pose0 = self.__align_obj_pose_to_onet(
                obj_pose0, obj_geom).astype(np.float32)

        return color0, depth0, cam_pose0, obj_pose0, camera_info, obj_mask0, obj_name, obj_geom

    def __getitem__(self, index):
        if self.keyframe_win == 1:
            return self.get_one_frame(index, get_color=self.get_color, get_depth=self.get_color, get_mask=self.get_mask)
        elif self.keyframe_win == 2:
            return self.get_image_pair(index)
        else:
            raise NotImplementedError()

    def get_image_pair(self, index):
        # the index we want from search sorted is shifted for one
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index+1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = self.obj_vis_sampling_frames[seq_idx][frame_idx]
        next_idx = self.obj_vis_sampling_frames[seq_idx][frame_idx +
                                                         random.choice(self.keyframes)]

        color0 = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1 = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        obj_pose0 = self.obj_pose_seq[seq_idx][this_idx]
        obj_pose1 = self.obj_pose_seq[seq_idx][next_idx]

        # the relative allocentric transform of objects
        transform = functools.reduce(np.dot,
                                     [np.linalg.inv(cam_pose1), obj_pose1, np.linalg.inv(obj_pose0), cam_pose0]).astype(np.float32)

        # the validity of the object is up the object mask
        obj_index = 1  # object index is in default to be 1
        obj_mask0 = self.__load_binary_mask_tensor(
            self.object_mask_seq[seq_idx][this_idx], obj_index)
        obj_mask1 = self.__load_binary_mask_tensor(
            self.object_mask_seq[seq_idx][next_idx], obj_index)

        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        obj_name = self.obj_names[seq_idx]
        # pair_name = '{:}/{:06d}to{:06d}'.format(obj_name, this_idx, next_idx)
        pair_name = {'seq': obj_name,
                     'seq_idx': seq_idx,
                     'frame0': this_idx,
                     'frame1': next_idx}

        return color0, color1, depth0, depth1, transform, calib, obj_mask0, obj_mask1, pair_name

    def get_original_size_batch(self, index):
        # the index we want from search sorted is shifted for one
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index+1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = self.obj_vis_sampling_frames[seq_idx][frame_idx]
        next_idx = self.obj_vis_sampling_frames[seq_idx][frame_idx +
                                                         random.choice(self.keyframes)]

        color0 = self.__load_rgb_tensor(
            self.image_seq[seq_idx][this_idx], do_resize=False)
        color1 = self.__load_rgb_tensor(
            self.image_seq[seq_idx][next_idx], do_resize=False)

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])

        depth0 = self.__load_depth_tensor(
            self.depth_seq[seq_idx][this_idx], do_resize=False)
        depth1 = self.__load_depth_tensor(
            self.depth_seq[seq_idx][next_idx], do_resize=False)

        cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        obj_pose0 = self.obj_pose_seq[seq_idx][this_idx]
        obj_pose1 = self.obj_pose_seq[seq_idx][next_idx]

        # the relative allocentric transform of objects
        transform = functools.reduce(np.dot,
                                     [np.linalg.inv(cam_pose1), obj_pose1, np.linalg.inv(obj_pose0), cam_pose0]).astype(np.float32)

        # the validity of the object is up the object mask
        obj_index = 1  # object index is in default to be 1
        obj_mask0 = self.__load_binary_mask_tensor(
            self.object_mask_seq[seq_idx][this_idx], obj_index, do_resize=False)
        obj_mask1 = self.__load_binary_mask_tensor(
            self.object_mask_seq[seq_idx][next_idx], obj_index, do_resize=False)

        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)

        obj_name = self.obj_names[seq_idx]
        # pair_name = '{:}/{:06d}to{:06d}'.format(obj_name, this_idx, next_idx)
        pair_name = {'seq': obj_name,
                     'seq_idx': seq_idx,
                     'frame0': this_idx,
                     'frame1': next_idx}

        return color0, color1, depth0, depth1, transform, calib, obj_mask0, obj_mask1, pair_name

    def __load_rgb_tensor(self, path, do_resize=True):
        """ Load the rgb image
        """
        image = imread(path)[:, :, :3]
        image = image.astype(np.float32) / 255.0
        if do_resize:
            image = resize(image, None, fx=self.fx_s, fy=self.fy_s)
        return image

    def __load_depth_tensor(self, path, do_resize=True):
        """ Load the depth
        """
        depth = imread(path).astype(np.float32) / 1e3
        if do_resize:
            depth = resize(depth, None, fx=self.fx_s,
                           fy=self.fy_s, interpolation=INTER_NEAREST)
        depth = np.clip(depth, 1e-1, 1e2)  # the valid region of the depth
        return depth[np.newaxis, :]

    def __load_binary_mask_tensor(self, path, seg_index, do_resize=True):
        """ Load a binary segmentation mask (numbers)
            If the object matches the specified index, return true;
            Otherwise, return false
        """
        obj_mask = imread(path)
        mask = (obj_mask == seg_index)
        if do_resize:
            mask = resize(mask.astype(np.float), None, fx=self.fx_s,
                          fy=self.fy_s, interpolation=INTER_NEAREST)
        return mask.astype(np.bool)[np.newaxis, :]

    def check_single_view_batch(self, index):
        color0, depth0, T_wc, T_wo, intrins, obj_mask0, obj_name, obj_geom = self.get_one_frame(
            index)

        def random_color():
            return np.random.rand(3) * 255

        # visualize input
        data_check_scene = trimesh.scene.Scene()

        # pointcloud
        c_obj_pcl = obj_geom['view_obj_pointcloud']
        # set object pose (the one to be optimised)
        T_OC0 = np.matmul(np.linalg.inv(T_wo), T_wc)
        # ground-truth transformation from object-coord to canonical-coord
        o_obj_pcl = warp_pointcloud(T_OC0, c_obj_pcl)
        pointcloud_geom = trimesh.PointCloud(
            vertices=o_obj_pcl, colors=random_color())
        data_check_scene.add_geometry(pointcloud_geom)

        # occupancy points
        c_obj_onet_occ_id = (obj_geom['points.occ'] != 0)
        c_obj_occ_points = obj_geom['points'][c_obj_onet_occ_id]
        occ_geom = trimesh.PointCloud(
            vertices=c_obj_occ_points, colors=random_color())
        data_check_scene.add_geometry(occ_geom)
        # data_check_scene.show()

        # partial tsdf
        vol_dim = 64
        voxel_size = 1.0 / vol_dim
        obj_tsdf_volume = TSDFVolumeTorch(
            [vol_dim, vol_dim, vol_dim], [-0.5, -0.5, -0.5], voxel_size, margin=3)
        # create object tsdf volume

        # ground-truth transformation from object-coord to canonical-coord
        masked_obj_depth = obj_mask0 * depth0
        obj_tsdf_volume.integrate(masked_obj_depth,
                                  intrins,
                                  T_wc,
                                  obs_weight=1.,
                                  T_wo=T_wo,
                                  )

        verts, faces, _ = obj_tsdf_volume.get_mesh()
        # tsdf_vol, weight_vol = obj_tsdf_volume.get_volume()
        partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces)
        data_check_scene.add_geometry(partial_tsdf)
        data_check_scene.show()

        # # sdf points
        # zero_level_points = obj_geom['sdf_points.zero_sdf_points']
        # zero_level_geom = trimesh.PointCloud(vertices=zero_level_points, colors=random_color())
        # data_check_scene.add_geometry(zero_level_geom)
        # data_check_scene.show()
        # off_surface_level_points = obj_geom['sdf_points.near_surface_points']
        # off_surface_geom = trimesh.PointCloud(vertices=off_surface_level_points, colors=random_color())
        # data_check_scene.add_geometry(off_surface_geom)
        # data_check_scene.show()

    def __align_obj_pose_to_onet(self, T_wo, obj_geom):
        # the alignment of ground-truth objectc coordinate with onet re-scaled coordinate
        T_BC = np.asarray(((1, 0, 0, 0),
                           (0, -1, 0, 0),
                           (0, 0, -1, 0),
                           (0, 0, 0, 1)))
        T_scale = np.eye(4)
        # scale from fusion (scaled to unit cube)
        # T_scale[0:3, 3] = field_data['model_scale']['loc']
        # T_scale[0:3, 0:3] *= field_data['model_scale']['scale']

        # scale from mesh sampling (using boudingbox of original mesh
        T_scale[0:3, 3] = obj_geom['points.loc']
        T_scale[0:3, 0:3] *= obj_geom['points.scale']

        T_oo = np.matmul(T_BC, T_scale)
        T_wo = np.matmul(T_wo, T_oo)

        return T_wo


def extract_info_pickle(info_pkl):

    with open(info_pkl, 'rb') as p:
        info = pickle.load(p)

        color_seq = [x.split('final/')[1] for x in info['color']]
        depth_seq = [x.split('final/')[1] for x in info['depth']]
        invalid_seq = [x.split('final/')[1] for x in info['invalid']]
        mask_seq = [x.split('final/')[1] for x in info['object_mask']]

        # in this rendering setting, there is only one object
        camera_poses_seq = info['pose']
        object_poses_seq = info['object_poses']['Model_1']
        object_visible_frames = info['object_visible_frames']['Model_1']

        calib_seq = info['calib']

    return color_seq, depth_seq, invalid_seq, mask_seq, \
        camera_poses_seq, object_poses_seq, object_visible_frames, calib_seq


if __name__ == '__main__':
    from src import config as src_config

    mode = 'train'
    onet_cfg_file = '/media/binbin/backup/checkpoint/conv_onet/pose_predictor/simo_nocs/2021-10-26-21-45-43/config.yaml'
    cfg = src_config.load_config(onet_cfg_file,
                                 '/media/binbin/code/object-level/convolutional_occupancy_networks/configs/default.yaml')

    dataset = MovingObjects3D(root=cfg['data']['simo_path'],
                              category=cfg['data']['classes'],
                              require_pcd=cfg['data']['require_pcd'],
                              load_type=mode,
                              keyframes=[1],
                              keyframe_win=cfg['data']['multi_view'],
                              sampling_freq=cfg['data']['sampling_freq'],
                              load_onet_path=cfg['data']['path'],
                              onet_cfg=cfg,
                              )
    dataset.check_single_view_batch(0)
#     from data.dataloader import load_data
#     import torchvision.utils as torch_utils

#     # loader = MovingObjects3D('', load_type='train', keyframes=[1])
#     loader = load_data('MovingObjects3D', keyframes=[1], load_type='train')
#     torch_loader = data.DataLoader(loader, batch_size=16, shuffle=False, num_workers=4)

#     for batch in torch_loader:
#         color0, color1, depth0, depth1, transform, K, mask0, mask1, names = batch
#         B,C,H,W=color0.shape

#         bcolor0_img = torch_utils.make_grid(color0, nrow=4)
#         bcolor1_img = torch_utils.make_grid(color1, nrow=4)
#         # bdepth0_img = torch_utils.make_grid(depth0, nrow=4)
#         # bdepth1_img = torch_utils.make_grid(depth1, nrow=4)
#         bmask0_img = torch_utils.make_grid(mask0.view(B,1,H,W)*255, nrow=4)
#         bmask1_img = torch_utils.make_grid(mask1.view(B,1,H,W)*255, nrow=4)

#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.imshow(bcolor0_img.numpy().transpose((1,2,0)))
#         plt.figure()
#         plt.imshow(bcolor1_img.numpy().transpose((1,2,0)))
#         # plt.figure()
#         # plt.imshow(bdepth0_img.numpy().transpose((1,2,0)))
#         # plt.figure()
#         # plt.imshow(bdepth1_img.numpy().transpose((1,2,0)))
#         plt.figure()
#         plt.imshow(bmask0_img.numpy().transpose((1,2,0)))
#         plt.figure()
#         plt.imshow(bmask1_img.numpy().transpose((1,2,0)))
#         plt.show()