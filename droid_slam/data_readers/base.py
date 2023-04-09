
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
import re

from .augmentation import RGBDAugmentor
from .rgbd_utils import *

class RGBDDataset(data.Dataset):
    def __init__(self, name, split_mode, datapath, n_frames=4, crop_size=[384,512], fmin=8.0, fmax=75.0, do_aug=True):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name+'-droid-midas-'+split_mode

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples
        
        if do_aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))
        
        cache_path = osp.join(cur_path, 'cache', '{}.pickle'.format(self.name))

        if osp.isfile(cache_path):
            scene_info = pickle.load(open(cache_path, 'rb'))[0]
        else:
            scene_info = self._build_dataset()
            with open(cache_path, 'wb') as cachefile:
                pickle.dump((scene_info,), cachefile)

        self.scene_info = scene_info
        self._build_dataset_index()
                
    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info:
            if not self.__class__.is_test_scene(scene):
                graph = self.scene_info[scene]['graph']
                for i in graph:
                    if len(graph[i][0]) > self.n_frames:
                        self.dataset_index.append((scene, i))
            else:
                print("Reserving {} for validation".format(scene))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph
    
    @staticmethod
    def read_pfm(path):
        """Read pfm file.
        Args:
            path (str): path to file
        Returns:
            tuple: (data, scale)
        """
        with open(path, "rb") as file:

            color = None
            width = None
            height = None
            scale = None
            endian = None

            header = file.readline().rstrip()
            if header.decode("ascii") == "PF":
                color = True
            elif header.decode("ascii") == "Pf":
                color = False
            else:
                raise Exception("Not a PFM file: " + path)

            dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
            if dim_match:
                width, height = list(map(int, dim_match.groups()))
            else:
                raise Exception("Malformed PFM header.")

            scale = float(file.readline().decode("ascii").rstrip())
            if scale < 0:
                # little-endian
                endian = "<"
                scale = -scale
            else:
                # big-endian
                endian = ">"

            data = np.fromfile(file, endian + "f")
            shape = (height, width, 3) if color else (height, width)

            data = np.reshape(data, shape)
            data = np.flipud(data)

            return data, scale
        
    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        midasdepth_list = self.scene_info[scene_id]['midasdepth']

        inds = [ ix ]
        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
            frames = frame_graph[ix][0][k]

            # prefer frames forward in time
            if np.count_nonzero(frames[frames > ix]):
                ix = np.random.choice(frames[frames > ix])
            
            elif np.count_nonzero(frames):
                ix = np.random.choice(frames)

            inds += [ ix ]

        images, depths, poses, intrinsics, midasdepths = [], [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            midasdepths.append(self.read_pfm(midasdepth_list[i])[0])
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        midasdepths = np.stack(midasdepths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        disps = torch.from_numpy(1.0 / depths)
        midasdisps = torch.from_numpy(midasdepths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.aug is not None:
            images, poses, disps, midasdisps, intrinsics = \
                self.aug(images, poses, disps, midasdisps, intrinsics)

        # scale scene
        m_s = midasdisps[midasdisps>0].mean()
        midasdisps = midasdisps / m_s

        if len(disps[disps>0.01]) > 0:
            s = disps[disps>0.01].mean()
            disps = disps / s
            poses[...,:3] *= s
        
        a_list, b_list = [], []
        for i in range(disps.shape[0]):
            # write_depth('result/val/midas'+str(inds[i]), midasdisps[i].numpy(), False)
            # write_depth('result/val/gt'+str(inds[i]), disps[i].numpy(), False)
            depth = disps[i].reshape(-1)
            midepth = midasdisps[i].reshape(-1)

            midepth = torch.stack((midepth, torch.ones_like(midepth)), dim=1)
            left = torch.linalg.inv(torch.matmul(midepth.transpose(1,0), midepth))
            right = torch.matmul(midepth.transpose(1,0), depth)
            solve = torch.matmul(left, right)
            a, b = solve[0], solve[1]
            # depth_s = a*midepth[:,0]+b
            # re_error = ((depth - depth_s)).abs().mean()
            # print('relative error is {}'.format(re_error))
            a_list.append(a)
            b_list.append(b)

        a = torch.stack(a_list, dim=0)[..., None, None]
        b = torch.stack(b_list, dim=0)[..., None, None]
        midasdisps = a*midasdisps+b

        return images, poses, disps, midasdisps, intrinsics 

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return
