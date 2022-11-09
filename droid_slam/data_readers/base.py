
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

from .augmentation import RGBDAugmentor
from .rgbd_utils import *
from torch_scatter import scatter_sum
NCAR = 135

class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, n_frames=4, crop_size=[384,512], fmin=8.0, fmax=75.0, do_aug=True):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name

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
    
    @staticmethod
    def trackinfo(objectmasks, inds):
        framenumber = len(inds)
        count = [torch.tensor([0])]
        trackidlist = []
        area = torch.zeros(NCAR).long()
        for i in range(framenumber):
            arearank = torch.bincount(objectmasks[i]. flatten())
            area += scatter_sum(arearank, torch.arange(arearank.shape[0]), dim = 0, dim_size= NCAR)
            valid = arearank[arearank!=0].shape[0]
            count.append(count[-1]+valid)
            trackidlist.append(torch.argsort(arearank)[-valid:])

        trackid = torch.concat(trackidlist)
        arearank = torch.argsort(area)
        fre = torch.bincount(trackid)
        frerank = torch.where(fre == torch.amax(fre))[0]
        TRACKID = torch.from_numpy(np.intersect1d(frerank, arearank[-frerank.shape[0]:]))#找出出现频率最大的几辆车
        if TRACKID.shape[0] == 1 and TRACKID == torch.tensor([0]):
            frerank = torch.where(torch.isin(fre, torch.tensor([torch.amax(fre),torch.amax(fre)-1, torch.amax(fre)-2])))[0]
            TRACKID = torch.from_numpy(np.intersect1d(frerank, arearank[-frerank.shape[0]:]))
        TRACKID = TRACKID[TRACKID!=0]-1
        
        bins = torch.concat(count)
        Apperance = []
        N_app = 0
        for id in TRACKID:
            ids = torch.nonzero(trackid == id+1).squeeze(-1)
            frames = torch.bucketize(ids,bins,right =True)-1
            N_app += len(frames)
            Apperance.append(frames)

        # print('trackid is {}'.format(TRACKID))
        # print('appear is {}'.format(Apperance))
        # print('frames are {}'.format(inds))
        return TRACKID, N_app, Apperance

    @staticmethod
    def construct_objectmask(TRACKID, mask):
        single_masklist = []
        for id in TRACKID:
            single_mask = torch.where(mask == (id+1), 1.0, 0.0)
            sampled_mask = torch.nn.functional.interpolate(single_mask.unsqueeze(1), size = (30,101)).int()
            single_masklist.append(sampled_mask.squeeze(1))
        return torch.stack(single_masklist, dim = 0)

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

    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        objectmasks_list = self.scene_info[scene_id]['objectmasks']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        # inds = [ ix ]
        # while len(inds) < self.n_frames:
            # get other frames within flow threshold
        k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
        frames = frame_graph[ix][0][k]

        while(1):
            if len(frames) < self.n_frames-1:
                inds = np.random.choice(frame_graph[ix][0], self.n_frames -1, replace = False)
            else:
                inds = np.random.choice(frame_graph[ix][0][k], self.n_frames -1, replace = False)
            
            inds = np.sort(np.append(ix, inds))
            if (len(np.unique(inds)) == len(inds)):
                break
        
            # prefer frames forward in time
            # if np.count_nonzero(frames[frames > ix]):
            #     ix = np.random.choice(frames[frames > ix])
            
            # elif np.count_nonzero(frames):
            #     ix = np.random.choice(frames)

            # print('ix is {}'.format(ix))
            # print(['inds are {}'.format(inds)])
            # if np.isin(ix, inds) == False:
            #     inds += [ ix ]

        # for i in range(1, self.n_frames):
        #     inds.append(ix + 5*i)

        #读取mask并确定要追踪的车的id
        images, depths, poses, intrinsics, objectmasks = [], [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            objectmasks.append(self.__class__.objectmask_read(objectmasks_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images = np.stack(images).astype(np.float32)
        objectmasks = np.stack(objectmasks).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        objectmasks = torch.from_numpy(objectmasks).int()
        intrinsics = torch.from_numpy(intrinsics)
        intrinsics[:, 0:2] *= (101/ 1242)
        intrinsics[:, 2:4] *= (30/ 375)

        # if self.aug is not None:
        #     images, objectmasks, poses, disps, intrinsics = \
        #             self.aug(images, objectmasks, poses, disps, intrinsics)

        TRACKID, N_app, Apperance = self.trackinfo(objectmasks, inds)
        objectposes = self.__class__.objectpose_read(self.root, inds, TRACKID, Apperance).float()
        objectmasks = self.construct_objectmask(TRACKID, objectmasks)

        trackinfo = {
            'trackid': TRACKID.to('cuda'),
            'apperance': [x.to('cuda') for x in Apperance],
            'n_app': N_app,
            'frames': inds,
        }

        # scale scene
        # if len(disps[disps>0.01]) > 0:
        #     s = disps[disps>0.01].mean()
        #     disps = disps / s
        #     poses[...,:3] *= s

        return images.to('cuda'), poses.to('cuda'), objectposes.to('cuda'), objectmasks.to('cuda'), disps.to('cuda'), intrinsics.to('cuda'), trackinfo

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self
