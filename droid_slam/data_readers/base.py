
import imghdr
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import re

import sys
sys.path.append("..")

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
import random

from .augmentation import RGBDAugmentor
from .rgbd_utils import *
from torch_scatter import scatter_sum
from geom.projective_ops import coords_grid, crop, batch_grid
from geom.flow_vis_utils import write_depth

NCAR = 135

class RGBDDataset(data.Dataset):
    def __init__(self, name, split_mode, datapath, n_frames=4, crop_size=[384,512], fmin=8.0, fmax=75.0, obfmin=8.0, obfmax=75.0, do_aug=True):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name+'-'+split_mode

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples

        self.obfmin = obfmin # exclude very easy examples
        self.obfmax = obfmax # exclude very hard examples

        self.h1 = 288
        self.w1 = 960
        self.scale = 8
        self.cropscale = 2

        # self.alltrackid = [1,2,3]
        self.alltrackid = [1,2]
        
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
            objectinfo = self.scene_info[scene]['object']
            cameragraph = self.scene_info[scene]['graph']
            for id in list(objectinfo.keys()):
                if len(objectinfo[id][0])<self.n_frames:
                    continue
                objectgraph = objectinfo[id][2]
                # for i in objectgraph:
                    # self.dataset_index.append((scene, id, i))
                frameidx_list = objectinfo[id][0]
                for i in objectgraph:
                    k = (objectgraph[i][1] > self.obfmin) & (objectgraph[i][1] < self.obfmax)
                    if k.any():
                        camera_frame = frameidx_list[i]
                        camera_idx = np.isin(cameragraph[camera_frame][0], frameidx_list[objectgraph[i][0][k]])
                        factor_camera = cameragraph[camera_frame][1][camera_idx]
                        m = (factor_camera > self.fmin) & (factor_camera < self.fmax)
                        if m.any():
                            self.dataset_index.append((scene, id, i))

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
        area = torch.zeros(NCAR)
        for i in range(framenumber):
            arearank = torch.bincount(objectmasks[i].int().flatten())
            area += scatter_sum(arearank, torch.arange(arearank.shape[0]), dim = 0, dim_size= NCAR)
            valid = arearank[arearank!=0].shape[0]
            count.append(count[-1]+valid)
            trackidlist.append(torch.argsort(arearank)[-valid:])

        trackid = torch.concat(trackidlist)
        arearank = torch.argsort(area)
        fre = torch.bincount(trackid)
        frerank = torch.where(fre == torch.amax(fre))[0]
        TRACKID = torch.from_numpy(np.intersect1d(arearank[-frerank.shape[0]:], frerank))#找出出现频率最大的几辆车
        # if TRACKID.shape[0] > 8:
        #     TRACKID = torch.from_numpy(np.intersect1d(arearank[-8:], frerank))
        # if TRACKID.shape[0] == 1 and TRACKID == torch.tensor([0]):
        #     frerank = torch.where(torch.isin(fre, torch.tensor([torch.amax(fre),torch.amax(fre)-1])))[0]
        #     TRACKID = torch.from_numpy(np.intersect1d(arearank[-frerank.shape[0]:], frerank))
        #     if TRACKID.shape[0] == 1 and TRACKID == torch.tensor([0]):
        #         frerank = torch.where(torch.isin(fre, torch.tensor([torch.amax(fre),torch.amax(fre)-1,torch.amax(fre)-2])))[0]
        #         TRACKID = torch.from_numpy(np.intersect1d(arearank[-frerank.shape[0]:], frerank))
        if len(TRACKID) > 1:
            TRACKID = TRACKID[TRACKID!=0]
            TRACKID = TRACKID[torch.randint(len(TRACKID), (1,))]-1
            # TRACKID = arearank[torch.isin(arearank, TRACKID).nonzero()[-1]]-1#最大面积
        else:
            TRACKID = TRACKID-2
        
        bins = torch.concat(count)
        Apperance = []
        N_app = 0
        for id in TRACKID:
            ids = torch.nonzero(trackid == id+1).squeeze(-1)
            frames = torch.bucketize(ids,bins,right =True)-1
            N_app += len(frames)
            Apperance.append(frames)
            if id == -2:
                Apperance = [torch.arange(framenumber)]

        return TRACKID, N_app, Apperance

    @staticmethod
    def cornerinfo(objectmask):
        coords = torch.nonzero(objectmask)
        corner_min = torch.amin(coords, dim = 0)[1:]
        corner_max = torch.amax(coords, dim = 0)[1:]
        rec = corner_max - corner_min
        return corner_min, rec
    
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

    @staticmethod
    def construct_objectmask(TRACKID, mask):
        single_masklist, cropmask_list = [], []
        for id in TRACKID:
            # single_mask = torch.where(mask == (id+1), 1.0, 0.0)
            # single_cropmask = torch.where(cropmask == (id+1), 1.0, 0.0)
            # sampled_mask = torch.nn.functional.interpolate(single_mask.unsqueeze(1), size = (30,101), mode = 'bicubic').int()
            single_masklist.append(torch.where(mask == (id+1.0), 1.0, 0.0))
            # cropmask_list.append(torch.where(cropmask == (id+1), 1.0, 0.0))
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
        #visualzie depth
        # for i in range(disps.shape[0]):
        #     write_depth(os.path.join('../DeFlowSLAM/datasets/vkitti_depth_visualize', str(i)), disps[i], False)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.savefig('./result/matrix/camera.png', bbox_inches='tight')

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def build_object_frame_graph(self, poses, depths, intrinsics, object, f=8, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d, object = prepare_object_distance_matrix_flow(poses, disps, intrinsics, object)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # for i in range(len(d)):
        #     plt.imshow(d[i])
        #     plt.savefig('./result/matrix/trackid_{}.png'.format(i), bbox_inches='tight')

        for n, key in enumerate(object.keys()):
            graph = {}
            d[n] *= f
            for i in range(d[n].shape[0]):
                j, = np.where(d[n][i] < max_flow)
                graph[i] = (j, d[n][i,j])
            object[key].append(graph)
        return object

    def __getitem__(self, index):
        """ return training video """

        # index = index % len(self.dataset_index)
        index = np.random.randint(30)

        scene_id, trackid, ix = self.dataset_index[index]
        objectinfo = self.scene_info[scene_id]['object']
        objectgraph = objectinfo[trackid][2]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        insmasks_list = self.scene_info[scene_id]['objectmasks']
        # midasdepth_list = self.scene_info[scene_id]['midasdepth']

        frameidx_list = objectinfo[trackid][0]
        objectposes_list = objectinfo[trackid][1]
        
        initial_ix = ix
        inds = [ ix ]

        # while len(inds) < self.n_frames:
        #     # get other frames within flow threshold
        #     k = (objectgraph[ix][1] > self.obfmin) & (objectgraph[ix][1] < self.obfmax)
        #     object_frames = objectgraph[ix][0][k]
        #     # print('object flow {}'.format(frameidx_list[object_frames]))

        #     camera_idx = frameidx_list[ix]
        #     m = (frame_graph[camera_idx][1] > self.fmin) & (frame_graph[camera_idx][1] < self.fmax)
        #     camera_frames = frame_graph[camera_idx][0][m]
        #     # print('camera flow {}'.format(camera_frames))

        #     intersect = np.intersect1d(frameidx_list[object_frames], camera_frames)
        #     frames = np.isin(frameidx_list, intersect).nonzero()[0]
        #     # print(frames)
        #     in20 = np.logical_and(np.abs(frames-ix)<10, 0<np.abs(frames-ix))

        #     if np.count_nonzero(frames[np.logical_and(frames>ix, in20)]):
        #         ix = np.random.choice(frames[np.logical_and(frames>ix, in20)])
        #         # print(ix)
        #     elif np.count_nonzero(frames[in20]):
        #         ix = np.random.choice(frames[in20])
        #         # print(ix)
        #     else:
        #         ix = np.random.choice(frames)
        #         # print(ix)
        #     inds += [ ix ]
        # inds = np.array(inds)

        # if len(np.unique(inds)) < 5:
        ix = initial_ix
        inds = [ix]
        allindex = np.arange(len(frameidx_list))
        while len(inds) < self.n_frames:
            ix = np.random.choice(allindex[np.abs(allindex-ix)==1])
            inds += [ ix ]
            inds = list(set(inds))
        inds = np.array(inds)
            
        print('scene is {}'.format(scene_id))
        print('trackid is {}'.format(trackid))
        print('frames are {}'.format(inds))
        print('camera frames are {}'.format(frameidx_list[inds]))
        # for i in range(len(inds)-1):
        #     camera_frame = frameidx_list[inds[i]]
        #     next_ca_frame = frameidx_list[inds[i+1]]
        #     print('object flow is {}'.format(objectgraph[inds[i]][1][objectgraph[inds[i]][0] == inds[i+1]]))
        #     print('camera flow is {}'.format(frame_graph[camera_frame][1][frame_graph[camera_frame][0] == next_ca_frame]))

        #读取mask并确定要追踪的车的id
        images, depths, poses, intrinsics, objectmasks, objectposes, insmasks, highdepths, midasdepths = [], [], [], [], [], [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[frameidx_list[i]]))
            depths.append(self.__class__.depth_read(depths_list[frameidx_list[i]]))#1/8 resolution, 1/2 resolution
            poses.append(poses_list[frameidx_list[i]])
            intrinsics.append(intrinsics_list[frameidx_list[i]])
            # midasdepths.append(self.read_pfm(midasdepth_list[frameidx_list[i]])[0])
            objectposes.append(objectposes_list[i])
            insmasks.append(self.__class__.objectmask_read(insmasks_list[frameidx_list[i]])[1])#1 resolution

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        objectposes = np.stack(objectposes).astype(np.float32)
        # highdepths = np.stack(highdepths).astype(np.float32)
        # midasdepths = np.stack(midasdepths).astype(np.float32)

        insmasks = np.stack(insmasks).astype(np.float32)
        objectmasks = np.where(insmasks == (trackid+1.0), 1.0, 0.0).astype(np.float32)

        # for n, idx in enumerate(inds):
        #     vis_image = images[n].copy()
        #     vis_image[objectmasks[n]>0] = np.array([255.0,255.0,255.0])
        #     cv2.imwrite('./result/object/mask_'+ str(n)+'.png', vis_image)

        images = torch.from_numpy(images)
        _, h0, w0 = images.shape[:3]

        depths = torch.from_numpy(depths)
        # highdepths = torch.from_numpy(highdepths)
        # midasdepths = torch.from_numpy(midasdepths)

        poses = torch.from_numpy(poses)
        objectmasks = torch.from_numpy(objectmasks)
        objectposes = torch.from_numpy(objectposes)
        # insmasks = torch.from_numpy(insmasks)
        intrinsics = torch.from_numpy(intrinsics)
        intrinsics[:, 0:2] *= ((self.w1//self.scale)/ w0)
        intrinsics[:, 2:4] *= ((self.h1//self.scale)/ h0)

        images = images.permute(0, 3, 1, 2)
        images = torch.nn.functional.interpolate(images, size = (self.h1,self.w1), mode = 'bilinear')

        if self.aug is not None:
            highimages = self.aug(images)
        # highimages = images

        # lowimages = torch.nn.functional.interpolate(highimages, size = (self.h1//2,self.w1//2), mode = 'bilinear')

        corners, recs = [], []
        highmask = torch.nn.functional.interpolate(objectmasks[:, None], size = (self.h1//2, self.w1//2)).squeeze(1)
        lowmask = torch.nn.functional.interpolate(objectmasks[:, None], size = (self.h1//8, self.w1//8)).squeeze(1)
        mask = highmask.clone()
        for level in range(3):
            corner, rec = self.cornerinfo(highmask)
            highmask = torch.nn.functional.interpolate(highmask[:, None], size = (self.h1//(2**(level+2)), self.w1//(2**(level+2)))).squeeze(1)
            corners.append(corner)
            recs.append(rec)
        corners = torch.stack(corners)
        recs = torch.stack(recs)
        
        # img = lowimages.permute(0,2,3,1).numpy().astype(np.uint8).copy()
        # reccorner = corners[0].tolist()
        # recrec = recs[0].tolist()
        # for i in range(len(img)):
        #     cv2.rectangle(img[i], (reccorner[1], reccorner[0]) , (reccorner[1]+recrec[1], reccorner[0]+recrec[0]), color=(0, 255, 255), thickness=1)
        #     cv2.imwrite('./result/crop/'+str(frameidx_list[inds][i])+'.png',img[i])

        #normalize
        lowdepths = torch.nn.functional.interpolate(depths[:, None], size = (self.h1//8, self.w1//8)).squeeze(1)
        highdepths = torch.nn.functional.interpolate(depths[:, None], size = (self.h1//2, self.w1//2)).squeeze(1)
        
        disps = 1.0/lowdepths
        highdisps = 1.0/highdepths

        # for i in range(depths.shape[0]):
        #     write_depth('./result/depth/'+str(i)+'.png', disps[i], False)
        #     write_depth('./result/depth/'+str(frameidx_list[inds][i])+'_high.png', highdisps[i], False)

        # m_s = highmidasdepths[highmidasdepths>0].mean()
        # lowmidasdepths = lowmidasdepths / m_s
        # highmidasdepths = highmidasdepths / m_s

        trackinfo = {
            'frames': torch.from_numpy(frameidx_list[inds]),
            'corner': corners,
            'rec': recs,
        }

        depth_valid = (0.2<1/disps)*(1/disps<50)
        high_depth_valid = (0.2<1/highdisps)*(1/highdisps<50)        
        
        if len(disps[disps>0.01]) > 0:
            s = disps[disps>0.01].mean()
            disps = disps / s
            highdisps = highdisps / s
            poses[...,:3] *= s
            objectposes[...,:3] *= s

        # a_list, b_list = [], []
        # for i in range(disps.shape[0]):
        #     depth = highdisps[i].reshape(-1)
        #     midepth = highmidasdepths[i].reshape(-1)
        #     midepth = torch.stack((midepth, torch.ones_like(midepth)), dim=1)
        #     left = torch.linalg.inv(torch.matmul(midepth.transpose(1,0), midepth))
        #     right = torch.matmul(midepth.transpose(1,0), depth)
        #     solve = torch.matmul(left, right)
        #     a, b = solve[0], solve[1]
        #     a_list.append(a)
        #     b_list.append(b)

        # a = torch.stack(a_list, dim=0)[..., None, None]
        # b = torch.stack(b_list, dim=0)[..., None, None]
        # highmidasdepths = a*highmidasdepths+b
        # lowmidasdepths = a*lowmidasdepths+b

        return (highimages, poses, objectposes, \
            lowmask, disps, highdisps,\
            mask, intrinsics, s, \
            depth_valid, high_depth_valid), trackinfo

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self

def depth_vis(disps, mode, max_depth):
    depth = 100*((((1.0/disps).clamp(max=max_depth)) * (655.35/max_depth)).cpu().detach().numpy())

    for i in range(disps.shape[0]):
        cv2.imwrite('./result/depth/depth' + mode +'_{}.png'.format(i),depth[i].astype(np.uint16))

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


