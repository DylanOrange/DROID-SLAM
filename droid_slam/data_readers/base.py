
import imghdr
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

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

        self.h1 = 240
        self.w1 = 808
        self.scale = 8
        self.cropscale = 2

        self.alltrackid = [1,2,3]
        
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
            # if not self.__class__.is_test_scene(scene):
            # dataset_index = []
            objectinfo = self.scene_info[scene]['object']
            for id in self.alltrackid:
                objectgraph = objectinfo[id][3]
                for i in objectgraph:
                    k = (objectgraph[i][1] > self.obfmin) & (objectgraph[i][1] < self.obfmax)
                    if k.any():
                        self.dataset_index.append((scene, id, i))
            # self.dataset_index[scene] = dataset_index
            # else:
            #     print("Reserving {} for validation".format(scene))

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
    def cornerinfo(objectmask, TRACKID):
        rec = torch.tensor([0,0])
        corners = []
        for id in TRACKID:
            mask = torch.where(objectmask == (id+1), 1.0, 0.0)
            coords = torch.nonzero(mask)
            corner_min = torch.min(coords, dim = 0).values[1:]
            corner_max = torch.max(coords, dim = 0).values[1:]
            rec = torch.maximum(corner_max - corner_min, rec)
            corners.append(corner_min)
        corners = torch.stack(corners, dim = 0)
        return corners, rec

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

    def build_object_frame_graph(self, poses, depths, intrinsics, object, f=8, max_flow=100):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.vkittidepth_read(fn)[0]
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
        # scene_id = random.sample(self.dataset_index.keys(), 1)[0] 
        # scene_id  = self.root
        # objectinfo = self.scene_info[scene_id]['object']
        # trackid = random.sample(self.alltrackid, 1)[0]
        # while(1):
        #     trackid = random.sample(self.alltrackid, 1)[0]  # 随机一个字典中的key，第二个参数为限制个数
        #     objectgraph = objectinfo[trackid][3]
        #     # graph = self.scene_info[scene]['graph']
        #     dataset_index= []
        #     for i in objectgraph:
        #             k = (objectgraph[i][1] > self.obfmin) & (objectgraph[i][1] < self.obfmax)
        #             if k.any():
        #                 dataset_index.append(i)
        #     if len(dataset_index) > self.n_frames-2:
        #         break
        # index = random.randint(0,len(self.dataset_index[scene_id])-1)
        # print('generated index is {}'.format(index))
        index = index % len(self.dataset_index)
        scene_id, trackid, ix = self.dataset_index[index]
        objectinfo = self.scene_info[scene_id]['object']
        objectgraph = objectinfo[trackid][3]

        # frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        sampledmasks_list = self.scene_info[scene_id]['objectmasks']

        frameidx_list =objectinfo[trackid][0]
        objectposes_list = objectinfo[trackid][1]
        objectmasks_list = objectinfo[trackid][2]

        inds = [ ix ]
        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            k = (objectgraph[ix][1] > self.obfmin) & (objectgraph[ix][1] < self.obfmax)
            frames = objectgraph[ix][0][k]

            if np.count_nonzero(frames[frames > ix]):
                ix = np.random.choice(frames[frames > ix])
            
            elif np.count_nonzero(frames):
                ix = np.random.choice(frames)
            
            inds += [ ix ]

        inds = np.array(inds)

        # print('scene is {}'.format(scene_id))
        # print('trackid is {}'.format(trackid))
        # print('object graph is {}'.format(inds))
        # for i in range(len(inds)-1):
        #     print('object flow is {}'.format(objectgraph[inds[i]][1][objectgraph[inds[i]][0] == inds[i+1]]))
        #     print('camera flow is {}'.format(frame_graph[inds[i]][1][frame_graph[inds[i]][0] == inds[i+1]]))


        # n_all = len(self.dataset_index)
        # step = 2
        # half = (self.n_frames//2)*step
        # if half<=ix<=n_all-half:
        #     inds = np.arange(ix - half, ix +half,step)
        # elif ix<20:
        #     inds = np.arange(ix, ix + half*2,step)
        # else:
        #     inds= np.arange(ix - half*2, ix,step)
        
        # inds =np.sort(inds)

        # inds_droid = [ idx ]
        # while len(inds_droid) < self.n_frames:
        #     # get other frames within flow threshold
        #     k = (frame_graph[idx][1] > self.fmin) & (frame_graph[idx][1] < self.fmax)
        #     frames = frame_graph[idx][0][k]

        #     # prefer frames forward in time
        #     if np.count_nonzero(frames[frames > idx]):
        #         idx = np.random.choice(frames[frames > idx])
            
        #     elif np.count_nonzero(frames):
        #         idx = np.random.choice(frames)

        #     inds_droid += [ idx ]

        # print('camera graph is {}'.format(inds_droid))
        # for i in range(len(inds_droid)-1):
        #     print('camera flow is {}'.format(frame_graph[inds_droid[i]][1][frame_graph[inds_droid[i]][0] == inds_droid[i+1]]))

        #读取mask并确定要追踪的车的id
        images, depths, poses, intrinsics, objectmasks, objectposes, sampledmasks, highdepths = [], [], [], [], [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[frameidx_list[i]]))
            depth, highdepth = self.__class__.vkittidepth_read(depths_list[frameidx_list[i]])
            depths.append(depth)
            highdepths.append(highdepth)
            poses.append(poses_list[frameidx_list[i]])
            intrinsics.append(intrinsics_list[frameidx_list[i]])

            objectmasks.append(objectmasks_list[i])
            objectposes.append(objectposes_list[i])

            # objectmasks.append(self.__class__.objectmask_read(objectmasks_list[i])[0])
            sampledmasks.append(self.__class__.objectmask_read(sampledmasks_list[frameidx_list[i]])[0])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        objectmasks = np.stack(objectmasks).astype(np.float32)
        objectposes = np.stack(objectposes).astype(np.float32)
        highdepths = np.stack(highdepths).astype(np.float32)

        sampledmasks = np.stack(sampledmasks).astype(np.float32)

        # for n, idx in enumerate(inds):
        #     vis_image = images[n].copy()
        #     vis_image[np.isin(sampledmasks[n], trackid+1)] = np.array([255.0,255.0,255.0])
        #     cv2.imwrite('./result/object/mask_'+ str(n)+'.png', vis_image)

        images = torch.from_numpy(images)
        N, h0, w0 = images.shape[:3]

        depths = torch.from_numpy(depths)
        highdepths = torch.from_numpy(highdepths)

        poses = torch.from_numpy(poses)
        objectmasks = torch.from_numpy(objectmasks)[None]
        objectposes = torch.from_numpy(objectposes)[None]
        sampledmasks = torch.from_numpy(sampledmasks)
        intrinsics = torch.from_numpy(intrinsics)
        intrinsics[:, 0:2] *= ((self.w1//self.scale)/ w0)
        intrinsics[:, 2:4] *= ((self.h1//self.scale)/ h0)


        # TRACKID, N_app, Apperance = self.trackinfo(objectmasks, inds)#

        # objectmasks = torch.nn.functional.interpolate(objectmasks[:, None].double(), size = (self.h1//self.cropscale, self.w1//self.cropscale), mode= 'bicubic').int().squeeze(1)#5,375,1242 ->5,120,404
        # corner, rec = self.cornerinfo(sampledmasks, TRACKID)

        images = images.permute(0, 3, 1, 2)
        images = torch.nn.functional.interpolate(images, size = (self.h1,self.w1), mode = 'bilinear')

        if self.aug is not None:
            # images, objectmasks, poses, disps, intrinsics = \
            #         self.aug(images, objectmasks, poses, disps, intrinsics)
            images = self.aug(images)

        # if TRACKID != -2:
        #     objectposes = self.__class__.objectpose_read(self.root, inds, TRACKID, Apperance)
        # else: 
        #     objectposes = torch.zeros(1,self.n_frames,7, dtype = poses.dtype)
        #     objectposes[...,-1] = 1.0

        quanmask = torch.nn.functional.interpolate(sampledmasks[:, None], size = (self.h1, self.w1)).squeeze(1)
        highmask = torch.where(quanmask == (trackid+1.0), 1.0, 0.0)[None]
        # quanmask = self.construct_objectmask(trackid, quanmask)

        # fullmasks = self.construct_objectmask(TRACKID, sampledmasks)#7,5,120,404

        # cropmasks = crop(fullmasks[..., None], corner, rec).squeeze(-1)#7,5,99,217
        
        # B = len(TRACKID)
        # objectmasks = fullmasks.view(B*N, -1, self.h1//self.cropscale, self.w1//self.cropscale)#35,1,120,404
        # objectmasks = torch.nn.functional.interpolate(objectmasks, size = (self.h1//self.scale, self.w1//self.scale))#35,1,120,404 ->35,1,30,101
        # objectmasks = objectmasks.view(B, N, self.h1//self.scale, self.w1//self.scale)#7,5,30,101

        # sampleddepths = torch.nn.functional.interpolate(depths[:, None], size = (self.h1, self.w1))#5,375,1242 ->5,240,808
        disps = 1.0/depths
        highdisps = 1.0/highdepths
        # max_depth = (1/disps).max()
        # depth_vis(disps,'gt', max_depth)

        # cropdepths = torch.nn.functional.interpolate(depths[:, None], size = (self.h1//self.cropscale, self.w1//self.cropscale))#5,1,375,1242 ->5,1,120,404
        # fulldepths = 1.0/cropdepths.squeeze(1)

        # cropdepths = cropdepths.transpose(0,1).expand(B,-1,-1,-1)
        # cropdepths = crop(cropdepths[..., None], corner, rec, depth =True)#7,5,99,217
        # cropdisps = 1.0/cropdepths.squeeze(-1)

        # if torch.isin(12, TRACKID) and 1.0/torch.mean(cropdisps[cropmasks>0.0]) > 20.0:
        #     cropdisps[:] = -0.1
            
        # batchgrid = batch_grid(corner, rec)
        Apperance = [torch.arange(self.n_frames)]
        trackinfo = {
            'trackid': torch.tensor([trackid]).to('cuda'),
            'apperance': [x.to('cuda') for x in Apperance],
            # 'n_app': N_app,
            'frames': frameidx_list[inds],
            # 'corner': corner.to('cuda'),
            # 'rec': rec.to('cuda'),
            # 'grid': tuple(t.to('cuda') for t in batchgrid)
        }

        # scale scened
        # objectdistance = (sampleddepths.squeeze(1))[(quanmask.int())[0]>0.0]
        # meandistance = objectdistance.mean()
        # print('real scale mean depth is {}'.format(meandistance))

        # if TRACKID != -2:
        #     print('real scale least depth is {}'.format(objectdistance.min()))
        #     print('real scale farest depth is {}'.format(objectdistance.max()))

        if len(disps[disps>0.01]) > 0:
            s = disps[disps>0.01].mean()
            disps = disps / s
            highdisps = highdisps / s
            poses[...,:3] *= s
            objectposes[...,:3] *= s

        # print('trackid is {}'.format(trackid))
        # print('poses is {}'.format(poses))
        # print('objectposes is {}'.format(objectposes))
        return images.to('cuda'), poses.to('cuda'), objectposes.to('cuda'), objectmasks.to('cuda'), disps.to('cuda'), highdisps.to('cuda'), highmask.to('cuda'), intrinsics.to('cuda'), trackinfo, s.to('cuda')

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self

def depth_vis(disps, mode, max_depth):
    depth = 100*((((1.0/disps).clamp(max=max_depth)) * (655.35/max_depth)).cpu().detach().numpy())

    for i in range(disps.shape[0]):
        cv2.imwrite('./result/depth/depth' + mode +'_{}.png'.format(i),depth[i].astype(np.uint16))
