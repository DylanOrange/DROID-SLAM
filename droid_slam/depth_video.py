from tkinter import N
import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict
from lietorch import SE3

from droid_net import cvx_upsample
import geom.projective_ops as pops
import geom.ba as ba
from scipy.spatial.transform import Rotation as R
import cv2

def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat


class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0", n_object = 1, trackID = [1]):

        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.device = device
        self.trackID = trackID

        ### state attributes ###
        self.tstamp = torch.zeros(
            buffer, device=device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(
            buffer, 3, ht, wd, device=device, dtype=torch.uint8)
        self.objectmask = torch.zeros(
            buffer, n_object, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.dirty = torch.zeros(
            buffer, device=device, dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device=device,
                               dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(
            buffer, 7, device=device, dtype=torch.float).share_memory_()
        self.objectposes = torch.zeros(
            buffer, n_object, 7, device=device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(
            buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(
            buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(
            buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(
            buffer, 4, device=device, dtype=torch.float).share_memory_()
        self.faflow = torch.zeros(
            buffer, ht//8, wd//8, 2, device=device, dtype=torch.float).share_memory_()
        self.baflow = torch.zeros(
            buffer, ht//8, wd//8, 2, device=device, dtype=torch.float).share_memory_()
        self.dispgt = torch.ones(
            buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.objectgt = torch.zeros(
            buffer, n_object, 7, device=device, dtype=torch.float).share_memory_()
        self.posegt = torch.zeros(
            buffer, 7, device=device, dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(
            buffer, c, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8,
                                dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8,
                                dtype=torch.half, device=device).share_memory_()

        # initialize poses to identity transformation
        # self.poses[:] = torch.as_tensor(
        #     [0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        # self.poses[:] = torch.as_tensor(
        #     [0.37732657, -271.64998866, -1.05244905, -0.00972557, 0.90465151, -0.00738998, 0.42597706], dtype=torch.float, device=device)#c2w
        self.poses[:] = torch.as_tensor(
            [-7.061051, 271.5504, -2.342589, 0.00972557, -0.9046515, 0.00738998, 0.42597707], dtype=torch.float, device=device) #w2c, extrinsics
        # initialize object poses to identity transformation
        # self.objectposes[:] = torch.as_tensor(
        #     [11.76709, -270.0, -10.63032, 0.0, 0.95476813, 0.0, 0.29735134], dtype=torch.float, device=device) #o2w
        # self.objectposes[:,:] = torch.as_tensor(
        #    [3.65031972, 270.0, -15.43189153, 0.0, 0.95476813, 0.0, -0.29735134], dtype=torch.float, device=device) #w2o
        self.objectposes[:,:] = torch.as_tensor(
            [ -3.0760, 270.0000, -24.5112,  -0.0000,  -0.9526,  -0.0000,   0.3043], dtype=torch.float, device=device)
        self.objectgt[:,:] = torch.as_tensor(
            [ -3.0760, 270.0000, -24.5112,  -0.0000,  -0.9526,  -0.0000,   0.3043], dtype=torch.float, device=device) 
        # self.objectposes[:] = torch.as_tensor(
        #    [[3.65031972, 270.0, -15.43189153, 0.0, 0.95476813, 0.0, -0.29735134],
        #    [8.37189733, 270.0, -38.33762701, 0.0, -0.95356543, 0.0, 0.30118595]], dtype=torch.float, device=device) #w2o
        # self.objectposes[:] = torch.as_tensor(
        #     [8.37189733, 270.0, -38.33762701, 0.0, 0.95356543, 0.0, -0.30118595], dtype=torch.float, device=device) #w2o
        # self.objectposes[:] = torch.as_tensor(
        #     [0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.objectgraph = OrderedDict()

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]
        
        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.objectmask[index] = item[9]#这里维度前面有没有1都可以
        
        if len(item) > 10 and item[10] is not None:
            self.posegt[index] = item[10]
        
        if len(item) > 11 and item[11] is not None:
            self.objectgt[index] = item[11]
        
        if len(item) > 12 and item[12] is not None:
            self.dispgt[index] = item[12]

        if len(item) > 13 and item[13] is not None:
            self.objectposes[index] = item[13]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.objectmask[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            # print(self.counter.value)
            self.__item_setter(self.counter.value, item)

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj, device):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value, :3] *= s
            self.dirty[:self.counter.value] = True

    def reproject(self, ii, jj):
        """ project points from ii -> jj """

        ii, jj = DepthVideo.format_indicies(ii, jj, self.device)
        Gs = lietorch.SE3(self.poses[None]) 
        ObjectGs = lietorch.SE3(self.objectposes.transpose(1,0)) 
        objectmasks = self.objectmask.transpose(1,0)

        validmasklist = []
        for n, id in enumerate(self.trackID):
            if id in self.objectgraph.keys():
                object_app = torch.as_tensor(self.objectgraph[id], device=ii.device)
                ii_stamp = self.tstamp[ii]
                jj_stamp = self.tstamp[jj]
                validmask = (torch.isin(ii_stamp, object_app) & torch.isin(jj_stamp, object_app))

            else:
                validmask = torch.zeros_like(ii,dtype=torch.bool)

            validmasklist.append(validmask)

        validmask = torch.stack(validmasklist, dim=0)

        coords, valid_mask = \
            pops.dyprojective_transform(
                Gs, self.disps[None], self.intrinsics[None], ii, jj, validmask, ObjectGs, objectmasks)

        return coords, valid_mask

    # def reproject(self, ii, jj):
    #     """ project points from ii -> jj """
    #     ii, jj = DepthVideo.format_indicies(ii, jj, self.device)
    #     Gs = lietorch.SE3(self.poses[None])

    #     coords, valid_mask = \
    #         pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

    #     return coords, valid_mask
    
    def getgtflow(self, ii, jj, validmask):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj, self.device)
        Gs = lietorch.SE3(self.posegt[None])#1,350,7
        objectGs = lietorch.SE3(self.objectgt.transpose(1,0))#2,350,7
        objectmask = self.objectmask.transpose(1,0)

        coords, valid_mask = \
            pops.dyprojective_transform(Gs, self.dispgt[None], self.intrinsics[None], ii, jj, validmask, objectGs, objectmask)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
            
        ii, jj = DepthVideo.format_indicies(ii, jj, self.device)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """
        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)
    
    def dynamicba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=True):
        """ dense bundle adjustment (DBA) """

        min = torch.min(torch.cat([ii,jj],dim=0))
        max = torch.max(torch.cat([ii,jj],dim=0))

        disps = self.disps[min:max+1].unsqueeze(0)
        poses = SE3(self.poses[min:max+1].unsqueeze(0))
        objectposes = SE3(self.objectposes[min:max+1].transpose(1,0)) 
        objectmask = self.objectmask[min:max+1].transpose(1,0)
        intrinsics = self.intrinsics[min:max+1].unsqueeze(0)

        validmasklist = []
        app = {}
        # print('current window is {}'.format(self.tstamp[min:max+1]))
        for n, id in enumerate(self.trackID):
            if id in self.objectgraph.keys():
                # TODO:这里在最后traj-filter只会优化object-app里的，即插帧插出来的无法优化
                object_app = torch.as_tensor(self.objectgraph[id], device=ii.device)
                ii_stamp = self.tstamp[ii]
                jj_stamp = self.tstamp[jj]
                validmask = (torch.isin(ii_stamp, object_app) & torch.isin(jj_stamp, object_app))
                app[n] = torch.arange(max+1-min, device=min.device)[torch.isin(self.tstamp[min:max+1], object_app)]

            else:
                validmask = torch.zeros_like(ii,dtype=torch.bool)
                app[n] = torch.arange(0,device=min.device)
            # print('object in this window apps in {}'.format(app[n]))
            validmasklist.append(validmask)

        validmask = torch.stack(validmasklist, dim=0)
        # target, weight = self.getgtflow(ii,jj,validmask)#1,2,30,101,2

        ii = ii - min
        jj = jj - min
        t0 = t0 - min

        for i in range(itrs):
            # if (motion_only == False):
            #    poses, objectposes, disps= ba.dynamictestBA(target, weight, objectposes, objectmask,  app, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=t0)
            # else:
            poses, objectposes, disps = ba.dynamictestmoBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=t0)
        
        self.poses[min:max+1] = poses.data
        self.objectposes[min:max+1] = objectposes.data.transpose(1,0)
        # self.disps[min:max+1] = disps


def vis(mask, index):
    h, w = mask.shape
    image = torch.zeros((h,w,3),device=mask.device)
    image[mask>0] = torch.tensor((255.0,255.0,255.0), device=image.device)
    cv2.imwrite('objectmask_'+str(index) +'.png', image.cpu().numpy())