
from configparser import Interpolation
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from scipy.spatial.transform import Rotation as R
from lietorch import SE3
from torch.functional import split
from .base import RGBDDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'vkitti2_test.txt')
test_split = open(test_split).read().split()


def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat


class binbin(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 1.0
    split = {
        'train': ('15-deg-left','15-deg-right','30-deg-left'),
        'val': ('clone',),
        'test': ('30-deg-right',)
    }
    scene = {
        'train': ('chair',),
        'val':('chair',),
    }
    def __init__(self, split_mode='train', **kwargs):
        self.split_mode = split_mode
        # self.midasdepth = '../MiDaS/output'
        super(binbin, self).__init__(name='binbin', split_mode = self.split_mode, **kwargs)

    @staticmethod
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building binbin dataset")

        scene_info = {}
        for scene in binbin.scene[self.split_mode]:
            if self.split_mode == 'train':
                splits = os.listdir(osp.join(self.root, scene))[:21]
            else:
                splits = os.listdir(osp.join(self.root, scene))[21:]
            for split in splits:
                images = sorted(
                    glob.glob(osp.join(self.root, scene, split, 'color/*.png')))
                depths = sorted(
                    glob.glob(osp.join(self.root, scene, split, 'depth/*.png')))
                instancemasks = sorted(
                    glob.glob(osp.join(self.root, scene, split, 'instance/*.png')))
                invalids = sorted(
                    glob.glob(osp.join(self.root, scene, split, 'invalid/*.png')))

                info_file = osp.join(self.root, scene, split, 'info.pkl')
                poses, objectposes, intrinsics = binbin.info_read(info_file)#c2w, o2w

                poses = SE3(torch.from_numpy(poses)).inv().data.numpy()#w2c

                # points = poses[:,:3].T
                # objectpoints = objectposes[:,:3].T
    
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection = '3d')
                # ax.plot(points[0], points[1], points[2], marker = 'x')
                # ax.plot(objectpoints[0], objectpoints[1], objectpoints[2], marker = '+')
                # ax.scatter(*points.T[0], color = 'red')
                # plt.show()

                masks = []
                for mask_file in instancemasks:
                    masks.append(binbin.objectmask_read(mask_file))
                objectmasks = np.stack(masks)

                object = {}
                object[1] = [np.arange(len(images), dtype=np.int32), objectposes, objectmasks]

                intrinsics = [intrinsics] * len(images)

                graph = self.build_frame_graph(poses, depths, intrinsics)

                objectinfo = self.build_object_frame_graph(poses, depths, intrinsics, object)

                scene_info[scene+'-'+split] = {'images': images, 'depths': depths,'objectmasks': instancemasks, 'invalids':invalids,
                                        'poses': poses, 'intrinsics': intrinsics, 'graph': graph, 'object': objectinfo}

        return scene_info

    @staticmethod
    def objectpose_read(datapath, ids, trackid, apperance):
        objectpose_list = []
        filepath = osp.join(datapath, '15-deg-left', 'pose.txt')
        # print('trackid is {}'.format(trackid))
        # print('appear is {}'.format(apperance))
        # print('frames are {}'.format(ids))

        for n, id in enumerate(trackid):
            objectpose = torch.zeros((len(ids), 7), dtype = torch.float)
            idx = apperance[n]
            raw_mat = torch.from_numpy(np.loadtxt(filepath, dtype = np.float32, delimiter=' ', skiprows=1))
            mask = ((raw_mat[:,1] == 0) & (raw_mat[:,2] == id)) & (torch.isin(raw_mat[:, 0], torch.tensor(ids)))
            mat = raw_mat[mask]
            raw_pose = mat[:,7:13]
            r = raw_pose[:,3] +torch.pi/2
            rotation = R.from_euler('y', r)
            o2w = torch.concat((raw_pose[:, 0:3], torch.from_numpy(rotation.as_quat()).float()), dim=1)
            objectpose[idx] = o2w
            noidx = torch.from_numpy(np.setdiff1d(idx, np.arange(len(ids))))
            objectpose[noidx] = torch.tensor([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype = torch.float)
            objectpose_list.append(objectpose)
        if objectpose_list == []:
            print('no trackid !! {}'.format(ids))
        objectposes = torch.stack(objectpose_list, dim = 0)
        return objectposes

    def object_read(self, datapath, trackid, maskpath):
        object = {}
        for id in trackid:
            raw_mat = np.loadtxt(datapath, dtype = np.float32, delimiter=' ', skiprows=1)
            mask = (raw_mat[:,1] == 0) & (raw_mat[:,2] == id)
            mat = raw_mat[mask]
            raw_pose = mat[:,7:13]
            r = raw_pose[:,3] +np.pi/2
            rotation = R.from_euler('y', r)
            poses = np.concatenate((raw_pose[:, 0:3], rotation.as_quat().astype(np.float32)), axis=1)
            indexlist = mat[:,0].astype(np.int32)
            objectmasks = []
            for i in indexlist:
                mask = self.objectmask_read(maskpath[i])[0]
                objectmasks.append(np.where(mask == (id+1.0), 1.0, 0.0))
            objectmasks = np.stack(objectmasks).astype(np.float32) 
            object[id] =[indexlist, poses, objectmasks]
        return object

    @staticmethod
    def info_read(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        poses = np.stack(data['pose'])
        r = rmat_to_quad(poses[:, 0:3, 0:3])
        t = poses[:, :3, 3] / binbin.DEPTH_SCALE
        poses = np.concatenate((t, r), axis=1)#w2c
        
        objectposes = np.stack(data['object_poses']['Model_1'])
        r = rmat_to_quad(objectposes[:, 0:3, 0:3])
        t = objectposes[:, :3, 3] / binbin.DEPTH_SCALE
        objectposes = np.concatenate((t, r), axis=1).astype(np.float32)#w2c

        intrinsics = np.asarray(data['calib'])

        return poses, objectposes, intrinsics
    
    @staticmethod
    def calib_read():
        return np.array([725.0087, 725.0087, 620.5, 187])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def objectmask_read(mask_file):
        mask = Image.open(mask_file)
        return np.array(mask)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / (binbin.DEPTH_SCALE*1000)
        depth[depth<1e-1] = 1.0
        depth[depth>1e2] = 1.0
        depth[depth == np.nan] = 1.0
        depth[depth == np.inf] = 1.0
        depth[depth == 0] = 1.0
        return depth
    
    @staticmethod
    def vkittidepth_read(depth_file):
        # depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
        #                    cv2.IMREAD_ANYDEPTH) / (VKitti2.DEPTH_SCALE*100)
        # resize(mask.resize((101,30)), (101,30), interpolation = cv2.INTER_NEAREST)
        depth = Image.open(depth_file)                
        depthlow = np.array(depth.resize((101,30), Image.NEAREST)) / (binbin.DEPTH_SCALE*100)
        depthlow[depthlow == np.nan] = 1.0
        depthlow[depthlow == np.inf] = 1.0
        depthlow[depthlow == 0] = 1.0

        depthhigh = np.array(depth.resize((404,120), Image.NEAREST)) / (binbin.DEPTH_SCALE*100)
        depthhigh[depthhigh == np.nan] = 1.0
        depthhigh[depthhigh == np.inf] = 1.0
        depthhigh[depthhigh == 0] = 1.0

        return depthlow, depthhigh #1/8, 1