
from configparser import Interpolation
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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


class Car4(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 1.0
    split = {
        'train': ('car4-full',),
        'val': ('room4-full',),
        'test': ('car4-full',)
    }
    obejctid = {
        'train':['truck-1', 'car-2'],
        'val':['car-2'],
    }
    intrinsics = {
        'train': np.array([564.3, 564.3, 480, 270]),
        'val': np.array([360, 360, 320, 240]),
    }
    scene = ['car4-full']

    def __init__(self, split_mode='train', **kwargs):
        self.split_mode = split_mode
        # self.objectid = ['truck-1', 'car-2']
        # self.midasdepth = '../MiDaS/output'
        super(Car4, self).__init__(name='Car4', split_mode = self.split_mode, **kwargs)

    @staticmethod
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building car4 dataset")

        scene_info = {}
        for scene in Car4.scene:
            for split in Car4.split[self.split_mode]:
                images = sorted(
                    glob.glob(osp.join(self.root, split, 'colour/*.png')))
                depths = sorted(
                    glob.glob(osp.join(self.root, split, 'depth_original/*.exr')))
                instancemasks = sorted(
                    glob.glob(osp.join(self.root, split, 'mask_id/*.png')))

                trajectory_path = osp.join(self.root, split, 'trajectories')

                poses = np.loadtxt(
                    osp.join(trajectory_path, 'gt-cam-0.txt'), delimiter=' ')[:, 1:]#c2w
                
                poses = SE3(torch.from_numpy(poses)).inv().data.numpy()#w2c
                
                object = self.object_read(trajectory_path, Car4.obejctid[self.split_mode], instancemasks)

                intrinsics = [Car4.intrinsics[self.split_mode]] * len(images)

                graph = self.build_frame_graph(poses, depths, intrinsics)

                objectinfo = self.build_object_frame_graph(poses, depths, intrinsics, object)

                scene_info[scene+'-'+split] = {'images': images, 'depths': depths,'objectmasks': instancemasks, 
                                        'poses': poses, 'intrinsics': intrinsics, 'graph': graph, 'object': objectinfo}

        return scene_info

    def object_read(self, datapath, trackid, maskpath):
        object = {}
        for id in trackid:
            raw_mat = np.loadtxt(os.path.join(datapath, 'gt-'+id+'.txt'), dtype = np.float32, delimiter=' ')
            poses = raw_mat[:,1:]
            indexlist = (raw_mat[:,0]).astype(np.int32)
            objectmasks = []
            num_id = int(id.split('-')[-1])
            valid_list = []
            index_list = []
            for i, index in enumerate(indexlist):
                mask = self.objectmask_read(maskpath[index-1])
                valid = np.where(mask == num_id, 1.0, 0.0)
                if np.count_nonzero(valid)!=0:
                    valid_list.append(i)
                    index_list.append(index-1)
                    objectmasks.append(np.where(mask == num_id, 1.0, 0.0))
            objectmasks = np.stack(objectmasks).astype(np.float32) 
            object[num_id] =[np.asarray(index_list, dtype = np.int32), poses[valid_list], objectmasks]
        return object

    @staticmethod
    def calib_read():
        return np.array([564.3, 564.3, 480, 270])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def objectmask_read(mask_file):
        mask = np.array(Image.open(mask_file))
        if mask.shape[-1] == 3:
            mask = mask[...,0]
        return mask

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / (Car4.DEPTH_SCALE)
        if depth.shape[-1] == 3:
            depth = depth[...,0]
        depth[depth>5.0] = 1.0
        depth[depth == np.nan] = 1.0
        depth[depth == np.inf] = 1.0
        depth[depth == 0] = 1.0
        return depth
    
    @staticmethod
    def vkittidepth_read(depth_file):
        # depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
        #                    cv2.IMREAD_ANYDEPTH) / (car4.DEPTH_SCALE*100)
        # resize(mask.resize((101,30)), (101,30), interpolation = cv2.INTER_NEAREST)
        depth = Image.open(depth_file)                
        depthlow = np.array(depth.resize((101,30), Image.NEAREST)) / (car4.DEPTH_SCALE*100)
        depthlow[depthlow == np.nan] = 1.0
        depthlow[depthlow == np.inf] = 1.0
        depthlow[depthlow == 0] = 1.0

        depthhigh = np.array(depth.resize((404,120), Image.NEAREST)) / (car4.DEPTH_SCALE*100)
        depthhigh[depthhigh == np.nan] = 1.0
        depthhigh[depthhigh == np.inf] = 1.0
        depthhigh[depthhigh == 0] = 1.0

        return depthlow, depthhigh #1/8, 1