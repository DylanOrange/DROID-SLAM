
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

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'vkitti2_test.txt')
test_split = open(test_split).read().split()


def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat


class Own_easy(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 1.0
    split = {
        'train': ('train',),
        'val': ('val',),
        'test': ('test',)
    }

    def __init__(self, split_mode='train', **kwargs):
        self.split_mode = split_mode
        # self.midasdepth = '../MiDaS/output'
        super(Own_easy, self).__init__(name='Own_easy', split_mode = self.split_mode, **kwargs)

    @staticmethod
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building own dataset")

        #'room0_car' 深度多了两百帧
        scene_info = {}
        # scene_dir = os.listdir(osp.join(self.root, self.split_mode))
        scene_dir = ['room0_car', 'apartment5_car', 'apartment0_car']
        for scene in scene_dir:
            images = sorted(
                glob.glob(osp.join(self.root, self.split_mode, scene, 'color/*.png')))
            depths = sorted(
                glob.glob(osp.join(self.root, self.split_mode, scene, 'depth/*.exr')))
            instancemasks = sorted(
                glob.glob(osp.join(self.root, self.split_mode, scene, 'mask/*.png')))
            
            objectpose_path = osp.join(self.root, self.split_mode, scene, 'object.txt')

            poses = np.loadtxt(
                osp.join(self.root, self.split_mode, scene, 'camera.txt'), delimiter=' ')[:, 1:]#c2w
            
            poses = SE3(torch.from_numpy(poses)).inv().data.numpy()#w2c

            object = self.object_read(objectpose_path, instancemasks)

            intrinsics = [self.calib_read(osp.join(self.root, self.split_mode, scene, 'calibration.txt'))] * len(images)

            graph = self.build_frame_graph(poses, depths, intrinsics)

            objectinfo = self.build_object_frame_graph(poses, depths, intrinsics, object)

            scene_info[scene] = {'images': images, 'depths': depths,'objectmasks': instancemasks, 
                                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph, 'object': objectinfo}

        return scene_info

    def object_read(self, datapath, maskpath):
        object = {}
        objectposes = np.loadtxt(datapath, dtype = np.float32, delimiter=' ')[:, 1:]#c2w
        indexlist = np.arange(objectposes.shape[0]).astype(np.int32)
        objectmasks = []
        for index in indexlist:
            objectmask = self.objectmask_read(maskpath[index])[0]
            objectmasks.append(objectmask)
        objectmasks = np.stack(objectmasks)
        object[1] =[indexlist, objectposes, objectmasks]

        # object = {}
        # for id in trackid:
        #     raw_mat = np.loadtxt(os.path.join(datapath, 'gt-'+id+'.txt'), dtype = np.float32, delimiter=' ')
        #     poses = raw_mat[:,1:]
        #     indexlist = (raw_mat[:,0]).astype(np.int32)
        #     objectmasks = []
        #     num_id = int(id.split('-')[-1])
        #     valid_list = []
        #     index_list = []
        #     for i, index in enumerate(indexlist):
        #         mask = self.objectmask_read(maskpath[index-1])
        #         valid = np.where(mask == num_id, 1.0, 0.0)
        #         if np.count_nonzero(valid)!=0:
        #             valid_list.append(i)
        #             index_list.append(index-1)
        #             objectmasks.append(np.where(mask == num_id, 1.0, 0.0))
        #     objectmasks = np.stack(objectmasks).astype(np.float32) 
        #     object[num_id] =[np.asarray(index_list, dtype = np.int32), poses[valid_list], objectmasks]
        return object

    @staticmethod
    def calib_read(path):
        calib = np.loadtxt(path, delimiter=' ')
        return calib

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def objectmask_read(mask_file):
        mask = cv2.imread(mask_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        objectmask = (mask != [0,0,0]).all(2).astype(np.float32)

        # vis_image = np.zeros((480,640,3))
        # vis_image[objectmask>0] = np.array([255,255,255])

        # diff = mask - vis_image
        # cv2.imwrite('test.png', vis_image)
        return objectmask[8//2::8, 8//2::8], objectmask #1, 1/8

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / (Own_easy.DEPTH_SCALE)
        if depth.shape[-1] == 3:
            depth = depth[...,0]
        depth[depth>10.0] = 1.0
        depth[depth == np.nan] = 1.0
        depth[depth == np.inf] = 1.0
        depth[depth == 0] = 1.0
        return depth