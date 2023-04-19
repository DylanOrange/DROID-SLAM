
from configparser import Interpolation
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

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


class VKitti2(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 1.0
    split = {
        'train': ('15-deg-left','15-deg-right','30-deg-left'),
        'val': ('clone',),
        'test': ('30-deg-right',)
    }
    scene = {
        'train': ('Scene06',),
        'val':('Scene18',),
    }
    def __init__(self, split_mode='train', **kwargs):
        self.split_mode = split_mode
        # self.midasdepth = '../MiDaS/output'
        super(VKitti2, self).__init__(name='VKitti2-Scene06', split_mode = self.split_mode, **kwargs)

    @staticmethod
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building VKitti2 dataset")

        scene_info = {}
        for scene in VKitti2.scene[self.split_mode]:
            for split in VKitti2.split['train']:
                images = sorted(
                    glob.glob(osp.join(self.root, scene, split, 'frames/rgb/Camera_0/*.jpg')))
                depths = sorted(
                    glob.glob(osp.join(self.root, scene, split, 'frames/depth/Camera_0/*.png')))
                instancemasks = sorted(
                    glob.glob(osp.join(self.root, scene, split, 'frames/instanceSegmentation/Camera_0/*.png')))
                # midasdepth = sorted(
                #     glob.glob(osp.join(self.midasdepth, scene, split, '*.pfm')))
                # 注意camera pose的选择

                objectposepath = osp.join(self.root, scene, split, 'pose.txt')

                poses = np.loadtxt(
                    osp.join(self.root, scene, split, 'extrinsic.txt'), delimiter=' ', skiprows=1)[::2, 2:]
                poses = poses.reshape(-1, 4, 4)
                r = rmat_to_quad(poses[:, 0:3, 0:3])
                t = poses[:, :3, 3] / VKitti2.DEPTH_SCALE
                poses = np.concatenate((t, r), axis=1)#w2c

                # translation + Quaternion
                with open(osp.join(self.root, scene, split, 'info.txt')) as f:
                    objectid = [int(line.split()[0]) for line in f.readlines()[1:-1]]
                
                object = self.object_read(objectposepath, objectid, instancemasks)#index, poses, objectmask at 1/8 resolution

                intrinsics = [VKitti2.calib_read()] * len(images)

                graph = self.build_frame_graph(poses, depths, intrinsics)

                objectinfo = self.build_object_frame_graph(poses, depths, intrinsics, object)

                scene_info[scene+'-'+split] = {'images': images, 'depths': depths,'objectmasks': instancemasks, 
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
    def calib_read():
        return np.array([725.0087, 725.0087, 620.5, 187])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def objectmask_read(mask_file):
        mask = Image.open(mask_file)
        return np.array(mask), np.array(mask.resize((101,30), Image.NEAREST)) #1, 1/8

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / (VKitti2.DEPTH_SCALE*100)
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
        depthlow = np.array(depth.resize((101,30), Image.NEAREST)) / (VKitti2.DEPTH_SCALE*100)
        depthlow[depthlow == np.nan] = 1.0
        depthlow[depthlow == np.inf] = 1.0
        depthlow[depthlow == 0] = 1.0

        depthhigh = np.array(depth.resize((404,120), Image.NEAREST)) / (VKitti2.DEPTH_SCALE*100)
        depthhigh[depthhigh == np.nan] = 1.0
        depthhigh[depthhigh == np.inf] = 1.0
        depthhigh[depthhigh == 0] = 1.0

        return depthlow, depthhigh #1/8, 1

    @staticmethod
    def flow_read(flow_file):
        bgr = cv2.imread(flow_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        h, w, _c = bgr.shape
        # 因为裁减了图片，所以光流也要变化
        # b: invalid(max 2**16-1), g: flow_y, r: flow_x
        # invalid = bgr[..., 0] == 0
        out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
        out_flow[..., 0] *= w - 1
        out_flow[..., 1] *= h - 1
        val = (bgr[..., 0] > 0).astype(np.float32)
        # out_flow[invalid] = 0
        # out_flow = np.concatenate(
        #     [out_flow, bgr[..., 0:1] / (2**16 - 1.0)], axis=-1)
        return out_flow, val

    @staticmethod
    def dymask_read(mask_file):
        content = np.load(mask_file)
        return content[..., 0], content[..., 1]


class VKitti2Stream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VKitti2Stream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/VKitti2'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class VKitti2TestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VKitti2TestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt',
                           self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
