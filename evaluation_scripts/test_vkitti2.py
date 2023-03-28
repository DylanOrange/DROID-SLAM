from asyncio import FastChildWatcher
from pickle import FALSE, TRUE
import sys

from sklearn.linear_model import MultiTaskLasso
sys.path.append('droid_slam')
from droid import Droid

import glob
import torch.nn.functional as F
import argparse
import time
import os
import cv2
import lietorch
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
from lietorch import SE3

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.core.trajectory import PosePath3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

def read_vkitti2_poses_file(file_path) -> PosePath3D:

    raw_mat = np.loadtxt(file_path, delimiter=' ', skiprows=1)[::2, 2:]
    mat = np.array(raw_mat).astype(float)
    # mat = mat[:framenumber]

    poses = [np.linalg.inv(np.array([[r[0], r[1], r[2], r[3]],
                                     [r[4], r[5], r[6], r[7]],
                                     [r[8], r[9], r[10], r[11]],
                                     [r[12], r[13], r[14], r[15]]])) for r in mat]
    
    # poses = [np.array([[r[0], r[1], r[2], r[3]],
    #                             [r[4], r[5], r[6], r[7]],
    #                             [r[8], r[9], r[10], r[11]],
    #                             [r[12], r[13], r[14], r[15]]]) for r in mat]
    return poses

def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat

def upsample_inter(mask):
    batch, num, ht, wd, dim = mask.shape
    mask = mask.permute(0, 1, 4, 2, 3).contiguous()
    mask = mask.view(batch*num, dim, ht, wd)
    mask = F.interpolate(mask, scale_factor=8, mode='bilinear',
                         align_corners=True, recompute_scale_factor=True)
    mask = mask.permute(0, 2, 3, 1).contiguous()
    return mask.view(batch, num, 8*ht, 8*wd, dim)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def depth_read(depth_file):
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                        cv2.IMREAD_ANYDEPTH) / (100.0)
    depth[depth == np.nan] = 1.0
    depth[depth == np.inf] = 1.0
    depth[depth == 0] = 1.0
    return depth

def objectposes_read(file_path, trackID):

    objectpose = {}
    raw_mat = np.loadtxt(file_path, delimiter=' ', skiprows=1)
    mask = ((raw_mat[:,1] == 0) & (raw_mat[:,2] == trackID))
    mat = raw_mat[mask]
    raw_pose = mat[:,7:13]
    idx = mat[:,0]

    r = raw_pose[:,3] +np.pi/2
    rotation = R.from_euler('y', r)
    poses = np.concatenate((raw_pose[:, 0:3], rotation.as_quat()), axis=1)
    o2w = torch.as_tensor(poses, dtype=torch.double)

    # o2w = SE3(o2w)
    # w2o = o2w.inv().data

    for i, index in enumerate(idx):
        objectpose[int(index)] = o2w[i]

    return objectpose

def pose_read(file_path):
    poses = np.loadtxt(file_path, delimiter=' ', skiprows=1)[::2, 2:]
    poses = poses.reshape(-1, 4, 4)
    r = rmat_to_quad(poses[:, 0:3, 0:3])
    t = poses[:, :3, 3] 
    poses = torch.as_tensor(np.concatenate((t, r), axis=1), dtype=torch.double)

    #比较要用c2w
    w2c = SE3(poses)
    c2w = w2c.inv().data

    return c2w

def image_stream(datapath, trackID, image_size=[240, 808], mode='val'):
    """ image generator """

    fx, fy, cx, cy = 725.0087, 725.0087, 620.5, 187
    h1, w1 = image_size[0], image_size[1]
    Id = torch.as_tensor([ -3.0760, 270.0000, -24.5112,  -0.0000,  -0.9526,  -0.0000,   0.3043], dtype=torch.double)

    split = {
        'train': 'clone',
        'val': '15-deg-left',
        'test': '30-deg-right'
    }

    images_list = sorted(glob.glob(os.path.join(
        datapath, split[mode], 'frames/rgb/Camera_0/*.jpg')))
    
    mask_list = sorted(glob.glob(os.path.join(
        datapath, split[mode], 'frames/instanceSegmentation/Camera_0/*.png')))
    
    depths_list = sorted(glob.glob(os.path.join(
        datapath, split[mode], 'frames/depth/Camera_0/*.png')))
    
    # pose_path = os.path.join(datapath, split[mode], 'extrinsic.txt')

    # objectpose_path = os.path.join(datapath, split[mode], 'pose.txt')

    for t, imfile in enumerate(images_list):
        #read image
        image = cv2.imread(imfile)
        h0, w0, _ = image.shape

        #read depths
        depth = (cv2.resize(depth_read(depths_list[t]), (w1//8, h1//8)))#1,30,101
        dispgt = torch.from_numpy(1.0 / depth)

        #read mask
        mask = Image.open(mask_list[t])

        #read gtpose
        pose_gt = poses_gt[t]

        # visualize object mask
        # if ((t+1)%20 == 0):
        #     vis_image = np.copy(image)
        #     maskarray = np.array(mask)
        #     for id in trackID:
        #         orignial_mask = np.where((maskarray == (id+1)), 1, 0)
        #         vis_image[np.nonzero(orignial_mask)] = np.array([255,255,255])
        #     cv2.imwrite(os.path.join('./intermediate', 'mask_{}.png'.format(t)), vis_image)

        #downsample objectmask
        maskarray = np.array(mask.resize((w1//8,h1//8)))#30,101
        instance_mask = torch.from_numpy(maskarray)
        objectmask_list = []
        objectpose_list = []

        if not trackID:
                objectmasks = torch.zeros(1,h1//8,w1//8)
                objectpose_gt = Id[None]
        
        else:
            for id in trackID:
                objectmask = np.where((maskarray == (id+1)), 1, 0)
                objectmask_list.append(torch.as_tensor(objectmask))
                if torch.isin(t, torch.tensor(list(objectpose_dict[id].keys()))):
                    objectpose_list.append(objectpose_dict[id][t])
                else:
                    objectpose_list.append(Id)
            objectmasks = torch.stack(objectmask_list, dim = 0)#2,30,101
            objectpose_gt = torch.stack(objectpose_list, dim = 0)#2,7

        #instance segementation mask

        #downsample image
        image = cv2.resize(image, (w1, h1))#240,808
        image = torch.as_tensor(image).permute(2, 0, 1)

        #read intrinsics
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0:2] *= (w1 / w0)
        intrinsics[2:4] *= (h1 / h0)

        yield t, image[None], objectmasks[None], instance_mask[None], intrinsics, pose_gt[None], objectpose_gt[None], dispgt[None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default = '../DeFlowSLAM/datasets/vkitti2/Scene20')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", default="checkpoints/12_2_objectgraph_smconsobflow_oldconf_032000.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[240, 808])
    parser.add_argument("--disable_vis", action="store_true", default= True)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--filter_obthresh", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--keyframe_thresh", type=float, default=0.0)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--framenumber", type=int, default=100)
    parser.add_argument("--trackID", type=int, default=[0,1,3])
    parser.add_argument("--disable_object", type=int, default=False)
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]
    args.stereo = False
    if args.disable_object == True:
        args.trackID = []
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)

    #read objectpose
    objectpose_path = os.path.join(args.datapath, '15-deg-left', 'pose.txt')
    objectpose_dict = {}
    for id in args.trackID:
        objectpose_gt = objectposes_read(objectpose_path, id)
        objectpose_dict[id] = objectpose_gt

    #read pose
    pose_path = os.path.join(args.datapath, '15-deg-left', 'extrinsic.txt')
    poses_gt = pose_read(pose_path)

    # traj_ref = read_vkitti2_poses_file(os.path.join(args.datapath, '15-deg-left/extrinsic.txt'))

    droid.objectposegt = objectpose_dict
    # droid.posegt = traj_ref

    for (t, image, objectmask, instance_mask, intrinsics, posegt, objectposegt, dispgt) in tqdm(image_stream(args.datapath, args.trackID)):
        if not args.disable_vis:
            show_image(image[0])
        droid.track(t, image, objectmask, instance_mask, posegt, objectposegt, dispgt, intrinsics=intrinsics)

    # print(droid.video.poses[:10])
    traj_est, obtraj_est = droid.terminate(image_stream(args.datapath, args.trackID), need_inv=True)

    ### run evaluation ###

    print("#"*20 + " Results...")

    traj_est = PosePath3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, [6,3,4,5]])

    traj_ref = PosePath3D(
        positions_xyz=poses_gt[:, :3].numpy(),
        orientations_quat_wxyz=poses_gt[:, [6,3,4,5]].numpy())
    
    # obtraj_est_dict = {}
    # obtraj_ref_dict = {}
    # for n, id in enumerate(args.trackID):
    #     exist = objectpose_dict[id].shape[0]
    #     obtraj_est_dict[id] = PosePath3D(
    #         positions_xyz=obtraj_est[:exist, n, :3],
    #         orientations_quat_wxyz=obtraj_est[:exist, n, [6,3,4,5]])
    #     obtraj_ref_dict[id] = PosePath3D(
    #         positions_xyz=objectpose_dict[id][:,:3],
    #         orientations_quat_wxyz=objectpose_dict[id][:,[6,3,4,5]])

    # est_file = './result/joint/cam_est.txt'
    # gt_file = './result/joint/cam_ref.txt'
    # file_interface.write_kitti_poses_file(gt_file, traj_ref)
    # file_interface.write_kitti_poses_file(est_file, traj_est)
    # ob_estfile = './result/joint/object_est.txt'
    # ob_reffile = './result/joint/object_ref.txt'
    # file_interface.write_kitti_poses_file(ob_estfile, obtraj_est_dict[args.trackID[0]])
    # file_interface.write_kitti_poses_file(ob_reffile, obtraj_ref_dict[args.trackID[0]])

    # print(traj_est.positions_xyz[:5])
    # print(traj_est.orientations_quat_wxyz[:5])
    # print(traj_ref.positions_xyz[:5])
    # print(traj_ref.orientations_quat_wxyz[:5])

    result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                          pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) #input c2w
    print('---camera pose----')
    print(result)
    
    # print('---object pose---')
    # for id in args.trackID:
    #     result = main_ape.ape(obtraj_ref_dict[id], obtraj_est_dict[id], est_name='traj',
    #                         pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) # input o2w
    #     print('------------result for car id ' + str(id)+'-----------')
    #     print(result)
