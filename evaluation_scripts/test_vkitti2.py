import sys
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


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def depth_read(depth_file):
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                        cv2.IMREAD_ANYDEPTH) / 100
    depth[depth == np.nan] = 1.0
    depth[depth == np.inf] = 1.0
    depth[depth == 0] = 1.0
    return depth


def image_stream(datapath, image_size=[240, 808], mode='val'):
    """ image generator """

    fx, fy, cx, cy = 725.0087, 725.0087, 620.5, 187

    # read all png images in folder
    split = {
        'train': 'clone',
        'val': '15-deg-left',
        'test': '30-deg-right'
    }
    images_list = sorted(glob.glob(os.path.join(
        datapath, split[mode], 'frames/rgb/Camera_0/*.jpg')))
    
    depths_list = sorted(glob.glob(os.path.join(
        datapath, split[mode], 'frames/depth/Camera_0/*.png')))

    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)

        depth = depth_read(depths_list[t])

        h0, w0, _ = image.shape
        h1, w1 = image_size[0], image_size[1]

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1 % 8, :w1-w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        depth = cv2.resize(depth, (w1,h1), cv2.INTER_NEAREST)
        depth = torch.as_tensor(depth)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0:2] *= (w1 / w0)
        intrinsics[2:4] *= (h1 / h0)

        yield t, image[None], depth, intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default = '/home/stud/lud/DeFlowSLAM/datasets/vkitti2/Scene18')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=700)
    parser.add_argument("--image_size", default=[240, 808])
    parser.add_argument("--disable_vis", default=True)

    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.25)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    # 2
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]
    args.stereo = False

    seq_id = args.datapath.split('/')[-1]  # for vis
    args.vis = True  # for vis
    args.loop_clo = True

    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)

    tstamps = []
    for (t, image, depth, intrinsics) in tqdm(image_stream(args.datapath)):
        if not args.disable_vis:
            show_image(image[0])
        droid.track(t, image, depth, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.datapath))

    # est_file = os.path.join(args.datapath, args.weights.split('/')[-1].split('.')[
    #                         0]+'_'+args.datapath.split('/')[-1]+'_cam0_est_quad.txt')
    # np.savetxt(est_file, traj_est, delimiter=' ')

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.trajectory import PosePath3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    def read_vkitti2_poses_file(file_path) -> PosePath3D:
        """
        parses pose file in Virtual KITTI 2 format (first 3 rows of SE(3) matrix per line)
        :param file_path: the trajectory file path (or file handle)
        :return: trajectory.PosePath3D
        """
        raw_mat = np.loadtxt(file_path, delimiter=' ', skiprows=1)[::2, 2:]
        error_msg = ("Virtual KITTI 2 pose files must have 16 entries per row "
                     "and no trailing delimiter at the end of the rows (space)")
        if raw_mat is None or (len(raw_mat) > 0 and len(raw_mat[0]) != 16):
            raise file_interface.FileInterfaceException(error_msg)
        try:
            mat = np.array(raw_mat).astype(float)
        except ValueError:
            raise file_interface.FileInterfaceException(error_msg)
        # yapf: disable
        poses = [np.linalg.inv(np.array([[r[0], r[1], r[2], r[3]],
                                         [r[4], r[5], r[6], r[7]],
                                         [r[8], r[9], r[10], r[11]],
                                         [r[12], r[13], r[14], r[15]]])) for r in mat]
        # yapf: enable
        if not hasattr(file_path, 'read'):  # if not file handle
            print("Loaded {} poses from: {}".format(len(poses), file_path))
        return PosePath3D(poses_se3=poses)

    gt_file = os.path.join(args.datapath, '15-deg-left/extrinsic.txt')
    traj_ref = read_vkitti2_poses_file(gt_file)

    traj_est = PosePath3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:])

    # est_file = os.path.join(args.datapath, args.weights.split('/')[-1].split('.')[
    #                         0]+'_'+args.datapath.split('/')[-1]+'_cam0_est.txt')
    # file_interface.write_kitti_poses_file(est_file, traj_est)

    # traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                          pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)