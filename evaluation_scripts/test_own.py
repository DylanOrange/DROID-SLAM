import sys
sys.path.append('droid_slam')
from droid import Droid

import glob
import torch.nn.functional as F
import argparse
import time
import os
import cv2
from lietorch import SE3
import torch
import numpy as np
from tqdm import tqdm


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(datapath, image_size=[360, 480], mode='val'):
    """ image generator """

    calib = np.loadtxt(os.path.join(datapath, 'calibration.txt'), delimiter=' ')

    images_list = sorted(glob.glob(os.path.join(
        datapath, 'color/*.png')))

    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)

        h0, w0, _ = image.shape
        h1, w1 = image_size[0], image_size[1]

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1 % 8, :w1-w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor(calib.copy())
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default = '/storage/slurm/xiny/dataset/cofusion/own/dataset/val/room0_truck')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=300)
    parser.add_argument("--image_size", default=[360, 480])
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
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.datapath)):
        if not args.disable_vis:
            show_image(image[0])
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.datapath))

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.trajectory import PosePath3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    def read_poses_file(file_path) -> PosePath3D:
        """
        parses pose file in Virtual KITTI 2 format (first 3 rows of SE(3) matrix per line)
        :param file_path: the trajectory file path (or file handle)
        :return: trajectory.PosePath3D
        """
        poses = np.loadtxt(file_path, delimiter=' ')[:, 1:]#c2w
        traj_ref = PosePath3D(positions_xyz=poses[:, :3],orientations_quat_wxyz=poses[:, 3:])
        return traj_ref

    gt_file = os.path.join(args.datapath, 'camera.txt')
    traj_ref = read_poses_file(gt_file)

    traj_est = PosePath3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:])

    result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                          pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)