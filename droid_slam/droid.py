import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.core.trajectory import PosePath3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        #求解w2c, 返回c2w
        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

    def evaluate(self):

        pred_pose = self.video.poses.cpu()
        gt_pose = self.video.posegt.cpu()
        counter = self.video.counter.value
        timestamp = self.video.tstamp[:counter].long().cpu()

        gt_objectpose = self.video.objectgt[:counter].cpu()
        pred_objectpose = self.video.objectposes[:counter].cpu()

        traj_est = PosePath3D(positions_xyz=pred_pose[:counter, :3],orientations_quat_wxyz=pred_pose[:counter, [6,3,4,5]])
        traj_ref = PosePath3D(positions_xyz=gt_pose[:counter, :3],orientations_quat_wxyz=gt_pose[:counter, [6,3,4,5]])

        obtraj_est_dict = {}
        obtraj_ref_dict = {}

        for n, id in enumerate(self.trackID):           
            
            object_timestamp = torch.as_tensor(list(self.objectposegt[id].keys()))

            keyob_timestamp = torch.as_tensor(np.intersect1d(object_timestamp, timestamp))

            gtidx = torch.isin(timestamp, keyob_timestamp)

            obtraj_est_dict[id] = PosePath3D(
                positions_xyz=pred_objectpose[gtidx, n, :3],
                orientations_quat_wxyz=pred_objectpose[gtidx][:,n, [6,3,4,5]])
            obtraj_ref_dict[id] = PosePath3D(
                positions_xyz=gt_objectpose[gtidx, n, :3],
                orientations_quat_wxyz=gt_objectpose[gtidx][:,n, [6,3,4,5]])

        result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) #input c2w
        print('---camera pose----')
        print(result)
    
        print('---object pose---')
        for id in self.trackID:
            result = main_ape.ape(obtraj_ref_dict[id], obtraj_est_dict[id], est_name='traj',
                                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) # input o2w
            print('------------result for car id ' + str(id)+'-----------')
            print(result)

