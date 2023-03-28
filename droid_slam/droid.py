from time import time
import torch
from lietorch import SE3
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
        self.args = args
        self.load_weights(args.weights)
        self.disable_vis = args.disable_vis

        self.objectposegt = None
        self.posegt = None

        self.trackID  = args.trackID

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, device=args.device, stereo=args.stereo, n_object=len(args.trackID), trackID=args.trackID)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh, obthresh=args.filter_obthresh, device=args.device)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(
                target=droid_visualization, args=(self.video, args.device))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            self.net, self.video, args.device)

    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights, self.args.device).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to(self.args.device).eval()

    def track(self, tstamp, image, objectmask, instance_mask, posegt, objectposegt, dispgt, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            
            self.filterx.track(tstamp, image, objectmask, instance_mask, posegt, objectposegt, dispgt, depth, intrinsics)

            self.frontend()

    def terminate(self, stream=None, need_inv=True):
        """ terminate the visualization process, return poses [t, q] """

        self.evaluate()

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        self.evaluate()

        camera_trajectory, object_trajectory = self.traj_filler(stream)
        if need_inv:
            return camera_trajectory.inv().data.cpu().numpy(), object_trajectory.inv().data.cpu().numpy(), 
        else:
            # for vkitti2(already w2c)
            return camera_trajectory.data.cpu().numpy(), object_trajectory.data.cpu().numpy(), 
    
    def evaluate(self):

        pred_pose = SE3(self.video.poses).inv().data.cpu()#w2c->c2w
        gt_pose = self.video.posegt.cpu()#已经转换为c2w
        counter = self.video.counter.value
        timestamp = self.video.tstamp[:counter].long().cpu()

        gt_objectpose = self.video.objectgt[:counter].cpu()#已经是o2w
        pred_objectpose = SE3(self.video.objectposes[:counter]).inv().data.cpu()#w2o->o2w

        traj_est = PosePath3D(positions_xyz=pred_pose[:counter, :3],orientations_quat_wxyz=pred_pose[:counter, [6,3,4,5]])
        traj_ref = PosePath3D(positions_xyz=gt_pose[:counter, :3],orientations_quat_wxyz=gt_pose[:counter, [6,3,4,5]])

        result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) #input c2w
        print('---camera pose----')
        print(result)

        obtraj_est_dict = {}
        obtraj_ref_dict = {}

        for n, id in enumerate(self.trackID):           
            
            object_timestamp = torch.as_tensor(list(self.objectposegt[id].keys()))

            keyob_timestamp = torch.as_tensor(np.intersect1d(object_timestamp, timestamp))

            gtidx = torch.isin(timestamp, object_timestamp)
            print(torch.equal(gtidx,torch.isin(timestamp, keyob_timestamp)))

            obtraj_est_dict[id] = PosePath3D(
                positions_xyz=pred_objectpose[gtidx, n, :3],
                orientations_quat_wxyz=pred_objectpose[gtidx][:,n, [6,3,4,5]])
            obtraj_ref_dict[id] = PosePath3D(
                positions_xyz=gt_objectpose[gtidx, n, :3],
                orientations_quat_wxyz=gt_objectpose[gtidx][:,n, [6,3,4,5]])
    
        print('---object pose---')
        for id in self.trackID:
            result = main_ape.ape(obtraj_ref_dict[id], obtraj_est_dict[id], est_name='traj',
                                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) # input o2w
            print('------------result for car id ' + str(id)+'-----------')
            print(result)
