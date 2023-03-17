import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(
            video, net.update, args.device, max_factors=48)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def __update(self):
        """ add edges, perform update """

        print('frontend update')
        self.count += 1#update的次数
        self.t1 += 1#总是等于上次更新时的关键帧数量，总是滞后一帧

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)#去掉更新次数超过25次的

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
                                         rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0,
                                                  self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])#sens和disps什么关系?->disps_sens全是0, disps仍是disps

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        d = self.video.distance(
            [self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)
        
        #检测倒数第二和第三帧的光流是否足够大
        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.objectposes[self.t1] = self.video.objectposes[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

    def __initialize(self):
        """ initialize the SLAM system """

        print('frontend initialization')
        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(t0 = 1, use_inactive=True)

        self.graph.add_proximity_factors(
            t0 = 0, t1 = 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        # TODO:
        for itr in range(8):
            self.graph.update(t0 = 1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.objectposes[self.t1] = self.video.objectposes[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            # self.graph.print_edges()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
            # self.graph.print_edges()
