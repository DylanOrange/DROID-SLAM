import cv2
import torch
import lietorch

from lietorch import SE3
from collections import OrderedDict
from factor_graph import FactorGraph
from droid_net import DroidNet
import geom.projective_ops as pops


class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """

    def __init__(self, net, video, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    def __fill(self, tstamps, images, intrinsics, objectmasks, posegts, objectposegts, dispgts):
        """ fill operator """

        tt = torch.as_tensor(tstamps, device=self.device)
        images = torch.stack(images, 0) #stack will add one more dimension
        objectmasks = torch.cat(objectmasks, 0) 
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images[:,:,[2,1,0]].to(self.device) / 255.0
        posegt = torch.cat(posegts,0)
        objectposegt = torch.cat(objectposegts, 0)#16,2,7
        dispgt = torch.cat(dispgts,0)#16,,30,101
        
        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(tstamps)

        ts = self.video.tstamp[:N]#0,5,10,15
        Ps = SE3(self.video.poses[:N])
        ObPs = SE3(self.video.objectposes[:N])#8,2,6

        t0 = torch.as_tensor([ts[ts<=t].shape[0] - 1 for t in tstamps])#对于每个tstamps, 小于tstamps的有几个，找到对应关键帧的插值区间
        t1 = torch.where(t0<N-1, t0+1, t0)

        dt = ts[t1] - ts[t0] + 1e-3#关键帧中间有几帧的间隔
        dP = Ps[t1] * Ps[t0].inv()#关键帧中间的pose之差
        ObdP = ObPs[t1] * ObPs[t0].inv()

        v = dP.log() / dt.unsqueeze(-1)#每帧平均应该有多大的pose变化
        w = v * (tt - ts[t0]).unsqueeze(-1)#ts[t0]是从第几帧关键帧开始插帧

        obv = ObdP.log() / dt.view(M,1,1)#16,2,6
        obw = obv * (tt - ts[t0]).view(M,1,1)

        Gs = SE3.exp(w) * Ps[t0]
        ObGs = SE3.exp(obw) * ObPs[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        self.video.counter.value += M
        self.video[N:N+M] = (tt, images[:,0], Gs.data, 1, None, intrinsics / 8.0, fmap ,None, None, objectmasks, posegt, objectposegt, dispgt, ObGs.data)

        graph = FactorGraph(self.video, self.update)
        graph.add_factors(t0.to(self.device), torch.arange(N, N+M).to(self.device))#插帧用的关键帧和新加进来的帧连接起来
        graph.add_factors(t1.to(self.device), torch.arange(N, N+M).to(self.device))

        for itr in range(6):
            # print('traj update')
            graph.update(N, N+M, motion_only=True)
    
        Gs = SE3(self.video.poses[N:N+M].clone())
        ObGs = SE3(self.video.objectposes[N:N+M].clone())
        self.video.counter.value -= M

        return [ Gs ], [ObGs]

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images """

        # store all camera poses
        pose_list = []
        Obpose_list = []

        tstamps = []
        images = []
        intrinsics = []
        objectmasks = []
        posegts  = []
        objectposegts = []
        dispgts = []
        
        for (tstamp, image, objectmask,instance_mask, intrinsic, posegt, objectposegt, dispgt) in image_stream:
            tstamps.append(tstamp)
            images.append(image)
            intrinsics.append(intrinsic)
            objectmasks.append(objectmask)
            posegts.append(posegt)
            objectposegts.append(objectposegt)
            dispgts.append(dispgt)

            if len(tstamps) == 16:
                poses, Obposes = self.__fill(tstamps, images, intrinsics, objectmasks, posegts, objectposegts, dispgts)
                pose_list += poses
                Obpose_list += Obposes
                tstamps, images, intrinsics, objectmasks, posegts, objectposegts, dispgts= [], [], [], [], [], [], []

        if len(tstamps) > 0:
                poses, Obposes = self.__fill(tstamps, images, intrinsics, objectmasks, posegts, objectposegts, dispgts)
                pose_list += poses
                Obpose_list += Obposes

        # stitch pose segments together
        return lietorch.cat(pose_list, 0), lietorch.cat(Obpose_list, 0)

