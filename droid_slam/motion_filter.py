import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops 
from modules.corr import CorrBlock


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, obthresh = 2.5, device="cuda:0"):

        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.obthresh = obthresh
        self.device = device

        self.count = 0

        self.instance_mask = None

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[
            :, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[
            :, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, objectmask, instance_mask, posegt, objectposegt, dispgt, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        # Id = torch.as_tensor(
        #     [0.37732657, -271.64998866, -1.05244905, -0.00972557, 0.90465151, -0.00738998, 0.42597706], dtype=torch.float)
        Id = self.video.poses[0].squeeze() # initialization
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2, 1, 0]].to(self.device) / 255.0 #1,1,3,240,808
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            print('---add one keyframe to video---')
            #count number of pixels for each car
            arearank = torch.bincount(instance_mask.flatten())
            valid = arearank[arearank>0].shape[0]
            trackid = torch.argsort(arearank)[-valid:]
            trackid = trackid[trackid!=0]-1
            
            net, inp = self.__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap#([1, 128, 30, 101])
            self.video.append(
                tstamp, image[0], Id, None, depth, intrinsics / 8.0, gmap, net[0, 0], inp[0, 0], objectmask, posegt, objectposegt, dispgt, None)#第一帧net,inp只加了128个维度中的一个

            for id in trackid:
                self.video.objectgraph[int(id)] = [tstamp]

            self.instance_mask = instance_mask
        ### only add new frame if there is enough motion ###
        else:
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]
            corr = CorrBlock(self.fmap[None, [0]], gmap[None, [0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, _ = self.update(
                self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            flow_camera = delta[..., 0:2].norm(dim=-1).mean()

            #check dynamic flow
            flow_object = []
            for id in self.video.objectgraph.keys():
                flow_object.append(delta[0, ..., 0:2][self.instance_mask == id+1].norm(dim=-1).mean())

            #check new coming car
            arearank = torch.bincount(instance_mask.flatten())
            valid = arearank[arearank>0].shape[0]
            trackid = torch.argsort(arearank)[-valid:]
            trackid = trackid[trackid!=0]-1

            allexistid = torch.isin(trackid, torch.tensor(list(self.video.objectgraph.keys())))

            # if flow_camera > self.thresh or (torch.stack(flow_object)>self.obthresh).any() or not allexistid.all():#有足够相机流
            if flow_camera > self.thresh:#有足够相机流
                print('---add one keyframe to video---')
                self.count = 0#这个count查了关键帧之间间隔了多少帧
                net, inp = self.__context_encoder(inputs[:, [0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(
                    tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0], objectmask, posegt, objectposegt, dispgt, None)
                
                self.instance_mask = instance_mask
                for idx, id in enumerate(self.video.objectgraph):
                    if flow_object[idx]> self.obthresh:
                        self.video.objectgraph[id].append(tstamp)

                #add new car
                for id in trackid[allexistid == False]:
                    self.video.objectgraph[int(id)] = [tstamp]
                print('existing trackid is {}'.format(self.video.objectgraph.keys()))
            else:
                self.count += 1
