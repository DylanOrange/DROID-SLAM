import cv2
from matplotlib.transforms import BboxBase
import torch
from lietorch import SE3
import numpy as np
import random
from collections import OrderedDict
from droid_net import DroidNet
from segmenter import Segmenter
from torchvision.ops import box_iou
import geom.projective_ops as pops 
from modules.corr import CorrBlock
from torch.cuda.amp import autocast as autocast


COLOR = np.random.uniform(0, 255, size=(200, 3))

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
        self.bbox = None

        self.carcount = 0
        self.trackid  = None

        self.iou_threshold = 0.5

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[
            :, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[
            :, None, None]

        self.segmenter = Segmenter(self.MEAN, self.STDV)

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image, None, None).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image, None, None).squeeze(0)

    # @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, objectmask, instance_mask, posegt, objectposegt, dispgt, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image.to(self.device) #1,1,3,240,808

        masks, bbox = self.segmenter.instance_segmentation_api(inputs[0], tstamp)

        inputs = inputs[None, :, [2, 1, 0]] / 255.0 #1,1,3,240,808
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        with autocast():
        # extract features
            gmap = self.__feature_encoder(inputs)

            ### always add first frame to the depth video ###
            if self.video.counter.value == 0:
                print('---add one keyframe to video---')           

                self.trackid = [0,1,3]
                if len(bbox) !=  0:
                    self.bbox = bbox
                    self.carcount = len(bbox)
                    trackid  = torch.arange(self.carcount)
                    self.trackid = trackid
                
                net, inp = self.__context_encoder(inputs[:, [0]])
                self.net, self.inp, self.fmap = net, inp, gmap#([1, 128, 30, 101])
                self.video.append(
                    tstamp, image[0], SE3(posegt).inv().data[0], None, 1.0/dispgt, intrinsics / 8.0, gmap, net[0, 0], inp[0, 0], objectmask, posegt, objectposegt, dispgt, None)#第一帧net,inp只加了128个维度中的一个

                for id in self.trackid:
                    self.video.objectgraph[int(id)] = [tstamp]

                self.instance_mask = instance_mask

                img = image[0].permute(1,2,0).numpy()
                vis_bbox = bbox.cpu().numpy()
                vis_mask = masks.cpu().numpy()
                for i, id in enumerate(trackid):
                    rgb_mask, color = self.random_colour_masks(vis_mask[i], id.cpu().numpy())
                    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                    cv2.rectangle(img, (vis_bbox[i][0], vis_bbox[i][1]) , (vis_bbox[i][2], vis_bbox[i][3]) ,color=color, thickness=1)
                    cv2.putText(img,'car_'+str(int(id)), (vis_bbox[i][0], vis_bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,thickness=1, lineType=cv2.LINE_AA)
                cv2.imwrite('./result/segmentation/'+str(tstamp)+'.png',img)

            ### only add new frame if there is enough motion ###
            else:
                # index correlation volume
                coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]
                corr = CorrBlock(self.fmap[None, [0]], gmap[None, [0]])(coords0)

                # approximate flow magnitude using 1 update iteration
                _, delta, _ = self.update(self.net[None], self.inp[None], corr)

                # check motion magnitue / add new frame to video
                flow_camera = delta[..., 0:2].norm(dim=-1).mean()

                if len(bbox)!=0:
                    if self.bbox is not None:
                        iou = box_iou(bbox, self.bbox)
                        #如果最大的iou>0.5，判断为同一个，加入trackid list, <0.7的在trackid 后面加新的id
                        corr = torch.amax(iou, dim = 1)
                        corrid = torch.argmax(iou[corr>=self.iou_threshold], dim = 1)
                        trackid = torch.cat((self.trackid[corrid], torch.arange(self.carcount, self.carcount + len(iou[corr<self.iou_threshold]))))
                        self.carcount += len(iou[corr<self.iou_threshold])
                        self.bbox = bbox
                        self.trackid = trackid
                    else:
                        self.bbox = bbox
                        self.carcount = len(bbox)
                        trackid  = torch.arange(self.carcount)
                        self.trackid = trackid

                # #visualize data association
                # img = image[0].permute(1,2,0).numpy()
                # vis_bbox = bbox.cpu().numpy()
                # vis_mask = masks.cpu().numpy()
                # for i, id in enumerate(trackid):
                #     rgb_mask, color = self.random_colour_masks(vis_mask[i], id.cpu().numpy())
                #     img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                #     cv2.rectangle(img, (vis_bbox[i][0], vis_bbox[i][1]) , (vis_bbox[i][2], vis_bbox[i][3]) ,color=color, thickness=1)
                #     cv2.putText(img,'car_'+str(int(id)), (vis_bbox[i][0], vis_bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,thickness=1, lineType=cv2.LINE_AA)
                # cv2.imwrite('./result/segmentation/'+str(tstamp)+'.png',img)

                #check dynamic flow
                flow_object = []
                for id in self.video.objectgraph.keys():
                    flow_object.append(delta[0, ..., 0:2][self.instance_mask == id+1].norm(dim=-1).mean())

                allexistid = torch.isin(trackid, torch.as_tensor(list(self.video.objectgraph.keys())))

                # for id in trackid[allexistid == False]:
                #     self.video.objectgraph[int(id)] = [tstamp]
                # print('existing trackid is {}'.format(self.video.objectgraph.keys()))

                # if flow_camera > self.thresh or (torch.stack(flow_object)>self.obthresh).any() or not allexistid.all():#有足够相机流
                if flow_camera > self.thresh:#有足够相机流
                    print('---add one keyframe to video---')
                    self.count = 0#这个count查了关键帧之间间隔了多少帧
                    net, inp = self.__context_encoder(inputs[:, [0]])
                    self.net, self.inp, self.fmap = net, inp, gmap
                    self.video.append(
                        tstamp, image[0], None, None, 1/dispgt, intrinsics / 8.0, gmap, net[0], inp[0], objectmask, posegt, objectposegt, dispgt, None)
                    
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

        # print(list(self.video.objectgraph.keys()))

    @staticmethod
    def random_colour_masks(image, id):
        # COLOR = np.random.uniform(0, 255, size=(100, 3))
        # colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        color = COLOR[id]
        r[image == 1], g[image == 1], b[image == 1] = color
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask, color
