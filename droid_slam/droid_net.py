import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3, SO3, Sim3
from geom.ba import BA, dynamicBA, fulldynamicBA, cameraBA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies
from geom.flow_vis_utils import flow_to_image

from torch_scatter import scatter_mean


def cvx_upsample(data, mask, time, disps = None):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, time, time, ht, wd)
    mask = torch.softmax(mask, dim=2)

    #NOTE: raft这里data先乘了8
    if disps:
        up_data = F.unfold(data, [3,3], padding=1)
    else:
        up_data = F.unfold(time*data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, time*ht, time*wd, dim)

    return up_data

def upsample_flow(flow, mask, time, disps = None):
    batch, num, ht, wd, dim = flow.shape
    flow = flow.view(batch*num, ht, wd, dim)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(flow, mask, time, disps).view(batch, num, time*ht, time*wd, dim)

def upsample4(flow, mode = 'nearest'):
    B, N, ht, wd, ch = flow.shape
    flow = flow.permute(0,1,4,2,3).reshape(B*N, ch, ht, wd)
    upsampled_flow = 4*F.interpolate(flow, scale_factor=4, mode=mode)
    upsampled_flow = upsampled_flow.permute(0,2,3,1).reshape(B, N, 4*ht, 4*wd, ch)
    return  upsampled_flow

class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        # self.upmask_flow = nn.Sequential(
        #     nn.Conv2d(128, 4*4*9, 1, padding=0))

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))#14,128,30,101

        net = net.view(batch, num, 128, ht, wd)#1,14,128,30,101
        net_less = scatter_mean(net, ix, dim=1)#1,5,128,30,101

        # net = net.view(-1, 128, ht, wd)#14,128,30,101
        net_less = net_less.view(-1, 128, ht, wd)#5,128,30,101

        # less_size = net_less.shape[0]
        # net = torch.cat([net_less, net], dim = 0)
        net = self.relu(self.conv2(net_less))#19,128,30,101

        # net_less = net[0:less_size]
        # net_more = net[less_size:]

        eta = self.eta(net_less).view(batch, -1, ht, wd)
        upmask_disp = self.upmask(net_less).view(batch, -1, 8*8*9, ht, wd)
        # upmask_flow = self.upmask_flow(net_more).view(batch, -1, 4*4*9, ht, wd)#1,14,576,30,101

        return .01 * eta, upmask_disp

class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        # self.dyweight = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 2, 3, padding=1),
        #     GradientClip(),
        #     nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

        # self.mask_flow = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 4*4*9, 1, padding=0))

        # self.mask_weight = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 4*4*9, 1, padding=0))

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)
        # dyweight = self.dyweight(net).view(*output_dim)

        # mask_flow = .25*self.mask_flow(net).view(*output_dim)
        # mask_weight = .25*self.mask_weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()
        # dyweight = dyweight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()


    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(self, Gs, Ps, ObjectGs, ObjectPs, images, objectmasks, disps, gtdisps, intrinsics, trackinfo, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """
        #Ps is ground truth
        objectmasks = objectmasks[0]
        # corners = trackinfo['corner'][0]
        # rec = trackinfo['rec'][0]
        # cropmasks = cropmasks[0]
        # cropdisps = cropdisps[0]
        # fullmasks = fullmasks[0]

        # B = objectmasks.shape[0]

        ii, jj, _ = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        # coords_crop = pops.coords_grid(ht//2, wd//2, device=images.device)
        
        validmasklist = []
        for n in range(len(trackinfo['trackid'][0])):
            validmasklist.append(torch.isin(ii, trackinfo['apperance'][n][0]) & torch.isin(jj, trackinfo['apperance'][n][0]))
        validmask = torch.stack(validmasklist, dim=0)

        #NOTE: 先用低分辨率试一试
        # coords1, _ = pops.dyprojective_transform(Gs, disps, intrinsics, ii, jj, validmask, ObjectGs, objectmasks)
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)

        target = coords1.clone()

        # highintrinsics = intrinsics.clone()
        # highintrinsics[...,:] *= 4

        lowgtflow, lowmask = pops.projective_transform(Ps, gtdisps, intrinsics, ii, jj)
        # lowgtflow, lowmask = pops.dyprojective_transform(Ps, gtdisps, intrinsics, ii, jj, validmask, ObjectPs, objectmasks)
        # highgtflow, highmask = pops.dyprojective_transform(Ps, fulldisps, highintrinsics, ii, jj, validmask, ObjectPs, fullmasks)

        Gs_list, disp_list, ObjectGs_list, flow_low_list, static_residual_list, dyna_residual_list = [], [], [], [], [], []

        for step in range(num_steps):
            Gs = Gs.detach()
            ObjectGs = ObjectGs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features
            corr = corr_fn(coords1.float())
            resd = target - coords1
            flow = coords1 - coords0

            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0).float()

            net, delta, weight, eta, mask_disp = \
                self.update(net, inp, corr, motion, ii, jj)

            target = coords1 + delta

            # flow_inter = upsample_flow(target - coords0, mask_flow, 4) + coords_crop
            # cropweight = upsample_flow(weight, mask_weight,4)

            # flow_inter = highgtflow.clone()
            # flow_inter = torch.normal(mean=highgtflow, std=1e+0)

            # vis_highmask = highmask.squeeze(-1)
            # vis_fullmask = fullmasks[:, ii]
            # flow_inter_vis = flow_to_image(flow_inter[0,0].cpu().numpy(), vis_highmask[0,0].cpu().numpy())
            # dyna_flow_inter_vis = flow_to_image(flow_inter[0,0].cpu().numpy(), vis_fullmask[0,0].cpu().numpy())
            # cv2.imwrite('flow_inter_{}.png'.format(0),flow_inter_vis)
            # cv2.imwrite('dyna_flow_inter_{}.png'.format(0),dyna_flow_inter_vis)

            # cropflow = pops.crop(flow_inter.expand(B,-1,-1,-1,-1), corners, rec)
            # cropweight = pops.crop(cropweight.expand(B,-1,-1,-1, -1), corners, rec)
            # cropweight = pops.crop(cropweight.expand(B,-1,-1,-1, -1), corners, rec)

            # lowmask = lowmask.expand(-1,-1,-1,-1,2)

            # upsampled_disps = upsample_flow(disps[..., None], mask_disp, 4, True)
            # cropdisps = pops.crop(upsampled_disps.expand(B,-1,-1,-1, -1), corners, rec)[..., 0]

            weight = lowmask.expand(-1,-1,-1,-1,2)
            eta = torch.zeros_like(eta)
            for i in range(4):
                Gs, disps = BA(lowgtflow, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)
                # Gs, ObjectGs, disps = dynamicBA(lowgtflow, weight, ObjectGs, objectmasks, trackinfo, validmask, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            # coords1, valid_static = pops.dyprojective_transform(Gs, disps, intrinsics, ii, jj, validmask, ObjectGs, objectmasks)
            coords1, valid_static = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            # coords_resi, valid_dyna = pops.dyprojective_transform(Gs, cropdisps, highintrinsics, ii, jj, validmask, ObjectGs, cropmasks, batch = True, batch_grid = trackinfo['grid'])
        
            # residual = (cropflow - coords_resi)*cropmasks[:,ii, ..., None]*valid
            static_residual = (target - coords1)*valid_static
            dyna_residual = (target - coords1)*objectmasks[:,ii, ..., None]
            
            # loss, r_err, t_err = geoloss(Ps, Gs, ii, jj)
            # ob_loss, ob_r_err, ob_t_err = geoloss(ObjectPs, ObjectGs, ii, jj)
            # print('-----------------')
            # print('frames are {}'.format(trackinfo['frames']))
            # print('trackid is {}'.format(trackinfo['trackid'].item()))
            # print('loss is {}'.format(loss.item()))
            # print('r_err is {}'.format(r_err.item()))
            # print('t_err is {}'.format(t_err.item()))

            # print('ob_loss is {}'.format(ob_loss.item()))
            # print('ob_r_err is {}'.format(ob_r_err.item()))
            # print('ob_t_err is {}'.format(ob_t_err.item()))

            # if ob_loss > 2.0:
            #     print('{} has too large loss!'.format(trackinfo['trackid'].item()))
            #     print('frames are {}'.format(trackinfo['frames']))
            #     print('its ob_loss is {}'.format(ob_loss.item()))
            #     print('its r_error is {}'.format(ob_r_err.item()))
            #     print('its t_err is {}'.format(ob_t_err.item()))

            # print('-----------------')
            Gs_list.append(Gs)
            ObjectGs_list.append(ObjectGs)
            disp_list.append(upsample_flow(disps[..., None], mask_disp, 8, True)[...,0])
            static_residual_list.append(static_residual)
            dyna_residual_list.append(dyna_residual[dyna_residual>0])
            flow_low_list.append(target)

        return Gs_list, ObjectGs_list, disp_list, static_residual_list, dyna_residual_list, flow_low_list

def add_neighborhood_factors(t0, t1, r=2):
    """ add edges between neighboring frames within radius r """

    ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1))
    ii = ii.reshape(-1).to(dtype=torch.long)
    jj = jj.reshape(-1).to(dtype=torch.long)

    keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
    return ii[keep], jj[keep]

def flow_error(gt, pred, val = None):

    if val is not None:
        epe = val * (pred - gt).norm(dim=-1)
        epe = epe.reshape(-1)[val.reshape(-1) > 0.5]
    else:
        epe = (pred - gt).norm(dim=-1)
        epe = epe.reshape(-1)

    f_error = epe.mean()
    onepx = (epe<1.0).float().mean()

    return f_error, onepx

def flow_metrics(lowerror_list, higherror_list, dynamicerror_list, lowerrorpx_list, higherrorpx_list, dynamicerrorpx_list):

    lowerror = torch.mean(torch.stack(lowerror_list))
    lowerrorpx = torch.mean(torch.stack(lowerrorpx_list))
    higherror = torch.mean(torch.stack(higherror_list))
    higherrorpx = torch.mean(torch.stack(higherrorpx_list))
    dynamicerror = torch.mean(torch.stack(dynamicerror_list))
    dynamicerrorpx = torch.mean(torch.stack(dynamicerrorpx_list))

    metrics = {
                'lowerror':lowerror,
                'lowerrorpx':lowerrorpx,
                'higherror':higherror,
                'higherrorpx':higherrorpx,
                'dynamicerror':dynamicerror,
                'dynamicerrorpx':dynamicerrorpx,
    }
    return metrics

def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s

def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err

def geoloss(objectposes, object_est, ii, jj):
    dP = objectposes[:,jj] * objectposes[:,ii].inv()
    dG = object_est[:,jj] * object_est[:,ii].inv()

    d = (dG * dP.inv()).log()

    tau, phi = d.split([3,3], dim=-1)
    geodesic_loss = tau.norm(dim=-1).mean() + phi.norm(dim=-1).mean()

    # s = fit_scale(dP, dG)
    # dG = dG.scale(s[:,None])

    dE = Sim3(dG * dP.inv()).detach()
    r_err, t_err, s_err = pose_metrics(dE)
    r_err = r_err.mean()
    t_err = t_err.mean()

    return geodesic_loss, r_err, t_err