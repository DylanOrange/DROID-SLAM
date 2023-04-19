import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock, AltCorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3, SO3, Sim3
from geom.ba import BA, dynamicBA, cameraBA, midasBA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies
from geom.flow_vis_utils import flow_to_image

from torch_scatter import scatter_mean
import os
import pickle as pk


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

def upsample4(flow, disps = None):
    B, N, ht, wd, ch = flow.shape
    flow = flow.permute(0,1,4,2,3).reshape(B*N, ch, ht, wd)
    if not disps:
        upsampled_flow = 4*F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=True)
    else:
        upsampled_flow = F.interpolate(flow, scale_factor=4, mode='nearest')
    upsampled_flow = upsampled_flow.permute(0,2,3,1).reshape(B, N, 4*ht, 4*wd, ch)
    return upsampled_flow

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

        # self.upmask_disp = nn.Sequential(
        #     nn.Conv2d(128, 4*4*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))#14,128,30,101

        net = net.view(batch, num, 128, ht, wd)#1,14,128,30,101
        net = scatter_mean(net, ix, dim=1)#1,5,128,30,101

        # net = net.view(-1, 128, ht, wd)#14,128,30,101
        net = net.view(-1, 128, ht, wd)#5,128,30,101

        # less_size = net_less.shape[0]
        # net = torch.cat([net_less, net], dim = 0)
        net = self.relu(self.conv2(net))#19,128,30,101

        # net_less = net[0:less_size]
        # net_more = net[less_size:]

        eta = self.eta(net).view(batch, -1, ht, wd)
        # upmask_disp = self.upmask_disp(net).view(batch, -1, 4*4*9, ht, wd)
        # upmask_flow = self.upmask_flow(net_more).view(batch, -1, 4*4*9, ht, wd)#1,14,576,30,101

        return .01 * eta

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

        corr = self.corr_encoder(corr)#196->128
        flow = self.flow_encoder(flow)#4->64
        net = self.gru(net, inp, corr, flow)#128,128,128,64

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

        # if ii is not None:
        #     eta = self.agg(net, ii.to(net.device))
        #     return net, delta, weight, eta

        # else:
        return net, delta, weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()


    def extract_features(self, images, corners, recs):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps, fmaps_high = self.fnet(images, corners, recs)
        net, net_high = self.cnet(images, corners, recs)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        net_high, inp_high = net_high.split([128,128], dim=2)
        net_high = torch.tanh(net_high)
        inp_high = torch.relu(inp_high)

        return [fmaps, fmaps_high], [net, net_high], [inp, inp_high]


    def forward(self, Gs, Ps, ObjectGs, ObjectPs, images, objectmasks, highmasks, \
                disps, gtdisps, highgtdisps, intrinsics, trackinfo, depth_valid, high_depth_valid, save, total_steps, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """
        #Ps is ground truth
        corners = trackinfo['corner'][0]
        recs = trackinfo['rec'][0]

        ii, jj, _ = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        validmask = torch.ones_like(ii,dtype=torch.bool)[None]

        gtflow, gtmask = pops.dyprojective_transform(Ps, disps, intrinsics, ii, jj, validmask, ObjectPs, objectmasks)

        # depth_valid = depth_valid[:, ii, ..., None]
        fmaps, net_all, inp_all = self.extract_features(images, corners, recs)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)

        all_Gs_list, all_disp_list, all_ObGs_list, all_flow_list, all_static_residual_list = [], [], [], [], []

        # objectmasks_list, weight_list, all_weight_list, valid_list, intrinsics_list = [], [], [], [], []

        print('-----')
        print('before optimization')
        loss, r_err, t_err = geoloss(Ps, Gs, ii, jj)
        ob_loss, ob_r_err, ob_t_err = geoloss(ObjectPs, ObjectGs, ii, jj)

        print('geo loss is {}'.format(loss.item()))
        print('r_err is {}'.format(r_err.item()))
        print('t_err is {}'.format(t_err.item()))

        print('ob_loss is {}'.format(ob_loss.item()))
        print('ob_r_err is {}'.format(ob_r_err.item()))
        print('ob_t_err is {}'.format(ob_t_err.item()))
        print('-----')
	
        for index in range(2):
            coords1, _ = pops.dyprojective_transform(Gs, disps, intrinsics, ii, jj, validmask, ObjectGs, objectmasks)
            corr_fn = CorrBlock(fmaps[index][:,ii], fmaps[index][:,jj], num_levels=4, radius=3)

            target = coords1.clone()
            net, inp = net_all[index][:,ii], inp_all[index][:,ii]

            Gs_list, disp_list, ObGs_list, flow_list, static_residual_list = [], [], [], [], []

            for step in range(num_steps):
                Gs = Gs.detach()
                ObjectGs = ObjectGs.detach()
                disps = disps.detach()
                coords1 = coords1.detach()
                target = target.detach()

                # extract motion features
                corr = corr_fn(coords1)
                resd = target - coords1
                flow = coords1 - coords0

                motion = torch.cat([flow, resd], dim=-1)
                motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

                net, delta, weight = \
                    self.update(net, inp, corr, motion, ii, jj)

                print('predicted weight is {}'.format(weight.mean().item()))
                print('predicted dynamic weight is {}'.format(weight[objectmasks[:,ii]>0.5].mean().item()))
                print('predicted flow loss is {}'.format((gtflow - target).abs().mean().item()))
                target = coords1 + delta
                print('predicted flow delta is {}'.format(delta.mean().item()))
                # for i in range(2):
                #     Gs, ObjectGs, a, b, midasdisps = midasBA(gtflow, gtmask, ObjectGs, objectmasks, trackinfo, validmask, \
                #                                     eta, Gs, gtdisps, midasdisps, intrinsics, ii, jj, a, b, fixedp=2)
                for i in range(2):
                    Gs, ObjectGs = dynamicBA(target, weight, ObjectGs, objectmasks, trackinfo, validmask, \
                                                    None, Gs, disps, intrinsics, ii, jj, fixedp=2)
                # evaluate_depth(gtdisps, depth_valid, a*midasdisps+b)
                coords1, valid_static = pops.dyprojective_transform(Gs, disps, intrinsics, ii, jj, \
                                                                    validmask, ObjectGs, objectmasks)
                static_residual = (target - coords1)*valid_static

                Gs_list.append(Gs)
                ObGs_list.append(ObjectGs)
                disp_list.append(disps)
                static_residual_list.append(static_residual)
                flow_list.append(target)
                # weight_list.append(weight)
            
            print('-----')
            print('after optimization')
            loss, r_err, t_err = geoloss(Ps, Gs, ii, jj)
            ob_loss, ob_r_err, ob_t_err = geoloss(ObjectPs, ObjectGs, ii, jj)

            print('geo loss is {}'.format(loss.item()))
            print('r_err is {}'.format(r_err.item()))
            print('t_err is {}'.format(t_err.item()))

            print('ob_loss is {}'.format(ob_loss.item()))
            print('ob_r_err is {}'.format(ob_r_err.item()))
            print('ob_t_err is {}'.format(ob_t_err.item()))
            print('-----')

            # intrinsics_list.append(intrinsics)
            # all_weight_list.append(weight_list)
            # objectmasks_list.append(objectmasks)
            # valid_list.append(depth_valid)
            all_Gs_list.append(Gs_list)
            all_ObGs_list.append(ObGs_list)
            all_disp_list.append(disp_list)
            all_static_residual_list.append(static_residual_list)
            all_flow_list.append(flow_list)

            if index == 1:
                # if save:
                #     save_debug_result(all_Gs_list, all_ObGs_list, all_disp_list, all_flow_list, valid_list,\
                #                       objectmasks_list, all_weight_list, intrinsics_list, trackinfo, ii, jj, total_steps)
                # for i in range(disp_list[-1].shape[1]):
                #     write_depth(os.path.join('result/multiscale', str(trackinfo['frames'][0][i].item())+'pred_cropdepth'), \
                #             1/(disp_list[-1][0,i].detach().cpu().numpy()), False)
                break

            objectmasks = pops.crop(highmasks, corners[0], recs[0])

            # upsampled_disps = upsample_flow(disps.unsqueeze(-1), mask_disp, 4, True).squeeze(-1)
            # upsampled_disps = upsample4(disps.unsqueeze(-1), True).squeeze(-1)
            disps = pops.crop(highgtdisps, corners[0], recs[0])
            # gtdisps = pops.crop(highgtdisps, corners[0], recs[0])
            # disps = pops.crop(midasdisps, corners[0], recs[0])
            # disps = gtdisps
            # depth_valid = pops.crop(high_depth_valid, corners[0], recs[0])[:, ii, ..., None]

            # upsampled_flow = upsample_flow(coords1 - coords0, mask_flow, 4, False)
            # upsampled_flow = upsample4(coords1 - coords0, False)
            coords0 = pops.coords_grid(recs[0][0], recs[0][1], device=coords1.device)
            # coords1 = pops.crop(upsampled_flow, corners[0], recs[0]) + coords0

            intrinsics[...,:] *= 4
            intrinsics[..., 2] -= corners[0][1]
            intrinsics[..., 3] -= corners[0][0]
            gtflow, gtmask = pops.dyprojective_transform(Ps, disps, intrinsics, ii, jj, validmask, ObjectPs, objectmasks)

            # print('----------')
            # # gtflow = pops.crop(highgtflow, corners[0], recs[0]) + coords0
            # # gtmask = pops.crop(highmask, corners[0], recs[0])
            # gtcropdisps = pops.crop(highgtdisps, corners[0], recs[0])
            # maxdepth = (1.0/highgtdisps).max()
            # for i in range(upsampled_disps.shape[1]):
            #     write_depth(os.path.join('result/multiscale', str(trackinfo['frames'][0][i].item())+'_upsampled_depth'), \
            #                 ((1/upsampled_disps[0,i]).clamp(max = maxdepth).detach().cpu().numpy()), False)
            #     write_depth(os.path.join('result/multiscale', str(trackinfo['frames'][0][i].item())+'_gt_depth'), \
            #                 1/(gtcropdisps[0,i].detach().cpu().numpy()), False)
            #     write_depth(os.path.join('result/multiscale', str(trackinfo['frames'][0][i].item())+'_gt_highdepth'), \
            #                 1/(highgtdisps[0,i].detach().cpu().numpy()), False)

            # flow = (lowgtflow - coords0).squeeze(0)
            # upsampled_flow = 4*F.interpolate(flow.permute(0,3,1,2), scale_factor=4, mode='bilinear', align_corners=True).permute(0,2,3,1)
            # upsampled_coords1 = (upsampled_flow + highcoords0)[None]

            # for i in range(highgtflow.shape[1]):
            #     gtflow = flow_to_image(highgtflow[0,i].cpu().numpy(), highmask[0,i,...,0].cpu().numpy())
            #     cv2.imwrite('./result/multiscale/highgtflow_{}.png'.format(i),gtflow)

            #     upsampledflow = flow_to_image(upsampled_coords1[0,i].cpu().numpy(), highmask[0,i,...,0].cpu().numpy())
            #     cv2.imwrite('./result/multiscale/upsampledflow_{}.png'.format(i),upsampledflow)


            # crophighgtflow = pops.crop(highgtflow , corners[0], recs[0]) + coords0
            # cropgtflow, cropgtmask = pops.dyprojective_transform(Ps, disps, intrinsics, ii, jj, \
            #                                                      validmask, ObjectPs, objectmasks)

            # for i in range(cropgtflow.shape[1]):
            #     # highgtflow = flow_to_image(highgtflow[0,i].cpu().numpy(), highmask[0,i,...,0].cpu().numpy())
            #     # cv2.imwrite('./result/multiscale/highgtflow_{}.png'.format(i),highgtflow)
            #     gtflow = flow_to_image(crophighgtflow[0,i].cpu().numpy(), cropgtmask[0,i,...,0].cpu().numpy())
            #     cv2.imwrite('./result/multiscale/crophighgtflow_{}.png'.format(i),gtflow)
            #     cropflow = flow_to_image(cropgtflow[0,i].cpu().numpy(), cropgtmask[0,i,...,0].cpu().numpy())
            #     cv2.imwrite('./result/multiscale/cropgtflow_{}.png'.format(i),cropflow)

            #     cropsampledflow = flow_to_image(coords1[0,i].cpu().numpy(), cropgtmask[0,i,...,0].cpu().numpy())
            #     cv2.imwrite('./result/multiscale/cropupsampledflow_{}.png'.format(i),cropsampledflow)
            
            # # dynaflow2 = dynaflow.clone()
            # # dynaflow2[:,:2] = 0.0
            # diff_flow = (target - lowgtflow)
            # # diff_flow2 = (target - lowgtflow)[dynaflow2>0.0]

            # error_static_flow = torch.mean(torch.abs(diff_flow))
            # error_dyna_flow = torch.mean(torch.abs(diff_flow[dynaflow>0.0]))

            # diff_disps = (disps - gtdisps)
            # error_dyan_depth = torch.mean(torch.abs(diff_disps[objectmasks.long()>0.0]))
            # error_static_depth = torch.mean(torch.abs(diff_disps))

            # # for i in range(target.shape[1]):
            # #     error_flow = torch.mean(torch.abs(diff_flow[:,i][dynaflow[:,i]>0.0]))
            # #     print('error of dynamic flow' + str(i) + ' is {}'.format(error_flow))

            # print('overall depth is {}'.format(1.0/torch.mean(gtdisps[objectmasks>0.0])))

            # weight_vis = valid[...,0]*weight[...,0]
            # weight_vis[weight_vis>0.5] = 1.0
            # for i in range(target.shape[1]):
                
            #     if torch.count_nonzero(dynaflow[0,i]) != 0:
            #         gt_dyna_flow = flow_to_image(lowgtflow[0,i].detach().cpu().numpy(), dynaflow[0,i].detach().cpu().numpy())
            #         cv2.imwrite('./result/gt_dyna_flow/flow_gtdyna_'+ str(step) + '_' + str(i) +'.png', gt_dyna_flow)

            #         pred_dyna_flow = flow_to_image(target[0,i].detach().cpu().numpy(), dynaflow[0,i].detach().cpu().numpy())
            #         cv2.imwrite('./result/pred_dyna_flow/flow_preddyna_'+ str(step) + '_' + str(i) +'.png', pred_dyna_flow)

            #     if (weight_vis[0,i].max() == 1.0):
            #         target_flow = flow_to_image(target[0,i].detach().cpu().numpy(), weight_vis[0,i].detach().cpu().numpy())
            #         cv2.imwrite('./result/pred_flow/flow_pred_'+ str(step) + '_' + str(i) +'.png', target_flow)
            
            # if ob_r_err.item()>0.1:
            #     print('bad optimization!')
        
        return all_Gs_list, all_ObGs_list, all_disp_list, all_static_residual_list, all_flow_list

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

    # s = fit_scale(dP, dG)
    # dG = dG.scale(s[:,None])

    d = (dG * dP.inv()).log()

    tau, phi = d.split([3,3], dim=-1)
    geodesic_loss = tau.norm(dim=-1).mean() + phi.norm(dim=-1).mean()

    dE = Sim3(dG * dP.inv()).detach()
    r_err, t_err, s_err = pose_metrics(dE)
    r_err = r_err.mean()
    t_err = t_err.mean()

    return geodesic_loss, r_err, t_err

def depth_vis(disps, mode, max_depth):
    depth = 100*((((1.0/disps).clamp(max=max_depth)) * (655.35/max_depth)).cpu().detach().numpy())

    for i in range(disps.shape[1]):
        cv2.imwrite('./result/depth/depth' + mode +'_{}.png'.format(i),depth[0,i].astype(np.uint16))

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def evaluate_depth(gtdepth, mask, depth):
    low_gt = (gtdepth)[mask]
    low_pred = (depth)[mask]

    low_diff = low_gt - low_pred
    abs_low_diff = torch.abs(low_diff)
    squared_diff = low_diff*low_diff
    abs_low_error = torch.mean(abs_low_diff)

    re_low_error = torch.mean((abs_low_diff/low_gt))
    rmse_low = torch.sqrt(torch.mean(squared_diff))

    print('abs_error {}'.format(abs_low_error))
    print('rmse {}'.format(rmse_low))

def save_debug_result(all_Gs_list, all_ObGs_list, all_disp_list, all_flow_list, valid_list,\
                    objectmasks_list, all_weight_list, intrinsics_list, trackinfo, ii, jj, step):
    
    debug_file = {
        'lgs':all_Gs_list[0][-1].detach().cpu(),
        'hgs':all_Gs_list[1][-1].detach().cpu(),
        'logs':all_ObGs_list[0][-1].detach().cpu(),
        'hogs':all_ObGs_list[1][-1].detach().cpu(),
        'ldisps':all_disp_list[0][-1].detach().cpu(),
        'hdisps':all_disp_list[1][-1].detach().cpu(),
        'lflow':all_flow_list[0][-1].detach().cpu(),
        'hflow':all_flow_list[1][-1].detach().cpu(),
        'lmask':objectmasks_list[0].detach().cpu(),
        'hmask':objectmasks_list[1].detach().cpu(),
        'lweight':all_weight_list[0][-1].detach().cpu(),
        'hweight':all_weight_list[1][-1].detach().cpu(),
        'lintrinsics':intrinsics_list[0].detach().cpu(),
        'hintrinsics':intrinsics_list[1].detach().cpu(),
        'ii':ii.detach().cpu(),
        'jj':jj.detach().cpu(),
        'frames':trackinfo['frames'][0].cpu(),
        'lvalid':valid_list[0].detach().cpu(),
        'hvalid':valid_list[1].detach().cpu()
        }
    
    with open(os.path.join('result/debug/wockwoweight', 'debug_'+str(step)+'.pkl'), 'wb') as tt:
                pk.dump(debug_file,tt)
