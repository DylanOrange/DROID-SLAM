from cmath import nan
from collections import OrderedDict
from re import S
import numpy as np
import torch
import cv2
from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list
from .projective_ops import projective_transform, dyprojective_transform, crop, coords_grid


def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err


def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s


def geodesic_loss(Ps, poses, graph, gamma=0.9, do_scale=True, object = False):
    """ Loss function for training network """
    
    ii, jj, kk = graph_to_edge_list(graph)
    
    dP = Ps[:,jj] * Ps[:,ii].inv()
    # dP = dP[None]

    metrics_list, geoloss_list = [], []
    for index in range(2):
        Gs = poses[index]

        n = len(Gs)
        geodesic_loss = 0.0
        metrics = {}

        for i in range(n):
            # Gs[i] = Gs[i].inv()
            w = gamma ** (n - i - 1)
            dG = Gs[i][:,jj] * Gs[i][:,ii].inv()
            # if object == True:
            #     dG = dG[None]

            if do_scale:
                s = fit_scale(dP, dG)
                dG = dG.scale(s[:,None])
            
            # pose error
            d = (dG * dP.inv()).log()

            if isinstance(dG, SE3):
                tau, phi = d.split([3,3], dim=-1)
                geodesic_loss += w * (
                    tau.norm(dim=-1).mean() + 
                    phi.norm(dim=-1).mean())

            elif isinstance(dG, Sim3):
                tau, phi, sig = d.split([3,3,1], dim=-1)
                geodesic_loss += w * (
                    tau.norm(dim=-1).mean() + 
                    phi.norm(dim=-1).mean() + 
                    0.05 * sig.norm(dim=-1).mean())
                
            dE = Sim3(dG * dP.inv()).detach()
            r_err, t_err, s_err = pose_metrics(dE)

        geoloss_list.append(geodesic_loss)

        if object == False:
            metrics = {
                'rot_error': r_err.mean().item(),
                'tr_error': t_err.mean().item(),
                'bad_rot': (r_err < .1).float().mean().item(),
                'bad_tr': (t_err < .01).float().mean().item(),
            } 
        else:
            metrics = {
                'ob_rot_error': r_err.mean().item(),
                'ob_tr_error': t_err.mean().item(),
                'ob_bad_rot': (r_err < .1).float().mean().item(),
                'ob_bad_tr': (t_err < .01).float().mean().item(),
            }
        if index == 1:
            for key_old, value in list(metrics.items()):
                metrics['high_'+key_old] = metrics.pop(key_old) 

        metrics_list.append(metrics)

    return geoloss_list, metrics_list

def smooth_loss(Gs, gamma=0.9):
    """ Loss function for training network """

    n = len(Gs)
    sum_smooth_loss = 0.0
    N = Gs[0].shape[1]
    for i in range(n):
        smooth_loss = 0.0
        dG = []
        w = gamma ** (n - i - 1)
        for j in range(N-1):
            dG.append(Gs[i][:,j] * Gs[i][:,j+1].inv())
        for k in range(N-2):
            tau, phi = (dG[k]*dG[k+1].inv()).log().split([3,3], dim=-1)
            smooth_loss += w * (tau.norm(dim=-1) + phi.norm(dim=-1))
        sum_smooth_loss += smooth_loss/(N-2)

    metrics = {'smooth_loss': sum_smooth_loss.item()} 
    return sum_smooth_loss, metrics


def residual_loss(residuals, gamma=0.9):
    """ loss on system residuals """

    residual_loss_list, metrics_list = [], []
    for index in range(2):
        residual_loss = 0.0
        metrics = {}
        residual = residuals[index]

        n = len(residual)

        for i in range(n):
            w = gamma ** (n - i - 1)
            residual_loss += w * residual[i].abs().mean()

        metrics = {'residual': residual_loss.item()}

        if index == 1:
            for key_old, value in list(metrics.items()):
                metrics['high_'+key_old] = metrics.pop(key_old)  

        residual_loss_list.append(residual_loss)
        metrics_list.append(metrics)

    return residual_loss_list, metrics_list


def flow_loss(Ps, disps, highdisps, poses_est, disps_est, ObjectPs, objectposes_est, \
              objectmasks, highobjectmask, trackinfo, intrinsics, graph, flow_list, scale, gamma=0.9):
    """ optical flow loss """

    ii, jj, kk = graph_to_edge_list(graph)

    validmask = torch.ones_like(ii, dtype=torch.bool, device='cuda')[None]

    highintrinsics = intrinsics.clone()
    highintrinsics[...,:] *= 4

    corner = trackinfo['corner'][0][0]
    rec = trackinfo['rec'][0][0]

    cropintrinsics = highintrinsics.clone()
    cropintrinsics[..., 2] -= corner[1]
    cropintrinsics[..., 3] -= corner[0]

    lowgtflow, lowmask = dyprojective_transform(Ps, disps, intrinsics, ii, jj, validmask, ObjectPs, objectmasks)
    highgtflow, highmask = dyprojective_transform(Ps, highdisps, highintrinsics, ii, jj, validmask, ObjectPs, highobjectmask)

    lowmask = lowmask * (disps[:,ii] > 0).float().unsqueeze(dim=-1)
    highmask = highmask * (highdisps[:,ii] > 0).float().unsqueeze(dim=-1)

    highcoords0 = coords_grid(highdisps.shape[2], highdisps.shape[3], device=highdisps.device)
    highgtflow = crop(highgtflow - highcoords0, corner, rec) + coords_grid(rec[0], rec[1], device=highgtflow.device)
    highmask = crop(highmask, corner, rec)
    highobjectmask = crop(highobjectmask, corner, rec)
    highdisps = crop(highdisps, corner, rec)

    n = len(poses_est[0])

    error_lowflow = 0
    error_dylow = 0
    error_induced_low = 0
    error_lowdepth = 0

    error_highflow = 0
    error_dyhigh = 0
    error_induced_high = 0
    error_highdepth = 0

    s_disps = highdisps*scale
    s_lowdisps = disps* scale

    for i in range(n):
        w = gamma ** (n - i - 1)

        #low resolution flow
        i_error_low = lowmask*(lowgtflow - flow_list[0][i]).abs()
        error_lowflow += w*(i_error_low.mean())

        #low resolution dyna flow
        i_error_dylow = i_error_low[objectmasks[:,ii]>0.5]
        error_dylow += w * i_error_dylow.mean()

        #low resolution pose and depth
        flow_low_induced, lowmask_induced = dyprojective_transform(poses_est[0][i], disps_est[0][i], \
                                                                   intrinsics, ii, jj, validmask, objectposes_est[0][i], objectmasks)

        v = (lowmask_induced * lowmask).squeeze(dim=-1)
        i_error_induced_low = v * (lowgtflow - flow_low_induced).norm(dim=-1)
        error_induced_low += w * i_error_induced_low.mean()

        #low resolution absolute depth
        diff_disp = torch.abs(s_lowdisps - disps_est[0][i]*scale)
        i_error_lowdepth = torch.mean((diff_disp)[s_lowdisps>0])
        error_lowdepth += w*i_error_lowdepth

        # #low resolution dyna depth
        # diff_dyna = torch.abs(s_lowdisps - disps_est[0][i]*scale)
        # i_error_lowdepth = torch.mean((diff_dyna/s_lowdisps)[objectmasks>0])
        # error_lowdepth += w*i_error_lowdepth

        #high resolution flow
        i_error_high = highmask*(highgtflow - flow_list[1][i]).abs()
        error_highflow += w*(i_error_high.mean())

        #high resolution dyna flow
        i_error_dyhigh = i_error_high[highobjectmask[:,ii]>0.5]
        error_dyhigh += w * i_error_dyhigh.mean()

        #high resolution pose and depth
        flow_high_induced, highmask_induced = dyprojective_transform(poses_est[1][i], disps_est[1][i], \
                                                                    cropintrinsics, ii, jj, validmask, objectposes_est[1][i], highobjectmask)

        v = (highmask_induced * highmask).squeeze(dim=-1)
        i_error_induced_high = v * (highgtflow - flow_high_induced).norm(dim=-1)
        error_induced_high += w * i_error_induced_high.mean()

        #high resolution absolute depth
        diff_disp = torch.abs(s_disps - disps_est[1][i]*scale)
        i_error_highdepth = torch.mean((diff_disp)[s_disps>0])
        error_highdepth += w*i_error_highdepth

        # #high resolution dyna depth
        # diff_dyna = torch.abs(s_disps - disps_est[1][i]*scale)
        # i_error_highdepth = torch.mean((diff_dyna/s_disps)[highobjectmask>0])
        # error_highdepth += w*i_error_highdepth

    #depth evaluation
    
    #depth visualization
    # gthighdepth = 1.0/(disps*scale)
    # gthighdepth = 100*gthighdepth.cpu().numpy()
    # gtlowdepth = 1.0/(lowdisps*scale)
    # gtlowdepth = 100*gtlowdepth.cpu().numpy()

    # prehighdepth = 1.0/(disps_est[-1]*scale)
    # prehighdepth = 100*prehighdepth.clamp(max=655.35).cpu().detach().numpy()
    # prelowdepth = 1.0/(low_dispest[-1]*scale)
    # prelowdepth = 100*prelowdepth.clamp(max=655.35).cpu().detach().numpy()

    # for i in range(5):
    #     cv2.imwrite('./result/objectflow/gthighdepth_{}.png'.format(i),gthighdepth[0,i].astype(np.uint16))
    #     cv2.imwrite('./result/objectflow/gtlowdepth_{}.png'.format(i),gtlowdepth[0,i].astype(np.uint16))
    #     cv2.imwrite('./result/objectflow/prehighdepth_{}.png'.format(i),prehighdepth[0,i].astype(np.uint16))
    #     cv2.imwrite('./result/objectflow/prelowdepth_{}.png'.format(i),prelowdepth[0,i].astype(np.uint16))
    
    #depth evaluation for the last optimization
    s_disps_est = disps_est[1][-1]* scale
    s_low_dispest = disps_est[0][-1] * scale

    valid_high = (1.0/s_disps < 30.0)*(1.0/s_disps > 0.2)
    high_gt = (s_disps)[valid_high]
    high_pred = (s_disps_est)[valid_high]

    high_diff = high_gt - high_pred
    abs_high_diff = torch.abs(high_diff)
    squared_diff = high_diff*high_diff
    abs_high_error = torch.mean(abs_high_diff)
    abs_high_dyna_error = torch.mean(torch.abs(s_disps - s_disps_est)[highobjectmask>0])

    re_high_error = torch.mean((abs_high_diff/high_gt))
    rmse_high = torch.sqrt(torch.mean(squared_diff))
    
    valid_low = (1.0/s_lowdisps < 30.0)*(1.0/s_lowdisps > 0.2)
    low_gt = (s_lowdisps)[valid_low]
    low_pred = (s_low_dispest)[valid_low]

    low_diff = low_gt - low_pred
    abs_low_diff = torch.abs(low_diff)
    squared_diff = low_diff*low_diff
    abs_low_error = torch.mean(abs_low_diff)
    abs_low_dyna_error = torch.mean(torch.abs(s_lowdisps - s_low_dispest)[objectmasks>0])

    re_low_error = torch.mean((abs_low_diff/low_gt))
    rmse_low = torch.sqrt(torch.mean(squared_diff))

    epe_low = i_error_low[lowmask[..., 0] > 0.5]
    epe_high = i_error_high[highmask[..., 0] > 0.5]

    metrics = {
        'low_f_error': epe_low.mean().item(),
        'low_1px': (epe_low<1.0).float().mean().item(),

        'low_dyna_f_error': i_error_dylow.mean().item(),
        'low_dyna_1px': (i_error_dylow<1.0).float().mean().item(),

        'high_f_error': epe_high.mean().item(),
        'high_1px': (epe_high<1.0).float().mean().item(),

        'high_dyna_f_error': i_error_dyhigh.mean().item(),
        'high_dyna_1px': (i_error_dyhigh<1.0).float().mean().item(),

        'low depth RMSE': rmse_low.item(),
        'high depth RMSE': rmse_high.item(),

        'abs_high_error':abs_high_error.item(),
        'abs_low_error':abs_low_error.item(),

        'abs_low_dyna_error': abs_low_dyna_error.item(),
        'abs_high_dyna_error': abs_high_dyna_error.item(),

        're_high_error': re_high_error.item(),
        're_low_error':re_low_error.item(),

        'abs_depth_high_error': i_error_highdepth.item(),
        'abs_depth_low_error':i_error_lowdepth.item(),

        # 're_dyna_high_error': i_error_highdepth.item(),
        # 're_dyna_low_error':i_error_lowdepth.item(),

    }
    return error_lowflow, error_dylow, error_induced_low, error_lowdepth, \
            error_highflow, error_dyhigh, error_induced_high, error_highdepth, \
            metrics