from cmath import nan
from collections import OrderedDict
from re import S
import numpy as np
import torch
import cv2
from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list
from .projective_ops import projective_transform, dyprojective_transform


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


def geodesic_loss(Ps, Gs, graph, gamma=0.9, do_scale=True, object = False, trackinfo = None):
    """ Loss function for training network """

    # relative pose
    # Ps = Ps.inv()
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Ps[:,jj] * Ps[:,ii].inv()

    if object == True:
        validmasklist = []
        for n in range(len(trackinfo['trackid'][0])):
            validmasklist.append(torch.isin(ii.to('cuda'), trackinfo['apperance'][n][0]) & torch.isin(jj.to('cuda'), trackinfo['apperance'][n][0]))
        validmask = torch.stack(validmasklist, dim=0)
        dP = dP[validmask]

    n = len(Gs)
    geodesic_loss = 0.0

    for i in range(n):
        # Gs[i] = Gs[i].inv()
        w = gamma ** (n - i - 1)
        dG = Gs[i][:,jj] * Gs[i][:,ii].inv()
        if object == True:
            dG = dG[validmask]

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
        # if geodesic_loss>0.5:
        #     print(trackinfo['trackid'])
        #     print(trackinfo['frames'])
        #     # print(Ps.data)
        #     # print(Gs[0].data)

    return geodesic_loss, metrics


def residual_loss(residuals, gamma=0.9):
    """ loss on system residuals """
    residual_loss = 0.0
    n = len(residuals)

    for i in range(n):
        w = gamma ** (n - i - 1)
        residual_loss += w * residuals[i].abs().mean()

    return residual_loss, {'residual': residual_loss.item()}


# def flow_loss(Ps, disps, poses_est, disps_est, ObjectPs, objectposes_est, objectmasks, cropmasks, cropdisps, fullmasks, fulldisps, quanmask, trackinfo, intrinsics, graph, flow_low_list, flow_high_list, gamma=0.9):
#     """ optical flow loss """

#     # graph = OrderedDict()
#     # for i in range(N):
#     #     graph[i] = [j for j in range(N) if abs(i-j)==1]

#     ii, jj, kk = graph_to_edge_list(graph)

#     validmasklist = []
#     for n in range(len(trackinfo['trackid'][0])):
#         validmasklist.append(torch.isin(ii.to('cuda'), trackinfo['apperance'][n][0]) & torch.isin(jj.to('cuda'), trackinfo['apperance'][n][0]))
#     validmask = torch.stack(validmasklist, dim=0)

#     # coords0, val0 = projective_transform(Ps, disps, intrinsics, ii, jj)
#     highintrinsics = intrinsics.clone()
#     highintrinsics[...,:] *= 4

#     # quanintrinsics = intrinsics.clone()
#     # quanintrinsics[...,:] *= 8

#     lowdisps = disps[:,:,3::8,3::8]
#     lowgtflow, lowmask = dyprojective_transform(Ps, lowdisps, intrinsics, ii, jj, validmask, ObjectPs, objectmasks[0])
#     highgtflow, highmask = dyprojective_transform(Ps, fulldisps, highintrinsics, ii, jj, validmask, ObjectPs, fullmasks[0])

#     highmask = highmask * (fulldisps[:,ii] > 0).float().unsqueeze(dim=-1)
#     lowmask = lowmask * (lowdisps[:,ii] > 0).float().unsqueeze(dim=-1)
#     # dyflow, dynamask0 = dyprojective_transform(Ps, disps, quanintrinsics, ii, jj, validmask, ObjectPs, quanmask[0])

#     n = len(poses_est)
#     error_low = 0
#     error_high = 0
#     error_dyna = 0
#     error_st = 0

#     for i in range(n):
#         w = gamma ** (n - i - 1)

#         i_error_low = (lowgtflow - flow_low_list[i]).abs()
#         error_low += w*(lowmask*i_error_low).mean()

#         i_error_high = (highgtflow - flow_high_list[i]).abs()
#         error_high += w*(highmask*i_error_high).mean()
        
#         coords_resi, dynamask1 = dyprojective_transform(poses_est[i], disps_est[i], highintrinsics, ii, jj, validmask, objectposes_est[i], fullmasks[0])

#         stmask = dynamask1*highmask
#         # dymask = fullmasks[0,:,ii, ..., None]*dynamask1*highmask

#         epe_st = (highgtflow - coords_resi).norm(dim=-1).reshape(-1)[stmask.reshape(-1)>0.5]
#         error_st += w * epe_st.mean()

#         # epe_dyna = (highgtflow - coords_resi).norm(dim=-1).reshape(-1)[dymask.reshape(-1)>0.5]
#         # error_dyna += w * epe_dyna.mean()

#     epe_low = (flow_low_list[-1] - lowgtflow).norm(dim=-1)
#     epe_low = epe_low.reshape(-1)[lowmask.reshape(-1) > 0.5]

#     epe_high = (flow_high_list[-1] - highgtflow).norm(dim=-1)
#     epe_high = epe_high.reshape(-1)[highmask.reshape(-1) > 0.5]

#     metrics = {
#         'low_f_error': epe_low.mean().item(),
#         'low_1px': (epe_low<1.0).float().mean().item(),

#         'high_f_error': epe_high.mean().item(),
#         'high_1px': (epe_high<1.0).float().mean().item(),

#         'st_f_error': epe_st.mean().item(),
#         'st_1px': (epe_st<1.0).float().mean().item(),

#         # 'dyna_f_error': epe_dyna.mean().item(),
#         # 'dyna_1px': (epe_dyna<1.0).float().mean().item(),
#     }

#     return error_low, error_high, error_st, metrics

def flow_loss(Ps, disps, poses_est, disps_est, ObjectPs, objectposes_est, objectmasks, quanmask, trackinfo, intrinsics, graph, flow_low_list, low_dispest, scale, gamma=0.9):
    """ optical flow loss """

    ii, jj, kk = graph_to_edge_list(graph)

    validmasklist = []
    for n in range(len(trackinfo['trackid'][0])):
        validmasklist.append(torch.isin(ii.to('cuda'), trackinfo['apperance'][n][0]) & torch.isin(jj.to('cuda'), trackinfo['apperance'][n][0]))
    validmask = torch.stack(validmasklist, dim=0)

    lowdisps = disps[:,:,3::8,3::8].clone()

    highintrinsics = intrinsics.clone()
    highintrinsics[...,:] *= 8

    # lowgtflow, lowmask = projective_transform(Ps, lowdisps, intrinsics, ii, jj)
    # highgtflow, highmask = projective_transform(Ps, disps, highintrinsics, ii, jj)

    lowgtflow, lowmask = dyprojective_transform(Ps, lowdisps, intrinsics, ii, jj, validmask, ObjectPs, objectmasks[0])
    highgtflow, highmask = dyprojective_transform(Ps, disps, highintrinsics, ii, jj, validmask, ObjectPs, quanmask[0])

    lowmask = lowmask * (lowdisps[:,ii] > 0).float().unsqueeze(dim=-1)
    highmask = highmask * (disps[:,ii] > 0).float().unsqueeze(dim=-1)

    n = len(poses_est)
    error_low = 0
    error_dyna = 0
    error_high = 0
    weighted_error_depth = 0

    s_disps = disps*scale
    s_lowdisps = lowdisps* scale

    valid_dyna = (1.0/s_lowdisps < 30.0)*(1.0/s_lowdisps > 0.2)*(objectmasks[0]>0)
    dymask = objectmasks[0,:,ii]

    for i in range(n):
        w = gamma ** (n - i - 1)

        #看预测的流准不准
        i_error_low = (lowgtflow - flow_low_list[i]).abs()
        l1_lowflow = lowmask*i_error_low
        error_low += w*(l1_lowflow.mean())

        #看预测的深度和Pose准不准
        # coords_resi, highmask1 = projective_transform(poses_est[i], disps_est[i], highintrinsics, ii, jj)
        coords_resi, highmask1 = dyprojective_transform(poses_est[i], disps_est[i], highintrinsics, ii, jj, validmask, objectposes_est[i], quanmask[0])

        diff_dyna = torch.abs(s_lowdisps - low_dispest[i]*scale)
        error_depth = torch.mean((diff_dyna/s_lowdisps)[valid_dyna])
        weighted_error_depth += w*error_depth

        #动态区域流
        epe_dyna = i_error_low[dymask>0.5]
        error_dyna += w * epe_dyna.mean()

        v = (highmask1 * highmask).squeeze(dim=-1)
        epe_high = v * (highgtflow - coords_resi).norm(dim=-1)
        error_high += w * epe_high.mean()

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
    
    s_disps_est = disps_est[-1]* scale
    s_low_dispest = low_dispest[-1] * scale

    valid_high = (1.0/s_disps < 30.0)*(1.0/s_disps > 0.2)
    high_gt = (s_disps)[valid_high]
    high_pred = (s_disps_est)[valid_high]

    high_diff = high_gt - high_pred
    abs_high_diff = torch.abs(high_diff)
    squared_diff = high_diff*high_diff
    abs_high_error = torch.mean(abs_high_diff)
    abs_re_high_error = torch.mean(abs_high_diff/high_gt)
    rmse_high = torch.sqrt(torch.mean(squared_diff))
    
    valid_low = (1.0/s_lowdisps < 30.0)*(1.0/s_lowdisps > 0.2)

    low_diff = s_lowdisps - s_low_dispest
    abs_low_diff = torch.abs(low_diff)
    squared_diff = low_diff*low_diff
    abs_low_error = torch.mean(abs_low_diff[valid_low])
    abs_low_dyna_error = torch.mean(abs_low_diff[valid_dyna])
    abs_re_low_error = torch.mean((abs_low_diff/s_lowdisps)[valid_low])
    # valid_dyna = valid_low*objectmasks[0]
    # error_depth = torch.mean((abs_low_diff/s_lowdisps)[valid_dyna.bool()])
    rmse_low = torch.sqrt(torch.mean(squared_diff[valid_low]))

    epe_low = l1_lowflow[lowmask[..., 0] > 0.5]
    epe_high = epe_high.reshape(-1)[v.reshape(-1) > 0.5]

    metrics = {
        'low_f_error': epe_low.mean().item(),
        'low_1px': (epe_low<1.0).float().mean().item(),

        'dyna_f_error': epe_dyna.mean().item(),
        'dyna_1px': (epe_dyna<1.0).float().mean().item(),

        'high_f_error': epe_high.mean().item(),
        'high_1px': (epe_high<1.0).float().mean().item(),

        'low depth RMSE': rmse_low.item(),
        'high depth RMSE': rmse_high.item(),

        'abs_high_error':abs_high_error.item(),
        'abs_low_error':abs_low_error.item(),
        'abs_low_dyna_error': abs_low_dyna_error.item(),

        'abs_re_high_error': abs_re_high_error.item(),
        'abs_re_low_error':abs_re_low_error.item(),

        'error_depth': error_depth.item(),
    }
    return error_low, error_high, error_dyna, weighted_error_depth, metrics
