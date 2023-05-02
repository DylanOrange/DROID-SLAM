from pickle import TRUE
import lietorch
import torch
import torch.nn.functional as F
from lietorch import SE3

from .chol import block_solve, schur_solve, block_solve_initial
import geom.projective_ops as pops

from torch_scatter import scatter_sum


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Full Bundle Adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords)*valid*weight
    print('residual is {}'.format(torch.mean((torch.abs(r)))))

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    # Ji = torch.zeros_like(Ji)
    # Jj = torch.zeros_like(Jj)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)
    # Jz = torch.zeros_like(Jz)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(w*r*Jz, dim=-1)
    Ck = torch.sum(w*Jz*Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    C = C + eta.view(*C.shape) + 1e-7
    # C = C + 1e-7

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w)
    
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps


def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)
    # w = .001 * (valid * weight).repeat(1,1,1,1,2).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)#B,N,D,-1
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)#B,N,D,D
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve_initial(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses
    
# def dynamicBA(target, weight, objectposes, objectmask, trackinfo, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

#     B, P, ht, wd = disps.shape#1,12,30,101
#     N = ii.shape[0]#42
#     D = poses.manifold_dim#6
#     # batch_grid = trackinfo['grid']

#     ### 1: compute jacobians and residuals ###
#     # coords, valid, (Ji_st, Jj_st, Jz_st) = pops.projective_transform(
#     #     poses, disps, intrinsics, ii, jj, jacobian=True)

#     coords, valid, (Jci, Jcj, Joi, Joj, Jz) = pops.dyprojective_transform(
#         poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, objectmask = objectmask, Jacobian = TRUE, batch = False, batch_grid = None)

#     # r = (target - coords)*valid*weight
#     # # residual = r[r!=0.0]
#     # print('residual is {}'.format(torch.mean((torch.abs(r)))))

#     r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
#     w = .001*(valid*weight).view(B,N,-1,1)

#     Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
#     Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
#     Joi = Joi.reshape(B, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
#     Joj = Joj.reshape(B, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
#     # Jci = torch.zeros_like(Jci)
#     # Jcj = torch.zeros_like(Jcj)

#     i = torch.arange(N).to('cuda')
#     ii_test = i*P + ii
#     jj_test = i*P + jj

#     hc = scatter_sum(Jci, ii_test, dim = 1,dim_size= N*P) + scatter_sum(Jcj, jj_test, dim = 1,dim_size= N*P)
#     hc = hc.view(B, N, P, -1, D)#1,14,5,6060,6
#     hc = hc[:, :, fixedp:]#1,14,3,6060,6

#     U = 2*(P-fixedp)
#     _, _, _, k, _=hc.shape
#     h = torch.zeros(B, N, U, k, D, dtype=hc.dtype, device= hc.device)#2,14,6,8330,6

#     for i in range(B):
#         hoi = scatter_sum(Joi[i], ii_test, dim = 0,dim_size= N*P) + scatter_sum(Joj[i], jj_test, dim = 0,dim_size= N*P)
#         hoi = hoi.view(N, P, -1, D)
#         hoi_out = torch.zeros_like(hoi, dtype=hoi.dtype)
#         hoi_out[:, trackinfo['apperance'][i][0][fixedp:]] = hoi[:, trackinfo['apperance'][i][0][fixedp:]]
#         h[i] = torch.cat([hc[i], hoi_out[:, fixedp:]], dim=1)
#     #     hoi_list.append(hoi)
#     # ho = torch.cat(hoi_list, dim = 2)
#     # # ho = torch.zeros_like(ho)
#     # h = torch.cat([hc, ho], dim = 2) #1,14,6,6060,6
#     # h_test = h.transpose(2,3).contiguous()#1,14,6060,6,6

#     #w:2,14,8330,1
#     h = h.transpose(2,3).contiguous().view(B, N, k, U*D)#2,14,8330,36
#     wh= h*w#2,14,8330,36

#     # N_app = trackinfo['n_app'][0] + P-(N_car+1)*fixedp

#     # h_v = wh_test.view(B, N, -1, N_app*D)#1,24,6060,24*6
#     # h_vtrans = h_v.transpose(2,3)

#     v_test = torch.matmul(wh.transpose(2,3), r)#2,14,36,8330    2,14,8330,1
#     v_test = torch.sum(v_test, dim = 1)
#     v_test = v_test.view(B, U, D)

#     h = h.view(B, N*k, U*D)
#     wh = wh.view(B, N*k, U*D)
#     # h_test = h_test.view(B, -1, N_app*D)#1,84840,36
#     # wh_transpose = wh_test. view(B, -1, N_app*D)
#     # wh_transpose = wh_transpose.transpose(1,2)
#     H_test = torch.matmul(wh.transpose(1,2), h)###weight乘了两次！！！
#     H_test = H_test.view(B, U, D, U, D).transpose(2,3)

#     Jz = Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
#     # Jz = torch.zeros_like(Jz)

#     Eci = ((w*Jci).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030
#     Ecj = ((w*Jcj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030

#     Eoi = ((w*Joi).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030
#     Eoj = ((w*Joj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030

#     w = w.view(B, N, ht*wd, -1)#1,14,3030,2
#     r = r.view(B, N, ht*wd, -1)#1,14,3030,2
#     wk = torch.sum(w*r*Jz, dim=-1)#1,18,3030
#     Ck = torch.sum(w*Jz*Jz, dim=-1)#1,18,3030
#     kx, kk = torch.unique(ii, return_inverse=True)#
#     M = kx.shape[0]#5

#     # # only optimize keyframe poses
#     # P = P + fixedp#4
#     # ii = ii - fixedp
#     # jj = jj - fixedp

#     Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
#         safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,15,6,3030

#     Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
#         safe_scatter_add_mat(Eoj, jj, kk, P , M)#6,15,6,3030

#     C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
#     w = safe_scatter_add_vec(wk, kk, M)#1,5,3030 

#     # C = C + 1e-7
#     C = C + eta.view(*C.shape) + 1e-7

#     Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
#     Eo = Eo.view(B, P, M, D, ht*wd)#6,3,5,6,30*101

#     E = torch.zeros(B, U, M, D, ht*wd, dtype=Ec.dtype, device = Ec.device)
#     for i in range(B):
#         E[i, :U//2] = Ec[i]
#         E[i, U//2 + trackinfo['apperance'][i][0][fixedp:]-fixedp] = Eo[i, trackinfo['apperance'][i][0]][fixedp:]
#     #     Eo_list.append(Eo[i, trackinfo['apperance'][i][0]][fixedp:])
#     # Eo = torch.cat(Eo_list, dim =0).unsqueeze(0)
#     # E = torch.cat([Ec,Eo], dim=1)

#     ### 3: solve the system ###
#     # dx = block_solve(H_test, v_test)#1,4,6,1,5,3030
#     dx, dz = schur_solve(H_test, E, C, v_test, w)#1,4,6,1,5,3030

#     P = P-fixedp
#     #NOTE:先用batch中的相机pose更新的平均值
#     update = torch.mean(dx[:,:P], dim = 0, keepdim = True)
#     poses = pose_retr(poses, update, torch.arange(P).to(device=dx.device) + fixedp)

#     # idx = P
#     for i in range(B):
#         # nextidx = idx+len(trackinfo['apperance'][i][0])-fixedp
#         objectposes[i] = pose_retr(objectposes[i, None], dx[i, P:][None], torch.arange(P).to(device=dx.device) + fixedp)
#         # idx = nextidx

#     disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
#     disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
#     disps = disps.clamp(min=0.0)
    
#     return poses, objectposes, disps, valid

def cameraBA(target, weight, objectposes, objectmask, trackinfo, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

    B, P, ht, wd = disps.shape#1,12,30,101
    N = ii.shape[0]#42
    D = poses.manifold_dim#6

    ### 1: compute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    # coords, valid, (Jci, Jcj, Joi, Joj, Jz) = pops.dyprojective_transform(
    #     poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, objectmask = objectmask, Jacobian = TRUE, batch = None, batch_grid = None)

    # r = (target - coords)*valid*weight
    # residual = r[r!=0.0]
    # print('residual is {}'.format(torch.mean((torch.abs(residual)))))

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    w = .001*(valid*weight).view(B,N,-1,1)

    Jci = Ji.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jcj = Jj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    # Joi = Joi.reshape(B, N, -1, D) *validmask[..., None, None]#1,18,30,101,2,6->1,18,6060,6
    # Joj = Joj.reshape(B, N, -1, D) *validmask[..., None, None]#1,18,30,101,2,6->1,18,6060,6

    Joi = torch.zeros_like(Jci)
    Joj = torch.zeros_like(Jcj)

    i = torch.arange(N).to('cuda')
    ii_test = i*P + ii
    jj_test = i*P + jj

    hc = scatter_sum(Jci, ii_test, dim = 1,dim_size= N*P) + scatter_sum(Jcj, jj_test, dim = 1,dim_size= N*P)
    hc = hc.view(B, N, P, -1, D)#1,14,5,6060,6
    hc = hc[:, :, fixedp:]#1,14,3,6060,6

    U = 2*(P-fixedp)
    _, _, _, k, _=hc.shape
    h = torch.zeros(B, N, U, k, D, dtype=hc.dtype, device= hc.device)#2,14,6,8330,6

    for i in range(B):
        hoi = scatter_sum(Joi[i], ii_test, dim = 0,dim_size= N*P) + scatter_sum(Joj[i], jj_test, dim = 0,dim_size= N*P)
        hoi = hoi.view(N, P, -1, D)
        hoi_out = torch.zeros_like(hoi, dtype=hoi.dtype)
        hoi_out[:, trackinfo['apperance'][i][0][fixedp:]] = hoi[:, trackinfo['apperance'][i][0][fixedp:]]
        h[i] = torch.cat([hc[i], hoi_out[:, fixedp:]], dim=1)

    #w:2,14,8330,1
    h = h.transpose(2,3).contiguous().view(B, N, k, U*D)#2,14,8330,36
    wh= h*w#2,14,8330,36
    v_test = torch.matmul(wh.transpose(2,3), r)#2,14,36,8330    2,14,8330,1
    v_test = torch.sum(v_test, dim = 1)
    v_test = v_test.view(B, U, D)

    h = h.view(B, N*k, U*D)
    wh = wh.view(B, N*k, U*D)
    H_test = torch.matmul(wh.transpose(1,2), h)###weight乘了两次！！！
    H_test = H_test.view(B, U, D, U, D).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
    # Jz = torch.zeros_like(Jz)

    Eci = ((w*Jci).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030
    Ecj = ((w*Jcj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030

    Eoi = ((w*Joi).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030
    Eoj = ((w*Joj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030

    w = w.view(B, N, ht*wd, -1)#1,14,3030,2
    r = r.view(B, N, ht*wd, -1)#1,14,3030,2
    wk = torch.sum(w*r*Jz, dim=-1)#1,18,3030
    Ck = torch.sum(w*Jz*Jz, dim=-1)#1,18,3030
    kx, kk = torch.unique(ii, return_inverse=True)#
    M = kx.shape[0]#5

    Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
        safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,15,6,3030

    Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
        safe_scatter_add_mat(Eoj, jj, kk, P , M)#6,15,6,3030

    C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
    w = safe_scatter_add_vec(wk, kk, M)#1,5,3030 

    C = C + eta.view(*C.shape) + 1e-7
    # C = C + 1e-7

    Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
    Eo = Eo.view(B, P, M, D, ht*wd)#6,3,5,6,30*101

    E = torch.zeros(B, U, M, D, ht*wd, dtype=Ec.dtype, device = Ec.device)
    for i in range(B):
        E[i] = torch.cat([Ec[i],Eo[i, trackinfo['apperance'][i][0]][fixedp:]], dim=0)

    ### 3: solve the system ###
    # dx = block_solve(H_test, v_test)#1,4,6,1,5,3030
    dx, dz = schur_solve(H_test, E, C, v_test, w)#1,4,6,1,5,3030

    P = P-fixedp
    #NOTE:先用batch中的相机pose更新的平均值
    update = torch.mean(dx[:,:P], dim = 0, keepdim = True)
    poses = pose_retr(poses, update, torch.arange(P).to(device=dx.device) + fixedp)

    # idx = P
    for i in range(B):
        objectposes[i] = pose_retr(objectposes[i, None], dx[i, P:][None], torch.arange(P).to(device=dx.device) + fixedp)

    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)
    
    return poses, objectposes, disps

# def dynamicBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

#     B, P, ht, wd = disps.shape#1,2,30,101
#     N = ii.shape[0]#2
#     D = poses.manifold_dim#6
#     DO = 3
#     N_car = objectmask.shape[0]

#     ### 1: co mpute jacobians and residuals ###
#     coords, valid, (Jci, Jcj, Joi, Joj) = pops.dyprojective_transform(
#         poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, \
#         objectmask = objectmask, Jacobian = True, batch = False)

#     r = (target - coords)
#     residual = r[r!=0.0]
#     print('residual is {}'.format(torch.mean(residual)))

#     r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
#     # weight[objectmask[:,ii]>0] = valid[objectmask[:,ii]>0].repeat(1,2)
#     w = .001*(valid*weight).view(B,N,-1,1) #1,18,3030,1
#     # w = .001*(valid*weight).repeat(1,1,1,1,2).view(B,N,-1,1)

#     Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
#     Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
#     Joi = Joi.reshape(N_car, N, -1, DO)#1,18,30,101,2,6->1,18,6060,6
#     Joj = Joj.reshape(N_car, N, -1, DO)#1,18,30,101,2,6->1,18,6060,6

#     i = torch.arange(N).to('cuda')
#     ii_scatter = i*P + ii
#     jj_scatter = i*P + jj

#     hc = scatter_sum(Jci, ii_scatter, dim = 1,dim_size= N*P) + scatter_sum(Jcj, jj_scatter, dim = 1,dim_size= N*P)
#     hc = hc.view(B, N, P, -1, D).permute(0,1,3,2,4)#1,18,6060,6,6
#     hc = hc[:, :, :, fixedp:].reshape(B, N, -1, (P-fixedp)*D)

#     ho = scatter_sum(Joi, ii_scatter, dim = 1, dim_size= N*P) + scatter_sum(Joj, jj_scatter, dim = 1, dim_size= N*P)
#     ho = ho.view(B, N, P, -1, DO).permute(0,1,3,2,4)#1,18,6,6060,3
#     ho = ho[:, :, :, fixedp:].reshape(B, N, -1, (P-fixedp)*DO)

#     h = torch.cat((hc, ho), dim = -1)

#     # U = h.shape[2]
#     k = hc.shape[2]

#     # h = h.transpose(2,3).contiguous().view(B, N, k, U*D)#2,14,8330,36
#     wh= h*w#2,14,8330,36

#     v = torch.matmul(wh.transpose(2,3), r)#2,14,36,8330    2,14,8330,1
#     v = torch.sum(v, dim = 1).view(B,-1,1)

#     h = h.view(B, N*k, -1)
#     wh = wh.view(B, N*k, -1)
#     H = torch.matmul(wh.transpose(1,2), h)###weight乘了两次！！！
#     # H = H.view(B, U, D, U, D).transpose(2,3)

#     # Jz = Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
#     # # Jz = torch.zeros_like(Jz)

#     # Eci = ((w*Jci).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,18,6,3030
#     # Ecj = ((w*Jcj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,18,6,3030

#     # Eoi = ((w*Joi).transpose(2,3).view(B,N,DO,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,18,3,3030
#     # Eoj = ((w*Joj).transpose(2,3).view(B,N,DO,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,18,3,3030

#     # w = w.view(B, N, ht*wd, -1)#1,18,3030,2
#     # r = r.view(B, N, ht*wd, -1)#1,18,3030,2
#     # wk = torch.sum(w*r*Jz, dim=-1)#1,18,3030
#     # Ck = torch.sum(w*Jz*Jz, dim=-1)#1,18,3030
#     # kx, kk = torch.unique(ii, return_inverse=True)#
#     # M = kx.shape[0]#5

#     # Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
#     #     safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,36,6,3030

#     # Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
#     #     safe_scatter_add_mat(Eoj, jj, kk, P , M)#1,36,3,3030

#     # C = safe_scatter_add_vec(Ck, kk, M).view(B,-1,1)#1,6,3030
#     # w = safe_scatter_add_vec(wk, kk, M).view(B,-1,1)#1,6,3030 

#     # # C = C + 1e-7 #eta, 5,30,101
#     # C = C + eta.view(*C.shape) + 1e-7

#     # Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:].permute(0,1,3,2,4)#1,5,7,6,30*101->1,5,6,7,30*101
#     # Eo = Eo.view(B, P, M, DO, ht*wd)[:, fixedp:].permute(0,1,3,2,4)#1,5,7,3,30*101->1,5,3,7,30*101

#     # E = torch.cat((Ec.reshape(B, (P-fixedp)*D, M, ht*wd), Eo.reshape(B, (P-fixedp)*DO, M, ht*wd)), dim=1)

#     # ## 3: solve the system ###
#     dx = block_solve(H, v)
#     # dx, dz = schur_solve(H, E, C, v, w)#1,4,6,1,5,3030
    
#     camera_dx = dx[:,:(P-fixedp)*D].reshape(-1, P-fixedp, D)
#     object_dx = dx[:,(P-fixedp)*D:].reshape(-1, P-fixedp, DO)
#     dx0 = torch.zeros(1,P-fixedp,1, device = object_dx.device)
#     object_dx = torch.cat((object_dx[:,:,0].unsqueeze(-1),dx0, object_dx[:,:,1].unsqueeze(-1),
#                            dx0,object_dx[:,:,2].unsqueeze(-1),dx0),
#                            dim = -1)
#     print('update value is {}'.format(dx.mean().item()))

#     P = P-fixedp
#     poses = pose_retr(poses, camera_dx, torch.arange(P).to(device=dx.device) + fixedp)
#     objectposes = pose_retr(objectposes, object_dx, torch.arange(P).to(device=dx.device) + fixedp)

#     # disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
#     # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
#     # disps = disps.clamp(min=0.0)
    
#     return poses, objectposes

def dynamictestBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

    B, P, ht, wd = disps.shape#1,2,30,101
    N = ii.shape[0]#2
    D = poses.manifold_dim#6
    N_car = objectmask.shape[0]

    ### 1: compute jacobians and residuals ###
    coords, valid, (Jci, Jcj, Joi, Joj, Jz) = pops.dyprojective_transform(
        poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, \
        objectmask = objectmask, Jacobian = True, batch = False, batch_grid = None)

    # r = (target - coords)*valid*weight
    # residual = r[r!=0.0]
    # print('residual is {}'.format(torch.mean((torch.abs(residual)))))

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    w = .001*(valid*weight).view(B,N,-1,1) #1,18,3030,1

    Ji = torch.cat((Jci, Joi), dim = 0).reshape(B+N_car, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jj = torch.cat((Jcj, Joj), dim = 0).reshape(B+N_car, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    wJiT = (w * Ji).transpose(2,3)#1,18,6,6060
    wJjT = (w * Jj).transpose(2,3)#1,18,6,6060.squeeze(0)18,6060,6

    Jz = Jz.reshape(B+N_car, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji) #1,18,6,6
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B+N_car,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B+N_car,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    
    w = w.view(B, N, ht*wd, -1)#1,14,3030,2
    r = r.view(B, N, ht*wd, -1)#1,14,3030,2
    wk = torch.sum(w*r*Jz, dim=-1)#1,18,3030
    Ck = torch.sum(w*Jz*Jz, dim=-1)#1,18,3030

    kx, kk = torch.unique(ii, return_inverse=True)#
    M = kx.shape[0]#5

    # P = P - fixedp
    # ii = ii - fixedp
    # jj = jj - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
    w = safe_scatter_add_vec(wk, kk, M)#1,5,3030 

    C = C + eta.view(-1, C.shape[1], C.shape[2]) + 1e-7

    H = H.view(B+N_car, P, P, D, D)
    E = E.view(B+N_car, P, M, D, ht*wd)

    #TODO:中间出现的车，在前端的前三帧固定不动
    for i in range(N_car):
        if len(app[i]) <= fixedp:
            #全部保留不优化,H,E,v置为0
            H[i+1] = E[i+1] = v[i+1] = 0
        else:
            first = app[i][0]+fixedp
            #保留前t0帧不优化，置为0
            H[i+1,:first] = 0
            H[i+1, :, :first] = 0
            E[i+1, :first] = 0
            v[i+1, :first] = 0

    P = P - fixedp

    H = H[:,fixedp:, fixedp:]
    E = E[:,fixedp:]
    v = v[:,fixedp:]

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w)#1,4,6,1,5,3030
    # print(dz.shape)
    # print('we optimize dx {}'.format(fixedp+torch.nonzero(torch.any(dx[1] != 0.0, dim = 1))))
    # print('we optimize dz {}'.format(torch.nonzero(torch.any(dz[1] != 0.0, dim = 1))))

    # P = P-fixedp
    poses = pose_retr(poses, dx[0, None], torch.arange(P).to(device=dx.device) + fixedp)

    # idx = P
    for i in range(N_car):
        # nextidx = idx+len(app[i])-fixedp
        objectposes[i] = pose_retr(objectposes[i, None], dx[i+1, None],  torch.arange(P).to(device=dx.device) + fixedp)
        # idx = nextidx

    for i in range(B+N_car):
        disps = disp_retr(disps, dz[i].view(B,-1,ht,wd), kx)
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)
    
    return poses, objectposes, disps

def dynamictestmoBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

    B, P, ht, wd = disps.shape#1,2,30,101
    N = ii.shape[0]#2
    D = poses.manifold_dim#6
    N_car = objectmask.shape[0]

    ### 1: compute jacobians and residuals ###
    coords, valid, (Jci, Jcj, Joi, Joj, Jz) = pops.dyprojective_transform(
        poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, objectmask = objectmask, Jacobian = True, batch = False, batch_grid = None)

    r = (target - coords)*valid*weight
    residual = r[r!=0.0]
    print('residual is {}'.format(torch.mean((torch.abs(residual)))))

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    w = .001*(valid*weight).view(B,N,-1,1) #1,18,3030,1

    Ji = torch.cat((Jci, Joi), dim = 0).reshape(B+N_car, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jj = torch.cat((Jcj, Joj), dim = 0).reshape(B+N_car, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    wJiT = (w * Ji).transpose(2,3)#1,18,6,6060
    wJjT = (w * Jj).transpose(2,3)#1,18,6,6060.squeeze(0)18,6060,6

    Hii = torch.matmul(wJiT, Ji) #1,18,6,6
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    H = H.view(B+N_car, P, P, D, D)

    for i in range(N_car):
        if len(app[i]) <= fixedp:
            #全部保留不优化,H,E,v置为0
            H[i+1] = v[i+1] = 0
        else:
            first = app[i][0]+fixedp
            #保留前t0帧不优化，置为0
            H[i+1,:first] = 0
            H[i+1, :, :first] = 0
            v[i+1, :first] = 0

    P = P - fixedp

    H = H[:,fixedp:, fixedp:]
    v = v[:,fixedp:]

    ### 3: solve the system ###
    dx = block_solve(H, v)#1,4,6,1,5,3030
    # print('we optimize dx {}'.format(fixedp+torch.nonzero(torch.any(dx[1] != 0.0, dim = 1))))

    poses = pose_retr(poses, dx[0, None], torch.arange(P).to(device=dx.device) + fixedp)

    for i in range(N_car):
        objectposes[i] = pose_retr(objectposes[i, None], dx[i+1, None],  torch.arange(P).to(device=dx.device) + fixedp)
    
    return poses, objectposes, disps

def midasBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, midasdisps, intrinsics, ii, jj, a, b, fixedp=0):

    app = app['apperance'][0][0]
    B, P, ht, wd = disps.shape#1,2,30,101
    N = ii.shape[0]#2
    D = poses.manifold_dim#6
    N_car = objectmask.shape[0]

    ### 1: co mpute jacobians and residuals ###
    coords, valid, (Jci, Jcj, Joi, Joj, Jz, Ja, Jb) = pops.dyprojective_transform(
        poses, b+a*midasdisps, intrinsics, ii, jj, validmask, objectposes = objectposes, \
        objectmask = objectmask, Jacobian = True, batch = False, midasdisps = midasdisps)

    r = (target - coords)*valid*weight
    residual = r[r!=0.0]
    print('residual is {}'.format(torch.mean((torch.abs(residual)))))

    Ja = torch.zeros_like(Ja)
    Jb = torch.zeros_like(Jb)

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    # w = .001*(valid*weight).view(B,N,-1,1) #1,18,3030,1
    w = .001*(valid*weight).repeat(1,1,1,1,2).view(B,N,-1,1)

    Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Joi = Joi.reshape(N_car, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
    Joj = Joj.reshape(N_car, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
    Ja = Ja.reshape(B, N, -1, 1)#1,18,6060,1
    Jb = Jb.reshape(B, N, -1, 1)#1,18,6060,1

    i = torch.arange(N).to('cuda')
    ii_scatter = i*P + ii
    jj_scatter = i*P + jj

    hc = scatter_sum(Jci, ii_scatter, dim = 1,dim_size= N*P) + scatter_sum(Jcj, jj_scatter, dim = 1,dim_size= N*P)
    hc = hc.view(B, N, P, -1, D)#1,14,5,6060,6

    hoi = scatter_sum(Joi, ii_scatter, dim = 1, dim_size= N*P) + scatter_sum(Joj, jj_scatter, dim = 1, dim_size= N*P)
    hoi = hoi.view(B, N, P, -1, D)

    h = torch.cat((hc[:, :, fixedp:], hoi[:, :, app[fixedp:]]), dim = 2)

    U = h.shape[2]
    k = hc.shape[3]

    ha = torch.zeros(B, N, k, P,device = hc.device)
    hb = ha.clone()
    for i in range(P):
        ha[:,ii==i,:,i] = Ja[:,ii==i,0]
        hb[:,ii==i,:,i] = Jb[:,ii==i,0]

    h = h.transpose(2,3).contiguous().view(B, N, k, U*D)#1,18,6060,48
    # h = torch.cat((h, ha, hb), dim = -1)#1,18,6060,60
    wh= h*w#2,14,8330,36

    v = torch.matmul(wh.transpose(2,3), r)#1,18,48,1
    v = torch.sum(v, dim = 1).view(B,U,D)

    h = h.view(B, N*k, U*D)#1,18*6060,48
    wh = wh.view(B, N*k, U*D)#1,18*6060,48
    H = torch.matmul(wh.transpose(1,2), h)###weight乘了两次！！！
    H = H.view(B, U, D, U, D).transpose(2,3)

    Jz = a[:, ii]*Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
    # Jz = torch.zeros_like(Jz)

    Eci = ((w*Jci).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030
    Ecj = ((w*Jcj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030

    Eoi = ((w*Joi).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030
    Eoj = ((w*Joj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030

    w = w.view(B, N, ht*wd, -1)#1,14,3030,2
    r = r.view(B, N, ht*wd, -1)#1,14,3030,2
    wk = torch.sum(w*r*Jz, dim=-1)#1,18,3030
    Ck = torch.sum(w*Jz*Jz, dim=-1)#1,18,3030
    kx, kk = torch.unique(ii, return_inverse=True)#
    M = kx.shape[0]#5

    Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
        safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,15,6,3030

    Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
        safe_scatter_add_mat(Eoj, jj, kk, P , M)#6,15,6,3030

    C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
    w = safe_scatter_add_vec(wk, kk, M)#1,5,3030 

    C = C  + 1e-7 #eta, 5,30,101
    # C = C + eta.view(*C.shape) + 1e-7

    Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
    Eo = Eo.view(B, N_car, P, M, D, ht*wd)#6,3,5,6,30*101

    E = torch.cat((Ec, Eo[:, 0, app[fixedp:]]), dim=1)

    dx, dz = schur_solve(H, E, C, v, w)#1,4,6,1,5,3030

    # da = dx[0, -2*P:-P, 0]
    # db = dx[0, -P:, 0]

    # dpose = dx[:,:-2*P].reshape(B,-1,D)

    P = P-fixedp
    poses = pose_retr(poses, dx[:,:P], torch.arange(P).to(device=dx.device) + fixedp)
    objectposes = pose_retr(objectposes, dx[:, P:], app[fixedp:])
    
    # a = a+da.view(B, -1, 1, 1)
    # b = b+db.view(B, -1, 1, 1)
    
    midasdisps = disp_retr(midasdisps, dz.view(B,-1,ht,wd), kx)
    midasdisps = torch.where(midasdisps > 10, torch.zeros_like(midasdisps), midasdisps)
    midasdisps = midasdisps.clamp(min=0.0)

    return poses, objectposes, a, b, midasdisps


def dynamicBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

    # app = app['apperance'][0][0]
    B, P, ht, wd = disps.shape#1,2,30,101
    N = ii.shape[0]#2
    D = poses.manifold_dim#6
    N_car = objectmask.shape[0]

    ### 1: co mpute jacobians and residuals ###
    coords, valid, (Jci, Jcj, Joi, Joj) = pops.dyprojective_transform(
        poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, \
        objectmask = objectmask, Jacobian = True, batch = False)

    r = (target - coords)
    # residual = r[r!=0.0]
    # residual = r[(valid*weight)[..., 0]>0.5]
    print('dyna residual is {}'.format(torch.mean(r[objectmask[:,ii]>0.5])))

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    w = .001*(valid*weight).view(B,N,-1,1) #1,18,3030,1
    # w = (valid*weight).repeat(1,1,1,1,2).view(B,N,-1,1)

    Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Joi = Joi.reshape(N_car, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
    Joj = Joj.reshape(N_car, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
    Jci = torch.zeros_like(Jci)
    Jcj = torch.zeros_like(Jcj)

    i = torch.arange(N).to('cuda')
    ii_scatter = i*P + ii
    jj_scatter = i*P + jj

    hc = scatter_sum(Jci, ii_scatter, dim = 1,dim_size= N*P) + scatter_sum(Jcj, jj_scatter, dim = 1,dim_size= N*P)
    hc = hc.view(B, N, P, -1, D)#1,14,5,6060,6

    hoi = scatter_sum(Joi, ii_scatter, dim = 1, dim_size= N*P) + scatter_sum(Joj, jj_scatter, dim = 1, dim_size= N*P)
    hoi = hoi.view(B, N, P, -1, D)

    h = torch.cat((hc[:, :, fixedp:], hoi[:, :, fixedp:]), dim = 2)

    U = h.shape[2]
    k = hc.shape[3]

    h = h.transpose(2,3).contiguous().view(B, N, k, U*D)#2,14,8330,36
    wh= h*w#2,14,8330,36

    v = torch.matmul(wh.transpose(2,3), r)#2,14,36,8330    2,14,8330,1
    v = torch.sum(v, dim = 1).view(B,U,D)

    h = h.view(B, N*k, U*D)
    wh = wh.view(B, N*k, U*D)
    H = torch.matmul(wh.transpose(1,2), h)###weight乘了两次！！！
    H = H.view(B, U, D, U, D).transpose(2,3)

    # Jz = Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
    # # Jz = torch.zeros_like(Jz)

    # Eci = ((w*Jci).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030
    # Ecj = ((w*Jcj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030

    # Eoi = ((w*Joi).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030
    # Eoj = ((w*Joj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030

    # w = w.view(B, N, ht*wd, -1)#1,14,3030,2
    # r = r.view(B, N, ht*wd, -1)#1,14,3030,2
    # wk = torch.sum(w*r*Jz, dim=-1)#1,18,3030
    # Ck = torch.sum(w*Jz*Jz, dim=-1)#1,18,3030
    # kx, kk = torch.unique(ii, return_inverse=True)#
    # M = kx.shape[0]#5

    # Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
    #     safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,15,6,3030

    # Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
    #     safe_scatter_add_mat(Eoj, jj, kk, P , M)#6,15,6,3030

    # C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
    # w = safe_scatter_add_vec(wk, kk, M)#1,5,3030 

    # # C = C + 1e-7 #eta, 5,30,101
    # C = C + eta.view(*C.shape) + 1e-7

    # Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
    # Eo = Eo.view(B, N_car, P, M, D, ht*wd)#6,3,5,6,30*101

    # E = torch.cat((Ec, Eo[:, 0, fixedp:]), dim=1)

    # # ## 3: solve the system ###
    # dx, dz = schur_solve(H, E, C, v, w)#1,4,6,1,5,3030

    dx = block_solve_initial(H, v)
    # print('update value is {}'.format(dx.mean().item()))

    P = P-fixedp
    poses = pose_retr(poses, dx[:,:P], torch.arange(P).to(device=dx.device) + fixedp)
    objectposes = pose_retr(objectposes, dx[:, P:], torch.arange(P).to(device=dx.device) + fixedp)

    # disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    # disps = disps.clamp(min=0.0)
    
    return poses, objectposes

# def dynamicBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

#     B, P, ht, wd = disps.shape#1,2,30,101
#     N = ii.shape[0]#2
#     D = poses.manifold_dim#6
#     DO = 3
#     N_car = objectmask.shape[0]

#     ### 1: co mpute jacobians and residuals ###
#     coords, valid, (Jci, Jcj, Joi, Joj) = pops.dyprojective_transform(
#         poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, \
#         objectmask = objectmask, Jacobian = True, batch = False)

#     r = (target - coords)
#     print('dyna residual is {}'.format(torch.mean(r[objectmask[:,ii]>0.5])))

#     r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
#     # weight[objectmask[:,ii]>0] = valid[objectmask[:,ii]>0].repeat(1,2)
#     # w = .001*(valid*weight).view(B,N,-1,1) #1,18,3030,1
#     w = .001*(valid*weight).repeat(1,1,1,1,2).view(B,N,-1,1) #1,18,3030,1

#     Joi = Joi.reshape(N_car, N, -1, DO)#1,18,30,101,2,6->1,18,6060,6
#     Joj = Joj.reshape(N_car, N, -1, DO)#1,18,30,101,2,6->1,18,6060,6

#     i = torch.arange(N).to('cuda')
#     ii_scatter = i*P + ii
#     jj_scatter = i*P + jj

#     ho = scatter_sum(Joi, ii_scatter, dim = 1, dim_size= N*P) + scatter_sum(Joj, jj_scatter, dim = 1, dim_size= N*P)
#     ho = ho.view(B, N, P, -1, DO).permute(0,1,3,2,4)#1,18,6,6060,3
#     ho = ho[:, :, :, fixedp:].reshape(B, N, -1, (P-fixedp)*DO)

#     k = ho.shape[2]

#     wh= ho*w#2,14,8330,36

#     v = torch.matmul(wh.transpose(2,3), r)#2,14,36,8330    2,14,8330,1
#     v = torch.sum(v, dim = 1).view(B,-1,1)

#     h = ho.view(B, N*k, -1)
#     wh = wh.view(B, N*k, -1)
#     H = torch.matmul(wh.transpose(1,2), h)###weight乘了两次！！！
  
#     dx = block_solve(H, v)
    
#     object_dx = dx.reshape(-1, P-fixedp, DO)
#     dx0 = torch.zeros(1,P-fixedp,1, device = object_dx.device)
#     object_dx = torch.cat((object_dx[:,:,0].unsqueeze(-1),dx0, object_dx[:,:,1].unsqueeze(-1),
#                            dx0,object_dx[:,:,2].unsqueeze(-1),dx0),
#                            dim = -1)
#     print('update value is {}'.format(dx.mean().item()))

#     P = P-fixedp
#     objectposes = pose_retr(objectposes, object_dx, torch.arange(P).to(device=dx.device) + fixedp)
    
#     return objectposes