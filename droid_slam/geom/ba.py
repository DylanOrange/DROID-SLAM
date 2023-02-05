from pickle import TRUE
import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve, schur_solve
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

    # r = (target - coords)*valid*weight
    # # residual = r[r!=0.0]
    # print('residual is {}'.format(torch.mean((torch.abs(r)))))

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

    # C = C + eta.view(*C.shape) + 1e-7
    C = C + 1e-7

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

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
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
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses
    
def dynamicBA(target, weight, objectposes, objectmask, trackinfo, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

    B, P, ht, wd = disps.shape#1,12,30,101
    N = ii.shape[0]#42
    D = poses.manifold_dim#6
    batch_grid = trackinfo['grid']

    ### 1: compute jacobians and residuals ###
    # coords, valid, (Ji_st, Jj_st, Jz_st) = pops.projective_transform(
    #     poses, disps, intrinsics, ii, jj, jacobian=True)

    coords, valid, (Jci, Jcj, Joi, Joj, Jz) = pops.dyprojective_transform(
        poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, objectmask = objectmask, Jacobian = TRUE, batch = False, batch_grid = None)

    # r = (target - coords)*valid*weight
    # # residual = r[r!=0.0]
    # print('residual is {}'.format(torch.mean((torch.abs(r)))))

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    w = .001*(valid*weight).view(B,N,-1,1)

    Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Joi = Joi.reshape(B, N, -1, D) *validmask[..., None, None]#1,18,30,101,2,6->1,18,6060,6
    Joj = Joj.reshape(B, N, -1, D) *validmask[..., None, None]#1,18,30,101,2,6->1,18,6060,6
    # Jci = torch.zeros_like(Jci)
    # Jcj = torch.zeros_like(Jcj)
    # Joi = torch.zeros_like(Joi)
    # Joj = torch.zeros_like(Joj)

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
    #     hoi_list.append(hoi)
    # ho = torch.cat(hoi_list, dim = 2)
    # # ho = torch.zeros_like(ho)
    # h = torch.cat([hc, ho], dim = 2) #1,14,6,6060,6
    # h_test = h.transpose(2,3).contiguous()#1,14,6060,6,6

    #w:2,14,8330,1
    h = h.transpose(2,3).contiguous().view(B, N, k, U*D)#2,14,8330,36
    wh= h*w#2,14,8330,36

    # N_app = trackinfo['n_app'][0] + P-(N_car+1)*fixedp

    # h_v = wh_test.view(B, N, -1, N_app*D)#1,24,6060,24*6
    # h_vtrans = h_v.transpose(2,3)

    v_test = torch.matmul(wh.transpose(2,3), r)#2,14,36,8330    2,14,8330,1
    v_test = torch.sum(v_test, dim = 1)
    v_test = v_test.view(B, U, D)

    h = h.view(B, N*k, U*D)
    wh = wh.view(B, N*k, U*D)
    # h_test = h_test.view(B, -1, N_app*D)#1,84840,36
    # wh_transpose = wh_test. view(B, -1, N_app*D)
    # wh_transpose = wh_transpose.transpose(1,2)
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

    # # only optimize keyframe poses
    # P = P + fixedp#4
    # ii = ii - fixedp
    # jj = jj - fixedp

    Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
        safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,15,6,3030

    Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
        safe_scatter_add_mat(Eoj, jj, kk, P , M)#6,15,6,3030

    C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
    w = safe_scatter_add_vec(wk, kk, M)#1,5,3030 

    # C = C + 1e-7
    C = C + eta.view(*C.shape) + 1e-7

    Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
    Eo = Eo.view(B, P, M, D, ht*wd)#6,3,5,6,30*101

    E = torch.zeros(B, U, M, D, ht*wd, dtype=Ec.dtype, device = Ec.device)
    for i in range(B):
        E[i] = torch.cat([Ec[i],Eo[i, trackinfo['apperance'][i][0]][fixedp:]], dim=0)
    #     Eo_list.append(Eo[i, trackinfo['apperance'][i][0]][fixedp:])
    # Eo = torch.cat(Eo_list, dim =0).unsqueeze(0)
    # E = torch.cat([Ec,Eo], dim=1)

    ### 3: solve the system ###
    # dx = block_solve(H_test, v_test)#1,4,6,1,5,3030
    dx, dz = schur_solve(H_test, E, C, v_test, w)#1,4,6,1,5,3030

    P = P-fixedp
    #NOTE:先用batch中的相机pose更新的平均值
    update = torch.mean(dx[:,:P], dim = 0, keepdim = True)
    poses = pose_retr(poses, update, torch.arange(P).to(device=dx.device) + fixedp)

    # idx = P
    for i in range(B):
        # nextidx = idx+len(trackinfo['apperance'][i][0])-fixedp
        objectposes[i] = pose_retr(objectposes[i, None], dx[i, P:][None], torch.arange(P).to(device=dx.device) + fixedp)
        # idx = nextidx

    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)
    
    return poses, objectposes, disps

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

def fulldynamicBA(target, weight, objectposes, objectmask, trackinfo, validmask, eta, poses, disps, intrinsics, ii, jj, fixedp=0):

    B, P, ht, wd = disps.shape#1,12,30,101
    N = ii.shape[0]#42
    D = poses.manifold_dim#6
    N_car = objectmask.shape[0]

    ### 1: compute jacobians and residuals ###

    coords, valid, (Jci, Jcj, Joi, Joj) = pops.dyprojective_transform(
        poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, objectmask = objectmask, Jacobian = TRUE)

    r = (target - coords)*valid*weight
    residual = r[r!=0.0]
    print('residual is {}'.format(torch.mean((torch.abs(residual)))))

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    # w = (valid*weight).view(B,N,-1,1)
    w = (valid*weight).repeat(1,1,1,1,2).view(B,N,-1,1)

    Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Joi = Joi.reshape(N_car, N, -1, D) *validmask[..., None, None]#1,18,30,101,2,6->1,18,6060,6
    Joj = Joj.reshape(N_car, N, -1, D) *validmask[..., None, None]#1,18,30,101,2,6->1,18,6060,6

    i = torch.arange(N).to('cuda')
    ii_test = i*P + ii
    jj_test = i*P + jj

    hc = scatter_sum(Jci, ii_test, dim = 1,dim_size= N*P) + scatter_sum(Jcj, jj_test, dim = 1,dim_size= N*P)
    hc = hc.view(B, N, P, -1, D)#1,14,5,6060,6
    hc = hc[:, :, fixedp:]#1,14,3,6060,6
    hoi_list = []
    for i in range(N_car):
        hoi = scatter_sum(Joi[i], ii_test, dim = 0,dim_size= N*P) + scatter_sum(Joj[i], jj_test, dim = 0,dim_size= N*P)
        hoi = hoi.view(B, N, P, -1, D)
        hoi = hoi[:, :, trackinfo['apperance'][i][0]]
        hoi = hoi[:, :, fixedp:]
        hoi_list.append(hoi)
    ho = torch.cat(hoi_list, dim = 2)
    # ho = torch.zeros_like(ho)
    h = torch.cat([hc, ho], dim = 2) #1,14,6,6060,6
    h_test = h.transpose(2,3).contiguous()#1,14,6060,6,6
    wh_test = h_test*w[..., None]

    N_app = trackinfo['n_app'][0] + P-(N_car+1)*fixedp

    h_v = wh_test.view(B, N, -1, N_app*D)#1,24,6060,24*6
    h_vtrans = h_v.transpose(2,3)
    v_test = torch.matmul(h_vtrans, r)
    v_test = torch.sum(v_test, dim = 1)
    v_test = v_test.view(B, N_app, D)
    h_test = h_test.view(B, -1, N_app*D)#1,84840,36
    wh_transpose = wh_test. view(B, -1, N_app*D)
    wh_transpose = wh_transpose.transpose(1,2)
    H_test = torch.matmul(wh_transpose, h_test)###weight乘了两次！！！
    H_test = H_test.view(B, N_app, D, N_app, D).transpose(2,3)

    # Jz = Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
    # # Jz = torch.zeros_like(Jz)

    # Eci = ((w*Jci).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030
    # Ecj = ((w*Jcj).transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030

    # Eoi = ((w*Joi).transpose(2,3).view(N_car,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030
    # Eoj = ((w*Joj).transpose(2,3).view(N_car,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030

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

    # C = C + eta.view(*C.shape) + 1e-7

    # Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
    # Eo = Eo.view(N_car, P, M, D, ht*wd)#6,3,5,6,30*101
    # Eo_list = []
    # for i in range(N_car):
    #     Eo_list.append(Eo[i, trackinfo['apperance'][i][0]][fixedp:])
    # Eo = torch.cat(Eo_list, dim =0).unsqueeze(0)
    # E = torch.cat([Ec,Eo], dim=1)

    ### 3: solve the system ###
    dx = block_solve(H_test, v_test)#1,4,6,1,5,3030
    # dx, dz = schur_solve(H_test, E, C, v_test, w)#1,4,6,1,5,3030

    P = P-fixedp
    poses = pose_retr(poses, dx[:, :P, :], torch.arange(P).to(device=dx.device) + fixedp)

    idx = P
    for i in range(N_car):
        nextidx = idx+len(trackinfo['apperance'][i][0])-fixedp
        objectposes[i] = pose_retr(objectposes[i, None], dx[:, idx:nextidx], trackinfo['apperance'][i][0][fixedp:])
        idx = nextidx

    # disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    # disps = disps.clamp(min=0.0)
    
    return poses, objectposes, disps