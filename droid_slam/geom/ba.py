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

    r = (target - coords).view(B, N, -1, 1)
    # print('residual is {}'.format(torch.mean((torch.abs(r)))))
    w = .001 * (valid*weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

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

# def midasBA(target, weight, objectposes, objectmask, app, validmask, eta, poses, disps, midasdisps, intrinsics, ii, jj, a, b, fixedp=0):

#     app = app['apperance'][0][0]
#     B, P, ht, wd = disps.shape#1,2,30,101
#     N = ii.shape[0]#2
#     D = poses.manifold_dim#6
#     N_car = objectmask.shape[0]

#     ### 1: co mpute jacobians and residuals ###
#     coords, valid, (Jci, Jcj, Joi, Joj, Jz, Ja, Jb) = pops.dyprojective_transform(
#         poses, b+a*midasdisps, intrinsics, ii, jj, validmask, objectposes = objectposes, \
#         objectmask = objectmask, Jacobian = True, batch = False, midasdisps = midasdisps)

#     r = (target - coords)*valid*weight
#     residual = r[r!=0.0]
#     print('residual is {}'.format(torch.mean((torch.abs(residual)))))

#     r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
#     # w = .001*(valid*weight).view(B,N,-1,1) #1,18,3030,1
#     w = .001*(valid*weight).repeat(1,1,1,1,2).view(B,N,-1,1)

#     Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
#     Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
#     Joi = Joi.reshape(N_car, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
#     Joj = Joj.reshape(N_car, N, -1, D)#1,18,30,101,2,6->1,18,6060,6
#     Ja = Ja.reshape(B, N, -1, 1)#1,18,6060,1
#     Jb = Jb.reshape(B, N, -1, 1)#1,18,6060,1

#     i = torch.arange(N).to('cuda')
#     ii_scatter = i*P + ii
#     jj_scatter = i*P + jj

#     hc = scatter_sum(Jci, ii_scatter, dim = 1,dim_size= N*P) + scatter_sum(Jcj, jj_scatter, dim = 1,dim_size= N*P)
#     hc = hc.view(B, N, P, -1, D)#1,14,5,6060,6

#     hoi = scatter_sum(Joi, ii_scatter, dim = 1, dim_size= N*P) + scatter_sum(Joj, jj_scatter, dim = 1, dim_size= N*P)
#     hoi = hoi.view(B, N, P, -1, D)

#     h = torch.cat((hc[:, :, fixedp:], hoi[:, :, app[fixedp:]]), dim = 2)

#     U = h.shape[2]
#     k = hc.shape[3]

#     ha = torch.zeros(B, N, k, P,device = hc.device)
#     hb = ha.clone()
#     for i in range(P):
#         ha[:,ii==i,:,i] = Ja[:,ii==i,0]
#         hb[:,ii==i,:,i] = Jb[:,ii==i,0]

#     h = h.transpose(2,3).contiguous().view(B, N, k, U*D)#1,18,6060,48
#     # h = torch.cat((h, ha, hb), dim = -1)#1,18,6060,60
#     wh= h*w#2,14,8330,36

#     v = torch.matmul(wh.transpose(2,3), r)#1,18,48,1
#     v = torch.sum(v, dim = 1).view(B,U,D)

#     h = h.view(B, N*k, U*D)#1,18*6060,48
#     wh = wh.view(B, N*k, U*D)#1,18*6060,48
#     H = torch.matmul(wh.transpose(1,2), h)###weight乘了两次！！！
#     H = H.view(B, U, D, U, D).transpose(2,3)

#     Jz = a[:, ii]*Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
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

#     Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
#         safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,15,6,3030

#     Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
#         safe_scatter_add_mat(Eoj, jj, kk, P , M)#6,15,6,3030

#     C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
#     w = safe_scatter_add_vec(wk, kk, M)#1,5,3030 

#     C = C  + 1e-7 #eta, 5,30,101
#     # C = C + eta.view(*C.shape) + 1e-7

#     Ec = Ec.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
#     Eo = Eo.view(B, N_car, P, M, D, ht*wd)#6,3,5,6,30*101

#     E = torch.cat((Ec, Eo[:, 0, app[fixedp:]]), dim=1)

#     dx, dz = schur_solve(H, E, C, v, w)#1,4,6,1,5,3030

#     # da = dx[0, -2*P:-P, 0]
#     # db = dx[0, -P:, 0]

#     # dpose = dx[:,:-2*P].reshape(B,-1,D)

#     P = P-fixedp
#     poses = pose_retr(poses, dx[:,:P], torch.arange(P).to(device=dx.device) + fixedp)
#     objectposes = pose_retr(objectposes, dx[:, P:], app[fixedp:])
    
#     # a = a+da.view(B, -1, 1, 1)
#     # b = b+db.view(B, -1, 1, 1)
    
#     midasdisps = disp_retr(midasdisps, dz.view(B,-1,ht,wd), kx)
#     midasdisps = torch.where(midasdisps > 10, torch.zeros_like(midasdisps), midasdisps)
#     midasdisps = midasdisps.clamp(min=0.0)

#     return poses, objectposes, a, b, midasdisps

def midasBA(target, weight, eta, poses, midasdisps, scale, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Full Bundle Adjustment """

    B, P, ht, wd = midasdisps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, scale*midasdisps, intrinsics, ii, jj, midasdisps, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    # print('residual is {}'.format(torch.mean((torch.abs(r)))))
    w = .001 * (weight*valid).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

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
    scale = scale+dz.view(B,-1,ht,wd)
    # disps = a*disps
    # disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    midasdisps = torch.where(scale*midasdisps > 10, torch.zeros_like(midasdisps), midasdisps)
    midasdisps = midasdisps.clamp(min=0.0)
    scale = scale.clamp(min=0.0)

    return poses, midasdisps, scale
