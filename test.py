
import sys
from tkinter import N
from xml.etree.ElementTree import TreeBuilder
sys.path.append("..") 
import torch
from lietorch import SE3
from torch_scatter import scatter_sum
# from droid_slam.geom.chol import block_solve, schur_solve
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import glob
from PIL import Image
import os
from functools import reduce
import flow_vis as vis

import evo
from evo.core.trajectory import PosePath3D
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

import torch.nn.functional as F

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def block_solve(H, b, ep=0.0001, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)

    I = torch.eye(N*D).to(H.device)
    H = H + (ep + lm*H) * I

    b = b.reshape(B, N*D, 1)

    x = CholeskySolver.apply(H,b)#1,18,18, 1,18,1
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.0001, lm=0.0001, sless=False):
    """ solve using shur complement """
    
    B, P, M, D, HW = E.shape#1,8,5,6,30*101
    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)#1,48,48
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)#1,48,5*30*101
    Q = (1.0 / C).view(B, M*HW, 1)#1,5*3030,1

    # damping
    I = torch.eye(P*D).to(H.device)
    H = H + (ep + lm*H) * I
    
    v = v.reshape(B, P*D, 1)#1,24,1
    w = w.reshape(B, M*HW, 1)#1,5*3030,1

    Et = E.transpose(1,2)#1,5*3030,24
    S = H - torch.matmul(E, Q*Et)#matmul(24,5*3010   5*3030, 24)
    v = v - torch.matmul(E, Q*w)

    dx = CholeskySolver.apply(S, v)
    dy = torch.linalg.solve(S,v)
    
    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx)    
    dx = dx.reshape(B, P, D)
    dz = dz.reshape(B, M, HW)

    return dx, dz

def coords_grid(ht, wd):
    y, x = torch.meshgrid(
        torch.arange(ht).float(),
        torch.arange(wd).float())
    return torch.stack([x, y], dim=-1)

def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat

def flow_read(flow_file):
    bgr = cv2.imread(flow_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    val = (bgr[..., 0] > 0).astype(np.float32)
    return out_flow, val

def depth_read(depth_file):
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                        cv2.IMREAD_ANYDEPTH) / (100.0)
    depth[depth == np.nan] = 1.0
    depth[depth == np.inf] = 1.0
    depth[depth == 0] = 1.0
    return depth

def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    out = scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1])
    return poses.retr(out)

def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    out = scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])
    return disps + out

def extract_intrinsics(intrinsics):
    return intrinsics[..., None, None, :].unbind(dim=-1)

def iproj(disps, intrinsics):
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy#投影到深度为1的面上
    pts = torch.stack([X, Y, i, disps], dim=-1)

    J = torch.zeros_like(pts)
    J[..., -1] = 1.0
    return pts, J

def actp(Gij, X0, jacobian=False, Gijobject = None, objectmask = None, fullmask  = None):
    """ action on point cloud """
    #X0: 1,12,30,101,4
    #objectmask: 2,12,30,101
    static = (1-fullmask)[..., None] #2,12,30,101,1
    cam_motion = Gij[:, :, None, None] * X0 #1,12,1,1,7 * 1,12,30,101,4
    cam_filtered = static * cam_motion#2,12,30,101,4

    ob_motion = Gijobject[:, :, None, None] * X0 #2,12,1,1,7* 1,12,30,101,4 ->2,12,30,101,4
    ob_filtered = torch.sum(objectmask[..., None]*ob_motion, dim = 0, keepdim= True)
    X1 = cam_filtered+ob_filtered

    # X1 = Gij[:, :, None, None] * X0

    X, Y, Z, d = X1.unbind(dim=-1)#2,12,30,101,4
    o = torch.zeros_like(d)
    B, N, H, W = d.shape

    J1 = torch.stack([
        d,  o,  o,  o,  Z, -Y,
        o,  d,  o, -Z,  o,  X,
        o,  o,  d,  Y, -X,  o,
        o,  o,  o,  o,  o,  o,
    ], dim=-1).view(B, N, H, W, 4, 6)
        
    return X1, J1

def proj(Xs, intrinsics):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5*0.2, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy

    coords = torch.stack([x, y], dim=-1)

    B, N, H, W = d.shape
    o = torch.zeros_like(d)
    proj_jac = torch.stack([
        fx*d,     o, -fx*X*d*d,  o,
        o,  fy*d, -fy*Y*d*d,  o,
        # o,     o,    -D*d*d,  d,
    ], dim=-1).view(B, N, H, W, 2, 4)

    return coords, proj_jac

def projective_transform(poses, depths, intrinsics, ii, jj, validmask = None, objectposes = None, objectmask = None):
    """ map points from ii->jj """
    
    #输入前面都有Batch 1，除了Objectmask
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii])#1,2,30,101,4

    # transform: both pose i and j are w2c
    Gij = poses[:, jj] * poses[:, ii].inv()
    # print(Gij.data)
    Gij.data[:, ii == jj] = torch.as_tensor(
        [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype = Gij.data.dtype,device=Gij.device)

    validobjectmask = objectmask[:, ii]*validmask[..., None, None]#2,12,30,101
    # objectmask = objectmask[:, ii]#2,12,30,101
    fullmask = torch.sum(objectmask[:, ii], dim = 0, keepdim = True)
    Gijobject =  poses[:, jj] * objectposes[:, jj].inv() * objectposes[:, ii] * poses[:, ii].inv()#2,12,1
    Gjj = poses[:, jj] * objectposes[:, jj].inv()#2,12,1
    Gijobject.data[:, ii == jj] = torch.as_tensor(
        [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype = Gij.data.dtype,device=Gij.device) 

    #X1, with object X2, without object
    X1, J1 = actp(Gij, X0, Gijobject = Gijobject, objectmask = validobjectmask, fullmask = fullmask)

    # project (pinhole)
    x1, Jp1 = proj(X1, intrinsics[:, jj])

    # exclude points too close to camera
    valid = ((X1[..., 2] > 0.2) & (X0[..., 2] > 0.2)).float()
    valid = valid.unsqueeze(-1)

        # Ji transforms according to dual adjoint
    Jcj = torch.matmul(Jp1, J1)#2,12,30,101,2,6
    Jci = -Gij[:, :, None, None, None].adjT(Jcj)

    Jcoi = -Gijobject[:, :, None, None, None].adjT(Jcj)

    Jci = torch.sum(Jcoi*validobjectmask[..., None, None], dim=0, keepdim=True) + Jci*(1-fullmask[..., None, None])

    Joi = Gjj[:, :, None, None, None].adjT(Jcj)
    Joj = -Joi

    Joi = Joi*validobjectmask[..., None, None]
    Joj = Joj*validobjectmask[..., None, None]

    Jz = torch.sum((Gijobject[:, :, None, None] * Jz) *validobjectmask[..., None], dim = 0, keepdim=True) + (Gij[:, :, None, None] * Jz) * (1 - fullmask[..., None])
    Jz = torch.matmul(Jp1, Jz.unsqueeze(-1))

    return x1, valid, (Jci, Jcj, Joi, Joj, Jz)

def transform(disp, transformation, intrinsics, ii, jj):
    X0, _ = iproj(disp[ii], intrinsics)
    X1, _ = actp(transformation, X0)
    x1, _ = proj(X1, intrinsics)

    valid = ((X1[..., 2] > 0.2) & (X0[..., 2] > 0.2)).float()
    valid = valid.unsqueeze(-1)

    return x1, X0, valid

def dynamicBA(target, weight, objectposes, objectmask, apperance, validmask, poses, disps, intrinsics, ii, jj, N_app, fixedp=0):

    B, P, ht, wd = disps.shape#1,12,30,101
    N = ii.shape[0]#42
    D = poses.manifold_dim#6
    N_car = objectmask.shape[0]

    ### 1: compute jacobians and residuals ###
    coords, valid, (Jci, Jcj, Joi, Joj, Jz) = projective_transform(
        poses, disps, intrinsics, ii, jj, validmask, objectposes = objectposes, objectmask = objectmask)

    r = (target - coords)*valid*weight
    # residual = r[r!=0.0]
    # print('residual is {}'.format(torch.mean((torch.abs(residual)))))


    epe = weight.squeeze(dim=-1) * (coords - target).norm(dim=-1)
    epe = epe.reshape(-1)[weight.reshape(-1) > 0.5]
    f_error = epe.mean().item()
    print('residual is {}'.format(f_error))


    r = r.view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    w = (valid*weight).repeat(1,1,1,1,2).view(B,N,-1,1)
    # w_test = (weight*valid)[..., None]
    # w = torch.ones_like(w)

    Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Joi = Joi.reshape(N_car, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Joj = Joj.reshape(N_car, N, -1, D) #1,18,30,101,2,6->1,18,6060,6

    Jci_test = w*Jci
    Jcj_test = w*Jcj
    Joi_test = w*Joi*validmask[..., None, None]
    Joj_test = w*Joj*validmask[..., None, None]

    i = torch.arange(N)
    ii_test = i*P + ii
    jj_test = i*P + jj

    hc = scatter_sum(Jci_test, ii_test, dim = 1,dim_size= N*P) + scatter_sum(Jcj_test, jj_test, dim = 1,dim_size= N*P)
    hc = hc.view(B, N, P, -1, D)#1,70,19,6060,6
    hc = hc[:, :, fixedp:]#1,70,19,6060,6
    hoi_list = []
    for i in range(N_car):
        hoi = scatter_sum(Joi_test[i], ii_test, dim = 0,dim_size= N*P) + scatter_sum(Joj_test[i], jj_test, dim = 0,dim_size= N*P)
        hoi = hoi.view(B, N, P, -1, D)
        hoi = hoi[:, :, apperance[i]]
        hoi = hoi[:, :, fixedp:]
        hoi_list.append(hoi)
    ho = torch.cat(hoi_list, dim = 2)
    # ho = torch.zeros_like(ho)
    h = torch.cat([hc, ho], dim = 2) #1,38,22,6060,6
    h_test = h.transpose(2,3).contiguous()#1,38,6060,22,6

    N_app = N_app + P-(N_car+1)*fixedp

    h_v = h_test.view(B, N, -1, N_app*D)#1,38,6060,22*6
    h_vtrans = h_v.transpose(2,3)#1,38,22*6,6060
    v_test = torch.matmul(h_vtrans, r)#1,38,22*6,6060 1,38,6060,1 ->1,38,22*6,1
    v_test = torch.sum(v_test, dim = 1)#1,22*6,1
    v_test = v_test.view(B, N_app, D)#1,22,6
    h_test = h_test.view(B, -1, N_app*D)#1,38,6060,22,6 -> 1,230280,132
    h_transpose = h_test.transpose(1,2)#1,132,230280
    H_test = torch.matmul(h_transpose, h_test)#1,132,132
    H_test = H_test.view(B, N_app, D, N_app, D).transpose(2,3)#1,22,22,6,6

    Jz = Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
    Jz = torch.zeros_like(Jz)

    Eci = (Jci_test.transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030
    Ecj = (Jcj_test.transpose(2,3).view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,14,6,3030

    Eoi = (Joi_test.transpose(2,3).view(N_car,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030
    Eoj = (Joj_test.transpose(2,3).view(N_car,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#6,14,6,3030

    w = w.view(B, N, ht*wd, -1)#1,14,3030,2
    r = r.view(B, N, ht*wd, -1)#1,14,3030,2
    wk = torch.sum(w*r*Jz, dim=-1)#1,18,3030
    Ck = torch.sum(w*Jz*Jz, dim=-1)#1,18,3030
    # # print(r[:,0,2525:2626])
    kx, kk = torch.unique(ii, return_inverse=True)#
    M = kx.shape[0]#5

    # # only optimize keyframe poses
    # P = P + fixedp#4
    # ii = ii - fixedp
    # jj = jj - fixedp

    Ec_test = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
        safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,15,6,3030

    Eo_test = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
        safe_scatter_add_mat(Eoj, jj, kk, P , M)#6,15,6,3030

    C_test = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
    w_test = safe_scatter_add_vec(wk, kk, M)#1,5,3030

    C_test = C_test + 1e-7 #eta, 5,30,101

    Ec_test = Ec_test.view(B, P, M, D, ht*wd)[:, fixedp:]#1,3,5,6,30*101
    Eo_test = Eo_test.view(N_car, P, M, D, ht*wd)#6,3,5,6,30*101
    Eo_list = []
    for i in range(N_car):
        Eo_list.append(Eo_test[i, apperance[i][fixedp:]])
    Eo_test = torch.cat(Eo_list, dim =0).unsqueeze(0)
    E_test = torch.cat([Ec_test,Eo_test], dim=1)
    
    # w = w.view(B,N,-1,1)
    # r = r.view(B, N, -1, 1)
    # wJoiT = (w * Joi).transpose(2,3)#1,18,6,6060
    # wJojT = (w * Joj).transpose(2,3)#1,18,6,6060
    # wJciT = (w* Jci).transpose(2,3)#1,18,6,6060
    # wJcjT = (w* Jcj).transpose(2,3)#1,18,6,6060

    # Jz = Jz.reshape(B, N, ht*wd, -1)#1,18,3030,2
    # # Jz = torch.zeros_like(Jz)
    # Hcii = torch.matmul(wJciT, Jci) #1,18,6,6
    # Hcij = torch.matmul(wJciT, Jcj)
    # Hcji = torch.matmul(wJcjT, Jci)
    # Hcjj = torch.matmul(wJcjT, Jcj)

    # Hoii = torch.matmul(wJoiT, Joi) #1,18,6,6
    # Hoij = torch.matmul(wJoiT, Joj)
    # Hoji = torch.matmul(wJojT, Joi)
    # Hojj = torch.matmul(wJojT, Joj)

    # Hcoii = torch.matmul(wJciT, Joi) #1,18,6,6
    # Hcoij = torch.matmul(wJciT, Joj)
    # Hcoji = torch.matmul(wJcjT, Joi)
    # Hcojj = torch.matmul(wJcjT, Joj)

    # vci = torch.matmul(wJciT, r).squeeze(-1)#1,18, 6
    # vcj = torch.matmul(wJcjT, r).squeeze(-1)

    # voi = torch.matmul(wJoiT, r).squeeze(-1)#1,18, 6
    # voj = torch.matmul(wJojT, r).squeeze(-1)

    # Eci = (wJciT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,38,6,3030,2  1,38,1,3030,2 ->1,38,6,3030
    # Ecj = (wJcjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,38,6,3030

    # Eoi = (wJoiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,38,6,3030
    # Eoj = (wJojT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)#1,38,6,3030

    # w = w.view(B, N, ht*wd, -1)#1,18,3030,2
    # r = r.view(B, N, ht*wd, -1)#1,18,3030,2
    # wk = torch.sum(w*r*Jz, dim=-1)#1,38,3030
    # Ck = torch.sum(w*Jz*Jz, dim=-1)#1,38,3030
    # # print(r[:,0,2525:2626])
    # kx, kk = torch.unique(ii, return_inverse=True)#
    # M = kx.shape[0]#5

    # # only optimize keyframe poses
    # P = P- fixedp#4
    # ii = ii - fixedp
    # jj = jj - fixedp

    # Hc = safe_scatter_add_mat(Hcii, ii, ii, P, P) + \
    #     safe_scatter_add_mat(Hcij, ii, jj, P, P) + \
    #     safe_scatter_add_mat(Hcji, jj, ii, P, P) + \
    #     safe_scatter_add_mat(Hcjj, jj, jj, P, P)#1,16,6,6
    
    # Ho = safe_scatter_add_mat(Hoii, ii, ii, P, P) + \
    #     safe_scatter_add_mat(Hoij, ii, jj, P, P) + \
    #     safe_scatter_add_mat(Hoji, jj, ii, P, P) + \
    #     safe_scatter_add_mat(Hojj, jj, jj, P, P)#1,16,6,6
    
    # Hco = safe_scatter_add_mat(Hcoii, ii, ii, P, P) + \
    #     safe_scatter_add_mat(Hcoij, ii, jj, P, P) + \
    #     safe_scatter_add_mat(Hcoji, jj, ii, P, P) + \
    #     safe_scatter_add_mat(Hcojj, jj, jj, P, P)#1,16,6,6

    # Ec = safe_scatter_add_mat(Eci, ii, kk, P, M) + \
    #     safe_scatter_add_mat(Ecj, jj, kk, P, M)#1,20,6,3030

    # Eo = safe_scatter_add_mat(Eoi, ii, kk, P, M) + \
    #     safe_scatter_add_mat(Eoj, jj, kk, P, M)#1,20,6,3030

    # vc = safe_scatter_add_vec(vci, ii, P) + \
    #     safe_scatter_add_vec(vcj, jj, P)#1,4,6
    
    # vo = safe_scatter_add_vec(voi, ii, P) + \
    #     safe_scatter_add_vec(voj, jj, P)#1,4,6

    # C = safe_scatter_add_vec(Ck, kk, M)#1,5,3030
    # w = safe_scatter_add_vec(wk, kk, M)#1,5,3030

    # C = C  + 1e-7 #eta, 5,30,101

    # Hc = Hc.view(B, P, P, D, D)#1,4,4,6,6
    # Ho = Ho.view(B, P, P, D, D)#1,4,4,6,6
    # Hco = Hco.view(B, P, P, D, D)
    # Hoc = Hco.permute(0,2,1,4,3)#1,4,4,6,6
    # Ec = Ec.view(B, P, M, D, ht*wd)#1,11,11,6,30*101
    # Eo = Eo.view(B, P, M, D, ht*wd)#1,11,11,6,30*101

    # v = torch.cat([vc,vo], dim=1)
    # E = torch.cat([Ec,Eo], dim=1)
    # H = torch.cat([torch.cat([Hc, Hco], dim = 2), torch.cat([Hoc, Ho], dim = 2)], dim = 1)
    ### 3: solve the system ###
    dx = block_solve(H_test, v_test)#1,4,6,1,5,3030
    # dx, dz = schur_solve(H_test, E_test, C_test, v_test, w_test)#1,4,6,1,5,3030
    P = P-fixedp
    # print('update value {}'.format(torch.mean(torch.abs(dz[dz!=0]))))
    poses = pose_retr(poses, dx[:, :P, :], torch.arange(P).to(device=dx.device) + fixedp)
    # print('after update {}'.format(poses.data))

    idx = P
    for i in range(N_car):
        nextidx = idx+len(apperance[i])-fixedp
        objectposes[i] = pose_retr(objectposes[i, None], dx[:, idx:nextidx], apperance[i][fixedp:])
        idx = nextidx
    # incre_object = dx[:,P:,:].view(N_car, P, D)
    # objectposes = pose_retr(objectposes, incre_object, torch.arange(P).to(device=dx.device) + fixedp).data
    # disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    # disps = disps.clamp(min=0.0)

    return poses, objectposes, disps

def dynamicmoBA(target, weight, objectposes, objectmask, poses, disps, intrinsics, ii, jj, fixedp=1):
    """ Motion only bundle adjustment """

    _, P, _, _ = disps.shape#1,2,30,101
    N = ii.shape[0]#2
    D = poses.manifold_dim#6
    _, B, _, _ =objectmask.shape 

    ### 1: compute jacobians and residuals ###
    coords, valid, (Jci, Jcj, Joi, Joj, Jz) = projective_transform(
        poses, disps, intrinsics, ii, jj, objectposes = objectposes, objectmask = objectmask)

    r = (target - coords)*valid
    residual = r[r!=0.0]
    print('residual is {}'.format(torch.mean(torch.abs(residual))))

    r = (target - coords).view(B, N, -1, 1) #1,18,30,101,2-> 1,18,6060,1
    w = valid.repeat(1,1,1,1,2).view(B,N,-1,1) #1,18,3030,1
    w = torch.ones_like(w)

    ### 2: construct linear system ###

    Jci = Jci.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    Jcj = Jcj.reshape(B, N, -1, D) #1,18,30,101,2,6->1,18,6060,6
    wJciT = (w * Jci).transpose(2,3)#1,18,6,6060
    wJcjT = (w * Jcj).transpose(2,3)#1,18,6,6060.squeeze(0)18,6060,6
    wJoiT = (w * Joi).transpose(2,3)#1,18,6,6060
    wJojT = (w * Joj).transpose(2,3)#1,18,6,6060

    Hcii = torch.matmul(wJciT, Jci) #1,18,6,6
    Hcij = torch.matmul(wJciT, Jcj)
    Hcji = torch.matmul(wJcjT, Jci)
    Hcjj = torch.matmul(wJcjT, Jcj)

    Hoii = torch.matmul(wJoiT, Joi) #1,18,6,6
    Hoij = torch.matmul(wJoiT, Joj)
    Hoji = torch.matmul(wJojT, Joi)
    Hojj = torch.matmul(wJojT, Joj)

    Hcoii = torch.matmul(wJciT, Joi) #1,18,6,6
    Hcoij = torch.matmul(wJciT, Joj)
    Hcoji = torch.matmul(wJcjT, Joi)
    Hcojj = torch.matmul(wJcjT, Joj)

    vci = torch.matmul(wJciT, r).squeeze(-1)#1,18, 6
    vcj = torch.matmul(wJcjT, r).squeeze(-1)

    voi = torch.matmul(wJoiT, r).squeeze(-1)#1,18, 6
    voj = torch.matmul(wJojT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P  - fixedp#3
    ii = ii  - fixedp#-1,0
    jj = jj  - fixedp#0,-1

    Hc = safe_scatter_add_mat(Hcii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hcij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hcji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hcjj, jj, jj, P, P)#1,16,6,6

    Ho = safe_scatter_add_mat(Hoii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hoij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hoji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hojj, jj, jj, P, P)#1,16,6,6

    Hco = safe_scatter_add_mat(Hcoii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hcoij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hcoji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hcojj, jj, jj, P, P)#1,16,6,6

    vc = safe_scatter_add_vec(vci, ii, P) + \
            safe_scatter_add_vec(vcj, jj, P)#1,4,6
        
    vo = safe_scatter_add_vec(voi, ii, P) + \
            safe_scatter_add_vec(voj, jj, P)#1,4,6
    
    Hc = Hc.view(B, P, P, D, D)#1,4,4,6,6
    Ho = Ho.view(B, P, P, D, D)#1,4,4,6,6
    Hco = Hco.view(B, P, P, D, D)
    Hoc = Hco.permute(0,2,1,4,3)#1,4,4,6,6

    v = torch.cat([vc,vo], dim=1)
    H = torch.cat([torch.cat([Hc, Hco], dim = 2), torch.cat([Hoc, Ho], dim = 2)], dim = 1)
    ### 3: solve the system ###

    dx = block_solve(H, v)#1,3,3,6,6, 1,3,6

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx[:, :P, :], torch.arange(P).to(device=dx.device) + fixedp).data
    objectposes = pose_retr(objectposes, dx[:, P:, :], torch.arange(P).to(device=dx.device) + fixedp).data
    return SE3(poses), SE3(objectposes)


def objectposes_read(file_path, trackID):

    objectpose = {}
    for id in trackID:
        raw_mat = np.loadtxt(file_path, delimiter=' ', skiprows=1)
        mask = ((raw_mat[:,1] == 0) & (raw_mat[:,2] == id))
        mat = raw_mat[mask]
        raw_pose = mat[:,7:13]

        r = raw_pose[:,3] +np.pi/2
        rotation = R.from_euler('y', r)
        o2w = np.zeros((raw_pose.shape[0],7))
        o2w[:,0:3] = raw_pose[:, 0:3]
        o2w[:,3:7] = rotation.as_quat()
        # o2w = torch.as_tensor(o2w, dtype=torch.double)

        # o2w = SE3(o2w)
        # w2o = o2w.inv()
        objectpose[id]= o2w
    return objectpose

def add_neighborhood_factors(t0, t1, r=2):
    """ add edges between neighboring frames within radius r """

    ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1))
    ii = ii.reshape(-1).to(dtype=torch.long)
    jj = jj.reshape(-1).to(dtype=torch.long)

    keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
    return ii[keep], jj[keep]

dataset_path = '../autodl-tmp/vkitti/'
sceneID= '20'
index  = [128, 132, 136, 140, 144]
framenumber = len(index)
w1,h1 = 1242, 375
h0,w0 = 375,1242
cut = 1
CAR_NUMBER = 6
NCAR = 135
#objectmask
mask_path = sorted(glob.glob(os.path.join(dataset_path, 'Scene'+sceneID, '15-deg-left/frames/instanceSegmentation/Camera_0/*.png')))[:150]
trackidlist = []
count = [torch.tensor([0])]
area = torch.zeros(NCAR).long()
for i in index:
        mask = Image.open(mask_path[i])
        fullmask = torch.from_numpy(np.array(mask))
        arearank = torch.bincount(fullmask. flatten())
        area += scatter_sum(arearank, torch.arange(arearank.shape[0]), dim = 0, dim_size= NCAR)
        valid = arearank[arearank!=0].shape[0]
        count.append(count[-1]+valid)
        trackidlist.append(torch.argsort(arearank)[-valid:])
        # frequency = (np.bincount(fullmask.flatten())).astype(int)
        # trackidlist.append(np.argsort(frequency)[-(CAR_NUMBER+1):-1])#找出出现面积最大的十辆车
trackid = torch.concat(trackidlist)
arearank = torch.argsort(area)
fre = torch.bincount(trackid)
frerank = torch.where(fre == torch.amax(fre))[0]
TRACKID = torch.from_numpy(np.intersect1d(frerank, arearank[-frerank.shape[0]:]))#找出出现频率最大的几辆车
if TRACKID.shape[0] > 8:
    TRACKID = torch.from_numpy(np.intersect1d(frerank, arearank[-8:]))
if TRACKID.shape[0] == 1 and TRACKID == torch.tensor([0]):
    frerank = torch.where(torch.isin(fre, torch.tensor([torch.amax(fre),torch.amax(fre)-1])))[0]
    TRACKID = torch.from_numpy(np.intersect1d(frerank, arearank[-frerank.shape[0]:]))
TRACKID = TRACKID[TRACKID!=0]-1
# TRACKID = torch.tensor([0,9])

bins = torch.concat(count)
Apperance = []
N_app = 0
for id in TRACKID:
    ids = torch.nonzero(trackid == id+1).squeeze(-1)
    frames = torch.bucketize(ids,bins,right =True)-1
    N_app += len(frames)
    Apperance.append(frames)

objectgt_path = os.path.join(dataset_path, 'Scene'+sceneID, '15-deg-left/pose.txt')
objectposes_list = []
for n, id in enumerate(TRACKID):
    objectpose = torch.zeros((len(index), 7)).double()
    idx = Apperance[n]
    raw_mat = torch.from_numpy(np.loadtxt(objectgt_path, delimiter=' ', skiprows=1))
    mask = ((raw_mat[:,1] == 0) & (raw_mat[:,2] == id)) & (torch.isin(raw_mat[:, 0], torch.tensor(index)))
    mat = raw_mat[mask]
    raw_pose = mat[:,7:13]
    r = raw_pose[:,3] +torch.pi/2
    rotation = R.from_euler('y', r)
    o2w = torch.concat((raw_pose[:, 0:3], torch.from_numpy(rotation.as_quat())), dim=1)
    # print(o2w.shape[0])
    objectpose[idx] = o2w
    objectposes_list.append(objectpose)
objectposes = torch.stack(objectposes_list, dim = 0)

objectposes =  SE3(objectposes).inv()#2,1,6

mask_list = []
fullmask_list = []
single_mask_list = []
single_fullmask_list = []

for id in TRACKID:
    for i in index:
        mask = Image.open(mask_path[i])
        fullmask = np.array(mask)
        mask = np.array(mask.resize((w1//cut, h1//cut)))
        mask = np.where(mask == (np.array(id)+1), 1.0, 0.0)
        fullmask = np.where(fullmask == (np.array(id)+1), 1.0, 0.0)
        mask_list.append(mask)
        fullmask_list.append(fullmask)
    masks = np.stack(mask_list, axis=0)
    fullmasks = np.stack(fullmask_list, axis = 0)
    mask_list = []
    fullmask_list = []
    single_mask_list.append(masks)
    single_fullmask_list.append(fullmasks)
mask = np.stack(single_fullmask_list,axis = 0)
fullmask = torch.from_numpy(np.sum(np.stack(single_fullmask_list, axis = 0),axis = 0))

#camera pose
poses = np.loadtxt(os.path.join(dataset_path, 'Scene'+sceneID, '15-deg-left/extrinsic.txt'), delimiter=' ', skiprows=1)[::2, 2:]
poses = poses.reshape(-1, 4, 4)
r = rmat_to_quad(poses[:, 0:3, 0:3])
t = poses[:, :3, 3] 
poses = torch.as_tensor(np.concatenate((t, r), axis=1), dtype=torch.double)
poses = SE3((poses[index])[None])#2,7

#disp
depths_path = sorted(glob.glob(os.path.join(dataset_path, 'Scene'+sceneID, '15-deg-left/frames/depth/Camera_0/*.png')))[:150]
depth_list = []
for i in index:
    depth = depth_read(depths_path[i])
    depth_list.append(torch.from_numpy(depth[None]))
depths = torch.stack(depth_list,dim=0)
# depths = depths*fullmask.unsqueeze(1)
depths = torch.nn.functional.interpolate(depths, size = (h1//cut, w1//cut), mode = 'nearest')
disps = 1.0 / depths.squeeze(1)#2,375,1242
depth_valid = (depths < 655.35)
disps = disps[None]

# sample = torch.rand(1,1,10,10)
# print(sample)
# sampled = torch.nn.functional.interpolate(sample, size = (5, 5), mode = 'bilinear')
# print(sampled)

#image
image_path = sorted(glob.glob(os.path.join(dataset_path, 'Scene'+sceneID, '15-deg-left/frames/rgb/Camera_0/*.jpg')))[:150]
image_list = []
for i in index:
    image = cv2.imread(image_path[i])
    image_list.append(image)
images = np.stack(image_list, axis=0)

#visualize mask
# fullmask = np.array(fullmask)
# for i in range(framenumber):
#     vis_image = np.copy(images[i])
#     vis_image[np.nonzero(fullmask[i])] = np.array([255,255,255])
#     cv2.imwrite('mask'+str(i)+'.png', vis_image)
objectmask = (torch.from_numpy(mask))
objectmask = torch.nn.functional.interpolate(objectmask.transpose(0,1), size = (h1//cut, w1//cut), mode = 'nearest').transpose(0,1)

#intrinsics
fx, fy, cx, cy = 725.0087, 725.0087, 620.5, 187
intrinsics = torch.as_tensor([fx, fy, cx, cy])
intrinsics[0:2] *= ((w1//cut)/ w0)
intrinsics[2:4] *= ((h1//cut)/ h0)
intrinsics = intrinsics[None, None]
intrinsics = intrinsics.repeat(1,framenumber,1)

#artificial flow
ii, jj = add_neighborhood_factors(0,framenumber)
validmasklist = []
for n in range(len(TRACKID)):
   validmasklist.append(torch.isin(ii, Apperance[n]) & torch.isin(jj, Apperance[n]))
validmask = torch.stack(validmasklist, dim=0)

# objectmask = F.interpolate(objectmask[:,None], (h1//8, w1//8))
flow, fullweight, (_, _, _, _, _)  = projective_transform(poses, disps, intrinsics, ii, jj, validmask, objectposes, objectmask)
depth_valid = depth_valid[ii][None, ..., None]
# weight = torch.ones_like(weight)
# weight = depth_valid[ii][None, ..., None]

#add noise
# fullweight[:,:,:20,:20,:] = 0.0
# fullweight[:,:,:,180:,:] = 0.0

# flow = torch.nn.functional.interpolate(flow[0].permute(0,3,1,2), size = (h1//cut, w1//cut), mode = 'nearest')
# flow = flow.permute(0,2,3,1).unsqueeze(0)

# disps = torch.nn.functional.interpolate(disps.transpose(0,1), size = (h1//cut, w1//cut), mode = 'nearest')
# disps = disps.transpose(0,1)

# objectmask = torch.nn.functional.interpolate(objectmask.transpose(0,1), size = (h1//cut, w1//cut), mode = 'nearest')
# objectmask = objectmask.transpose(0,1)

fullmask = torch.sum(objectmask[:, ii], dim = 0, keepdim = True)
weight = fullmask.unsqueeze(-1)
# fullweight = torch.ones_like(weight)

# for i in range(flow.shape[1]):
#     flow_image = vis.flow_to_image(np.array(flow[0,i]), np.array(fullmask[0,i]))
#     cv2.imwrite('flow{}.png'.format(i),flow_image)

flow_error = torch.normal(mean=flow, std=1e-1)
# for i in range(flow.shape[1]):
#     flow_image = vis.flow_to_image(np.array(flow_error[0,i]), np.array(fullmask[0,i]))
#     cv2.imwrite('flow_error{}.png'.format(i),flow_image)

epe = weight.squeeze(dim=-1) * (flow_error - flow).norm(dim=-1)
epe = epe.reshape(-1)[weight.reshape(-1) > 0.5]
f_error = epe.mean().item()
onepx = (epe<1.0).float().mean().item()
print(f_error)
print(onepx)

pose_est = poses.data.clone()
object_est = objectposes.data.clone()
pose_est = SE3(pose_est)
object_est = SE3(object_est)
pose_est.data[:,0,:] = poses.data[:, 0, :]
pose_est.data[:,1:,:] = poses.data[:, 1, :].unsqueeze(1)
object_est.data[:,0,:] = objectposes.data[:, 0, :]
object_est.data[:,1:,:] = objectposes.data[:, 1, :].unsqueeze(1)
# def depth_loss(pred, gt,valid):
#     valid_pixel = (pred - gt[None])*valid[None]
#     valid_pixel = valid_pixel[valid_pixel!=0]
#     return torch.mean(torch.abs(valid_pixel))

disps_error = disps.clone()
# disps_error = torch.ones_like(disps)
# disps_error[:] = disps[:,0]
# disps_error = torch.normal(mean=disps_error, std=1e-3)
# depth_error = (1/disps_error)
# loss_before = depth_loss(depth_error, depths, depth_valid)
# disps_error = disps_error*depth_valid[None]
# print('before {}'.format(loss_before))

for i in range(5):
    pose_est, object_est, disps_error = dynamicBA(flow_error, fullweight, object_est, objectmask, Apperance, validmask, pose_est, disps_error, intrinsics, ii, jj, N_app, fixedp=2)

# disps_error[disps_error == 0] = 1
# depth_error = (1/disps_error)
# loss_after = depth_loss(depth_error, depths, depth_valid)
# print('after {}'.format(loss_after))

flow_after, weight2, (_, _, _, _, _)  = projective_transform(pose_est, disps_error, intrinsics, ii, jj, validmask, object_est, objectmask)
val = weight*weight2
epe = val.squeeze(dim=-1) * (flow - flow_after).norm(dim=-1)
epe = epe.reshape(-1)[val.reshape(-1) > 0.5]
f_error = epe.mean().item()
onepx = (epe<1.0).float().mean().item()
print('after')
print(f_error)
print(onepx)


traj_ref = PosePath3D(
        positions_xyz=poses.data[0, :, :3],
        orientations_quat_wxyz=poses.data[0, :, 3:])

traj_est = PosePath3D(
        positions_xyz=pose_est.data[0, :, :3],
        orientations_quat_wxyz=pose_est.data[0, :, 3:])

result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) #input c2w
print('---camera pose----')
print(result)

print('---object pose---')

for i, id in enumerate(TRACKID):
    obtraj_ref = PosePath3D(
            positions_xyz=objectposes.data[i, Apperance[i], :3],
            orientations_quat_wxyz=objectposes.data[i, Apperance[i], 3:])

    obtraj_est = PosePath3D(
            positions_xyz=object_est.data[i, Apperance[i], :3],
            orientations_quat_wxyz=object_est.data[i, Apperance[i], 3:])

    result = main_ape.ape(obtraj_ref, obtraj_est, est_name='traj',
                            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) # input o2w
    print('----result for car {}----'.format(id))
    print(result)

print('---ground truth---')
print(objectposes.data[0, :, :3])
print('---estimated ---')
print(object_est.data[0, :, :3])

#after
# print('after disps error is {}'.format(torch.mean(torch.abs((disps_error - disps)*objectmask.squeeze(1)))))
# # transformation = poses[jj]*poses[ii].inv()
# transformation = poses[jj]*objectposes[jj].inv()*objectposes[ii]*poses[ii].inv()
# corr_after, pts_after, valid_after = transform(disps_error, transformation, intrinsics, ii, jj)
# corr_after = flowval * corr_after

# pts_after = transformation[:,None, None, :]*pts_after
# pts_after = flowval * pts_after

# pts_residual = pts_before - pts_after
# residual = (corr_before - corr_after)
# # ma_residual =  pts_before_ma - pts_after_ma
# # a = pts_before[pts_residual!=0]
# # b = pts_after[pts_residual!=0]

# # r_be = corr_before - flow
# # r_af = corr_after - flow
# # a = r_be[corr_before!=0]
# # b = r_af[corr_after!=0]
# # print(torch.mean(a))
# # print(torch.mean(b))
# # print('after {}'.format(transformation.data))
# a = residual[residual!=0]
# b = pts_residual[pts_residual!=0]
# print(torch.mean(a))
# print(torch.mean(b))
