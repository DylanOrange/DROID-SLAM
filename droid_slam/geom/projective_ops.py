import torch
import torch.nn.functional as F
import numpy as np
from lietorch import SE3, Sim3
import cv2
import matplotlib.pyplot as plt


MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)

def batch_grid(corners, rec):
    # B = corners.shape[0]
    # x_list, y_list = [], []
    # for i in range(B):
    #     y, x = torch.meshgrid(
    #         torch.arange(corners[i,0], corners[i,0]+rec[0]).float(),
    #         torch.arange(corners[i,1], corners[i,1]+rec[1]).float())
    #     y_list.append(y)
    #     x_list.append(x)
    # x = torch.stack(x_list).unsqueeze(1)
    # y = torch.stack(y_list).unsqueeze(1)

    y, x = torch.meshgrid(
        torch.arange(corners[0], corners[0]+rec[0]).float(),
        torch.arange(corners[1], corners[1]+rec[1]).float())
    return (x,y)

def crop(images, center, rec, depth = False):
    # B,N,_,_,ch = images.shape
    # output = torch.zeros(B, N, rec[0], rec[1], ch, dtype=images.dtype, device = images.device)
    # if depth:
    #     output[output == 0.0] = -0.1
    # for n in range(B):
    #     crop = images[n, :, center[n,0]:center[n,0]+rec[0], center[n,1]:center[n,1]+rec[1]]
    #     try:
    #         output[n] = crop
    #     except Exception as e:
    #         #padding
    #         # print(e)
    #         h,w = crop.shape[1:3]
    #         output[n,:, :h, :w] = crop
    output = images[:, :, center[0]:center[0]+rec[0], center[1]:center[1]+rec[1]]

    return output

def iproj(disps, intrinsics, jacobian=False, batch_grid = None):
    """ pinhole camera inverse projection """
    # ht, wd = disps.shape[2:]
    B,_, ht, wd = disps.shape
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    
    if batch_grid is not None:
        x, y = batch_grid[0][0], batch_grid[1][0]
    else:
        y, x = torch.meshgrid(
            torch.arange(ht, dtype = disps.dtype, device = disps.device),
            torch.arange(wd, dtype = disps.dtype, device = disps.device))#2,49,85

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        # Ja = -J*disps[..., None]*disps[..., None]
        # Jb = Ja/midasdisps[..., None]
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """ map points from ii->jj """

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    
    # transform
    Gij = poses[:,jj] * poses[:,ii].inv()

    Gij.data[:,ii==jj] = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype = Gij.data.dtype, device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian)
    
    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def induced_flow(poses, disps, intrinsics, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[...,:2] - coords0, valid

def induced_object_flow(poses, disps, intrinsics, objectposes, objectmasks, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    validmask = torch.ones(len(ii), device = objectmasks.device)
    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = dyprojective_transform(poses, disps, intrinsics, ii, jj, validmask, objectposes, objectmasks)

    return coords1[...,:2] - coords0, valid

def dyactp(Gij, X0, Gijobject = None, objectmask = None, fullmask  = None, jacobian = False, batch = False):
    """ action on point cloud """
    #X0: 1,12,30,101,4
    #objectmask: 2,12,30,101
    if batch:
        static = (1-objectmask)[..., None] #7,14,49,108,1
        cam_motion = Gij[:, :, None, None] * X0 #1,14,1,1,7 * 7,14,49,108,4-> 7,14,49,108,4
        cam_filtered = static * cam_motion#7,14,49,108,4

        ob_motion = Gijobject[:, :, None, None] * X0 #7,14,1,1,7* 7,14,49,108,4 ->7,14,49,108,4 
        ob_filtered = objectmask[..., None]*ob_motion#7,14,49,108,4 
        X1 = cam_filtered+ob_filtered#7,14,49,108,4 

    else:
        static = (1-fullmask)[..., None] #2,12,30,101,1
        cam_motion = Gij[:, :, None, None] * X0 #1,12,1,1,7 * 1,12,30,101,4
        cam_filtered = static * cam_motion#2,12,30,101,4

        ob_motion = Gijobject[:, :, None, None] * X0 #2,12,1,1,7* 1,12,30,101,4 ->2,12,30,101,4
        ob_filtered = torch.sum(objectmask[..., None]*ob_motion, dim = 0, keepdim= True)
        X1 = cam_filtered+ob_filtered

    # X1 = Gij[:, :, None, None] * X0
    if jacobian:
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

    return X1, None

def dyprojective_transform(poses, depths, intrinsics, ii, jj, validmask = None, objectposes = None, \
                           objectmask = None, Jacobian = False, return_depth = False, batch = False, batch_grid = None, midasdisps = None):
    """ map points from ii->jj """
    
    #输入前面都有Batch 1，除了Objectmask
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii], jacobian = Jacobian, batch_grid= batch_grid)#1,2,30,101,4

    # transform: both pose i and j are w2c
    Gij = poses[:, jj] * poses[:, ii].inv()
    # print(Gij.data)
    Gij.data[:, ii == jj] = torch.as_tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype = Gij.data.dtype,device=Gij.device)

    validobjectmask = objectmask[:, ii]*validmask[..., None, None]#2,12,30,101
    # objectmask = objectmask[:, ii]#2,12,30,101
    fullmask = torch.sum(objectmask[:, ii], dim = 0, keepdim = True)
    Gijobject =  poses[:, jj] * objectposes[:, jj].inv() * objectposes[:, ii] * poses[:, ii].inv()#cjTw * wToj * oiTw * wTci 
    Gjj = poses[:, jj] * objectposes[:, jj].inv()#2,12,1
    Gijobject.data[:, ii == jj] = torch.as_tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype = Gij.data.dtype,device=Gij.device) 

    #X1, with object X2, without object
    X1, J1 = dyactp(Gij, X0, Gijobject = Gijobject, objectmask = validobjectmask, fullmask = fullmask, jacobian = Jacobian, batch = batch)

    # project (pinhole)
    x1, Jp1 = proj(X1, intrinsics[:, jj], jacobian=Jacobian, return_depth=return_depth)

    # exclude points too close to camera
    # valid = ((X1[..., 2] > MIN_DEPTH) & (X1[...,3] > 0.0) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if Jacobian:
        # Ji transforms according to dual adjoint

        Jcj = torch.matmul(Jp1, J1)#2,12,30,101,2,6
        Jci = -Gij[:, :, None, None, None].adjT(Jcj)

        Jcoi = -Gijobject[:, :, None, None, None].adjT(Jcj)

        Joi = Gjj[:, :, None, None, None].adjT(Jcj)
        Joj = -Joi

        Jdof = torch.zeros(6,3, device = Joi.device)
        Jdof[0,0] = Jdof[2,1] = Jdof[4,2] = 1

        # Joi = Joi*validobjectmask[..., None, None]
        # Joj = Joj*validobjectmask[..., None, None]

        Joi = torch.matmul(Joi, Jdof)*validobjectmask[..., None, None]
        Joj = torch.matmul(Joj, Jdof)*validobjectmask[..., None, None]

        if batch:
            Jci = Jcoi*validobjectmask[..., None, None]+ Jci*(1-validobjectmask[..., None, None])
            Jz = (Gijobject[:, :, None, None] * Jz) *validobjectmask[..., None] + (Gij[:, :, None, None] * Jz) * (1 - validobjectmask[..., None])

        else:
            Jci = torch.sum(Jcoi*validobjectmask[..., None, None], dim=0, keepdim=True) + Jci*(1-fullmask[..., None, None])
            Jz = torch.sum((Gijobject[:, :, None, None] * Jz) *validobjectmask[..., None], dim = 0, keepdim=True) + (Gij[:, :, None, None] * Jz) * (1 - fullmask[..., None])
            # Jz = torch.cat(((Gij[:, :, None, None] * Jz) * (1 - fullmask[..., None]), (Gijobject[:, :, None, None] * Jz) *validobjectmask[..., None]), dim=0)

        # Jz = torch.matmul(Jp1, Jz.unsqueeze(-1))
        # Jb = Jz.clone()
        # Ja = Jz*midasdisps[:,ii,..., None, None]
        # Jb = -Jz*depths[:,ii,..., None, None]*depths[:,ii,..., None, None]
        # Ja = Jb/midasdisps[:,ii,..., None, None]

        return x1, valid, (Jci, Jcj, Joi, Joj)

    return x1, valid

def icp_residual(flow, images, poses, depths, intrinsics, ii, jj, validmask, objectposes, objectmask, jacobian=True):
    # warped_depth = warp(depths, images, flow, ii, jj)
    B, _, ht,wd = depths.shape

    Xi, _ = iproj(depths[:, ii], intrinsics[:, ii], jacobian = True, batch_grid= None)#1,22,30,101,4
    Xj, _ = iproj(depths[:, jj], intrinsics[:, jj], jacobian = True, batch_grid= None)#1,22,30,101,4

    Xi = Xi/(Xi[..., 3].unsqueeze(-1))
    Xj = Xj/(Xj[..., 3].unsqueeze(-1))

    warped_Xi= warp(Xj, images, flow, ii, jj)

    Gij = poses[:, ii] * objectposes[:,ii].inv() * objectposes[:,jj] * poses[:, jj].inv()
    Xj = Gij[:, :, None, None] * warped_Xi

    x1, _ = proj(Xj, intrinsics[:,ii], return_depth=True)
    di = depths[:,ii]*objectmask[:,ii]
    dj = x1[...,2]*objectmask[:,ii]
    valid = ((x1[...,0]>0) & (x1[...,0]<wd) & (x1[...,1]>0) & (x1[...,1]<ht) & ((dj-di)<3.0)).float()
    # for i in range(dj.shape[1]):
    #     write_depth('result/warp/'+str(i)+'_depth.png', di[0,i].cpu().numpy(), False)
    #     write_depth('result/warp/'+str(i)+'_warpdepth.png', dj[0,i].cpu().numpy(), False)

    normal = get_surface_normal_by_depth(1.0/depths[:, ii], intrinsics)#1,22,30,101,4
    residual = ((normal*(Xi-Xj)).sum(dim=-1))*objectmask[:,ii]
    # for i in range(residual.shape[1]):
    #     fig = plt.figure()
    #     plt.imshow(residual[0,i].cpu().numpy(), extent=[0, 404, 0, 120], cmap='RdGy')
    #     plt.colorbar()
    #     fig.savefig('./result/warp/' + str(i)+'_residual.png')
    #     plt.close(fig)
    
    if jacobian:
        
        Gii = poses[:, ii] * objectposes[:, ii].inv()#2,12,1
        Jn = normal.unsqueeze(-2)

        X, Y, Z, d = Xj.unbind(dim=-1)#2,12,30,101,4
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        J1 = torch.stack([
            d,  o,  o,  o,  Z, -Y,
            o,  d,  o, -Z,  o,  X,
            o,  o,  d,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(B, N, H, W, 4, 6)

        Jcj = torch.matmul(Jn, J1)#2,12,30,101,2,6

        Joj = Gii[:, :, None, None, None].adjT(Jcj)
        Joi = -Joj

        Ji = torch.matmul(Jn, J1)
        Jj = -Gij[:, :, None, None, None].adjT(Ji)
        
        Jdof = torch.zeros(6,3, device = Joi.device)
        Jdof[0,0] = Jdof[2,1] = Jdof[4,2] = 1

        Joi = torch.matmul(Joi, Jdof)
        Joj = torch.matmul(Joj, Jdof)

    return residual, valid, Joi, Joj

def warp(depths, images, flow, ii, jj):

    B, N, ht, wd, D = depths.shape
    images = images[:,:,:,3::8,3::8]

    grid_x = flow[..., 0]/(wd-1)
    grid_y = flow[..., 1]/(ht-1)
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, ht, wd, 2)
    grid = grid * 2 - 1

    # for i in range(ii.shape[0]):
    #     cv2.imwrite('result/warp/'+str(i)+'_image.png', images[:,ii][0,i].permute(1,2,0).cpu().numpy())
        # write_depth('result/warp/'+str(i)+'_depth.png', depths[:,ii][0,i].cpu().numpy(), False)

    depths = depths.permute(0,1,4,2,3).view(B*N, D, ht, wd)
    images = images[0, jj]

    warped_depths = F.grid_sample(
            depths, grid, mode='nearest', padding_mode="border", align_corners=False)
    warped_images = F.grid_sample(
            images, grid, mode='nearest', padding_mode="border", align_corners=False)

    warped_depths = warped_depths.permute(0,2,3,1).view(B,N,ht,wd,D)
    # for i in range(warped_depths.shape[1]):
    #     cv2.imwrite('result/warp/'+str(i)+'warpimage.png', warped_images[i].permute(1,2,0).cpu().numpy())
        # write_depth('result/warp/'+str(i)+'_warpdepth.png', warped_depths[0,i].cpu().numpy(), False)
    return warped_depths

def get_surface_normal_by_depth(depth, K):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    fx, fy = K[0,0,0], K[0,0,1]

    b, c, d = torch.gradient(depth[0])
    # b1, c1, d1 = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = c * du_dx
    dz_dy = d * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = torch.stack((-dz_dx, -dz_dy, torch.ones_like(depth)), dim=-1)
    # normalize to unit vector
    normal_unit = normal_cross / torch.linalg.norm(normal_cross, dim=-1, keepdims=True)
    normal_unit[~torch.isfinite(normal_unit).all(4)] = torch.tensor([0, 0, 1], dtype=torch.float, device = normal_unit.device)

    # vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    # for i in range(depth.shape[1]):
    #     cv2.imwrite('result/warp/{}_normal.png'.format(i), vis_normal(normal_unit[0,i].cpu().numpy()))

    normal = torch.cat((normal_unit, torch.zeros_like(depth).unsqueeze(-1)), dim = -1)
    return normal


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
