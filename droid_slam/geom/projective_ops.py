import torch
import torch.nn.functional as F

from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)

def batch_grid(corners, rec):
    B = corners.shape[0]
    x_list, y_list = [], []
    for i in range(B):
        y, x = torch.meshgrid(
            torch.arange(corners[i,0], corners[i,0]+rec[0]).float(),
            torch.arange(corners[i,1], corners[i,1]+rec[1]).float())
        y_list.append(y)
        x_list.append(x)
    x = torch.stack(x_list).unsqueeze(1)
    y = torch.stack(y_list).unsqueeze(1)
    return (x,y)

def crop(images, center, rec, depth = False):
    B,N,_,_,ch = images.shape
    output = torch.zeros(B, N, rec[0], rec[1], ch, dtype=images.dtype, device = images.device)
    if depth:
        output[output == 0.0] = -0.1
    for n in range(B):
        crop = images[n, :, center[n,0]:center[n,0]+rec[0], center[n,1]:center[n,1]+rec[1]]
        try:
            output[n] = crop
        except Exception as e:
            #padding
            # print(e)
            h,w = crop.shape[1:3]
            output[n,:, :h, :w] = crop
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
            torch.arange(ht).to(disps.device).float(),
            torch.arange(wd).to(disps.device).float())#2,49,85

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
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

    Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype = Gij.data.dtype, device="cuda")
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

def dyprojective_transform(poses, depths, intrinsics, ii, jj, validmask = None, objectposes = None, objectmask = None, Jacobian = False, return_depth = False, batch = False, batch_grid = None):
    """ map points from ii->jj """
    
    #输入前面都有Batch 1，除了Objectmask
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii], jacobian = Jacobian, batch_grid= batch_grid)#1,2,30,101,4

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
    X1, J1 = dyactp(Gij, X0, Gijobject = Gijobject, objectmask = validobjectmask, fullmask = fullmask, jacobian = Jacobian, batch = batch)

    # project (pinhole)
    x1, Jp1 = proj(X1, intrinsics[:, jj], jacobian=Jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X1[...,3] > 0.0) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if Jacobian:
        # Ji transforms according to dual adjoint
        Jcj = torch.matmul(Jp1, J1)#2,12,30,101,2,6
        Jci = -Gij[:, :, None, None, None].adjT(Jcj)

        Jcoi = -Gijobject[:, :, None, None, None].adjT(Jcj)

        Joi = Gjj[:, :, None, None, None].adjT(Jcj)
        Joj = -Joi

        Joi = Joi*validobjectmask[..., None, None]
        Joj = Joj*validobjectmask[..., None, None]

        if batch:
            Jci = Jcoi*validobjectmask[..., None, None]+ Jci*(1-validobjectmask[..., None, None])
            Jz = (Gijobject[:, :, None, None] * Jz) *validobjectmask[..., None] + (Gij[:, :, None, None] * Jz) * (1 - validobjectmask[..., None])

        else:
            Jci = torch.sum(Jcoi*validobjectmask[..., None, None], dim=0, keepdim=True) + Jci*(1-fullmask[..., None, None])
            Jz = torch.sum((Gijobject[:, :, None, None] * Jz) *validobjectmask[..., None], dim = 0, keepdim=True) + (Gij[:, :, None, None] * Jz) * (1 - fullmask[..., None])

        Jz = torch.matmul(Jp1, Jz.unsqueeze(-1))

        return x1, valid, (Jci, Jcj, Joi, Joj, Jz)

    return x1, valid