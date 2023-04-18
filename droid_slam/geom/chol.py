import torch
import torch.nn.functional as F
import geom.projective_ops as pops

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

# def block_solve(H, b, ep=0.0001, lm=0.0001):
#     """ solve normal equations """
#     B, N, _, D, _ = H.shape
#     H = H.permute(0,1,3,2,4).reshape(B, N*D, N*D)

#     I = torch.eye(N*D).to(H.device)
#     H = H + (ep + lm*H) * I

#     b = b.reshape(B, N*D, 1)

#     # x = CholeskySolver.apply(H,b)
#     x = torch.linalg.solve(H, b)
#     return x.reshape(B, N, D)

def block_solve(H, b, ep=0.0001, lm=0.0001):
    """ solve normal equations """
    _, K, _ = H.shape
    # H = H.permute(0,1,3,2,4).reshape(B, N*D, N*D)

    I = torch.eye(K).to(H.device)
    H = H + (ep + lm*H) * I

    # b = b.reshape(B, N*D, 1)

    # x = CholeskySolver.apply(H,b)
    x = torch.linalg.solve(H, b)
    return x


# def schur_solve(H, E, C, v, w, ep=0.0001, lm=0.0001, sless=False):
#     """ solve using shur complement """
    
#     B, P, M, D, HW = E.shape #1,5,7,6,30*101
#     H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
#     E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)#1,5,6,7,30*101->1,30,7*30*101
#     Q = (1.0 / C).view(B, M*HW, 1)

#     # damping
#     I = torch.eye(P*D).to(H.device)
#     H = H + (ep + lm*H) * I
    
#     v = v.reshape(B, P*D, 1)
#     w = w.reshape(B, M*HW, 1)

#     Et = E.transpose(1,2)
#     S = H - torch.matmul(E, Q*Et)
#     v = v - torch.matmul(E, Q*w)

#     # dx = CholeskySolver.apply(S, v)
#     try:
#         dx = torch.linalg.solve(S, v)
#     except Exception as e:
#         print(e)
#         dx = torch.zeros_like(v)

#     if sless:
#         return dx.reshape(B, P, D)

#     dz = Q * (w - Et @ dx)    
#     dx = dx.reshape(B, P, D)
#     dz = dz.reshape(B, M, HW)

#     return dx, dz

def schur_solve(H, E, C, v, w, ep=0.0001, lm=0.0001, sless=False):
    """ solve using shur complement """
    
    B, K, M, HW = E.shape #1,5,7,6,30*101
    # H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    E = E.reshape(B, K, M*HW)#1,5,6,7,30*101->1,30,7*30*101
    Q = 1.0 / C

    # damping
    I = torch.eye(K).to(H.device)
    H = H + (ep + lm*H) * I
    
    # v = v.reshape(B, P*D, 1)
    # w = w.reshape(B, M*HW, 1)

    Et = E.transpose(1,2)
    S = H - torch.matmul(E, Q*Et)
    v = v - torch.matmul(E, Q*w)

    # dx = CholeskySolver.apply(S, v)
    try:
        dx = torch.linalg.solve(S, v)
    except Exception as e:
        print(e)
        dx = torch.zeros_like(v)

    if sless:
        return dx

    dz = Q * (w - Et @ dx)    
    dx = dx
    dz = dz.reshape(B, M, HW)

    return dx, dz