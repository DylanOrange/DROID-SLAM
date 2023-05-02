from geom.ba import dynamicBA
import os
import pickle
import torch

with open(os.path.join('result/debug/', 'debug_2500.pkl'), 'rb') as tt:
    result = pickle.load(tt, encoding='latin1')

target = result['hflow']
weight = result['hweight']
depth_valid = result['hweight']
ObjectGs = result['hogs']
Gs = result['hgs']

disps = result['hdisps']
intrinsics = result['hintrinsics']
objectmasks = result['hmask']

ii = result['ii']
jj = result['jj']
validmask = torch.ones_like(ii,dtype=torch.bool)[None]
    
for i in range(10):
    Gs, ObjectGs, disps = dynamicBA(target, weight, ObjectGs, objectmasks, trackinfo, validmask, \
                                    None, Gs, disps, intrinsics, ii, jj, fixedp=2)