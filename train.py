from email.policy import strict
from pickle import FALSE, TRUE
import sys
sys.path.append('droid_slam')

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.graph_utils import build_frame_graph, build_object_frame_graph

# network
from droid_net import DroidNet
from logger import Logger

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def load_weights(model, weights):
    """ load trained model weights """

    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])
    # state_dict = torch.load(weights)
    # for key in state_dict.keys():
    #     state_dict.update({key.split('.', 1)[1]:state_dict.pop(key)})

    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

    model.load_state_dict(state_dict, strict = False)
    return model

def step(model, item, mode, logger, skip, save, total_steps, args):

    images, poses, objectposes, objectmasks, disps, \
		highdisps, highmask, intrinsics, scale, depth_valid, high_depth_valid = [x.to('cuda') for x in item[0]]
    trackinfo = item[1]
    for key in trackinfo.keys():
        trackinfo[key] = trackinfo[key].to('cuda')

    N = disps.shape[1]

    Ps = SE3(poses)#in system we use w2c
    ObjectPs = SE3(objectposes).inv()#物体从o2w到w2o
    
    Gs = SE3.IdentityLike(Ps)
    ObjectGs = SE3.IdentityLike(ObjectPs)
	
    # if np.random.rand() < 0.5:
    #     graph = build_object_frame_graph(poses, disps, intrinsics, objectposes, objectmasks, num=args.edges)
    # else:
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]

    Gs.data[:,0] = Ps.data[:,0].clone()
    ObjectGs.data[:, 0] = ObjectPs.data[:, 0].clone()
    Gs.data[:,1:] = Ps.data[:,1].clone()
    ObjectGs.data[:, 1:] = ObjectPs.data[:, 1].clone()
    
    # dGs = Gs[:,1]*Gs[:,0].inv()
    # ds = dGs.log()/(trackinfo['frames'][:,1] - trackinfo['frames'][:,0])
    # dOGs = ObjectGs[:,1]*ObjectGs[:,0].inv()
    # dOs = dOGs.log()/(trackinfo['frames'][:,1] - trackinfo['frames'][:,0])

    # for i in range(2, N):
    #     step = trackinfo['frames'][:,i] - trackinfo['frames'][:,0]
    #     dw = ds*step
    #     dOw = dOs*step
    #     Gs[:, i] = SE3.exp(dw)*Gs[:, 0]
    #     ObjectGs[:, i] = SE3.exp(dOw)*ObjectGs[:, 0]

    # disp0 = lowmidasdepths
    # disp0 = torch.ones_like(disps)

    r = 0
    while r < args.restart_prob:
        r = np.random.rand()

        if mode == 'val':
            with torch.no_grad():
                poses_est, objectposes_est, disps_est, static_residual_list, flow_list = model(Gs, Ps, ObjectGs, ObjectPs, images, \
                                                                                               objectmasks, highmask, disps, disps, highdisps, intrinsics.clone(), trackinfo,\
                                                                                               depth_valid, high_depth_valid, save, total_steps, graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False, object = False)
                Obgeo_loss, Obgeo_metrics = losses.geodesic_loss(ObjectPs, objectposes_est, graph, do_scale=False, object = True)
                static_resi_loss, static_resid_metrics = losses.residual_loss(static_residual_list)

                error_lowflow, error_dylow, error_induced_low, error_lowdepth, error_lowdynadepth, \
                error_highflow, error_dyhigh, error_induced_high, error_highdepth, error_highdynadepth, flow_metrics, \
                = losses.flow_loss(Ps, disps, highdisps, poses_est, disps_est, ObjectPs, objectposes_est, \
                                 objectmasks, highmask, trackinfo, intrinsics, graph, flow_list, scale)
               
        else:
            poses_est, objectposes_est, disps_est, static_residual_list, flow_list = model(Gs, Ps, ObjectGs, ObjectPs, images, \
                                                                                            objectmasks, highmask, disps, disps, highdisps, intrinsics.clone(), trackinfo,\
                                                                                            depth_valid, high_depth_valid, save, total_steps, graph, num_steps=args.iters, fixedp=2)
            
            geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False, object = False)
            Obgeo_loss, Obgeo_metrics = losses.geodesic_loss(ObjectPs, objectposes_est, graph, do_scale=False, object = True)
            static_resi_loss, static_resid_metrics = losses.residual_loss(static_residual_list)

            error_lowflow, error_dylow, error_induced_low, error_lowdepth, error_lowdynadepth, \
            error_highflow, error_dyhigh, error_induced_high, error_highdepth, error_highdynadepth, flow_metrics, \
            = losses.flow_loss(Ps, disps, highdisps, poses_est, disps_est, ObjectPs, objectposes_est, \
                                objectmasks, highmask, trackinfo, intrinsics, graph, flow_list, scale)

            loss = 0.5*(args.w1 * geo_loss[0] + args.w1 * 5* Obgeo_loss[0] + \
                args.w2 * static_resi_loss[0] +\
                args.w3 * error_induced_low) +\
                args.w1 * geo_loss[1] + args.w1 * 5* Obgeo_loss[1] + \
                args.w2 * static_resi_loss[1] +\
                args.w3 * error_induced_high
            
            loss.backward()

        Gs = poses_est[1][-1].detach()
        ObjectGs = objectposes_est[1][-1].detach()
        disps = disps_est[0][-1].detach()

        if skip:
            if flow_metrics['abs_high_dyna_error'] > 1.2*flow_metrics['abs_high_error'] and Obgeo_metrics[1]['high_ob_rot_error'] > 0.5:
                print('bad optimization!')
                return True
    metrics = {}

    for index in range(len(poses_est)):
        metrics.update(geo_metrics[index])
        metrics.update(Obgeo_metrics[index])
        metrics.update(static_resid_metrics[index])
    metrics.update(flow_metrics)

    loss = {
        'geo_loss':geo_loss[0].item(),
        'Obgeo_loss':Obgeo_loss[0].item(),
        'high_geo_loss':geo_loss[1].item(),
        'high_Obgeo_loss':Obgeo_loss[1].item(),

        'error_lowflow':error_lowflow.item(),
        'error_lowdepth':error_lowdepth.item(),
        'error_induced_low':error_induced_low.item(), 
        'error_dylow':error_dylow.item(),
        'error_lowdynadepth': error_lowdynadepth.item(),

        'error_highflow':error_highflow.item(), 
        'error_highdepth':error_highdepth.item(), 
        'error_induced_high':error_induced_high.item(), 
        'error_dyhigh':error_dyhigh.item(), 
        'error_highdynadepth': error_highdynadepth.item()
    }
    metrics.update(loss)

    # if gpu ==0:
    if mode == 'val':
        val_metrics = {}
        for key in metrics:
            newkey = 'val_'+key
            val_metrics[newkey] = metrics[key]
        logger.push(val_metrics)
        metrics.clear()
    
    else:
        logger.push(metrics)

    return False

    
def train(args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    # setup_ddp(gpu, args)

    model = DroidNet()
    model.cuda()
    model.train()

    # model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    if args.ckpt is not None:
        model = load_weights(model, args.ckpt)
    
    # for param in model.parameters():
    #     print(param)

    # fetch dataloader
    db = dataset_factory(['vkitti2'], split_mode='train', datapath=args.datapath, n_frames=args.n_frames, crop_size=[240, 808], fmin=args.fmin, fmax=args.fmax, obfmin=args.obfmin, obfmax=args.obfmax)
    test_db = dataset_factory(['vkitti2'], split_mode='val', datapath=args.datapath, n_frames=args.n_frames, crop_size=[240, 808], fmin=args.fmin, fmax=args.fmax, obfmin=args.obfmin, obfmax=args.obfmax)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     db, shuffle=True, num_replicas=args.world_size, rank=gpu)
    
    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_db, shuffle=True, num_replicas=args.world_size, rank=gpu)
    
    train_loader = DataLoader(db, batch_size=args.batch, shuffle = True)
    test_loader = DataLoader(test_db, batch_size=args.batch, shuffle = True)
    
    # train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)
    # test_loader = DataLoader(test_db, batch_size=args.batch, sampler=test_sampler, num_workers=2)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    skip = False
    total_steps = 0
    save = False

    while should_keep_training:
        for _, item in enumerate(train_loader):

            optimizer.zero_grad()

            if step(model, item, 'train', logger, skip, save, total_steps, args):
                print('jump train!')
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1

            if total_steps % 500 == 0:
                ##validation
                model.eval()
                eval_steps = 0

                for _, item in enumerate(test_loader):

                    if step(model, item, 'val', logger, skip, False, total_steps, args):
                        print('jump val!')
                        continue
                    eval_steps += 1

                    if eval_steps == 60:
                        model.train()
                        break

            if total_steps>80000:
                skip = True

            if total_steps % 2000 == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    # dist.destroy_process_group()
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--name', default='scene06', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore', default='droid.pth')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='../DeFlowSLAM/datasets/vkitti2', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=0)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--steps', type=int, default=160000)
    parser.add_argument('--lr', type=float, default=0.000025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)

    parser.add_argument('--fmin', type=float, default=0.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--obfmin', type=float, default=2.0)
    parser.add_argument('--obfmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.01)

    args = parser.parse_args()

    args.world_size = args.gpus
    print(args)

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    args.world_size = args.gpus
    train(args)

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12348'
    # mp.spawn(train, nprocs=args.gpus, args=(args,))

