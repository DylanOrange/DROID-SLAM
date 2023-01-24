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
from geom.graph_utils import build_frame_graph

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

def train(args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    # setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet()
    model.cuda()
    model.train()

    # model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    if args.ckpt is not None:
        model = load_weights(model, args.ckpt)

    # fetch dataloader
    db = dataset_factory(['vkitti2'], datapath=args.datapath, n_frames=args.n_frames, crop_size=[240, 808], fmin=args.fmin, fmax=args.fmax)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, shuffle = True)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, objectposes, objectmasks, disps, cropmasks, cropdisps, fullmasks, fulldisps, quanmask, intrinsics, trackinfo = item

            if torch.mean(cropdisps) < 0.0:
                continue
            # convert poses w2c -> c2w
            Ps = SE3(poses)#这里暂时使用w2c
            ObjectPs = SE3(objectposes[0]).inv()
            
            Gs = SE3.IdentityLike(Ps)
            ObjectGs = SE3.IdentityLike(ObjectPs)

            # randomize frame graph
            # if np.random.rand() < 0.5:
            #     graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            
            # else:
            graph = OrderedDict()
            for i in range(N):
                graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            for n in range(len(trackinfo['trackid'][0])):
                ObjectGs.data[n, trackinfo['apperance'][n][0][0]] = ObjectPs.data[n, trackinfo['apperance'][n][0][0]].clone()
                ObjectGs.data[n, trackinfo['apperance'][n][0][1:]] = ObjectPs.data[n, trackinfo['apperance'][n][0][1]].clone()
            disp0 = torch.ones_like(disps[:,:,3::8,3::8])
            # disp0 = disps[:,:,3::8,3::8]
            # perform random restarts

            r = 0
            while r < args.restart_prob:
                r = rng.random()
                
                # intrinsics0 = intrinsics / 8.0
                poses_est, objectposes_est, disps_est,  static_residual_list, dyna_residual_list, flow_low_list, flow_high_list = model(Gs, Ps, ObjectGs, ObjectPs, images, objectmasks, disp0, disps[:,:,3::8,3::8], cropmasks, cropdisps, fullmasks, fulldisps, intrinsics, trackinfo,
                    graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False, object = False, trackinfo = None)
                Obgeo_loss, Obgeo_metrics = losses.geodesic_loss(ObjectPs, objectposes_est, graph, do_scale=False, object = True, trackinfo = trackinfo)
                static_resi_loss, static_resid_metrics = losses.residual_loss(static_residual_list)
                dyna_resi_loss, dyna_resid_metrics = losses.residual_loss(dyna_residual_list)
                error_low, error_high, error_st, flow_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, ObjectPs, objectposes_est, objectmasks, cropmasks, cropdisps, fullmasks, fulldisps, quanmask, trackinfo, intrinsics, graph, flow_low_list, flow_high_list)

                loss =  args.w1*geo_loss + args.w1 * Obgeo_loss + args.w2 * static_resi_loss + args.w2 * dyna_resi_loss + args.w3 * error_low + args.w3 * error_high + args.w3 * error_st
                loss.backward()

                Gs = poses_est[-1].detach()
                ObjectGs = objectposes_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::4,3::4].detach()

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(Obgeo_metrics)
            metrics.update(static_resid_metrics)
            metrics.update(dyna_resid_metrics)
            metrics.update(flow_metrics)
            loss = {
                'geo_loss':geo_loss.item(),
                'Obgeo_loss':Obgeo_loss.item(),
                'error_low':error_low.item(),
                'error_high':error_high.item(),
                'error_st':error_st.item(),           
                'static_resi_loss':static_resi_loss.item(),
                'dyna_resi_loss':dyna_resi_loss.item(),
            }
            metrics.update(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1

            # if gpu == 0:
            logger.push(metrics)

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
    parser.add_argument('--name', default='test', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore', default='droid.pth')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='../DeFlowSLAM/datasets/vkitti2/Scene20', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=1)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--steps', type=int, default=80000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=5)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=1)

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=80.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

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
    # os.environ['MASTER_PORT'] = '12356'
    # mp.spawn(train, nprocs=args.gpus, args=(args,))

