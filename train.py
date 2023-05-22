from pickle import NONE
from subprocess import check_output
import sys
sys.path.append('droid_slam')

import cv2
import numpy as np
from collections import OrderedDict

import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.graph_utils import build_frame_graph
from data_readers.rgbd_utils import write_depth

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

def load_weights(model, checkpoint):
    """ load trained model weights """

    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in checkpoint['state'].items()])

    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

    model.load_state_dict(state_dict, strict = True)
    return model

def step(model, item, mode, logger, args, gpu):

    images, poses, disps, midasdisps, intrinsics, a, b = [x.to('cuda') for x in item]
    Ps = SE3(poses).inv()
    Gs = SE3.IdentityLike(Ps)
    N = disps.shape[1]

    if np.random.rand() < 0.5:
        graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)

    else:
        graph = OrderedDict()
        for i in range(N):
            graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]

    Gs.data[:,0] = Ps.data[:,0].clone()
    Gs.data[:,1:] = Ps.data[:,[1]].clone()
    #droid
    disp0 = torch.ones_like(disps[:,:,3::8,3::8])
    #midas
    # disp0 = midasdisps[:,:,3::8,3::8]
    intrinsics0 = intrinsics / 8.0

    # vis_image = images[0].permute(0,2,3,1).cpu().numpy()
    # for i in range(disps.shape[1]):
    #     cv2.imwrite('result/val/image'+str(i)+'.png', vis_image[i])
    #     write_depth('result/val/midas'+str(i), midasdisps[0,i].cpu().numpy(), False)
    #     write_depth('result/val/gt'+str(i), disps[0,i].cpu().numpy(), False)

    r = 0
    while r < args.restart_prob:
        r = np.random.rand()

        if mode == 'val':
            
            with torch.no_grad():
                poses_est, disps_est, residuals = model(Gs, Ps, images, disp0, disps, intrinsics0, 
                    graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

        else:
            poses_est, disps_est, residuals = model(Gs, Ps, images, disp0, disps, intrinsics0, 
                        graph, num_steps=args.iters, fixedp=2)

            geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
            res_loss, res_metrics = losses.residual_loss(residuals)
            flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

            loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss
            loss.backward()

        Gs = poses_est[-1].detach()
        disp0 = disps_est[-1][:,:,3::8,3::8].detach()

    metrics = {}
    metrics.update(geo_metrics)
    metrics.update(res_metrics)
    metrics.update(flo_metrics)

    loss = {
        'geo_loss':geo_loss.item(),
        'flow_loss':flo_loss.item(),
    }
    metrics.update(loss)

    if gpu == 0:
        if mode == 'val':
            val_metrics = {}
            for key in metrics:
                newkey = 'val_'+key
                val_metrics[newkey] = metrics[key]
            logger.push(val_metrics, True)
            metrics.clear()
        
        else:
            logger.push(metrics)

    return None

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)

    model = DroidNet()
    model.cuda()

    if args.ckpt == True:
        ckpt = sorted(os.listdir(os.path.join(args.savedir, 'ckpt')))[-1].split('.')[0]
        print('load ckpt!'+ckpt)
        checkpoint = torch.load(os.path.join(args.savedir, 'ckpt', ckpt+'.pth'))
        model = load_weights(model, checkpoint)

    model.train()
    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    # fetch dataloader
    db = dataset_factory(['tartan'], split_mode='train', datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)
    test_db = dataset_factory(['tartan'], split_mode='val', datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_db, shuffle=True, num_replicas=args.world_size, rank=gpu)
    
    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_db, batch_size=args.batch, sampler=test_sampler, num_workers=2)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    if args.ckpt == True:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        total_steps = checkpoint['steps']
        eval_steps = checkpoint['eval_steps']
    else:
        total_steps = 0
        eval_steps = 0

    logger = Logger(args.name, scheduler, total_steps, eval_steps)
    should_keep_training = True

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()
            step(model, item, 'train', logger, args, gpu)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            if total_steps%500 == 0:
                model.eval()

                for _, item in enumerate(test_loader):
                    step(model, item, 'val', logger, args, gpu)
                    eval_steps += 1
                    if eval_steps % 100 == 0:
                        model.train()
                        break

            if total_steps % 2000 == 0 and gpu == 0:
                MODEL_PATH = os.path.join(args.savedir, 'ckpt', '%s_%06d' % (args.name, total_steps)+'.pth')
                torch.save({'state':model.state_dict(), 
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'steps': total_steps,
                'eval_steps': eval_steps,
                }, MODEL_PATH)
                
            if total_steps >= args.steps:
                should_keep_training = False
                break

    dist.destroy_process_group()
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='droid', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore', default = False)
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='/storage/user/lud/lud/dataset/tartanair', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=2)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    args = parser.parse_args()

    args.world_size = args.gpus
    args.savedir = os.path.join('checkpoints', args.name)
    print(args)

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
        os.mkdir(os.path.join(args.savedir, 'ckpt'))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

