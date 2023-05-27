import sys

sys.path.append('../code')
import argparse
import torch

import os
from training.monosdf_train_online import MonoSDFTrainRunner_online
import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/room_im_grids.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    #parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=True, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='2022_11_25_18_38_06', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='5000', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=1, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--render_only', type=int, default=0, help='render only for test views')

    opt = parser.parse_args()

    '''
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    '''
    gpu = opt.local_rank

    # set distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    print(opt.local_rank)
    torch.cuda.set_device(opt.local_rank)
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='10006')
    torch.distributed.init_process_group(backend='nccl', init_method=dist_init_method, world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
    torch.distributed.barrier()


    trainrunner = MonoSDFTrainRunner_online(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis,
                                    render_only= opt.render_only
                                    )
    
    if opt.render_only:
        pass
    else:
        trainrunner.run()
