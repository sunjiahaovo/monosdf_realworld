import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import time

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth

import torch.distributed as dist

from training.common import *
from torch.autograd import Variable
import copy
import imageio
import copy
import threading

class MonoSDFTrainRunner_uncer():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        print(kwargs['conf'])
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        print("render_only = ", kwargs['render_only'])


        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if kwargs['render_only']:
                self.plots_dir = "../exps/replica_im8_grids_1/2022_11_30_19_02_06/plots"
        elif self.GPU_INDEX == 0:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
            
            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))


        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
            self.nepochs = int(self.max_total_iters / self.ds_len)
            print('RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           collate_fn=self.train_dataset.collate_fn,
                                                           num_workers=0
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        self.Grid_MLP = self.model.Grid_MLP
        self.model = self.model.to(device)

        self.model2 = self.conf.get_config('model')
        self.model2 = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        self.model2 = self.model2.to(device2)
        

        ## 加载位姿
        self.is_gt_pose = self.conf.get_float('train.is_gt_pose')
        if self.is_gt_pose:
            self.pose_all_est = self.train_dataset.pose_all_gt
        else:
            self.pose_all_est = self.train_dataset.pose_all
        self.pose_all_gt = self.train_dataset.pose_all_gt
        self.pose_all_init = copy.deepcopy(self.pose_all_est)


        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        self.BA_cam_lr = self.conf.get_float('train.BA_cam_lr')
        self.BA = self.conf.get_float('train.BA')
        self.BA_cam_size = int(self.conf.get_float('train.BA_cam_size'))
       
        self.grid_para_list = list(self.model.implicit_network.grid_parameters())
        self.net_para_list = list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters())
        self.density_para_list = list(self.model.density.parameters())
        camera_tensor = torch.tensor([1.0,0.,0.,0.,0.,0.,0.])
        self.camera_tensor_list = [Variable(camera_tensor.to(device), requires_grad=True)]

        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': self.grid_para_list, 'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': self.net_para_list,'lr': self.lr},
                {'name': 'density', 'params': self.density_para_list, 'lr': self.lr},
                {'name': 'camera', 'params': self.camera_tensor_list, 'lr': 0},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))


        if kwargs['render_only']:
            ckpt = '../exps/replica_im8_grids_1/2022_11_30_19_02_06/checkpoints/ModelParameters/30000.pth'
            print(ckpt)
            
            if not os.path.exists(ckpt):
                print('the ckpt path does not exists!!')
                return

            ckpt= torch.load(ckpt)['model_state_dict']
            self.model.load_state_dict(ckpt)
 

        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            # self.start_epoch = saved_model_state['epoch']
            self.start_epoch = 50

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).to(device)



        if kwargs['render_only']:
            # Replica
            # for n in range(0,100,10):
            #     self.render_test(160,data_type="test_all",index = n)
            # Unity
            for n in range(0,120,30):
                self.render_test(30000,data_type="test_all",index = n)

        #    ## L435
        #    for n in range(0,36,9):
        #         self.render_test(100,data_type="test_all",index = n)
        #    self.render_test(100,data_type="test_all",index = 80)
        
        

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):
        print("training...")
        epoch = 0
        self.iter_step = 0
        self.uncer_cal = True
        # self.save_checkpoints(self.iter_step)

        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))
        
        t_start = time.time()

        ## online 加载数据和重建
        n_index = 0
        ts = time.time()
        self.train_dataset.change_sampling_idx(-1)
        while(n_index < self.train_dataset.n_images and False):
            p = np.random.uniform(0,1)
            if p < 0.5 or n_index==0:
                n = n_index
            else:
                n = np.random.randint(n_index,size=1)[0]
            # print(self.iter_step,n)
            self.train_dataset.change_sampling_idx(self.num_pixels)
            indices, model_input, ground_truth = self.train_dataset.get_testdata(n,data_type = "train")
            index = indices.cpu().numpy()[0]
            model_input["intrinsics"] = model_input["intrinsics"].to(device)
            model_input["uv"] = model_input["uv"].to(device)
            c2w = self.pose_all_est[index]
            model_input['pose'] = c2w.to(device)[None,:,:]
            
            # print("t1_0 =",time.time()-ts)
            model_outputs = self.model(model_input, indices)
            # print("t1 =",time.time()-ts)

            loss_output = self.loss(model_outputs, ground_truth,uncer_cal=self.uncer_cal)
            loss = loss_output['loss']
            # print("t2 =",time.time()-ts)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            # print("t3 =",time.time()-ts)
       

            self.iter_step +=1

            if self.iter_step%50==0 and self.iter_step>0:
                # print(self.GPU_INDEX)
                if self.GPU_INDEX == 0 and self.iter_step%1000==0:
                 
                    self.model.eval()
                    self.model2.load_state_dict(self.model.state_dict())
                    self.render_test(self.iter_step,"train", n_index,device2)
                    # threading.Thread(target=self.render_test,args=(self.iter_step,"train", n_index,device2)).start()
                    # for n in range(0,120,30):
                    #     threading.Thread(target=self.render_test,args=(self.iter_step,"test_all", n_index,device2)).start()
                    self.save_checkpoints(self.iter_step)

                    self.model.train()
                n_index +=1
                # print(n_index)

            
            if self.GPU_INDEX == 0:
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],ground_truth['rgb'].to(device).reshape(-1,3))
                if self.iter_step % 10 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7},smooth_loss = {8}, depth_loss = {9}, normal_l1 = {10}, normal_cos = {11}, psnr = {12},uncer = {13}, bete={14}, alpha={15},speed={16}'
                            .format(self.expname, self.timestamp, epoch, self.iter_step, 50* self.train_dataset.n_images, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    loss_output['smooth_loss'].item(),
                                    loss_output['depth_loss'].item(),
                                    loss_output['normal_l1'].item(), 
                                    loss_output['normal_cos'].item(),
                                    psnr.item(),
                                    loss_output['uncer'].item(),
                                    self.model.density.get_beta().item(),
                                    1. / self.model.density.get_beta().item(),
                                    (self.iter_step+1)/(time.time()-ts)))
                # prinst("time = ", time.time()-t0)
                logfolder = os.path.join(self.plots_dir, 'logs')
                pose_loss = 0.0
                data = np.array([[self.iter_step,time.time()-ts,psnr.item(),loss.item(),loss_output['rgb_loss'].item(),loss_output['eikonal_loss'].item(),\
                                loss_output['smooth_loss'].item(),loss_output['depth_loss'].item(), loss_output['normal_l1'].item(), loss_output['normal_cos'].item(),\
                                pose_loss,loss_output['uncer'].item(),\
                                self.model.density.get_beta().item(),1. / self.model.density.get_beta().item()]])

                with open(f'{logfolder}/loss.txt', "ab") as f:
                    np.savetxt(f, data)
                            

        # self.uncer_cal = False  
        ## 数据加载完成，进一步优化
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0 :
                self.model.eval()

                self.model2.load_state_dict(self.model.state_dict())
                self.render_test(self.iter_step,"train", 60,device2)
                # threading.Thread(target=self.render_test,args=(self.iter_step,"train", 60,device2)).start()
            
                # 清除缓存
                for i in range(1):
                    torch.cuda.empty_cache()
                    time.sleep(0.5)

                self.model.train()
            
            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(self.iter_step)
                logfolder = os.path.join(self.plots_dir, 'logs')
                with open(f'{logfolder}/{epoch}_pose.txt', "ab") as f:
                    for pose_index in range(len(self.pose_all_est)):
                        np.savetxt(f, self.pose_all_est[pose_index].numpy())

            self.train_dataset.change_sampling_idx(self.num_pixels)

            t0 = time.time()
            num = 1
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                t_init = time.time()
                index = indices.cpu().numpy()[0]
                c2w = self.pose_all_est[index]

                if self.BA and index > 0:
                    optimize_frame = self.keyframe_selection_overlap(index,n=self.BA_cam_size)
                    self.camera_tensor_list = []   
                    self.gt_camera_tensor_list = []
                    self.init_camera_tensor_list = []
                    for frame in optimize_frame:
                        # print(frame)
                        c2w = self.pose_all_est[frame]
                        gt_c2w = self.pose_all_gt[frame]
                        init_c2w = self.pose_all_init[frame]
                        camera_tensor = get_tensor_from_camera(c2w.clone())
                        camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                        self.camera_tensor_list.append(camera_tensor)
                        gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                        self.gt_camera_tensor_list.append(gt_camera_tensor)
                        init_camera_tensor = get_tensor_from_camera(init_c2w)
                        self.init_camera_tensor_list.append(init_camera_tensor)
                    
                    initial_loss_camera_tensor = torch.abs(self.gt_camera_tensor_list[0].clone().to(device)-self.camera_tensor_list[0]).mean().item()
                    num = 5
                    if self.Grid_MLP:
                        self.optimizer = torch.optim.Adam([
                            {'name': 'encoding', 'params': self.grid_para_list, 'lr': self.lr * self.lr_factor_for_grid},
                            {'name': 'net', 'params': self.net_para_list,'lr': self.lr},
                            {'name': 'density', 'params': self.density_para_list, 'lr': self.lr},
                            {'name': 'camera_q', 'params': self.camera_tensor_list[0:4], 'lr': 0.3*self.BA_cam_lr},
                            {'name': 'camera_t', 'params': self.camera_tensor_list[4:], 'lr': self.BA_cam_lr},
                        ],betas=(0.9, 0.99), eps=1e-15)
                from torch.optim.lr_scheduler import StepLR
                scheduler = StepLR(self.optimizer, step_size=5, gamma=0.8)
            

                pose_loss = 0
                for k in range(num):
                    self.optimizer.zero_grad()
                    if self.BA and index > 0:
                        camera_tensor_id = 0
                        loss = 0
                        for frame in optimize_frame:
                            # print(frame)
                            self.train_dataset.change_sampling_idx(self.num_pixels)
                            indices, model_input, ground_truth = self.train_dataset.get_testdata(frame,data_type = "train")
                            model_input["intrinsics"] = model_input["intrinsics"].to(device)
                            model_input["uv"] = model_input["uv"].to(device)
                            c2w = get_camera_from_tensor(self.camera_tensor_list[camera_tensor_id])
                            model_input['pose'] = c2w[None,:,:]
                            model_outputs = self.model(model_input, indices)
                            loss_output = self.loss(model_outputs, ground_truth,uncer_cal=self.uncer_cal)
                            loss = loss + loss_output['loss']/self.BA_cam_size
                            camera_tensor_id += 1

                        # pose_loss = torch.mean((init_camera_tensor.to(device)-camera_tensor) ** 2)*0.
                        pose_loss = torch.mean(torch.abs(init_camera_tensor.to(device)-camera_tensor))
                        loss = loss +  pose_loss*0.0
                    else:
                        model_input["intrinsics"] = model_input["intrinsics"].to(device)
                        model_input['pose'] = c2w.to(device)[None,:,:]
                        model_input["uv"] = model_input["uv"].to(device)
                        # print(time.time()-t0)
                        model_outputs = self.model(model_input, indices)
                        # print(time.time()-t0)
                        loss_output = self.loss(model_outputs, ground_truth,uncer_cal=self.uncer_cal)
                        loss = loss_output['loss']

                    loss.backward()
                    self.optimizer.step()
                    scheduler.step()
                    # print(time.time()-t0)
                    
                    if self.BA and index > 0:
                        ## 位姿赋值 
                        camera_tensor_id = 0
                        for frame in optimize_frame:
                            c2w = get_camera_from_tensor(self.camera_tensor_list[camera_tensor_id].detach()).cpu()
                            self.pose_all_est[frame] =c2w
                            camera_tensor_id += 1
                        ## log
                        loss_camera_tensor = torch.abs(self.gt_camera_tensor_list[0].clone().to(device)-self.camera_tensor_list[0].detach()).mean().item()
                        print([epoch],data_index,"index = ",index,"pose_loss = ",pose_loss.item(),"loss = ",loss.item()) 
                        print(f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')  

                
                    psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                            ground_truth['rgb'].to(device).reshape(-1,3))
                    
                    self.iter_step += 1                
                    
                    if self.GPU_INDEX == 0:
                        if self.iter_step % 100 == 0:
                            print(
                                '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7},smooth_loss = {8}, depth_loss = {9}, normal_l1 = {10}, normal_cos = {11}, psnr = {12},uncer = {13}, bete={14}, alpha={15},speed={16}'
                                    .format(self.expname, self.timestamp, epoch, self.iter_step, 50* self.train_dataset.n_images, loss.item(),
                                            loss_output['rgb_loss'].item(),
                                            loss_output['eikonal_loss'].item(),
                                            loss_output['smooth_loss'].item(),
                                            loss_output['depth_loss'].item(),
                                            loss_output['normal_l1'].item(), 
                                            loss_output['normal_cos'].item(),
                                            psnr.item(),
                                            loss_output['uncer'].item(),
                                            self.model.density.get_beta().item(),
                                            1. / self.model.density.get_beta().item(),
                                            (data_index+1)/(time.time()-t0)))
                        # prinst("time = ", time.time()-t0)
                        logfolder = os.path.join(self.plots_dir, 'logs')
                        if self.BA:
                            if index > 0:
                                pose_loss = pose_loss.item()
                            data = np.array([[self.iter_step,time.time()-t0,psnr.item(),loss.item(),loss_output['rgb_loss'].item(),loss_output['eikonal_loss'].item(),\
                                            loss_output['smooth_loss'].item(),loss_output['depth_loss'].item(), loss_output['normal_l1'].item(), loss_output['normal_cos'].item(),\
                                            pose_loss,loss_output['uncer'].item(),\
                                            self.model.density.get_beta().item(),1. / self.model.density.get_beta().item()]])
                        else:
                            data = np.array([[self.iter_step,time.time()-t0,psnr.item(),loss.item(),loss_output['rgb_loss'].item(),loss_output['eikonal_loss'].item(),\
                                            loss_output['smooth_loss'].item(),loss_output['depth_loss'].item(), loss_output['normal_l1'].item(), loss_output['normal_cos'].item(),\
                                            pose_loss,loss_output['uncer'].item(),\
                                            self.model.density.get_beta().item(),1. / self.model.density.get_beta().item()]])

                        with open(f'{logfolder}/loss.txt', "ab") as f:
                            np.savetxt(f, data)
                        

                    self.train_dataset.change_sampling_idx(self.num_pixels)
                    self.scheduler.step()
            
                if self.BA and index > 0:
                    ## log
                    loss_camera_tensor = torch.abs(self.gt_camera_tensor_list[0].clone().to(device)-self.camera_tensor_list[0].detach()).mean().item()
                    print([epoch],data_index,"index = ",index,"pose_loss = ",pose_loss,"loss = ",loss.item()) 
                    print(f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')  
            self.BA_cam_lr = 0.9*self.BA_cam_lr
            self.lr = 0.9*self.lr
        self.save_checkpoints(self.iter_step)

    

    def keyframe_selection_overlap(self,index,n=4):
        c2w_loss_list = []
        c2w_cur_tensor = get_tensor_from_camera(self.pose_all_est[index]).numpy()
        for i in range(len(self.pose_all_est)):
            c2w_tensor = get_tensor_from_camera(self.pose_all_est[i]).numpy()
            c2w_loss_q = np.mean(abs(c2w_cur_tensor-c2w_tensor)[0:4])
            c2w_loss_t = np.mean(abs(c2w_cur_tensor-c2w_tensor)[4:])
            c2w_loss = c2w_loss_q*5 + c2w_loss_t
            c2w_loss_list.append([i,c2w_loss])
            
        c2w_loss_list = np.array(c2w_loss_list)
        c2w_loss_list = c2w_loss_list[np.lexsort(c2w_loss_list.T)]
        # print(c2w_loss_list)
        selected_keyframe_list = c2w_loss_list[0:n,0].astype(np.int32).tolist()
        print(selected_keyframe_list)
        ## 如果是第一帧,则将当前帧加入优化池
        idx = 0
        for frame in selected_keyframe_list:
            if frame == 0:
                selected_keyframe_list[idx] = c2w_loss_list[n,0].astype(np.int32)
            idx += 1
        return selected_keyframe_list


    def render_test(self,epoch,data_type = "train",index = 0,device=device):
            ts = time.time()

            self.train_dataset.change_sampling_idx(-1)
            indices, model_input, ground_truth = self.train_dataset.get_testdata(index,data_type = data_type)
            print(indices)
            model_input["intrinsics"] = model_input["intrinsics"].to(device)
            model_input["uv"] = model_input["uv"].to(device)
            model_input['pose'] = model_input['pose'].to(device)
            
            split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels,device=device)
            # print("t1 =",time.time()-ts)
            res = []
            for s in tqdm(split):
                # print("t1-1 =",time.time()-ts)
                out = self.model2(s, indices,device=device)
                # print("t1-2 =",time.time()-ts)
                d = {'rgb_values': out['rgb_values'].detach(),
                        'normal_map': out['normal_map'].detach(),
                        'depth_values': out['depth_values'].detach()}
                if 'rgb_un_values' in out:
                    d['rgb_un_values'] = out['rgb_un_values'].detach()
                
                if 'uncer_map' in out:
                    d['uncer_map'] = out['uncer_map'].detach()
                res.append(d)
            # torch.cuda.empty_cache()
            print("t2 =",time.time()-ts)
            # print(res)
            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            # print(model_outputs)
            plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'],device=device)

            uncer_map = plot_data["uncer_map"].reshape(384,384).cpu().numpy()
            # print(np.max(uncer_map),np.min(uncer_map))
            # print(uncer_map)
            print(uncer_map)
            uncer_data = 1/uncer_map
            rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
            

            plt.plot(self.model2.implicit_network,
                    indices,
                    plot_data,
                    self.plots_dir,
                    epoch,
                    self.img_res,
                    **self.plot_conf,
                    data_type = data_type,
                    device = device
                    )

            if data_type == "test_all":
                imageio.imwrite('{0}/test_all/uncer_{1}_{2}.png'.format(self.plots_dir, epoch, indices[0]),rgb8)

            torch.cuda.empty_cache()
            time.sleep(0.5)


    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt,device=device):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        # print(depth_map)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        uncer_map = model_outputs['uncer_map']

        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs,device=device)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs,device=device)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'uncer_map': uncer_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs,device=device):
        self.backproject = self.backproject.to(device)
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()


    def copy_model(self):

        uncermodel = copy.deepcopy(self.model)
        uncermodel = uncermodel.to(device2)
        uncermodel.copy(device2)
        return uncermodel

