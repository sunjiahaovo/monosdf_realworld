import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
import copy
from scipy.spatial.transform import Rotation as R

##  unity
def get_pose(location,u,v):
    sx = np.sin(u)
    cx = np.cos(u)
    sy = np.sin(v)
    cy = np.cos(v)
    return [[cy, sy*sx, -sy*cx, location[0]],
                        [0, cx, sx, location[1]],
                        [-sy, cy*sx, -cy*cx, location[2]],
                        [0,0,0,1]]


# Real Scene
def get_pose_real(location,u,v):
    v = v/np.pi*180
    u = -u/np.pi*180 
    r = R.from_euler('ZYX', [v, u, 0], degrees=True)
    pose = np.zeros((4,4))
    pose[0:3,0:3] = r.as_matrix()
    pose[0:3,3] = np.array(location)
    pose[3,3]=1
    # print("pose = ", pose)
    return pose





# Dataset with monocular depth / L435
class SceneDatasetDNIM(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None


        self.n_images = 101
        remove_list = [67,68,69,70,71,72]

        self.intrinsics_all = []
        self.pose_all = []
        self.pose_all_test = []
        self.pose_all_gt = []
        offset_T = np.loadtxt("../postprocess/offset_pnp_v1.txt")
        print(offset_T)
        poses = []
        for i in range(self.n_images):
            if i in remove_list:
                continue
            path = os.path.join(self.instance_dir,str(i)+".txt")
            pose = np.loadtxt(path,delimiter=" ")
            # pose = np.linalg.multi_dot([np.linalg.inv(offset_T),pose,offset_T]) 
            # # Blender
            # T_b2o = np.array([[0,0,-1,0],
            #                   [-1,0,0,0],
            #                   [0,1,0,0],
            #                   [0,0,0,1]])
            # pose = np.linalg.inv(T_b2o) @  pose @ T_b2o
            poses.append(pose)

        poses = np.stack(poses)

        # deal with invalid poses
        valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
        min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
        max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)
    
        center = (min_vertices + max_vertices) / 2.
        scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
        print(center, scale)

        # we should normalized to unit cube
        scale_mat = np.eye(4).astype(np.float32)
        scale_mat[:3, 3] = -center
        scale_mat[:3 ] *= scale 
        scale_mat = np.linalg.inv(scale_mat)

        K = np.eye(4)
        # camera_intrinsic = np.array([[911.87/2,0,640/2],[0,911.87/2,360/2],[0,0,1]]).astype(np.float64)
        camera_intrinsic = np.array([[911.87*384/720,0,384/2],[0,911.87*384/720,384/2],[0,0,1]]).astype(np.float64)
        K[:3, :3] = camera_intrinsic
        print(K)
        intrinsics = copy.deepcopy(K)

        ## load pose
        for i in range(self.n_images):
            if i in remove_list:
                continue
            path = os.path.join(self.instance_dir,str(i)+".txt")
            pose = np.loadtxt(path,delimiter=" ")
            # pose = np.linalg.multi_dot([np.linalg.inv(offset_T),pose,offset_T])
            
            pose_test = copy.deepcopy(pose)

            if i < 36:
                theta = i*10/180*np.pi
                x = -2.0*np.cos(theta)
                y = 1.3*np.sin(theta)
                z = 0.7
                object_center = np.array([0,0,0.8])
                dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
                v = np.arctan2(dy,dx) 
                pose_test = get_pose_real(np.array([x,y,z]),0,v)     
            else:
                t_noise = np.array([0,0.0,0.3])
                pose_test[:3,3] = pose_test[:3,3] + t_noise
            
            # ## Blender
            # T_b2o = np.array([[0,0,-1,0],
            #                   [-1,0,0,0],
            #                   [0,1,0,0],
            #                   [0,0,0,1]])
            # pose = np.linalg.inv(T_b2o) @  pose @ T_b2o
  
  
            # # ## 写入文件 gt pose
            # np.savetxt(os.path.join(self.instance_dir,"l435_out_384",str(i)+"_gt.txt"),pose)
            # print("before ",i,pose)
            # ## 加位姿噪声
            # if i > 0:
            #     dir_noise = np.random.randn(2)*5.0
            #     t_noise = np.random.randn(3)*0.05
            #     # print(dir_noise,t_noise)
            #     r4 = R.from_euler('ZYX', [0,  dir_noise[0],  dir_noise[1]], degrees=True)
            #     rm = r4.as_matrix()
            #     # print("rm = ", rm)
            #     pose[:3,:3] = rm @ pose[:3,:3]
            #     pose[:3,3] = pose[:3,3] + t_noise
            #     print("after ",i,pose)
            # # 写入文件 noise pose
            # np.savetxt(os.path.join(self.instance_dir,"l435_out_384",str(i)+"_5_5.txt"),pose)

            # 加载noise /gt pose
            pose = np.loadtxt(os.path.join(self.instance_dir,"l435_out_384",str(i)+".txt"))
            pose_gt = np.loadtxt(os.path.join(self.instance_dir,"l435_out_384",str(i)+".txt"))
            
                      
            pose = self.covert_pose(K, scale_mat, pose)
            pose_test = self.covert_pose(K, scale_mat, pose_test)
            pose_gt = self.covert_pose(K, scale_mat, pose_gt) ##  needed while loading pose txt
            # pose_gt = copy.deepcopy(pose) ## not needed while saving pose txt
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            self.pose_all_test.append(torch.from_numpy(pose_test).float())
            self.pose_all_gt.append(torch.from_numpy(pose_gt).float())

        self.rgb_images = []
        for i in range(self.n_images):
            if i in remove_list:
                continue
            path = os.path.join(self.instance_dir,"l435_out_384",str(i)+"_om_rgb.png")
            rgb = rend_util.load_rgb(path,self.img_res)
            # print(rgb.shape)
            rgb = rgb[0:3,:,:]
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []
        self.mask_images = []
        
        normal_im = np.ones_like(rgb)
        for i in range(self.n_images):
            if i in remove_list:
                continue
            # dpath = os.path.join(self.instance_dir,str(i)+"_depth.png")
            # depth = rend_util.load_depth(dpath,self.img_res,l435=True)/1000.0
            # depth = depth[0,:,:]

            dpath = os.path.join(self.instance_dir,"l435_out_384",str(i)+"_om_depth.npy")
            depth = np.load(dpath)

            npath = os.path.join(self.instance_dir,"l435_out_384",str(i)+"_om_normal.npy")
            normal = np.load(npath)

            # print(normal.shape)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
            self.normal_images.append(torch.from_numpy(normal).float())
            
            # print(depth)
            # depth[depth>6]=0
            mask = np.ones_like(depth)
            # mask[depth<0.1]=0
            # mask[depth>6]=0
            self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float()) 
            
        self.n_images = 95


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def covert_pose(self,K, scale_mat, pose):
        # print(i)
        # print("before :", pose)
        pose = np.array(pose)
        R = copy.deepcopy(pose[:3,:3])
        pose = K @ np.linalg.inv(pose)
        P = pose @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        # print("after :", pose)
        pose[:3,:3] = R
        # print("after1 :", pose)
        return pose
    
    def get_testdata(self, idx, data_type="train"):
        
        if data_type=="train":
            pose = self.pose_all[idx]
        else:
            pose = self.pose_all_test[idx]

        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": pose
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        batch_list = [(idx,sample,ground_truth)]


        return self.collate_fn(batch_list)


