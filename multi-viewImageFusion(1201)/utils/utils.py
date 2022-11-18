""" utils.py
"""

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datetime import datetime as dt
import warnings
import cv2
import torch.nn as nn
import torch.nn.functional as F

name_dataparallel = torch.nn.DataParallel.__name__
log10 = np.log(10)

class Get_gradient_xy(nn.Module):
    def __init__(self):
        super(Get_gradient_xy, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]] #y 纵向
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]] #x 横向
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)

        grad = [x_v, x_h]
        return grad

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x
        
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

def gen_blend_template(patch_size = 2048): #patch_size应为4的倍数
    weightTensor = torch.zeros((1,3,patch_size, patch_size), dtype=torch.float32).cuda() 
    weight = np.zeros((patch_size, patch_size))
    coord_grid = np.indices(weight.shape)
    # print(coord_grid[0])
    # print(coord_grid[1])
    four_dis = np.zeros((4, patch_size, patch_size))
    four_dis[0] = coord_grid[0] + 1 - 0
    four_dis[1] = coord_grid[1] + 1 - 0
    four_dis[2] = patch_size - coord_grid[0]
    four_dis[3] = patch_size - coord_grid[1]
    weight = np.min(four_dis, axis=0)
    weight = weight / (patch_size / 4.0) #weight.dtype=float64
    weightTensor[0, :, :, :] = torch.Tensor(weight)
    weightTensor[:, :, patch_size//4:patch_size//4+patch_size//2, patch_size//4:patch_size//4+patch_size//2] = 1
    return weightTensor 


def compute_psnr(x , label , max_diff):
    assert max_diff in [255,1,2]
    if max_diff == 255:
        x = x.clamp( 0 , 255  )
    elif max_diff == 1:
        x = x.clamp( 0 , 1 )
    elif max_diff == 2 :
        x = x.clamp( -1 , 1 )

    mse =  (( x - label ) **2 ).mean()
    return 10*torch.log( max_diff**2 / mse ) / log10

def save_model(model,dirname,epoch):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    torch.save( model.state_dict() , '{}/{}_epoch{}.pth'.format(dirname,type(model).__name__,epoch ) )

def save_checkpoints(dirname, Best_Img_PSNR, epoch_idx, net, net_solver):
    file_path = '{}/best.pth'.format(dirname)
    print('[INFO] {} Saving checkpoint to {}, BEST_PSNR={}\n'.format(dt.now(),file_path,Best_Img_PSNR))
    checkpoint = {
        'best_epoch': epoch_idx,
        'Best_Img_PSNR': Best_Img_PSNR,
        'net_state_dict': net.state_dict(),
        'net_solver_state_dict': net_solver.state_dict(),
    }
    torch.save(checkpoint, file_path)

def save_checkpoints1(dirname, epoch_idx, net, net_solver):
    file_path = '{}/{}.pth'.format(dirname, epoch_idx)
    print('[INFO] {} Saving checkpoint to {}, epoch={}\n'.format(dt.now(), file_path, epoch_idx))
    checkpoint = {
        'epoch': epoch_idx,
        'net_state_dict': net.state_dict(),
        'net_solver_state_dict': net_solver.state_dict(),
    }
    torch.save(checkpoint, file_path)

def save_checkpoints0(dirname, GorD, Img_PSNR, epoch_idx, net, net_solver): #专门针对GAN
    file_path = '{}/{}_{}_{:.6f}.pth'.format(dirname, GorD, epoch_idx, Img_PSNR)
    print('[INFO] {} Saving checkpoint to {}, PSNR={}\n'.format(dt.now(),file_path, Img_PSNR))
    checkpoint = {
        'epoch': epoch_idx,
        'Img_PSNR': Img_PSNR,
        'net_state_dict': net.state_dict(),
        'net_solver_state_dict': net_solver.state_dict(),
    }
    torch.save(checkpoint, file_path)


def load_checkpoints(net, net_solver, dirname, epoch_idx=None, strict= True):
    p = "{}/best.pth".format(dirname)
    if os.path.exists(p):
        checkpoint = torch.load(p)
        i = checkpoint['best_epoch']
        net.load_state_dict(checkpoint['net_state_dict'])
        Best_Img_PSNR = checkpoint['Best_Img_PSNR']
        net_solver.load_state_dict(checkpoint['net_solver_state_dict'])
        print('[INFO] {0} Recover complete. Best_Img_PSNR = {1} at epoch #{2}.'.format(dt.now(), Best_Img_PSNR, i))
        return i
    else:
        return -1

def load_checkpoints0(net, net_solver, weightname, epoch_idx=None, strict= True):
    p = weightname
    if os.path.exists(p):
        checkpoint = torch.load(p)
        i = checkpoint['best_epoch']
        net.load_state_dict(checkpoint['net_state_dict'])
        Img_PSNR = checkpoint['Best_Img_PSNR']
        net_solver.load_state_dict(checkpoint['net_solver_state_dict'])
        print('[INFO] {0} Recover complete. Img_PSNR = {1} at epoch #{2}.'.format(dt.now(), Img_PSNR, i))
        return i
    else:
        return -1

def load_ckpt_from_net(net,ckpt_path):
    model_dict = net.state_dict()
    weight_all_keys = model_dict.keys() 

    weight = torch.load(ckpt_path)
    i = weight['best_epoch']
    Best_Img_PSNR = weight['Best_Img_PSNR']
    print('[INFO] {0} Recover complete. Best_Img_PSNR = {1} at epoch #{2}.'.format(dt.now(), Best_Img_PSNR, i))
    weight = weight['net_state_dict']

    for k, v in weight.items():
        if k in weight_all_keys:
            model_dict[k] = v  
        if 'upconv' not in k and 'stem' in k and 'convd' not in k:    
            k_all = k.replace('.stem.0.','.conv0.')
            k_all = k_all.replace('.stem.2.','.conv1.')
            if k_all in weight_all_keys:
                
                model_dict[k_all] = v
    net.load_state_dict(model_dict)
    return net
    
def set_requires_grad(module , b ):
    for parm in module.parameters():
        parm.requires_grad = b

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.val, self.avg)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        # torch.nn.init.constant_(m.bias, 0)