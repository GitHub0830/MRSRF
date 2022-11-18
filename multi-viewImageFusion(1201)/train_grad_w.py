import torch.nn as nn
import torch
import torchvision
import os
from tqdm import tqdm
import sys
from time import time
import math

from utils.log import TensorBoardX
from dataLoaders.dataLoader import fusionDatasetNoOcc
import config.train_config as config
from utils.utils import *
from utils.losses import *
from models.pixelFusionModel_grad_w import pixelFusionNet, warp
from models.VGG19 import VGG19forPixelFusion, VGG19
import models
import argparse

import torch.nn.functional as F

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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    tb = TensorBoardX(config_filename = "config/train_config.py", code_filename="train_grad_w.py",
                     output_path=config.OUT['path'] , sub_dir="weight")
    log_file = open('{}/train.log'.format(tb.path), 'w')

    config.DATA["num_epochs"] = 600 
    config.DATA['train_batch_size'] = 4
    
    config.DATA['lr_milestone'] = [600] 
    config.DATA['learning_rate'] = 1e-4
    # config.DATA['lr_milestone'] = [320,480]
    # config.DATA['lr_decay'] = 0.5

    # config.DATA['lr_milestone'] = [160,320,480]
    # config.DATA['learning_rate'] = 5e-5
    # config.DATA['lr_decay'] = 0.5

    train_dataset = fusionDatasetNoOcc(config.DATA["train_Folder"], imgSize=320, refFolder='Ref2/')
    val_dataset = fusionDatasetNoOcc(config.DATA["val_Folder"], isTrain=False, imgSize=512, augment=False, refFolder='Ref2/') #理论上val时imgSize=None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.DATA["train_batch_size"], num_workers=config.DATA['train_num_workers'], shuffle=True, drop_last=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config.DATA["val_batch_size"], num_workers=config.DATA['val_num_workers'], shuffle=False, drop_last=True, pin_memory=True)
    #test
    test_dataset = fusionDatasetNoOcc (config.DATA["test_Folder"], isTrain=False, imgSize=512, augment=False, refFolder='Ref2/')
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.DATA["test_batch_size"], num_workers=config.DATA['test_num_workers'], shuffle=False, drop_last=True, pin_memory=True)
    
    # 搭网络
    level = 5
    net = pixelFusionNet(Level=level)

    print('Parameters in %s: %d.' % ('pixelFusionrNet', count_parameters(net)))
    # 初始化网络的权重
    net.apply(init_weights_xavier)
    # 搭solver
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.DATA['learning_rate'],
                                        betas=(config.DATA['momentum'], config.DATA['beta']))
    optimizer_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.DATA['lr_milestone'], gamma=config.DATA['lr_decay'])

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()

    vggnet =  VGG19()
    if torch.cuda.is_available():
        vggnet = torch.nn.DataParallel(vggnet).cuda()
    
    last_epoch = -1
    config.DATA["rsesume"] = None
    if config.DATA["resume"] is not None:
        _ = load_checkpoints(net, optimizer, config.DATA["resume"], config.DATA["resume_epoch"])
        print('Resume is not None, move to corresponding learning rate')
        # for i in range(0, last_epoch):
        #     optimizer_lr_scheduler.step()    


    first_val = True
    Best_Img_PSNR = 0

    num_batches = int(len(train_dataloader.dataset)/float(train_dataloader.batch_size))
    
    loss_file = open('{}/loss.log'.format(tb.path), 'w')
    for epoch in range(last_epoch+1, config.DATA["num_epochs"]):
        epoch_start_time = time()

        # AverageMeter
        fusion_losses = AverageMeter()
        img_PSNRs = AverageMeter()
        img_val_PSNRs = AverageMeter()
        img_test_PSNRs = AverageMeter()

        pbar = tqdm(np.arange(num_batches)) #, ncols=100

        net.train()
        for batch_idx, batch in enumerate(train_dataloader):
            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1,config.DATA["num_epochs"]))
            
            lrImg, refImg, hrImg, refImgWarped = batch
            lrImg.requires_grad = True
            refImg.requires_grad = True 
            hrImg.requires_grad = False
            refImgWarped.requires_grad = False

            lrImg = lrImg.cuda()
            refImg = refImg.cuda()
            hrImg = hrImg.cuda()
            refImgWarped = refImgWarped.cuda()

            optimizer.zero_grad() 
            output_hrImg, flows_final, delta_flows, grad = net(lrImg, refImg)

            # mse_loss = mseLoss(output_hrImg, hrImg)
            # pixel_loss = 10*charbonnierLoss(output_hrImg, hrImg)
            reconstruction_loss = perceptualLoss2(output_hrImg, hrImg, vggnet)
            warping_loss = warpingLoss(refImg, refImgWarped, flows_final, level, vggnet)
            regularization_loss = 5e-3*regularizationLoss(delta_flows)
            # gradLoss
            get_grad_nopadding = Get_gradient_nopadding()
            get_grad = Get_gradient()
            output_hrImg_grad = get_grad(output_hrImg)
            hrImg_grad = get_grad(hrImg)
            gradL1LossFun = gradL1Loss()
            grad_loss = 5e-2*gradL1LossFun(output_hrImg_grad, hrImg_grad) #师弟们是0.05 于是我将5e-3改为5e-2. 
            # 0702:当reconstruction_loss的权重为0.1时, grad_loss的权重为0.05, grad_branch_loss的权重为0.5, 所以当reconstruction_loss的权重为1时, 后两者的权重也扩大10倍
            hrImg_grad_nopadding = get_grad_nopadding(hrImg) 
            grad_branch_loss = 5e-1*gradL1LossFun(grad, hrImg_grad_nopadding) #师弟们给的0.5 于是我将5e-2改为5e-1
    
            fusion_loss = reconstruction_loss + warping_loss + regularization_loss + grad_loss + grad_branch_loss #
            # fusion_loss = mse_loss
        
            img_PSNR = PSNR(output_hrImg, hrImg)

            fusion_loss.backward()
            optimizer.step()

            fusion_losses.update(fusion_loss.item(), config.DATA['train_batch_size'])
            img_PSNRs.update(img_PSNR.item(), config.DATA['train_batch_size'])

            pbar.set_postfix(Train_Loss = fusion_losses.avg, Current_Loss=fusion_loss.item(), Rec=reconstruction_loss.item(), 
                            Warp=warping_loss.item(), Reg=regularization_loss.item()) # , Pixel=pixel_loss.item()
            # pbar.set_postfix(Train_Loss = fusion_losses.avg, Current_Loss=fusion_loss.item(), Mse=mse_loss.item()) 

        print('epoch:{}, recons:{}, 1*warp:{}, 5e-3*reg:{}, grad:{}, grad_branch:{}, fusionLoss:{}'.format(epoch,
                    1*reconstruction_loss, 1*warping_loss, regularization_loss, grad_loss, grad_branch_loss, fusion_loss), file=loss_file) # 
        # print('epoch:{}, mse:{}, fusionLoss:{}'.format(epoch, mse_loss, fusion_loss), file=loss_file) 

        net.eval()
        config.DATA['log_epoch'] = 5
        if first_val or epoch%config.DATA['log_epoch']==config.DATA['log_epoch']-1:
            with torch.no_grad(): 
                first_val = False
                for batch_idx, batch in enumerate(val_dataloader):
                    
                    _, lrImg, refImg, hrImg = batch
        
                    lrImg = lrImg.cuda()
                    refImg = refImg.cuda()
                    hrImg = hrImg.cuda()

                    ## 全图
                    # _, _, h_lr, w_lr = lrImg.shape
                    # scale = 1
                    # lr_patches = []
                    # ref_patches = []
                    # grids = []
                    # patch_size = 512
                    # stride = 500 #无overlap
                    # use_padding = False #法1用镜面反射/零填充; 法2不用填充,全是原始图像
                    # is_avg = False #取平均 or 反距离加权

                    # if not is_avg:
                    #     weight = gen_blend_template(patch_size=patch_size)
                    # else:
                    #     weight = torch.ones((1, 3, patch_size, patch_size), dtype=torch.float32).to(lrImg)

                    # if use_padding:
                    # # 法1 用镜面反射/零填充
                    #     for ind_row in range(0, h_lr - (patch_size - stride), stride):
                    #         for ind_col in range(0, w_lr - (patch_size - stride), stride):
                    #             lq_patch = lrImg[:, :, ind_row:ind_row + patch_size, ind_col:ind_col + patch_size]
                    #             ref_patch = refImg[:, :, ind_row:ind_row + patch_size, ind_col:ind_col + patch_size]

                    #             if lq_patch.shape[2] != patch_size or lq_patch.shape[3] != patch_size:
                    #                 if patch_size - lq_patch.shape[3] > lq_patch.shape[3] or patch_size - lq_patch.shape[2] > lq_patch.shape[2]: #要填充的区域大于patch原有的区域
                    #                     pad = torch.nn.ZeroPad2d((0, patch_size - lq_patch.shape[3], 0, patch_size - lq_patch.shape[2]))
                    #                 else:
                    #                     pad = torch.nn.ReflectionPad2d((0, patch_size - lq_patch.shape[3], 0, patch_size - lq_patch.shape[2]))
                    #                 lq_patch = pad(lq_patch)
                    #                 ref_patch = pad(ref_patch)
                    #             lr_patches.append(lq_patch)
                    #             ref_patches.append(ref_patch)
                    #             grids.append((ind_row * scale, ind_col * scale, patch_size * scale))
                    #     sr_patches = []
                    #     for i in range(len(lr_patches)):
                    #         sr, _, _, _ = net(lr_patches[i], ref_patches[i]) #inference, net有3个输出
                    #         sr_patches.append(sr)
                    #     patch_size = grids[0][2]
                    #     h_sr = grids[-1][0] + patch_size
                    #     w_sr = grids[-1][1] + patch_size
                    #     sr_image_large = torch.zeros((1, 3, h_sr, w_sr), dtype=torch.float32).to(sr_patches[0])
                    #     counter = torch.zeros((1, 3, h_sr, w_sr), dtype=torch.float32).to(sr_patches[0])
                    #     for i in range(len(sr_patches)):
                    #         sr_image_large[:, :, grids[i][0]:grids[i][0] + patch_size, grids[i][1]:grids[i][1] + patch_size] += sr_patches[i]*weight
                    #         counter[:, :, grids[i][0]:grids[i][0] + patch_size, grids[i][1]:grids[i][1] + patch_size] += weight
                    #     sr_image_large /= counter
                    #     output_hrImg = sr_image_large[:, :, 0:h_lr*scale, 0:w_lr*scale]
                    # else:
                    #     # 法2 不用填充,全是原始图像
                    #     hnum = (h_lr-patch_size)//stride + 2
                    #     wnum = (w_lr-patch_size)//stride + 2
                    #     for i in range(0, hnum): 
                    #         h_st = i * stride
                    #         if h_st + patch_size > h_lr:
                    #             h_st = h_lr-patch_size
                    #         for j in range(0, wnum):
                    #             w_st = j * stride
                    #             if w_st + patch_size > w_lr:
                    #                 w_st = w_lr - patch_size
                    #             lr_patches.append(lrImg[:,:, h_st:h_st+patch_size, w_st:w_st+patch_size])
                    #             ref_patches.append(refImg[:,:, h_st:h_st+patch_size, w_st:w_st+patch_size])
                    #             grids.append((h_st*scale, w_st*scale, patch_size*scale))
                    #     sr_patches = []
                    #     for i in range(len(lr_patches)):
                    #         sr_out, _, _, _ = net(lr_patches[i], ref_patches[i]) #inference, net有3个输出
                    #         sr_patches.append(sr_out)

                    #     srImg = torch.zeros((1, 3, h_lr, w_lr), dtype=torch.float32).to(sr_patches[0])
                    #     srIdx = torch.zeros((1, 3, h_lr, w_lr), dtype=torch.float32).to(sr_patches[0])
                    #     for i in range(len(sr_patches)):
                    #         srImg[:,:, grids[i][0]:grids[i][0]+patch_size, grids[i][1]:grids[i][1]+patch_size] += sr_patches[i]*weight
                    #         srIdx[:,:, grids[i][0]:grids[i][0]+patch_size, grids[i][1]:grids[i][1]+patch_size] += weight
                    #     srImg /= srIdx
                    #     output_hrImg = srImg
                    
                    output_hrImg, _, _, _  = net(lrImg, refImg)

                    img_val_PSNR = PSNR(output_hrImg, hrImg) 
                    img_val_PSNRs.update(img_val_PSNR.item(), config.DATA['val_batch_size'])

                epoch_end_time = time()
                pbar.set_postfix(Train_Loss=fusion_losses.avg,Train_PSNR=img_PSNRs.avg,Val_PSNR=img_val_PSNRs.avg,Time=epoch_end_time - epoch_start_time)

                if img_val_PSNRs.avg > Best_Img_PSNR:
                    Best_Img_PSNR = img_val_PSNRs.avg
                    save_checkpoints(tb.path, Best_Img_PSNR, epoch, net, optimizer)

                    # test
                    for batch_idx, batch in enumerate(test_dataloader):
                    
                        _, lrImg, refImg, hrImg = batch
            
                        lrImg = lrImg.cuda()
                        refImg = refImg.cuda()
                        hrImg = hrImg.cuda()

                        output_hrImg, _, _, _  = net(lrImg, refImg)

                        img_test_PSNR = PSNR(output_hrImg, hrImg) 
                        img_test_PSNRs.update(img_test_PSNR.item(), config.DATA['test_batch_size'])
                    
                    epoch_end_time = time()
                    pbar.set_postfix(Train_Loss=fusion_losses.avg,Train_PSNR=img_PSNRs.avg,Val_PSNR=img_val_PSNRs.avg,Test_PSNR=img_test_PSNRs.avg,
                                        Time=epoch_end_time - epoch_start_time)
        
        optimizer_lr_scheduler.step()

        print("Epoch [%d/%d] Train_Loss=%.5f Train_PSNR=%.6f Val_PSNR=%.6f Test_PSNR=%.6f Time=%.2fmin" % (epoch + 1, config.DATA["num_epochs"], fusion_losses.avg,
                                img_PSNRs.avg, img_val_PSNRs.avg, img_test_PSNRs.avg, (epoch_end_time - epoch_start_time)/60), file=log_file)

        pbar.close()

    loss_file.close()     
    log_file.close()


