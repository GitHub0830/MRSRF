import torch.nn as nn
import torch
import torchvision
import os
from tqdm import tqdm
import sys
from time import time
import math

from utils.log import TensorBoardX
from dataLoaders.dataLoader1 import fusionDatasetNoOcc1, fusionDatasetNoOcc1_x2, fusionDatasetNoOcc1_x4
import config.train_config  as config
from utils.utils import *
from utils.losses import *
from utils.occlusion import occlusion
# from models.pixelFusionModel_grad_w import pixelFusionNet, warp
from models.pixelFusionModel_grad_w import pixelFusionNet, warp
# from models.pixelFusionModel_grad_test import pixelFusionNet, warp #能输出mask
import models
import cv2
from utils.imgWarp import warp1
import numpy as np

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def visualizeFlow(flow, size): 
    #仅输出光流强度信息，更方便指示目标所在位置
    # print(flow.shape, size)
    hsv = np.zeros(size, dtype=np.uint8)
    hsv[:, :, 0] = 0
    hsv[:, :, 1] = 0
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def visualizeFlow_ori(flow, size):
    # print(flow.shape, size)
    hsv = np.zeros(size, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

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

def minRect(img):
    h, w = img.shape
    # print(img.shape, type(img))
    a, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    
    max_id = areas.index(max(areas))
    
    max_rect = cv2.minAreaRect(contours[max_id])
    max_box = cv2.boxPoints(max_rect)
    max_box = np.int0(max_box)
    # print(max_box)
    # [[  9 510] #左下
    # [  9   0] #左上
    # [510   0] #右上
    # [510 510]] #右下
    left = max_box[1][0]
    right = max_box[3][0]
    top = max_box[1][1]
    bottom = max_box[3][1]
    # print('left, right, top, bottom:', left, right, top, bottom)
    newImg = np.zeros((h, w), np.uint8)
    newImg[top:bottom, left:right] = 1
    return newImg


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
                           
    # dataset = fusionDatasetNoOcc1_x2(config.DATA["test_Folder"], isTrain=False, imgSize=1024, augment=False) #test_Folder
    # dataset = fusionDatasetNoOcc1(config.DATA["test_Folder"], isTrain=False, imgSize=1024, augment=False) #test_Folder
    dataset = fusionDatasetNoOcc1_x4(config.DATA["val_Folder"], isTrain=False, imgSize=1024, augment=False) #test_Folder
    scale = 4

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.DATA["val_batch_size"], num_workers=config.DATA['val_num_workers'], 
                                            shuffle=False, drop_last=True, pin_memory=True)
    
    level = 5
    net = pixelFusionNet(Level=level) 

    set_requires_grad(net, False)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.DATA['learning_rate'], betas=(config.DATA['momentum'], config.DATA['beta']))

    # net = net.cuda() #如果key中不涉及module
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()  

    
    # config.DATA["resume"] = "/home/disk60t/jjq/imageFusion/ckpt_P40pro//grad/2021-12-31T09:27:47.982984/" #x2训练完成 44.835918(545/600epoch)
    # last_epoch = load_checkpoints(net, optimizer, config.DATA["resume"], config.DATA["resume_epoch"])

    # config.DATA["resume"] = "/home/disk60t/jjq/imageFusion/ckpt_P40pro//grad/2021-12-21T09:06:37.038060/best_3343.pth" #x5 33.433785(130/600epoch) input=LR+ref
    # config.DATA["resume"] = "/home/disk60t/jjq/imageFusion/ckpt_P40pro/grad/2021-12-24T02:05:13.410636/best_4016.pth" #x2 40.166269(65/600epoch) input=LR_ds2_ups+ref_x2
    # config.DATA["resume"] = "/home/disk60t/jjq/imageFusion/ckpt_P40pro//grad/2021-12-24T14:47:54.888964/best_3150.pth" #x5 31.505291(105/600epoch) input=LR_ds5+ref
    # config.DATA["resume"] = "/home/disk60t/jjq/imageFusion/ckpt_P40pro/grad/2021-12-31T03:26:16.641448/best_3414.pth" #x5 修正后的数据 中间结果34.148304(425/450epoch) input=LR_small+ref1
    # config.DATA["resume"] = "/home/disk60t/jjq/imageFusion/ckpt_P40pro/grad/2021-12-31T03:26:16.641448/"  #x5 修正后的数据 最终结果34.341496(505/600epoch) input=LR_small+ref1
    # config.DATA["resume"] = "/home/disk60t/jjq/imageFusion/ckpt_P40pro//grad/2022-02-24T03:21:05.626752/"  #x4 40.400003(550/600epoch) input=LR_small_x4+ref1_x4
    config.DATA["resume"] = "/home/jjq/paper/ckpts_P40pro/MVIF_weight/2022-03-19T19:34:56.869181/" #x4 34.727039(290/300epoch) 重新造了LR_small_x4
    last_epoch = load_checkpoints(net, optimizer, config.DATA["resume"], config.DATA["resume_epoch"])


    config.DATA['output_path'] = "/home/jjq/paper/out/mvif_out/"
  
    test_psnr = dict()
    g_name= 'init'
    save_num = 0
    img_PSNRs = AverageMeter()
    img_lr_PSNRs = AverageMeter()

    FolderName = '290th3472_val_calcPSNR_512_OpticalFlow'  #
    out_dir = os.path.join(config.DATA['output_path'], FolderName)
    if not os.path.isdir(out_dir):
        mkdir(out_dir)
    
    flow_file = open(os.path.join(out_dir, 'flow_result.txt'), 'w')
    flowXs_mean = []
    flowXs_max = []
    flowYs_mean = []
    flowYs_max = []
    
    t1 = time()
    net.eval() 
    f_time = open("/home/jjq/paper/out/mvif_out/time.txt", 'w')
    print('The 512x512 image used', file=f_time )
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout, desc='testing'):
        
        imgPath, lrImg, refImg, hrImg = batch #, lrImg_small
        
        name = imgPath[0]
       
        if not g_name == name:
            g_name = name
            save_num = 0
        save_num = save_num + 1

        if not name in test_psnr:
            test_psnr[name] = {'n_samples': 0, 'psnr': [], 'lr_psnr': []}

        with torch.no_grad(): 
            lrImg = lrImg.cuda()
            refImg = refImg.cuda()
            hrImg = hrImg.cuda()
            # lrImg_small = lrImg_small.cuda()
            if 0:
                _, _, h_lr, w_lr = lrImg.shape
                scale = 1
                lr_patches = []
                ref_patches = []
                grids = []
                patch_size = 512
                stride = 500
                use_padding = False #法1用镜面反射/零填充; 法2不用填充,全是原始图像
                is_avg = False #取平均 or 反距离加权

                if not is_avg:
                    weight = gen_blend_template(patch_size=patch_size)
                else:
                    weight = torch.ones((1, 3, patch_size, patch_size), dtype=torch.float32).to(lrImg)

                if use_padding:
                # 法1 用镜面反射/零填充
                    for ind_row in range(0, h_lr - (patch_size - stride), stride):
                        for ind_col in range(0, w_lr - (patch_size - stride), stride):
                            lq_patch = lrImg[:, :, ind_row:ind_row + patch_size, ind_col:ind_col + patch_size]
                            ref_patch = refImg[:, :, ind_row:ind_row + patch_size, ind_col:ind_col + patch_size]

                            if lq_patch.shape[2] != patch_size or lq_patch.shape[3] != patch_size:
                                if patch_size - lq_patch.shape[3] > lq_patch.shape[3] or patch_size - lq_patch.shape[2] > lq_patch.shape[2]: #要填充的区域大于patch原有的区域
                                    pad = torch.nn.ZeroPad2d((0, patch_size - lq_patch.shape[3], 0, patch_size - lq_patch.shape[2]))
                                else:
                                    pad = torch.nn.ReflectionPad2d((0, patch_size - lq_patch.shape[3], 0, patch_size - lq_patch.shape[2]))
                                lq_patch = pad(lq_patch)
                                ref_patch = pad(ref_patch)
                            lr_patches.append(lq_patch)
                            ref_patches.append(ref_patch)
                            grids.append((ind_row * scale, ind_col * scale, patch_size * scale))
                    sr_patches = []
                    for i in range(len(lr_patches)):
                        sr, _, _, _ = net(lr_patches[i], ref_patches[i]) #inference, net有3个输出
                        sr_patches.append(sr)
                    patch_size = grids[0][2]
                    h_sr = grids[-1][0] + patch_size
                    w_sr = grids[-1][1] + patch_size
                    sr_image_large = torch.zeros((1, 3, h_sr, w_sr), dtype=torch.float32).to(sr_patches[0])
                    counter = torch.zeros((1, 3, h_sr, w_sr), dtype=torch.float32).to(sr_patches[0])
                    for i in range(len(sr_patches)):
                        sr_image_large[:, :, grids[i][0]:grids[i][0] + patch_size, grids[i][1]:grids[i][1] + patch_size] += sr_patches[i]*weight
                        counter[:, :, grids[i][0]:grids[i][0] + patch_size, grids[i][1]:grids[i][1] + patch_size] += weight
                    sr_image_large /= counter
                    output_hrImg = sr_image_large[:, :, 0:h_lr*scale, 0:w_lr*scale]
                else:
                    # 法2 不用填充,全是原始图像
                    hnum = (h_lr-patch_size)//stride + 2
                    wnum = (w_lr-patch_size)//stride + 2
                    for i in range(0, hnum): 
                        h_st = i * stride
                        if h_st + patch_size > h_lr:
                            h_st = h_lr-patch_size
                        for j in range(0, wnum):
                            w_st = j * stride
                            if w_st + patch_size > w_lr:
                                w_st = w_lr - patch_size
                            lr_patches.append(lrImg[:,:, h_st:h_st+patch_size, w_st:w_st+patch_size])
                            ref_patches.append(refImg[:,:, h_st:h_st+patch_size, w_st:w_st+patch_size])
                            grids.append((h_st*scale, w_st*scale, patch_size*scale))
                    sr_patches = []
                    for i in range(len(lr_patches)):
                        sr_out, _, _, _ = net(lr_patches[i], ref_patches[i]) #inference, net有3个输出
                        sr_patches.append(sr_out)

                    srImg = torch.zeros((1, 3, h_lr, w_lr), dtype=torch.float32).to(sr_patches[0])
                    srIdx = torch.zeros((1, 3, h_lr, w_lr), dtype=torch.float32).to(sr_patches[0])
                    for i in range(len(sr_patches)):
                        srImg[:,:, grids[i][0]:grids[i][0]+patch_size, grids[i][1]:grids[i][1]+patch_size] += sr_patches[i]*weight
                        srIdx[:,:, grids[i][0]:grids[i][0]+patch_size, grids[i][1]:grids[i][1]+patch_size] += weight
                    srImg /= srIdx
                    output_hrImg = srImg
            else:
                # output_hrImg, flows_final, _, _, mask = net(lrImg, refImg)
                t_model = time()
                output_hrImg, flows_final, _, _ = net(lrImg, refImg)
                print('%.4f' % (time()-t_model), file=f_time)
                # a

            # # print('mask.shape:', mask.shape) #torch.Size([1, 16, 512, 512])
            # mask = torch.sum(mask, dim=1) #将16个维度的值相加
            # mask = mask.unsqueeze(dim=1) 
            # mask = mask/mask.shape[1]

            # 我计算PSNR的方式
            # img_PSNR = PSNR(output_hrImg, hrImg) 
            # img_lr_PSNR = PSNR(lrImg, hrImg)
            # DCSR计算PSNR的方式
            # lrImg = lrImg[:,:, 4:504, 4:504] 
            # hrImg = hrImg[:,:, 4:504, 4:504]
            # output_hrImg = output_hrImg[:,:, 4:504, 4:504]
            # refImg = refImg[:,:, 4:504, 4:504]

            output_hrImg = quantize(output_hrImg)
            img_PSNR = calc_psnr(output_hrImg, hrImg, scale=scale) 
            img_lr_PSNR = calc_psnr(lrImg, hrImg, scale=scale) 

            img_PSNRs.update(img_PSNR.item(), config.DATA['val_batch_size'])
            img_lr_PSNRs.update(img_lr_PSNR.item(), config.DATA['val_batch_size'])

            test_psnr[name]['n_samples'] += 1
            test_psnr[name]['psnr'].append(img_PSNR)
            test_psnr[name]['lr_psnr'].append(img_lr_PSNR)

            nameWithoutsuffix = imgPath[0].split('.')[0]

            # smallImgName = nameWithoutsuffix + '_0small.png'
            lrImgName = nameWithoutsuffix + '_1lr.png'
            hrImgName = nameWithoutsuffix + '_3hr.png'
            fusionImgName = nameWithoutsuffix + '_2fusion.png'
            refImgName = nameWithoutsuffix + '_4ref.png'
            # maskName = nameWithoutsuffix + '_6mask.png'
            # maskName2 = nameWithoutsuffix + '_6mask_open.png'
            # maskName3 = nameWithoutsuffix + '_6mask_erode.png'
            # maskName4 = nameWithoutsuffix + '_6mask_erode1.png'

            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, smallImgName), 
            #             (lrImg_small.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))

            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, lrImgName), 
                        (lrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
        
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, hrImgName), 
                        (hrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
        
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, fusionImgName), 
                        (output_hrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))

            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, refImgName), 
                        (refImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))

            # mask = mask.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0)
            # #形态学处理
            # se = np.ones((5,5), np.uint8) 
            # se_hor = np.ones((1,10), np.uint8) 
            # se_ver = np.ones((10,1), np.uint8) 
            # mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se) #闭运算是先膨胀后腐蚀: 善于补小洞和去除小噪声
            # # mask_dilation = cv2.dilate(mask, se, iterations=1)
            # mask_erosion = cv2.erode(mask, se_ver, iterations=1)
            # mask_erosion1 = cv2.erode(mask_erosion, se_hor, iterations=1)
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, maskName3), (mask_erosion*255.0).astype(np.uint8))
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, maskName4), (mask_erosion1*255.0).astype(np.uint8))
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, maskName2), (mask_open* 255.0).astype(np.uint8))
            # #找最小包围框
            # mask_small = minRect((mask[:,:,0]*255.0).astype(np.uint8))
            # maskName3 = nameWithoutsuffix + '_6mask_small.png'
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, maskName3), (mask_small* 255.0).astype(np.uint8))

            flowImgName = nameWithoutsuffix + '_5flow.png' #最高尺度的光流
            flow = flows_final[level-1]
            flow = flow[0].cpu().numpy().transpose(1, 2, 0)
            flowImg = visualizeFlow_ori( flow, (flow.shape[0],flow.shape[1],3) )
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, flowImgName), flowImg)

            # warpedImgName = nameWithoutsuffix + '_5warped1.png'
            # refImgNumpy = refImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0)
            # refImgWarped = warp1(refImgNumpy, flow)
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, warpedImgName), 
            #             (refImgWarped* 255.0)[:,:,::-1].astype(np.uint8))

            # 输出flow的值
            flowX  = (flows_final[level-1][0].cpu().numpy())[0,:,:]
            flowXs_mean.append(abs(flowX).mean())
            flowXs_max.append(abs(flowX).max())
            flowY  = (flows_final[level-1][0].cpu().numpy())[1,:,:]
            flowYs_mean.append(abs(flowY).mean())
            flowYs_max.append(abs(flowY).max())
            print('{0}: meanX={1}\t maxX={2}\t meanY={3}\t maxY={4}'.format(nameWithoutsuffix, abs(flowX).mean(), abs(flowX).max(),
                    abs(flowY).mean(), abs(flowY).max()), file=flow_file)
            
    print('avg: \t {}\t {}\t {}\t {}'.format(np.mean(flowXs_mean), np.mean(flowXs_max), np.mean(flowYs_mean), np.mean(flowYs_max)), file=flow_file) 
    flow_file.close()

    t2 = time()

    ## Output test results
    print('============================ TEST RESULTS ============================')
    print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
    for name in test_psnr: 
        test_psnr[name]['psnr'] = torch.mean(torch.stack(test_psnr[name]['psnr']))
        test_psnr[name]['lr_psnr'] = torch.mean(torch.stack(test_psnr[name]['lr_psnr']))
    #     print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}\t PSNR_Ratio: {3}'.format(name, test_psnr[name]['n_samples'],
    #                                                                     test_psnr[name]['psnr'], test_psnr[name]['psnr']/test_psnr[name]['lr_psnr'])) #s输出太多了, 被我注释掉了

    result_file = open(os.path.join(config.DATA['output_path'], FolderName, 'test_result.txt'), 'w')
    sys.stdout = result_file
    print('============================ TEST RESULTS ============================')
    print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
    for name in test_psnr:
        print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}\t PSNR_Ratio: {3}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr'], test_psnr[name]['psnr']/test_psnr[name]['lr_psnr']))
    
    print('Time = {:.2f}s'.format(t2-t1))

    result_file.close()
       