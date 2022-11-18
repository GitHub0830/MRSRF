import torch.nn as nn
import torch
import torchvision
import os
from tqdm import tqdm
import sys
from time import time
import math

from utils.log import TensorBoardX
from dataLoaders.dataLoaderTest import fusionDataset
import config.train_config  as config
from utils.utils import *
from utils.losses import *
from models.pixelFusionModel_grad import pixelFusionNet
import models
import cv2
from utils.flow_to_img import flow_to_image
from models.pixelFusionModel import warp


def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def visualizeFlow(flow, size):
    # print(flow.shape, size)
    hsv = np.zeros(size, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imwrite('examples/outFlow_new.png', rgb)
    # cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)
    return rgb

def gen_blend_template(patch_size = 2048): #patch_size应为4的倍数
    weightTensor = torch.zeros((1,3,patch_size, patch_size), dtype=torch.float32).cuda() #这种情况下应该to？直接cuda()
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    config.DATA["val_Folder"] =  "/home/disk60t/HUAWEI/data/1010_测试/"                               
    dataset = fusionDataset(config.DATA["val_Folder"], isTrain=False, imgSize=None, augment=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.DATA["val_batch_size"], num_workers=config.DATA['val_num_workers'], 
                                            shuffle=False, drop_last=True, pin_memory=True)
    
    level = 6
    net = pixelFusionNet(Level=level) 

    set_requires_grad(net, False)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.DATA['learning_rate'], betas=(config.DATA['momentum'], config.DATA['beta']))

    # net = net.cuda() #如果key中不涉及module
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()  #如果key中涉及到module, 就是采用DataParallel的方式训练的

    # config.DATA['resume'] = "/home/disk60t/jjq/imgFusion/pixelFusionNet//naive/2021-02-10T10:20:55.342608" #31.798531(580/600epoch), grad
    # config.DATA['resume'] = "/home/disk60t/jjq/imgFusion/pixelFusionNet//naive/2021-03-11T01:56:36.634540" #30.281744(570/600epoch), level=6, grad
    config.DATA["resume"] = "/home/disk60t/jjq/imgFusion/pixelFusionNet/naive/2021-03-15T13:01:33.169683/" #30.605557(470/600epoch) level=6, 加大了gradLoss的权重值
    

    last_epoch = load_checkpoints(net, optimizer, config.DATA["resume"], config.DATA["resume_epoch"])

    config.DATA['output_path'] = '/home/disk60t/jjq/imgFusion/pixelFusionNet/Out'
    tt = time()
  
    test_psnr = dict()
    g_name= 'init'
    save_num = 0
    img_PSNRs = AverageMeter()
    img_lr_PSNRs = AverageMeter()

    FolderName = '470th3060_test_2_weight' #'380th3019_allVal'
    out_dir = os.path.join(config.DATA['output_path'], FolderName)
    if not os.path.isdir(out_dir):
        mkdir(out_dir)
    # flow_file = open(os.path.join(config.DATA['output_path'], FolderName, 'flow_result.txt'), 'w')
    # flowXs = []
    # flowYs = []

    net.eval() 
    # gname = 'ddd'
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout, desc='testing'):
        
        imgPath, lrImg, refImg = batch 
        
        fileName = imgPath[0].split('/')[0] #文件夹名
        # print(fileName)
      
        if not os.path.isdir(os.path.join(config.DATA['output_path'], FolderName, fileName)):
            print(os.path.join(config.DATA['output_path'], FolderName, fileName))
            os.makedirs(os.path.join(config.DATA['output_path'], FolderName, fileName))
        
        # if not g_name == name:
        #     g_name = name
        #     save_num = 0
        # save_num = save_num + 1

        with torch.no_grad(): 
            lrImg = lrImg.cuda()
            refImg = refImg.cuda()
            # hrImg = hrImg.cuda(async=True)
            if 1:
                _, _, h_lr, w_lr = lrImg.shape
                scale = 1
                lr_patches = []
                ref_patches = []
                grids = []
                patch_size = 512
                stride = 400
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
                        sr, _, _ = net(lr_patches[i], ref_patches[i]) #inference, net有3个输出
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
                        sr_out, _, _, _ = net(lr_patches[i], ref_patches[i]) 
                        sr_patches.append(sr_out)

                    srImg = torch.zeros((1, 3, h_lr, w_lr), dtype=torch.float32).to(sr_patches[0])
                    srIdx = torch.zeros((1, 3, h_lr, w_lr), dtype=torch.float32).to(sr_patches[0])
                    for i in range(len(sr_patches)):
                        srImg[:,:, grids[i][0]:grids[i][0]+patch_size, grids[i][1]:grids[i][1]+patch_size] += sr_patches[i]*weight
                        # print(sr_patches[i][:, 0, 0:8, 0:8])
                        # print(weight[:, 0, 0:8, 0:8])
                        # tmp = sr_patches[i]*weight
                        # print(tmp[:, 0, 0:8, 0:8])
                        # a
                        srIdx[:,:, grids[i][0]:grids[i][0]+patch_size, grids[i][1]:grids[i][1]+patch_size] += weight
                    srImg /= srIdx
                    output_hrImg = srImg
            else:
                output_hrImg = net(lrImg, refImg)

            # img_PSNR = PSNR(output_hrImg, hrImg) 
            # img_PSNRs.update(img_PSNR.item(), config.DATA['val_batch_size'])

            # img_lr_PSNR = PSNR(lrImg, hrImg)
            # img_lr_PSNRs.update(img_lr_PSNR.item(), config.DATA['val_batch_size'])

            # test_psnr[name]['n_samples'] += 1
            # test_psnr[name]['psnr'].append(img_PSNR)
            # test_psnr[name]['lr_psnr'].append(img_lr_PSNR)



            imgName = imgPath[0].split('/')[-1] # 01.png

            # lrImgName = nameWithoutsuffix + '_0lr.png'
            # hrImgName = nameWithoutsuffix + '_2hr.png'
            # fusionImgName = nameWithoutsuffix + 'fusion.png'
            # refImgName = nameWithoutsuffix + '_3ref.png'

            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, lrImgName), 
            #             (lrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
        
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, hrImgName), 
            #             (hrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
        
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, fileName, imgName), 
                        (output_hrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))

            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, refImgName), 
            #             (refImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))


        

       