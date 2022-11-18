import torch.nn as nn
import torch
import torchvision
import os
from tqdm import tqdm
import sys
from time import time
import math

from utils.log import TensorBoardX
from dataLoaders.dataLoader1 import fusionDatasetNoOcc
import config.train_config  as config
from utils.utils import *
from utils.losses import *
from models.pixelFusionModel_grad_w import pixelFusionNet, warp
import models
import cv2
from utils.flow_to_img import flow_to_image
from torchvision.utils import make_grid
from PCV.tools import pca


def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)


def sigmoid(x):
    # print(x[0].shape)
    return 1.0 / (1 + np.exp(-x.astype(np.float64)))

# def pca(X,k):#k is the components you want
#     #mean of each feature
#     n_samples, n_features = X.shape
#     mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
#     #normalization
#     norm_X=X-mean
#     #scatter matrix
#     scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
#     #Calculate the eigenvectors and eigenvalues
#     eig_val, eig_vec = np.linalg.eig(scatter_matrix)
#     eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
#     # sort eig_vec based on eig_val from highest to lowest
#     eig_pairs.sort(reverse=True)
#     # select the top k eig_vec
#     feature=np.array([ele[1] for ele in eig_pairs[:k]])
#     #get new data
#     data=np.dot(norm_X,np.transpose(feature))
#     return data

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

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # tb = TensorBoardX(config_filename = "config/train_config.py", code_filename="test.py", output_path=config.OUT['path'] , sub_dir="test")
    config.DATA['val_Folder'] = "/home/disk60t/HUAWEI/data/20201118/val/"
    dataset = fusionDatasetNoOcc(config.DATA["val_Folder"], isTrain=False, imgSize=1024, augment=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.DATA["val_batch_size"], num_workers=config.DATA['val_num_workers'], 
                                            shuffle=False, drop_last=True, pin_memory=True)
    
    level = 5
    net = pixelFusionNet(Level=level) 

    set_requires_grad(net, False)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.DATA['learning_rate'], betas=(config.DATA['momentum'], config.DATA['beta']))

    # net = net.cuda() #如果key中不涉及module
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()  #如果key中涉及到module, 就是采用DataParallel的方式训练的

    
    # config.DATA['resume'] = "/home/disk60t/jjq/imgFusion/pixelFusionNet//naive/2021-02-10T10:20:55.342608" #31.798531(580/600epoch), grad
    config.DATA["resume"] = "/home/disk60t/jjq/imgFusion/pixelFusionNet//naive/2021-04-30T09:34:21.800417" #32.346720(550/600epoch)
    last_epoch = load_checkpoints(net, optimizer, config.DATA["resume"], config.DATA["resume_epoch"])

    config.DATA['output_path'] = '/home/disk60t/jjq/imgFusion/pixelFusionNet/Out'
    tt = time()
  
    test_psnr = dict()
    g_name= 'init'
    save_num = 0
    img_PSNRs = AverageMeter()
    img_lr_PSNRs = AverageMeter()

    FolderName = '550th3234_w_pca' #'380th3019_allVal'
    out_dir = os.path.join(config.DATA['output_path'], FolderName)
    if not os.path.isdir(out_dir):
        mkdir(out_dir)
    flow_file = open(os.path.join(config.DATA['output_path'], FolderName, 'flow_result.txt'), 'w')
    flowXs_mean = []
    flowXs_max = []
    flowYs_mean = []
    flowYs_max = []

    numIter = 0

    net.eval() 
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout, desc='testing'):
        
        imgPath, lrImg, refImg, hrImg = batch 
        
        name = imgPath[0]
        # print(imgPath) []为什么返回的是list?
        # print(name)
        # print(type(name))
        # a
        if not g_name == name:
            g_name = name
            save_num = 0
        save_num = save_num + 1

        if not name in test_psnr:
            test_psnr[name] = {'n_samples': 0, 'psnr': [], 'lr_psnr': []}

        with torch.no_grad(): 
                
            lrImg = lrImg.cuda(async=True)
            refImg = refImg.cuda(async=True)
            hrImg = hrImg.cuda(async=True)

            # output_hrImg, flows_final, delta_flows, grad = net(lrImg, refImg)
            output_hrImg, flows_final, delta_flows, grad, target_feas, ref_feas, ref_weights = net(lrImg, refImg)

            img_PSNR = PSNR(output_hrImg, hrImg) 
            img_PSNRs.update(img_PSNR.item(), config.DATA['val_batch_size'])

            img_lr_PSNR = PSNR(lrImg, hrImg)
            img_lr_PSNRs.update(img_lr_PSNR.item(), config.DATA['val_batch_size'])

            test_psnr[name]['n_samples'] += 1
            test_psnr[name]['psnr'].append(img_PSNR)
            test_psnr[name]['lr_psnr'].append(img_lr_PSNR)

            nameWithoutsuffix = imgPath[0].split('.')[0]
            # 专门针对all_val数据
            # nameWithoutsuffix = nameWithoutsuffix.split('_')[1]+'_'+nameWithoutsuffix.split('_')[0]                                                 ##第2处
            # print(nameWithoutsuffix)

            lrImgName = nameWithoutsuffix + '_0lr.png'
            hrImgName = nameWithoutsuffix + '_2hr.png'
            fusionImgName = nameWithoutsuffix + '_1fusion.png'
            refImgName = nameWithoutsuffix + '_3ref.png'
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, lrImgName), 
                        (lrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
        
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, hrImgName), 
                        (hrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
        
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, fusionImgName), 
                        (output_hrImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))

            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, refImgName), 
                        (refImg.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
            
            # flowImgName = nameWithoutsuffix + '_4flow' #所有尺度的光流
            # for i in range(len(flows_final)):
            #     flow = flows_final[i]
            #     flow = flow[0].cpu().numpy().transpose(1, 2, 0)
            #     # flowImg = flow_to_image(flow)
            #     flowImg = visualizeFlow( flow, (flow.shape[0],flow.shape[1],3) )
            #     cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, flowImgName+str(level-1-i)+'.png'), flowImg)

            # flowImgName = nameWithoutsuffix + '_4flow.png' #最高尺度的光流
            # flow = flows_final[level-1]
            # flow = flow[0].cpu().numpy().transpose(1, 2, 0)
            # flowImg = visualizeFlow( flow, (flow.shape[0],flow.shape[1],3) )
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, flowImgName), flowImg)

            # warpedImgName = nameWithoutsuffix + '_5warped.png' #refWarped图像
            # warpedRef = warp(refImg, flows_final[level-1])
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, warpedImgName), 
            #             (warpedRef.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))

            # gradImgName = nameWithoutsuffix + '_6grad.png' #输出的grad图像
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, gradImgName), 
            #             (grad.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
            
            # get_grad_nopadding = Get_gradient_nopadding()
            # hrImg_grad_nopadding = get_grad_nopadding(hrImg)
            # hrGradImgName = nameWithoutsuffix + '_7hrGrad.png' #hrGrad图像
            # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, hrGradImgName), 
            #             (hrImg_grad_nopadding.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))

            feaName = nameWithoutsuffix + '_5fea' #看weight图对特征的选择
            numFeas = target_feas.shape[1]
            m, n = target_feas.shape[-2:]
            # print(m, n)
            target_lists = []
            # print(numFeas)
            for i in range(numFeas):
                target_fea = target_feas[0][i]
                # print(target_fea.shape) #torch.Size([16, 1024, 1024])
                target_fea = target_fea.cpu().numpy()
                target_lists.append(target_fea*255.0)
                # target_fea = (((target_fea - np.min(target_fea)) / (np.max(target_fea) - np.min(target_fea)))*255.0).astype(np.uint8)
                # target_fea = (sigmoid(target_fea)*255.0).astype(np.uint8)
                # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, feaName+str(i)+'_target.png'), target_fea)
                ref_fea = ref_feas[0][i]
                ref_fea = ref_fea.cpu().numpy()
                # ref_fea = (((ref_fea - np.min(ref_fea)) / (np.max(ref_fea) - np.min(ref_fea)))* 255.0).astype(np.uint8)
                ref_fea = (sigmoid(ref_fea)*255.0).astype(np.uint8)
                # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, feaName+str(i)+'_ref.png'), ref_fea)
                ref_w = ref_weights[0][i]
                ref_w = ref_w.cpu().numpy()
                # ref_w = (((ref_w - np.min(ref_w)) / (np.max(ref_w) - np.min(ref_w)))* 255.0).astype(np.uint8)
                ref_w = (sigmoid(ref_w)*255.0).astype(np.uint8)
                # cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, feaName+str(i)+'_w.png'), ref_w)

            target_mat = np.array([fea.flatten() for fea in target_lists], 'f')
            V, S, immean = pca.pca(target_mat) #https://blog.csdn.net/dianchen5200/article/details/101952027
            # print(immean.reshape(m, n)[0:10, 0:10])
            # print(V[0].reshape(m, n)[0:10, 0:10])
            # a
            cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, feaName+str(i)+'_target_pcaM.png'), immean.reshape(m, n).astype(np.uint8))
            for i in range(5):
                target_pca = V[i].reshape(m, n)
                target_pca = (((target_fea - np.min(target_pca)) / (np.max(target_pca) - np.min(target_pca)))*255.0).astype(np.uint8)
                # target_pca = (target_pca).astype(np.uint8)
                cv2.imwrite(os.path.join(config.DATA['output_path'], FolderName, feaName+str(i)+'_target_pca.png'), target_pca)

            # target_fea = target_feas[0]
            # ref_fea = ref_feas[0]
            # ref_w = ref_weights[0]  
            # # tb.add_image_single('blur_left', blurry_img_left[0].cpu(), epoch*len(train_dataloader)+curr_iter , 'train')
            # tb.add_image_single('target_fea', make_grid(target_fea.detach().cpu().unsqueeze(dim=1), nrow=4, padding=20, normalize=False, pad_value=1), numIter , 'val')
            # tb.add_image_single('ref_fea', make_grid(ref_fea.detach().cpu().unsqueeze(dim=1), nrow=4, padding=20, normalize=False, pad_value=1), numIter , 'val')
            # tb.add_image_single('ref_w', make_grid(ref_w.detach().cpu().unsqueeze(dim=1), nrow=4, padding=20, normalize=False, pad_value=1), numIter , 'val')
            # numIter += 1
                

            ## 输出flow的值                                                                                                                             ##第3处
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

    # Output test results
    print('============================ TEST RESULTS ============================')
    print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
    for name in test_psnr:
        # test_psnr[name]['psnr'] = np.mean(test_psnr[name]['psnr'], axis=0)
        test_psnr[name]['psnr'] = torch.mean(torch.stack(test_psnr[name]['psnr']))
        # test_psnr[name]['lr_psnr'] = np.mean(test_psnr[name]['lr_psnr'], axis=0)
        test_psnr[name]['lr_psnr'] = torch.mean(torch.stack(test_psnr[name]['lr_psnr']))
        print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}\t PSNR_Ratio: {3}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr'], test_psnr[name]['psnr']/test_psnr[name]['lr_psnr'])) ##第4处，还需修改dataloader

    result_file = open(os.path.join(config.DATA['output_path'], FolderName, 'test_result.txt'), 'w')
    sys.stdout = result_file
    print('============================ TEST RESULTS ============================')
    print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
    for name in test_psnr:
        print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}\t PSNR_Ratio: {3}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr'], test_psnr[name]['psnr']/test_psnr[name]['lr_psnr']))
    result_file.close()
       