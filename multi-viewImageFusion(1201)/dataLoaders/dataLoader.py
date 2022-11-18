from PIL import Image, ImageOps
from torchvision.transforms import transforms
import torch
import os
import numpy as np
import random
import math
import cv2
# import sys  

# reload(sys)  
# sys.setdefaultencoding('utf8')  

def augmentation(imgs, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        for i in range(len(imgs)):
            imgs[i] = ImageOps.flip(imgs[i])
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            for i in range(len(imgs)):
                imgs[i] = ImageOps.mirror(imgs[i])
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].rotate(180)
            info_aug['trans'] = True
    
    return imgs

def random_mask(mask):
    p = random.random()
    if p < 0.1:
        mask *= 0.0
    elif p < 0.3:
        h, w = mask.shape
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        k = math.tan(random.random() * math.pi)
        
        rows = torch.arange(h).float() 
        cols = torch.arange(w).float() 
        grid = torch.meshgrid(rows, cols) 

        value = (grid[0] - y) - k * (grid[1] - x)
        mask[value > 0] = 0.0
    return mask

class fusionDatasetNoOcc(torch.utils.data.dataset.Dataset):
    # 可输入不同高低频比例融合的HR；中心区域验证；hrImgGray是为了计算HR图像的梯度来作为额外的监督, 如果不是灰度图就需要每个通道单独求梯度。
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True, refFolder='Ref1/'):
        super(fusionDatasetNoOcc, self).__init__() 
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'HR1.0/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print(len(self.imglist))
        # print(self.imglist)

        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
        # self.HRFolder = HRFolder
        self.refFolder = refFolder
        print('refFolder:', refFolder)
    
    def __getitem__(self, idx):
        # lrImg = Image.open(self.imgFolder + 'LR/'+ self.imglist[idx])
        lrImg = cv2.imread(self.imgFolder + 'LR_small/'+ self.imglist[idx])
        lrImg = cv2.resize(lrImg, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        lrImg = Image.fromarray(cv2.cvtColor(lrImg, cv2.COLOR_BGR2RGB))

        refImg = Image.open(self.imgFolder + self.refFolder + self.imglist[idx])
        hrImg = Image.open(self.imgFolder + 'HR1.0/' + self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped/'+ self.imglist[idx])
           
            if self.augment: 
                lrImg, refImg, hrImg, refImgWarped = augmentation([lrImg, refImg, hrImg, refImgWarped])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                left = int(np.floor(np.random.uniform(4 , w - 4 - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(4, h- 4 - self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)

        else: 
            left = 0
            top = 0
            crop_region = (left, top, left+512, top+512)
            lrImg = lrImg.crop(crop_region)
            refImg = refImg.crop(crop_region)
            hrImg = hrImg.crop(crop_region)


        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:
            refImgWarped = self.toTensor(refImgWarped)
         
            # 采用random_mask进行数据扩充
            # mask = torch.ones(*refImg.shape[1:]).float()
            # mask = random_mask(mask)
            # mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            # refImg = refImg*mask
            # return lrImg, refImg, hrImg, refImgWarped, mask
            
            return lrImg, refImg, hrImg, refImgWarped
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDatasetNoOcc_x2(torch.utils.data.dataset.Dataset):
    # 可输入不同高低频比例融合的HR；中心区域验证；hrImgGray是为了计算HR图像的梯度来作为额外的监督, 如果不是灰度图就需要每个通道单独求梯度。
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True, HRFolder='HR1.0_x2/'):
        super(fusionDatasetNoOcc_x2, self).__init__() 
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'LR_x2/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print('fusionDatasetNoOcc_x2:', len(self.imglist))
        # print(self.imglist)

        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
        self.HRFolder = HRFolder
    
    def __getitem__(self, idx):
        # lrImg = Image.open(self.imgFolder + 'LR_x2/'+ self.imglist[idx])
        lrImg = cv2.imread(self.imgFolder + 'LR_small_x2/'+ self.imglist[idx])
        lrImg = cv2.resize(lrImg, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        lrImg = Image.fromarray(cv2.cvtColor(lrImg, cv2.COLOR_BGR2RGB))


        refImg = Image.open(self.imgFolder + 'Ref1_x2/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + self.HRFolder + self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped_x2/'+ self.imglist[idx])
           
            if self.augment: 
                lrImg, refImg, hrImg, refImgWarped = augmentation([lrImg, refImg, hrImg, refImgWarped])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                left = int(np.floor(np.random.uniform(4 , w - 4 - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(4, h- 4 - self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)

        else: 
            left = 0
            top = 0
            crop_region = (left, top, left+512, top+512)
            lrImg = lrImg.crop(crop_region)
            refImg = refImg.crop(crop_region)
            hrImg = hrImg.crop(crop_region)


        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:
            refImgWarped = self.toTensor(refImgWarped)
         
            # 采用random_mask进行数据扩充
            # mask = torch.ones(*refImg.shape[1:]).float()
            # mask = random_mask(mask)
            # mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            # refImg = refImg*mask
            # return lrImg, refImg, hrImg, refImgWarped, mask
            
            return lrImg, refImg, hrImg, refImgWarped
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDatasetNoOcc_x4(torch.utils.data.dataset.Dataset):
    # 可输入不同高低频比例融合的HR；中心区域验证；hrImgGray是为了计算HR图像的梯度来作为额外的监督, 如果不是灰度图就需要每个通道单独求梯度。
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True, HRFolder='HR1.0_x4/'):
        super(fusionDatasetNoOcc_x4, self).__init__() 
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'LR_x4/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print('fusionDatasetNoOcc_x4:', len(self.imglist))
        # print(self.imglist)

        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
        self.HRFolder = HRFolder
    
    def __getitem__(self, idx):
        # lrImg = Image.open(self.imgFolder + 'LR_x2/'+ self.imglist[idx])
        lrImg = cv2.imread(self.imgFolder + 'LR_small_x4/'+ self.imglist[idx])
        lrImg = cv2.resize(lrImg, dsize=(0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        lrImg = Image.fromarray(cv2.cvtColor(lrImg, cv2.COLOR_BGR2RGB))


        refImg = Image.open(self.imgFolder + 'Ref1_x4/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + self.HRFolder + self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped_x4/'+ self.imglist[idx])
           
            if self.augment: 
                lrImg, refImg, hrImg, refImgWarped = augmentation([lrImg, refImg, hrImg, refImgWarped])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                left = int(np.floor(np.random.uniform(4 , w - 4 - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(4, h- 4 - self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)

        else: 
            left = 0
            top = 0
            crop_region = (left, top, left+512, top+512)
            lrImg = lrImg.crop(crop_region)
            refImg = refImg.crop(crop_region)
            hrImg = hrImg.crop(crop_region)


        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:
            refImgWarped = self.toTensor(refImgWarped)
         
            # 采用random_mask进行数据扩充
            # mask = torch.ones(*refImg.shape[1:]).float()
            # mask = random_mask(mask)
            # mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            # refImg = refImg*mask
            # return lrImg, refImg, hrImg, refImgWarped, mask
            
            return lrImg, refImg, hrImg, refImgWarped
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDatasetWarp(torch.utils.data.dataset.Dataset):
    # 参考图像经RAFT对齐到lr了
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True, HRFolder='HR/'):
        super(fusionDatasetWarp, self).__init__() #为什么dataloader的模型初始化不用super？
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'LR/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print(len(self.imglist))
        # print(self.imglist)

        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
        self.HRFolder = HRFolder
    
    def __getitem__(self, idx):
        lrImg = Image.open(self.imgFolder + 'LR/'+ self.imglist[idx])
        refImg = Image.open(self.imgFolder + 'warped1/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + self.HRFolder + self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            # refImgWarped = Image.open(self.imgFolder + 'refWarped/'+ self.imglist[idx])
           
            if self.augment: 
                # refImg = augmentationRef(refImg)
                lrImg, refImg, hrImg = augmentation([lrImg, refImg, hrImg])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                left = int(np.floor(np.random.uniform(200 , w - 200 - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(200, h- 200 - self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                # refImgWarped = refImgWarped.crop(crop_region)

        else: 
            left = 1250
            top = 1000
            crop_region = (left, top, left+1024, top+1024)
            lrImg = lrImg.crop(crop_region)
            refImg = refImg.crop(crop_region)
            hrImg = hrImg.crop(crop_region)


        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:
        
            # 采用random_mask进行数据扩充
            # mask = torch.ones(*refImg.shape[1:]).float()
            # mask = random_mask(mask)
            # mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            # refImg = refImg*mask
            # return lrImg, refImg, hrImg, refImgWarped, mask
            
            return lrImg, refImg, hrImg
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDatasetNoOcc1(torch.utils.data.dataset.Dataset):
    # 之前的行人数据集TUD, 有着不同的对齐程度；后面为我所用了, 全图验证
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDatasetNoOcc1, self).__init__() 
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'LR/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print(len(self.imglist))
        # print(self.imglist)

        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
    
    def __getitem__(self, idx):
        lrImg = Image.open(self.imgFolder + 'LR/'+ self.imglist[idx])
        refImg = Image.open(self.imgFolder + 'Ref/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + 'HR2.0/'+ self.imglist[idx])
        # prefix = self.imglist[idx].split('-')[0]
        # print(prefix)
        splitLen = len(self.imglist[idx].split('_'))
        # print(splitLen, self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped/'+ self.imglist[idx])
           
            if self.augment: 
                # refImg = augmentationRef(refImg)
                lrImg, refImg, hrImg, refImgWarped = augmentation([lrImg, refImg, hrImg, refImgWarped])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # if prefix not in ['neg', 'pos']:
                if splitLen < 2:
                    # Image.crop(left, top, right, bottom)
                    left = int(np.floor(np.random.uniform(4 , w - 4 - self.imgSize + 1)))
                    top = int(np.floor( np.random.uniform(4, h- 4 - self.imgSize + 1)))
                else:
                    # print(self.imglist[idx])
                    left = int(np.floor(np.random.uniform(0 , w-self.imgSize + 1)))
                    top = int(np.floor( np.random.uniform(0, h-self.imgSize + 1)))
                    print('should be no used')
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)


        # else:
        #     # if prefix not in ['neg', 'pos']:
        #     if splitLen < 2:
        #         left = 1250
        #         top = 1000
        #         crop_region = (left, top, left+1024, top+1024)
        #     else:
        #         left = 0
        #         top = 0
        #         crop_region = (left, top, 512, 512)
        #     lrImg = lrImg.crop(crop_region)
        #     refImg = refImg.crop(crop_region)
        #     hrImg = hrImg.crop(crop_region)


        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:
            refImgWarped = self.toTensor(refImgWarped)

            # 采用random_mask进行数据扩充
            # mask = torch.ones(*refImg.shape[1:]).float()
            # mask = random_mask(mask)
            # mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            # refImg = refImg*mask
            # return lrImg, refImg, hrImg, refImgWarped, mask
            
            return lrImg, refImg, hrImg, refImgWarped
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDatasetNoOccCrop(torch.utils.data.dataset.Dataset):
    #先从4096x3072的图中裁剪多个520x520的patch, 再进行读取, 旨在充分利用数据
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDatasetNoOccCrop, self).__init__() #为什么dataloader的模型初始化不用super？
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'LR/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print(len(self.imglist))
        # print(self.imglist)

        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
    
    def __getitem__(self, idx):
        lrImg = Image.open(self.imgFolder + 'LR/'+ self.imglist[idx])
        refImg = Image.open(self.imgFolder + 'Ref/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + 'HR/'+ self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped/'+ self.imglist[idx])
           
            if self.augment: 
                # refImg = augmentationRef(refImg)
                lrImg, refImg, hrImg, refImgWarped = augmentation([lrImg, refImg, hrImg, refImgWarped])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                left = int(np.floor(np.random.uniform(0 , w - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(0, h- self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)

        else: 
            left = 1250
            top = 1000
            crop_region = (left, top, left+1024, top+1024)
            lrImg = lrImg.crop(crop_region)
            refImg = refImg.crop(crop_region)
            hrImg = hrImg.crop(crop_region)


        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:
            refImgWarped = self.toTensor(refImgWarped)

            # 采用random_mask进行数据扩充
            # mask = torch.ones(*refImg.shape[1:]).float()
            # mask = random_mask(mask)
            # mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            # refImg = refImg*mask
            # return lrImg, refImg, hrImg, refImgWarped, mask
            
            return lrImg, refImg, hrImg, refImgWarped
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDataset(torch.utils.data.dataset.Dataset):
    # 处理含遮挡的数据
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDataset, self).__init__() #为什么dataloader的模型初始化不用super？
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'LR/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print(len(self.imglist))
        # print(self.imglist)

        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
    
    def __getitem__(self, idx):
        lrImg = Image.open(self.imgFolder + 'LR/'+ self.imglist[idx])
        # refImg = Image.open(self.imgFolder + 'Ref/'+ self.imglist[idx])
        refImg = Image.open(self.imgFolder + 'refOcclusion/'+ self.imglist[idx]) #针对有遮挡的数据
        hrImg = Image.open(self.imgFolder + 'HR/'+ self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped/'+ self.imglist[idx])
            refImgMask = Image.open(self.imgFolder + 'refOcclusion/'+ self.imglist[idx].split('.')[0]+'_mask.png')

            if self.augment:
                refImg, refImgMask = augmentationRef1(refImg, refImgMask)
                lrImg, refImg, hrImg, refImgWarped, refImgMask = augmentation([lrImg, refImg, hrImg, refImgWarped, refImgMask])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                # left = int(np.floor(np.random.uniform(0 , w - self.imgSize + 1)))
                # top = int(np.floor( np.random.uniform(0, h- self.imgSize + 1)))
                left = int(np.floor(np.random.uniform(200 , w - 200 - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(200, h- 200 - self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)
                refImgMask = refImgMask.crop(crop_region)
            
        # else: 
        #     # left = 1250
        #     # top = 1000
        #     center_w = 2048
        #     center_h = 1536
        #     crop_region = (center_w-1500, center_h-1000, center_w+1500, center_h+1000)
        #     lrImg = lrImg.crop(crop_region)
        #     refImg = refImg.crop(crop_region)
        #     hrImg = hrImg.crop(crop_region)

        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:
            refImgWarped = self.toTensor(refImgWarped)
            refImgMask = self.toTensor(refImgMask)

            # 采用random_mask进行数据扩充
            mask = torch.ones(*refImg.shape[1:]).float()
            mask = random_mask(mask)
            mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            refImg = refImg*mask
            # refImgWarped = refImgWarped*mask

            return lrImg, refImg, hrImg, refImgWarped, mask, refImgMask
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    dataset = fusionDataset(imgFolder='./data/', imgSize=256, isTrain=True, augment=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)
    print(len(dataloader))



