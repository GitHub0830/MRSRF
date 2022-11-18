from PIL import Image, ImageOps
from torchvision.transforms import transforms
import torch
import os
import numpy as np
import random
import math
import cv2

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

def augmentationRef(ref): 
    if random.random() > 0.0: #0.1
        ref = cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2BGR) #Image to numpy
    
        h, w, _ = ref.shape
        x = random.randint(-5, 5)
        y = random.randint(-5, 5)
        m = np.float32([[1, 0, x], [0, 1, y]])
        ref = cv2.warpAffine(ref, m, (w, h))

        ref = Image.fromarray(cv2.cvtColor(ref,cv2.COLOR_BGR2RGB)) # numpy 转 image
    
    return ref 


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
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDatasetNoOcc, self).__init__() 
        self.imgFolder = imgFolder 
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
        refImg = Image.open(self.imgFolder + 'warped1/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + 'HR/'+ self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped/'+ self.imglist[idx])
           
            if self.augment: #需要先augment再crop?
                refImg = augmentationRef(refImg)
                lrImg, refImg, hrImg, refImgWarped = augmentation([lrImg, refImg, hrImg, refImgWarped])

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                left = int(np.floor(np.random.uniform(200 , w - 200 - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(200, h- 200 - self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)

        # else: #按道理没有这个的，但是由于val图像太大导致out of memeory，所以就裁剪出指定的patch
            # left = 1250
            # top = 1000
            # crop_region = (left, top, left+1024, top+1024)
            # lrImg = lrImg.crop(crop_region)
            # refImg = refImg.crop(crop_region)
            # hrImg = hrImg.crop(crop_region)


        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        if self.isTrain:

            refImgWarped = self.toTensor(refImgWarped)

            # # 采用random_mask进行数据扩充
            # mask = torch.ones(*refImg.shape[1:]).float()
            # mask = random_mask(mask)
            # mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            # refImg = refImg*mask
            
            return lrImg, refImg, hrImg, refImgWarped#, mask
            
            # return lrImg, refImg, hrImg, refImgWarped
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDatasetNoOcc1(torch.utils.data.dataset.Dataset):
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDatasetNoOcc1, self).__init__() #为什么dataloader的模型初始化不用super？
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'HR1.0/')]
        self.imglist = sorted([filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]) #仅获取文件夹中的图像文件
        print('x5', len(self.imglist))
        # print(self.imglist)

        self.toTensor = transforms.ToTensor()
    
    def __getitem__(self, idx):
        # lrImg = Image.open(self.imgFolder + 'LR/'+ self.imglist[idx])
        lrImg_small = cv2.imread(self.imgFolder + 'LR_small/'+ self.imglist[idx])
        lrImg = cv2.resize(lrImg_small, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        lrImg = Image.fromarray(cv2.cvtColor(lrImg, cv2.COLOR_BGR2RGB))
        # lrImg_small = Image.fromarray(cv2.cvtColor(lrImg_small, cv2.COLOR_BGR2RGB))

        refImg = Image.open(self.imgFolder + 'Ref1/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + 'HR1.0/'+ self.imglist[idx])
        # prefix = self.imglist[idx].split('-')[0]
        # print(prefix)

        left = 15
        top = 15
        hr_size = 512
        crop_region = (left, top, left+hr_size, top+hr_size)
        lrImg = lrImg.crop(crop_region)
        refImg = refImg.crop(crop_region)
        hrImg = hrImg.crop(crop_region)
        # crop_region_small = (left//5, top//5, left//5+hr_size//5, top//5+hr_size//5)
        # lrImg_small = lrImg_small.crop(crop_region_small)

        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)
        # lrImg_small = self.toTensor(lrImg_small)

        # print(self.imglist[idx]) #00158.png
        # print(type(self.imglist[idx])) #<class 'str'>
        return self.imglist[idx], lrImg, refImg, hrImg #, lrImg_small

    def __len__(self):
        return len(self.imglist)

class fusionDatasetNoOcc1_x2(torch.utils.data.dataset.Dataset):
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDatasetNoOcc1_x2, self).__init__() #为什么dataloader的模型初始化不用super？
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'HR1.0_x2/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print('x2', len(self.imglist))
        # print(self.imglist)

        self.toTensor = transforms.ToTensor()
    
    def __getitem__(self, idx):
        # lrImg = Image.open(self.imgFolder + 'LR_x2/'+ self.imglist[idx])
        lrImg = cv2.imread(self.imgFolder + 'LR_small_x2/'+ self.imglist[idx])
        lrImg = cv2.resize(lrImg, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        lrImg = Image.fromarray(cv2.cvtColor(lrImg, cv2.COLOR_BGR2RGB))

        refImg = Image.open(self.imgFolder + 'Ref1_x2/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + 'HR1.0_x2/'+ self.imglist[idx])
        prefix = self.imglist[idx].split('-')[0]
        # print(prefix)

        left = 0
        top = 0
        crop_region = (left, top, left+512, top+512)
        lrImg = lrImg.crop(crop_region)
        refImg = refImg.crop(crop_region)
        hrImg = hrImg.crop(crop_region)

        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        # print(self.imglist[idx]) #00158.png
        # print(type(self.imglist[idx])) #<class 'str'>
        return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)

class fusionDatasetNoOcc1_x4(torch.utils.data.dataset.Dataset):
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDatasetNoOcc1_x4, self).__init__() #为什么dataloader的模型初始化不用super？
        self.imgFolder = imgFolder #'./data/'
        allfiles = [f for f in os.listdir(self.imgFolder + 'HR1.0_x4/')]
        self.imglist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]] #仅获取文件夹中的图像文件
        print('x4', len(self.imglist))
        # print(self.imglist)

        self.toTensor = transforms.ToTensor()
    
    def __getitem__(self, idx):
        # lrImg = Image.open(self.imgFolder + 'LR_x2/'+ self.imglist[idx])
        lrImg = cv2.imread(self.imgFolder + 'LR_small_x4/'+ self.imglist[idx])
        lrImg = cv2.resize(lrImg, dsize=(0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        lrImg = Image.fromarray(cv2.cvtColor(lrImg, cv2.COLOR_BGR2RGB))

        refImg = Image.open(self.imgFolder + 'Ref1_x4/'+ self.imglist[idx])
        hrImg = Image.open(self.imgFolder + 'HR1.0_x4/'+ self.imglist[idx])
        prefix = self.imglist[idx].split('-')[0]
        # print(prefix)

        left = 0
        top = 0
        crop_region = (left, top, left+512, top+512)
        lrImg = lrImg.crop(crop_region)
        refImg = refImg.crop(crop_region)
        hrImg = hrImg.crop(crop_region)

        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        hrImg = self.toTensor(hrImg)

        # print(self.imglist[idx]) #00158.png
        # print(type(self.imglist[idx])) #<class 'str'>
        return self.imglist[idx], lrImg, refImg, hrImg

    def __len__(self):
        return len(self.imglist)
        
class fusionDataset(torch.utils.data.dataset.Dataset):
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
        refImg = Image.open(self.imgFolder + 'refOcclusion35/'+ self.imglist[idx]) #针对有遮挡的数据
        hrImg = Image.open(self.imgFolder + 'HR/'+ self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open(self.imgFolder + 'refWarped/'+ self.imglist[idx])
            refImgMask = Image.open(self.imgFolder + 'refOcclusion35/'+ self.imglist[idx].split('.')[0]+'_mask.png')

            if self.augment: #需要先augment再crop?
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
            refImgMask = self.toTensor(refImgMask)

            # 采用random_mask进行数据扩充
            mask = torch.ones(*refImg.shape[1:]).float()
            mask = random_mask(mask)
            mask = mask.view(1, refImg.shape[1], refImg.shape[2])
            refImg = refImg*mask

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



