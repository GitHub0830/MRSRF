from PIL import Image, ImageOps
from torchvision.transforms import transforms
import torch
import os
import numpy as np
import random


def augmentation(img_lr, img_ref, img_hr, ref_warped, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_lr = ImageOps.flip(img_lr)
        img_ref = ImageOps.flip(img_ref)
        img_hr = ImageOps.flip(img_hr)
        ref_warped = ImageOps.flip(ref_warped)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_lr = ImageOps.mirror(img_lr)
            img_ref = ImageOps.mirror(img_ref)
            img_hr = ImageOps.mirror(img_hr)
            ref_warped = ImageOps.mirror(ref_warped)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_lr = img_lr.rotate(180)
            img_ref = img_ref.rotate(180)
            img_hr = img_hr.rotate(180)
            ref_warped = ref_warped.rotate(180)
            info_aug['trans'] = True
    
    return img_lr, img_ref, img_hr, ref_warped

class fusionDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, imgFolder, imgSize=1024, isTrain=True, augment=True):
        super(fusionDataset, self).__init__() #为什么dataloader的模型初始化不用super？
        self.imgFolder = imgFolder #'./data/'
        allFolders = [f for f in os.listdir(self.imgFolder)]
        allFolders = [f for f in allFolders if len(f.split('.'))==1]
        print(allFolders)
        imglist = []
        for i in range(len(allFolders)):
            for idx in range(1, 26):
                imgName = '%02d' % idx + '.png'
                imglist.append(allFolders[i] + '/LR/' + imgName)
                # print(imgName)
        self.imglist = imglist
        print(len(self.imglist))
    
        self.imgSize = imgSize
        self.isTrain = isTrain 
        self.augment = augment
        self.toTensor = transforms.ToTensor()
    
    
    def __getitem__(self, idx):
        lrImg = Image.open(self.imgFolder + self.imglist[idx])
        # print(self.imgFolder + self.imglist[idx])
        refImg = Image.open(self.imgFolder + self.imglist[idx].replace('LR','Ref'))
        # print(self.imgFolder + self.imglist[idx].replace('LR','Ref'))
        
        # hrImg = Image.open(self.imgFolder + 'HR/'+ self.imglist[idx])

        if self.isTrain:
            # ref经光流warp后的图像，为计算warpingLoss
            refImgWarped = Image.open('/home/disk60t/HUAWEI/data/20201118一阶段训练图片/warped/img2_HR_OF_Mask_TRC/'+ self.imglist[idx])

            if self.augment: #需要先augment再crop?
                lrImg, refImg, hrImg, refImgWarped = augmentation(lrImg, refImg, hrImg, refImgWarped)

            if self.imgSize:
                w, h = lrImg.size #5289x4356
                # Image.crop(left, top, right, bottom)
                left = int(np.floor(np.random.uniform(0 , w - self.imgSize + 1)))
                top = int(np.floor( np.random.uniform(0 , h - self.imgSize + 1)))
                crop_region = (left, top, left+self.imgSize, top+self.imgSize)
                lrImg = lrImg.crop(crop_region)
                refImg = refImg.crop(crop_region)
                hrImg = hrImg.crop(crop_region)
                refImgWarped = refImgWarped.crop(crop_region)
            

        lrImg = self.toTensor(lrImg)
        refImg = self.toTensor(refImg)
        # hrImg = self.toTensor(hrImg)

        if self.isTrain:
            refImgWarped = self.toTensor(refImgWarped)
            return lrImg, refImg, hrImg, refImgWarped
        else:
            # print(self.imglist[idx]) #00158.png
            # print(type(self.imglist[idx])) #<class 'str'>
            return self.imglist[idx], lrImg, refImg

    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    dataset = fusionDataset(imgFolder='./data/', imgSize=256, isTrain=True, augment=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)
    print(len(dataloader))



