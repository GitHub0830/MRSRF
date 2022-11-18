
import torch
import torch.nn as nn
import os
import sys 
sys.path.append("..")
from models.pixelFusionModel import warp
from models.VGG19 import VGG19forPixelFusion

import torch.nn.functional as F
import math


def smoothLoss(flows, grad): #https://blog.csdn.net/weixin_42923416/article/details/106024897 参考但并没有借鉴
    flow_v = (flows[-1][:, 0]).unsqueeze(1)
    flow_h = (flows[-1][:, 1]).unsqueeze(1)
    grad_h = grad[0] #(b,c,h,w) 水平
    grad_v = grad[1] #竖直方向

    mse_loss = nn.MSELoss(reduction = 'mean') #elementwise_mean

    grad_v_weight = grad_v[:,:,:-1,:-1]
    flow_v_left = flow_v[:,:,:-1,:-1] * (1-grad_v_weight)
    flow_v_right = flow_v[:,:,:-1,1:] * (1-grad_v_weight)
    flow_v_top = flow_v[:,:,:-1,:-1] * (1-grad_v_weight)
    flow_v_bottom = flow_v[:,:,1:,:-1] * (1-grad_v_weight)
    MSE_v = (mse_loss(flow_v_left,flow_v_right) + mse_loss(flow_v_top,flow_v_bottom))

    grad_h_weight = grad_h[:,:,:-1,:-1] 
    flow_h_left = flow_h[:,:,:-1,:-1] * (1-grad_h_weight)
    flow_h_right = flow_h[:,:,:-1,1:] * (1-grad_h_weight)
    flow_h_top = flow_h[:,:,:-1,:-1] * (1-grad_h_weight)
    flow_h_bottom = flow_h[:,:,1:,:-1] * (1-grad_h_weight)
    MSE_h = (mse_loss(flow_h_left,flow_h_right) + mse_loss(flow_h_top,flow_h_bottom))

    return (MSE_v + MSE_h)/4.0
    

def mseLoss(output, target):
    mse_loss = nn.MSELoss(reduction = 'mean') #elementwise_mean
    MSE = mse_loss(output, target)
    return MSE

def charbonnierLoss(pred, target, eps=1e-4):
    return torch.mean(torch.sqrt((pred - target)**2 + eps**2))


def PSNR(output, target, max_diff=1.0):
    # assert max_diff in [255, 1, 2]
    # if max_diff == 255:
    #     output = output.clamp(0.0, 255.0)
    # elif max_diff == 1:
    #     output = output.clamp(0.0, 1.0)
    # elif max_diff == 2:
    #     output = output.clamp(-1.0, 1.0)
    output = output.clamp(0.0,1.0)
    # print(torch.max(target())
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_diff**2 / mse)


# 以下2个函数来自DCSR
def quantize(img, rgb_range=1.0):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
def calc_psnr(sr, hr, scale, rgb_range=1.0, dataset=None):
    if hr.nelement() == 1: return 0
    if (sr.shape != hr.shape):
        # print(sr.shape, hr.shape) 
        # torch.Size([1, 3, 6016, 8064]) torch.Size([1, 3, 3024, 4032])
        h_min = min(sr.shape[2], hr.shape[2])
        w_min = min(sr.shape[3], hr.shape[3])
        sr = sr[:,:,:h_min, :w_min]
        hr = hr[:,:,:h_min, :w_min]
    # cv2.imwrite('./sr.png', (sr.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
    # cv2.imwrite('./hr.png', (hr.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0)[:,:,::-1].astype(np.uint8))
    # a

    diff = (sr - hr) / rgb_range
    # if dataset:
    #     shave = scale
    #     if diff.size(1) > 1:
    #         gray_coeffs = [65.738, 129.057, 25.064]
    #         convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    #         diff = diff.mul(convert).sum(dim=1)
    # else:
    shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * torch.log10(mse) #torch.log10和math.log10的差异大吗？


def perceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''
   
    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]
    # print('perceptualLoss：', loss)
    return loss


## Loss for pixelImageFusionNet

def l1_loss(R, F):
    '''
    参考对象：https://github.com/Blade6570/PhotographicImageSynthesiswithCascadedRefinementNetworks-Pytorch/blob/master/Cascaded_Network_LM_256.py
    '''
    # E=torch.mean(torch.mean(label_images* torch.mean(torch.abs(R-F),1).unsqueeze(1),2),2)
    E = torch.mean(torch.abs(R-F))
    return E 

def perceptualLoss2(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_2, conv4_2, conv5_2 feature 
    '''
   
    weights = [1, 1/1.6, 1/2.3, 1/1.8, 1/2.8, 10/0.8] #[1, 0.625, 0.4347]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
  
    # print('len(VGG_features_real)=3?', len(features_real)) #3
    loss = 0
    # print(loss)
    # loss = loss.cuda()
    loss += l1_loss(fakeIm, realIm)*weights[0]
    for i in range(len(features_real)):
        loss_i = l1_loss(features_fake[i], features_real_no_grad[i]) #加了个cuda()就不报错了？
        # print('loss', loss)
        # print('loss_i', loss_i)
        loss = loss + loss_i * weights[i+1]
    return loss


def warpingLoss(refImg, GT, flows, level, vggnet):
    ref = refImg
    gt = GT
    loss = 0
    #torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, 
    # ceil_mode=False, count_include_pad=True)
    ref = nn.functional.avg_pool2d(ref, kernel_size=2, stride=2)
    gt = nn.functional.avg_pool2d(gt, kernel_size=2, stride=2)
    warpedRef = warp(ref, flows[level-2]) #仅仅倒数第2层

    loss += perceptualLoss2(warpedRef, gt, vggnet)
    return loss

def warpingLoss_randomMask0(refImg, GT, flows, mask, isObjectMask, level, vggnet):
    ref = refImg
    gt = GT
    loss = 0
    ref = nn.functional.avg_pool2d(ref, kernel_size=2, stride=2)
    gt = nn.functional.avg_pool2d(gt, kernel_size=2, stride=2)

    ##针对有遮挡的ref
    if isObjectMask:
        mask = 1 - mask
        ref = ref * mask
    warpedRef = warp(ref, flows[level-2]) #仅仅倒数第2层   

    M = nn.functional.interpolate(mask, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
    warpedMask = warp(M, flows[level-2])
    loss += perceptualLoss2(warpedRef, gt*warpedMask, vggnet)

    return loss

def warpingLoss_randomMask(refImg, GT, flows, mask, isObjectMask, objectMask, level, vggnet):
    ref = refImg
    gt = GT
    loss = 0
    gt = nn.functional.avg_pool2d(gt, kernel_size=2, stride=2)

    MASK = mask
    ##针对有遮挡的ref
    if isObjectMask:
        objectMask = 1 - objectMask
        ref = ref * objectMask
        MASK = mask * objectMask
    ref = nn.functional.avg_pool2d(ref, kernel_size=2, stride=2)

    warpedRef = warp(ref, flows[level-2]) #仅仅倒数第2层   
    M = nn.functional.interpolate(MASK, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
    warpedMask = warp(M, flows[level-2])
    loss += perceptualLoss2(warpedRef, gt*warpedMask, vggnet)

    return loss

def warpingLoss_padding(refImg, GT, flows, vggnet):
    ref = refImg
    gt = GT
    loss = 0
    #torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, 
    # ceil_mode=False, count_include_pad=True)
    ref = nn.functional.avg_pool2d(ref, kernel_size=2, stride=2)
    gt = nn.functional.avg_pool2d(gt, kernel_size=2, stride=2)
    warpedRef = warp(ref, flows) #仅仅倒数第2层
    loss += perceptualLoss2(warpedRef, gt, vggnet)
    return loss

def regularizationLoss(delta_flows): #约束每个deltaFlow小
    loss = 0
    # print(len(delta_flows)) #4
    for i in range(len(delta_flows)):
        # loss += torch.norm(delta_flows[i]) #tensor([ 97.6682, 100.8805]
        # delta_flows_vector0 = delta_flows[i][:,0,:,:].contiguous().view(1, -1)
        # delta_flows_vector1 = delta_flows[i][:,1,:,:].contiguous().view(1, -1)
        # loss += (torch.norm(delta_flows_vector0) + torch.norm(delta_flows_vector1))/2
        # loss1 += torch.sqrt(delta_flows[i][:,0,:,:]**2 + delta_flows[i][:,1,:,:]**2).mean()
        # loss1 += torch.sqrt( (delta_flows[i][:,0,:,:]**2 + delta_flows[i][:,1,:,:]**2).sum())
        # loss1 += (torch.sqrt( (delta_flows[i][:,0,:,:]**2).sum()) + torch.sqrt( (delta_flows[i][:,1,:,:]**2).sum()))/2
        loss += torch.sqrt((delta_flows[i]**2).sum(1)).mean()
    return loss

def regularizationLoss1(delta_flows): #约束deltaFlow的累加和要小
    loss = 0
    delta_flows_acc = delta_flows[-1]    
    for level in range(len(delta_flows)-1): # i越小图像分辨率越抵
        tmp = delta_flows[level]#.detach()
        for j in range(len(delta_flows)-1-level): # 需要上采样的次数
            tmp = nn.functional.interpolate(tmp*2, scale_factor=2, mode='bilinear', align_corners=True)  ##之前没×2，导致光流好小
        
        delta_flows_acc = delta_flows_acc + tmp  

    loss = torch.sqrt((delta_flows_acc**2).sum(1)).mean()
    return loss

def regularizationLoss_padding(delta_flows, padNum):
    loss = 0
    # print(len(delta_flows)) #4
    for i in range(len(delta_flows)):
        _,_,h,w = delta_flows[i].shape
        padNum_i = padNum // (2**(len(delta_flows)-i-1))
        loss += torch.sqrt((delta_flows[i][:, :, padNum_i:h-padNum_i, padNum_i:w-padNum_i]**2).sum(1)).mean()
    return loss

## 专门针对grad 约束直线
_reduction_modes = ['none', 'mean', 'sum']
# @weighted_loss
def grad_l1Loss(pred, target, reduction='mean'):
    return F.l1_loss(pred, target, reduction=reduction)

class gradL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(gradL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. 'f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return grad_l1Loss(pred, target, reduction=self.reduction)

## 使得无ref区域也能生成较好的融合结果
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 测perceptualLoss2
    # vggnet = VGG19forPixelFusion()
    # if torch.cuda.is_available():
    #     vggnet = torch.nn.DataParallel(vggnet).cuda()
    # x = torch.randn(2, 3, 256, 256).cuda()
    # y = torch.randn(2, 3, 256, 256).cuda()
    # loss = perceptualLoss2(x, y, vggnet)

    # 测warpingLoss
    # vggnet = VGG19forPixelFusion()
    # if torch.cuda.is_available():
    #     vggnet = torch.nn.DataParallel(vggnet).cuda()
    # x = torch.randn(2, 3, 256, 256).cuda()
    # y = torch.randn(2, 3, 256, 256).cuda()
    # loss = warpingLoss(x, y, 4, vggnet)
    # print(loss)

    # 测regularizationLoss
    # a = torch.randn(3, 2, 4, 4).cuda()
    # b = torch.randn(3, 2, 4, 4).cuda()
    # delta_flows = [a, b]
    # loss, loss1 = regularizationLoss(delta_flows)
    # print(loss, loss1)

    # 测charbonnierLoss
    a = torch.randn(1, 3, 2, 2).cuda()
    b = torch.randn(1, 3, 2, 2).cuda()
    print(a)
    print(b)
    loss = l1_loss(a, b)
    loss1 = charbonnierLoss(a, b)
    print(loss, loss1)