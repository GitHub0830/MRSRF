import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

def conv_A(level=0): #特征提取
    convs = []
    convs.append(nn.Conv2d(3, 2**(4+level), 3, 1, 1))
    return nn.Sequential(*convs)

def residualFlowPredict(in_channel):
    ops = []
    ops.append(nn.Conv2d(in_channel, 32, 3, 1, 1))
    ops.append(nn.ReLU())
    ops.append(nn.Conv2d(32, 64, 3, 1, 1))
    ops.append(nn.ReLU())
    ops.append(nn.Conv2d(64, 64, 1, 1, 0))
    ops.append(nn.ReLU())
    ops.append(nn.Conv2d(64, 16, 1, 1, 0))
    ops.append(nn.ReLU())
    ops.append(nn.Conv2d(16, 2, 1, 1, 0))
    return nn.Sequential(*ops)

def warp(x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

  
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flow
    # print(x.is_cuda)
    # print(vgrid.is_cuda)

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask


def fusion(in_channel, level):
    ops = []
    ops.append(nn.Conv2d(in_channel, 2**(level+4), 3, 1, 1))
    ops.append(nn.ReLU())
    ops.append(nn.Conv2d(2**(level+4), 2**(level+4), 3, 1, 1))
    return nn.Sequential(*ops)

class pixelFusionNet(nn.Module):
    def __init__(self, Level=8):
        super(pixelFusionNet, self).__init__()
        self.level = Level
        self.A0 = nn.Conv2d(3, 2**(4+0), 3, 1, 1)
        self.A1 = nn.Conv2d(2**(4+0), 2**(4+1), 3, 1, 1) 
        self.A2 = nn.Conv2d(2**(4+1), 2**(4+2), 3, 1, 1)

        self.avgPooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flowPredict0 = residualFlowPredict(in_channel=2* 2**4) #前面那个系数2是我加的，对于特征的通道数我还不是很清楚
        self.flowPredict1 = residualFlowPredict(in_channel=2* (2**4 + 2**5))
        self.flowPredict = residualFlowPredict(in_channel=2 *(2**4 + 2**5 + 2**6))

        conv_before_fusion = []
        for l in range(0, Level-1):
            if l == 0:
                conv_before_fusion.append(nn.Conv2d(2**(l+1+4), 2*2*2**(l+4), 1, 1, 0)) #卷积核大小是我自己写的
            else: 
                if l == 1:
                    conv_before_fusion.append(nn.Conv2d(2**(l+1+4), 2*2*2**(l+4), 1, 1, 0))
                else:
                    conv_before_fusion.append(nn.Conv2d(2**(l+1+4), 2*2*2**(l+4), 1, 1, 0))
        self.conv_before_fusion = nn.ModuleList(conv_before_fusion)
                    
        fusions = []
        for l in range(Level): #顺序，与融合过程顺序相反
            if l == 0:
                fusions.append(fusion(in_channel=2* 2**4 + 2*2*2**(l+4), level=0))
            else: 
                if l == 1:
                    fusions.append(fusion(in_channel=2* (2**4 + 2**5) + 2*2*2**(l+4), level=1))
                else:
                    if l==Level-1:
                        fusions.append(fusion(in_channel=2 *(2**4 + 2**5 + 2**6), level=l))
                    else:
                        fusions.append(fusion(in_channel=2 *(2**4 + 2**5 + 2**6) + 2*2*2**(l+4), level=l))
        self.fusions = nn.ModuleList(fusions)

        self.finalConv = nn.Conv2d(2**(0+4), 3, 1, 1, 0)

    def forward(self, x1, x2): 
        ## 构建图像金字塔
        targetImg = []
        targetImg.append(x1)
        refImg = []
        refImg.append(x2)
        for i in range(1, self.level):
            targetImg.append(self.avgPooling(targetImg[i-1]))
            refImg.append(self.avgPooling(refImg[i-1]))
        ## 提取特征
        target_l0 = []
        ref_l0 = []
        for i in range(self.level):
            target_l0.append(self.A0(targetImg[i])) #2**4个通道
            ref_l0.append(self.A0(refImg[i]))
        target_l1 = []
        ref_l1 = []
        for i in range(1, self.level):
            target_l1.append(torch.cat((target_l0[i], self.A1(self.avgPooling(target_l0[i-1]))), dim=1)) #2**4+2**5个通道
            ref_l1.append(torch.cat((ref_l0[i], self.A1(self.avgPooling(ref_l0[i-1]))), dim=1))
        target_l2 = []
        ref_l2 = []
        for i in range(1, self.level-1): #有点混乱，但我不知道怎么写比较好
            target_l2.append(torch.cat((target_l1[i], self.A2(self.avgPooling(self.A1(self.avgPooling(target_l0[i-1]))))), dim=1))
            ref_l2.append(torch.cat((ref_l1[i], self.A2(self.avgPooling(self.A1(self.avgPooling(ref_l0[i-1]))))), dim=1))
        
        ## warping变换
        ref_final = []
        b, c, h, w = targetImg[self.level-1].size()
        flow_init = torch.zeros(b, 2, h, w).cuda()
        
        for i in range(self.level-3, -1, -1): #从coarse到fine
            ref_init = warp(ref_l2[i], flow_init)
            delta_flow = self.flowPredict(torch.cat((target_l2[i], ref_init), dim=1))
            flow_final = flow_init + delta_flow
            ref_final.append(warp(ref_l2[i], flow_final))
            flow_final_up = nn.functional.interpolate(flow_final, scale_factor=2, mode='bilinear', align_corners=False) 
            flow_init = flow_final_up
        
        # Level 1
        ref_init = warp(ref_l1[0], flow_init)
        delta_flow = self.flowPredict1(torch.cat((target_l1[0], ref_init), dim=1))
        flow_final = flow_init + delta_flow
        ref_final_l1 = warp(ref_l1[0], flow_final)
        flow_final_up = nn.functional.interpolate(flow_final, scale_factor=2, mode='bilinear', align_corners=False) 
        flow_init = flow_final_up
        # Level 0
        ref_init = warp(ref_l0[0], flow_init)
        delta_flow = self.flowPredict0(torch.cat((target_l0[0], ref_init), dim=1))
        flow_final = flow_init + delta_flow
        ref_final_l0 = warp(ref_l0[0], flow_final)
        # flow_final_up = nn.functional.interpolate(flow_final, scale_factor=2, mode='bilinear', align_corners=False) 
        # flow_init = flow_final_up

        ## Fusion融合
        num = 0
        for i in range(self.level-3, -1, -1):
            if i < self.level-3:
                fea_cat = torch.cat((target_l2[i], ref_final[num], fea_up), dim=1)
                fea_fusion = self.fusions[i+2](fea_cat) 
            else:
                fea_cat = torch.cat((target_l2[i], ref_final[num]), dim=1)
                fea_fusion = self.fusions[i+2](fea_cat)
            
            fea_up = nn.functional.interpolate(fea_fusion, scale_factor=2)  
            fea_up = self.conv_before_fusion[i+1](fea_up)
            num += 1
        
        # Level 1
        fea_cat = torch.cat((target_l1[0], ref_final_l1, fea_up), dim=1)
        # 默认参数就是mode='nearest', align_corners=False，但是我把它在函数里写出来就不对
        fea_fusion = self.fusions[1](fea_cat) 
        fea_up = nn.functional.interpolate(fea_fusion, scale_factor=2) 
        
        fea_up = self.conv_before_fusion[0](fea_up)
        # Level 0
        fea_cat = torch.cat((target_l0[0], ref_final_l0, fea_up), dim=1)
        fea_fusion = self.fusions[0](fea_cat) 

        output = self.finalConv(fea_fusion)
        return output


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    # 下面代码是测试model
    # target = torch.randn(8, 3, 128, 128).cuda()
    # ref = torch.randn(8, 3, 128, 128).cuda()
    # model = pixelFusionNet(Level=8).cuda()
    # fusionResult = model(target, ref)
    # print(fusionResult.size())


    # 下面代码是测试warp函数
    I = Image.open("./models/test_warp.jpg")
    I_tensor = transforms.ToTensor()(I)
    I_tensor = I_tensor.cuda()
    I_tensor = I_tensor.unsqueeze(dim=0)
    B, C, H, W = I_tensor.size()
    flow = torch.ones(B, 2, H, W)*10
    flow = flow.cuda()
    output = warp(I_tensor, flow)
    output = tensor_to_PIL(output)
    output.save('./output.jpg')



















