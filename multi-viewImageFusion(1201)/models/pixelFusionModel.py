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
    # print('x:', x.size())
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        # grid = grid.cuda() #
        grid = grid.to(x)
    # print('grid:', grid.size())
    # print('flow:', flow.size())
    vgrid = Variable(grid) + flow

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask


def fusion(in_channel, level):
    ops = []
    ops.append(nn.Conv2d(in_channel, 2**(level+4), 3, 1, 1))
    ops.append(nn.ReLU())
    ops.append(nn.Conv2d(2**(level+4), 2**(level+4), 3, 1, 1))
    ops.append(nn.ReLU())
    return nn.Sequential(*ops)


class pixelFusionNet(nn.Module):
    def __init__(self, Level=8):
        super(pixelFusionNet, self).__init__()
        self.level = Level

        # 提特征时各层之间不共享权重
        A0 = []
        for i in range(self.level):
            A0.append(
                nn.Sequential(
                    nn.Conv2d(3, 2**(4+0), 3, 1, 1),
                    nn.Conv2d(2**(4+0), 2**(4+0), 3, 1, 1),
                )
            )
        self.A0 = nn.ModuleList(A0)

        A1 = []
        for i in range(self.level-1):
            A1.append(
                nn.Sequential(
                    nn.Conv2d(2**(4+0), 2**(4+1), 3, 1, 1),
                    nn.Conv2d(2**(4+1), 2**(4+1), 3, 1, 1),
                )
            )
        self.A1 = nn.ModuleList(A1)

        A2 = []
        for i in range(self.level-2):
            A2.append(
                nn.Sequential(
                    nn.Conv2d(2**(4+1), 2**(4+2), 3, 1, 1),
                    nn.Conv2d(2**(4+2), 2**(4+2), 3, 1, 1),
                )
            )
        self.A2 = nn.ModuleList(A2)

        self.avgPooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flowPredict0 = residualFlowPredict(in_channel=2* 2**4) #前面那个系数2是我加的，对于特征的通道数我还不是很清楚
        self.flowPredict1 = residualFlowPredict(in_channel=2* (2**4 + 2**5))
        self.flowPredict = residualFlowPredict(in_channel=2 *(2**4 + 2**5 + 2**6))

        conv_before_fusion = []
        for l in range(0, Level-1):
            if l == 0:
                conv_before_fusion.append(nn.Conv2d(2**(l+1+4), 2*2*2**(l+4), 3, 1, 1)) #卷积核大小是我自己写的
            elif l == 1: 
                conv_before_fusion.append(nn.Conv2d(2**(l+1+4), 2*2*2**(l+4), 3, 1, 1))
            else:
                conv_before_fusion.append(nn.Conv2d(2**(l+1+4), 2*2*2**(l+4), 3, 1, 1))
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
        # print(x1.requires_grad)
        ## 构建图像金字塔
        targetImg = []
        targetImg.append(x1)
        refImg = []
        refImg.append(x2)
        for i in range(1, self.level):
            targetImg.append(self.avgPooling(targetImg[i-1]))
            refImg.append(self.avgPooling(refImg[i-1]))
        # print(refImg[-1].requires_grad) #False
        
        ## 提取特征
        target_l0 = []
        ref_l0 = []
        for i in range(self.level):
            target_l0.append(self.A0[i](targetImg[i])) #2**4个通道
            ref_l0.append(self.A0[i](refImg[i]))
        # print(ref_l0[0].requires_grad) #True

        target_l1 = []
        ref_l1 = []
        for i in range(self.level-1):
            target_l1.append(self.A1[i](self.avgPooling(target_l0[i]))) #2**5个通道
            ref_l1.append(self.A1[i](self.avgPooling(ref_l0[i])))

        target_l2 = []
        ref_l2 = []
        for i in range(self.level-2):
            target_l2.append(self.A2[i](self.avgPooling(target_l1[i]))) #2**6个通道
            ref_l2.append(self.A2[i](self.avgPooling(ref_l1[i])))
        
        ## 特征聚合
        target_fea = []
        ref_fea = []
        for i in range(self.level):
            if i == 0:
                target_fea.append(target_l0[i])
                ref_fea.append(ref_l0[i])
            elif i == 1:
                target_fea.append(torch.cat((target_l0[i], target_l1[i-1]), dim=1))
                ref_fea.append(torch.cat([ref_l0[i], ref_l1[i-1]], dim=1))
            else:
                target_fea.append(torch.cat((target_l0[i], target_l1[i-1], target_l2[i-2]), dim=1))
                ref_fea.append(torch.cat([ref_l0[i], ref_l1[i-1], ref_l2[i-2]], dim=1))

        ## warping变换
        ref_final = []
        flows_final = [] #为了计算WarpingLoss
        delta_flows = [] #为了计算regularizationLoss
        b, c, h, w = targetImg[self.level-1].size()
        flow_init = torch.zeros(b, 2, h, w, device=x1.device)#.to(x1)
        
        for i in range(0, self.level): #从coarse到fine
            i = self.level-1-i
            ref_init = warp(ref_fea[i], flow_init)
            if i == 0:
                delta_flow = self.flowPredict0(torch.cat((target_fea[i], ref_init), dim=1))
            elif i == 1:
                delta_flow = self.flowPredict1(torch.cat((target_fea[i], ref_init), dim=1))
            else:
                delta_flow = self.flowPredict(torch.cat((target_fea[i], ref_init), dim=1))
            delta_flows.append(delta_flow)
            flow_final = flow_init + delta_flow
            flows_final.append(flow_final)
            ref_final.append(warp(ref_fea[i], flow_final))
            flow_init = nn.functional.interpolate(flow_final*2, scale_factor=2, mode='bilinear', align_corners=True)  ##之前没×2，导致光流好小
        
        #为了计算flow consistency check loss
        # flowsBack_final = [] 
        # flowBack_init = torch.zeros(b, 2, h, w, device=x1.device)#.to(x1)
        
        # for i in range(0, self.level): #从coarse到fine
        #     i = self.level-1-i
        #     target_init = warp(target_fea[i], flowBack_init)
        #     if i == 0:
        #         delta_flowBack = self.flowPredict0(torch.cat((ref_fea[i], target_init), dim=1))
        #     elif i == 1:
        #         delta_flowBack = self.flowPredict1(torch.cat((ref_fea[i], target_init), dim=1))
        #     else:
        #         delta_flowBack = self.flowPredict(torch.cat((ref_fea[i], target_init), dim=1))
        #     # delta_flows.append(delta_flow)
        #     flowBack_final = flowBack_init + delta_flowBack
        #     flowsBack_final.append(flowBack_final)
        #     # ref_final.append(warp(ref_fea[i], flow_final))
        #     flowBack_init = nn.functional.interpolate(flowBack_final, scale_factor=2, mode='bilinear', align_corners=True) 

        ## Fusion融合
        fea_cat = torch.cat((target_fea[-1], ref_final[0]), dim=1)
        fea_fusion = self.fusions[-1](fea_cat)


        for i in range(0, self.level-1): #从coarse到fine
            i_reverse = self.level-2-i

            fea_up = nn.functional.interpolate(fea_fusion, scale_factor=2, mode='nearest')   #
            fea_up = self.conv_before_fusion[i_reverse](fea_up)

            fea_cat = torch.cat((target_fea[i_reverse], ref_final[i+1], fea_up), dim=1)
            fea_fusion = self.fusions[i_reverse](fea_cat)

        output = self.finalConv(fea_fusion)

        return output, flows_final, delta_flows

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    # 下面代码是测试model
    target = torch.randn(2, 3, 256, 256).cuda()
    target.requires_grad = False
    ref = torch.randn(2, 3, 256, 256).cuda()
    ref.requires_grad = False
    model = pixelFusionNet(Level=4).cuda()
    # model.requires_grad = True
    # print(model.requires_grad)
    
    fusionResult = model(target, ref)
    print(fusionResult.size())


    # 下面代码是测试warp函数
    # I = Image.open("./models/test_warp.jpg")
    # I_tensor = transforms.ToTensor()(I)
    # I_tensor = I_tensor.cuda()
    # I_tensor = I_tensor.unsqueeze(dim=0)
    # B, C, H, W = I_tensor.size()
    # flow = torch.ones(B, 2, H, W)*10
    # flow = flow.cuda()
    # output = warp(I_tensor, flow)
    # output = tensor_to_PIL(output)
    # output.save('./output.jpg')



















