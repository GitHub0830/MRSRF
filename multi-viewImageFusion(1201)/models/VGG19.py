# from models.submodules import *
import torch
import torchvision.models
import os

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        '''
         use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = [2, 7, 14]
        vgg19 = torchvision.models.vgg19(pretrained=True)

        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1]+1])

        # 参考https://github.com/crowsonkb/vgg_loss/blob/master/vgg_loss.py 添加的
        self.model.eval()
        self.model.requires_grad_(False)


    def forward(self, x):
        x = (x-0.5)/0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)

        return features


class VGG19forPixelFusion(torch.nn.Module):
    def __init__(self):
        super(VGG19forPixelFusion, self).__init__()
        '''
         use vgg19 conv1_2, conv2_2, conv3_2, conv4_2, conv5_2 feature
         "Photographic Image Synthesis with Cascaded Refinement Networks"
        '''
        self.feature_list = [2, 7, 12, 21, 30]
        vgg19 = torchvision.models.vgg19(pretrained=True)

        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1]+10]) #把前31层(0~30)都存进model
        # print(self.feature_list[-1]+1) #31
        # print(self.model) #各layer的定义

    def forward(self, x):
        x = (x-0.5)/0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
        return features


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    # 下面代码是测试model
    x = torch.randn(2, 3, 256, 256).cuda()
    model = VGG19forPixelFusion().cuda()
    out = model(x)
    # print(out)


