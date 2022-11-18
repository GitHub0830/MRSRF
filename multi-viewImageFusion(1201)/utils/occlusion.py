

# import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.nn.functional as F 


'''
在网址：https://blog.csdn.net/dou3516/article/details/109181203
pixelFusionModel_grad.py中的warp函数参考下写出来的
'''

# def get_pixel_value(img, x, y):
#     """
#     Utility function to get pixel value for coordinate vectors x and y from a  4D tensor image.
#     Input
#     - img: tensor of shape (B, H, W, C)
#     - x: flattened tensor of shape (B*H*W, )
#     - y: flattened tensor of shape (B*H*W, )
#     Returns
#     - output: tensor of shape (B, H, W, C)
#     """
#     # shape = tf.shape(x)
#     shape = x.shape
#     print(shape)
#     batch_size = shape[0]
#     height = shape[1]
#     width = shape[2]

#     # if x.is_cuda:
#     #     print('x in cuda')

#     batch_idx = torch.arange(0, batch_size).view(batch_size,-1)#输出tensor为long(int64型)且不包括end元素
#     # print('0', batch_idx.is_cuda)
#     # batch_idx = batch_idx.reshape(batch_size, 1, 1)
#     b = batch_idx.view(batch_size, 1, 1).repeat(1, height, width)
#     print(b.shape, y.shape, x.shape) #torch.Size([3, 256, 256]) torch.Size([3, 256, 256]) torch.Size([3, 256, 256])
#     if x.is_cuda:
#         b = b.to(x)
#     print('b', b.dtype)
#     # indices = tf.stack([b, y, x], 3)
#     if b.is_cuda:
#         print('b in cuda')
#     indices = torch.stack([b.long(), y.long(), x.long()], dim=1)
#     # print(indices.shape, img.shape)

#     # torch.gather(input, dim, index, out=None) → Tensor
#     return torch.gather(img, dim=1, index=indices) #dim=?


# def get_pixel_value(img, x, y):



def warp(img, flow, H, W):
#    H = 256
#    W = 256

    B, C, H, W = img.size()
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if img.is_cuda:
        # grid = grid.cuda() #
        grid = grid.to(img)
    flows = Variable(grid) + flow


    max_y = torch.tensor(H-1).int().to(img) #dtype=torch.int32
    max_x = torch.tensor(W-1).int().to(img) 

    sample_grid_x = flows[:,0,:,:]
    sample_grid_y = flows[:,1,:,:]
    sample_grid_x = sample_grid_x / (W - 1) * 2 - 1
    sample_grid_y = sample_grid_y / (H - 1) * 2 - 1
    sample_grid = torch.cat([sample_grid_x.unsqueeze(3), sample_grid_y.unsqueeze(3)], dim=3)
    output = F.grid_sample(img, sample_grid, align_corners=True)
    # print(output.shape) #torch.Size([3, 2, 256, 256])
    return output

    # # print('0',x.is_cuda)
    # x0 = x.to(x)
    # y0 = y.to(x)
    # x0 = x0.int()
    # x1 = x0 + 1
    # y0 = y0.int()
    # y1 = y0 + 1
    # # clip to range [0, H/W] to not violate img boundaries
    # x0 = torch.clamp(x0, 0, max_x)
    # x1 = torch.clamp(x1, 0, max_x)
    # y0 = torch.clamp(y0, 0, max_y)
    # y1 = torch.clamp(y1, 0, max_y)
    # # get pixel value at corner coords
    # Ia = get_pixel_value(img, x0, y0)
    # Ib = get_pixel_value(img, x0, y1)
    # Ic = get_pixel_value(img, x1, y0)
    # Id = get_pixel_value(img, x1, y1)
    # # recast as float for delta calculation
    # x0 = x0.float()
    # x1 = x1.float()
    # y0 = y0.float()
    # y1 = y1.float()
    # # calculate deltas
    # wa = (x1-x) * (y1-y)
    # wb = (x1-x) * (y-y0)
    # wc = (x-x0) * (y1-y)
    # wd = (x-x0) * (y-y0)
    # # add dimension for addition
    # wa = torch.unsqueeze(wa, dim=1)
    # wb = torch.unsqueeze(wb, dim=1)
    # wc = torch.unsqueeze(wc, dim=1)
    # wd = torch.unsqueeze(wd, dim=1)
    # # compute output
    # out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    # return out

def length_sq(x):
    # return tf.reduce_sum(tf.square(x), 3, keepdims=True)  
    return torch.sum(torch.pow(x, 2), dim=1, keepdims=True)

def occlusion(flow_fw, flow_bw):
    ## from SelFlow: Self-Supervised Learning of Optical Flow
    ## https://github.com/ppliuboy/SelFlow/blob/master/utils.py
    # x_shape = tf.shape(flow_fw)
    x_shape = flow_fw.shape
    H = x_shape[2]
    W = x_shape[3]    
    flow_bw_warped = warp(flow_bw, flow_fw, H, W)
    flow_fw_warped = warp(flow_fw, flow_bw, H, W)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw =  0.05 * mag_sq_fw + 0.5
    occ_thresh_bw =  0.05 * mag_sq_bw + 0.5
    # occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh_fw, tf.float32)
    # occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh_bw, tf.float32)
    occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()
    return occ_fw, occ_bw

if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    flow_fw = torch.randn(3, 2, 256, 256).cuda()
    flow_bw = torch.randn(3, 2, 256, 256).cuda()
    occ_fw, occ_bw = occlusion(flow_fw, flow_bw)
    print(occ_bw.shape)
