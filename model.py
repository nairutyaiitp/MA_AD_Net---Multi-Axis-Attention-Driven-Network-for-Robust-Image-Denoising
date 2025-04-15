import os
import math
import torch
import torch.nn as nn
import numpy as np
import functools
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 8

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


class DCTLayer(nn.Module):
    """DCT Transform Layer"""
    def __init__(self, size=8):
        super(DCTLayer, self).__init__()
        self.size = size
        
    def forward(self, x):
        # This is a placeholder for DCT transformation
        # In a real implementation, you would use proper DCT calculations
        # For now, we'll use a simulated version with convolutions
        b, c, h, w = x.shape
        x_blocks = x.view(b, c, h // self.size, self.size, w // self.size, self.size)
        x_blocks = x_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_blocks = x_blocks.view(b, c, h // self.size * w // self.size, self.size * self.size)
        
        # Simulated DCT transformation
        simulated_dct = F.adaptive_avg_pool2d(x, (h // 2, w // 2))
        simulated_dct = F.interpolate(simulated_dct, size=(h, w), mode='bilinear', align_corners=False)
        
        return simulated_dct


# Spatial Attention Module as shown in the diagram
class SAM(nn.Module):
    def __init__(self, channels=32, kernel_size=3, stride=1, bias=False):
        super().__init__()
        # Sequential convolution and PReLU pairs as in SAM
        self.seq1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), bias=bias),
            nn.PReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0, bias=bias),
            nn.PReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), bias=bias),
            nn.PReLU()
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0, bias=bias),
            nn.PReLU()
        )
        
    def forward(self, x):
        x1 = self.seq1(x)
        x2 = self.seq2(x1)
        x3 = self.seq3(x2)
        x4 = self.seq4(x3)
        return x4


# Channel Attention Module as shown in the diagram
class CAM(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CAM, self).__init__()
        
        # Max pooling branch
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_conv = nn.Conv2d(channel, channel, 1, padding=0, bias=bias)
        
        # Average pooling branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_conv = nn.Conv2d(channel, channel, 1, padding=0, bias=bias)
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(channel, channel, 1, padding=0, bias=bias)
        
    def forward(self, x):
        # Max pooling path
        max_path = self.max_pool(x)
        max_path = self.max_conv(max_path)
        
        # Avg pooling path
        avg_path = self.avg_pool(x)
        avg_path = self.avg_conv(avg_path)
        
        # Combine paths and apply final conv
        y = max_path + avg_path
        y = self.final_conv(y)
        
        # Apply attention
        return x * y.sigmoid() + x


# Spatial Attention Layer
class SpatialAttention(nn.Module):
    def __init__(self, channel, bias=False):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel//2, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//2, 1, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.conv(x)
        return x * y + x


# Multi-Axis Feature Fusion Block
class MAFF(nn.Module):
    def __init__(self, channels, bias=False):
        super(MAFF, self).__init__()
        
        # Different kernel size convolutions for multi-axis feature extraction
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0, bias=bias),
            nn.PReLU()
        )
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.PReLU()
        )
        self.conv5x5_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, bias=bias),
            nn.PReLU()
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, bias=bias),
            nn.PReLU()
        )
        self.conv5x5_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, bias=bias),
            nn.PReLU()
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.PReLU()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0, bias=bias),
            nn.PReLU()
        )
        
        # Feature fusion for the first group (1,3,5,7)
        self.fusion1 = nn.Conv2d(channels*4, channels, 1, bias=bias)
        # Feature fusion for the second group (5,3,1)
        self.fusion2 = nn.Conv2d(channels*3, channels, 1, bias=bias)
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1, bias=bias),
            nn.Sigmoid()
        )
        
        # DCT layers
        self.dct_in = DCTLayer()
        self.dct_out = DCTLayer()
        
    def forward(self, x):
        # Apply DCT
        x_dct = self.dct_in(x)
        
        # First branch - top path
        c1_1 = self.conv1x1_1(x_dct)
        c3_1 = self.conv3x3_1(x_dct)
        c5_1 = self.conv5x5_1(x_dct)
        c7 = self.conv7x7(x_dct)
        
        # Fusion of first branch
        f1 = torch.cat([c1_1, c3_1, c5_1, c7], dim=1)
        f1 = self.fusion1(f1)
        
        # Second branch - bottom path
        c5_2 = self.conv5x5_2(x_dct)
        c3_2 = self.conv3x3_2(x_dct)
        c1_2 = self.conv1x1_2(x_dct)
        
        # Fusion of second branch
        f2 = torch.cat([c5_2, c3_2, c1_2], dim=1)
        f2 = self.fusion2(f2)
        
        # Final fusion
        f = torch.cat([f1, f2], dim=1)
        f = self.final_fusion(f)
        
        # Apply inverse DCT and add residual
        out = self.dct_out(f * x_dct)
        return out


# Dilated Multi-scale Attention Module
class DMAM(nn.Module):
    def __init__(self, channels, bias=False):
        super(DMAM, self).__init__()
        
        # Convolution layers with dilated convolutions
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=bias)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=bias)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=6, dilation=6, bias=bias)
        self.prelu3 = nn.PReLU()
        
        # Final fusion
        self.fusion = nn.Conv2d(channels*3, channels, 1, bias=bias)
        
    def forward(self, x):
        c1 = self.prelu1(self.conv1(x))
        c2 = self.prelu2(self.conv2(x))
        c3 = self.prelu3(self.conv3(x))
        
        cat = torch.cat([c1, c2, c3], dim=1)
        out = self.fusion(cat)
        return out


# MultiAxis Encoder Block
class MAEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, down=True):
        super(MAEncoder, self).__init__()
        
        # Sobel-Based Gradient extraction
        self.edge_detect = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.PReLU()
        )
        
        # Main convolution path
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=False),
            nn.PReLU()
        )
        
        # Spatial and Channel Attention
        self.sam = SAM(out_channels)
        self.cam = CAM(out_channels)
        
        # Fusion
        self.fusion = nn.Conv2d(out_channels*2, out_channels, 1, bias=False)
        
        self.down = down
        if self.down:
            self.down_sample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)
            
    def forward(self, x):
        edge = self.edge_detect(x)
        x = x + edge
        
        conv = self.conv(x)
        
        # Apply attention
        sam_out = self.sam(conv)
        cam_out = self.cam(conv)
        
        # Fusion
        fused = torch.cat([sam_out, cam_out], dim=1)
        conv_out = self.fusion(fused)
        
        if self.down:
            down = self.down_sample(conv_out)
            return conv_out, down
        else:
            return conv_out


# MultiAxis Decoder Block
class MADecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MADecoder, self).__init__()
        
        # Upsampling
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.PReLU()
        )
        
        # Fusion with skip connection
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=False),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=False),
            nn.PReLU()
        )
        
        # DMAM module
        self.dmam = DMAM(out_channels)
        
    def forward(self, x, skip):
        up = self.up_sample(x)
        cat = torch.cat([up, skip], dim=1)
        fused = self.fusion(cat)
        out = self.dmam(fused)
        return out


# Transformer components
class LayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, h, w):
        return self.fn(self.norm(x), h, w)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0, bias=False),
            nn.GELU(),
        )
        self.out_linear = nn.Linear(hidden_dim, dim)
        
    def forward(self, x, h, w):
        b = x.shape[0]
        x = self.net(x)
        x = x.transpose(1, 2).contiguous().view(b, self.hidden_dim, h, w)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.out_linear(x)
        return x


class MSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        
    def forward(self, x, h, w):
        b = x.shape[0]
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class TAB(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNorm(dim, MSA(dim, heads=heads, dim_head=dim_head)),
                LayerNorm(dim, FFN(dim, mlp_dim))
            ]))
            
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        for attn, ff in self.layers:
            x = attn(x, H, W) + x
            x = ff(x, H, W) + x
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return x


# Main Attention Driven Block
class AD_Block(nn.Module):
    def __init__(self, first=True, channel=32, num_head=8, dim_head=64, mlp_dim=512, tab_depth=2):
        super(AD_Block, self).__init__()
        
        self.first_channel = 3
        self.first = first
        
        if not self.first:
            self.transmission = conv(3, channel, 3)
            self.gate = SAM(channels=channel)
            self.first_channel = channel*2
            
        # Edge detection
        self.edge_detect = nn.Sequential(
            nn.Conv2d(self.first_channel, self.first_channel, 3, padding=1, groups=self.first_channel, bias=False),
            nn.PReLU()
        )
            
        # Downsampling path
        self.down_1 = MAEncoder(in_channels=self.first_channel, out_channels=channel)
        self.down_2 = MAEncoder(in_channels=channel, out_channels=channel*2)
        self.down_3 = MAEncoder(in_channels=channel*2, out_channels=channel*4)
        
        # Bottleneck with MAFF and transformer
        self.maff = MAFF(channel*4)
        self.bottle_conv = conv(channel*4, channel*8, 3)
        self.former = TAB(channel*8, heads=num_head, dim_head=dim_head, mlp_dim=mlp_dim, depth=tab_depth)
        
        # Upsampling path
        self.up_4 = MADecoder(in_channels=channel*8, out_channels=channel*4)
        self.up_5 = MADecoder(in_channels=channel*4, out_channels=channel*2)
        self.up_6 = MADecoder(in_channels=channel*2, out_channels=channel)
        
        # Final layers for noise map estimation
        self.final_conv1 = conv(channel, channel//2, 3)
        self.final_conv2 = conv(channel//2, 3, 3)
        self.act = nn.PReLU()
        
    def forward(self, input, pre_f=None):
        N_input = input
        
        # If not first stage, use previous features
        if not self.first and pre_f is not None:
            pre_f = self.gate(pre_f)
            conv = self.transmission(input)
            N_input = torch.cat([conv, pre_f], dim=1)
            
        # Edge enhancement
        edge = self.edge_detect(N_input)
        N_input = N_input + edge
            
        # Encoder
        conv1, down1 = self.down_1(N_input)
        conv2, down2 = self.down_2(down1)
        conv3, down3 = self.down_3(down2)
        
        # Bottleneck with MAFF and transformer
        maff_out = self.maff(down3)
        bottle = self.act(self.bottle_conv(maff_out))
        conv4 = self.former(bottle)
        
        # Decoder
        conv5 = self.up_4(conv4, conv3)
        conv6 = self.up_5(conv5, conv2)
        conv7 = self.up_6(conv6, conv1)
        
        # Noise map estimation
        x = self.act(self.final_conv1(conv7))
        noise_map = self.final_conv2(x)
        
        # Clean image estimation
        output = input - noise_map
        
        return output, conv7


# MA_AD_Net: MultiAxis Attention Driven Network for robust image denoising
class MA_AD_Net(nn.Module):
    def __init__(self, channel=32, num_head=8, dim_head=64, mlp_dim=512, tab_depth=2):
        super(MA_AD_Net, self).__init__()
        
        # Progressive stages
        self.ad_block_1 = AD_Block(first=True, channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, tab_depth=tab_depth)
        self.ad_block_2 = AD_Block(first=False, channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, tab_depth=tab_depth)
        self.ad_block_3 = AD_Block(first=False, channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, tab_depth=tab_depth)
        self.ad_block_4 = AD_Block(first=False, channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, tab_depth=tab_depth)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            
    def forward(self, input):
        # Apply padding
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        
        # Progressive refinement
        output_1, transmission = self.ad_block_1(input)
        output_2, transmission = self.ad_block_2(output_1, transmission)
        output_3, transmission = self.ad_block_3(output_2, transmission)
        output_4, _ = self.ad_block_4(output_3, transmission)
        
        # Remove padding
        output_1 = pad_tensor_back(output_1, pad_left, pad_right, pad_top, pad_bottom)
        output_2 = pad_tensor_back(output_2, pad_left, pad_right, pad_top, pad_bottom)
        output_3 = pad_tensor_back(output_3, pad_left, pad_right, pad_top, pad_bottom)
        output_4 = pad_tensor_back(output_4, pad_left, pad_right, pad_top, pad_bottom)
        
        return output_1, output_2, output_3, output_4


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    input_size = 256
    model = MA_AD_Net(channel=40).cuda()

    macs, params = get_model_complexity_info(model, (3, input_size, input_size), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)

    '''hw * hw * num_heads * head_dims * 2 * num_tab / 1e9  +
       channel * channel * h * w * 2 * num_sam / 1e9'''
    att_flops = 32*32*32*32*8*64*2*8/1e9 + 40*40*256*256*6/1e9

    macs = float(macs.split(' ')[0]) + att_flops
    print('{:<30}  {:<5} G'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
