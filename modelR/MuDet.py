import sys

import torch

sys.path.append("..")
import torch.nn as nn
from modelR.backbones.darknet53_siames import Darknet53
from modelR.necks.neck_GGHL import Neck
from modelR.head.head_GGHL import Head
from modelR.layers.convolutions import Convolutional
from utils.utils_basic import *
from modelR.backbones.resnetv2 import ResNet


class MulLeaBlock(nn.Module):
    def __init__(self, channel):
        super(MulLeaBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel*2, out_channels=self.inter_channel*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel*2, out_channels=channel*2, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x0, x, x_dsm):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c//2, -1)
        x_theta = self.conv_theta(x_dsm).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x0).view(b, c, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel*2, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x0
        return out

class MuDet(nn.Module):
    def __init__(self, init_weights=True, inputsize= int(cfg.TRAIN["TRAIN_IMG_SIZE"]), weight_path=None):
        super(MuDet, self).__init__()
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = self.__nC + 4 + 5 + 1
        self.__backnone = Darknet53()
        self.__backnone_dsm = ResNet()
        self.softmax = nn.Softmax(dim=1)
        self.__conv2 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.__conv3 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.__conv4 = Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.mulLea4 = MulLeaBlock(channel=512)
        self.mulLea3 = MulLeaBlock(channel=256)
        self.mulLea2 = MulLeaBlock(channel=128)
        self.__fpn = Neck(fileters_in=[1024, 512, 256, 128], fileters_out=self.__out_channel)

        self.__conv2hd = Convolutional(filters_in=128, filters_out=1, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.__conv3hd = Convolutional(filters_in=256, filters_out=1, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.__conv4hd = Convolutional(filters_in=512, filters_out=1, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')

        self.__conv2hu = Convolutional(filters_in=128, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.__conv3hu = Convolutional(filters_in=256, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.__conv4hu = Convolutional(filters_in=512, filters_out=1024, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')

        self.__head_s = Head(nC=self.__nC, stride=self.__strides[0])
        self.__head_m = Head(nC=self.__nC, stride=self.__strides[1])
        self.__head_l = Head(nC=self.__nC, stride=self.__strides[2])
        if init_weights:
            self.__init_weights()

    def hedis(self, a32, b32, c32, x_32_mix, x_32, x_32_dsm):
        m1 = torch.where(a32 > c32, torch.ones_like(a32), torch.zeros_like(a32))
        m2 = torch.where(b32 > c32, torch.ones_like(a32), torch.zeros_like(a32))
        m_mix = m1 * m2
        m_1 = m1 - m1 * m2
        m_2 = m2 - m1 * m2
        #print(x_32_mix.shape, x_32.shape, x_32_dsm.shape)
        z32_mix = m_mix * (x_32_mix + x_32 + x_32_dsm)
        z32_1 = m_1 * (x_32_mix + x_32) * a32
        z32_2 = m_2 * (x_32_mix + x_32_dsm) * b32
        z32 = z32_mix + z32_1 + z32_2
        return z32

    def forward(self, x_rgb, x_dsm):
        out = []
        x_8, x_16, x_32 = self.__backnone(x_rgb)
        x_8_dsm, x_16_dsm, x_32_dsm = self.__backnone_dsm(x_dsm)
        x_32_mix = self.mulLea4(x_32, self.__conv4(x_32), x_32_dsm)
        x_16_mix = self.mulLea3(x_16, self.__conv3(x_16), x_16_dsm)
        x_8_mix = self.mulLea2(x_8, self.__conv2(x_8), x_8_dsm)

        #He-Dis
        theta = 0.3
        a32 = torch.sigmoid(self.__conv4hd(self.__conv4(x_32)))
        a16 = torch.sigmoid(self.__conv3hd(self.__conv3(x_16)))
        a8 = torch.sigmoid(self.__conv2hd(self.__conv2(x_8)))

        b32 = torch.sigmoid(self.__conv4hd(x_32_dsm))
        b16 = torch.sigmoid(self.__conv3hd(x_16_dsm))
        b8 = torch.sigmoid(self.__conv2hd(x_8_dsm))

        c32 = torch.ones_like(a32) * theta
        c16 = torch.ones_like(a16) * theta
        c8 = torch.ones_like(a8) * theta

        z32 = self.hedis(a32, b32, c32, x_32_mix, x_32, self.__conv4hu(x_32_dsm))
        z16 = self.hedis(a16, b16, c16, x_16_mix, x_16, self.__conv3hu(x_16_dsm))
        z8 = self.hedis(a8, b8, c8, x_8_mix, x_8, self.__conv2hu(x_8_dsm))

        x_s, x_m, x_l = self.__fpn(z32, z16, z8)
        x_l = x_l * (1 + a32 + b32 - a32*b32)
        x_m = x_m * (1 + a16 + b16 - a16*b16)
        x_s = x_s * (1 + a8 + b8 - a8*b8)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def __init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                #print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)
                #print("initing {}".format(m))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                #print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        print("load darknet weights : ", weight_file)
        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1
                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    #print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                #print("loading weight {}".format(conv_layer))