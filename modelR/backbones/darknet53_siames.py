import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.convolutions import Convolutional
from ..layers.conv_blocks import Residual_block

class Darknet53(nn.Module):

    def __init__(self, pre_weight=None):
        super(Darknet53, self).__init__()
        fi = [32, 64, 128, 256, 512, 1024]
        self.__conv = Convolutional(filters_in=3, filters_out=fi[0], kernel_size=3, stride=1, pad=1, norm='bn',
                                    activate='leaky')

        self.__conv_5_0 = Convolutional(filters_in=fi[0], filters_out=fi[1], kernel_size=3, stride=2, pad=1, norm='bn',activate='leaky')
        self.__rb_5_0 = Residual_block(filters_in=fi[1], filters_out=fi[1], filters_medium=fi[0])

        self.__conv_5_1 = Convolutional(filters_in=fi[1], filters_out=fi[2], kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_1_0 = Residual_block(filters_in=fi[2], filters_out=fi[2], filters_medium=fi[1])
        self.__rb_5_1_1 = Residual_block(filters_in=fi[2], filters_out=fi[2], filters_medium=fi[1])

        self.__conv_5_2 = Convolutional(filters_in=fi[2], filters_out=fi[3], kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_2_0 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])
        self.__rb_5_2_1 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])
        self.__rb_5_2_2 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])
        self.__rb_5_2_3 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])
        self.__rb_5_2_4 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])
        self.__rb_5_2_5 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])
        self.__rb_5_2_6 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])
        self.__rb_5_2_7 = Residual_block(filters_in=fi[3], filters_out=fi[3], filters_medium=fi[2])

        self.__conv_5_3 = Convolutional(filters_in=fi[3], filters_out=fi[4], kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_3_0 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])
        self.__rb_5_3_1 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])
        self.__rb_5_3_2 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])
        self.__rb_5_3_3 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])
        self.__rb_5_3_4 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])
        self.__rb_5_3_5 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])
        self.__rb_5_3_6 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])
        self.__rb_5_3_7 = Residual_block(filters_in=fi[4], filters_out=fi[4], filters_medium=fi[3])


        self.__conv_5_4 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_4_0 = Residual_block(filters_in=fi[5], filters_out=fi[5], filters_medium=fi[4])
        self.__rb_5_4_1 = Residual_block(filters_in=fi[5], filters_out=fi[5], filters_medium=fi[4])
        self.__rb_5_4_2 = Residual_block(filters_in=fi[5], filters_out=fi[5], filters_medium=fi[4])
        self.__rb_5_4_3 = Residual_block(filters_in=fi[5], filters_out=fi[5], filters_medium=fi[4])


    def forward(self, x):
        x = self.__conv(x)
        x0_0 = self.__conv_5_0(x)

        #x0_0 = self.__focus(x)
        x0_1 = self.__rb_5_0(x0_0)

        x1_0 = self.__conv_5_1(x0_1)
        x1_1 = self.__rb_5_1_0(x1_0)
        x1_2 = self.__rb_5_1_1(x1_1)

        x2_0 = self.__conv_5_2(x1_2)
        x2_1 = self.__rb_5_2_0(x2_0)
        x2_2 = self.__rb_5_2_1(x2_1)
        x2_3 = self.__rb_5_2_2(x2_2)
        x2_4 = self.__rb_5_2_3(x2_3)
        x2_5 = self.__rb_5_2_4(x2_4)
        x2_6 = self.__rb_5_2_5(x2_5)
        x2_7 = self.__rb_5_2_6(x2_6)
        x2_8 = self.__rb_5_2_7(x2_7)

        x3_0 = self.__conv_5_3(x2_8)
        x3_1 = self.__rb_5_3_0(x3_0)
        x3_2 = self.__rb_5_3_1(x3_1)
        x3_3 = self.__rb_5_3_2(x3_2)
        x3_4 = self.__rb_5_3_3(x3_3)
        x3_5 = self.__rb_5_3_4(x3_4)
        x3_6 = self.__rb_5_3_5(x3_5)
        x3_7 = self.__rb_5_3_6(x3_6)
        x3_8 = self.__rb_5_3_7(x3_7)

        x4_0 = self.__conv_5_4(x3_8)
        x4_1 = self.__rb_5_4_0(x4_0)
        x4_2 = self.__rb_5_4_1(x4_1)
        x4_3 = self.__rb_5_4_2(x4_2)
        x4_4 = self.__rb_5_4_3(x4_3)

        return x2_8, x3_8, x4_4
