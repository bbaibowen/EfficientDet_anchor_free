import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel = 3,strides = 1,padding=1,
                 bias = True,act = True,bn = True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=kernel,stride=strides,padding=padding,
                              bias=bias)
        self.act = nn.ReLU(True) if act else None
        self.bn = nn.BatchNorm2d(in_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class BiBlock(nn.Module):

    def __init__(self,feature_size):
        super(BiBlock,self).__init__()


        self.maxpooling = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.p7_to_p6 = ConvBlock(feature_size,feature_size)
        self.p6_to_p5 = ConvBlock(feature_size,feature_size)
        self.p5_to_p4 = ConvBlock(feature_size,feature_size)

        self.p3 = ConvBlock(feature_size,feature_size)
        self.p4 = ConvBlock(feature_size,feature_size)
        self.p5 = ConvBlock(feature_size,feature_size)
        self.p6 = ConvBlock(feature_size,feature_size)
        self.p7 = ConvBlock(feature_size,feature_size)



    def forward(self, pyramids):
        p3,p4,p5,p6,p7 = pyramids

        p7_to_p6 = F.upsample(p7,size=p6.shape[-2:])
        # p7_to_p6 = F.upsample(p7,scale_factor=2)
        p7_to_p6 = self.p7_to_p6(p7_to_p6 + p6)

        p6_to_p5 = F.upsample(p7_to_p6,p5.shape[-2:])
        # p6_to_p5 = F.upsample(p7_to_p6,scale_factor=2)
        p6_to_p5 = self.p6_to_p5(p6_to_p5 + p5)

        p5_to_p4 = F.upsample(p6_to_p5,size=p4.shape[-2:])
        # p5_to_p4 = F.upsample(p6_to_p5,scale_factor=2)
        p5_to_p4 = self.p5_to_p4(p5_to_p4 + p4)

        p4_to_p3 = F.upsample(p5_to_p4,size=p3.shape[-2:])
        # p4_to_p3 = F.upsample(p5_to_p4,scale_factor=2)
        p3 = self.p3(p4_to_p3 + p3)

        p3_to_p4 = self.maxpooling(p3)
        p4 = self.p4(p3_to_p4 + p5_to_p4 + p4)

        p4_to_p5 = self.maxpooling(p4)
        p5 = self.p5(p4_to_p5 + p6_to_p5 + p5)

        p5_to_p6 = self.maxpooling(p5)
        p5_to_p6 = F.upsample(p5_to_p6,size=p6.shape[-2:])
        p6 = self.p6(p5_to_p6 + p7_to_p6 + p6)

        p6_to_p7 = self.maxpooling(p6)
        p6_to_p7 = F.upsample(p6_to_p7,size=p7.shape[-2:])
        p7 = self.p7(p6_to_p7 + p7)

        return p3,p4,p5,p6,p7


class BiFpn(nn.Module):

    def __init__(self,in_channels,out_channels,len_input,bi = 3):
        super(BiFpn,self).__init__()
        assert len_input <= 5
        self.len_input = len_input
        self.bi = bi
        self.default = 5 - len_input
        for i in range(len_input):
            setattr(self, 'p{}'.format(str(i)), ConvBlock(in_channels=in_channels[i], out_channels=out_channels,
                                                          kernel=1, strides=1, padding=0, act=False, bn=False))
        if self.default > 0:
            for i in range(self.default):
                setattr(self,'make_pyramid{}'.format(str(i)),ConvBlock(in_channels=in_channels[-1] if i == 0 else out_channels,out_channels=out_channels,kernel=3,strides=2,
                                                                       padding=1,act=False,bn=False))
        for i in range(bi):
            setattr(self, 'biblock{}'.format(str(i)), BiBlock(out_channels))

    def forward(self, inputs):
        pyramids = []
        for i in range(self.len_input):
            pyramids.append(getattr(self,'p{}'.format(str(i)))(inputs[i]))

        if self.default > 0:
            x = inputs[-1]
            for i in range(self.default):
                x = getattr(self,'make_pyramid{}'.format(str(i)))(x)
                pyramids.append(x)

        for i in range(self.bi):
            pyramids = getattr(self,'biblock{}'.format(str(i)))(pyramids)

        return pyramids


if __name__ == '__main__':
    p3 = torch.randn(2,512,64,64)
    p4 = torch.randn(2,1024,32,32)
    p5 = torch.randn(2,2048,16,16)

    b = BiFpn([512,1024,2048],256,3,3)
    py = b([p3,p4,p5])
    for i in py:
        print(i.shape)

