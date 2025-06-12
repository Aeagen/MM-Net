from collections import OrderedDict

import torch
from torch import nn, nn as nn
from torch.nn import functional as F

def default_norm_layer(planes, groups=16):
    groups_ = min(groups, planes)
    if planes % groups_ > 0:
        divisor = 16
        while planes % divisor > 0:
            divisor /= 2
        groups_ = int(planes // divisor)
    return nn.GroupNorm(groups_, planes)


def get_norm_layer(norm_type="group"):
    if "group" in norm_type:
        try:
            grp_nb = int(norm_type.replace("group", ""))
            return lambda planes: default_norm_layer(planes, groups=grp_nb)
        except ValueError as e:
            print(e)
            print('using default group number')
            return default_norm_layer
    elif norm_type == "none":
        return None
    else:
        return lambda x: nn.InstanceNorm3d(x, affine=True)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBnRelu(nn.Sequential):

    def __init__(self, inplanes, planes, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )
        else:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )

class UBlock222(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock222, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout))
                ])
        )

class UBlock333_Spatial(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock333_Spatial, self).__init__()
        self.conv1 = ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)
        self.spatial = SpatialGate(norm_layer)
    def forward(self, x):
        # x1 = x
        x1 = self.conv1(x)
        spatial = self.spatial(x1)
        return spatial

class UBlock333_sig(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock333_sig, self).__init__()
        self.conv1 = ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)
        self.score = nn.Sequential(
            # ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout),
            nn.Conv3d(midplanes, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x1 = x
        x1 = self.conv1(x)
        score = self.score(x1)
        att_x = torch.mul(x1, score)
        return att_x

class UBlock333_inner(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock333_inner, self).__init__()
        # self.conv1 = ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)
        self.score = nn.Sequential(
            ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout),
            nn.Conv3d(midplanes, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = x
        # x1 = self.conv1(x)
        score = self.score(x1)
        att_x = torch.mul(x, score)
        return att_x


class UBlock333(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock333, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                ])
        )


class UBlock(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                    ('ConvBnRelu2', ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ])
        )
class UBlock1(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1, 1, 1), dropout=0):
        super(UBlock1, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                    ('ConvBnRelu2', ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                    ('ConvBnRelu3', ConvBnRelu(outplanes, outplanes, norm_layer, dilation=dilation[2], dropout=dropout)),
                    ('ConvBnRelu4', ConvBnRelu(outplanes, outplanes, norm_layer, dilation=dilation[3], dropout=dropout)),
                ])
        )

class UBlock_single(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock_single, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, outplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                ])
        )
class UBlockNoCbam(nn.Sequential):
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlockNoCbam, self).__init__(
            OrderedDict(
                [
                    ('UBlock', UBlock(inplanes, midplanes, outplanes, norm_layer, dilation=dilation, dropout=dropout)),
                ])
        )


class UBlockCbam(nn.Sequential):
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlockCbam, self).__init__(
            OrderedDict(
                [
                    ('UBlock', UBlock(inplanes, midplanes, outplanes, norm_layer, dilation=dilation, dropout=dropout)),
                    ('CBAM', CBAM(outplanes, norm_layer=norm_layer)),
                ])
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(UBlock(4, 4))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=None):
        super(BasicConv, self).__init__()
        bias = False
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, norm_layer=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, norm_layer=norm_layer)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out + x

class CBA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBA, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        # x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x)
        return x_out + x

class SE_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = F.sigmoid(y)
        return x*y.expand_as(x)


class Spatial_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Spatial_Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 3),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, 1, 3)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = F.sigmoid(y)
        return x * y

class BamSpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, dilation_ratio=2, norm_layer=None):
        super(BamSpatialAttention, self).__init__()

        self.body = nn.Sequential(
            nn.Conv3d(channel, channel//reduction, 1),

            norm_layer(channel//reduction),
            nn.Conv3d(channel//reduction, channel//reduction, 3, padding=dilation_ratio, dilation=dilation_ratio),
            norm_layer(channel // reduction),
            nn.ReLU(False),

            norm_layer(channel // reduction),
            nn.Conv3d(channel // reduction, channel // reduction, 3, padding=dilation_ratio, dilation=dilation_ratio),
            norm_layer(channel // reduction),
            nn.ReLU(False),

            nn.Conv3d(channel//reduction, 1, 1)
        )
    def forward(self, x):
        x1 = x
        x_out = self.body(x)
        scale = torch.sigmoid(x_out)
        return x1 * scale