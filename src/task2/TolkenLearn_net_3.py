
"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from src.models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM, CBA, UBlock333, UBlock333_sig
from task2.Tolekn_block.tokenLearn_3D import AxialBlock
class Conv_1x1x1(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = norm_layer(outplanes)
        self.act = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.norm(x))

        return x


class Conv_3x3x1(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.norm = norm_layer(outplanes)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_1x3x3(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = norm_layer(outplanes)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class Conv_3x3x3(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)
        self.norm = norm_layer(outplanes)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class HDC_module(nn.Module):
    # inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, activation):
        super(HDC_module, self).__init__()
        self.inplanes = inplanes
        self.midplanes = midplanes
        self.outplanes = outplanes
        self.inter_dim = inplanes // 4
        self.out_inter_dim = midplanes // 4
        self.act = activation
        self.conv_1x3x3_1 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, norm_layer, activation)
        self.conv_5x5x1_1 = nn.Conv3d(self.out_inter_dim, self.out_inter_dim, kernel_size=(5, 5, 1), padding=(2, 2, 0), stride=1, bias=True)
        self.conv_7x7x1_1 = nn.Conv3d(self.out_inter_dim, self.out_inter_dim, kernel_size=(7, 7, 1), padding=(3, 3, 0), stride=1, bias=True)
        self.con1 = nn.Conv3d(self.out_inter_dim * 2, self.out_inter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_1x3x3_2 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, norm_layer, activation)
        self.conv_5x5x1_2 = nn.Conv3d(self.out_inter_dim, self.out_inter_dim, kernel_size=(5, 5, 1), padding=(2, 2, 0), stride=1, bias=True)
        self.conv_7x7x1_2 = nn.Conv3d(self.out_inter_dim, self.out_inter_dim, kernel_size=(7, 7, 1), padding=(3, 3, 0), stride=1, bias=True)
        self.con2 = nn.Conv3d(self.out_inter_dim * 2, self.out_inter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_1x3x3_3 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, norm_layer, activation)
        self.conv_5x5x1_3 = nn.Conv3d(self.out_inter_dim, self.out_inter_dim, kernel_size=(5, 5, 1), padding=(2, 2, 0), stride=1, bias=True)
        self.conv_7x7x1_3 = nn.Conv3d(self.out_inter_dim, self.out_inter_dim, kernel_size=(7, 7, 1), padding=(3, 3, 0), stride=1, bias=True)
        self.con3 = nn.Conv3d(self.out_inter_dim * 2, self.out_inter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_1x1x1_1 = Conv_1x1x1(inplanes, midplanes, norm_layer, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(midplanes, midplanes, norm_layer, activation)
        if self.inplanes > self.midplanes:
            self.conv_1x1x1_3 = Conv_1x1x1(inplanes, midplanes, norm_layer, activation)
        self.conv_3x3x1 = Conv_3x3x1(midplanes, midplanes, norm_layer, activation)
        if self.outplanes != self.midplanes:
            self.conv_1x1x1 = Conv_1x1x1(self.midplanes, self.outplanes, norm_layer, activation)
    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        # x_1 = x
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_1x3x3_1(x2)
        x2_1 = self.conv_5x5x1_1(x2)
        x2_2 = self.conv_7x7x1_1(x2)
        x2 = torch.cat((x2_1, x2_2), dim=1)
        x2 = self.con1(x2)
        # print("x2",x2.shape)
        # print("x3",x3.shape)
        x3 = self.conv_1x3x3_2(x2 + x3)
        x3_1 = self.conv_5x5x1_1(x3)
        x3_2 = self.conv_7x7x1_1(x3)
        x3 = torch.cat((x3_1, x3_2), dim=1)
        x3 = self.con2(x3)
        x4 = self.conv_1x3x3_3(x3 + x4)
        x4_1 = self.conv_5x5x1_1(x4)
        x4_2 = self.conv_7x7x1_1(x4)
        x4 = torch.cat((x4_1, x4_2), dim=1)
        x4 = self.con3(x4)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.inplanes > self.outplanes:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_3x3x1(x_1 + x)
        if self.midplanes != self.outplanes:
            x_1 = self.conv_1x1x1(x_1)
        return x_1
class HDC_module1(nn.Module):
    # inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, activation):
        super(HDC_module1, self).__init__()
        self.inplanes = inplanes
        self.midplanes = midplanes
        self.outplanes = outplanes
        self.inter_dim = inplanes // 4
        self.out_inter_dim = midplanes // 4
        self.act = activation
        self.conv_1x3x3_1 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, norm_layer, activation)
        self.conv_1x3x3_2 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, norm_layer, activation)
        self.conv_1x3x3_3 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, norm_layer, activation)
        self.conv_1x1x1_1 = Conv_1x1x1(inplanes, midplanes, norm_layer, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(midplanes, midplanes, norm_layer, activation)
        if self.inplanes > self.midplanes:
            self.conv_1x1x1_3 = Conv_1x1x1(inplanes, midplanes, norm_layer, activation)
        self.conv_3x3x1 = Conv_3x3x1(midplanes, midplanes, norm_layer, activation)
        if self.outplanes != self.midplanes:
            self.conv_1x1x1 = Conv_1x1x1(self.midplanes, self.outplanes, norm_layer, activation)
    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        # x_1 = x
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_1x3x3_1(x2)
        x3 = self.conv_1x3x3_2(x2 + x3)
        x4 = self.conv_1x3x3_3(x3 + x4)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.inplanes > self.outplanes:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_3x3x1(x_1 + x)
        if self.midplanes != self.outplanes:
            x_1 = self.conv_1x1x1(x_1)
        return x_1

class Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer):
        super(Conv_down, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        self.norm = norm_layer(out_dim)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        down0 = self.downsample0(x)  # 128 -> 64
        print("down0",down0.shape)
        down1 = self.encoder1(down0)
        down_2 = self.downsample1(down1)  # 64->32
        down2 = self.encoder2(down_2)

        down_3 = self.downsample2(down2)  # 32->16
        down3 = self.encoder3(down_3)

        down_4 = self.downsample3(down3)  # 16->8
        down4 = self.encoder4(down_4)


        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder

        up3 = self.upsample(bottom_2)  # 8->16
        # print("down2:{}, up2:{}".format(down3.shape, up3.shape))
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)  # 16->32

        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)  # 32->64
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))
        up0 = self.upsample(up1)  # 64->128
        out = self.outconv(up0)

        if self.deep_supervision:
            output4 = self.deep_bottom(bottom)
            output3 = self.deep_bottom2(bottom_2)
            output2 = self.deep3(up3)
            output1 = self.deep2(up2)
            return out, output4, output3, output2, output1
        return out

devices = torch.device('cuda')
class pseudo_down(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer, activation, first=False):
        super(pseudo_down, self).__init__()
        self.inplane = in_dim * 8
        self.outplane = out_dim
        self.conv = Conv_1x1x1(self.inplane, self.outplane, norm_layer=norm_layer, activation=activation)
    def forward(self, image, num=2):
        # print("image:", image.shape)
        x1 = torch.Tensor([]).to(devices)
        for i in range(num):
            for j in range(num):
                for k in range(num):
                    x3 = image[:, :, k::num, i::num, j::num]
                    x1 = torch.cat((x1, x3), dim=1)
        # print("x1:",x1.shape)
        x1 = self.conv(x1)
        return x1
class EquiUnet(Unet):
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]  # [48, 96, 192, 384]
        print(features)
        self.act = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        # self.downsample0 = Conv_down(inplanes,  features[0], norm_layer)
        self.downsample0 = pseudo_down(inplanes,  features[0], norm_layer, self.act, True)
        # self.encoder1 = UBlockCbam(features[0], features[0], features[0], norm_layer, dropout=dropout)
        self.encoder1 = nn.Sequential(
            AxialBlock(features[0], features[0], norm_layer=norm_layer),
            # HDC_module1(features[0], features[0], features[0], norm_layer, self.act),
            # AxialBlock(features[0], features[0], norm_layer=norm_layer),

        )
        # self.downsample1 = Conv_down(features[0], features[1], norm_layer)
        self.downsample1 = pseudo_down(features[0],  features[1], norm_layer, self.act)
        self.encoder2 = nn.Sequential(
            AxialBlock(features[1], features[1], norm_layer=norm_layer),
            # HDC_module1(features[1], features[1], features[1], norm_layer, self.act),
            AxialBlock(features[1], features[1], norm_layer=norm_layer),
        )
        self.downsample2 = Conv_down(features[1], features[2], norm_layer)
        # self.downsample2 = pseudo_down(features[1],  features[2], norm_layer, self.act)
        self.encoder3 = nn.Sequential(
            AxialBlock(features[2], features[2], norm_layer=norm_layer),
            # HDC_module1(features[2], features[2], features[2], norm_layer, self.act),
            AxialBlock(features[2], features[2], norm_layer=norm_layer),
            AxialBlock(features[2], features[2], norm_layer=norm_layer),
        )
        # self.downsample3 = Conv_down(features[2], features[3], norm_layer)
        self.downsample3 = pseudo_down(features[2], features[3], norm_layer, self.act)
        self.encoder4 = nn.Sequential(
            AxialBlock(features[3], features[3],  norm_layer=norm_layer),
            # HDC_module1(features[3], features[3], features[3], norm_layer, self.act),
            AxialBlock(features[3], features[3], norm_layer=norm_layer),
            # AxialBlock(features[3], features[3], norm_layer=norm_layer),
        )
        self.bottom = nn.Sequential(
            AxialBlock(features[3], features[3], norm_layer=norm_layer),

            # UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)
        )

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

        self._init_weights()

class Att_EquiUnet(Unet):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False,  dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)
        self.act = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision

        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = HDC_module(features[0], features[1], features[1], norm_layer, self.act)
        self.encoder3 = HDC_module(features[1], features[2], features[2], norm_layer, self.act)
        self.encoder4 = HDC_module(features[2], features[3], features[3], norm_layer, self.act)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

        self._init_weights()

if __name__ == "__main__":
    from thop import profile
    from src.models import get_norm_layer
    device = torch.device('cuda')
    image_size = 128
    x = torch.rand((1, 4, 128, 128, 128), device=device)
    print("x size: {}".format(x.size()))
    model = EquiUnet(inplanes=4, num_classes=3, width=32,norm_layer=get_norm_layer('group')).to(device)
    flops, params = profile(model, inputs=(x,))
    print("***********")
    print(flops, params)
    print("***********")