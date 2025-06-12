import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import SpatialGate


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""
kernal = {'8':8, '11':11, '48':64}
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False, depth=False,dropout_rate=0.0, mlp=True, fc=True):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.depth = depth
        self.drop = dropout_rate
        self.mlp = mlp
        self.fc = fc
        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.norm = nn.LayerNorm(in_planes, eps=1e-6)
        self.dropout = nn.Dropout(p=self.drop)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)
        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

        self.mlp = nn.Sequential(
            nn.Linear(in_planes, out_planes * 2, bias=False),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(p=self.drop),
            nn.Linear(out_planes * 2, in_planes, bias=False),
            nn.Dropout(p=self.drop)
        )
        self.fc = nn.Linear(self.out_planes * 2, self.out_planes, bias=False)
    def transformer(self, x, N, W, D, C, H):

        x = x.contiguous().view(N * W * D, C, H)
        in_x = x
        x = self.norms(x)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W * D, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)

        print("q.shape",q.shape, q_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W * D, 3, self.groups, H, H).sum(dim=1)

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W * D, self.out_planes * 2, H)
        if self.fc:
            stacked_output = stacked_output.view(N * W * D, self.out_planes * 2, H)
            stacked_output = stacked_output.permute(0, 2, 1)
            stacked_output = self.dropout(self.fc(stacked_output))
        else:
            stacked_output = self.bn_output(stacked_output).view(N, W, D, self.out_planes, 2, H).sum(dim=-2)
        stacked_output = stacked_output.view(N * W * D, self.out_planes, H)
        stacked_output = in_x + stacked_output
        # output = stacked_output.view(N, W, D, self.out_planes, H)


        # mlp
        if self.mlp:
            in_x = stacked_output
            stacked_output = stacked_output.permute(0, 2, 1).contiguous()
            stacked_output = self.dropout(self.norm(stacked_output))
            stacked_output = self.mlp(stacked_output)
            output = stacked_output.permute(0, 2, 1).contiguous()
            stacked_output = output + in_x
        output = stacked_output.view(N, W, D, self.out_planes, H)
        if self.width:
            output = output.permute(0, 3, 1, 2, 4)
        elif self.depth:
            output = output.permute(0, 3, 4, 1, 2)
        else:
            output = output.permute(0, 3, 1, 4, 2)
        if self.stride > 1:
            output = self.pooling(output)
        # print("output1", output.shape)
        return output
    def norms(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1).contiguous()
        return x
    def forward(self, x):
        # x.shape = N, C, D, H, W
        # N, C, D, H, W = x.shape
        # D = str(D)
        # kernal_size = kernal[D]
        if self.width:
            # print("x", x.shape)
            x = x.permute(0, 2, 3, 1, 4)
            N, D, H, C, W = x.shape
            x = x.contiguous().view(N * D * H, C, W)

            output = self.transformer(x, N, D, H, C, W)
            return output
        elif self.depth:
            x = x.permute(0, 3, 4, 1, 2)
            N, H, W, C, D = x.shape
            x = x.contiguous().view(N * H * W, C, D)
            output = self.transformer(x, N, H, W, C, D)
            return output
        else:
            x = x.permute(0, 2, 4, 1, 3)
            N, D, W, C, H = x.shape
            # print("x1",x.shape)
            x = x.contiguous().view(N * D * W, C, H)
            output = self.transformer(x, N, D, W, C, H)
            return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.dropout(self.norm(x))

class Flatten(nn.Module):
    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return x
class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1,  groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.))
        self.inplanes = inplanes
        self.planes = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.depth_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, depth=True)
        self.conv_up1 = conv1x1(width,width)
        self.conv_up2 = conv1x1(width,inplanes)
        self.bn2 = norm_layer(width)
        self.relu = nn.ReLU(inplace=False)
        self.stride = stride
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(inplanes, inplanes // 16),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes // 16, inplanes),
        )
        self.pool_types = ['avg']
        self.SpatialGate = SpatialGate(norm_layer)
    def channel(self, x):
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

    def forward1(self, x):
        # position_6
        identity = x
        # print(x.shape)
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.depth_block(out)

        out = self.relu(out)
        out = self.conv_up1(out)
        out = self.bn2(out)
        if self.inplanes != self.planes:
            out = self.conv_up2(out)
        out += identity

        out = self.relu(out)

        # out = self.SpatialGate(out)

        return out
    def forward(self, x):
        # position_6
        identity = x
        # print(x.shape)
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.hight_block(out)
        out2 = self.width_block(out)
        out3 = self.depth_block(out)

        out = out1 + out2 + out3
        out = self.relu(out)
        out = self.conv_up1(out)
        out = self.bn2(out)
        if self.inplanes != self.planes:
            out = self.conv_up2(out)
        out += identity

        out = self.relu(out)

        # out = self.SpatialGate(out)

        return out

if __name__ == "__main__":
    # from thop import profile
    device = torch.device('cuda')
    image_size = 128
    x = torch.rand((1, 64, 32, 32, 32), device=device)
    print("x size: {}".format(x.size()))
    model = AxialBlock(inplanes=64, planes=64,kernel_size=32).to(device)
    # flops, params = profile(model, inputs=(x,))
    # print("***********")
    # print(flops, params)
    # print("***********")
    out = model(x)
    print("out size: {}".format(out.size()))
