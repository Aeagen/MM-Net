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
    def __init__(self, in_planes, out_planes, groups=8, dropout_rate=0.0):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.drop = dropout_rate
        self.norm = nn.LayerNorm(in_planes, eps=1e-6)
        self.dropout = nn.Dropout(p=self.drop)
        self.Encoder1DBlock = Encoder1DBlock(self.groups, self.in_planes)
        self.TokenLearnerModule = TokenLearnerModule(d_model=in_planes, num_token=self.groups)
        self.TokenFuser = TokenFuser(n_token=groups, n_head=self.groups, d_model=self.in_planes)
    def transformer(self, x):
        N, C, H, W, D = x.shape
        x = self.norms(x)

        h = round(math.pow(H*W*D, 1/3))
        qkv = x.reshape(N, C, h, h, h)
        qkv = qkv.permute(0, 2, 3, 4, 1)  # [b, h, w, d, c]
        residual = qkv  # [b, h, w, d, c]
        qkv = self.TokenLearnerModule(qkv)  # [b, h, w, d, c] -> [N, n_token, C]
        # Transformer
        x = self.Encoder1DBlock(qkv)  # [bs, n_token, c]
        # TokenFuser module
        x = self.TokenFuser(x, residual)  # [b, h, w, d, c]
        x = x + residual
        x = x.view(N, H * W * D, C)
        x = self.norm(x).permute(0, 2, 1)
        output = x.view(N, C, H, W, D)
        return output
    def norms(self, x):
        N, C, H, W, D = x.shape
        x = x.view(N, C, H*W*D)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(N, C, H, W, D)
        return x
    def forward(self, x):
        output = self.transformer(x)
        return output

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
class TokenLearnerModule(nn.Module):
    def __init__(self, d_model, num_token=8):
        super(TokenLearnerModule, self).__init__()
        self.num_token = num_token
        self.use_sum_pooling = False
        self.conv1 = nn.Conv3d(d_model, num_token, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv3d(num_token, num_token, kernel_size=3, stride=1, padding=1, bias=False)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm3d(num_token)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        """Applies learnable tokenization to the 2D inputs.
            Args:
              inputs: Inputs of shape `[bs, h, w, d, c]`.
            Returns:
              Output of shape `[bs, n_token, c]`.
        """
        reidual = inputs
        inputs = inputs.permute(0, 4, 1, 2, 3)

        bs, c, h, w, d = inputs.shape

        selected = inputs
        selected = selected.reshape(bs, c, h * w * d).permute(0, 2, 1)
        selected = self.norm(selected)
        selected = selected.permute(0, 2, 1).reshape(bs, c, h, w, d)
        # spatial attention
        selected = self.gelu(self.bn(self.conv1(selected)))  # Shape: [bs, n_token, h, w, d].

        for _ in range(2):
            selected = self.gelu(self.conv2(selected))  # Shape: [bs, n_token, h, w, d].
        #-----
        selected = selected.reshape([bs, -1, h * w * d])  # Shape: [bs, h*w, n_token].
        selected = nn.Sigmoid()(selected)[:, None, ...].permute(0, 2, 3, 1)  # Shape: [bs, n_token, h*w*d, 1].
        feat = reidual
        feat = feat.reshape([bs, h * w * d, -1])[:, None, ...]  # Shape: [bs, 1, h*w*d, c].

        if self.use_sum_pooling:
            inputs = torch.sum(feat * selected, dim=2)
        else:
            inputs = torch.mean(feat * selected, dim=2)
        return inputs

class Encoder1DBlock(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(Encoder1DBlock, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.mlp = PositionwiseFeedForward(self.d_model, self.d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs):
        """Applies Encoder1DBlock module.
            Args:
              inputs: Input data. [bs, n_token, c]
              deterministic: Deterministic or not (to apply dropout).
            Returns:
              Output after transformer encoder block. [bs, n_token, c]
            """
        # Attention block.
        residual = inputs
        n, n_token, c = inputs.shape
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        assert inputs.ndim == 3
        x_1 = self.layer_norm(inputs)
        q = self.w_qs(x_1).reshape(n, n_token, n_head, d_k)
        k = self.w_ks(x_1).reshape(n, n_token, n_head, d_k)
        v = self.w_vs(x_1).reshape(n, n_token, n_head, d_k)
        # q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(n, n_token, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        # MLP block.
        y = self.mlp(q)

        return y

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        s = q / self.temperature
        attn = torch.matmul(s, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):

        residual = x

        x = self.w_2(self.act(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
class TokenFuser(nn.Module):
    """Token fusion module.
    Attributes:
    use_normalization: Whether to use LayerNorm layers. This is needed when
      using sum pooling in the TokenLearner module.
    dropout_rate: Dropout rate.
    """
    def __init__(self, n_token, n_head, d_model):
        super(TokenFuser, self).__init__()
        self.use_normalization = True
        self.dropout_rate = 0.
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.n_token = n_token
        self.norm = nn.LayerNorm(self.d_model)
        self.linear = nn.Linear(self.n_token, self.n_token, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.d_model, self.n_token, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.n_token),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.n_token, self.n_token, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(self.n_token),
            nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.d_model * 2, self.d_model, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.d_model),
            nn.ReLU(inplace=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(self.dropout_rate)
        self.channel_attention = ChannelGate(self.d_model, self.n_token)
    def forward(self, inputs, original):
        """Applies token fusion to the generate 2D ouputs.
        Args:
          inputs: Inputs of shape `[bs, n_token, c]`.
          original: Inputs of shape `[bs, h, w, c]`.
          deterministic: Weather we are in the deterministic mode (e.g inference
            time) or not.
        Returns:
          Output of shape `[bs, h, w, c]`.
        """
        bs, h, w, d, c = original.shape
        channel_original = original.permute(0, 4, 1, 2, 3)
        channel_inputs = inputs
        if self.use_normalization:
          inputs =self.norm(inputs)

        inputs = inputs.permute(0, 2, 1)  # [bs , c , n_token]
        inputs = self.linear(inputs)
        inputs = inputs.permute(0, 2, 1)    # [bs , n_token, c]

        if self.use_normalization:
          inputs = self.norm(inputs)
        # spatial attention and TokenLearn
        original = original.reshape(bs, h * w * d, c)
        original = self.norm(original)
        original = original.permute(0, 2, 1).reshape(bs, c, h, w, d)
        mix = self.conv1(original)
        mix = self.conv2(mix)
        mix = mix[:, None, ...]   # [bs, 1, n_token, h, w, d]
        mix = self.sigmoid(mix)
        mix = mix.permute(0, 3, 4, 5, 2, 1)        # [bs, n_token, h, w.,d] -> [bs, h, w, d, n_token, 1]
        inputs = inputs[:, None, None, None, ...]  # [bs, n_token, c]       -> [bs, 1, 1, 1, n_token, c]
        inputs = inputs * mix    # [bs, h, w, d, n_token, c]
        inputs = torch.sum(inputs, dim=-2)  # [bs, h, w, d, c]

        # channel attention and TolkenLearn
        channel = self.channel_attention(channel_original, channel_inputs)
        s_c = torch.cat([inputs, channel], dim=-1).permute(0, 4, 1, 2, 3)
        inputs = self.conv3(s_c)

        inputs = self.drop(inputs)
        inputs = inputs.permute(0, 2, 3, 4, 1)

        return inputs
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, reduction_ratio),
            nn.GELU(),
            nn.Linear(reduction_ratio, reduction_ratio)
        )
        self.pool_types = pool_types
        self.reduce_conv = nn.Conv3d(self.gate_channels, reduction_ratio, kernel_size=1, stride=1, bias=False)
    def forward(self, x, tolekn):
        # print("x:",x.shape)
        re_channel = self.reduce_conv(x)
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
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(re_channel)
        scale = scale[:, None, ...]   # [bs, 1, n_tolken, h, w, d]
        tolekn = tolekn.permute(0, 2, 1)[:, :, :, None, None, None]  # [bs, c, n_tolken, 1, 1, 1]
        mix = tolekn * scale   # [bs, c, n_tolken, h, w, d]
        mix = torch.sum(mix, dim=2).permute(0, 2, 3, 4, 1)  # [bs, c, h, w, d] -> [bs, h, w, d, c]
        return mix
class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, n_head=8, norm_layer=None):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.inplanes = inplanes
        self.planes = planes
        self.conv = conv1x1(inplanes, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self. EncoderModFuser = AxialAttention(self.inplanes, self.planes, groups=n_head)
        self.relu = nn.ReLU(inplace=False)
        self.SpatialGate = SpatialGate(norm_layer)
    def forward(self, x):
        identity = x
        print("x:",x.shape)
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.EncoderModFuser(out)
        out += identity
        out = self.SpatialGate(out)

        return out

if __name__ == "__main__":
    from thop import profile
    device = torch.device('cuda')
    image_size = 128
    x = torch.rand((1, 32, 64, 64, 64), device=device)
    print("x size: {}".format(x.size()))
    model = AxialBlock(inplanes=32, planes=32).to(device)
    flops, params = profile(model, inputs=(x,))
    print("***********")
    print(flops, params)
    print("***********")
    out = model(x)
    print("out size: {}".format(out.size()))
