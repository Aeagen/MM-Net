import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class Transformer(nn.Module):
    def __init__(self,n_f,num_filters,dropout=0.1):
        super(Transformer, self).__init__()
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)
        # self.ccnet = CrissCrossAttention(self.n_f,dropout)
        self.Conv_3x3x3 = Conv_3x3x3(self.n_f * 3, self.n_f, self.activation)
        self.softmax = F.Softmax(dim=-1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self,q,k,v):
        bs_1, c_1, h_1, w_1, d_1 = q.size()
        # print("x:",x.shape) # [2, 32, 8, 8, 8]
        # W H
        for i in range(q.shape[4]):
            s = i
            x_q = q[:, :, :, :, s:s + 1].contiguous().view(bs_1*c_1, h_1, w_1 * 1).permute(0,2,1)
            x_k = k[:, :, :, :, s:s + 1].contiguous().view(bs_1*c_1, h_1, w_1 * 1)
            x_v = v[:, :, :, :, s:s + 1].contiguous().view(bs_1*c_1, h_1, w_1 * 1)

            A = self.dropout(self.softmax(torch.matmul(x_q, x_k)))
            if i == 0:
                AVa = torch.matmul(A,x_v).view(bs_1, c_1, h_1, w_1,1)
            else:
                AV_1 = torch.matmul(A,x_v).view(bs_1, c_1, h_1, w_1,1)
                AVa = torch.cat((AVa, AV_1), dim=4)
        # print("Ava",AVa.shape)
        # H D
        for i in range(q.shape[3]):
            s = i
            x_q = q[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1)
            x_k = k[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1).permute(0, 2, 1)
            x_v = v[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1)

            A = self.dropout(self.softmax(torch.matmul(x_q, x_k)))
            if i == 0:
                AVb = torch.matmul(A, x_v).view(bs_1, c_1, h_1, 1, d_1)
            else:
                AV_1 = torch.matmul(A, x_v).view(bs_1, c_1, h_1, 1, d_1)
                AVb = torch.cat((AVb, AV_1), dim=3)
        # print("Avb", AVb.shape)
        # W D
        for i in range(q.shape[2]):
            s = i
            x_q = q[:, :, s:s+1, :, :].contiguous().view(bs_1 * c_1, 1*w_1 , d_1)
            x_k = k[:, :, s:s+1, :, :].contiguous().view(bs_1 * c_1, 1*w_1 , d_1).permute(0, 2, 1)
            x_v = v[:, :, s:s+1, :, :].contiguous().view(bs_1 * c_1, 1*w_1 , d_1)

            A = self.dropout(self.softmax(torch.matmul(x_q, x_k)))
            if i == 0:
                AVc = torch.matmul(A, x_v).view(bs_1, c_1, 1, w_1, d_1)
            else:
                AV_1 = torch.matmul(A, x_v).view(bs_1, c_1, 1, w_1, d_1)
                AVc = torch.cat((AVc, AV_1), dim=2)
        # print("Avc", AVc.shape)
        x = torch.cat((AVa, AVb), dim=1)
        x = torch.cat((x, AVc), dim=1)
        x = self.Conv_3x3x3(x)
        return x



class Transformer_encoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(Transformer_encoder, self).__init__()
        self.n_f = in_dim
        self.dk = math.sqrt(in_dim)
        self.activation = nn.ReLU(inplace=False)
        # self.ccnet = CrissCrossAttention(self.n_f,dropout)
        self.conv3d_Wq = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3d_Wk = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3d_Wv = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.Conv_3x3x3 = Conv_3x3x3(self.n_f * 3, out_dim, self.activation)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm3d(out_dim)
    def forward(self, q):
        k = q
        v = q
        q = self.conv3d_Wq(q)
        k = self.conv3d_Wk(k)
        v = self.conv3d_Wv(v)
        bs_1, c_1, h_1, w_1, d_1 = q.size()
        # print("x:",x.shape) # [2, 32, 8, 8, 8]
        # W H
        for i in range(q.shape[4]):
            s = i
            # H W * W H ->H H * H W ->HW
            #  Q  *  K  -> A * V    ->Z
            x_q = q[:, :, :, :, s:s + 1].contiguous().view(bs_1*c_1, h_1, w_1 * 1)
            x_k = k[:, :, :, :, s:s + 1].contiguous().view(bs_1*c_1, h_1, w_1 * 1).permute(0,2,1)
            x_v = v[:, :, :, :, s:s + 1].contiguous().view(bs_1*c_1, h_1, w_1 * 1)

            A = self.dropout(self.softmax(torch.matmul(x_q/self.dk, x_k)))
            if i == 0:
                AVa = torch.matmul(A,x_v).view(bs_1, c_1, h_1, w_1,1)
            else:
                AV_1 = torch.matmul(A,x_v).view(bs_1, c_1, h_1, w_1,1)
                AVa = torch.cat((AVa, AV_1), dim=4)
            # W H * H W ->W W * W H ->W H -> H W
            #  Q  *  K  -> A * V    ->Z
            x_q_1 = q[:, :, :, :, s:s + 1].contiguous().view(bs_1 * c_1, h_1, w_1 * 1).permute(0, 2, 1)
            x_k_1 = k[:, :, :, :, s:s + 1].contiguous().view(bs_1 * c_1, h_1, w_1 * 1)
            x_v_1 = v[:, :, :, :, s:s + 1].contiguous().view(bs_1 * c_1, h_1, w_1 * 1).permute(0, 2, 1)
            A_1 = self.dropout(self.softmax(torch.matmul(x_q_1 / self.dk, x_k_1)))
            if i == 0:
                AVa_1 = torch.matmul(A_1, x_v_1).view(bs_1, c_1, w_1, h_1, 1)
            else:
                AV_1_1 = torch.matmul(A_1, x_v_1).view(bs_1, c_1, w_1, h_1, 1)
                AVa_1 = torch.cat((AVa_1, AV_1_1), dim=4)
        AVa_1 = AVa_1.permute(0, 1, 3, 2, 4)
        AVa = (AVa + AVa_1) / 2
        # print("Ava",AVa.shape)
        # H D
        for i in range(q.shape[3]):
            s = i
            # H D * D H ->H H * H D ->H D
            #  Q  *  K  -> A * V    ->Z
            x_q = q[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1)
            x_k = k[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1).permute(0, 2, 1)
            x_v = v[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1)

            A = self.dropout(self.softmax(torch.matmul(x_q/self.dk, x_k)))
            if i == 0:
                AVb = torch.matmul(A, x_v).view(bs_1, c_1, h_1, 1, d_1)
            else:
                AV_1 = torch.matmul(A, x_v).view(bs_1, c_1, h_1, 1, d_1)
                AVb = torch.cat((AVb, AV_1), dim=3)

            # D H * H D ->D D * D H ->D H -> H D
            #  Q  *  K  -> A * V    ->Z
            x_q_1 = q[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1).permute(0, 2, 1)
            x_k_1 = k[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1)
            x_v_1 = v[:, :, :, s:s+1, :].contiguous().view(bs_1 * c_1, h_1*1, d_1).permute(0, 2, 1)
            A_1 = self.dropout(self.softmax(torch.matmul(x_q_1 / self.dk, x_k_1)))
            if i == 0:
                AVb_1 = torch.matmul(A_1, x_v_1).view(bs_1, c_1, d_1, 1, h_1)
            else:
                AV_1_1 = torch.matmul(A_1, x_v_1).view(bs_1, c_1, d_1, 1, h_1)
                AVb_1 = torch.cat((AVb_1, AV_1_1), dim=3)
        AVb_1 = AVb_1.permute(0, 1, 4, 3, 2)
        AVb = (AVb + AVb_1) / 2
        # print("Avb", AVb.shape)
        # W D
        for i in range(q.shape[2]):
            s = i
            # W D * D W -> W W * W D -> W D
            #  Q  *  K  ->  A  *  V  ->  Z
            x_q = q[:, :, s:s+1, :, :].contiguous().view(bs_1 * c_1, 1*w_1 , d_1)
            x_k = k[:, :, s:s+1, :, :].contiguous().view(bs_1 * c_1, 1*w_1 , d_1).permute(0, 2, 1)
            x_v = v[:, :, s:s+1, :, :].contiguous().view(bs_1 * c_1, 1*w_1 , d_1)

            A = self.dropout(self.softmax(torch.matmul(x_q/self.dk, x_k)))
            if i == 0:
                AVc = torch.matmul(A, x_v).view(bs_1, c_1, 1, w_1, d_1)
            else:
                AV_1 = torch.matmul(A, x_v).view(bs_1, c_1, 1, w_1, d_1)
                AVc = torch.cat((AVc, AV_1), dim=2)
            # D W * W D -> D D * D W -> D W -> W D
            #  Q  *  K  ->  A  *  V  ->Z
            x_q_1 = q[:, :, s:s + 1, :, :].contiguous().view(bs_1 * c_1, 1 * w_1, d_1).permute(0, 2, 1)
            x_k_1 = k[:, :, s:s + 1, :, :].contiguous().view(bs_1 * c_1, 1 * w_1, d_1)
            x_v_1 = v[:, :, s:s + 1, :, :].contiguous().view(bs_1 * c_1, 1 * w_1, d_1).permute(0, 2, 1)

            A_1 = self.dropout(self.softmax(torch.matmul(x_q_1 / self.dk, x_k_1)))
            if i == 0:
                AVc_1 = torch.matmul(A_1, x_v_1).view(bs_1, c_1, 1, d_1, w_1)
            else:
                AV_1_1 = torch.matmul(A_1, x_v_1).view(bs_1, c_1, 1, d_1, w_1)
                AVc_1 = torch.cat((AVc_1, AV_1_1), dim=2)
        AVc_1 = AVc_1.permute(0, 1, 2, 4, 3)
        AVc = (AVc + AVc_1) / 2
        # print("Avc", AVc.shape)
        x = torch.cat((AVa, AVb), dim=1)
        x = torch.cat((x, AVc), dim=1)
        x = self.norm(self.Conv_3x3x3(x))
        return x + q