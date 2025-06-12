import torch.nn.functional as F
import torch
import logging
import torch.nn as nn


__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss']

cross_entropy = F.cross_entropy


def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3 # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1) # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()

def dice(output, target,eps =1e-5): # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sigmoid_dice_loss(output, target,alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:,0,...],(target==1).float(),eps=alpha)
    loss2 = dice(output[:,1,...],(target==2).float(),eps=alpha)
    loss3 = dice(output[:,2,...],(target == 4).float(),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    return loss1+loss2+loss3


def softmax_dice_loss(output, target,eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],(target==1).float())
    loss2 = dice(output[:,2,...],(target==2).float())
    loss3 = dice(output[:,3,...],(target==4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    return loss1+loss2+loss3


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()
    if target.dim() == 4:
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[...] # [class, N*H*W*D]

    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels

    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
        # print("class_weights",class_weights)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)
    # print('TC 1:{:.4f} | WT 2:{:.4f} | ET 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())

    shape.insert(1, n_class)

    shape = tuple(shape)

    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

# ---------------------------------------------------------------------------------------------------------
from torch.nn.modules.loss import _Loss
class FocalLoss(_Loss):
    '''
    Focal_Loss = - [alpha * (1 - p)^gamma *log(p)]  if y = 1;
               = - [(1-alpha) * p^gamma *log(1-p)]  if y = 0;
        average over batchsize; alpha helps offset class imbalance; gamma helps focus on hard samples
    '''
    def __init__(self, alpha=0.9, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, eps=1e-8):

        alpha = self.alpha
        gamma = self.gamma
        focal_ce = - (alpha * torch.pow((1-y_pred), gamma) * torch.log(torch.clamp(y_pred, eps, 1.0)) * y_true
                      + (1-alpha) * torch.pow(y_pred, gamma) * torch.log(torch.clamp(1-y_pred, eps, 1.0)) * (1-y_true))
        focal_loss = torch.mean(focal_ce)

        return focal_loss

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, combine):
        super(SoftDiceLoss, self).__init__()
        self.combine = combine

    def forward(self, y_pred, y_true, eps=1e-8):   # put 2,1,4 together
        if self.combine:
            y_pred[:, 0, :, :, :] = torch.sum(y_pred, dim=1)
            y_pred[:, 1, :, :, :] = torch.sum(y_pred[:, 1:, :, :, :], dim=1)
            y_true[:, 0, :, :, :] = torch.sum(y_true, dim=1)
            y_true[:, 1, :, :, :] = torch.sum(y_true[:, 1:, :, :, :], dim=1)
        if y_true.dim() == 4:
            y_true[y_true == 4] = 3  # label [4] -> [3]
            y_true = expand_target(y_true, n_class=y_pred.size()[1])  # [N,H,W,D] -> [N,4，H,W,D]
        y_pred = flatten(y_pred)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
        y_true = flatten(y_true)[1:, ...]  # [class, N*H*W*D]
        # intersection = torch.sum(torch.mul(y_pred, y_true), dim=[-3, -2, -1])
        intersection = (y_pred*y_true).sum(-1)
        union = (y_pred * y_pred).sum(-1) + (y_true*y_true).sum(-1) + eps

        dice = 2 * intersection / union   # (bs, 3)
        # dice_loss = 1 - torch.mean(dice)  # loss small, better
        dice_loss = 1 - dice

        return dice_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(torch.log(torch.mul(std, std))) - 1


def CombinedLoss(y_pred, y_true, vae, ms):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''
    k1 = 0.1
    k2 = 0.1

    l2_loss = nn.MSELoss()
    kl_loss = CustomKLLoss()


    est_mean, est_std = (ms[:, :128], ms[:, 128:])
    # seg_pred, seg_truth = (y_pred[:,:3,:,:,:], y_true[:,:3,:,:,:])   # 10; (3+4)
    # vae_pred, vae_truth = (y_pred[:,3:,:,:,:], y_true[:,3:,:,:,:])
    dice_loss = GeneralizedDiceLoss(y_pred, y_true)
    l2_loss = l2_loss(y_pred, vae)  ### 7; 4
    kl_div = kl_loss(est_mean, est_std)

    combined_loss = dice_loss + k1 * l2_loss + k2 * kl_div

    return combined_loss

