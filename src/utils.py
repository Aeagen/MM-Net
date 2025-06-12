import os
import pathlib
import pprint
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
from torch import distributed as dist
# from torch.cuda.amp import autocast
from src.tta import apply_simple_tta
from torch.autograd import Variable
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from cropPatch.crop import crop_pad, stride_size, cal_crop_num_img





def concatenate(targets, stack_seg):
    image_numpy = stack_seg.cpu().data.numpy()  # 18, 3, 128, 128, 128
    targets_size = targets
    crop_n1, crop_n2, crop_n3 = cal_crop_num_img(targets_size, (image_numpy.shape[2], image_numpy.shape[3], image_numpy.shape[4]))
    # print("crop_n1", crop_n1,  crop_n2,  crop_n3)
    dim1 = targets_size[0]
    dim2 = targets_size[1]
    dim3 = targets_size[2]
    output = image_concatenate(image_numpy, crop_n1, crop_n2, crop_n3, dim1,  dim2, dim3)
    return output
def image_concatenate(image, crop_num1, crop_num2, crop_num3, dim1, dim2, dim3):
    """concatenate images
    Args :
        image : output images (should be square)
        crop_num2 (int) : number of crop in horizontal way (2)
        crop_num1 (int) : number of crop in vertical way (2)
        dim1(int) : vertical size of output (512)
        dim2(int) : horizontal size_of_output (512)
    Return :
        div_array : numpy arrays of numbers of 1,2,4
    """
    crop_size = image.shape[3]  # size of crop
    empty_array1 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    empty_array2 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    empty_array3 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    # empty_array4 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    empty_array1 = np.expand_dims(empty_array1,axis=0)
    empty_array2 = np.expand_dims(empty_array2,axis=0)
    empty_array3 = np.expand_dims(empty_array3,axis=0)
    # empty_array4 = np.expand_dims(empty_array4,axis=0)
    # empty_array = np.concatenate([empty_array1,empty_array2,empty_array3,empty_array4],axis=0)
    empty_array = np.concatenate([empty_array1,empty_array2,empty_array3],axis=0)
    dim0_stride = stride_size(dim1, crop_num1, crop_size)  # vertical stride
    dim1_stride = stride_size(dim2, crop_num2, crop_size)  # vertical stride
    dim2_stride = stride_size(dim3, crop_num3, crop_size)  # horizontal stride
    index = 0
    for d in range(crop_num1):
        for i in range(crop_num2):
            for j in range(crop_num3):
                # add image to empty_array at specific position
                empty_array[:,dim0_stride*d:dim0_stride*d + crop_size, dim1_stride*i:dim1_stride*i + crop_size,
                            dim2_stride*j:dim2_stride*j + crop_size] += image[index]
                index += 1
    return empty_array
def stride_size(image_len, crop_num, crop_size):
    """return stride size
    Args :
        image_len(int) : length of one size of image (width or height)
        crop_num(int) : number of crop in certain direction
        crop_size(int) : size of crop
    Return :
        stride_size(int) : stride size
    """
    return int((image_len - crop_size)/(crop_num - 1))


def save_args(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    pprint.pprint(config)
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with config_file.open("w") as file:
        yaml.dump(config, file)


def master_do(func, *args, **kwargs):
    """Help calling function only on the rank0 process id ddp"""
    try:
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
    except AssertionError:
        # not in DDP setting, just do as usual
        func(*args, **kwargs)


def save_checkpoint(state: dict, save_folder: pathlib.Path):
    """Save Training state."""
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# TODO remove dependency to args
def reload_ckpt(args, model, optimizer, scheduler):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))


def reload_ckpt_bis(ckpt, model, optimizer=None):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt}' (epoch {start_epoch})")
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


class WeightSWA(object):
    """
    SWA or fastSWA
    Taken from https://github.com/benathi/fastswa-semi-sup
    """

    def __init__(self, swa_model):
        self.num_params = 0
        self.swa_model = swa_model  # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_params += 1
        print("Updating SWA. Current num_params =", self.num_params)
        if self.num_params == 1:
            print("Loading State Dict")
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            inv = 1. / float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.add_(-inv * swa_p.data)
                swa_p.data.add_(inv * src_p.data)

    def reset(self):
        self.num_params = 0


def save_metrics(epoch, metrics, swa, writer, current_epoch, teacher=False, save_folder=None):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{'_swa' if swa else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


def generate_segmentations(data_loader, model, writer, args, set_kernal=True):
    metrics_list = []
    for i, batch in enumerate(data_loader):
        # measure data loading time
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0]
        crops_idx = batch["crop_indexes"]
        print("inputs", inputs.shape)
        inputs, pads = pad_batch1_to_compatible_size(inputs)
        # print("pads", pads)
        inputs = inputs.cuda()
        inputs = inputs.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            if args.deep_sup:
                pre_segs, _ = model(inputs)
            else:
                pre_segs = model(inputs)
            pre_segs = torch.sigmoid(pre_segs)
        # remove pads
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        print("maxz:{}, maxy:{},maxx:{}".format(maxz,maxy,maxx))
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
        # pre_segs = pre_segs[:, :, 0:155, 0:240, 0:240].cpu()
        segs = torch.zeros((1, 3, 155, 240, 240))
        print("pre_segs", pre_segs.shape)
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        # segs[0, :, :, :, :] = pre_segs[0]
        # print("0:{},1:{},2:{}".format(slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])))
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)
        print(f"Writing {args.seg_folder}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/{patient_id}.nii.gz")
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.save_folder / 'results.csv'), index=False)

def generate_segmentations_axial(data_loader, model, writer, args, set_kernal=True):
    metrics_list = []
    targets = (1,3,160,256,256)
    for i, batch in enumerate(data_loader):
        # measure data loading time
        stacked_img = torch.Tensor([]).cuda()
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0]
        crops_idx = batch["crop_indexes"]
        zmin, zmax = crops_idx[0][0], crops_idx[0][1]
        ymin, ymax = crops_idx[1][0], crops_idx[1][1]
        xmin, xmax = crops_idx[2][0], crops_idx[2][1]
        Z = int(zmax - zmin)
        Y = int(ymax - ymin)
        X = int(xmax - xmin)
        # inputs, pads = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.cuda()
        for index in range(inputs.size()[2]):
            with torch.no_grad():
                input = Variable(inputs[:, :, index, :, :, :].cuda())
                input = input.cuda()
                input = input.type(torch.FloatTensor).cuda()
                if args.deep_sup:
                    pre_segs, _ = model(input)
                else:
                    # pre_segs = apply_simple_tta(model, input, True).to(inputs.device)
                    pre_segs = model(input)
                # pre_segs = torch.sigmoid(pre_segs)
                stacked_img = torch.cat((stacked_img, pre_segs))
        # remove pads
        c1, c2, c3 = cal_crop_num_img((Z, Y, X), (128, 128, 128))
        print("x1:{}, c2:{}. c3:{}".format(c1, c2, c3))
        div_array = division_array(128, c1, c2, c3, Z, Y, X)
        print("staced", stacked_img.shape)
        print("div_array", div_array.shape)
        pre_segs = concatenate((Z, Y, X), stacked_img)
        pre_segs = pre_segs / (div_array + 0.00001)

        pre_segs = torch.from_numpy(pre_segs).permute(0, 1, 2, 3)
        # pre_segs = pre_segs[:, :155, :240, :240].cpu().numpy()

        segs = torch.zeros((1, 3, 155, 240, 240)).cpu().numpy()
        print("pre_segs", pre_segs.shape)
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs
        pre_seg = segs[0]
        segs_et = pre_seg[0] > 0.3
        segs_net = pre_seg[1] > 0.3
        segs_ed = pre_seg[2] > 0.4

        et = segs_et
        net = np.logical_and(segs_net, np.logical_not(et))
        ed = np.logical_and(segs_ed, np.logical_not(segs_net))
        labelmap = np.zeros(segs_et.shape)

        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        print(f"Writing {args.seg_folder}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/{patient_id}.nii.gz")





def division_array(crop_size, crop_num0, crop_num1, crop_num2, dim1, dim2, dim3):
    """Make division array
    Args :
        crop_size(int) : size of cropped image
        crop_num2 (int) : number of crop in horizontal way
        crop_num1 (int) : number of crop in vertical way
        dim1(int) : vertical size of output
        dim2(int) : horizontal size_of_output
    Return :
        div_array : numpy array of numbers of 1,2,4
    """
    empty_array1 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    empty_array2 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    empty_array3 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    # empty_array4 = np.zeros([dim1, dim2, dim3]).astype("float16")  # to make sure no overflow
    empty_array1 = np.expand_dims(empty_array1, axis=0)
    empty_array2 = np.expand_dims(empty_array2, axis=0)
    empty_array3 = np.expand_dims(empty_array3, axis=0)
    # empty_array4 = np.expand_dims(empty_array4, axis=0)
    div_array = np.concatenate([empty_array1, empty_array2, empty_array3], axis=0)
    # div_array = np.zeros([dim1, dim2, dim3])  # make division array
    one_array1 = np.ones([crop_size, crop_size, crop_size]).astype("float16")  # one array to be added to div_array
    one_array2 = np.ones([crop_size, crop_size, crop_size]).astype("float16")  # one array to be added to div_array
    one_array3 = np.ones([crop_size, crop_size, crop_size]).astype("float16")  # one array to be added to div_array
    # one_array4 = np.ones([crop_size, crop_size, crop_size])  # one array to be added to div_array
    one_array1 = np.expand_dims(one_array1, axis=0)
    one_array2 = np.expand_dims(one_array2, axis=0)
    one_array3 = np.expand_dims(one_array3, axis=0)
    # one_array4 = np.expand_dims(one_array4, axis=0)
    one_array = np.concatenate([one_array1, one_array2, one_array3], axis=0)
    dim0_stride = stride_size(dim1, crop_num1, crop_size)  # vertical stride
    dim1_stride = stride_size(dim2, crop_num1, crop_size)  # vertical stride
    dim2_stride = stride_size(dim3, crop_num2, crop_size)  # horizontal stride
    for d in range(crop_num0):
        for i in range(crop_num1):
            for j in range(crop_num2):
                # add ones to div_array at specific position
                div_array[:, dim0_stride*d:dim0_stride*d+crop_size, dim1_stride*i:dim1_stride*i + crop_size,
                          dim2_stride*j:dim2_stride*j + crop_size] += one_array
    # print("div_array", div_array.shape)
    return div_array


def update_teacher_parameters(model, teacher_model, global_step, alpha=0.99 / 0.999):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    # print("teacher updated!")


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]
