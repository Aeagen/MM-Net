import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace
from torch.autograd import Variable
import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
# from torch.cuda.amp import autocast
import nibabel as nib
from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.models import get_norm_layer
from src.tta import apply_simple_tta
from src.utils import reload_ckpt_bis
from train_Axial import concatenate
from utils import division_array
from cropPatch.crop import crop_pad, stride_size, cal_crop_num_img

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default='', type=str, metavar='PATH',
                    help='path(s) to the trained models config yaml you want to use', nargs="+")
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="val", choices=["val","train","test"])
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)
parser.add_argument('--postprocess', action="store_true")


def main(args):
    # setup
    random.seed(args.seed)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")
    print(f"Working with {ngpus} GPUs")
    print(args.config)

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    save_folder = pathlib.Path(f"./preds/{current_experiment_time}")
    save_folder.mkdir(parents=True, exist_ok=True)

    with (save_folder / 'args.txt').open('w') as f:
        print(vars(args), file=f)

    args_list = []
    for config in args.config:
        config_file = pathlib.Path(config).resolve()
        print("config_file:",config_file)
        ckpt = config_file.with_name("model_best.pth.tar")
        print("ckpt:", ckpt)
        with config_file.open("r") as file:
            old_args = yaml.safe_load(file)
            old_args = SimpleNamespace(**old_args, ckpt=ckpt)
            # set default normalisation
            if not hasattr(old_args, "normalisation"):
                old_args.normalisation = "minmax"
        print(old_args)
        args_list.append(old_args)

    if args.on == "test":
        args.pred_folder = save_folder / f"test_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    elif args.on == "val":
        args.pred_folder = save_folder / f"validation_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    else:
        args.pred_folder = save_folder / f"training_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)

    # Create model

    models_list = []
    normalisations_list = []
    for model_args in args_list:
        print(model_args.arch)
        model_maker = getattr(models, model_args.arch)

        model = model_maker(
            4, 3,
            width=model_args.width, deep_supervision=model_args.deep_sup,
            norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        print(f"Creating {model_args.arch}")

        reload_ckpt_bis(str(model_args.ckpt), model)
        models_list.append(model)
        normalisations_list.append(model_args.normalisation)
        print("reload best weights")
        # print(model)

    dataset_minmax = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="minmax")

    dataset_zscore = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="zscore")

    loader_minmax = torch.utils.data.DataLoader(
        dataset_minmax, batch_size=1, num_workers=2)

    loader_zscore = torch.utils.data.DataLoader(
        dataset_zscore, batch_size=1, num_workers=2)

    print("Val dataset number of batch:", len(loader_minmax))
    generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list, args)

def crop_idex(crop_idx):
    zmin, zmax = crop_idx[0][0], crop_idx[0][1]
    ymin, ymax = crop_idx[1][0], crop_idx[1][1]
    xmin, xmax = crop_idx[2][0], crop_idx[2][1]
    Z = int(zmax - zmin)
    Y = int(ymax - ymin)
    X = int(xmax - xmin)
    return Z, Y, X

def generate_segmentations(data_loaders, models, normalisations, args):
    # TODO: try reuse the function used for train...
    target = torch.zeros((1, 3, 160, 256, 256))
    # segs = torch.zeros((1, 3, 155, 240, 240))
    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        inputs = batch_minmax["image"]
        patient_id = batch_minmax["patient_id"][0]
        ref_img_path = batch_minmax["seg_path"][0]
        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]
        model_preds = []
        last_norm = None
        for model, normalisation in zip(models, normalisations):
            if normalisation == last_norm:
                pass
            elif normalisation == "minmax":
                Z, Y, X = crop_idex(crops_idx_minmax)
                inputs = inputs_minmax.cuda()
                crops_idx = crops_idx_minmax
            elif normalisation == "zscore":
                Z, Y, X = crop_idex(crops_idx_zscore)
                inputs = inputs_zscore.cuda()
                crops_idx = crops_idx_zscore
            model.cuda()  # go to gpu
            stacked_img = torch.Tensor([]).cuda()
            for index in range(inputs.size()[2]):
                with torch.no_grad():
                    # print("inputs", inputs.shape)
                    input = Variable(inputs[:, :, index, :, :, :].cuda())
                    if args.tta:

                        input = input.type(torch.FloatTensor).cuda()
                        pre_segs = apply_simple_tta(model, input, True,False).cuda()
                        # model_preds.append(pre_segs)
                    else:
                        if model.deep_supervision:
                            input = input.type(torch.FloatTensor).cuda()
                            pre_segs, _ = model(input)
                        else:
                            input = input.type(torch.FloatTensor).cuda()
                            pre_segs = model(input)
                        pre_segs = pre_segs.sigmoid_().cpu().cuda()
                    # remove pads
                stacked_img = torch.cat((stacked_img, pre_segs))
            c1, c2, c3 = cal_crop_num_img((Z, Y, X), (128, 128, 128))
            # print("x1:{}, c2:{}. c3:{}".format(c1, c2, c3))
            div_array = division_array(128, c1, c2, c3, Z, Y, X)
            # print("staced", stacked_img.shape)
            # print("div_array", div_array.shape)
            pre_segs = concatenate((Z, Y, X), stacked_img)
            pre_segs = pre_segs / (div_array + 0.00001)

            pre_segs = torch.from_numpy(pre_segs).permute(0, 1, 2, 3)
            # pre_segs = pre_segs[:, :155, :240, :240].cpu().numpy()

            segs = torch.zeros((1, 3, 155, 240, 240)).cpu().numpy()
            print("pre_segs", pre_segs.shape)
            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs

            # pre_seg = segs[0]
            # segs_et = pre_seg[0] > 0.4
            # segs_net = pre_seg[1] > 0.4
            # segs_ed = pre_seg[2] > 0.6

            pre_seg = segs[0]
            segs_et = pre_seg[0] > 0.4
            segs_net = pre_seg[1] > 0.3
            segs_ed = pre_seg[2] > 0.4

            et = segs_et
            net = np.logical_and(segs_net, np.logical_not(et))
            ed = np.logical_and(segs_ed, np.logical_not(segs_net))
            labelmap = np.zeros(segs_et.shape)

            # et = segs[0]
            # net = np.logical_and(segs[1], np.logical_not(et))
            # ed = np.logical_and(segs[2], np.logical_not(segs[1]))
            # labelmap = np.zeros(segs[0].shape)

            labelmap[et] = 4
            labelmap[net] = 1
            labelmap[ed] = 2
            if args.postprocess:
                print("et:", len(labelmap[et]))
                ET_voxels = len(labelmap[et])
                if ET_voxels < 500:
                    print("et:", len(labelmap[et]))
                    labelmap[et] = 1

            oname = f"{args.pred_folder}/{patient_id}.nii.gz"
            labelmap = sitk.GetImageFromArray(labelmap)
            print(f"Writing {args.pred_folder}/{patient_id}.nii.gz")
            sitk.WriteImage(labelmap, f"{args.pred_folder}/{patient_id}.nii.gz")

            image = sitk.ReadImage(oname)
            img = sitk.GetArrayFromImage(image)
            out = sitk.GetImageFromArray(img)
            out.SetOrigin((-0.0, -239.0, 0.0))
            sitk.WriteImage(out, oname)

if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
