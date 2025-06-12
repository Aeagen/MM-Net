import argparse
import pathlib
import time
from datetime import datetime
import yaml
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch.nn.parallel
import torch.optim
import random
import torch.utils.data
from torch.autograd import Variable
from ranger import Ranger
from torch.utils.tensorboard import SummaryWriter
from str2bool import find_last
from save_history import *
from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import determinist_collate
from src.loss import EDiceLoss
from src.models import get_norm_layer, DataAugmenter
from src.utils import save_args, AverageMeter, ProgressMeter, reload_ckpt, save_checkpoint, reload_ckpt_bis, \
    count_parameters, WeightSWA, save_metrics, generate_segmentations_axial
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from utils import concatenate, division_array
from cropPatch.crop import cal_crop_num_img
print("---------")
parser = argparse.ArgumentParser(description='Brats Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='Unet',
                    help='model architecture (default: Unet)')
parser.add_argument('--width', default=48, help='base number of features for Unet (x2 per downsampling)', type=int)
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr') # default lr = 1e-4
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
# Warning: untested option!!
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint. Warning: untested option')
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--restore', action="store_true")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--deep_sup', action="store_true")
parser.add_argument('--no_fp16', action="store_true")
parser.add_argument('--seed', default=16111990, help="seed for train/val split")
parser.add_argument('--warm', default=3, type=int, help="number of warming up epochs")

parser.add_argument('--val', default=3, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--norm_layer', default='group')
parser.add_argument('--swa', action="store_true", help="perform stochastic weight averaging at the end of the training")
parser.add_argument('--swa_repeat', type=int, default=5, help="how many warm restarts to perform")
parser.add_argument('--optim', choices=['adam', 'sgd', 'ranger', 'adamw'], default='ranger')
parser.add_argument('--com', help="add a comment to this run!")
parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)
parser.add_argument('--warm_restart', action='store_true', help='use scheduler warm restarts with period of 30')
parser.add_argument('--full', default=False, type=bool, help='Fit the network on the full training set')
parser.add_argument('--val_seg_csv_path', default='./history/history_valid.csv', type=str, metavar='PATH',
                    help='path to latest checkpoint. Warning: untested option')


def main(args):
    """ The main training function.

    Only works for single node (be it single or multi-GPU)

    Parameters
    ----------
    args :
        Parsed arguments
    """
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    header = ['epoch', 'val_dice', 'train_loss']
    save_file_name = args.val_seg_csv_path
    folder_index = find_last(save_file_name, '/')
    save_dir = save_file_name[:folder_index]

    # setup
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")

    print(f"Working with {ngpus} GPUs")
    if args.optim.lower() == "ranger":
        # No warm up if ranger optimizer
        args.warm = 0

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    args.exp_name = f"{'debug_' if args.debug else ''}{current_experiment_time}_" \
                    f"_fold{args.fold if args.full else 'FULL'}" \
                    f"_{args.arch}_{args.width}" \
                    f"_batch{args.batch_size}" \
                    f"_optim{args.optim}" \
                    f"_{args.optim}" \
                    f"_lr{args.lr}-wd{args.weight_decay}_epochs{args.epochs}_deepsup{args.deep_sup}" \
                    f"_{'fp16' if not args.no_fp16 else 'fp32'}" \
                    f"_warm{args.warm}_" \
                    f"_norm{args.norm_layer}{'_swa' + str(args.swa_repeat) if args.swa else ''}" \
                    f"_dropout{args.dropout}" \
                    f"_warm_restart{args.warm_restart}" \
                    f"{'_' + args.com.replace(' ', '_') if args.com else ''}"
    args.save_folder = pathlib.Path(f"./runs/{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.save_folder = args.save_folder.resolve()
    save_args(args)
    t_writer = SummaryWriter(str(args.save_folder))

    # Create model
    print(f"Creating {args.arch}")

    model_maker = getattr(models, args.arch)

    model = model_maker(
        4, 3,
        width=args.width, deep_supervision=args.deep_sup,
        norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)
    # model = model_maker(
    #     4, 3)
    print(f"total number of trainable parameters {count_parameters(model)}")

    if args.swa:
        # Create the average model
        swa_model = model_maker(
            4, 3,
            width=args.width, deep_supervision=args.deep_sup,
            norm_layer=get_norm_layer(args.norm_layer), Train=True)
        # swa_model = model_maker(
        #     4, 3)
        for param in swa_model.parameters():
            param.detach_()
        swa_model = swa_model.cuda()
        swa_model_optim = WeightSWA(swa_model)

    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    # print(model)
    model_file = args.save_folder / "model.txt"
    with model_file.open("w") as f:
        print(model, file=f)

    criterion = EDiceLoss().cuda()
    metric = criterion.metric
    print("create Loss funcation success")

    rangered = False  # needed because LR scheduling scheme is different for this optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay, eps=1e-4)


    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)

    elif args.optim == "adamw":
        print(f"weight decay argument will not be used. Default is 11e-2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    elif args.optim == "ranger":
        optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        rangered = True
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
    # optionally resume from a checkpoint
        # ------------------------------reload checkpoint------
    if args.restore:
        args.epochs = 0
        print("config_file:", args.resume)
        config_file = pathlib.Path(args.resume).resolve()
        print("config_file:", config_file)
        ckpt = config_file.with_name("model_best.pth.tar")
        print("ckpt:", ckpt)
        with config_file.open("r") as file:
            old_args = yaml.safe_load(file)
            old_args = SimpleNamespace(**old_args, ckpt=ckpt)
            # set default normalisation
            if not hasattr(old_args, "normalisation"):
                old_args.normalisation = "minmax"
        print(old_args)

        model_args = old_args
        print(model_args.arch)
        model_maker = getattr(models, model_args.arch)
        model = model_maker(
            4, 3,
            width=args.width, deep_supervision=args.deep_sup,
            norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)
        print(f"Creating {model_args.arch}")

        reload_ckpt_bis(str(model_args.ckpt), model)
        model = model.cuda()
        print("reload best weights")
    # ---------------------------------------------------------------------------

    if args.debug:
        args.epochs = 2
        args.warm = 0
        args.val = 1

    if args.full:
        train_dataset, val_dataset, bench_dataset = get_datasets(args.seed, args.debug, full=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False,
            pin_memory=False, num_workers=args.workers)

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=1, num_workers=args.workers)
        print("Val dataset number of batch:", len(val_loader))
    else:

        train_dataset, val_dataset, bench_dataset = get_datasets(args.seed, args.debug, fold_number=args.fold, full=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False,
            pin_memory=False, num_workers=args.workers)
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False,
        #     pin_memory=False, num_workers=args.workers, collate_fn=determinist_collate)

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=1, num_workers=args.workers)
        print("Val dataset number of batch:", len(val_loader))

    print("Train dataset number of batch:", len(train_loader))

    # create grad scaler
    # scaler = GradScaler()
    scaler = None
    # Actual Train loop

    best = np.inf
    print("start warm-up now!")
    if args.warm != 0:
        tot_iter_train = len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda cur_iter: (1 + cur_iter) / (tot_iter_train * args.warm))

    patients_perf = []

    if not args.resume:
        for epoch in range(args.warm):
            ts = time.perf_counter()
            model.train()
            training_loss = step(train_loader, model, criterion, metric, args.deep_sup, optimizer, epoch, t_writer,
                                     scheduler, save_folder=args.save_folder,
                                 no_fp16=args.no_fp16, patients_perf=patients_perf)
            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss = step_val(val_loader, model, criterion, metric, args.deep_sup, optimizer, epoch,
                                           t_writer, save_folder=args.save_folder,
                                           no_fp16=args.no_fp16)

                t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, epoch)

    if args.warm_restart:
        print('Total number of epochs should be divisible by 30, else it will do odd things')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               args.epochs + 30 if not rangered else round(
                                                                   args.epochs * 0.5))
    print("start training now!")
    if args.swa:
        # c = 15, k=3, repeat = 5
        c, k, repeat = 30, 3, args.swa_repeat
        epochs_done = args.epochs
        reboot_lr = 0
        if args.debug:
            c, k, repeat = 2, 1, 2
    for epoch in range(args.start_epoch + args.warm, args.epochs + args.warm):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()
            model.train()
            training_loss = step(train_loader, model, criterion, metric, args.deep_sup, optimizer, epoch, t_writer,
                                 save_folder=args.save_folder,
                                 no_fp16=args.no_fp16, patients_perf=patients_perf, set_kernal=False)
            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss = step_val(val_loader, model, criterion, metric, args.deep_sup, optimizer,
                                           epoch,
                                           t_writer,
                                           save_folder=args.save_folder,
                                           no_fp16=args.no_fp16, patients_perf=patients_perf, set_kernal=False)
                    # 保存val 和 train 损失函数
                    print("save history")
                    values = [epoch, validation_loss, training_loss]
                    export_history(header, values, save_dir, save_file_name)

                t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, epoch)

                if validation_loss < best:
                    best = validation_loss
                    model_dict = model.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=args.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")

            if args.swa:
                if (args.epochs - epoch - c) == 0:
                    reboot_lr = optimizer.param_groups[0]['lr']

            if not rangered:
                scheduler.step()
                print("scheduler stepped!")
            else:
                if epoch / args.epochs > 0.5:
                    scheduler.step()
                    print("scheduler stepped!")

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break

    if args.swa:
        swa_model_optim.update(model)
        print("SWA Model initialised!")
        for i in range(repeat):
            optimizer = torch.optim.Adam(model.parameters(), args.lr / 2, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c + 10)
            for swa_epoch in range(c):
                # do_epoch for one epoch
                ts = time.perf_counter()
                model.train()
                swa_model.train()
                current_epoch = epochs_done + i * c + swa_epoch
                training_loss = step(train_loader, model, criterion, metric, args.deep_sup, optimizer,
                                     current_epoch, t_writer,
                                     scaler, no_fp16=args.no_fp16, patients_perf=patients_perf)
                te = time.perf_counter()
                print(f"Train Epoch done in {te - ts} s")

                t_writer.add_scalar(f"SummaryLoss/train", training_loss, current_epoch)

                # update every k epochs and val:
                print(f"cycle number: {i}, swa_epoch: {swa_epoch}, total_cycle_to_do {repeat}")
                if (swa_epoch + 1) % k == 0:
                    swa_model_optim.update(model)
                    model.eval()
                    swa_model.eval()
                    with torch.no_grad():
                        validation_loss = step_val(val_loader, model, criterion, metric, args.deep_sup, optimizer,
                                               current_epoch,
                                               t_writer, save_folder=args.save_folder, no_fp16=args.no_fp16)
                        swa_model_loss = step_val(val_loader, swa_model, criterion, metric, args.deep_sup, optimizer,
                                              current_epoch,
                                              t_writer, swa=True, save_folder=args.save_folder,
                                              no_fp16=args.no_fp16)

                    t_writer.add_scalar(f"SummaryLoss/val", validation_loss, current_epoch)
                    t_writer.add_scalar(f"SummaryLoss/swa", swa_model_loss, current_epoch)
                    t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, current_epoch)
                    t_writer.add_scalar(f"SummaryLoss/overfit_swa", swa_model_loss - training_loss, current_epoch)
                scheduler.step()
        epochs_added = c * repeat
        save_checkpoint(
            dict(
                epoch=args.epochs + epochs_added, arch=args.arch,
                state_dict=swa_model.state_dict(),
                optimizer=optimizer.state_dict()
            ),
            save_folder=args.save_folder, )
    else:
        save_checkpoint(
            dict(
                epoch=args.epochs, arch=args.arch,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            ),
            save_folder=args.save_folder, )

    try:
        df_individual_perf = pd.DataFrame.from_records(patients_perf)
        print(df_individual_perf)
        df_individual_perf.to_csv(f'{str(args.save_folder)}/patients_indiv_perf.csv')
        reload_ckpt_bis(f'{str(args.save_folder)}/model_best.pth.tar', model)
        generate_segmentations_axial(bench_loader, model, t_writer, args, set_kernal=True)
    except KeyboardInterrupt:
        print("Stopping right now!")
def step(data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer,
         scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None, set_kernal=False):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # TODO monitor teacher loss
    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []
    print(f"fp 16: {not no_fp16}")
    # TODO: not recreate data_aug for each epoch...
    data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()
    alpha = [0.2, 0.4, 0.6, 0.8]
    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        targets = batch["label"].cuda(non_blocking=True)
        inputs = batch["image"].cuda()
        patient_id = batch["patient_id"]
        inputs = inputs.type(torch.FloatTensor).cuda()
        if set_kernal:
            inputs, pads = pad_batch1_to_compatible_size(inputs)
            print("valid_input:",inputs.shape)
        # data augmentation step
        if mode == "train":
            inputs = data_aug(inputs)
        if deep_supervision:

            segs, deeps = model(inputs)
            if mode == "train":  # revert the data aug
                segs, deeps = data_aug.reverse([segs, deeps])
            loss_ = torch.stack(
                [criterion(segs, targets)] + [criterion(de, targets) * alp for
                                              de, alp in zip(deeps, alpha)])
            # print("seg:{},deep_0:{},deep_1:{},deep_2:{}".format(loss_[0],loss_[1],loss_[2],loss_[3]))
            loss_ = torch.mean(loss_)
            print("---------------mean_Loss:{}---------------".format(loss_))
        else:
            inputs = inputs.type(torch.FloatTensor).cuda()
            segs = model(inputs)
            if mode == "train":
                segs = data_aug.reverse(segs)
            loss_ = criterion(segs, targets)
        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")

        if not model.training:
            metric_ = metric(segs, targets)
            metrics.extend(metric_)
        loss_ = loss_.requires_grad_()
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        # compute gradient and do SGD step
        # if model.training:
        #     scaler.scale(loss_).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()
        #     writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)
    if not model.training:
        save_metrics(epoch, metrics, swa, writer, epoch, False, save_folder)

    if mode == "train":
        writer.add_scalar(f"SummaryLoss/train", losses.avg, epoch)
    else:
        writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    return losses.avg

def step_val(data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer,
         scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None, set_kernal=False):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # TODO monitor teacher loss
    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []
    print(f"fp 16: {not no_fp16}")
    # TODO: not recreate data_aug for each epoch...
    data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()
    alpha = [0.2, 0.4, 0.6, 0.8]
    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        targets = batch["label"].cuda(non_blocking=True)
        inputs = batch["image"].cuda()
        # inputs, pads = pad_batch1_to_compatible_size(inputs)
        patient_id = batch["patient_id"]
        crops_idx = batch["crop_indexes"]
        zmin, zmax = crops_idx[0][0], crops_idx[0][1]
        ymin, ymax = crops_idx[1][0], crops_idx[1][1]
        xmin, xmax = crops_idx[2][0], crops_idx[2][1]
        Z = int(zmax - zmin)
        Y = int(ymax - ymin)
        X = int(xmax - xmin)
        print("Z:{},Y:{},X:{}".format(Z,Y,X))
        inputs = inputs.type(torch.FloatTensor).cuda()
        # print("inputs", inputs.shape)
        if set_kernal:
            inputs, pads = pad_batch1_to_compatible_size(inputs)
            print("valid_input:",inputs.shape)
        # data augmentation step
        stacked_img = torch.Tensor([]).cuda()
        total_loss = 0
        for index in range(inputs.size()[2]):
            with torch.no_grad():
                image_v = Variable(inputs[:, :, index, :, :, :]).cuda()
                # mask_v = Variable(targets[:, :, index, :, :, :]).cuda()
                if deep_supervision:
                    segs, deeps = model(image_v)
                    if mode == "train":  # revert the data aug
                        segs, deeps = data_aug.reverse([segs, deeps])

                    # loss_ = torch.stack(
                    #     [criterion(segs, targets)] + [criterion(deep, targets) * alp for
                    #                                   deep, alp in zip(deeps, alpha)])
                    # # print("seg:{},deep_0:{},deep_1:{},deep_2:{}".format(loss_[0],loss_[1],loss_[2],loss_[3]))
                    # loss_ = torch.mean(loss_)
                    # print("---------------mean_Loss:{}---------------".format(loss_))
                else:
                    segs = model(image_v)
                stacked_img = torch.cat((stacked_img, segs))
        c1, c2, c3 = cal_crop_num_img((Z, Y, X), (128, 128, 128))
        print("x1:{}, c2:{}. c3:{}".format(c1, c2, c3))
        div_array = division_array(128, c1, c2, c3, Z, Y, X)
        print("staced", stacked_img.shape)
        print("div_array", div_array.shape)
        pre_segs = concatenate((Z, Y, X), stacked_img)
        pre_segs = pre_segs / (div_array + 0.00001)
        # pre_segs = pre_segs > 0.5
        pre_segs = torch.from_numpy(pre_segs).unsqueeze(0).numpy()
        pre_segs = torch.from_numpy(pre_segs).to(targets.device)
        loss_ = criterion(pre_segs, targets)
        # loss_ = criterion(segs, targets)
        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")

        if not model.training:
            metric_ = metric(pre_segs, targets)
            metrics.extend(metric_)
        loss_ = loss_.requires_grad_()
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        # compute gradient and do SGD step
        # if model.training:
        #     scaler.scale(loss_).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()
        #     writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)
    print("model_train",model.training)
    if not model.training:
        save_metrics(epoch, metrics, swa, writer, epoch, False, save_folder)

    if mode == "train":
        writer.add_scalar(f"SummaryLoss/train", losses.avg, epoch)
    else:
        writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    return losses.avg


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
