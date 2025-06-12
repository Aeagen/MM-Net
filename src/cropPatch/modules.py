import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import *
import torch.nn as nn
import torch.nn.functional as F
from accuracy import accuracy_check, accuracy_check_for_batch
import csv
import os
from metrics import *
from save_history import *
import re
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
device=torch.device("cuda")
def train_model(model, data_train, criterion, optimizer):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    model.train()
    for batch, (images, masks) in enumerate(data_train):
        # plt.imshow(images[0,0],cmap="gray")
        # plt.show()
        # plt.imshow(masks[0],cmap="gray")
        # plt.show()
        # print(np.unique(masks[0]))
        images = images.to(device)
        masks = masks.to(device)
        # 修改的地方
        # outputs = model(images)
        out1,out2,out3,out4,out5,out6 = model(images)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(out1.cpu().data.numpy()[0][1])
        # plt.subplot(1, 2, 2)
        # plt.imshow(masks.cpu().data.numpy()[0])
        # plt.show()
        # print(outputs)
        # print(images.shape)
        # print(masks.shape)
        # print(masks.shape, outputs.shape)
        # 修改的地方
        # loss0 = criterion(out0, masks)
        loss1 = criterion(out1, masks)
        loss2 = criterion(out2, masks)
        loss3 = criterion(out3, masks)
        loss4 = criterion(out4, masks)
        loss5 = criterion(out5, masks)
        loss6 = criterion(out6, masks)
        loss = loss1+loss2+loss3+loss4+loss5+loss6

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(out1.cpu().data.numpy()[0][1])
        # plt.subplot(1, 2, 2)
        # plt.imshow(masks.cpu().data.numpy()[0])
        # plt.show()

        # loss = criterion(outputs, masks)
        # 修改的地方
        #loss = (4*loss_outputs+3*loss3+2*loss2+loss1)/10
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
    # total_loss = get_loss_train(model, data_train, criterion)


def get_loss_train(model, data_train, criterion):
    """
        Calculate loss over train set
    """
    model.eval()
    total_acc = 0
    total_loss = 0
    num=0
    for batch, (images, masks) in enumerate(data_train):
        with torch.no_grad():
            images = images.to(device)
            masks = masks.to(device)
            # 修改的地方
            out1,out2,out3,out4,out5,out6 = model(images)
            # outputs = model(images)
            # loss = criterion(outputs, masks)
            # loss0 = criterion(out0, masks)
            loss1 = criterion(out1, masks)
            loss2 = criterion(out2, masks)
            loss3 = criterion(out3, masks)
            loss4 = criterion(out4, masks)
            loss5 = criterion(out5, masks)
            loss6 = criterion(out6, masks)
            loss =  loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            preds = torch.argmax(out1, dim=1).float()
            acc = accuracy_check_for_batch(masks.cpu(), preds.cpu(), images.size()[0])
            total_acc = total_acc + acc
            total_loss = total_loss + loss.cpu().item()
            num +=1
            # if batch>=20:
            #     break
    return total_acc / (num), total_loss / (num)
def validate_model2(model, data_val, criterion, epoch, header2,save_dir,save_file_name2,make_prediction=True, save_folder_name='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    model.eval()
    value=[None]*166
    value[0]=epoch
    num = 0
    total_val_acc = 0
    total_val_loss = 0
    total_val_dice = 0
    total_val_jac = 0
    total_val_Sp = 0
    total_val_sn = 0
    total_val_recall = 0
    total_val_recall = 0
    total_dict = {}
    for batch, (images_v, masks_v, original_msk,name,in_size) in enumerate(data_val):
        pad=16
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda())
                mask_v = Variable(masks_v[:, index, :, :].squeeze(1).cuda())
                # print(image_v.shape, mask_v.shape)

                logit = F.softmax(model(image_v), 1)  # 000
                logit += F.softmax(model(image_v.flip(dims=(2,))).flip(dims=(2,)), 1)
                logit += F.softmax(model(image_v.flip(dims=(3,))).flip(dims=(3,)), 1)
                logit += F.softmax(model(image_v.flip(dims=(2,3))).flip(dims=(2,3)), 1)
                logit += F.softmax(model(image_v.permute(0,1,3,2).flip(dims=(2,))).flip(dims=(2,)).permute(0,1,3,2), 1)
                logit += F.softmax(model(image_v.permute(0,1,3,2).flip(dims=(3,))).flip(dims=(3,)).permute(0,1,3,2), 1)
                logit += F.softmax(model(image_v.permute(0,1,3,2).flip(dims=(2,3))).flip(dims=(2,3)).permute(0,1,3,2), 1)
                output_v=logit/7.0
                output_v=output_v[:,:,pad:-pad,pad:-pad]

                print(name)
                total_val_loss = total_val_loss + criterion(output_v, mask_v).cpu().item()
                # print('out', output_v.shape)
                output_v = torch.argmax(output_v, dim=1).float()
                # print(output_v.max(),output_v.min())
                stacked_img = torch.cat((stacked_img, output_v))
        if make_prediction:
            name = re.sub("\D", "", name[0])

            diex=int(name)
            im_name = diex  # TODO: Change this to real image name so we know
            pred_msk = save_prediction_image(stacked_img, im_name, epoch, save_folder_name,in_size=in_size)
            jac = dice_coeff(pred_msk, original_msk)
            value[diex]=(jac["jac"])
            total_val_dice = total_val_dice + jac["f1"]
            total_val_jac = total_val_jac + jac["jac"]
            total_val_Sp = total_val_Sp + jac["sp"]
            total_val_sn = total_val_sn + jac["sn"]
            total_val_recall = total_val_recall + jac["rc"]
            total_val_acc = total_val_acc + jac["acc"]

            num = num + 1
            # total_val_dice = total_val_dice/num
            # total_val_jac = total_val_jac/num
            # total_val_sp = total_val_sp/num
            # total_val_sn = total_val_sn/num
            # total_val_recall = total_val_recall/num
            # total_val_acc = total_val_acc/num
    #export_history(header2, value, save_dir, save_file_name2)
    total_dict['dice'] = total_val_dice
    total_dict['jac'] = total_val_jac
    total_dict['sp'] = total_val_Sp
    total_dict['sn'] = total_val_sn
    total_dict['recall'] = total_val_recall
    total_dict['acc'] = total_val_acc
        # return total_val_dice / (num), total_val_jac / (num), total_val_acc / num
    return total_dict, num

def validate_model(model, data_val, criterion, epoch, header2,save_dir,save_file_name2,make_prediction=True, save_folder_name='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    model.eval()
    value=[None]*166
    value[0]=epoch
    num = 0
    total_val_acc = 0
    total_val_loss = 0
    total_val_dice = 0
    total_val_jac = 0
    total_val_Sp = 0
    total_val_sn = 0
    total_val_AUC = 0
    total_val_recall = 0
    total_dict = {}
    for batch, (images_v, masks_v, original_msk,name,in_size) in enumerate(data_val):
        pad=0
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index,:,:, :, :].unsqueeze(0).cuda())
                mask_v = Variable(masks_v[:, index, :, :].squeeze(1).cuda())
                out1, out2, out3, out4, out5, out6 = model(image_v)

                total_val_loss = total_val_loss + criterion(out1, mask_v).cpu().item()
                # print('out', output_v.shape)
                output_v = torch.argmax(out1, dim=1).float()
                # print(output_v.max(),output_v.min())
                stacked_img = torch.cat((stacked_img, output_v))
                # print('stack', stacked_img.shape)
        if make_prediction:
            name=re.sub("\D","",name[0])
            diex=int(name)
            im_name = diex  # TODO: Change this to real image name so we know
            pred_msk = save_prediction_image(stacked_img, im_name, epoch, save_folder_name,in_size=in_size)
            # plt.imshow(pred_msk)
            # plt.show()
            # print(pred_msk.shape)
            # print(original_msk.shape)
            jac = dice_coeff(pred_msk, original_msk)
            # value[diex]=(jac["jac"])
            total_val_dice = total_val_dice + jac["f1"]
            total_val_jac = total_val_jac + jac["jac"]
            total_val_Sp = total_val_Sp + jac["sp"]
            total_val_sn = total_val_sn + jac["sn"]
            total_val_recall = total_val_recall + jac["rc"]
            total_val_acc = total_val_acc + jac["acc"]
            total_val_AUC = total_val_AUC + jac["AUC"]
            #print(jac["AUC"])
            num = num + 1
            # total_val_dice = total_val_dice/num
            # total_val_jac = total_val_jac/num
            # total_val_sp = total_val_sp/num
            # total_val_sn = total_val_sn/num
            # total_val_recall = total_val_recall/num
            # total_val_acc = total_val_acc/num
    #export_history(header2, value, save_dir, save_file_name2)
    total_dict['dice'] = total_val_dice
    total_dict['jac'] = total_val_jac
    total_dict['sp'] = total_val_Sp
    total_dict['sn'] = total_val_sn
    total_dict['recall'] = total_val_recall
    total_dict['acc'] = total_val_acc
    total_dict['AUC'] = total_val_AUC
        # return total_val_dice / (num), total_val_jac / (num), total_val_acc / num
    return total_dict, num


def test_model(model_path, data_test, epoch, save_folder_name='prediction'):
    """
        Test run
    """
    model = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
    model.eval()
    for batch, (images_t) in enumerate(data_test):
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_t.size()[1]):
            with torch.no_grad():
                image_t = Variable(images_t[:, index, :, :].unsqueeze(0).cuda())
                output_t = model(image_t)
                # print(image_t.shape, output_t.shape)
                output_t = torch.argmax(output_t, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_t))
        im_name = batch  # TODO: Change this to real image name so we know
        _ = save_prediction_image(stacked_img, im_name, epoch, save_folder_name)
    print("Finish Prediction!")
def cal_crop_num_img(img_size, in_size):
    if img_size[0] % in_size[0] == 0:
        crop_n1 = math.ceil(img_size[0] / in_size[0]) + 1
    else:
        crop_n1 = math.ceil(img_size[0] / in_size[0])

    if img_size[1] % in_size[1] == 0:
        crop_n2 = math.ceil(img_size[1] / in_size[1]) + 1
    else:
        crop_n2 = math.ceil(img_size[1] / in_size[1])
    return crop_n1,crop_n2

def save_prediction_image(stacked_img, im_name, epoch, save_folder_name="result_images", in_size=[512,512],output_shape=[1152,1500],
                          save_im=True):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
    """
    insize=[0,0]
    insize[0]=int(in_size[0])
    insize[1]=int(in_size[1])
    crop_n1,crop_n2 = cal_crop_num_img(output_shape,insize)
    # if output_shape[0] >= 1024:
    #     crop_n1 = 3
    # if output_shape[1] >= 1024:
    #     crop_n2 = 3

    div_arr = division_array(stacked_img.size(1), crop_n1, crop_n2, output_shape[0], output_shape[1])
    # print('div_arr', div_arr.shape)
    img_cont = image_concatenate(stacked_img.cpu().data.numpy(), crop_n1, crop_n2, output_shape[0], output_shape[1])
    # print('img_cont', img_cont.shape)
    img_cont = polarize((img_cont) / (div_arr+0.0000000000001))
    img_cont_np = img_cont.astype('uint8')
    img_cont = Image.fromarray(img_cont_np * 255)
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    img_cont.save(desired_path + export_name)
    return img_cont_np


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


"""
def test_SEM(model, data_test,  folder_to_save):
    '''Test the model with test dataset
    Args:
        model: model to be tested
        data_test (DataLoader): test dataset
        folder_to_save (str): path that the predictions would be saved
    '''
    for i, (images) in enumerate(data_test):

        print(images)
        stacked_img = torch.Tensor([])
        for j in range(images.size()[1]):
            image = Variable(images[:, j, :, :].unsqueeze(0).cuda())
            output = model(image.cuda())
            print(output)
            print("size", output.size())
            output = torch.argmax(output, dim=1).float()
            print("size", output.size())
            stacked_img = torch.cat((stacked_img, output))
        div_arr = division_array(388, 2, 2, 512, 512)
        print(stacked_img.size())
        img_cont = image_concatenate(stacked_img.data.numpy(), 2, 2, 512, 512)
        final_img = (img_cont*255/div_arr)
        print(final_img)
        final_img = final_img.astype("uint8")
        break
    return final_img
"""

if __name__ == '__main__':
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_train_load = torch.utils.data.DataLoader(dataset=SEM_train,
                                                 num_workers=3, batch_size=10, shuffle=True)
    get_loss_train()
