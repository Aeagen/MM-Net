import numpy as np
from PIL import Image
import glob
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
from random import randint
from torch.utils.data.dataset import Dataset
from pre_processing import *
from mean_std import *
import math
import cv2 as cv
import matplotlib.pyplot as plt
Training_MEAN = 0.4911
Training_STDEV = 0.1658

def cal_crop_num_img(img_size, in_size):
    if img_size[0] % in_size[0] == 0:
        crop_n1 = math.ceil(img_size[0] / in_size[0]) + 1
    else:
        crop_n1 = math.ceil(img_size[0] / in_size[0])

    if img_size[1] % in_size[1] == 0:
        crop_n2 = math.ceil(img_size[1] / in_size[1]) + 1
    else:
        crop_n2 = math.ceil(img_size[1] / in_size[1])
    if img_size[2] % in_size[2] == 0:
        crop_n3 = math.ceil(img_size[2] / in_size[2]) + 1
    else:
        crop_n3 = math.ceil(img_size[2] / in_size[2])
    return crop_n1, crop_n2, crop_n3


class SEMDataTrain(Dataset):

    def __init__(self, image_path, mask_path, in_size=512, out_size=512):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        # all file names
        # self.mask_arr = glob.glob(str(mask_path) + "/*")
        # self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.mask_arr = self.read_file(mask_path)
        self.image_arr = self.read_file(image_path)
        # self.mask2_arr= glob.glob(str("/media/y/D/LZR/EYE_blood2/eye_blood/DRIVE/training/mask") + str("/*"))
        self.in_size, self.out_size = in_size, out_size
        # Calculate len
        self.data_len = len(self.mask_arr)
        # calculate mean and stdev

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        """
        # GET IMAGE
        """
        single_image_name = self.image_arr[index]
        # mask2=self.mask2_arr[index]
        #print(single_image_name)
        # mask2_np=np.array(Image.open(mask2))
        img_as_img = Image.open(single_image_name)
        # img_as_img.show()
        img_as_np = np.array(img_as_img)

        img_as_np=img_as_np[:,:,1]

        #img_as_np[np.where(mask2_np==0)]=0
        # Augmentation
        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        img_as_np = flip(img_as_np, flip_num)

        # Noise Determine {0: Gaussian_noise, 1: uniform_noise
        # if randint(0, 1):
        #     # Gaussian_noise
        #     gaus_sd, gaus_mean = randint(0, 20), 0
        #     img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
        # else:
        #     # uniform_noise
        #     l_bound, u_bound = randint(-20, 0), randint(0, 20)
        #     img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

        # Brightness
        pix_add = randint(-10, 10)
        img_as_np = change_brightness(img_as_np, pix_add)
        # plt.imshow(img_as_np,cmap="gray")
        # plt.show()
        # 对比度受限的自适应直方图均衡化(CLAHE)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_as_np = clahe.apply(img_as_np)
        # plt.imshow(img_as_np,cmap="gray")
        # plt.show()
        # Gamma校正
        #img_as_np = np.power(img_as_np / float(np.max(img_as_np)), 1.2)


        # Elastic distort {0: distort, 1:no distort}
        sigma = randint(6, 12)
        # sigma = 4, alpha = 34
        #img_as_np, seed = add_elastic_transform(img_as_np, alpha=34, sigma=sigma, pad_size=20)

        # Crop the image
        img_height, img_width = img_as_np.shape[0], img_as_np.shape[1]
        pad_size = 0
        #img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
        y_loc, x_loc = randint(0, img_height+2*pad_size-self.out_size), randint(0, img_width+pad_size*2-self.out_size)
        img_as_np = cropping(img_as_np, crop_size=self.in_size, dim1=y_loc, dim2=x_loc)
        # plt.imshow(img_as_np, cmap="gray")
        # plt.show()
        '''
        # Sanity Check for image
        img1 = Image.fromarray(img_as_np)
        img1.show()
        '''
        # Normalize the image
        img_as_np = normalization1(img_as_np)
        # print(img_as_np.shape)
        # print(img_as_np.min(), img_as_np.max())
        # plt.imshow(img_as_np, cmap="gray")
        # plt.show()
        img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        """
        # GET MASK
        """
        single_mask_name = self.mask_arr[index]
        #print(single_mask_name)
        msk_as_img = Image.open(single_mask_name)
        # msk_as_img.show()
        msk_as_np = np.array(msk_as_img)
        #msk_as_np = np.zeros(shape=msk_as_np2.shape, dtype=np.uint8)
        # msk_as_np[np.where(msk_as_np2 == True)] = 255
        # msk_as_np[np.where(msk_as_np2 == False)] = 0
        # flip the mask with respect to image
        msk_as_np = flip(msk_as_np, flip_num)

        # elastic_transform of mask with respect to image

        # sigma = 4, alpha = 34, seed = from image transformation
        # msk_as_np, _ = add_elastic_transform(
        #    msk_as_np, alpha=34, sigma=sigma, seed=seed, pad_size=20)
        #msk_as_np.flags.writeable = True
        #msk_as_np = approximate_image(msk_as_np)  # images only with 0 and 255
        #msk_as_np = np.pad(msk_as_np, pad_size, mode="symmetric")

        # Crop the mask
        msk_as_np = cropping(msk_as_np, crop_size=self.out_size, dim1=y_loc, dim2=x_loc)
        # plt.imshow(msk_as_np, cmap="gray")
        # plt.show()
        '''
        # Sanity Check for mask
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        '''

        # Normalize mask to only 0 and 1
        msk_as_np[msk_as_np >= 189] = 255
        msk_as_np[msk_as_np < 189] = 0
        msk_as_np = msk_as_np/255
        # plt.imshow(msk_as_np, cmap="gray")
        # plt.show()
        # print(np.unique(msk_as_np))
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

import re

class SEMDataVal(Dataset):
    def __init__(self, image_path, mask_path, in_size=512, out_size=512):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # paths to all images and masks
        self.image_path=image_path
        self.mask_arr = self.read_file(mask_path)
        self.image_arr = self.read_file(image_path)
        # self.mask2_arr=glob.glob(str("/media/y/D/LZR/EYE_blood2/eye_blood/DRIVE/test/mask") + str("/*"))
        self.in_size = in_size
        self.out_size = out_size
        self.data_len = len(self.mask_arr)

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index : an integer variable that calls (indext)th image in the
                    path
        Returns:
            Tensor: 4 cropped data on index which is converted to Tensor
        """

        single_image = self.image_arr[index]
        # mask2=self.mask2_arr[index]
        name=single_image.replace(self.image_path,"")
        name = re.sub("\D", "", name)
        # mask2_np=np.array(Image.open(mask2))
        img_as_img = Image.open(single_image)
        # img_as_img.show()
        # Convert the image into numpy array
        img_as_np = np.array(img_as_img)
        plt.imshow(img_as_np, cmap="gray")
        plt.show()
        '''
        DRIVE数据集需要选取通道
        '''
        img_as_np=img_as_np[:,:,1]

        #img_as_np[np.where(mask2_np==0)]=0
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_as_np = clahe.apply(img_as_np)

        # Gamma校正
        #img_as_np = np.power(img_as_np / float(np.max(img_as_np)), 1.2)
        img_size = img_as_np.shape
        # Make 4 cropped image (in numpy array form) using values calculated above
        # Cropped images will also have paddings to fit the model.
        pad_size = 0

        crop_n1,crop_n2 = cal_crop_num_img(img_size,[self.in_size,self.in_size])
        img_as_np = multi_cropping(img_as_np,
                                   crop_size=self.in_size,
                                   crop_num1=crop_n1, crop_num2=crop_n2)                         #DRIVE：2
        #print(crop_n1,crop_n2)
        # Empty list that will be filled in with arrays converted to tensor
        processed_list = []

        for array in img_as_np:

            # SANITY CHECK: SEE THE CROPPED & PADDED IMAGES
            #array_image = Image.fromarray(array)
            array = np.pad(array, pad_size, mode="symmetric")
            # Normalize the cropped arrays
            img_to_add = normalization1(array)
            #print(img_to_add.shape)
            # Convert normalized array into tensor
            processed_list.append(img_to_add)
        #print(processed_list.shape)
        img_as_tensor = torch.Tensor(processed_list)
        #  return tensor of 4 cropped images
        #  top left, top right, bottom left, bottom right respectively.

        """
        # GET MASK
        """
        single_mask_name = self.mask_arr[index]
        msk_as_img = Image.open(single_mask_name)
        # msk_as_img.show()
        msk_as_np = np.array(msk_as_img)
        # plt.imshow(msk_as_np, cmap="gray")
        # plt.show()
        # msk_as_np = np.zeros(shape=msk_as_np2.shape, dtype=np.uint8)
        # msk_as_np[np.where(msk_as_np2 == True)] = 255
        # msk_as_np[np.where(msk_as_np2 == False)] = 0
        msk_as_np[msk_as_np >= 189] = 255
        msk_as_np[msk_as_np < 189] = 0
        msk_as_img=msk_as_np
        plt.imshow(msk_as_img, cmap="gray")
        plt.show()
        # Normalize mask to only 0 and 1
        msk_as_np = multi_cropping(msk_as_np,
                                   crop_size=self.out_size,
                                   crop_num1=crop_n1, crop_num2=crop_n2)
        # msk_as_np[msk_as_np>0]=255
        # msk_as_np[msk_as_np >= 42] = 255
        # msk_as_np[msk_as_np < 42] = 0
        msk_as_np = msk_as_np/255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        original_msk = torch.from_numpy(np.asarray(msk_as_img))
        return (img_as_tensor, msk_as_tensor, original_msk,name,[self.in_size,self.in_size])

    def __len__(self):

        return self.data_len

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

class SEMDataTest(Dataset):

    def __init__(self, image_path, in_size=572, out_size=388):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # paths to all images and masks

        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size = in_size
        self.out_size = out_size
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        '''Get specific data corresponding to the index
        Args:
            index: an integer variable that calls(indext)th image in the
                path
        Returns:
            Tensor: 4 cropped data on index which is converted to Tensor
        '''

        single_image = self.image_arr[index]
        img_as_img = Image.open(single_image)
        # img_as_img.show()
        # Convert the image into numpy array
        img_as_np = np.asarray(img_as_img)

        pad_size = int((self.in_size - self.out_size)/2)
        img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
        img_as_np = multi_cropping(img_as_np,
                                   crop_size=self.in_size,
                                   crop_num1=2, crop_num2=2)

        # Empty list that will be filled in with arrays converted to tensor
        processed_list = []

        for array in img_as_np:

            # SANITY CHECK: SEE THE PADDED AND CROPPED IMAGES
            # array_image = Image.fromarray(array)

            # Normalize the cropped arrays
            img_to_add = normalization2(array, max=1, min=0)
            # Convert normalized array into tensor
            processed_list.append(img_to_add)

        img_as_tensor = torch.Tensor(processed_list)
        #  return tensor of 4 cropped images
        #  top left, top right, bottom left, bottom right respectively.
        return img_as_tensor

    def __len__(self):

        return self.data_len


if __name__ == "__main__":

    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')
    SEM_val = SEMDataVal('../data/val/images', '../data/val/masks')

    imag_1, msk = SEM_val.__getitem__(0)
    print(imag_1.shape)