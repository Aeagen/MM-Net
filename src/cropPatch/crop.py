import math
import numpy as np
from src.cropPatch.pre_processing import division_array, image_concatenate
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

def cal_crop_num_img_seg(img_size, in_size):
    if img_size[2] % in_size[0] == 0:
        crop_n1 = math.ceil(img_size[2] / in_size[0]) + 1
    else:
        crop_n1 = math.ceil(img_size[2] / in_size[0])

    if img_size[3] % in_size[1] == 0:
        crop_n2 = math.ceil(img_size[3] / in_size[1]) + 1
    else:
        crop_n2 = math.ceil(img_size[3] / in_size[1])
    if img_size[4] % in_size[2] == 0:
        crop_n3 = math.ceil(img_size[4] / in_size[2]) + 1
    else:
        crop_n3 = math.ceil(img_size[4] / in_size[2])
    return crop_n1, crop_n2, crop_n3

def crop_pad(image, crop_size, set=False):
    image_size = image.shape  # 1,3,160,256,256
    # print("image", image_size)
    if set:
        crop_n1, crop_n2, crop_n3 = cal_crop_num_img_seg(image_size, crop_size)
    else:
        crop_n1, crop_n2, crop_n3 = cal_crop_num_img(image_size[1:], crop_size)
        # print("crop1:{},crop2:{},crop3:{}".format(crop_n1, crop_n2, crop_n3))
    img_as_np = multi_cropping(image,
                               crop_size=crop_size[0],
                               crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)
    return img_as_np
def multi_cropping(image, crop_size, crop_num1, crop_num2, crop_num3):
    """crop the image and pad it to in_size
    Args :
        images : numpy arrays of images
        crop_size(int) : size of cropped image
        crop_num2 (int) : number of crop in horizontal way
        crop_num1 (int) : number of crop in vertical way
    Return :
        cropped_imgs : numpy arrays of stacked images
    """
    # print(image.shape)
    img_depth, img_height, img_width = image.shape[1], image.shape[2], image.shape[3]
    # assert crop_size*crop_num1 >= img_height and crop_size * \
    #     crop_num2 >= img_width, "Whole image cannot be sufficiently expressed"
    # assert crop_num1 <= img_height - crop_size + 1 and crop_num2 <= img_width - \
    #     crop_size + 1, "Too many number of crops"

    cropped_imgs = []
    # int((img_height - crop_size)/(crop_num1 - 1))
    dim0_stride = stride_size(img_depth, crop_num1, crop_size)

    dim1_stride = stride_size(img_height, crop_num2, crop_size)
    # int((img_width - crop_size)/(crop_num2 - 1))
    dim2_stride = stride_size(img_width, crop_num3, crop_size)
    for d in range(crop_num1):
        for i in range(crop_num2):
            for j in range(crop_num3):
                # print("j", j)
                cropped_imgs.append(cropping(image, crop_size,
                                             dim0_stride*d, dim1_stride*i, dim2_stride*j))
    return np.asarray(cropped_imgs)

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

def cropping(image, crop_size, dim0, dim1, dim2):
    """crop the image and pad it to in_size
    Args :
        images : numpy array of images
        crop_size(int) : size of cropped image
        dim1(int) : vertical location of crop
        dim2(int) : horizontal location of crop
    Return :
        cropped_img: numpy array of cropped image
    """
    cropped_img = image[:, dim0:dim0+crop_size, dim1:dim1+crop_size, dim2:dim2+crop_size]
    # print("croped", cropped_img.shape)
    return cropped_img

def prediction_image(stacked_img, ori_shape):  # stack_img: 18, 3, 128, 128, 128  ori_shape: 160, 256, 256
    stacked_img = stacked_img.cpu().data.numpy()
    stack_img_shape = np.shape(stacked_img)
    in_size = np.shape(stacked_img)[2:]  # 128, 128, 128

    pad = (5, 16, 16)
    crop_n1, crop_n2, crop_n3 = cal_crop_num_img(ori_shape, in_size)

    div_arr = division_array(in_size, crop_n1, crop_n2, crop_n3, ori_shape) + 0.000001
    img_cont_np = image_concatenate(stacked_img,crop_n1, crop_n2, crop_n3,ori_shape[0], ori_shape[1], ori_shape[2])
    print("div",div_arr.shape)
    print("img_cont_np",img_cont_np.shape)
    probability = img_cont_np / div_arr
    print("probability11", probability.shape)
    probability = probability[:, :-pad[0], :-pad[1], :-pad[2]]
    print("probability", probability.shape)
    return probability