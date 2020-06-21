from matplotlib import image
from matplotlib import pyplot as plt
import scipy

from model import get_model
from torch import nn
import torch.optim as optim
import torch

import numpy as np
from skimage.transform import rescale
import argparse

from tqdm import tqdm
from perceptual_loss import VGGPerceptualLoss


# symmetric patch size
def get_random_patch(img, size):
    image_width = img.shape[2]
    image_height = img.shape[1]
    x = np.random.randint(0, image_width - size, dtype=np.int)
    y = np.random.randint(0, image_height - size, dtype=np.int)
    crop = img[:, y:y + size, x:x + size]
    return crop


def pre_process(img):
    # image is in format [H, W, C] with values in range [0, 255], return normalized version.
    mean = np.mean(img, axis=(0, 1))  # has C values (3)
    std = np.std(img, axis=(0, 1))  # has C values (3)
    img = (img - mean) / std
    img = np.transpose(img, [2, 0, 1]) # C, H, W
    img = np.expand_dims(img, 0)
    img = torch.tensor(img).float()

    if torch.cuda.is_available():
        img = img.cuda()

    return img


def post_process(img):
    # get image in range [-1,1] (result of tanh)
    # return values in range [0, 255]
    img = (img + 1.0) * (255.0 / 2)
    img = img.astype(int)
    return img


def main():
    gpu = torch.cuda.is_available()
    if gpu:
        print(torch.cuda.get_device_name(0))

    orig_res_np = image.imread('building.jpg')
    low_res_np = rescale(orig_res_np, 0.5, anti_aliasing=True, multichannel=True)
    low_res_np = rescale(low_res_np, 2, multichannel=True)

    image.imsave('tmp\\orig_ers.jpg', orig_res_np)
    image.imsave('tmp\\low_res.jpg', low_res_np)

    orig_res = pre_process(orig_res_np)
    low_res = pre_process(low_res_np)

    model = get_model()
    criterion = VGGPerceptualLoss()

    if gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters())

    # training!
    for step in tqdm(range(1000)):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        restored = model(low_res)

        loss = criterion(restored, orig_res)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(loss)
            image.imsave('tmp\\restored_in_step_{}.jpg'.format(step), restored.detach().numpy())

            upsampled_np = rescale(orig_res_np, 2, multichannel=True)
            image.imsave('tmp\\upsampled.jpg', upsampled_np)
            upsampled = pre_process(upsampled_np)

            super_resolution = model(upsampled)
            image.imsave('tmp\\super_resolution_in_step_{}.jpg'.format(step), super_resolution.detach().numpy())



            # save upsampled
            # perform superresolution
            # print loss
            pass


if __name__=='__main__':
    main()




