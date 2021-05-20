#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deep_sdf.utils


def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 2, offset=None, scale=None
):
    start = time.time()
    txt_filename = filename

    decoder.eval()

    # NOTE: the pixel_origin is actually the (bottom, left, down) corner, not the middle
    pixel_origin = [-1, -1]
    pixel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 2, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 2, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 1] = overall_index % N
    samples[:, 0] = (overall_index.long() / N) % N

    # transform first 3 columns
    # to be the x, y coordinate
    samples[:, 0] = (samples[:, 0] * pixel_size) + pixel_origin[1]
    samples[:, 1] = (samples[:, 1] * pixel_size) + pixel_origin[0]

    num_samples = N ** 2

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:2].cuda()

        # To be noted, the stop indice of a slice can be larger than the length of the overall list.
        samples[head : min(head + max_batch, num_samples), 2] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 2]
    sdf_values = sdf_values.reshape(N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_txt(
        sdf_values.data.cpu(),
        pixel_origin,
        N,
        txt_filename,
        offset,
        scale,
    )

def longest(coords):
    
    index = 0
    length = 0
    
    for i in range(len(coords)):
        if coords[i].shape[0] > length:
            index = i
            length = coords[i].shape[0]
    
    return index

def convert_sdf_samples_to_txt(
    pytorch_2d_sdf_tensor,
    pixel_grid_origin,
    N,
    txt_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_2d_sdf_tensor: a torch.FloatTensor of shape (n,n)
    :pixel_grid_origin: a list of two floats: the left, down origin of the pixel grid
    :pixel_size: float, the size of the pixels
    :txt_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_2d_sdf_array = pytorch_2d_sdf_tensor.numpy()

    # This will be replaced by the Marching Square algorithm (skimage.measure.find_contour) in the blank shape optimisation case.
    # To clarify, contours extracted here are not required by image-based surrogate models, while can be used for the reconstruction of CAD models.
    coords = skimage.measure.find_contour(
        numpy_2d_sdf_array, 0.0
    )

    # Find the largest array in coords
    blank = coords[longest(coords)]
    
    # Convert the sdf values to 0-1 values.
    numpy_2d_sdf_array[numpy_2d_sdf_array <= 0] = 1
    numpy_2d_sdf_array[numpy_2d_sdf_array >0] = 0

    # Save pixel-based blank images to .txt files.
    logging.debug("saving blank images to %s" % (txt_filename_out + 'image.txt'))
    numpy_2d_sdf_array.write(txt_filename_out + 'image.txt')

    # Save the coordinates of blank shapes to .txt files.
    logging.debug("saving blank shapes to %s" % (txt_filename_out + 'shape.txt'))
    blank.write(txt_filename_out + 'shape.txt')