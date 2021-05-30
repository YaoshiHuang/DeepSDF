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
    decoder, latent_vec, filename, N=256, max_batch=32 ** 2, offset=None, scale=None, number_sample=30000
):
    start = time.time()
    npy_filename = filename

    decoder.eval()

    # NOTE: the pixel_origin is actually the (bottom, left, down) corner, not the middle
    pixel_origin = [-1, -1]
    pixel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 2, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 2, 3)

    # transform first 2 columns
    # to be the x, y index
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

    convert_sdf_samples_to_npy(
        sdf_values.data.cpu(),
        pixel_origin,
        N,
        npy_filename,
        offset,
        scale,
        number_sample
    )

def longest(coords):
    
    index = 0
    length = 0
    
    for i in range(len(coords)):
        if coords[i].shape[0] > length:
            index = i
            length = coords[i].shape[0]
    
    return index

def Sample(
    blank_sample,
    number_sample
):
    blank_ref = np.insert(blank_sample, blank_sample.shape[0], blank_sample[0], axis = 0)
    blank_ref = np.delete(blank_ref, 0, axis = 0)
    diff = blank_ref - blank_sample
    cdf = np.sqrt(diff[:,0] ** 2 + diff[:,1] ** 2)
    total_len = 0
    
    for i in range(len(cdf)):
        total_len = total_len + cdf[i]
        cdf[i] = total_len
        
    for j in range(number_sample):
        rnd_1 = np.random.rand() * total_len
        rnd_2 = np.random.rand()
        
        ele_1 = blank_ref[np.where(cdf >= rnd)[0][0]]
        ele_2 = blank_ref[np.where(cdf >= rnd)[0][0] + 1]
        ele_sampled = ele_1 * rnd_2 + ele_2 * (1 - rnd_2)
        
        blank_sample = np.insert(blank_sample, blank_sample.shape[0], ele_sampled, axis = 0)

def convert_sdf_samples_to_npy(
    pytorch_2d_sdf_tensor,
    pixel_grid_origin,
    N,
    npy_filename_out,
    offset=None,
    scale=None,
    number_sample
):
    """
    Convert sdf samples to .ply

    :param pytorch_2d_sdf_tensor: a torch.FloatTensor of shape (n,n)
    :pixel_grid_origin: a list of two floats: the left, down origin of the pixel grid
    :pixel_size: float, the size of the pixels
    :npy_filename_out: string, path of the filename to save to

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

    # Save pixel-based blank images to *.npy files.
    logging.debug("saving blank images to %s" % (npy_filename_out + 'image.npy'))
    np.save(npy_filename_out + 'image.npy', numpy_2d_sdf_array)

    # Save the coordinates of blank shapes to *.npy files.
    logging.debug("saving blank shapes to %s" % (npy_filename_out + 'shape.npy'))
    np.save(npy_filename_out + 'shape.npy', blank)
    
    # Generate more sampled points on the blank shapes.
    blank_sample = blank.copy()
    Sample(blank_sample, number_sample-blank_sample.shape[0])
    
    # Save the coordinates of sampled blank shapes to *.npy files.
    logging.debug("saving blank shapes to %s" % (npy_filename_out + 'sample.npy'))
    np.save(npy_filename_out + 'sample.npy', blank_sample)